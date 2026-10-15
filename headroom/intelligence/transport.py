"""HTTP transport to llama-server and the JevK5 ``/v1/systemone`` API.

Every byte Headroom sends to a decision endpoint goes through
:func:`http_json`, which takes an explicit timeout derived from a
:class:`Deadline`. A decision's whole budget — ``/tokenize``, every
``/completion`` pass, the missing-letter retry and >16-option knockout passes —
is one monotonic deadline; each operation gets at most the remaining time, and
an exhausted budget raises :class:`DeadlineExceeded` before any socket opens.

Destinations are the managed llama-server on ``127.0.0.1`` or an endpoint the
operator configured. With ``HEADROOM_OFFLINE`` set, a non-loopback endpoint is
refused through :func:`headroom.offline.guard_egress`.
"""

from __future__ import annotations

import ipaddress
import json
import math
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from headroom.offline import guard_egress, is_offline

# The prompt's chat-template marker; with parse_special it must be ONE token.
SPECIAL_TOKEN_PROBE = "<|im_start|>"


class TransportError(RuntimeError):
    """A decision endpoint failed or answered with an unusable shape."""


class DeadlineExceeded(TransportError):
    """The decision's overall wall-clock budget is exhausted."""


class ContextOverflow(TransportError):
    """The prompt does not fit the running server's effective context."""


class Deadline:
    """A monotonic wall-clock budget shared by every step of one decision."""

    def __init__(self, seconds: float, *, clock: Any = time.monotonic) -> None:
        self._clock = clock
        self.started = clock()
        self.expires = self.started + max(0.0, float(seconds))

    def remaining(self) -> float:
        return self.expires - self._clock()

    def budget(self, cap: float | None = None) -> float:
        """Seconds the next operation may use; raises when nothing is left."""
        left = self.remaining()
        if left <= 0.005:
            raise DeadlineExceeded("decision deadline exhausted")
        return min(left, cap) if cap is not None else left

    def elapsed_ms(self) -> float:
        return (self._clock() - self.started) * 1000.0


def is_loopback_url(url: str) -> bool:
    host = (urlsplit(url).hostname or "").strip("[]")
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def http_json(
    method: str, url: str, payload: Mapping[str, Any] | None, *, timeout: float
) -> tuple[int, Any]:
    """One JSON request. Returns ``(status, parsed_body_or_None)``.

    Raises :class:`TransportError` on connection failure or timeout. The
    caller passes the remaining deadline budget as ``timeout``.
    """
    if is_offline() and not is_loopback_url(url):
        guard_egress("JevK5 decision endpoint", url)
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=max(0.01, timeout)) as response:
            body = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read() if hasattr(exc, "read") else b""
        status = exc.code
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
        raise TransportError(f"{method} {urlsplit(url).path} failed: {type(exc).__name__}") from exc
    try:
        parsed = json.loads(body) if body else None
    except ValueError:
        parsed = None
    return status, parsed


# ---------------------------------------------------------------------------
# llama-server capability probes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LlamaProbe:
    health: bool
    tokenize_special: bool
    completion_logprobs: bool
    n_ctx: int | None
    detail: str

    @property
    def ok(self) -> bool:
        return self.health and self.tokenize_special and self.completion_logprobs


def llama_health(base_url: str, *, timeout: float = 2.0) -> tuple[bool, str]:
    try:
        status, body = http_json("GET", base_url + "/health", None, timeout=timeout)
    except TransportError as exc:
        return False, str(exc)
    if status == 200:
        return True, "ok"
    message = body.get("error", {}).get("message") if isinstance(body, dict) else None
    return False, f"status {status}" + (f": {message}" if message else "")


def probe_llama_server(base_url: str, *, timeout: float = 10.0) -> LlamaProbe:
    """Require ``/health``, ``/tokenize`` (parse_special) and ``/completion`` logprobs."""
    deadline = Deadline(timeout)
    healthy, detail = llama_health(base_url, timeout=deadline.budget(3.0))
    if not healthy:
        return LlamaProbe(False, False, False, None, f"/health: {detail}")
    try:
        status, body = http_json(
            "POST",
            base_url + "/tokenize",
            {"content": SPECIAL_TOKEN_PROBE, "add_special": False, "parse_special": True},
            timeout=deadline.budget(),
        )
        tokens = body.get("tokens") if isinstance(body, dict) else None
        special_ok = status == 200 and isinstance(tokens, list) and len(tokens) == 1
        if not special_ok:
            return LlamaProbe(True, False, False, None, "/tokenize did not honor parse_special")
        status, body = http_json(
            "POST",
            base_url + "/completion",
            {
                "prompt": tokens,
                "n_predict": 1,
                "n_probs": 40,
                "temperature": 0,
                "cache_prompt": False,
            },
            timeout=deadline.budget(),
        )
        top = _top_logprobs(body) if status == 200 else None
        if not top:
            return LlamaProbe(True, True, False, None, "/completion returned no top_logprobs")
        n_ctx = _server_ctx(base_url, deadline)
        return LlamaProbe(True, True, True, n_ctx, "ok")
    except TransportError as exc:
        return LlamaProbe(True, False, False, None, str(exc))


def _top_logprobs(body: Any) -> list[dict[str, Any]] | None:
    try:
        top = body["completion_probabilities"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(top, list) or not top:
        return None
    for entry in top:
        if not isinstance(entry, dict) or not math.isfinite(float(entry.get("logprob", "nan"))):
            return None
    return top


def _server_ctx(base_url: str, deadline: Deadline) -> int | None:
    """Per-slot context the server actually runs with (``/props``)."""
    try:
        status, body = http_json("GET", base_url + "/props", None, timeout=deadline.budget(2.0))
    except TransportError:
        return None
    if status != 200 or not isinstance(body, dict):
        return None
    settings = body.get("default_generation_settings") or {}
    value = settings.get("n_ctx") or body.get("n_ctx")
    try:
        return int(value) if value else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# JevK5GGUF adapter with a whole-decision deadline
# ---------------------------------------------------------------------------


def make_deadline_adapter(
    base_url: str,
    *,
    deadline: Deadline,
    temperature: float,
    knockout_temperature: float,
    top_k: int,
    max_prompt_tokens: int,
) -> Any:
    """A fresh ``JevK5GGUF`` bound to one decision's deadline.

    One instance per decision: the upstream adapter mutates ``missing``,
    ``last_pass_tokens`` and ``last_seconds`` on every call, so sharing one
    across concurrent decisions would bleed metrics and missing-letter state
    between requests. The llama-server process itself stays shared.
    """
    from jevk5.gguf import JevK5GGUF

    class _DeadlineJevK5GGUF(JevK5GGUF):  # type: ignore[misc, valid-type]
        def _post(self, path: str, payload: dict) -> dict:
            status, body = http_json(
                "POST", self.url + path, payload, timeout=deadline.budget(self.timeout_s)
            )
            if status != 200 or not isinstance(body, dict):
                raise TransportError(f"{path} returned status {status}")
            if path == "/tokenize":
                tokens = body.get("tokens")
                if not isinstance(tokens, list):
                    raise TransportError("/tokenize returned no tokens")
                if len(tokens) + 1 > max_prompt_tokens:
                    raise ContextOverflow(
                        f"prompt needs {len(tokens) + 1} tokens; effective context is "
                        f"{max_prompt_tokens}"
                    )
            elif path == "/completion" and _top_logprobs(body) is None:
                raise TransportError("/completion returned no finite top_logprobs")
            return body

    return _DeadlineJevK5GGUF(
        url=base_url,
        temperature=temperature,
        knockout_temperature=knockout_temperature,
        top_k=top_k,
        # Upper bound for any single HTTP operation; the deadline caps it further.
        timeout_s=max(0.05, deadline.remaining()),
        method="knockout",
    )


# ---------------------------------------------------------------------------
# Operator-managed /v1/systemone client
# ---------------------------------------------------------------------------


class SystemOneClient:
    """Typed client for a JevK5-compatible ``/v1/systemone`` service.

    This API is distinct from llama-server's and is never handed to
    ``JevK5GGUF``.
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def decide(self, state: Any, question: Mapping[str, Any], *, deadline: Deadline) -> dict:
        status, body = http_json(
            "POST",
            self.base_url + "/v1/systemone",
            {"state": state, "questions": {"q": dict(question)}},
            timeout=deadline.budget(),
        )
        if status != 200 or not isinstance(body, dict):
            raise TransportError(f"/v1/systemone returned status {status}")
        answer = (body.get("answers") or {}).get("q")
        if not isinstance(answer, dict) or answer.get("type") != question.get("type"):
            raise TransportError("/v1/systemone answer has an unexpected shape")
        return answer

    def probe(self, *, timeout: float = 5.0) -> tuple[bool, str]:
        """Probe the actual ``/v1/systemone`` route with a minimal typed question."""
        try:
            answer = self.decide(
                "The sky is blue.",
                {"type": "noul", "instructions": "Is the sky described as blue?"},
                deadline=Deadline(timeout),
            )
        except TransportError as exc:
            return False, str(exc)
        value = answer.get("noul")
        if not isinstance(value, int | float) or not 0.0 <= float(value) <= 1.0:
            return False, "/v1/systemone noul answer is not a probability"
        return True, "ok"
