"""DecisionAdvisor: the single owner of every JevK5 call.

JevK5 is an optional *advisory* classifier. Every method returns an
:class:`Advice` (or ``None`` when the advisory must be ignored) and never
raises into a caller: a timeout, an open circuit, an exhausted per-request
call budget, a missing answer letter, or low confidence all mean "no advice",
and the caller's deterministic behavior continues unchanged.

Per decision:

* one monotonic deadline (default 1200 ms) covers ``/tokenize``, every
  ``/completion`` pass, the missing-letter retry and knockout passes;
* a short-lived ``JevK5GGUF`` instance per decision (no shared mutable
  adapter state between concurrent requests);
* if a required option letter is absent from ``top_logprobs`` the decision is
  retried once with ``n_probs`` raised to the retry bound (default 128); still
  missing → the advisory is unavailable for that decision (a failure);
* top-choice confidence below ``min_confidence`` (0.60) is ignored;
* the utility weight is ``min(weight, weight_cap)`` (0.25, cap 0.40).

Across decisions: a 30 s TTL cache (a cached answer is held to the same
confidence floor as a fresh one), at most N calls per request — one
:class:`RequestBudget` shared by every consumer that passes the same request
key — and a circuit breaker that opens for 60 s after 3 consecutive failures.
JevK5 is English-only: a decision whose text is mostly non-Latin script is not
sent. A state that does not fit the running server's context is refused for
that decision only (it says nothing about service health). Raw state text is
never logged or stored.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import logging
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from headroom.agent.interfaces import DecisionHints
from headroom.intelligence.settings import IntelligenceSettings, ServicePolicy
from headroom.intelligence.transport import (
    ContextOverflow,
    Deadline,
    DeadlineExceeded,
    SystemOneClient,
    TransportError,
    make_deadline_adapter,
)

log = logging.getLogger(__name__)

# Rough upper bound used before tokenization: never send a state that cannot
# fit the configured local-decision budget.
_CHARS_PER_TOKEN = 3.0
# Request-keyed budgets live this long (a request never takes longer) and the
# table is bounded so abandoned keys cannot grow it.
_BUDGET_TTL_S = 300.0
_BUDGET_MAX_KEYS = 4096
# English-only gate: at least this share of the letters must be Latin script.
_MIN_LATIN_SHARE = 0.85
_MIN_LETTERS_FOR_GATE = 12


def english_enough(*texts: str) -> bool:
    """Whether the decision text is predominantly Latin-script (JevK5 is English-only).

    Code, paths and numbers carry few letters and pass; prose in a non-Latin
    script (CJK, Cyrillic, Arabic, ...) does not. Accented Latin letters count
    as Latin so names and loanwords in English text are not penalized.
    """
    latin = other = 0
    for text in texts:
        for ch in text:
            if not ch.isalpha():
                continue
            if ch.isascii() or "À" <= ch <= "ɏ":
                latin += 1
            else:
                other += 1
    total = latin + other
    if total < _MIN_LETTERS_FOR_GATE:
        return True
    return latin / total >= _MIN_LATIN_SHARE


def _question_text(question: Mapping[str, Any]) -> list[str]:
    parts = [str(question.get("instructions") or "")]
    criteria = question.get("criteria")
    if isinstance(criteria, Mapping):
        parts.extend(str(v) for v in criteria.values())
    elif isinstance(criteria, list):
        parts.extend(str(v) for v in criteria)
    return parts


class MissingLetters(TransportError):
    """Required answer letters were absent from top_logprobs even after retry."""


@dataclass(frozen=True)
class Advice:
    family: str
    kind: str  # "noul" | "choice" | "score"
    value: Any  # P(true) | chosen option id | expected score
    confidence: float
    probabilities: Mapping[str, float]
    weight: float
    source: str  # "llama-server" | "systemone"
    latency_ms: float
    cached: bool = False
    # Metrics of THIS decision only, read from its own short-lived adapter.
    input_tokens: int = 0
    pass_tokens: int = 0
    missing_retried: bool = False

    def to_hints(self, query_terms: Sequence[str] = ()) -> DecisionHints:
        return DecisionHints(
            family=self.family,
            source="jevk5",
            query_terms=tuple(query_terms),
            advisory=dict(self.probabilities),
            choice=str(self.value) if self.kind == "choice" else None,
            confidence=self.confidence,
            weight=self.weight,
        )


class RequestBudget:
    """Per-request advisor call allowance (default 4). Thread-safe."""

    def __init__(self, calls: int) -> None:
        self._remaining = max(0, int(calls))
        self._lock = threading.Lock()

    def take(self) -> bool:
        with self._lock:
            if self._remaining <= 0:
                return False
            self._remaining -= 1
            return True

    @property
    def remaining(self) -> int:
        with self._lock:
            return self._remaining


@dataclass
class _Breaker:
    failures_to_open: int
    open_seconds: float
    consecutive_failures: int = 0
    opened_at: float | None = None
    trial_in_flight: bool = False

    def allow(self, now: float) -> bool:
        if self.opened_at is None:
            return True
        if now - self.opened_at < self.open_seconds:
            return False
        if self.trial_in_flight:
            return False
        self.trial_in_flight = True  # half-open: exactly one trial
        return True

    def success(self) -> None:
        self.consecutive_failures = 0
        self.opened_at = None
        self.trial_in_flight = False

    def release_trial(self) -> None:
        """End a half-open trial that said nothing about service health."""
        self.trial_in_flight = False

    def failure(self, now: float) -> None:
        self.consecutive_failures += 1
        self.trial_in_flight = False
        if self.opened_at is not None or self.consecutive_failures >= self.failures_to_open:
            self.opened_at = now

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        return "half_open" if self.trial_in_flight else "open"


@dataclass
class AdvisorStats:
    calls: int = 0
    cache_hits: int = 0
    successes: int = 0
    failures: int = 0
    timeouts: int = 0
    low_confidence: int = 0
    missing_letter_retries: int = 0
    missing_letter_failures: int = 0
    budget_exhausted: int = 0
    breaker_rejections: int = 0
    unavailable: int = 0
    non_english: int = 0
    context_overflow: int = 0
    last_failure: str | None = None
    last_latency_ms: float | None = None
    by_family: dict[str, int] = field(default_factory=dict)


Endpoint = tuple[str, str]  # (kind, base_url) where kind is "llama-server" | "systemone"


class DecisionAdvisor:
    """See module docstring. ``endpoint_provider`` returns the live endpoint or None."""

    def __init__(
        self,
        settings: IntelligenceSettings,
        endpoint_provider: Callable[[], Endpoint | None],
        *,
        clock: Callable[[], float] = time.monotonic,
        adapter_factory: Callable[..., Any] = make_deadline_adapter,
        systemone_factory: Callable[[str], SystemOneClient] = SystemOneClient,
        effective_ctx: Callable[[], int | None] | None = None,
    ) -> None:
        self.settings = settings
        self._endpoint_provider = endpoint_provider
        self._clock = clock
        self._adapter_factory = adapter_factory
        self._systemone_factory = systemone_factory
        self._effective_ctx = effective_ctx or (lambda: None)
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, tuple[float, Advice]] = OrderedDict()
        self._breaker = _Breaker(settings.breaker_failures, settings.breaker_open_s)
        self.stats = AdvisorStats()
        self.missing_letter_unhealthy = False
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._budgets: OrderedDict[str, tuple[float, RequestBudget]] = OrderedDict()

    # -- public typed API ----------------------------------------------------

    def request_budget(self, request_key: str | None = None) -> RequestBudget:
        """The call allowance for one request.

        Every consumer that passes the same ``request_key`` (the proxy request
        id) draws from ONE budget, so all JevK5 consumers of a request together
        make at most ``calls_per_request`` calls. Without a key the budget is
        private to the caller (an off-request decision such as a host hook).
        """
        if not request_key:
            return RequestBudget(self.settings.calls_per_request)
        now = self._clock()
        with self._lock:
            entry = self._budgets.get(request_key)
            if entry is None or now - entry[0] > _BUDGET_TTL_S:
                entry = (now, RequestBudget(self.settings.calls_per_request))
                self._budgets[request_key] = entry
            self._budgets.move_to_end(request_key)
            while len(self._budgets) > _BUDGET_MAX_KEYS:
                self._budgets.popitem(last=False)
            return entry[1]

    def noul(
        self, family: str, state: Any, instructions: str, *, budget: RequestBudget
    ) -> Advice | None:
        return self.decide(family, state, {"type": "noul", "instructions": instructions}, budget)

    def choose(
        self,
        family: str,
        state: Any,
        instructions: str,
        options: Sequence[str] | Mapping[str, str],
        *,
        budget: RequestBudget,
    ) -> Advice | None:
        criteria = dict(options) if isinstance(options, Mapping) else list(options)
        return self.decide(
            family, state, {"type": "choice", "instructions": instructions, "criteria": criteria},
            budget,
        )

    def score(
        self,
        family: str,
        state: Any,
        instructions: str,
        levels: Sequence[str],
        *,
        budget: RequestBudget,
    ) -> Advice | None:
        return self.decide(
            family, state, {"type": "score", "instructions": instructions, "criteria": list(levels)},
            budget,
        )

    async def adecide(
        self, family: str, state: Any, question: Mapping[str, Any], budget: RequestBudget
    ) -> Advice | None:
        """Event-loop-safe variant: runs on a small bounded pool, late results dropped."""
        loop = asyncio.get_running_loop()
        with self._lock:
            if self._executor is None:
                self._executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="headroom-jevk5"
                )
            executor = self._executor
        future = loop.run_in_executor(executor, self.decide, family, state, question, budget)
        try:
            return await asyncio.wait_for(
                future, timeout=self.settings.decision_timeout_ms / 1000.0 + 0.05
            )
        except asyncio.TimeoutError:
            return None

    # -- core ----------------------------------------------------------------

    def _validate(self, question: Mapping[str, Any]) -> str | None:
        kind = question.get("type")
        if kind not in ("noul", "choice", "score"):
            return "unknown question type"
        if not isinstance(question.get("instructions"), str) or not question["instructions"]:
            return "missing instructions"
        criteria = question.get("criteria")
        if kind == "choice":
            count = len(criteria) if isinstance(criteria, list | dict) else 0
            if count < 2 or count > self.settings.max_options:
                return "choice needs 2..max_options options"
        if kind == "score" and (not isinstance(criteria, list) or len(criteria) < 2):
            return "score needs at least two levels"
        return None

    def _cache_key(self, family: str, state: Any, question: Mapping[str, Any]) -> str:
        raw = json.dumps(
            {"f": family, "s": state, "q": question}, sort_keys=True, default=str
        ).encode("utf-8", "surrogatepass")
        return hashlib.sha256(raw).hexdigest()

    def decide(
        self,
        family: str,
        state: Any,
        question: Mapping[str, Any],
        budget: RequestBudget,
    ) -> Advice | None:
        settings = self.settings
        if settings.policy is ServicePolicy.OFF:
            return None
        problem = self._validate(question)
        if problem:
            log.debug("advisor rejected malformed question (%s)", problem)
            return None
        state_text = state if isinstance(state, str) else json.dumps(
            state, default=str, ensure_ascii=False
        )
        if len(state_text) / _CHARS_PER_TOKEN > settings.max_state_tokens * 1.5:
            return None  # cannot fit the local-decision budget; do not send
        if not english_enough(state_text, *_question_text(question)):
            with self._lock:
                self.stats.non_english += 1
            return None  # JevK5 is English-only: no advice rather than noise
        key = self._cache_key(family, state, question)
        now = self._clock()
        with self._lock:
            cached = self._cache.get(key)
            if cached and now - cached[0] <= settings.cache_ttl_s:
                self._cache.move_to_end(key)
                self.stats.cache_hits += 1
                if cached[1].confidence < settings.min_confidence:
                    self.stats.low_confidence += 1
                    return None  # the floor applies to a cached answer too
                return _replace_cached(cached[1])
        if not budget.take():
            with self._lock:
                self.stats.budget_exhausted += 1
            return None
        endpoint = self._endpoint_provider()
        if endpoint is None:
            with self._lock:
                self.stats.unavailable += 1
            return None
        with self._lock:
            if not self._breaker.allow(now):
                self.stats.breaker_rejections += 1
                return None
            self.stats.calls += 1
            self.stats.by_family[family] = self.stats.by_family.get(family, 0) + 1
        deadline = Deadline(settings.decision_timeout_ms / 1000.0, clock=self._clock)
        try:
            answer, metrics = self._call(endpoint, state, question, deadline)
            latency = deadline.elapsed_ms()
            if deadline.remaining() <= 0:
                raise DeadlineExceeded("result arrived after the decision deadline")
            advice = _advice_from_answer(
                family, question, answer, endpoint[0], latency, settings.effective_weight,
                metrics,
            )
        except DeadlineExceeded as exc:
            self._record_failure(exc, timeout=True)
            return None
        except ContextOverflow:
            # This state is too large for the running context: refuse this one
            # decision. It is not a service failure, so the breaker is untouched.
            with self._lock:
                self._breaker.release_trial()
                self.stats.context_overflow += 1
            return None
        except MissingLetters as exc:
            with self._lock:
                self.stats.missing_letter_failures += 1
                self.missing_letter_unhealthy = True
            self._record_failure(exc)
            return None
        except (TransportError, ValueError, KeyError, TypeError, ImportError) as exc:
            # A socket timeout sized to the remaining budget is the deadline firing.
            self._record_failure(exc, timeout=deadline.remaining() <= 0.01)
            return None
        with self._lock:
            self._breaker.success()
            self.stats.successes += 1
            self.stats.last_latency_ms = advice.latency_ms
            self._cache[key] = (self._clock(), advice)
            self._cache.move_to_end(key)
            while len(self._cache) > settings.cache_max_entries:
                self._cache.popitem(last=False)
        if advice.confidence < settings.min_confidence:
            with self._lock:
                self.stats.low_confidence += 1
            return None
        return advice

    def _record_failure(self, exc: BaseException, *, timeout: bool = False) -> None:
        with self._lock:
            self.stats.failures += 1
            if timeout:
                self.stats.timeouts += 1
            self.stats.last_failure = type(exc).__name__
            self._breaker.failure(self._clock())

    def _call(
        self, endpoint: Endpoint, state: Any, question: Mapping[str, Any], deadline: Deadline
    ) -> tuple[dict, dict[str, Any]]:
        """One decision; returns ``(answer, this decision's own adapter metrics)``."""
        kind, url = endpoint
        if kind == "systemone":
            answer = self._systemone_factory(url).decide(state, question, deadline=deadline)
            return answer, {"input_tokens": int(answer.get("input_tokens") or 0)}
        ctx = self._effective_ctx() or self.settings.ctx_size
        max_prompt = min(ctx, self.settings.ctx_size)
        adapter = self._adapter_factory(
            url,
            deadline=deadline,
            temperature=self.settings.temperature,
            knockout_temperature=self.settings.knockout_temperature,
            top_k=self.settings.top_k,
            max_prompt_tokens=max_prompt,
        )
        answer = adapter.decide(state, dict(question))
        used = adapter
        if getattr(adapter, "missing", 0):
            with self._lock:
                self.stats.missing_letter_retries += 1
            retry = self._adapter_factory(
                url,
                deadline=deadline,
                temperature=self.settings.temperature,
                knockout_temperature=self.settings.knockout_temperature,
                top_k=self.settings.missing_retry_n_probs,
                max_prompt_tokens=max_prompt,
            )
            answer = retry.decide(state, dict(question))
            used = retry
            if getattr(retry, "missing", 0):
                raise MissingLetters("answer letters absent from top_logprobs after retry")
        with self._lock:
            self.missing_letter_unhealthy = False
        return answer, {
            "input_tokens": int(answer.get("input_tokens") or 0),
            "pass_tokens": int(getattr(used, "last_pass_tokens", 0) or 0),
            "missing_retried": used is not adapter,
        }

    # -- diagnostics ---------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._lock:
            stats = self.stats
            return {
                "policy": self.settings.policy.value,
                "breaker": self._breaker.state,
                "consecutive_failures": self._breaker.consecutive_failures,
                "missing_letter_unhealthy": self.missing_letter_unhealthy,
                "cache_entries": len(self._cache),
                "stats": {
                    "calls": stats.calls,
                    "cache_hits": stats.cache_hits,
                    "successes": stats.successes,
                    "failures": stats.failures,
                    "timeouts": stats.timeouts,
                    "low_confidence": stats.low_confidence,
                    "missing_letter_retries": stats.missing_letter_retries,
                    "missing_letter_failures": stats.missing_letter_failures,
                    "budget_exhausted": stats.budget_exhausted,
                    "breaker_rejections": stats.breaker_rejections,
                    "unavailable": stats.unavailable,
                    "non_english": stats.non_english,
                    "context_overflow": stats.context_overflow,
                    "last_failure": stats.last_failure,
                    "last_latency_ms": stats.last_latency_ms,
                },
                "limits": {
                    "decision_timeout_ms": self.settings.decision_timeout_ms,
                    "calls_per_request": self.settings.calls_per_request,
                    "cache_ttl_s": self.settings.cache_ttl_s,
                    "min_confidence": self.settings.min_confidence,
                    "weight": self.settings.effective_weight,
                    "breaker_failures": self.settings.breaker_failures,
                    "breaker_open_s": self.settings.breaker_open_s,
                },
            }

    def shutdown(self) -> None:
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


def _replace_cached(advice: Advice) -> Advice:
    return Advice(**{**advice.__dict__, "cached": True})


def _finite_prob(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number) or number < -1e-6 or number > 1 + 1e-6:
        raise ValueError("advisor returned a non-probability")
    return min(1.0, max(0.0, number))


def _advice_from_answer(
    family: str,
    question: Mapping[str, Any],
    answer: Mapping[str, Any],
    source: str,
    latency_ms: float,
    weight: float,
    metrics: Mapping[str, Any] | None = None,
) -> Advice:
    kind = str(question["type"])
    if answer.get("type") != kind:
        raise ValueError("answer type does not match the question")
    if kind == "noul":
        p_true = _finite_prob(answer["noul"])
        probabilities = {"true": p_true, "false": 1.0 - p_true}
        value: Any = p_true
    else:
        raw = answer.get("probabilities")
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError("answer has no probabilities")
        probabilities = {str(k): _finite_prob(v) for k, v in raw.items()}
        total = sum(probabilities.values())
        if not math.isfinite(total) or abs(total - 1.0) > 1e-3:
            raise ValueError("answer probabilities are not normalized")
        if kind == "choice":
            value = str(answer.get("choice") or max(probabilities, key=probabilities.__getitem__))
            if value not in probabilities:
                raise ValueError("choice is not one of the options")
        else:
            value = float(answer["score"])
            if not math.isfinite(value):
                raise ValueError("score is not finite")
    confidence = _finite_prob(max(probabilities.values()))
    return Advice(
        family=family,
        kind=kind,
        value=value,
        confidence=confidence,
        probabilities=probabilities,
        weight=weight,
        source=source,
        latency_ms=latency_ms,
        input_tokens=int((metrics or {}).get("input_tokens") or 0),
        pass_tokens=int((metrics or {}).get("pass_tokens") or 0),
        missing_retried=bool((metrics or {}).get("missing_retried")),
    )


__all__ = [
    "Advice",
    "AdvisorStats",
    "ContextOverflow",
    "DecisionAdvisor",
    "MissingLetters",
    "RequestBudget",
    "english_enough",
]
