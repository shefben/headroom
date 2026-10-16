"""Claude Code PreToolUse/PostToolUse command hook installed by ``headroom wrap claude``.

Usage (written into the hook ``command`` by wrap)::

    <python> -m headroom.host_hook claude [--proxy http://127.0.0.1:8787]

It reads the hook JSON Claude Code writes on stdin, forwards it to the local
Headroom proxy (``POST /v1/agent/host-hook``) together with the wrap session
token Claude Code already carries in ``ANTHROPIC_CUSTOM_HEADERS``, and prints
the JSON the proxy returns (if any) on stdout.

Fail-open by construction: a missing/unreachable proxy, a non-loopback proxy
URL, a timeout, malformed input or any error prints nothing and exits 0, so
Claude Code proceeds exactly as if no hook were installed. The hook itself
never decides anything (never exit code 2): the only permission field it can
ever print is the proxy's ``deny`` for an explicitly forbidden or
hard-protected change in a trusted wrap workspace (unified-21 Phase 6). It
never sends tool data to a non-loopback address. Standard library only, so
the per-call start-up cost stays small.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

ENDPOINT = "/v1/agent/host-hook"
PROXY_ENV = "HEADROOM_HOOK_PROXY_URL"
DEBUG_ENV = "HEADROOM_HOOK_DEBUG"
MAX_STDIN_BYTES = 64 * 1024 * 1024
#: Below this serialized size a result is never worth a round trip.
MIN_RESULT_CHARS = 1500
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})
_POST_TOOLS = frozenset({"Bash", "PowerShell", "Grep", "Glob"})
_PRE_TOOLS = frozenset(
    {
        "Edit",
        "Write",
        "MultiEdit",
        "NotebookEdit",
        "Bash",
        "PowerShell",
        # Unified-21 #20: the workflow tool runs only behind this boundary.
        "mcp__headroom__headroom_workflow",
    }
)
_URL_ENVS = (
    PROXY_ENV,
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_VERTEX_BASE_URL",
    "ANTHROPIC_FOUNDRY_BASE_URL",
)


def _debug(message: str) -> None:
    if os.environ.get(DEBUG_ENV):
        sys.stderr.write(f"headroom host hook: {message}\n")


def loopback_base(url: str | None) -> str | None:
    """``scheme://host:port`` of ``url`` when it names a loopback host, else None."""
    if not url:
        return None
    try:
        parts = urllib.parse.urlsplit(url.strip())
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or (parts.hostname or "").lower() not in _LOOPBACK:
        return None
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError:
        return None
    return f"{parts.scheme}://{host}" + (f":{port}" if port else "")


def custom_headers(env: Mapping[str, str]) -> dict[str, str]:
    """Headroom headers from ``ANTHROPIC_CUSTOM_HEADERS`` (``Name: value`` lines)."""
    out: dict[str, str] = {}
    for line in (env.get("ANTHROPIC_CUSTOM_HEADERS") or "").splitlines():
        name, sep, value = line.partition(":")
        if not sep:
            continue
        key = name.strip().lower()
        if key in ("x-headroom-session-token", "x-headroom-project"):
            out[key] = value.strip()
    return out


def proxy_base(argv: Sequence[str], env: Mapping[str, str]) -> str | None:
    explicit = env.get(PROXY_ENV)
    if explicit:
        return loopback_base(explicit)
    if "--proxy" in argv:
        idx = list(argv).index("--proxy")
        if idx + 1 < len(argv):
            found = loopback_base(argv[idx + 1])
            if found:
                return found
    for name in _URL_ENVS[1:]:
        found = loopback_base(env.get(name))
        if found:
            return found
    return None


def worth_forwarding(payload: Mapping[str, Any]) -> bool:
    event = payload.get("hook_event_name")
    tool = str(payload.get("tool_name") or "")
    if event == "PreToolUse":
        return tool in _PRE_TOOLS
    if event != "PostToolUse":
        return False
    if tool.startswith("mcp__headroom"):
        return False
    if tool not in _POST_TOOLS and not tool.startswith("mcp__"):
        return False
    try:
        size = len(json.dumps(payload.get("tool_response"), ensure_ascii=False))
    except (TypeError, ValueError):
        return False
    return size >= MIN_RESULT_CHARS


def forward(
    base: str, client: str, payload: Mapping[str, Any], env: Mapping[str, str], timeout: float
) -> Any:
    headers = {"Content-Type": "application/json"}
    for key, value in custom_headers(env).items():
        headers[key] = value
    body = json.dumps({"client": client, "payload": payload}).encode("utf-8")
    request = urllib.request.Request(base + ENDPOINT, data=body, headers=headers, method="POST")
    # Loopback only: never route the payload through an environment HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data.get("output") if isinstance(data, dict) else None


def main(argv: Sequence[str] | None = None, *, stdin: Any = None, stdout: Any = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    client = args[0] if args and not args[0].startswith("-") else "claude"
    env = os.environ
    reader = stdin if stdin is not None else sys.stdin.buffer
    writer = stdout if stdout is not None else sys.stdout
    try:
        raw = reader.read(MAX_STDIN_BYTES + 1)
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        if len(raw) > MAX_STDIN_BYTES:
            _debug("payload too large; passing through")
            return 0
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict) or not worth_forwarding(payload):
            return 0
        base = proxy_base(args, env)
        if base is None:
            _debug("no loopback Headroom proxy configured; passing through")
            return 0
        timeout = 3.0 if payload.get("hook_event_name") == "PreToolUse" else 20.0
        output = forward(base, client, payload, env, timeout)
        if isinstance(output, dict) and output:
            writer.write(json.dumps(output))  # ASCII-safe on any console encoding
            writer.flush()
    except Exception as exc:  # noqa: BLE001 - fail open: the host proceeds unchanged
        _debug(f"{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
