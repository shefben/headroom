"""Interception matrix for change scope (#23) and tool contracts (#19).

Where a decision may take effect depends on who executes the tool:

* **Trusted host pre-tool boundary** — Claude Code's ``PreToolUse`` command
  hook that ``headroom wrap claude`` installs (:mod:`headroom.host_hook`,
  forwarded to ``POST /v1/agent/host-hook``). The hook runs *before* the host
  executes the call, so a decision here is enforceable: a FORBIDDEN change
  answers ``permissionDecision: "deny"`` with a stable reason; warnings and
  repair suggestions travel as compact ``additionalContext`` (no permission
  field, so the host's own permission prompts are unaffected); everything
  else prints nothing. Nothing ever rewrites the tool input.
* **Proxy / TurnHook path** — every client, but only after the client already
  executed the calls (their results are in the request). It is
  **observe-only**: decisions are recorded with ``enforcement="observe"``,
  never reported as "blocked", and feed the after-edit scope check and the
  learned-recovery observations.

Fail-open everywhere except an explicitly enforceable hard-protected action at
the trusted host boundary for a trusted (wrap-registered) workspace.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

from headroom.agent.change_scope import (
    FEATURE_CHANGE_SCOPE,
    PlannedChange,
    ScopeDecision,
    Verdict,
    describe,
    planned_changes,
)
from headroom.log_safety import describe_exception

log = logging.getLogger(__name__)

FEATURE_TOOL_CONTRACTS = "agent_tool_contracts"
HOST_PRETOOL = "host_pretool"
OBSERVE_ONLY = "observe_only"
STAGE_OBSERVE = "agent_enforcement_observe"
LISTENER_WRITES = "change_scope_writes"
PREFIX = "Headroom"
MAX_CONTEXT_CHARS = 600
_MAX_OBSERVED_PER_REQUEST = 32
_SEEN_LIMIT = 8192

#: Which clients expose a genuine pre-tool hook Headroom installs.
ENFORCEMENT_CAPABILITIES: dict[str, str] = {
    "claude": HOST_PRETOOL,
    "codex": OBSERVE_ONLY,
    "opencode": OBSERVE_ONLY,
    "copilot": OBSERVE_ONLY,
    "cursor": OBSERVE_ONLY,
    "aider": OBSERVE_ONLY,
    "gemini": OBSERVE_ONLY,
}

_lock = threading.Lock()
_host_evaluated: OrderedDict[str, str] = OrderedDict()
_seen_calls: OrderedDict[str, None] = OrderedDict()
_counters: dict[str, int] = {
    "host_pretool_calls": 0,
    "host_prevented": 0,
    "host_warned": 0,
    "host_repair_suggested": 0,
    "host_allowed": 0,
    "observed_calls": 0,
    "observed_failures": 0,
    "observed_would_block": 0,
    "observed_warn": 0,
    "warned_then_failed": 0,
    "prevented_then_retried": 0,
}


def enforcement_capability(client: str, *, hook_installed: bool = True) -> str:
    cap = ENFORCEMENT_CAPABILITIES.get((client or "").lower(), OBSERVE_ONLY)
    if cap == HOST_PRETOOL and not hook_installed:
        return OBSERVE_ONLY
    return cap


def _count(name: str, n: int = 1) -> None:
    with _lock:
        _counters[name] = _counters.get(name, 0) + n


def counters() -> dict[str, int]:
    with _lock:
        return dict(_counters)


def reset() -> None:
    with _lock:
        _host_evaluated.clear()
        _seen_calls.clear()
        for key in _counters:
            _counters[key] = 0


def _remember_host(call_id: str, verdict: str) -> None:
    if not call_id:
        return
    with _lock:
        _host_evaluated[call_id] = verdict
        _host_evaluated.move_to_end(call_id)
        while len(_host_evaluated) > _SEEN_LIMIT:
            _host_evaluated.popitem(last=False)


def host_verdict(call_id: str) -> str | None:
    with _lock:
        return _host_evaluated.get(call_id)


def _services(runtime: Any) -> tuple[Any, Any]:
    if runtime is None:
        return None, None
    return runtime.services.get("change_scope"), runtime.services.get("tool_contracts")


def active_runtime() -> Any:
    from headroom.agent.install import _installed

    return _installed.get("runtime")


# ---------------------------------------------------------------------------
# Host pre-tool boundary
# ---------------------------------------------------------------------------


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= MAX_CONTEXT_CHARS else text[: MAX_CONTEXT_CHARS - 1] + "…"


def render_pretool(kind: str, message: str) -> dict[str, Any] | None:
    """Claude Code ``PreToolUse`` hook stdout for a decision kind.

    ``BLOCK`` denies (the only permission field Headroom ever sets);
    ``WARN``/``REPAIR_SUGGEST`` add context only; ``ALLOW`` prints nothing.
    """
    if kind == "BLOCK":
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": _clip(f"{PREFIX} blocked this call: {message}"),
            }
        }
    if kind in ("WARN", "REPAIR_SUGGEST") and message:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "additionalContext": _clip(f"{PREFIX} tool check: {message}"),
            }
        }
    return None


def scope_decisions(
    firewall: Any,
    identity: Any,
    tool_name: str,
    tool_input: Mapping[str, Any],
    *,
    cwd: str | None,
    enforceable: bool,
    observe: bool = False,
) -> list[ScopeDecision]:
    """Scope decisions for every planned change of one call."""
    from headroom.agent.change_scope import shell_cwd, shell_of

    changes: list[PlannedChange] = planned_changes(tool_name, tool_input)
    if not changes:
        return []
    call_cwd = cwd
    command = tool_input.get("command") if isinstance(tool_input, Mapping) else None
    if isinstance(command, str):
        call_cwd = shell_cwd(command, cwd, shell_of(tool_name)) or cwd
    out: list[ScopeDecision] = []
    for change in changes[:16]:
        if observe:
            out.append(firewall.observe_change(identity, change, cwd=call_cwd))
        else:
            out.append(firewall.authorize(identity, change, cwd=call_cwd, enforceable=enforceable))
    return out


def summarize_scope(decisions: Sequence[ScopeDecision]) -> tuple[str, str, str]:
    """(kind, reason_code, message) of the most severe scope decision."""
    order = {Verdict.BLOCK: 2, Verdict.WARN: 1, Verdict.ALLOW: 0}
    if not decisions:
        return "ALLOW", "scope.none", ""
    worst = max(decisions, key=lambda d: order[d.verdict])
    if worst.verdict is Verdict.ALLOW:
        return "ALLOW", worst.reason_code, ""
    if worst.reason_code == "scope.untrusted_workspace":
        return "WARN", worst.reason_code, ""  # never surfaced: nothing to tell the agent
    return worst.verdict.value, worst.reason_code, describe(worst) + f" [{worst.reason_code}]"


def host_pretool(
    payload: Mapping[str, Any],
    *,
    identity: Any,
    client: str,
    runtime: Any = None,
) -> dict[str, Any]:
    """Evaluate one ``PreToolUse`` payload at the trusted host boundary."""
    runtime = runtime if runtime is not None else active_runtime()
    firewall, validator = _services(runtime)
    tool_name = str(payload.get("tool_name") or "")
    tool_input = payload.get("tool_input") if isinstance(payload.get("tool_input"), dict) else {}
    call_id = str(payload.get("tool_use_id") or "")
    cwd = str(payload.get("cwd") or "") or None
    capability = enforcement_capability(client)
    _count("host_pretool_calls")
    if firewall is None and validator is None:
        return {"output": None, "stage": "runtime_unavailable", "capability": capability}
    # Enforceable only at a genuine pre-tool host for a trusted workspace.
    enforceable = capability == HOST_PRETOOL and bool(identity.can_authorize_paths())
    if validator is not None:
        decision = validator.evaluate(
            tool_name,
            tool_input,
            identity=identity,
            source=client,
            cwd=cwd,
            enforcement=HOST_PRETOOL if enforceable else OBSERVE_ONLY,
            call_id=call_id,
        )
        kind, reason, message = decision.kind.value, decision.reason_code, decision.message()
        details: dict[str, Any] = decision.to_dict()
    else:
        decisions = scope_decisions(
            firewall, identity, tool_name, tool_input, cwd=cwd, enforceable=enforceable
        )
        kind, reason, message = summarize_scope(decisions)
        details = {"scope": [d.to_dict() for d in decisions]}
    if kind == "BLOCK" and not enforceable:
        kind = "WARN"  # never claim a block the boundary cannot enforce
    _remember_host(call_id, kind)
    _count(
        {
            "BLOCK": "host_prevented",
            "WARN": "host_warned",
            "REPAIR_SUGGEST": "host_repair_suggested",
        }.get(kind, "host_allowed")
    )
    return {
        "output": render_pretool(kind, message),
        "stage": f"pretool:{kind.lower()}",
        "capability": capability,
        "reason_code": reason,
        "decision": details,
    }


# ---------------------------------------------------------------------------
# Proxy path (observe only)
# ---------------------------------------------------------------------------


def executed_calls(provider: str, messages: Any) -> list[dict[str, Any]]:
    """``{call_id, tool_name, input, output, is_error}`` of executed calls, in order."""
    if not isinstance(messages, list):
        return []
    extractor = _extractor()
    try:
        if provider == "anthropic":
            return extractor.extract_tool_results_from_messages(messages)
        if provider == "openai":
            return extractor.extract_tool_results_from_openai_messages(messages)
    except Exception as exc:  # noqa: BLE001 - observation must never break a request
        log.debug("tool call extraction failed: %s", describe_exception(exc))
    return []


_EXTRACTOR: Any = None


def _extractor() -> Any:
    """One backend-less TrafficLearner reused for its pure extract/pair helpers."""
    global _EXTRACTOR
    if _EXTRACTOR is None:
        from headroom.memory.traffic_learner import TrafficLearner

        _EXTRACTOR = TrafficLearner(backend=None, agent_type="headroom-enforcement")
    return _EXTRACTOR


def _fresh(session_id: str, calls: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with _lock:
        for call in calls:
            call_id = str(call.get("call_id") or "")
            if not call_id:
                continue
            key = f"{session_id}|{call_id}"
            if key in _seen_calls:
                continue
            _seen_calls[key] = None
            out.append(call)
        while len(_seen_calls) > _SEEN_LIMIT:
            _seen_calls.popitem(last=False)
    return out[-_MAX_OBSERVED_PER_REQUEST:]


def observe_calls(
    runtime: Any, identity: Any, calls: Sequence[dict[str, Any]], tools: Any = None
) -> None:
    """Worker-thread side of the proxy path: classify executed calls (observe only)."""
    firewall, validator = _services(runtime)
    for call in calls:
        tool_name = str(call.get("tool_name") or "")
        tool_input = call.get("input") if isinstance(call.get("input"), dict) else {}
        call_id = str(call.get("call_id") or "")
        failed = bool(call.get("is_error"))
        prior = host_verdict(call_id)
        _count("observed_calls")
        if failed:
            _count("observed_failures")
            if prior in ("WARN", "REPAIR_SUGGEST"):
                _count("warned_then_failed")
        if validator is not None:
            validator.observe(
                tool_name,
                tool_input,
                identity=identity,
                call_id=call_id,
                output=str(call.get("output") or ""),
                is_error=failed,
                evaluated_at_host=prior is not None,
                tools=tools,
            )
        if prior is not None or firewall is None or failed:
            continue  # the host already decided, or nothing was changed
        decisions = scope_decisions(
            firewall,
            identity,
            tool_name,
            tool_input,
            cwd=identity.root,
            enforceable=False,
            observe=True,
        )
        kind, _, _ = summarize_scope(decisions)
        if any(d.verdict is Verdict.WARN and d.scope.value == "FORBIDDEN" for d in decisions):
            _count("observed_would_block")
        elif kind == "WARN":
            _count("observed_warn")
    if firewall is not None:
        firewall.check_diff(identity)


def observe_stage(ctx: Any, task: Any, runtime: Any) -> None:
    """Request stage (priority-80 agent hook): queue executed calls for observation."""
    from headroom.agent.hook import identity_for

    identity = identity_for(ctx)
    if identity is None or not identity.trusted:
        return
    calls = _fresh(identity.session_id, executed_calls(str(ctx.provider or ""), ctx.messages))
    tools = ctx.tools
    if not calls:
        runtime.submit_task(lambda: _diff_only(runtime, identity))
        return
    runtime.submit_task(lambda: observe_calls(runtime, identity, calls, tools))


def _diff_only(runtime: Any, identity: Any) -> None:
    firewall, _ = _services(runtime)
    if firewall is not None:
        firewall.check_diff(identity)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def _feature(config: Any, name: str) -> bool:
    rollout = getattr(config, "rollout", None)
    try:
        return bool(rollout is not None and rollout.is_enabled(name))
    except KeyError:
        return False


def enforcement_health(config: Any) -> dict[str, Any]:
    """``/health.agent_enforcement``: what each path can actually do (counts only)."""
    runtime = active_runtime()
    firewall, validator = _services(runtime)
    block: dict[str, Any] = {
        "change_scope_enabled": _feature(config, FEATURE_CHANGE_SCOPE),
        "tool_contracts_enabled": _feature(config, FEATURE_TOOL_CONTRACTS),
        "capabilities": dict(ENFORCEMENT_CAPABILITIES),
        "proxy_path": OBSERVE_ONLY,
        "counters": counters(),
    }
    if firewall is not None:
        block["change_scope"] = firewall.status()
    if validator is not None:
        block["tool_contracts"] = validator.status()
    return block
