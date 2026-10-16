"""Complexity-aware effort and verbosity routing (unified-21 #10).

Two independent decisions per request:

* **Reasoning effort** — only ever *lowered*, and only an effort value the
  client already sent (Anthropic ``output_config.effort``, Responses
  ``reasoning.effort``, chat ``reasoning_effort``). No thinking parameter is
  ever added, and ``thinking`` is never toggled. Effort is never lowered for
  a new user ask, an error/failed retry, novel errors, contradicted evidence,
  an unresolved completion gate, or an explicit user request for deep
  thinking. On a mechanical continuation the remaining complexity signals
  (new symbols, multi-file scope, unknown requirements, uncertainty) give a
  0-4 score; JevK5 scores 0-4 only when the deterministic score is the
  uncertain middle. A short answer with high effort is a valid outcome.
* **Visible verbosity** — the output shaper's conversation-stable level
  (verbosity learning, controller, env; system-tail placement) stays the
  base. A complex new ask may *relax* it for that one reply through a
  trailing per-turn block placed after the live turn (never inside the
  cached system prompt), only on models that accept it.

Both levers run only for the ``treatment`` arm of the existing
``HEADROOM_OUTPUT_HOLDOUT`` conversation-stable assignment.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from headroom.log_safety import describe_exception

log = logging.getLogger(__name__)

FEATURE_COMPLEXITY = "agent_complexity_routing"
EFFORT_ORDER = ("low", "medium", "high", "xhigh", "max")
EFFORT_FLOOR = "medium"
TURN_BLOCK_TAG = "headroom_turn_verbosity"
MAX_SESSIONS = 1024

_EXPLICIT_DEEP = re.compile(
    r"\b(?:ultrathink|megathink|think\s+(?:very\s+)?(?:hard(?:er)?|deeply|carefully|longer)|"
    r"think\s+step\s+by\s+step|take\s+your\s+time|be\s+thorough)\b",
    re.IGNORECASE,
)
_EXPLAIN = re.compile(
    r"\b(?:why|explain|walk me through|how does|how do|compare|trade-?offs?)\b", re.IGNORECASE
)
_HEDGE = re.compile(
    r"\b(?:not sure|unsure|unclear|might be|may be|possibly|perhaps|i think|i suspect|"
    r"it seems|appears to|hard to tell|could be)\b",
    re.IGNORECASE,
)
_DONE_CLAIM = re.compile(
    r"\b(?:done|complete[d]?|finished|fixed|resolved|all tests pass(?:ing)?|ready)\b",
    re.IGNORECASE,
)
_ERROR_LINE = re.compile(
    r"(?m)^.*(?:Traceback|Error\b|error(?:\[|:)|Exception\b|FAILED\b|panicked at|fatal:).*$"
)
_PATH = re.compile(r"(?<![\w@])(?:[\w.\-]+/)+[\w.\-]+\.[A-Za-z0-9]{1,8}\b")
_SYMBOL = re.compile(r"\b(?:[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+|[a-z]+(?:_[a-z0-9]+){1,})\b")


@dataclass(frozen=True)
class ComplexitySignals:
    turn_kind: str
    errors: bool = False
    novel_errors: bool = False
    failed_retries: bool = False
    contradictions: bool = False
    unresolved_gate: bool = False
    explicit_deep: bool = False
    unknown_requirements: bool = False
    new_symbols: bool = False
    multi_file: bool = False
    uncertainty: bool = False
    explains: bool = False

    @property
    def protected(self) -> str | None:
        """Why effort must not be lowered (``None`` = it may be)."""
        if self.turn_kind != "mechanical_continuation":
            return self.turn_kind
        for name in (
            "errors",
            "novel_errors",
            "failed_retries",
            "contradictions",
            "unresolved_gate",
            "explicit_deep",
        ):
            if getattr(self, name):
                return name
        return None

    @property
    def score(self) -> int:
        """Deterministic 0-4 complexity of a non-protected turn."""
        return min(
            4,
            sum(
                1
                for flag in (
                    self.unknown_requirements,
                    self.new_symbols,
                    self.multi_file,
                    self.uncertainty,
                )
                if flag
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass(frozen=True)
class RoutingDecision:
    arm: str
    signals: ComplexitySignals | None
    score: int | None = None
    score_source: str = "deterministic"
    effort_before: str | None = None
    effort_after: str | None = None
    verbosity_base: int = 0
    verbosity_turn: int | None = None
    protected: str | None = None
    labels: tuple[str, ...] = ()
    changed: bool = False


@dataclass
class _SessionMemory:
    errors: OrderedDict[str, None] = field(default_factory=OrderedDict)
    symbols: OrderedDict[str, None] = field(default_factory=OrderedDict)


_sessions: OrderedDict[str, _SessionMemory] = OrderedDict()
_lock = threading.Lock()
_counters: dict[str, int] = {
    "routed": 0,
    "effort_lowered": 0,
    "effort_kept_protected": 0,
    "effort_absent": 0,
    "verbosity_relaxed": 0,
    "jevk5_scores": 0,
    "control_arm": 0,
}


def _bump(name: str) -> None:
    with _lock:
        _counters[name] = _counters.get(name, 0) + 1


def status() -> dict[str, Any]:
    with _lock:
        return {"sessions": len(_sessions), **_counters}


def _memory(session_key: str) -> _SessionMemory:
    with _lock:
        mem = _sessions.get(session_key)
        if mem is None:
            mem = _SessionMemory()
            _sessions[session_key] = mem
        _sessions.move_to_end(session_key)
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)
        return mem


def _remember(store: OrderedDict[str, None], items: Sequence[str], cap: int = 2048) -> int:
    """Add ``items``; return how many were new."""
    new = 0
    with _lock:
        for item in items:
            if item not in store:
                store[item] = None
                new += 1
        while len(store) > cap:
            store.popitem(last=False)
    return new


# ---------------------------------------------------------------------------
# Structure readers (Anthropic / chat / Responses)
# ---------------------------------------------------------------------------


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        out = []
        for b in value:
            if isinstance(b, Mapping):
                if isinstance(b.get("text"), str):
                    out.append(b["text"])
                elif b.get("type") == "tool_result":
                    out.append(_text(b.get("content")))
        return "\n".join(out)
    return ""


@dataclass
class _Turn:
    kind: str
    latest_user: str = ""
    last_assistant: str = ""
    trailing_results: list[tuple[str, str, bool]] = field(default_factory=list)  # (tool, text, err)
    recent_calls: list[tuple[str, Mapping[str, Any]]] = field(default_factory=list)


def _read_messages(messages: Sequence[Any]) -> _Turn:
    from headroom.proxy.output_turn_policy import classify_turn

    msgs = [m for m in messages if isinstance(m, Mapping)]
    kind = classify_turn(list(msgs)).value
    names: dict[str, str] = {}
    calls: list[tuple[str, Mapping[str, Any]]] = []
    latest_user = ""
    last_assistant = ""
    for m in msgs:
        content = m.get("content")
        if m.get("role") == "assistant":
            text = _text(content)
            if text.strip():
                last_assistant = text
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, Mapping) and b.get("type") == "tool_use":
                        names[str(b.get("id"))] = str(b.get("name") or "")
                        inp = b.get("input") if isinstance(b.get("input"), Mapping) else {}
                        calls.append((str(b.get("name") or ""), inp))
            for call in m.get("tool_calls") or []:
                if isinstance(call, Mapping):
                    fn = call.get("function") if isinstance(call.get("function"), Mapping) else {}
                    names[str(call.get("id"))] = str(fn.get("name") or "")
                    calls.append((str(fn.get("name") or ""), {"arguments": fn.get("arguments")}))
        elif m.get("role") == "user":
            if isinstance(content, str):
                latest_user = content
            elif isinstance(content, list):
                t = "\n".join(
                    str(b.get("text") or "")
                    for b in content
                    if isinstance(b, Mapping) and b.get("type") == "text"
                )
                if t.strip():
                    latest_user = t
    trailing: list[tuple[str, str, bool]] = []
    if msgs:
        last = msgs[-1]
        if last.get("role") == "user" and isinstance(last.get("content"), list):
            for b in last["content"]:
                if isinstance(b, Mapping) and b.get("type") == "tool_result":
                    trailing.append(
                        (
                            names.get(str(b.get("tool_use_id")), ""),
                            _text(b.get("content")),
                            bool(b.get("is_error")),
                        )
                    )
        # OpenAI chat: trailing role=tool messages.
        idx = len(msgs) - 1
        while idx >= 0 and msgs[idx].get("role") == "tool":
            m = msgs[idx]
            trailing.append(
                (names.get(str(m.get("tool_call_id")), ""), _text(m.get("content")), False)
            )
            idx -= 1
        if idx < len(msgs) - 1 and kind == "unknown":
            kind = "mechanical_continuation"
    return _Turn(kind, latest_user, last_assistant, trailing, calls[-8:])


def _read_responses(items: Any) -> _Turn:
    from headroom.proxy.output_shaper import classify_responses_turn

    kind = classify_responses_turn(items).value
    if not isinstance(items, list):
        return _Turn(kind, latest_user=items if isinstance(items, str) else "")
    names: dict[str, str] = {}
    calls: list[tuple[str, Mapping[str, Any]]] = []
    latest_user = last_assistant = ""
    trailing: list[tuple[str, str, bool]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        itype = item.get("type")
        if itype == "function_call":
            names[str(item.get("call_id"))] = str(item.get("name") or "")
            calls.append((str(item.get("name") or ""), {"arguments": item.get("arguments")}))
        elif item.get("role") == "user":
            t = _text(item.get("content"))
            if t.strip():
                latest_user = t
        elif item.get("role") == "assistant":
            t = _text(item.get("content"))
            if t.strip():
                last_assistant = t
    for item in reversed(items):
        if not isinstance(item, Mapping) or not str(item.get("type") or "").endswith("_output"):
            break
        out = item.get("output")
        trailing.append(
            (
                names.get(str(item.get("call_id")), ""),
                out if isinstance(out, str) else str(out),
                False,
            )
        )
    return _Turn(kind, latest_user, last_assistant, trailing, calls[-8:])


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------


def assess(turn: _Turn, *, session_key: str, snapshot: Any = None) -> ComplexitySignals:
    mem = _memory(session_key)
    trailing_text = "\n".join(t for _, t, _ in turn.trailing_results)
    error_lines = [ln.strip()[:160] for ln in _ERROR_LINE.findall(trailing_text)]
    errors = any(err for _, _, err in turn.trailing_results) or bool(error_lines)
    sigs = [hashlib.sha256(re.sub(r"\d+", "#", ln).encode()).hexdigest()[:12] for ln in error_lines]
    novel = _remember(mem.errors, sigs) > 0
    failed_tools = [tool for tool, _, err in turn.trailing_results if err]
    failed_retries = len(failed_tools) >= 2 or (
        bool(failed_tools) and sum(1 for name, _ in turn.recent_calls if name in failed_tools) >= 2
    )
    contradictions = False
    unresolved_gate = False
    unknown = snapshot is None
    if snapshot is not None:
        contradictions = any(
            getattr(f, "status", "") == "CONTRADICTED" for f in getattr(snapshot, "evidence", ())
        )
        state = snapshot.state
        unknown = state.goal is None
        live_acceptance = [a for a in state.acceptance if a.live]
        open_acceptance = [a for a in live_acceptance if a.state != "SATISFIED"]
        claimed = bool(_DONE_CLAIM.search(turn.last_assistant or ""))
        test_ran = any(
            "pytest" in str(inp) or "test" in name.lower() for name, inp in turn.recent_calls[-2:]
        )
        unresolved_gate = bool(open_acceptance) and (claimed or test_ran)
    symbols = list(dict.fromkeys(_SYMBOL.findall(trailing_text)))[:400]
    new_symbols = _remember(mem.symbols, symbols) > 8
    paths: set[str] = set()
    for _name, inp in turn.recent_calls[-6:]:
        for key in ("file_path", "path", "notebook_path"):
            value = inp.get(key) if isinstance(inp, Mapping) else None
            if isinstance(value, str) and value:
                paths.add(value)
    paths.update(_PATH.findall(trailing_text)[:50])
    multi_file = len(paths) >= 3
    uncertainty = len(_HEDGE.findall(turn.last_assistant or "")) >= 2
    return ComplexitySignals(
        turn_kind=turn.kind,
        errors=errors,
        novel_errors=novel and errors,
        failed_retries=failed_retries,
        contradictions=contradictions,
        unresolved_gate=unresolved_gate,
        explicit_deep=bool(_EXPLICIT_DEEP.search(turn.latest_user or "")),
        unknown_requirements=unknown,
        new_symbols=new_symbols,
        multi_file=multi_file,
        uncertainty=uncertainty,
        explains=bool(_EXPLAIN.search(turn.latest_user or "")),
    )


def lowered_effort(current: str, score: int) -> str:
    """The routed effort for a non-protected turn; never raises the client's."""
    if current not in EFFORT_ORDER:
        return current
    idx = EFFORT_ORDER.index(current)
    drop = {0: 2, 1: 1}.get(score, 0)
    floor = EFFORT_ORDER.index(EFFORT_FLOOR)
    target = max(idx - drop, floor) if idx > floor else idx
    return EFFORT_ORDER[min(target, idx)]


def _jev_score(
    advisor: Any, signals: ComplexitySignals, request_id: str | None = None
) -> int | None:
    if advisor is None:
        return None
    try:
        # The request's shared allowance (arbiter + catalog + this call <= 4).
        budget = advisor.request_budget(request_id)
        advice = advisor.score(
            "effort_complexity",
            {"signals": signals.to_dict()},
            "How much reasoning does the agent's next step need (0 trivial .. 4 very hard)?",
            ["0", "1", "2", "3", "4"],
            budget=budget,
        )
    except Exception as exc:  # noqa: BLE001 - advisory only
        log.debug("complexity advisor failed: %s", describe_exception(exc))
        return None
    if advice is None:
        return None
    try:
        return max(0, min(4, int(round(float(advice.value)))))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Effort locations and the per-turn verbosity block
# ---------------------------------------------------------------------------


def _get_effort(body: Mapping[str, Any], shape: str) -> str | None:
    if shape == "anthropic":
        oc = body.get("output_config")
        value = oc.get("effort") if isinstance(oc, Mapping) else None
    elif shape == "responses":
        rs = body.get("reasoning")
        value = rs.get("effort") if isinstance(rs, Mapping) else None
    else:
        value = body.get("reasoning_effort")
    return value if isinstance(value, str) and value in EFFORT_ORDER else None


def _set_effort(body: dict[str, Any], shape: str, value: str) -> None:
    if shape == "anthropic":
        body["output_config"] = {**body["output_config"], "effort": value}
    elif shape == "responses":
        body["reasoning"] = {**body["reasoning"], "effort": value}
    else:
        body["reasoning_effort"] = value


def turn_verbosity_block(level: int) -> str:
    from headroom.proxy.output_verbosity_policy import steering_text

    guidance = steering_text(level) if level > 0 else None
    body = guidance or (
        "Answer with whatever detail this question needs; earlier brevity guidance "
        "does not limit this reply."
    )
    return f'<{TURN_BLOCK_TAG} level="{level}">For this reply only: {body}</{TURN_BLOCK_TAG}>'


def _place_turn_block(body: dict[str, Any], shape: str, model: str, block: str) -> bool:
    from headroom.agent.task_state import plan_injection

    if shape == "anthropic":
        new, _ = plan_injection("anthropic", model, list(body.get("messages") or []), block)
        if new is None:
            return False
        body["messages"] = new
        return True
    if shape == "responses":
        items = body.get("input")
        if not isinstance(items, list):
            return False
        new, _ = plan_injection("openai", model, list(items), block)
        if new is None:
            return False
        body["input"] = new
        return True
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    body["messages"] = [*messages, {"role": "system", "content": block}]
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _enabled(config: Any, name: str) -> bool:
    rollout = getattr(config, "rollout", None)
    try:
        return bool(rollout is not None and rollout.is_enabled(name))
    except KeyError:
        return False


def _snapshot_and_advisor(session_hint: str | None) -> tuple[Any, Any, str]:
    from headroom.agent.hook import _active_runtime
    from headroom.agent.identity import resolve_identity
    from headroom.proxy.project_context import (
        get_current_project,
        get_registered_cwd,
        get_registered_session,
    )

    identity = resolve_identity(
        session_token=get_registered_session(),
        registered_cwd=get_registered_cwd(),
        attribution=get_current_project(),
    )
    runtime = _active_runtime()
    snapshot = advisor = None
    if runtime is not None:
        advisor = getattr(runtime, "advisor", None)
        compiler = runtime.services.get("task_state")
        if compiler is not None and identity.trusted:
            snapshot = compiler.snapshot(identity.workspace_id, identity.session_id)
    return snapshot, advisor, f"{identity.workspace_id}|{session_hint or identity.session_id}"


def route_request(
    body: dict[str, Any],
    *,
    shape: str,
    config: Any,
    model: str,
    request_id: str | None = None,
    session_key: str | None = None,
) -> RoutingDecision | None:
    """Route effort and per-turn verbosity in place; ``None`` when off. Never raises."""
    if not _enabled(config, FEATURE_COMPLEXITY):
        return None
    try:
        return _route(
            body,
            shape=shape,
            config=config,
            model=model,
            session_key=session_key,
            request_id=request_id,
        )
    except Exception as exc:  # noqa: BLE001 - routing never fails a request
        log.warning("[%s] complexity routing failed: %s", request_id, describe_exception(exc))
        return None


def _route(
    body: dict[str, Any],
    *,
    shape: str,
    config: Any,
    model: str,
    session_key: str | None,
    request_id: str | None = None,
) -> RoutingDecision:
    from headroom.proxy.output_savings import (
        assign_arm,
        conversation_key_from_body,
        conversation_key_from_responses_body,
    )
    from headroom.proxy.output_shaper import (
        OutputShaperSettings,
        output_holdout_fraction,
        resolve_verbosity_level,
        shaper_enabled_for,
        steering_allowed_for,
    )

    conversation = (
        conversation_key_from_responses_body(body)
        if shape == "responses"
        else conversation_key_from_body(body)
    )
    arm = assign_arm(conversation, output_holdout_fraction())
    if arm != "treatment":
        _bump("control_arm")
        return RoutingDecision(arm=arm, signals=None)
    snapshot, advisor, skey = _snapshot_and_advisor(session_key)
    turn = (
        _read_responses(body.get("input"))
        if shape == "responses"
        else _read_messages(body.get("messages") or [])
    )
    signals = assess(turn, session_key=skey, snapshot=snapshot)
    _bump("routed")
    labels: list[str] = []
    changed = False

    # -- effort -------------------------------------------------------------
    effort = _get_effort(body, shape)
    effort_after = effort
    protected = signals.protected
    score: int | None = None
    source = "deterministic"
    if effort is None:
        _bump("effort_absent")
    elif protected is not None:
        _bump("effort_kept_protected")
    else:
        score = signals.score
        if score == 2:
            jev = _jev_score(advisor, signals, request_id)
            if jev is not None:
                score, source = jev, "jevk5"
                _bump("jevk5_scores")
        effort_after = lowered_effort(effort, score)
        if effort_after != effort:
            _set_effort(body, shape, effort_after)
            changed = True
            _bump("effort_lowered")
            labels.append(f"agent:effort:{effort}->{effort_after}:s{score}")

    # -- verbosity (independent) -----------------------------------------------
    base = 0
    shaper = shaper_enabled_for(config)
    if shaper:
        settings = OutputShaperSettings.from_env(
            enabled=True, steering_enabled=steering_allowed_for(config)
        )
        base, _src = resolve_verbosity_level(settings)
    turn_level: int | None = None
    if (
        base > 0
        and signals.turn_kind == "new_user_ask"
        and (signals.explains or signals.contradictions or signals.explicit_deep)
    ):
        desired = min(base, 1)
        if desired != base and _place_turn_block(body, shape, model, turn_verbosity_block(desired)):
            turn_level = desired
            changed = True
            _bump("verbosity_relaxed")
            labels.append(f"agent:verbosity_turn:L{base}->L{desired}")
    return RoutingDecision(
        arm=arm,
        signals=signals,
        score=score,
        score_source=source,
        effort_before=effort,
        effort_after=effort_after,
        verbosity_base=base,
        verbosity_turn=turn_level,
        protected=protected,
        labels=tuple(labels),
        changed=changed,
    )


def reset() -> None:
    with _lock:
        _sessions.clear()
        for k in _counters:
            _counters[k] = 0


__all__ = [
    "ComplexitySignals",
    "EFFORT_ORDER",
    "FEATURE_COMPLEXITY",
    "RoutingDecision",
    "assess",
    "lowered_effort",
    "route_request",
    "status",
    "turn_verbosity_block",
]
