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
  0-4 score. JevK5 is consulted only when :func:`should_consult_advisor` finds
  that score genuinely uncertain, and then only about the single boundary the
  routed effort turns on, so its answer is always one of the two scores the
  deterministic logic was already choosing between. A short answer with high
  effort is a valid outcome.
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

from headroom.intelligence.policy import (
    FAMILY_COMPLEXITY,
    Uncertainty,
    ambiguity_threshold,
    ask_choice,
    consumer_stats,
    should_consult,
)
from headroom.intelligence.policy import applied as policy_applied
from headroom.intelligence.policy import clip as _clip
from headroom.log_safety import describe_exception

log = logging.getLogger(__name__)

#: This consumer's JevK5 counters (``/health.agent_runtime.intelligence.consumers``).
_STATS = consumer_stats("complexity_router")

FEATURE_COMPLEXITY = "agent_complexity_routing"
EFFORT_ORDER = ("low", "medium", "high", "xhigh", "max")
EFFORT_FLOOR = "medium"
TURN_BLOCK_TAG = "headroom_turn_verbosity"
MAX_SESSIONS = 1024
#: Thresholds the three positive-evidence booleans fire at. The uncertainty
#: policy reads the raw counts against them, so a turn that sits on a
#: threshold is treated as the borderline case it is.
NEW_SYMBOL_THRESHOLD = 8
PATH_THRESHOLD = 3
HEDGE_THRESHOLD = 2
#: Score boundaries of :func:`lowered_effort` (drop 2 / 1 / 0).
SCORE_BOUNDARIES = (1.0, 2.0)
#: How close the continuous score must come to a boundary to count as borderline.
BOUNDARY_BAND = 0.40
#: Positive evidence (0..3) a turn needs before its effort is worth a question.
#: Below it the turn is a clean mechanical continuation: deterministic, no call.
MIN_EVIDENCE = 0.30

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
    # Magnitudes behind the three positive-evidence booleans above, plus what
    # the projector knew. The booleans are the deterministic score; these give
    # the uncertainty policy a continuous view of how close each threshold is.
    new_symbol_count: int = 0
    path_count: int = 0
    hedge_count: int = 0
    snapshot_available: bool = False
    open_work: bool = False

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
        counters = {"sessions": len(_sessions), **_counters}
    counters["jevk5"] = _STATS.snapshot()
    return counters


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
    open_work = False
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
        # Known, still-unsatisfied acceptance work: the task carries more
        # complexity than this turn's text shows (an uncertainty signal, not a
        # protected turn — only ``unresolved_gate`` protects).
        open_work = bool(open_acceptance)
    symbols = list(dict.fromkeys(_SYMBOL.findall(trailing_text)))[:400]
    new_symbol_count = _remember(mem.symbols, symbols)
    new_symbols = new_symbol_count > NEW_SYMBOL_THRESHOLD
    paths: set[str] = set()
    for _name, inp in turn.recent_calls[-6:]:
        for key in ("file_path", "path", "notebook_path"):
            value = inp.get(key) if isinstance(inp, Mapping) else None
            if isinstance(value, str) and value:
                paths.add(value)
    paths.update(_PATH.findall(trailing_text)[:50])
    multi_file = len(paths) >= PATH_THRESHOLD
    hedge_count = len(_HEDGE.findall(turn.last_assistant or ""))
    uncertainty = hedge_count >= HEDGE_THRESHOLD
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
        new_symbol_count=new_symbol_count,
        path_count=len(paths),
        hedge_count=hedge_count,
        snapshot_available=snapshot is not None,
        open_work=open_work,
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


def positive_evidence(signals: ComplexitySignals) -> float:
    """Continuous strength of the three *positive* complexity signals (0..3)."""
    return (
        _clip(signals.new_symbol_count / NEW_SYMBOL_THRESHOLD)
        + _clip(signals.path_count / PATH_THRESHOLD)
        + _clip(signals.hedge_count / HEDGE_THRESHOLD)
    )


def soft_score(signals: ComplexitySignals) -> float:
    """The deterministic score without its thresholds rounded away (0..4).

    ``score`` counts four booleans; this counts how far each of them is from
    firing. Where the two land on different sides of a :data:`SCORE_BOUNDARIES`
    boundary, the deterministic answer rests on a threshold, not on evidence.
    """
    unknown = 1.0 if signals.unknown_requirements else 0.0
    return min(4.0, unknown + positive_evidence(signals))


def effective_boundaries(effort: str | None) -> tuple[float, ...]:
    """Score boundaries that actually change the routed effort for ``effort``.

    ``lowered_effort`` clamps at :data:`EFFORT_FLOOR`, so for a client effort
    close to the floor some boundaries route to the same value on both sides.
    Asking JevK5 about a boundary whose answer cannot move the effort is pure
    latency, so those are dropped — and an effort already at (or below) the
    floor leaves none, which skips the family entirely for that request.
    """
    if effort is None or effort not in EFFORT_ORDER:
        return ()
    return tuple(
        b
        for b in SCORE_BOUNDARIES
        if lowered_effort(effort, int(b) - 1) != lowered_effort(effort, int(b))
    )


def should_consult_advisor(
    signals: ComplexitySignals,
    deterministic_score: int,
    *,
    effort: str | None = None,
) -> Uncertainty:
    """Whether the effort decision is genuinely uncertain (unified-21 #10).

    Uncertain means one of:

    * **borderline threshold** — the continuous score sits within
      :data:`BOUNDARY_BAND` of a boundary where :func:`lowered_effort` changes
      its answer;
    * **signals disagree** — the thresholded score and the continuous one fall
      on opposite sides of such a boundary;
    * **task state is heavier than the turn** — the projector still has
      unsatisfied acceptance work while this turn looks simple.

    It is *never* uncertain when the advisor's answer cannot change the routed
    effort (the client's effort is already at the floor), and never when there
    is no material positive evidence at all (a clean mechanical continuation:
    the obviously safe case this must stay out of).
    """
    soft = soft_score(signals)
    evidence = positive_evidence(signals)
    detail = {"score": deterministic_score, "soft": round(soft, 2)}
    boundaries = effective_boundaries(effort)
    if not boundaries:
        # Nothing JevK5 could answer would move the effort: don't ask.
        return Uncertainty(blocked="no_effort_change_possible", detail=detail)
    if evidence < MIN_EVIDENCE:
        return Uncertainty(blocked="no_complexity_evidence", detail=detail)
    reasons: list[str] = []
    score = 0.0
    nearest = min(boundaries, key=lambda b: abs(soft - b))
    distance = abs(soft - nearest)
    detail["boundary"] = nearest
    threshold = ambiguity_threshold()
    if distance <= BOUNDARY_BAND:
        # Inside the band the routed effort rests on a threshold, not on
        # evidence: decisive on its own. BOUNDARY_BAND is the tuning knob.
        reasons.append("borderline_threshold")
        score += threshold + (1.0 - threshold) * 0.4 * (1.0 - distance / BOUNDARY_BAND)
    if any((soft - b) * (deterministic_score - b) < 0 for b in boundaries):
        reasons.append("signals_disagree")
        score += threshold
    if signals.open_work and soft < SCORE_BOUNDARIES[-1]:
        reasons.append("task_state_heavier")
        score += 0.5 * threshold
    if signals.explains and deterministic_score < 2:
        reasons.append("explanatory_ask")
        score += 0.3 * threshold
    return Uncertainty(score=_clip(score), reasons=tuple(reasons), detail=detail)


#: What each side of a boundary means, in the terms the turn is actually in.
#: The question JevK5 gets is the *one* binary choice the routed effort turns
#: on, not a five-level scale over a bag of flags: measured on the published 4B
#: model, a concrete two-way question comes back at 0.86-0.999 confidence while
#: a flags-and-levels one sits at ~0.5 and is rejected by the 0.60 floor, so
#: the scale shape alone decided whether this family was ever usable.
_BOUNDARY_OPTIONS: Mapping[float, tuple[str, str]] = {
    1.0: (
        "a routine mechanical step: carry on and finish it directly",
        "one real step of reasoning is needed before acting",
    ),
    2.0: (
        "one step of reasoning is enough",
        "several interacting unknowns: this needs careful reasoning",
    ),
}


def turn_description(signals: ComplexitySignals) -> str:
    """A content-free sentence describing the turn's shape for the advisor.

    Counts and flags only — never a path, an identifier, a command or any user
    or tool text — so the advisor gets something it can reason about without
    any payload leaving the process.
    """
    parts = [f"The agent is in a {signals.turn_kind.replace('_', ' ')}."]
    parts.append(
        f"{signals.path_count} file path(s) in play"
        if signals.path_count
        else "no file paths in play"
    )
    parts.append(
        f"{signals.new_symbol_count} identifier(s) not seen before in this session"
        if signals.new_symbol_count
        else "no new identifiers"
    )
    parts.append(
        f"{signals.hedge_count} hedging phrase(s) in the agent's own last message"
        if signals.hedge_count
        else "no hedging in the agent's own last message"
    )
    parts.append(
        "the task goal is not known yet" if signals.unknown_requirements else "the task goal is known"
    )
    if signals.open_work:
        parts.append("acceptance criteria are still unsatisfied")
    if signals.explains:
        parts.append("the user asked for an explanation or a comparison")
    return parts[0] + " " + ", ".join(parts[1:]) + "."


def _jev_score(
    advisor: Any,
    signals: ComplexitySignals,
    deterministic_score: int,
    *,
    effort: str | None = None,
    request_id: str | None = None,
) -> int | None:
    """JevK5's complexity score for an uncertain turn, or ``None``.

    It is asked only about the single boundary the routed effort turns on, and
    may only answer with one of that boundary's two sides — so the result is
    always one of the two scores the deterministic logic was already choosing
    between, and can never route above what the client sent.
    """
    uncertainty = should_consult_advisor(signals, deterministic_score, effort=effort)
    # The request's shared allowance (arbiter + catalog + this call <= 4).
    budget = None
    if advisor is not None:
        try:
            budget = advisor.request_budget(request_id)
        except Exception as exc:  # noqa: BLE001 - advisory only
            log.debug("complexity advisor budget failed: %s", describe_exception(exc))
            return None
    if not should_consult(
        FAMILY_COMPLEXITY, uncertainty, advisor=advisor, budget=budget, stats=_STATS
    ):
        return None
    boundary = float(uncertainty.detail.get("boundary") or SCORE_BOUNDARIES[0])
    below, at_or_above = _BOUNDARY_OPTIONS.get(boundary, _BOUNDARY_OPTIONS[1.0])
    advice = ask_choice(
        advisor,
        FAMILY_COMPLEXITY,
        {"turn": turn_description(signals), "signals": signals.to_dict()},
        "How much reasoning does the agent's next step need?",
        {"below": below, "at_or_above": at_or_above},
        budget=budget,
        stats=_STATS,
    )
    if advice is None:
        return None
    choice = str(getattr(advice, "value", ""))
    if choice not in ("below", "at_or_above"):
        _STATS.skipped("malformed")
        _STATS.bump("deterministic_fallback")
        return None
    value = int(boundary) - 1 if choice == "below" else int(boundary)
    policy_applied(FAMILY_COMPLEXITY, advice, _STATS)
    return max(0, min(4, value))


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
        jev = _jev_score(advisor, signals, score, effort=effort, request_id=request_id)
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
    "effective_boundaries",
    "lowered_effort",
    "positive_evidence",
    "route_request",
    "should_consult_advisor",
    "soft_score",
    "status",
    "turn_description",
    "turn_verbosity_block",
]
