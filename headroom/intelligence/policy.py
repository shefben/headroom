"""One JevK5 consultation policy for every advisory consumer (unified-21).

The advisor is a *bounded* classifier, so the question is never "could JevK5
have an opinion" but "is this decision genuinely uncertain and can the answer
still change the outcome". Every consumer therefore runs the same ladder:

1. deterministic hard constraints (SafetyGate, protected turns, core tools) —
   JevK5 never sees a candidate they rejected;
2. deterministic easy decisions — a confident deterministic answer is used
   as-is and nothing is sent;
3. **uncertainty detection** (:class:`Uncertainty`) from cheap numbers the
   consumer already has: closeness of the competing options, disagreement
   between the components that produced them, how many options sit on the
   decision frontier, rank stability under a small perturbation, and whether
   the answer could change the outcome at all;
4. consult JevK5 only for an uncertain decision, and only while the advisor
   is ready and the request's shared call allowance has room;
5. the advisor's own confidence gate;
6. bounded blending by the consumer (the capped advisor weight);
7. the deterministic answer otherwise.

:class:`ConsumerStats` records that ladder per consumer so ``/health`` can
prove which subsystem actually used the advisor. Counters only: no request id,
path, tool argument or user string ever reaches a counter or a log line here.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

#: The decision families a live proxy may consult. Fixed set: ``by_family``
#: and the consumer table must stay low-cardinality.
FAMILY_COMPLEXITY = "effort_complexity"
FAMILY_ARBITER = "transform_arbiter"
FAMILY_TOOL_CATALOG = "tool_catalog"
FAMILY_TASK_STATE = "task_state_target"
#: Claude's trusted pre-tool boundary only (repair-suggestion order).
FAMILY_REPAIR_RANK = "tool_repair_rank"
FAMILIES = (
    FAMILY_COMPLEXITY,
    FAMILY_ARBITER,
    FAMILY_TOOL_CATALOG,
    FAMILY_TASK_STATE,
    FAMILY_REPAIR_RANK,
)

#: An :class:`Uncertainty` at or above this is worth one advisor call.
DEFAULT_AMBIGUITY_THRESHOLD = 0.50
_AMBIGUITY_ENV = "HEADROOM_JEVK5_AMBIGUITY"


def ambiguity_threshold() -> float:
    """Uncertainty needed before a consumer may consult JevK5 (0..1)."""
    raw = (os.environ.get(_AMBIGUITY_ENV) or "").strip()
    if not raw:
        return DEFAULT_AMBIGUITY_THRESHOLD
    try:
        return min(1.0, max(0.0, float(raw)))
    except ValueError:
        return DEFAULT_AMBIGUITY_THRESHOLD


def clip(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Uncertainty:
    """How unsure the *deterministic* answer is, with the reasons that said so.

    ``score`` 0 = the deterministic answer is obviously safe, 1 = maximally
    uncertain. ``blocked`` carries a hard "do not ask" reason that no amount of
    uncertainty overrides (the answer cannot change the outcome, there is only
    one option left, ...).
    """

    score: float = 0.0
    reasons: tuple[str, ...] = ()
    blocked: str | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ambiguous(self) -> bool:
        return self.blocked is None and self.score >= ambiguity_threshold()

    @property
    def reason(self) -> str:
        if self.blocked is not None:
            return self.blocked
        if self.ambiguous:
            return ",".join(self.reasons) or "ambiguous"
        return "deterministic_margin"


def closeness(margin: float, band: float) -> float:
    """1.0 when two options are tied, 0.0 once they are ``band`` apart."""
    if band <= 0:
        return 0.0
    return clip(1.0 - (max(0.0, float(margin)) / band))


def frontier_density(scores: Sequence[float], band: float) -> float:
    """Share of options sitting within ``band`` of the best one, beyond the first.

    0.0 = one option owns the decision; 1.0 = every option is on the frontier.
    """
    values = [float(s) for s in scores]
    if len(values) < 2 or band <= 0:
        return 0.0
    top = max(values)
    near = sum(1 for s in values if top - s <= band)
    return clip((near - 1) / (len(values) - 1))


def component_disagreement(
    components: Sequence[Mapping[str, float]], keys: Sequence[str] | None = None
) -> float:
    """How much the scoring components disagree about which option wins.

    0.0 = every component prefers the same option; 1.0 = each component
    prefers a different one. Ties inside a component are resolved by index, so
    the result depends only on the numbers (no scheduling, no ordering luck).
    """
    if len(components) < 2:
        return 0.0
    names = list(keys) if keys else sorted({k for c in components for k in c})
    if len(names) < 2:
        return 0.0
    winners: set[int] = set()
    for name in names:
        best, best_i = None, 0
        for i, comp in enumerate(components):
            value = float(comp.get(name, 0.0) or 0.0)
            if best is None or value > best:
                best, best_i = value, i
        winners.add(best_i)
    return clip((len(winners) - 1) / (len(names) - 1))


def rank_unstable(
    components: Sequence[Mapping[str, float]],
    weights: Mapping[str, float],
    *,
    epsilon: float = 0.20,
) -> bool:
    """Whether the winner changes when one component weight is nudged.

    Deterministic sensitivity analysis: every weight is scaled by
    ``1 +- epsilon`` in turn (others renormalized implicitly by the shared
    denominator cancelling out of the comparison). A winner that survives all
    of them is a stable deterministic answer.
    """
    if len(components) < 2:
        return False

    def winner(scale: Mapping[str, float]) -> int:
        best, best_i = None, 0
        for i, comp in enumerate(components):
            total = sum(
                float(weights.get(k, 0.0)) * float(scale.get(k, 1.0)) * float(comp.get(k, 0.0) or 0.0)
                for k in weights
            )
            if best is None or total > best:
                best, best_i = total, i
        return best_i

    base = winner({})
    for key in weights:
        for factor in (1.0 - epsilon, 1.0 + epsilon):
            if winner({key: factor}) != base:
                return True
    return False


def spread(values: Sequence[float]) -> float:
    values = [float(v) for v in values]
    if len(values) < 2:
        return 0.0
    return clip(max(values) - min(values))


# ---------------------------------------------------------------------------
# Per-consumer telemetry
# ---------------------------------------------------------------------------

#: Reasons a request for advice came back empty. Low cardinality by design.
_OUTCOMES = (
    "ok",
    "low_confidence",
    "budget_exhausted",
    "timeout",
    "failure",
    "breaker",
    "unavailable",
    "non_english",
    "context_overflow",
    "malformed",
    "not_ready",
)


class ConsumerStats:
    """Thread-safe JevK5 counters for one subsystem (counts only)."""

    __slots__ = ("name", "_lock", "_counts", "_skips")

    def __init__(self, name: str) -> None:
        self.name = name
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {
            "eligible": 0,
            "calls_requested": 0,
            "advice_received": 0,
            "advice_applied": 0,
            "advice_ignored_low_confidence": 0,
            "deterministic_fallback": 0,
            "budget_exhausted": 0,
            "not_ready": 0,
        }
        self._skips: dict[str, int] = {}

    def bump(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._counts[name] = self._counts.get(name, 0) + n

    def skipped(self, reason: str) -> None:
        with self._lock:
            self._skips[reason] = self._skips.get(reason, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {**self._counts, "skipped": dict(self._skips)}

    def reset(self) -> None:
        with self._lock:
            for key in self._counts:
                self._counts[key] = 0
            self._skips.clear()


_registry: dict[str, ConsumerStats] = {}
_registry_lock = threading.Lock()


def consumer_stats(name: str) -> ConsumerStats:
    """The process-wide counters for one JevK5 consumer."""
    with _registry_lock:
        stats = _registry.get(name)
        if stats is None:
            stats = ConsumerStats(name)
            _registry[name] = stats
        return stats


def consumer_snapshot() -> dict[str, Any]:
    """Every registered consumer's counters (for ``/health``)."""
    with _registry_lock:
        consumers = list(_registry.values())
    return {stats.name: stats.snapshot() for stats in sorted(consumers, key=lambda s: s.name)}


def reset_consumers() -> None:
    with _registry_lock:
        consumers = list(_registry.values())
    for stats in consumers:
        stats.reset()


# ---------------------------------------------------------------------------
# Readiness and the gate
# ---------------------------------------------------------------------------


def advisor_readiness(advisor: Any) -> str:
    """Cheap, network-free readiness of an advisor: see ``DecisionAdvisor.readiness``.

    ``"absent"`` when there is no advisor at all. Anything other than
    ``"ready"`` means a call would almost certainly say nothing, so consumers
    skip it — but the advisor stays fail-open, because readiness can change
    between this check and the call.
    """
    if advisor is None:
        return "absent"
    readiness = getattr(advisor, "readiness", None)
    if isinstance(readiness, str):
        return readiness
    return "ready"  # a test double or an older advisor: let it answer


def should_consult(
    family: str,
    uncertainty: Uncertainty,
    *,
    advisor: Any,
    budget: Any,
    stats: ConsumerStats,
) -> bool:
    """The shared gate. Records why it said no and logs it at DEBUG."""
    readiness = advisor_readiness(advisor)
    if readiness != "ready":
        stats.bump("not_ready")
        stats.skipped(readiness)
        log.debug("jevk5 skipped family=%s reason=%s", family, readiness)
        return False
    if budget is None:
        stats.skipped("no_budget")
        log.debug("jevk5 skipped family=%s reason=no_budget", family)
        return False
    remaining = getattr(budget, "remaining", None)
    if isinstance(remaining, int) and remaining <= 0:
        stats.bump("budget_exhausted")
        stats.skipped("budget_exhausted")
        log.debug("jevk5 skipped family=%s reason=budget_exhausted", family)
        return False
    if not uncertainty.ambiguous:
        stats.bump("deterministic_fallback")
        stats.skipped(uncertainty.reason)
        log.debug(
            "jevk5 skipped family=%s reason=%s score=%.2f",
            family,
            uncertainty.reason,
            uncertainty.score,
        )
        return False
    stats.bump("eligible")
    log.debug(
        "jevk5 eligible family=%s score=%.2f reasons=%s%s",
        family,
        uncertainty.score,
        "|".join(uncertainty.reasons) or "-",
        "".join(f" {k}={v}" for k, v in sorted(uncertainty.detail.items())),
    )
    return True


# ---------------------------------------------------------------------------
# Asking
# ---------------------------------------------------------------------------


def _question(kind: str, instructions: str, criteria: Any) -> dict[str, Any]:
    question: dict[str, Any] = {"type": kind, "instructions": instructions}
    if criteria is not None:
        question["criteria"] = criteria
    return question


def ask(
    advisor: Any,
    family: str,
    state: Any,
    question: Mapping[str, Any],
    *,
    budget: Any,
    stats: ConsumerStats,
) -> Any:
    """One advisor call with the outcome recorded. Never raises.

    Uses ``decide_outcome`` when the advisor offers it (so a refusal is
    attributed), and the typed ``choose``/``score``/``noul`` API otherwise.
    """
    stats.bump("calls_requested")
    log.debug("jevk5 called family=%s", family)
    kind = str(question.get("type"))
    criteria = question.get("criteria")
    try:
        outcome = getattr(advisor, "decide_outcome", None)
        if callable(outcome):
            advice, reason = outcome(family, state, dict(question), budget=budget)
        elif kind == "choice":
            advice = advisor.choose(
                family, state, question["instructions"], criteria, budget=budget
            )
            reason = "ok" if advice is not None else "unavailable"
        elif kind == "score":
            advice = advisor.score(family, state, question["instructions"], criteria, budget=budget)
            reason = "ok" if advice is not None else "unavailable"
        else:
            advice = advisor.noul(family, state, question["instructions"], budget=budget)
            reason = "ok" if advice is not None else "unavailable"
    except Exception as exc:  # noqa: BLE001 - advisory only; deterministic path continues
        from headroom.log_safety import describe_exception

        stats.skipped("error")
        log.debug("jevk5 failed family=%s reason=%s", family, describe_exception(exc))
        return None
    if advice is None:
        if reason == "low_confidence":
            stats.bump("advice_ignored_low_confidence")
        elif reason == "budget_exhausted":
            stats.bump("budget_exhausted")
        stats.skipped(reason if reason in _OUTCOMES else "unavailable")
        stats.bump("deterministic_fallback")
        log.debug("jevk5 ignored family=%s reason=%s", family, reason)
        return None
    stats.bump("advice_received")
    return advice


def ask_choice(
    advisor: Any,
    family: str,
    state: Any,
    instructions: str,
    options: Mapping[str, str] | Sequence[str],
    *,
    budget: Any,
    stats: ConsumerStats,
) -> Any:
    criteria = dict(options) if isinstance(options, Mapping) else list(options)
    return ask(
        advisor,
        family,
        state,
        _question("choice", instructions, criteria),
        budget=budget,
        stats=stats,
    )


def ask_score(
    advisor: Any,
    family: str,
    state: Any,
    instructions: str,
    levels: Sequence[str],
    *,
    budget: Any,
    stats: ConsumerStats,
) -> Any:
    return ask(
        advisor,
        family,
        state,
        _question("score", instructions, list(levels)),
        budget=budget,
        stats=stats,
    )


def applied(family: str, advice: Any, stats: ConsumerStats) -> None:
    """Record that bounded advice actually changed (or confirmed) the outcome."""
    stats.bump("advice_applied")
    log.debug(
        "jevk5 applied family=%s confidence=%.2f latency_ms=%.0f",
        family,
        float(getattr(advice, "confidence", 0.0) or 0.0),
        float(getattr(advice, "latency_ms", 0.0) or 0.0),
    )


__all__ = [
    "DEFAULT_AMBIGUITY_THRESHOLD",
    "FAMILIES",
    "FAMILY_ARBITER",
    "FAMILY_COMPLEXITY",
    "FAMILY_REPAIR_RANK",
    "FAMILY_TASK_STATE",
    "FAMILY_TOOL_CATALOG",
    "ConsumerStats",
    "Uncertainty",
    "advisor_readiness",
    "ambiguity_threshold",
    "applied",
    "ask",
    "ask_choice",
    "ask_score",
    "clip",
    "closeness",
    "component_disagreement",
    "consumer_snapshot",
    "consumer_stats",
    "frontier_density",
    "rank_unstable",
    "reset_consumers",
    "should_consult",
    "spread",
]
