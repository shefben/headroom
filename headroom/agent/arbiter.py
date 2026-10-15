"""Multi-candidate Transform Arbiter (unified-21 #6).

The one owner of "which form of this live-zone unit is forwarded". It takes
over from the SafetyGate ladder in the tool-admission dispatcher:

1. Candidates of the applicable kinds (:data:`KINDS`: ``original``,
   ``lossless``, ``type_specific``, ``query_aware``, ``structural``,
   ``delta``, ``indexed_externalization``) arrive with their provenance and
   preparation latency. The original is always a candidate.
2. **Hard filters run before any scoring**: the SafetyGate check (invariants,
   lossy-risk/volatile policy, verified markers, unknown policy -> reject),
   and actual billed-token cost after cache effects and expected retrieval
   overhead — a compressed form that does not save billed cost is rejected.
3. Survivors get a deterministic utility with the fixed weights
   (:data:`WEIGHTS`, sum 1.0): relevance .28, soft recall .20, recovery .16,
   net token savings .14, cache economics .10, learned prior .07,
   latency .05. Utility depends only on the complete input (latency uses a
   deterministic estimate; the measured time is recorded, never scored), so
   the winner is independent of scheduling and of tool-result order.
4. Stable tie-break: less irreversible loss, then lower net cost, then the
   original.
5. JevK5 compares at most :data:`MAX_FINALISTS` finalists only when the task
   is ambiguous or the top utilities are within :data:`JEV_MARGIN`; the
   blend is ``(1-w)*utility + w*probability`` with the advisor's capped
   weight (default .25, max .40). No advisor/advice => ``w = 0``.

Work is bounded per *request* (:class:`RequestLimits`), not just per block.
Accounting keeps realized token/cost savings separate from estimated avoided
turns/time; no payload is ever logged.
"""

from __future__ import annotations

import logging
import re
import statistics
import threading
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from headroom.agent.feedback import NEUTRAL_PRIOR, FeedbackKey, RetentionFeedback
from headroom.log_safety import describe_exception

log = logging.getLogger(__name__)

FEATURE_ARBITER = "agent_arbiter"

KINDS = (
    "original",
    "lossless",
    "type_specific",
    "query_aware",
    "structural",
    "delta",
    "indexed_externalization",
)

WEIGHTS: Mapping[str, float] = {
    "relevance": 0.28,
    "soft_recall": 0.20,
    "recovery": 0.16,
    "net_savings": 0.14,
    "cache": 0.10,
    "prior": 0.07,
    "latency": 0.05,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

MAX_FINALISTS = 16
JEV_MARGIN = 0.15
LATENCY_BUDGET_MS = 50.0
#: Selective retrieval returns at most this many tokens (indexed CCR default).
RETRIEVAL_TOKENS_CAP = 1200
#: Fixed cost of one retrieval round-trip (tool call + framing), in tokens.
RETRIEVAL_CALL_TOKENS = 120
#: Per-kind expected retrieval probability before feedback has learned one.
DEFAULT_RETRIEVAL_P: Mapping[str, float] = {
    "original": 0.0,
    "lossless": 0.0,
    "delta": 0.05,
    "type_specific": 0.15,
    "query_aware": 0.15,
    "structural": 0.20,
    "indexed_externalization": 0.50,
}
#: Recovery score of a form that is exact only via one CCR retrieval.
CCR_RECOVERY = 0.9
#: Deterministic preparation-latency estimate (ms) per kind and per 1k chars.
_LATENCY_BASE = {
    "original": 0.0,
    "lossless": 0.5,
    "indexed_externalization": 0.5,
    "delta": 2.0,
    "type_specific": 3.0,
    "structural": 3.0,
    "query_aware": 4.0,
}
_LATENCY_PER_KCHAR = 0.05

_TIE_EPS = 1e-9


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """One applicable form of the unit (``kind='original'`` = forward as-is)."""

    kind: str
    text: str
    tier: str = "aggressive"
    label: str = ""
    provenance: Any = None  # headroom.transforms.safety_gate.Provenance
    reversible: bool = False  # presentation itself is exact (lossless)
    prep_ms: float = 0.0  # measured, recorded only

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown candidate kind {self.kind!r}")


@dataclass
class CandidateRecord:
    kind: str
    label: str
    tier: str
    tokens: int
    allowed: bool
    reasons: tuple[str, ...] = ()
    soft_recall: float = 1.0
    invariant_recall: float = 1.0
    relevance: float = 1.0
    recovery: float = 1.0
    irreversible_loss: float = 0.0
    exact_recoverable: bool = True
    volatile_introduced: int = 0
    retrieval_p: float = 0.0
    retrieval_overhead: float = 0.0
    net_cost: float = 0.0
    prior: float = NEUTRAL_PRIOR
    est_latency_ms: float = 0.0
    prep_ms: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    utility: float = 0.0
    blended: float = 0.0
    candidate: Candidate | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "tokens": self.tokens,
            "allowed": self.allowed,
            "reasons": list(self.reasons),
            "utility": round(self.utility, 6),
            "blended": round(self.blended, 6),
            "net_cost": round(self.net_cost, 2),
            "soft_recall": round(self.soft_recall, 4),
            "relevance": round(self.relevance, 4),
        }


@dataclass
class RequestLimits:
    """Per-request bound on candidate expansion and ML-backed views."""

    max_candidates: int = 64
    max_views: int = 24
    used_candidates: int = 0
    used_views: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def take_view(self) -> bool:
        with self._lock:
            if self.used_views >= self.max_views:
                return False
            self.used_views += 1
            return True

    def take_candidates(self, n: int) -> int:
        with self._lock:
            grant = max(0, min(n, self.max_candidates - self.used_candidates))
            self.used_candidates += grant
            return grant


@dataclass
class ArbiterContext:
    original: str
    task: Any = None  # gate task payload
    terms: tuple[str, ...] = ()
    policy: Any = None
    model: str = ""
    tokenizer: Any = None
    cache_mode: str = "none"  # "write" | "none"
    cache_ttl_seconds: int | None = None
    feedback: RetentionFeedback | None = None
    feedback_key: FeedbackKey | None = None  # strategy filled per candidate
    ambiguous: bool = False
    advisor: Any = None
    advisor_budget: Any = None
    limits: RequestLimits | None = None


@dataclass(frozen=True)
class ArbiterDecision:
    winner: CandidateRecord
    records: tuple[CandidateRecord, ...]
    margin: float
    jev_weight: float = 0.0
    jev_confidence: float | None = None
    allowance: int | None = None
    pressure: float = 0.0
    insufficient: bool = False
    cache_mode: str = "none"

    @property
    def candidate(self) -> Candidate | None:
        return None if self.winner.kind == "original" else self.winner.candidate

    @property
    def rejections(self) -> tuple[str, ...]:
        return tuple(r for rec in self.records if not rec.allowed for r in rec.reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner.kind,
            "label": self.winner.label,
            "margin": round(self.margin, 6),
            "jev_weight": self.jev_weight,
            "jev_confidence": self.jev_confidence,
            "allowance": self.allowance,
            "pressure": round(self.pressure, 4),
            "insufficient": self.insufficient,
            "cache_mode": self.cache_mode,
            "candidates": [r.summary() for r in self.records],
        }


@dataclass(frozen=True)
class Evaluation:
    """Hard-filtered, allowance-independent candidate metrics (step 1-2)."""

    original_tokens: int
    records: tuple[CandidateRecord, ...]

    @property
    def survivors(self) -> tuple[CandidateRecord, ...]:
        return tuple(r for r in self.records if r.allowed)

    @property
    def min_tokens(self) -> int:
        return min(r.tokens for r in self.survivors)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clip(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _tokens(tokenizer: Any, text: str) -> int:
    if tokenizer is not None:
        try:
            return int(tokenizer.count_text(text))
        except Exception:  # noqa: BLE001 - fall back to the byte estimate
            pass
    return max(1, (len(text.encode("utf-8", "replace")) + 3) // 4)


def cache_multiplier(cache_mode: str, ttl_seconds: int | None) -> float:
    """Billed multiplier for live-zone input under the request's cache mix."""
    if cache_mode != "write":
        return 1.0
    return 2.0 if (ttl_seconds or 0) >= 3600 else 1.25


def estimated_latency_ms(kind: str, original: str) -> float:
    return _LATENCY_BASE.get(kind, 3.0) + _LATENCY_PER_KCHAR * (len(original) / 1000.0)


_SALIENT = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]{3,}")
SALIENT_SAMPLE = 48


_DIGITS = re.compile(r"[0-9]+")


def salient_terms(original: str) -> tuple[str, ...]:
    """Distinct *non-repetitive* identifier tokens, sampled evenly over the text.

    Tokens whose digit-normalized shape recurs (``test_case_0001`` ...,
    ``crate_17``) are the repetition a structural view is meant to fold, so
    they do not count as information a form must keep.
    """
    tokens = _SALIENT.findall(original)
    shapes: dict[str, int] = {}
    for tok in tokens:
        shape = _DIGITS.sub("#", tok)
        shapes[shape] = shapes.get(shape, 0) + 1
    seen = list(dict.fromkeys(t for t in tokens if shapes[_DIGITS.sub("#", t)] <= 2))
    if len(seen) <= SALIENT_SAMPLE:
        return tuple(seen)
    step = len(seen) / SALIENT_SAMPLE
    return tuple(seen[int(i * step)] for i in range(SALIENT_SAMPLE))


def relevance_retained(original: str, text: str, terms: Sequence[str]) -> float:
    """Fraction of query terms present in the original that survive.

    With no query term in the original, falls back to a weaker signal: half
    credit plus half the coverage of salient identifiers sampled across the
    whole original (so a form that hides nearly everything still scores low).
    """
    present = [t for t in terms if t and t in original]
    if present:
        return sum(1 for t in present if t in text) / len(present)
    salient = salient_terms(original)
    if not salient:
        return 1.0
    coverage = sum(1 for t in salient if t in text) / len(salient)
    return 0.5 + 0.5 * coverage


def confidence_bin(margin: float) -> str:
    if margin < 0.05:
        return "lt_0.05"
    if margin < JEV_MARGIN:
        return "lt_0.15"
    if margin < 0.30:
        return "lt_0.30"
    return "ge_0.30"


# ---------------------------------------------------------------------------
# Arbiter
# ---------------------------------------------------------------------------


class _Telemetry:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.decisions: Counter[str] = Counter()
        self.families: Counter[str] = Counter()
        self.strategies: Counter[str] = Counter()
        self.rejections: Counter[str] = Counter()
        self.confidence: Counter[str] = Counter()
        self.cache_mix: Counter[str] = Counter()
        self.overhead_ms: deque[float] = deque(maxlen=2048)
        self.realized_tokens = 0
        self.realized_cost_units = 0.0
        self.estimated_avoided_turns = 0.0
        self.jev_calls = 0
        self.jev_applied = 0
        self.pressured = 0
        self.insufficient = 0
        self.allocations = 0
        self.last_allocation: dict[str, Any] | None = None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            ms = sorted(self.overhead_ms)
            p50 = statistics.median(ms) if ms else 0.0
            p95 = ms[min(len(ms) - 1, int(round(0.95 * (len(ms) - 1))))] if ms else 0.0
            return {
                "decisions": dict(self.decisions),
                "decision_families": dict(self.families),
                "selected_strategy": dict(self.strategies),
                "candidate_rejections": dict(self.rejections),
                "confidence_bins": dict(self.confidence),
                "cache_mix": dict(self.cache_mix),
                "local_overhead_ms": {"p50": round(p50, 3), "p95": round(p95, 3), "n": len(ms)},
                "realized": {
                    "tokens_saved": self.realized_tokens,
                    "cost_units_saved": round(self.realized_cost_units, 2),
                },
                "estimated": {"avoided_turns": round(self.estimated_avoided_turns, 3)},
                "jev": {"calls": self.jev_calls, "applied": self.jev_applied},
                "pressured_units": self.pressured,
                "insufficient_budget_units": self.insufficient,
                "allocations": self.allocations,
                "last_allocation": self.last_allocation,
            }


class TransformArbiter:
    """Stateless selection plus process-wide, payload-free telemetry."""

    def __init__(self) -> None:
        self.telemetry = _Telemetry()

    # -- step 1-2: metrics + hard filters ------------------------------------

    def evaluate(self, candidates: Sequence[Candidate], ctx: ArbiterContext) -> Evaluation:
        from headroom.transforms.safety_gate import SafetyGate

        original = ctx.original
        orig_tokens = _tokens(ctx.tokenizer, original)
        mult = cache_multiplier(ctx.cache_mode, ctx.cache_ttl_seconds)
        orig_cost = orig_tokens * mult
        records: list[CandidateRecord] = [
            CandidateRecord(
                kind="original",
                label="original",
                tier="original",
                tokens=orig_tokens,
                allowed=True,
                net_cost=orig_cost,
                est_latency_ms=0.0,
            )
        ]
        pool = [c for c in candidates if c.kind != "original" and c.text and c.text != original]
        if ctx.limits is not None:
            grant = ctx.limits.take_candidates(len(pool))
            if grant < len(pool):
                for c in pool[grant:]:
                    records.append(
                        CandidateRecord(
                            c.kind,
                            c.label,
                            c.tier,
                            _tokens(ctx.tokenizer, c.text),
                            False,
                            ("request_candidate_cap",),
                            candidate=c,
                        )
                    )
                pool = pool[:grant]
        for cand in pool:
            tokens = _tokens(ctx.tokenizer, cand.text)
            rec = CandidateRecord(
                kind=cand.kind,
                label=cand.label or cand.kind,
                tier=cand.tier,
                tokens=tokens,
                allowed=False,
                prep_ms=cand.prep_ms,
                est_latency_ms=estimated_latency_ms(cand.kind, original),
                candidate=cand,
            )
            try:
                decision = SafetyGate.check(
                    original,
                    cand.text,
                    task=ctx.task,
                    policy=ctx.policy,
                    provenance=cand.provenance,
                    tier=cand.tier,
                    model=ctx.model,
                )
            except Exception as exc:  # noqa: BLE001 - an unscorable candidate is rejected
                rec.reasons = (f"gate_error:{type(exc).__name__}",)
                records.append(rec)
                continue
            rec.invariant_recall = decision.invariant_recall
            rec.soft_recall = (
                decision.facts_preserved / decision.facts_total if decision.facts_total else 1.0
            )
            rec.irreversible_loss = decision.irreversible_loss_ratio
            rec.exact_recoverable = decision.exact_recoverable
            rec.volatile_introduced = decision.volatile_introduced
            if not decision.allow:
                rec.reasons = decision.reasons or ("gate_reject",)
                records.append(rec)
                continue
            rec.relevance = relevance_retained(original, cand.text, ctx.terms)
            if cand.reversible:
                rec.recovery = 1.0
            elif decision.exact_recoverable:
                rec.recovery = CCR_RECOVERY
            else:
                rec.recovery = _clip(1.0 - decision.irreversible_loss_ratio)
            key = (
                replace(ctx.feedback_key, strategy=cand.kind)
                if ctx.feedback_key is not None
                else None
            )
            learned = (
                ctx.feedback.retrieval_rate(key)
                if ctx.feedback is not None and key is not None
                else None
            )
            if cand.reversible:
                p = 0.0
            else:
                p = DEFAULT_RETRIEVAL_P.get(cand.kind, 0.2) if learned is None else learned
            rec.retrieval_p = p
            rec.retrieval_overhead = p * (
                min(orig_tokens, RETRIEVAL_TOKENS_CAP) + RETRIEVAL_CALL_TOKENS
            )
            rec.net_cost = tokens * mult + rec.retrieval_overhead
            if ctx.feedback is not None and key is not None:
                rec.prior = ctx.feedback.prior(key)
            if rec.net_cost >= orig_cost:
                rec.reasons = ("no_billed_savings",)
                records.append(rec)
                continue
            rec.allowed = True
            records.append(rec)
        return Evaluation(orig_tokens, tuple(records))

    # -- step 3-5: utility, tie-break, JevK5 -------------------------------------

    def choose(
        self,
        evaluation: Evaluation,
        ctx: ArbiterContext,
        *,
        allowance: int | None = None,
        pressure: float = 0.0,
        insufficient: bool = False,
        family: str = "",
    ) -> ArbiterDecision:
        import time

        started = time.perf_counter()
        orig = evaluation.records[0]
        orig_tokens = evaluation.original_tokens
        need = (orig_tokens - allowance) if allowance is not None else 0
        survivors = list(evaluation.survivors)
        for rec in survivors:
            saved_cost = max(0.0, orig.net_cost - rec.net_cost)
            if need > 0:
                fits = rec.tokens <= (allowance or 0)
                savings = 1.0 if fits else _clip((orig_tokens - rec.tokens) / max(need, 1))
            else:
                savings = _clip(saved_cost / orig.net_cost) if orig.net_cost else 0.0
            stability = 1.0 - (0.5 if rec.volatile_introduced > 0 else 0.0)
            future = (
                _clip((orig_tokens - rec.tokens) / max(orig_tokens, 1))
                if ctx.cache_mode == "write"
                else 0.0
            )
            comps = {
                "relevance": _clip(rec.relevance),
                "soft_recall": _clip(rec.soft_recall),
                "recovery": _clip(rec.recovery),
                "net_savings": savings,
                "cache": 0.5 * stability + 0.5 * future,
                "prior": _clip(rec.prior),
                "latency": 1.0 - _clip(rec.est_latency_ms / LATENCY_BUDGET_MS),
            }
            rec.components = comps
            rec.utility = sum(WEIGHTS[k] * v for k, v in comps.items())
            rec.blended = rec.utility

        def order(rec: CandidateRecord) -> tuple[Any, ...]:
            return (
                -round(rec.blended, 9),
                round(rec.irreversible_loss, 9),
                round(rec.net_cost, 6),
                0 if rec.kind == "original" else 1,
                KINDS.index(rec.kind),
                rec.label,
            )

        survivors.sort(key=order)
        margin = survivors[0].utility - survivors[1].utility if len(survivors) > 1 else 1.0
        jev_w, jev_conf = 0.0, None
        if len(survivors) > 1 and (ctx.ambiguous or margin < JEV_MARGIN):
            jev_w, jev_conf = self._jev_blend(survivors[:MAX_FINALISTS], ctx, family)
            if jev_w > 0:
                survivors.sort(key=order)
        winner = survivors[0]
        decision = ArbiterDecision(
            winner=winner,
            records=tuple(sorted(evaluation.records, key=lambda r: KINDS.index(r.kind))),
            margin=margin,
            jev_weight=jev_w,
            jev_confidence=jev_conf,
            allowance=allowance,
            pressure=pressure,
            insufficient=insufficient,
            cache_mode=ctx.cache_mode,
        )
        self._record(decision, family, evaluation, (time.perf_counter() - started) * 1000.0)
        return decision

    def _jev_blend(
        self, finalists: list[CandidateRecord], ctx: ArbiterContext, family: str
    ) -> tuple[float, float | None]:
        advisor, budget = ctx.advisor, ctx.advisor_budget
        if advisor is None or budget is None:
            return 0.0, None
        options = {
            f"c{i}": (
                f"{r.kind}: {r.tokens} tokens, recall {r.soft_recall:.2f}, "
                f"relevance {r.relevance:.2f}, recovery {r.recovery:.2f}"
            )
            for i, r in enumerate(finalists)
        }
        state = {
            "family": family,
            "goal": (" ".join(ctx.terms))[:400],
            "original_tokens": finalists[0].tokens if finalists[0].kind == "original" else None,
            "candidates": options,
        }
        with self.telemetry.lock:
            self.telemetry.jev_calls += 1
        try:
            advice = advisor.choose(
                "transform_arbiter",
                state,
                "Which form of this tool output best serves the task while keeping "
                "every needed fact?",
                options,
                budget=budget,
            )
        except Exception as exc:  # noqa: BLE001 - advisory only
            log.debug("arbiter advisor failed: %s", describe_exception(exc))
            return 0.0, None
        if advice is None:
            return 0.0, None
        w = max(0.0, min(0.40, float(getattr(advice, "weight", 0.0) or 0.0)))
        probs = getattr(advice, "probabilities", {}) or {}
        if w <= 0:
            return 0.0, None
        for i, rec in enumerate(finalists):
            p = _clip(float(probs.get(f"c{i}", 0.0) or 0.0))
            rec.blended = (1.0 - w) * rec.utility + w * p
        with self.telemetry.lock:
            self.telemetry.jev_applied += 1
        return w, float(getattr(advice, "confidence", 0.0) or 0.0)

    def _record(self, d: ArbiterDecision, family: str, evaluation: Evaluation, ms: float) -> None:
        t = self.telemetry
        with t.lock:
            t.decisions[d.winner.kind] += 1
            t.families[family or "unknown"] += 1
            t.strategies[d.winner.label or d.winner.kind] += 1
            for rec in evaluation.records:
                if not rec.allowed:
                    for reason in rec.reasons:
                        t.rejections[reason.split(":")[0]] += 1
            t.confidence[confidence_bin(d.margin)] += 1
            t.cache_mix[d.cache_mode] += 1
            t.overhead_ms.append(ms + sum(r.prep_ms for r in evaluation.records))
            if d.pressure > 0:
                t.pressured += 1
            if d.insufficient:
                t.insufficient += 1
            orig = evaluation.records[0]
            if d.winner.kind != "original":
                t.realized_tokens += max(0, orig.tokens - d.winner.tokens)
                t.realized_cost_units += max(0.0, orig.net_cost - d.winner.net_cost)

    def note_allocation(self, allocation: Any) -> None:
        """Record the request budget/allocation summary (counts only)."""
        with self.telemetry.lock:
            self.telemetry.allocations += 1
            self.telemetry.last_allocation = allocation.to_dict()

    def note_avoided_turns(self, n: float) -> None:
        with self.telemetry.lock:
            self.telemetry.estimated_avoided_turns += max(0.0, n)

    def status(self) -> dict[str, Any]:
        return {"weights": dict(WEIGHTS), **self.telemetry.snapshot()}


_ARBITER: TransformArbiter | None = None
_ARBITER_LOCK = threading.Lock()


def get_arbiter() -> TransformArbiter:
    global _ARBITER
    if _ARBITER is None:
        with _ARBITER_LOCK:
            if _ARBITER is None:
                _ARBITER = TransformArbiter()
    return _ARBITER


def reset_arbiter() -> None:
    global _ARBITER
    with _ARBITER_LOCK:
        _ARBITER = None


__all__ = [
    "ArbiterContext",
    "ArbiterDecision",
    "Candidate",
    "CandidateRecord",
    "Evaluation",
    "FEATURE_ARBITER",
    "KINDS",
    "RequestLimits",
    "TransformArbiter",
    "WEIGHTS",
    "cache_multiplier",
    "get_arbiter",
    "relevance_retained",
    "reset_arbiter",
]
