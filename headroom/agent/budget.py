"""Global live-zone token budget and marginal-information allocator (unified-21 #8).

Two steps, both deterministic and order-independent:

1. :func:`compute_budget` — tokens actually available to the mutable live
   zone: the provider model's *effective* context limit (context-guard learned
   cap, ``context-1m`` beta honoured only for models whose real window is 1M —
   never "1M" merely because a client claims it), minus the frozen prefix,
   the mandatory system/tool definitions, the reserved generation and a
   safety margin. ``HEADROOM_CONTEXT_GUARD=0`` keeps the registry limit (the
   guard's own client-compaction nudge stays the guard's job), and
   ``x-headroom-keep-last-turns`` trimming is recorded but never drops task
   state (TaskState lives in the agent ledger, not in trimmed history).
2. :func:`allocate` — mandatory tokens first (the latest user turn and every
   unit's smallest invariant-safe form: user goal, active errors, user-named
   paths/symbols, tool statuses and truncation metadata are pinned there),
   then optional tokens by marginal ``relevance + novelty + anomaly +
   (1 - recoverability)`` per token, after a modest cross-source diversity
   share. When even the mandatory floor does not fit, the budget is marked
   ``insufficient`` and the arbiter prefers exact CCR externalization or the
   original (the context guard then drives client compaction) — it never
   fabricates a smaller "fit".
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

#: Fraction of the optional budget reserved for spreading across sources.
DIVERSITY_SHARE = 0.15
#: Safety margin: the larger of this many tokens and SAFETY_FRACTION of the window.
SAFETY_MIN_TOKENS = 1024
SAFETY_FRACTION = 0.02
#: Generation reserve when the request names no max_tokens.
DEFAULT_RESERVE_FRACTION = 0.10
DEFAULT_RESERVE_CAP = 32_000

#: Model id fragments whose real window is 1M with the ``context-1m`` beta.
_ONE_M_MODELS = (
    "claude-sonnet-4",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
)


def model_supports_1m(model: str) -> bool:
    """True only for model families with a real 1M window behind the beta."""
    name = (model or "").lower().replace("[1m]", "")
    for prefix in ("anthropic/", "bedrock/", "vertex_ai/"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return any(frag in name for frag in _ONE_M_MODELS)


@dataclass(frozen=True)
class BudgetInputs:
    """Request facts the handler knows and the pipeline transform does not."""

    model: str = ""
    model_limit: int = 0  # registry window, before any 1M / learned adjustment
    effective_limit: int = 0  # handler's context-guard effective limit (0 = derive)
    beta_header: str | None = None
    credential_scope: str = ""
    system_tokens: int = 0
    tools_tokens: int = 0
    max_output_tokens: int | None = None
    keep_last_turns: int | None = None
    cache_mode: str = "none"  # "write" | "none"
    cache_ttl_seconds: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "model_limit": self.model_limit,
            "effective_limit": self.effective_limit,
            "context_1m": bool(self.beta_header and "context-1m" in self.beta_header.lower()),
            "system_tokens": self.system_tokens,
            "tools_tokens": self.tools_tokens,
            "max_output_tokens": self.max_output_tokens,
            "keep_last_turns": self.keep_last_turns,
            "cache_mode": self.cache_mode,
        }


@dataclass(frozen=True)
class LiveZoneBudget:
    effective_limit: int
    frozen_tokens: int
    overhead_tokens: int
    reserved_generation: int
    safety_margin: int
    available: int
    context_guard: bool
    keep_last_turns: int | None = None
    cache_mode: str = "none"
    one_m_honoured: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "effective_limit": self.effective_limit,
            "frozen_tokens": self.frozen_tokens,
            "overhead_tokens": self.overhead_tokens,
            "reserved_generation": self.reserved_generation,
            "safety_margin": self.safety_margin,
            "available": self.available,
            "context_guard": self.context_guard,
            "keep_last_turns": self.keep_last_turns,
            "cache_mode": self.cache_mode,
            "one_m_honoured": self.one_m_honoured,
        }


def resolve_effective_limit(inputs: BudgetInputs) -> tuple[int, bool, bool]:
    """``(limit, context_guard_enabled, one_m_honoured)`` for this request."""
    from headroom.proxy.context_guard import (
        context_guard_enabled,
        effective_context_limit,
        has_context_1m_beta,
    )

    guard = context_guard_enabled()
    base = max(int(inputs.model_limit or 0), 0)
    if base <= 0:
        from headroom.models.registry import ModelRegistry

        base = ModelRegistry.get_context_limit(inputs.model or "", default=128_000)
    claimed_1m = has_context_1m_beta(inputs.beta_header)
    honour_1m = claimed_1m and model_supports_1m(inputs.model)
    if not guard:
        return base, False, False
    limit = int(inputs.effective_limit or 0) or effective_context_limit(
        inputs.model, base, inputs.beta_header, scope=inputs.credential_scope
    )
    if limit > base and not honour_1m:
        # The client claims 1M for a model without a real 1M window.
        limit = base
    return max(limit, 0), True, honour_1m and limit > base


def compute_budget(
    inputs: BudgetInputs,
    *,
    frozen_tokens: int,
) -> LiveZoneBudget:
    limit, guard, one_m = resolve_effective_limit(inputs)
    reserve = inputs.max_output_tokens
    if not isinstance(reserve, int) or reserve <= 0:
        reserve = min(DEFAULT_RESERVE_CAP, int(limit * DEFAULT_RESERVE_FRACTION))
    reserve = min(reserve, limit)
    margin = max(SAFETY_MIN_TOKENS, int(limit * SAFETY_FRACTION))
    overhead = max(0, int(inputs.system_tokens)) + max(0, int(inputs.tools_tokens))
    available = limit - max(0, frozen_tokens) - overhead - reserve - margin
    return LiveZoneBudget(
        effective_limit=limit,
        frozen_tokens=max(0, frozen_tokens),
        overhead_tokens=overhead,
        reserved_generation=reserve,
        safety_margin=margin,
        available=max(0, available),
        context_guard=guard,
        keep_last_turns=inputs.keep_last_turns,
        cache_mode=inputs.cache_mode,
        one_m_honoured=one_m,
    )


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetUnit:
    """One allocatable live-zone unit (a tool result)."""

    key: str  # stable identity (resource id + call id); never list position
    source: str  # tool family (diversity grouping)
    original_tokens: int
    min_tokens: int  # smallest invariant-safe candidate (mandatory floor)
    relevance: float = 0.0
    novelty: float = 1.0
    anomaly: float = 0.0
    recoverability: float = 1.0
    fixed: bool = False  # already presented (memoized): its tokens are committed

    @property
    def optional_tokens(self) -> int:
        return max(0, self.original_tokens - self.min_tokens)

    @property
    def value(self) -> float:
        return (
            _unit(self.relevance)
            + _unit(self.novelty)
            + _unit(self.anomaly)
            + (1.0 - _unit(self.recoverability))
        )


@dataclass(frozen=True)
class Allocation:
    budget: LiveZoneBudget
    mandatory_tokens: int
    optional_demand: int
    allowances: Mapping[str, int] = field(default_factory=dict)
    pressure: Mapping[str, float] = field(default_factory=dict)
    insufficient: bool = False

    def allowance(self, key: str, default: int | None = None) -> int | None:
        return self.allowances.get(key, default)

    def under_pressure(self, key: str) -> bool:
        return self.pressure.get(key, 0.0) > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget.to_dict(),
            "mandatory_tokens": self.mandatory_tokens,
            "optional_demand": self.optional_demand,
            "units": len(self.allowances),
            "pressured_units": sum(1 for v in self.pressure.values() if v > 0),
            "insufficient": self.insufficient,
        }


def _unit(x: float) -> float:
    if not isinstance(x, int | float) or math.isnan(x):
        return 0.0
    return max(0.0, min(1.0, float(x)))


def allocate(
    units: Sequence[BudgetUnit],
    budget: LiveZoneBudget,
    *,
    other_live_tokens: int = 0,
    diversity_share: float = DIVERSITY_SHARE,
) -> Allocation:
    """Allowance per unit. Independent of the order ``units`` arrive in."""
    ordered = sorted(units, key=lambda u: u.key)
    fixed_tokens = sum(u.original_tokens for u in ordered if u.fixed)
    live = [u for u in ordered if not u.fixed]
    mandatory = max(0, other_live_tokens) + fixed_tokens + sum(u.min_tokens for u in live)
    demand = sum(u.optional_tokens for u in live)
    remaining = budget.available - mandatory
    allowances: dict[str, int] = {u.key: u.original_tokens for u in ordered if u.fixed}
    if remaining >= demand:
        allowances.update({u.key: u.original_tokens for u in live})
        return Allocation(budget, mandatory, demand, allowances, {})
    insufficient = remaining < 0
    give: dict[str, int] = {u.key: 0 for u in live}
    pool = max(0, remaining)
    # Diversity: an equal share per source, each capped by that source's demand.
    by_source: dict[str, list[BudgetUnit]] = {}
    for u in live:
        if u.optional_tokens:
            by_source.setdefault(u.source, []).append(u)
    if by_source and pool:
        share = int(pool * max(0.0, min(1.0, diversity_share))) // len(by_source)
        for source in sorted(by_source):
            left = share
            for u in _by_density(by_source[source]):
                take = min(left, u.optional_tokens - give[u.key])
                give[u.key] += take
                left -= take
                pool -= take
                if left <= 0:
                    break
    for u in _by_density(live):
        if pool <= 0:
            break
        take = min(pool, u.optional_tokens - give[u.key])
        give[u.key] += take
        pool -= take
    pressure: dict[str, float] = {}
    for u in live:
        allowances[u.key] = u.min_tokens + give[u.key]
        short = u.optional_tokens - give[u.key]
        if short > 0:
            pressure[u.key] = 1.0 if insufficient else short / max(u.optional_tokens, 1)
    return Allocation(budget, mandatory, demand, allowances, pressure, insufficient)


def _by_density(units: Iterable[BudgetUnit]) -> list[BudgetUnit]:
    """Highest marginal value per optional token first; ties by stable key."""
    return sorted(
        units,
        key=lambda u: (-(u.value / max(u.optional_tokens, 1)), u.key),
    )


def with_fixed(unit: BudgetUnit, tokens: int) -> BudgetUnit:
    return replace(unit, original_tokens=tokens, min_tokens=tokens, fixed=True)


__all__ = [
    "Allocation",
    "BudgetInputs",
    "BudgetUnit",
    "LiveZoneBudget",
    "allocate",
    "compute_budget",
    "model_supports_1m",
    "request_budget_inputs",
    "resolve_effective_limit",
    "with_fixed",
]


def _json_tokens(tokenizer: Any, value: Any) -> int:
    if not value:
        return 0
    import json

    try:
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        return int(tokenizer.count_text(text)) if tokenizer is not None else len(text) // 4
    except Exception:  # noqa: BLE001 - an estimate is enough for a budget
        return 0


def _has_cache_control(value: Any) -> bool:
    if isinstance(value, dict):
        if isinstance(value.get("cache_control"), dict):
            return True
        return any(_has_cache_control(v) for k, v in value.items() if k == "content")
    if isinstance(value, list):
        return any(_has_cache_control(v) for v in value)
    return False


def request_budget_inputs(
    body: Mapping[str, Any],
    *,
    model: str,
    model_limit: int,
    effective_limit: int,
    tokenizer: Any,
    beta_header: str | None = None,
    credential_scope: str = "",
    keep_last_turns: str | int | None = None,
    cache_ttl_seconds: int | None = None,
    system_key: str = "system",
) -> BudgetInputs:
    """Handler-side :class:`BudgetInputs` from a provider request body."""
    try:
        klt = int(keep_last_turns) if keep_last_turns not in (None, "") else None
    except (TypeError, ValueError):
        klt = None
    max_out = (
        body.get("max_tokens") or body.get("max_completion_tokens") or body.get("max_output_tokens")
    )
    cache = (
        "write"
        if _has_cache_control(body.get("messages")) or _has_cache_control(body.get(system_key))
        else "none"
    )
    return BudgetInputs(
        model=model,
        model_limit=int(model_limit or 0),
        effective_limit=int(effective_limit or 0),
        beta_header=beta_header,
        credential_scope=credential_scope,
        system_tokens=_json_tokens(tokenizer, body.get(system_key)),
        tools_tokens=_json_tokens(tokenizer, body.get("tools")),
        max_output_tokens=int(max_out) if isinstance(max_out, int) and max_out > 0 else None,
        keep_last_turns=klt,
        cache_mode=cache,
        cache_ttl_seconds=cache_ttl_seconds,
    )
