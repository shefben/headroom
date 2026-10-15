"""SafetyGate: deterministic invariants, lossy-risk policy and backoff (unified-21 #7, #13).

The gate itself is implemented once, in Rust (``headroom_core::safety_gate``),
and exposed through ``headroom._core``; this module is the typed Python face
of that single implementation. Every lossy candidate the Python
``ContentRouter`` produces passes :meth:`SafetyGate.admit` before it may
replace original bytes, exactly as every live-zone block does in Rust.

Contract (see the Rust module docs for the full fact list):

* ``check(original, candidate, task, policy, provenance)`` returns
  ``ALLOW``/``REJECT`` with reason codes and the invariant recall.
* ``CompressionPolicy.max_lossy_ratio`` bounds *irreversible* loss;
  ``wire_drop_ratio`` (bytes removed from the wire) is reported separately and
  is unlimited when an exact retrieval path covers the dropped bytes.
  ``volatile_token_threshold`` bounds volatile tokens a candidate introduces.
  A missing or out-of-range policy fails to the original.
* The backoff ladder is aggressive -> conservative (policy-fit) ->
  structural/lossless -> CCR exact externalization -> original, evaluated by
  the Rust ``select``. JevK5 never takes part.

Phase 4+ transforms submit candidates through :meth:`SafetyGate.admit` (or
:meth:`SafetyGate.select` with their own rungs); none may bypass it.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

TIERS = ("aggressive", "conservative", "lossless", "ccr_exact")

#: Retrieval markers whose hash the gate verifies (same shapes the Rust gate
#: and ``ccr.tool_injection`` recognise).
_MARKER_HASH_RE = re.compile(r"(?:<<ccr:|hash=)([a-f0-9]{12,24})\b")


@dataclass(frozen=True)
class GateDecision:
    """One gate verdict (mirror of the Rust ``Decision``)."""

    allow: bool
    reasons: tuple[str, ...]
    invariant_recall: float
    facts_total: int
    facts_preserved: int
    facts_recoverable: int
    original_tokens: int
    candidate_tokens: int
    wire_drop_ratio: float
    irreversible_loss_ratio: float
    volatile_introduced: int
    exact_recoverable: bool
    tier: str = "aggressive"

    @property
    def verdict(self) -> str:
        return "ALLOW" if self.allow else "REJECT"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], tier: str = "aggressive") -> GateDecision:
        return cls(
            allow=bool(value.get("allow")),
            reasons=tuple(str(r) for r in value.get("reasons") or ()),
            invariant_recall=float(value.get("invariant_recall", 0.0)),
            facts_total=int(value.get("facts_total", 0)),
            facts_preserved=int(value.get("facts_preserved", 0)),
            facts_recoverable=int(value.get("facts_recoverable", 0)),
            original_tokens=int(value.get("original_tokens", 0)),
            candidate_tokens=int(value.get("candidate_tokens", 0)),
            wire_drop_ratio=float(value.get("wire_drop_ratio", 0.0)),
            irreversible_loss_ratio=float(value.get("irreversible_loss_ratio", 0.0)),
            volatile_introduced=int(value.get("volatile_introduced", 0)),
            exact_recoverable=bool(value.get("exact_recoverable")),
            tier=str(value.get("tier") or tier),
        )


@dataclass
class Provenance:
    """How a candidate was produced and what makes it recoverable."""

    content_type: str | None = None
    strategy: str = ""
    reversible: bool = False
    retrieval_available: bool = False
    ccr_payloads: dict[str, str] = field(default_factory=dict)
    source_truncated: bool = False
    cache_frozen: bool = False
    net_cost_gain: float | None = None
    original_tokens: int | None = None
    candidate_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_type": self.content_type,
            "strategy": self.strategy,
            "reversible": self.reversible,
            "retrieval_available": self.retrieval_available,
            "ccr_payloads": dict(self.ccr_payloads),
            "source_truncated": self.source_truncated,
            "cache_frozen": self.cache_frozen,
            "net_cost_gain": self.net_cost_gain,
            "original_tokens": self.original_tokens,
            "candidate_tokens": self.candidate_tokens,
        }


@dataclass
class GateCandidate:
    tier: str
    text: str
    provenance: Provenance
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        if self.tier not in TIERS:
            raise ValueError(f"unknown ladder tier {self.tier!r}")
        return {"tier": self.tier, "text": self.text, "provenance": self.provenance.to_dict()}


@dataclass(frozen=True)
class GateSelection:
    """Ladder outcome; ``candidate is None`` means the original wins."""

    candidate: GateCandidate | None
    tier: str
    decisions: tuple[GateDecision, ...]

    @property
    def first_decision(self) -> GateDecision | None:
        return self.decisions[0] if self.decisions else None


def policy_limits(policy: Any) -> dict[str, Any] | None:
    """The two enforced limits of a ``CompressionPolicy`` (``None`` if absent).

    Out-of-range or non-finite limits are passed through unchanged so the
    Rust gate rejects them (fail to original on unknown limits).
    """
    if policy is None:
        return None
    if isinstance(policy, Mapping):
        ratio = policy.get("max_lossy_ratio")
        volatile = policy.get("volatile_token_threshold")
    else:
        ratio = getattr(policy, "max_lossy_ratio", None)
        volatile = getattr(policy, "volatile_token_threshold", None)
    if ratio is None or volatile is None:
        return None
    try:
        ratio_f = float(ratio)
    except (TypeError, ValueError):
        ratio_f = math.nan
    return {
        "max_lossy_ratio": ratio_f if math.isfinite(ratio_f) else None,
        "volatile_token_threshold": int(volatile),
    }


def task_payload(task: Any, fallback_text: str = "") -> dict[str, Any]:
    """Task facts for the gate from a ``RelevanceQuery`` / its dict / plain text.

    Plain text (the legacy ``context`` string) gets the same exact-entity
    extraction the agent query builder uses, so user-named ids, paths and
    symbols are protected even when the agent runtime is off.
    """
    if task is not None and hasattr(task, "to_dict"):
        return dict(task.to_dict())
    if isinstance(task, Mapping):
        return dict(task)
    text = task if isinstance(task, str) else fallback_text
    text = (text or "").strip()
    if not text:
        return {"v": 1, "text": "", "exact_entities": []}
    from headroom.agent.query import extract_entities

    ents = extract_entities(text[:4000])
    exact = [*ents["paths"], *ents["symbols"], *ents["errors"], *ents["identifiers"]]
    return {"v": 1, "text": text[:1536], "exact_entities": list(dict.fromkeys(exact))[:24]}


def _core() -> Any:
    from headroom import _core

    return _core


class SafetyGate:
    """Typed facade over the Rust gate and ladder (stateless)."""

    @staticmethod
    def check(
        original: str,
        candidate: str,
        *,
        task: Any = None,
        policy: Any = None,
        provenance: Provenance | None = None,
        tier: str = "aggressive",
        model: str = "",
    ) -> GateDecision:
        raw = _core().safety_gate_check(
            original,
            candidate,
            json.dumps(task_payload(task)),
            _policy_json(policy),
            json.dumps((provenance or Provenance()).to_dict()),
            tier,
            model,
        )
        return GateDecision.from_dict(json.loads(raw), tier)

    @staticmethod
    def select(
        original: str,
        candidates: Sequence[GateCandidate],
        *,
        task: Any = None,
        policy: Any = None,
        model: str = "",
    ) -> GateSelection:
        if not candidates:
            return GateSelection(None, "original", ())
        raw = json.loads(
            _core().safety_gate_select(
                original,
                json.dumps([c.to_dict() for c in candidates]),
                json.dumps(task_payload(task)),
                _policy_json(policy),
                model,
            )
        )
        decisions = tuple(
            GateDecision.from_dict(d, str(d.get("tier") or "")) for d in raw.get("decisions", ())
        )
        chosen = raw.get("chosen")
        return GateSelection(
            candidates[chosen] if isinstance(chosen, int) else None,
            str(raw.get("tier") or "original"),
            decisions,
        )

    @staticmethod
    def fit(
        original: str,
        aggressive: str | None,
        *,
        task: Any = None,
        policy: Any = None,
        content_type: str = "plain_text",
        model: str = "",
    ) -> str | None:
        """Policy-fit conservative candidate (line-oriented content only)."""
        limits = _policy_json(policy)
        if limits is None:
            return None
        return _core().safety_gate_fit(
            original, aggressive, json.dumps(task_payload(task)), limits, content_type, model
        )

    @staticmethod
    def minify_json(original: str) -> str | None:
        return _core().safety_gate_minify_json(original)

    @staticmethod
    def ccr_stub(original: str, hash_key: str) -> str:
        return _core().safety_gate_ccr_stub(original, hash_key)

    @classmethod
    def admit(
        cls,
        original: str,
        aggressive: GateCandidate,
        *,
        task: Any = None,
        policy: Any = None,
        model: str = "",
        backoff: Callable[[], Iterable[GateCandidate]] | None = None,
    ) -> GateSelection:
        """Gate one aggressive candidate; on rejection evaluate ``backoff()``.

        The backoff rungs are only materialised when the aggressive candidate
        is rejected for a reason a smaller drop could fix (a candidate that
        does not save tokens cannot be rescued by keeping more).
        """
        first = cls.select(original, [aggressive], task=task, policy=policy, model=model)
        if first.candidate is not None or backoff is None:
            return first
        reasons = first.first_decision.reasons if first.first_decision else ()
        if reasons == ("no_token_savings",):
            return first
        rungs = [c for c in backoff() if c.text and c.text != original]
        second = cls.select(original, rungs, task=task, policy=policy, model=model)
        return GateSelection(second.candidate, second.tier, first.decisions + second.decisions)


def _policy_json(policy: Any) -> str | None:
    limits = policy_limits(policy)
    return None if limits is None else json.dumps(limits)


def marker_hashes(text: str) -> list[str]:
    """Distinct retrieval-marker hashes in ``text``, in order."""
    return list(dict.fromkeys(_MARKER_HASH_RE.findall(text or "")))


def verified_payloads(original: str, candidate: str, store: Any | None) -> dict[str, str]:
    """``hash -> stored payload`` for every marker the candidate introduces
    whose entry the store holds right now (read without feedback side effects).
    """
    if store is None:
        return {}
    known = set(marker_hashes(original))
    out: dict[str, str] = {}
    peek = getattr(store, "peek_original", None)
    if not callable(peek):
        return out
    for h in marker_hashes(candidate):
        if h in known:
            continue
        try:
            payload = peek(h)
        except Exception:  # noqa: BLE001 - an unreadable entry is simply unverified
            payload = None
        if isinstance(payload, str):
            out[h] = payload
    return out
