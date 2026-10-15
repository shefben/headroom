"""The one tool-result admission dispatcher (unified-21 #4, #5, #14).

Both entry points use it: the host pre-context hook (Claude Code PostToolUse,
before the result enters host history) and the proxy live-zone transform (any
client, after the fact). Per result it runs the storage sequence:

1. capture completeness from the envelope (never inferred for partials);
2. take the resource content hash (``envelope.version_hash``);
3. index the exact original in CCR;
4. retrieval self-test: read the stored bytes back and compare exactly;
5. build candidates: a structural delta against the last complete version
   of the same resource (when one is visible), then the family view(s);
6. run every candidate through the SafetyGate ladder;
7. replace the result only if a candidate is admitted.

Any failure at any step forwards the original. Outcomes are memoized per
(scope, call, content) so a result already admitted renders to the same bytes
on every later request (prefix-cache stable).
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from headroom.agent.interfaces import ToolResultEnvelope
from headroom.agent.tool_admission.delta import (
    Delta,
    ResourceVersion,
    ResourceVersionStore,
    compute_delta,
    version_for,
)
from headroom.agent.tool_admission.families import ToolFamily, workspace_relative
from headroom.agent.tool_admission.views import (
    INTERCEPTORS,
    MIN_VIEW_CHARS,
    FamilyView,
    ToolInterceptor,
    ViewContext,
    pin_repair,
)
from headroom.log_safety import describe_exception

log = logging.getLogger(__name__)

#: Marker every admitted result carries; a result that already has one (e.g.
#: admitted by the host hook) is never admitted again by the proxy.
ADMITTED_TAG = "[headroom"
MEMO_LIMIT = 4096
_FAMILY_MIN_CHARS = {ToolFamily.READ: 6000}
#: Families whose failure/summary/exit lines are pinned in every view.
_FACT_FAMILIES = frozenset({ToolFamily.SHELL, ToolFamily.TEST, ToolFamily.GIT})


@dataclass(frozen=True)
class AdmissionOutcome:
    """Result of one admission. ``text is None`` means forward the original."""

    text: str | None
    stage: str
    family: str
    label: str = ""
    reasons: tuple[str, ...] = ()
    ccr_hash: str | None = None
    original_bytes: int = 0
    admitted_bytes: int = 0
    tier: str = ""
    delta: Delta | None = None

    @property
    def admitted(self) -> bool:
        return self.text is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "family": self.family,
            "label": self.label,
            "reasons": list(self.reasons),
            "ccr_hash": self.ccr_hash,
            "original_bytes": self.original_bytes,
            "admitted_bytes": self.admitted_bytes,
            "tier": self.tier,
            "delta": (
                {
                    "base": self.delta.base_hash,
                    "current": self.delta.current_hash,
                    "added": self.delta.added,
                    "removed": self.delta.removed,
                    "changed": self.delta.changed,
                    "unchanged": self.delta.unchanged,
                    "unit": self.delta.unit,
                }
                if self.delta is not None
                else None
            ),
        }


@dataclass
class AdmissionRequest:
    """Everything one admission needs besides the envelope."""

    scope: str
    context: ViewContext
    task: Any = None
    policy: Any = None
    store: Any = None
    tokenizer: Any = None
    model: str = ""
    retrieval_available: bool = True
    source: str = "proxy"
    base_visible: Callable[[ResourceVersion], bool] | None = None


@dataclass
class _Counters:
    by_stage: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_family: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_source: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    bytes_in: int = 0
    bytes_out: int = 0
    deltas: int = 0


def _footer(envelope: ToolResultEnvelope, family: ToolFamily, key: str) -> str:
    lines = envelope.line_count
    partial = (
        f" PARTIAL original ({envelope.partial_reason}): it is not the complete output."
        if envelope.is_partial
        else ""
    )
    return (
        f"[headroom: {family.value} view of {lines} lines / {envelope.byte_length} bytes."
        f"{partial} Exact original: Retrieve more: hash={key}]"
    )


class ToolAdmissionDispatcher:
    """See module docstring. Thread-safe; one per process."""

    def __init__(
        self,
        interceptors: Mapping[ToolFamily, ToolInterceptor] | None = None,
        versions: ResourceVersionStore | None = None,
    ) -> None:
        self.interceptors: dict[ToolFamily, ToolInterceptor] = dict(interceptors or INTERCEPTORS)
        self.versions = versions or ResourceVersionStore()
        self._memo: OrderedDict[tuple[str, str, str, str], AdmissionOutcome] = OrderedDict()
        self._lock = threading.Lock()
        self._counters = _Counters()

    # -- public ---------------------------------------------------------------

    def admit(self, envelope: ToolResultEnvelope, request: AdmissionRequest) -> AdmissionOutcome:
        """Admit one result; never raises (failures forward the original)."""
        try:
            family = ToolFamily(envelope.tool_family)
        except ValueError:
            family = ToolFamily.OTHER
        key = (request.scope, envelope.tool_call_id, envelope.version_hash, family.value)
        if envelope.tool_call_id:
            with self._lock:
                memo = self._memo.get(key)
                if memo is not None:
                    self._memo.move_to_end(key)
                    return memo
        try:
            outcome = self._admit(envelope, family, request)
        except Exception as exc:  # noqa: BLE001 - a local failure passes the original through
            log.warning("tool admission failed: %s", describe_exception(exc))
            outcome = AdmissionOutcome(None, "error", family.value, reasons=(type(exc).__name__,))
        self._count(outcome, request.source, envelope)
        if envelope.tool_call_id and outcome.stage not in ("error",):
            with self._lock:
                self._memo[key] = outcome
                while len(self._memo) > MEMO_LIMIT:
                    self._memo.popitem(last=False)
        return outcome

    def status(self) -> dict[str, Any]:
        with self._lock:
            c = self._counters
            return {
                "stages": dict(c.by_stage),
                "admitted_by_family": dict(c.by_family),
                "sources": dict(c.by_source),
                "bytes_in": c.bytes_in,
                "bytes_out": c.bytes_out,
                "deltas": c.deltas,
                "memo": len(self._memo),
                "versions": self.versions.status(),
                "families": [f.value for f in self.interceptors],
            }

    def reset(self) -> None:
        with self._lock:
            self._memo.clear()
            self._counters = _Counters()
        self.versions.clear()

    # -- internals ------------------------------------------------------------

    def _count(self, outcome: AdmissionOutcome, source: str, env: ToolResultEnvelope) -> None:
        with self._lock:
            c = self._counters
            c.by_stage[outcome.stage] += 1
            if outcome.admitted:
                c.by_family[outcome.family] += 1
                c.by_source[source] += 1
                c.bytes_in += env.byte_length
                c.bytes_out += outcome.admitted_bytes
                if outcome.delta is not None:
                    c.deltas += 1

    def _admit(
        self, envelope: ToolResultEnvelope, family: ToolFamily, request: AdmissionRequest
    ) -> AdmissionOutcome:
        original = envelope.content
        size = len(original.encode("utf-8", "replace"))

        def out(stage: str, **kw: Any) -> AdmissionOutcome:
            return AdmissionOutcome(
                kw.pop("text", None), stage, family.value, original_bytes=size, **kw
            )

        interceptor = self.interceptors.get(family)
        if family is ToolFamily.OTHER or interceptor is None:
            return out("unsupported_family")
        if not original or len(original) < _FAMILY_MIN_CHARS.get(family, MIN_VIEW_CHARS):
            return out("small")
        if ADMITTED_TAG in original and "hash=" in original:
            return out("already_admitted")
        ctx = request.context
        if family is ToolFamily.READ and self._is_edit_target(envelope, ctx):
            return out("edit_target_exact")
        store = request.store
        if store is None:
            return out("no_ccr_store")

        # (5a) delta base lookup happens before this version is recorded.
        base = self.versions.get(request.scope, str(envelope.resource_id or ""))
        if base is not None and base.call_id == envelope.tool_call_id:
            base = None  # the record is this very call
        if base is not None and request.base_visible is not None and not request.base_visible(base):
            base = None

        # (3) index the exact original, (4) retrieval self-test.
        query = getattr(request.context, "query", "") or ""
        try:
            key = store.store(
                original,
                "",
                original_tokens=0,
                tool_name=envelope.tool_name or None,
                tool_call_id=envelope.tool_call_id or None,
                compression_strategy=f"tool_admission:{family.value}",
                query_context=query[:500] or None,
            )
        except Exception as exc:  # noqa: BLE001
            return out("ccr_store_failed", reasons=(type(exc).__name__,))
        if not store.verify_exact(key, original):
            return out("ccr_selftest_failed", ccr_hash=key)

        record = version_for(envelope, ccr_hash=key, family=family)

        def finish(outcome: AdmissionOutcome, presented_exact: bool) -> AdmissionOutcome:
            # A partial observation never replaces a complete base.
            if record.complete or base is None or not base.complete:
                self.versions.put(
                    request.scope, replace(record, presented_exact=presented_exact)
                )
            return outcome

        # Progressive disclosure: a file re-read after the model was shown a
        # lossy view of it gets the exact content (the reread is the signal).
        if family is ToolFamily.READ and base is not None and not base.presented_exact:
            return finish(out("progressive_disclosure", ccr_hash=key), True)

        # (5b) candidates.
        from headroom.transforms.safety_gate import GateCandidate, Provenance, SafetyGate

        def prov(text: str, label: str, content_type: str, reversible: bool) -> Provenance:
            tok = request.tokenizer
            return Provenance(
                content_type=content_type,
                strategy=f"tool_admission:{label}",
                reversible=reversible,
                retrieval_available=request.retrieval_available,
                ccr_payloads={key: original},
                source_truncated=envelope.is_partial,
                original_tokens=tok.count_text(original) if tok is not None else None,
                candidate_tokens=tok.count_text(text) if tok is not None else None,
            )

        candidates: list[GateCandidate] = []
        delta: Delta | None = None
        if base is not None:
            base_text = None
            if base.ccr_hash:
                try:
                    base_text = store.peek_original(base.ccr_hash)
                except Exception:  # noqa: BLE001 - base text gone: identity-only delta
                    base_text = None
            label = _label(envelope, ctx, family)
            delta = compute_delta(
                family, envelope, base, base_text=base_text, ccr_hash=key, label=label
            )
            if delta is not None:
                # Facts the family must always show (failing ids, summaries,
                # exit status, user-named entities) stay visible in a delta.
                pinned = pin_repair(
                    original,
                    delta.text,
                    ctx.entities,
                    facts=family in _FACT_FAMILIES,
                )
                delta = replace(delta, text=pinned)
                candidates.append(
                    GateCandidate(
                        "aggressive",
                        delta.text,
                        prov(delta.text, delta.label, "plain_text", False),
                        label=delta.label,
                    )
                )
        view: FamilyView | None = interceptor.view(envelope, ctx)
        footer = _footer(envelope, family, key)
        if view is not None:
            tier = "lossless" if view.reversible else ("conservative" if delta else "aggressive")
            body = f"{view.text}\n{footer}"
            candidates.append(
                GateCandidate(
                    tier,
                    body,
                    prov(body, view.label, view.content_type, view.reversible),
                    label=view.label,
                )
            )
            if view.conservative and not view.reversible:
                body_c = f"{view.conservative}\n{footer}"
                candidates.append(
                    GateCandidate(
                        "conservative",
                        body_c,
                        prov(body_c, view.label + "_conservative", view.content_type, False),
                        label=view.label + "_conservative",
                    )
                )
        if not candidates:
            return finish(out("no_view", ccr_hash=key), True)

        # (6) SafetyGate ladder: first admitted rung wins, else the original.
        selection = SafetyGate.select(
            original, candidates, task=request.task, policy=request.policy, model=request.model
        )
        chosen = selection.candidate
        reasons = tuple(r for d in selection.decisions for r in d.reasons)
        if chosen is None:
            return finish(out("gate_rejected", reasons=reasons, ccr_hash=key), True)
        text = chosen.text
        used_delta = delta if delta is not None and text == delta.text else None
        if used_delta is not None:
            exact = bool(base is not None and base.presented_exact)
        else:
            exact = bool(view is not None and view.reversible and text.startswith(view.text))
        # (7) replace: only an admitted candidate leaves this function.
        return finish(
            out(
                "admitted",
                text=text,
                label=chosen.label,
                reasons=reasons,
                ccr_hash=key,
                admitted_bytes=len(text.encode("utf-8", "replace")),
                tier=selection.tier,
                delta=used_delta,
            ),
            exact,
        )

    @staticmethod
    def _is_edit_target(envelope: ToolResultEnvelope, ctx: ViewContext) -> bool:
        if not ctx.working_set:
            return False
        args = ctx.tool_input if isinstance(ctx.tool_input, Mapping) else {}
        from headroom.agent.tool_admission.families import path_of

        raw = path_of(args)
        if not raw:
            return False
        rel = workspace_relative(raw, root=ctx.root, cwd=ctx.cwd)
        return rel in ctx.working_set or raw in ctx.working_set


def _label(envelope: ToolResultEnvelope, ctx: ViewContext, family: ToolFamily) -> str:
    name = envelope.tool_name or family.value
    args = ctx.tool_input if isinstance(ctx.tool_input, Mapping) else {}
    from headroom.agent.tool_admission.families import command_of, path_of

    target = path_of(args) or command_of(name, args) or ""
    target = " ".join(str(target).split())[:80]
    return f"{name} {target}".strip()


_DISPATCHER: ToolAdmissionDispatcher | None = None
_DISPATCHER_LOCK = threading.Lock()


def get_dispatcher() -> ToolAdmissionDispatcher:
    """The process-wide dispatcher (host hook endpoint and proxy transform share it)."""
    global _DISPATCHER
    if _DISPATCHER is None:
        with _DISPATCHER_LOCK:
            if _DISPATCHER is None:
                _DISPATCHER = ToolAdmissionDispatcher()
    return _DISPATCHER


def reset_dispatcher() -> None:
    global _DISPATCHER
    with _DISPATCHER_LOCK:
        if _DISPATCHER is not None:
            _DISPATCHER.versions.unbind()
        _DISPATCHER = None


__all__ = [
    "ADMITTED_TAG",
    "AdmissionOutcome",
    "AdmissionRequest",
    "ToolAdmissionDispatcher",
    "get_dispatcher",
    "reset_dispatcher",
]
