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
import re
import threading
import time
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
    arbiter: dict[str, Any] | None = None

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
            "arbiter": self.arbiter,
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
    # -- Phase 5 (#6/#8/#12/#15); all optional, ``None`` keeps the Phase-4 ladder.
    arbiter: Any = None  # headroom.agent.arbiter.TransformArbiter
    feedback: Any = None  # headroom.agent.feedback.RetentionFeedback
    speculation: Any = None  # headroom.agent.speculation.RequestSpeculation
    limits: Any = None  # headroom.agent.arbiter.RequestLimits (per request)
    advisor: Any = None
    advisor_budget: Any = None
    cache_mode: str = "none"
    cache_ttl_seconds: int | None = None


@dataclass
class PreparedAdmission:
    """One unit after steps 1-5 and the hard filters, before selection."""

    envelope: ToolResultEnvelope
    family: ToolFamily
    request: AdmissionRequest
    outcome: AdmissionOutcome | None = None  # decided early (forward original)
    size: int = 0
    key: str | None = None
    base: ResourceVersion | None = None
    record: ResourceVersion | None = None
    delta: Delta | None = None
    view: FamilyView | None = None
    view_from_speculation: bool = False
    gate_candidates: list[Any] = field(default_factory=list)
    evaluation: Any = None  # headroom.agent.arbiter.Evaluation
    arbiter_ctx: Any = None
    exact_on_finish: bool = False


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
        memo = self.memo_lookup(envelope, request)
        if memo is not None:
            return memo
        return self.decide(self.prepare(envelope, request))

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

    # -- two-phase API (proxy transform: prepare all units, allocate, decide) --

    def memo_lookup(
        self, envelope: ToolResultEnvelope, request: AdmissionRequest
    ) -> AdmissionOutcome | None:
        """The outcome already rendered for this exact (scope, call, content)."""
        if not envelope.tool_call_id:
            return None
        key = self._memo_key(envelope, request)
        with self._lock:
            memo = self._memo.get(key)
            if memo is not None:
                self._memo.move_to_end(key)
            return memo

    def prepare(self, envelope: ToolResultEnvelope, request: AdmissionRequest) -> PreparedAdmission:
        """Steps 1-5 plus candidate generation and hard filters; never raises."""
        family = _family_of(envelope)
        try:
            return self._prepare(envelope, family, request)
        except Exception as exc:  # noqa: BLE001 - a local failure passes the original through
            log.warning("tool admission failed: %s", describe_exception(exc))
            outcome = AdmissionOutcome(None, "error", family.value, reasons=(type(exc).__name__,))
            return PreparedAdmission(envelope, family, request, outcome=outcome)

    def decide(
        self,
        prepared: PreparedAdmission,
        *,
        allowance: int | None = None,
        pressure: float = 0.0,
        insufficient: bool = False,
    ) -> AdmissionOutcome:
        """Select (arbiter or ladder), record the version, count and memoize."""
        envelope, request = prepared.envelope, prepared.request
        if prepared.outcome is not None:
            outcome = prepared.outcome
            if prepared.exact_on_finish:
                self._finish(prepared, True)
        else:
            try:
                outcome = self._decide(prepared, allowance, pressure, insufficient)
            except Exception as exc:  # noqa: BLE001
                log.warning("tool admission failed: %s", describe_exception(exc))
                outcome = AdmissionOutcome(
                    None, "error", prepared.family.value, reasons=(type(exc).__name__,)
                )
        self._count(outcome, request.source, envelope)
        if envelope.tool_call_id and outcome.stage not in ("error",):
            with self._lock:
                self._memo[self._memo_key(envelope, request)] = outcome
                while len(self._memo) > MEMO_LIMIT:
                    self._memo.popitem(last=False)
        return outcome

    def speculate_view(self, envelope: ToolResultEnvelope, request: AdmissionRequest) -> bool:
        """Queue the family-view preview of ``envelope`` on the request's lane.

        Pure work only (the interceptor view); the authoritative path takes it
        if it is ready and recomputes it inline otherwise, so the outcome is
        identical either way.
        """
        spec = request.speculation
        if spec is None:
            return False
        family = _family_of(envelope)
        interceptor = self.interceptors.get(family)
        if interceptor is None or family is ToolFamily.OTHER:
            return False
        original = envelope.content
        if not original or len(original) < _FAMILY_MIN_CHARS.get(family, MIN_VIEW_CHARS):
            return False
        if ADMITTED_TAG in original and "hash=" in original:
            return False
        ctx = request.context
        if family is ToolFamily.READ and self._is_edit_target(envelope, ctx):
            return False
        return spec.submit(_view_key(envelope, family), lambda: interceptor.view(envelope, ctx))

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _memo_key(
        envelope: ToolResultEnvelope, request: AdmissionRequest
    ) -> tuple[str, str, str, str]:
        return (
            request.scope,
            envelope.tool_call_id,
            envelope.version_hash,
            _family_of(envelope).value,
        )

    def _prepare(
        self, envelope: ToolResultEnvelope, family: ToolFamily, request: AdmissionRequest
    ) -> PreparedAdmission:
        original = envelope.content
        size = len(original.encode("utf-8", "replace"))
        prepared = PreparedAdmission(envelope, family, request, size=size)

        def early(stage: str, **kw: Any) -> PreparedAdmission:
            prepared.outcome = AdmissionOutcome(
                kw.pop("text", None), stage, family.value, original_bytes=size, **kw
            )
            return prepared

        interceptor = self.interceptors.get(family)
        if family is ToolFamily.OTHER or interceptor is None:
            return early("unsupported_family")
        if not original or len(original) < _FAMILY_MIN_CHARS.get(family, MIN_VIEW_CHARS):
            return early("small")
        if ADMITTED_TAG in original and "hash=" in original:
            return early("already_admitted")
        ctx = request.context
        if family is ToolFamily.READ and self._is_edit_target(envelope, ctx):
            return early("edit_target_exact")
        store = request.store
        if store is None:
            return early("no_ccr_store")

        # (5a) delta base lookup happens before this version is recorded.
        base = self.versions.get(request.scope, str(envelope.resource_id or ""))
        if base is not None and base.call_id == envelope.tool_call_id:
            base = None  # the record is this very call
        if base is not None and request.base_visible is not None and not request.base_visible(base):
            base = None
        if base is not None and not base.presented_exact and request.feedback is not None:
            # Retention feedback (#12): re-reading / re-searching a resource the
            # model was shown lossy is a negative outcome for that decision.
            request.feedback.signal_reread(request.scope, str(envelope.resource_id or ""))

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
            return early("ccr_store_failed", reasons=(type(exc).__name__,))
        if not store.verify_exact(key, original):
            return early("ccr_selftest_failed", ccr_hash=key)
        prepared.key = key
        prepared.base = base
        prepared.record = version_for(envelope, ccr_hash=key, family=family)

        # Progressive disclosure: a file re-read after the model was shown a
        # lossy view of it gets the exact content (the reread is the signal).
        if family is ToolFamily.READ and base is not None and not base.presented_exact:
            prepared.exact_on_finish = True
            return early("progressive_disclosure", ccr_hash=key)

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
        prepared.delta = delta

        view: FamilyView | None = None
        vkey = _view_key(envelope, family)
        started = time.perf_counter()
        hit = False
        if request.speculation is not None:
            hit, value = request.speculation.take(vkey)
            if hit:
                view = value
        if not hit and (request.limits is None or request.limits.take_view()):
            view = interceptor.view(envelope, ctx)
        view_ms = (time.perf_counter() - started) * 1000.0
        prepared.view = view
        prepared.view_from_speculation = hit
        footer = _footer(envelope, family, key)

        if request.arbiter is None:
            # Phase-4 ladder: aggressive -> conservative -> lossless rungs.
            candidates: list[GateCandidate] = []
            if delta is not None:
                candidates.append(
                    GateCandidate(
                        "aggressive",
                        delta.text,
                        prov(delta.text, delta.label, "plain_text", False),
                        label=delta.label,
                    )
                )
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
                prepared.exact_on_finish = True
                return early("no_view", ccr_hash=key)
            prepared.gate_candidates = candidates
            return prepared

        # Unified-21 #6: every applicable candidate kind, for the arbiter.
        from headroom.agent.arbiter import ArbiterContext, Candidate

        cands: list[Candidate] = []
        if delta is not None:
            cands.append(
                Candidate(
                    "delta", delta.text, "aggressive", delta.label,
                    prov(delta.text, delta.label, "plain_text", False),
                )
            )
        if view is not None:
            body = f"{view.text}\n{footer}"
            if view.reversible:
                cands.append(
                    Candidate(
                        "lossless", body, "lossless", view.label,
                        prov(body, view.label, view.content_type, True), True, view_ms,
                    )
                )
            else:
                cands.append(
                    Candidate(
                        "type_specific", body, "conservative" if delta else "aggressive",
                        view.label, prov(body, view.label, view.content_type, False),
                        False, view_ms,
                    )
                )
                if view.conservative:
                    body_c = f"{view.conservative}\n{footer}"
                    label_c = view.label + "_conservative"
                    cands.append(
                        Candidate(
                            "structural", body_c, "conservative", label_c,
                            prov(body_c, label_c, view.content_type, False),
                        )
                    )
        terms = _query_terms(request)
        if terms and family in _QUERY_FIT_FAMILIES:
            fit = SafetyGate.fit(
                original,
                view.text if view is not None else None,
                task=request.task,
                policy=request.policy,
                content_type="plain_text",
                model=request.model,
            )
            if fit and fit != original:
                body_q = f"{fit}\n{footer}"
                cands.append(
                    Candidate(
                        "query_aware", body_q, "conservative", "query_fit",
                        prov(body_q, "query_fit", "plain_text", False),
                    )
                )
        if family is ToolFamily.JSON and not (view is not None and view.reversible):
            minified = SafetyGate.minify_json(original)
            if minified and len(minified) < len(original):
                cands.append(
                    Candidate(
                        "lossless", minified, "lossless", "json_minify",
                        prov(minified, "json_minify", "json", True), True,
                    )
                )
        if request.retrieval_available:
            stub = SafetyGate.ccr_stub(original, key)
            cands.append(
                Candidate(
                    "indexed_externalization", stub, "ccr_exact", "ccr_exact",
                    prov(stub, "ccr_exact", "plain_text", False),
                )
            )
        actx = ArbiterContext(
            original=original,
            task=request.task,
            terms=terms,
            policy=request.policy,
            model=request.model,
            tokenizer=request.tokenizer,
            cache_mode=request.cache_mode,
            cache_ttl_seconds=request.cache_ttl_seconds,
            feedback=request.feedback,
            feedback_key=_feedback_key(envelope, family, request),
            ambiguous=not terms,
            advisor=request.advisor,
            advisor_budget=request.advisor_budget,
            limits=request.limits,
        )
        prepared.arbiter_ctx = actx
        prepared.evaluation = request.arbiter.evaluate(cands, actx)
        return prepared

    def budget_unit(self, prepared: PreparedAdmission) -> Any:
        """The allocator's view of one prepared unit (``None`` when already decided)."""
        from headroom.agent.budget import BudgetUnit

        if prepared.evaluation is None or prepared.outcome is not None:
            return None
        env = prepared.envelope
        ev = prepared.evaluation
        terms = prepared.arbiter_ctx.terms if prepared.arbiter_ctx is not None else ()
        present = [t for t in terms if t and t in env.content]
        delta = prepared.delta
        novelty = 1.0
        if delta is not None:
            total = delta.added + delta.removed + delta.changed + delta.unchanged
            novelty = 1.0 - (delta.unchanged / total if total else 0.0)
        anomaly = (
            1.0
            if env.is_error or env.exit_code not in (None, 0) or _FAILURE.search(env.content)
            else 0.0
        )
        return BudgetUnit(
            key=unit_key(env),
            source=prepared.family.value,
            original_tokens=ev.original_tokens,
            min_tokens=ev.min_tokens,
            relevance=(len(present) / len(terms)) if terms else 0.0,
            novelty=novelty,
            anomaly=anomaly,
            recoverability=1.0 if prepared.request.retrieval_available else 0.0,
        )

    def _decide(
        self,
        prepared: PreparedAdmission,
        allowance: int | None,
        pressure: float,
        insufficient: bool,
    ) -> AdmissionOutcome:
        from headroom.transforms.safety_gate import SafetyGate

        envelope, family, request = prepared.envelope, prepared.family, prepared.request
        key, base, delta, view = prepared.key, prepared.base, prepared.delta, prepared.view
        size = prepared.size
        original = envelope.content

        def out(stage: str, **kw: Any) -> AdmissionOutcome:
            return AdmissionOutcome(
                kw.pop("text", None), stage, family.value, original_bytes=size, **kw
            )

        if prepared.evaluation is None:
            # (6) SafetyGate ladder: first admitted rung wins, else the original.
            selection = SafetyGate.select(
                original,
                prepared.gate_candidates,
                task=request.task,
                policy=request.policy,
                model=request.model,
            )
            chosen = selection.candidate
            reasons = tuple(r for d in selection.decisions for r in d.reasons)
            if chosen is None:
                self._finish(prepared, True)
                return out("gate_rejected", reasons=reasons, ccr_hash=key)
            text = chosen.text
            used_delta = delta if delta is not None and text == delta.text else None
            if used_delta is not None:
                exact = bool(base is not None and base.presented_exact)
            else:
                exact = bool(view is not None and view.reversible and text.startswith(view.text))
            self._finish(prepared, exact)
            # (7) replace: only an admitted candidate leaves this function.
            return out(
                "admitted",
                text=text,
                label=chosen.label,
                reasons=reasons,
                ccr_hash=key,
                admitted_bytes=len(text.encode("utf-8", "replace")),
                tier=selection.tier,
                delta=used_delta,
            )

        assert request.arbiter is not None and prepared.arbiter_ctx is not None
        decision = request.arbiter.choose(
            prepared.evaluation,
            prepared.arbiter_ctx,
            allowance=allowance,
            pressure=pressure,
            insufficient=insufficient,
            family=family.value,
        )
        reasons = decision.rejections
        winner = decision.candidate
        if winner is None:
            self._finish(prepared, True)
            stage = "arbiter_original" if len(prepared.evaluation.survivors) > 1 else "gate_rejected"
            return out(stage, reasons=reasons, ccr_hash=key, arbiter=decision.to_dict())
        text = winner.text
        used_delta = delta if winner.kind == "delta" else None
        if used_delta is not None:
            exact = bool(base is not None and base.presented_exact)
        else:
            exact = winner.reversible
        self._finish(prepared, exact)
        fkey = prepared.arbiter_ctx.feedback_key
        if not exact and request.feedback is not None and fkey is not None:
            from headroom.agent.tool_admission.families import path_of

            args = (
                request.context.tool_input if isinstance(request.context.tool_input, Mapping) else {}
            )
            raw = path_of(args) if family is ToolFamily.READ else None
            request.feedback.register(
                request.scope,
                replace(fkey, strategy=winner.kind),
                ccr_hash=key or "",
                resource_id=str(envelope.resource_id or ""),
                path=(
                    workspace_relative(raw, root=request.context.root, cwd=request.context.cwd)
                    if raw
                    else ""
                ),
            )
        return out(
            "admitted",
            text=text,
            label=winner.label,
            reasons=reasons,
            ccr_hash=key,
            admitted_bytes=len(text.encode("utf-8", "replace")),
            tier=winner.tier,
            delta=used_delta,
            arbiter=decision.to_dict(),
        )

    def _finish(self, prepared: PreparedAdmission, presented_exact: bool) -> None:
        record, base = prepared.record, prepared.base
        if record is None:
            return
        # A partial observation never replaces a complete base.
        if record.complete or base is None or not base.complete:
            self.versions.put(
                prepared.request.scope, replace(record, presented_exact=presented_exact)
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


_QUERY_FIT_FAMILIES = frozenset(
    {ToolFamily.SEARCH, ToolFamily.SHELL, ToolFamily.TEST, ToolFamily.LISTING}
)
_FAILURE = re.compile(
    r"(?m)^(?:E\s|FAILED\b|ERROR\b|error(?:\[|:)|Traceback\b|panicked at\b|fatal:)"
)
_LANG_BY_EXT = {
    ".py": "python", ".rs": "rust", ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".go": "go", ".java": "java",
    ".rb": "ruby", ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".hpp": "cpp",
    ".cs": "csharp", ".kt": "kotlin", ".swift": "swift", ".php": "php",
    ".md": "markdown", ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
}


def _family_of(envelope: ToolResultEnvelope) -> ToolFamily:
    try:
        return ToolFamily(envelope.tool_family)
    except ValueError:
        return ToolFamily.OTHER


def unit_key(envelope: ToolResultEnvelope) -> str:
    """Stable allocator identity: resource + call (never a list position)."""
    return f"{envelope.resource_id or ''}|{envelope.tool_call_id}"


def _view_key(envelope: ToolResultEnvelope, family: ToolFamily) -> tuple[str, str, str]:
    return ("view", family.value, envelope.version_hash + "|" + envelope.tool_call_id)


def _query_terms(request: AdmissionRequest) -> tuple[str, ...]:
    task = request.task
    terms: list[str] = []
    if isinstance(task, Mapping):
        terms.extend(str(e) for e in task.get("exact_entities") or ())
        text = str(task.get("text") or "")
        terms.extend(re.findall(r"[A-Za-z_][A-Za-z0-9_./-]{3,}", text))
    terms.extend(str(e) for e in request.context.entities or ())
    return tuple(dict.fromkeys(t for t in terms if t))[:48]


def _feedback_key(
    envelope: ToolResultEnvelope, family: ToolFamily, request: AdmissionRequest
) -> Any:
    from headroom.agent.feedback import FeedbackKey, workspace_hash
    from headroom.agent.tool_admission.families import path_of

    task = request.task if isinstance(request.task, Mapping) else {}
    if task.get("is_code"):
        task_class = "code"
    elif task.get("active_error"):
        task_class = "error"
    else:
        task_class = str(task.get("tool_intent") or "general")[:24]
    args = request.context.tool_input if isinstance(request.context.tool_input, Mapping) else {}
    raw = path_of(args) or ""
    ext = raw[raw.rfind(".") :].lower() if "." in raw.rsplit("/", 1)[-1] else ""
    return FeedbackKey(
        family=family.value,
        content_type="error" if envelope.is_error else family.value,
        task_class=task_class,
        model=request.model,
        language=_LANG_BY_EXT.get(ext, ""),
        workspace=workspace_hash(envelope.workspace or ""),
    )


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
    "PreparedAdmission",
    "unit_key",
    "ToolAdmissionDispatcher",
    "get_dispatcher",
    "reset_dispatcher",
]
