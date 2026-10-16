"""Proxy live-zone fallback for tool-result admission (unified-21 #4/#14).

For clients without a pre-context host hook (and for results a host hook did
not rewrite), tool results are admitted when they appear in the mutable live
zone of a request. This never claims the bytes stayed out of the client's own
history: the host keeps the original; only the forwarded request is smaller.

Runs as the first pipeline transform (before the legacy interceptors, the
cache aligner and the content router) for Anthropic and OpenAI-chat message
lists. Frozen prefix messages are never touched; every tool result in the
mutable tail goes through the one :class:`ToolAdmissionDispatcher`.
"""

from __future__ import annotations

import copy
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from headroom.agent.budget import BudgetInputs
from headroom.agent.interfaces import content_version_hash
from headroom.agent.tool_admission.dispatcher import (
    ADMITTED_TAG,
    AdmissionOutcome,
    AdmissionRequest,
    PreparedAdmission,
    ToolAdmissionDispatcher,
    get_dispatcher,
    unit_key,
)
from headroom.agent.tool_admission.families import (
    EDIT_TOOLS,
    build_envelope,
    parse_args,
    path_of,
    workspace_relative,
)
from headroom.agent.tool_admission.views import ViewContext
from headroom.config import TransformResult
from headroom.log_safety import describe_exception
from headroom.tokenizer import Tokenizer
from headroom.transforms.base import Transform

log = logging.getLogger(__name__)

_APPLYABLE = re.compile(
    r"\b(?:git apply|git am|format-patch|apply (?:the|this) (?:diff|patch)|patch file|"
    r"\.patch\b|\.diff\b|applyable|unified diff)\b",
    re.IGNORECASE,
)


def is_admitted(text: str) -> bool:
    return ADMITTED_TAG in text and "Retrieve more: hash=" in text


def _tool_calls(messages: list[dict[str, Any]]) -> dict[str, tuple[str, Mapping[str, Any]]]:
    index: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    bid = block.get("id")
                    if isinstance(bid, str):
                        index[bid] = (str(block.get("name") or ""), parse_args(block.get("input")))
        calls = msg.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if not isinstance(call, dict) or not isinstance(call.get("id"), str):
                    continue
                fn = call.get("function") or {}
                index[call["id"]] = (str(fn.get("name") or ""), parse_args(fn.get("arguments")))
    return index


def _visible_results(messages: list[dict[str, Any]]) -> dict[str, str]:
    """``call id -> version hash`` of every tool result visible in this request.

    A delta base is usable only when the exact base result is itself still
    visible here (same call id *and* same bytes): ids are not resource identity
    and are not guaranteed unique across conversations.
    """
    out: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "tool":
            text = _block_text(msg.get("content"))
            if text is not None and isinstance(msg.get("tool_call_id"), str):
                out[msg["tool_call_id"]] = content_version_hash(text)
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                text = _block_text(block.get("content"))
                cid = block.get("tool_use_id")
                if text is not None and isinstance(cid, str):
                    out[cid] = content_version_hash(text)
    return out


def working_set(
    calls: Mapping[str, tuple[str, Mapping[str, Any]]], *, root: str | None, cwd: str | None
) -> frozenset[str]:
    """Paths the agent edited/wrote in this conversation (kept exact on Read)."""
    out: set[str] = set()
    for name, args in calls.values():
        if name.lower() in EDIT_TOOLS:
            raw = path_of(args)
            if raw:
                out.add(workspace_relative(raw, root=root, cwd=cwd))
                out.add(raw)
    return frozenset(out)


def _block_text(inner: Any) -> str | None:
    if isinstance(inner, str):
        return inner
    if isinstance(inner, list):
        if any(not isinstance(b, dict) or b.get("type") != "text" for b in inner):
            return None  # images / documents: never collapsed
        return "\n".join(str(b.get("text") or "") for b in inner)
    return None


def _set_block_text(block: dict[str, Any], text: str) -> None:
    if isinstance(block.get("content"), list):
        block["content"] = [{"type": "text", "text": text}]
    else:
        block["content"] = text


class ToolAdmissionTransform(Transform):
    """Pipeline adapter over :class:`ToolAdmissionDispatcher` (frozen prefix untouched)."""

    name = "tool_admission"

    def __init__(
        self,
        *,
        retrieval_available: bool = True,
        dispatcher: ToolAdmissionDispatcher | None = None,
        arbiter_enabled: bool = False,
        feedback_enabled: bool = False,
        speculation_enabled: bool = False,
    ) -> None:
        self.retrieval_available = retrieval_available
        self._dispatcher = dispatcher
        #: Unified-21 Phase 5: global budget + multi-candidate arbiter (#8/#6),
        #: retention feedback (#12) and speculative preparation (#15).
        self.arbiter_enabled = arbiter_enabled
        self.feedback_enabled = feedback_enabled
        self.speculation_enabled = speculation_enabled

    @property
    def dispatcher(self) -> ToolAdmissionDispatcher:
        return self._dispatcher or get_dispatcher()

    def apply(
        self, messages: list[dict[str, Any]], tokenizer: Tokenizer, **kwargs: Any
    ) -> TransformResult:
        tokens_before = tokenizer.count_messages(messages)
        try:
            new_messages, applied = self._apply(messages, tokenizer, kwargs)
        except Exception as exc:  # noqa: BLE001 - admission must never fail a request
            log.warning("tool admission transform failed: %s", describe_exception(exc))
            new_messages, applied = messages, []
        tokens_after = (
            tokenizer.count_messages(new_messages) if new_messages is not messages else tokens_before
        )
        return TransformResult(
            messages=new_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=applied,
        )

    def _request_facts(self, kwargs: Mapping[str, Any]) -> tuple[str, str | None, Any, str]:
        from headroom.agent.identity import resolve_identity
        from headroom.proxy.project_context import (
            get_current_project,
            get_registered_cwd,
            get_registered_session,
        )

        cwd = get_registered_cwd()
        identity = resolve_identity(
            session_token=get_registered_session(),
            registered_cwd=cwd,
            attribution=get_current_project(),
        )
        relevance = kwargs.get("relevance_query")
        query = getattr(relevance, "text", "") or str(kwargs.get("context") or "")
        return f"{identity.workspace_id}|{identity.session_id}", identity.root or cwd, relevance, query

    def _apply(
        self, messages: list[dict[str, Any]], tokenizer: Tokenizer, kwargs: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        frozen = max(int(kwargs.get("frozen_message_count") or 0), 0)
        if frozen >= len(messages):
            return messages, []
        if not any(self._has_tool_result(m) for m in messages[frozen:]):
            return messages, []
        scope, root, relevance, query = self._request_facts(kwargs)
        calls = _tool_calls(messages)
        visible = _visible_results(messages)
        ws = working_set(calls, root=root, cwd=root)
        from headroom.transforms.safety_gate import task_payload

        task = task_payload(relevance, fallback_text=query)
        entities = tuple(task.get("exact_entities") or ())
        policy = kwargs.get("compression_policy")
        if policy is None:
            from headroom.transforms.content_router import default_gate_policy

            policy = default_gate_policy()
        from headroom.cache.compression_store import get_compression_store

        store = get_compression_store()
        dispatcher = self.dispatcher
        model = str(kwargs.get("model") or "")
        budget_inputs = self._budget_inputs(messages, kwargs, model)
        feedback = self._feedback()
        if feedback is not None:
            feedback.tick(scope)
            self._correction_signal(feedback, scope, messages)
        arbiter = None
        limits = None
        advisor = advisor_budget = None
        if self.arbiter_enabled:
            from headroom.agent.arbiter import RequestLimits, get_arbiter

            arbiter = get_arbiter()
            limits = RequestLimits()
            advisor = _advisor()
            # One JevK5 allowance per proxy request, shared with every other
            # consumer (catalog, effort routing) that passes the same id.
            advisor_budget = (
                advisor.request_budget(str(kwargs.get("request_id") or "") or None)
                if advisor is not None
                else None
            )
        speculation = None
        if self.speculation_enabled:
            from headroom.agent.speculation import RequestSpeculation

            speculation = RequestSpeculation()
        base = AdmissionRequest(
            scope=scope,
            context=ViewContext(query=query, entities=entities, working_set=ws, root=root, cwd=root),
            task=task,
            policy=policy,
            store=store,
            tokenizer=tokenizer,
            model=model,
            retrieval_available=self.retrieval_available,
            source="proxy",
            base_visible=lambda record, _v=visible: _v.get(record.call_id) == record.version_hash,
            arbiter=arbiter,
            feedback=feedback,
            speculation=speculation,
            limits=limits,
            advisor=advisor,
            advisor_budget=advisor_budget,
            cache_mode=budget_inputs.cache_mode,
            cache_ttl_seconds=budget_inputs.cache_ttl_seconds,
            arm=_arm(scope) if arbiter is not None and feedback is not None else "treatment",
        )
        applyable = bool(_APPLYABLE.search(query or ""))
        replacements: dict[tuple[int, int | None], tuple[str, str]] = {}
        try:
            units = self._collect_units(messages, frozen, calls, base, applyable)
            replacements = self._admit_units(
                units, dispatcher, tokenizer, messages, frozen, budget_inputs, feedback
            )
        finally:
            if speculation is not None:
                self._prepare_indexes(speculation, store, _admitted_hashes(replacements))
        if not replacements:
            return messages, []
        out = list(messages[:frozen])
        applied: list[str] = []
        for idx in range(frozen, len(messages)):
            msg = messages[idx]
            mine = {b: r for (m, b), r in replacements.items() if m == idx}
            if not mine:
                out.append(msg)
                continue
            if msg.get("role") == "tool":
                text, label = mine[None]
                clone = dict(msg)
                clone["content"] = (
                    text if isinstance(msg.get("content"), str) else [{"type": "text", "text": text}]
                )
                out.append(clone)
                applied.append(label)
                continue
            clone = copy.deepcopy(msg)
            for block_idx in sorted(b for b in mine if b is not None):
                text, label = mine[block_idx]
                _set_block_text(clone["content"][block_idx], text)
                applied.append(label)
            out.append(clone)
        return out, applied

    # -- Phase 5 request plumbing ---------------------------------------------------

    @staticmethod
    def _budget_inputs(
        messages: list[dict[str, Any]], kwargs: Mapping[str, Any], model: str
    ) -> BudgetInputs:
        given = kwargs.get("live_zone_budget")
        if isinstance(given, BudgetInputs):
            return given
        from headroom.models.registry import ModelRegistry

        ttl = kwargs.get("cache_ttl_seconds")
        return BudgetInputs(
            model=model,
            model_limit=ModelRegistry.get_context_limit(model or "", default=128_000),
            effective_limit=int(kwargs.get("model_limit") or 0),
            cache_mode=cache_mode_of(messages),
            cache_ttl_seconds=int(ttl) if isinstance(ttl, int | float) else None,
        )

    def _feedback(self) -> Any:
        if not self.feedback_enabled:
            return None
        from headroom.agent.feedback import get_feedback

        return get_feedback()

    @staticmethod
    def _correction_signal(feedback: Any, scope: str, messages: list[dict[str, Any]]) -> None:
        text = _latest_user_text(messages)
        if not text:
            return
        from headroom.agent.task_state import classify_user_turn

        if classify_user_turn(text) == "correction":
            feedback.signal_correction(scope, content_version_hash(text))

    def _collect_units(
        self,
        messages: list[dict[str, Any]],
        frozen: int,
        calls: Mapping[str, tuple[str, Mapping[str, Any]]],
        base: AdmissionRequest,
        applyable: bool,
    ) -> list[_Unit]:
        units: list[_Unit] = []
        for idx in range(frozen, len(messages)):
            msg = messages[idx]
            if msg.get("role") == "tool":
                text = _block_text(msg.get("content"))
                call_id = msg.get("tool_call_id")
                if text is not None and isinstance(call_id, str):
                    unit = self._unit(idx, None, text, call_id, calls, base, False, applyable)
                    if unit is not None:
                        units.append(unit)
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for b_idx, block in enumerate(content):
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                text = _block_text(block.get("content"))
                call_id = block.get("tool_use_id")
                if text is None or not isinstance(call_id, str):
                    continue
                unit = self._unit(
                    idx, b_idx, text, call_id, calls, base, bool(block.get("is_error")), applyable
                )
                if unit is not None:
                    units.append(unit)
        return units

    @staticmethod
    def _unit(
        msg_idx: int,
        block_idx: int | None,
        text: str,
        call_id: str,
        calls: Mapping[str, tuple[str, Mapping[str, Any]]],
        base: AdmissionRequest,
        is_error: bool,
        applyable: bool,
    ) -> _Unit | None:
        if not text or is_admitted(text):
            return None
        name, args = calls.get(call_id, ("", {}))
        if not name:
            return None  # orphaned result: no tool identity, never guessed
        envelope = build_envelope(
            tool_name=name,
            tool_input=args,
            content=text,
            call_id=call_id,
            session=base.scope,
            root=base.context.root,
            cwd=base.context.cwd,
            is_error=is_error,
        )
        ctx = ViewContext(
            query=base.context.query,
            entities=base.context.entities,
            working_set=base.context.working_set,
            root=base.context.root,
            cwd=base.context.cwd,
            tool_input=dict(args),
            applyable_patch=applyable,
        )
        request = AdmissionRequest(**{**base.__dict__, "context": ctx})
        return _Unit(msg_idx, block_idx, envelope, request)

    def _admit_units(
        self,
        units: list[_Unit],
        dispatcher: ToolAdmissionDispatcher,
        tokenizer: Tokenizer,
        messages: list[dict[str, Any]],
        frozen: int,
        budget_inputs: BudgetInputs,
        feedback: Any,
    ) -> dict[tuple[int, int | None], tuple[str, str]]:
        """Prepare every unit, allocate the live-zone budget, then decide.

        Memoized units are fixed (their rendered bytes are committed in the
        client's history and must not change). A unit whose resource already
        occurred earlier in this request is decided after it, in order, so a
        same-request reread still sees its base.
        """
        replacements: dict[tuple[int, int | None], tuple[str, str]] = {}
        fixed: list[tuple[_Unit, AdmissionOutcome]] = []
        fresh: list[_Unit] = []
        deferred: list[_Unit] = []
        seen_resources: set[str] = set()
        for unit in units:
            memo = dispatcher.memo_lookup(unit.envelope, unit.request)
            if memo is not None:
                fixed.append((unit, memo))
                continue
            rid = str(unit.envelope.resource_id or "")
            if rid and rid in seen_resources:
                deferred.append(unit)
            else:
                fresh.append(unit)
            if rid:
                seen_resources.add(rid)
        if feedback is not None:
            for unit in fresh + deferred:
                _signal_new_result(feedback, unit)
        # Speculation (#15): view previews for the next units run on this
        # request's single lane while the current unit is prepared inline.
        for unit in fresh[1:]:
            dispatcher.speculate_view(unit.envelope, unit.request)
        prepared = [dispatcher.prepare(u.envelope, u.request) for u in fresh]
        allocation = None
        if self.arbiter_enabled:
            allocation = self._allocate(
                prepared, fixed, dispatcher, tokenizer, messages, frozen, budget_inputs
            )
        for unit, prep in zip(fresh, prepared):
            kw: dict[str, Any] = {}
            if allocation is not None:
                key = unit_key(unit.envelope)
                kw = {
                    "allowance": allocation.allowance(key),
                    "pressure": allocation.pressure.get(key, 0.0),
                    "insufficient": allocation.insufficient,
                }
            outcome = dispatcher.decide(prep, **kw)
            if outcome.text is not None:
                replacements[(unit.msg_idx, unit.block_idx)] = (
                    outcome.text,
                    f"tool_admission:{outcome.family}:{outcome.label}",
                )
        for unit in deferred:
            outcome = dispatcher.admit(unit.envelope, unit.request)
            if outcome.text is not None:
                replacements[(unit.msg_idx, unit.block_idx)] = (
                    outcome.text,
                    f"tool_admission:{outcome.family}:{outcome.label}",
                )
        for unit, memo in fixed:
            if memo.text is not None:
                replacements[(unit.msg_idx, unit.block_idx)] = (
                    memo.text,
                    f"tool_admission:{memo.family}:{memo.label}",
                )
        return replacements

    def _allocate(
        self,
        prepared: list[PreparedAdmission],
        fixed: list[tuple[_Unit, AdmissionOutcome]],
        dispatcher: ToolAdmissionDispatcher,
        tokenizer: Tokenizer,
        messages: list[dict[str, Any]],
        frozen: int,
        budget_inputs: BudgetInputs,
    ) -> Any:
        from headroom.agent.arbiter import get_arbiter
        from headroom.agent.budget import BudgetUnit, allocate, compute_budget

        frozen_tokens = tokenizer.count_messages(messages[:frozen]) if frozen else 0
        budget = compute_budget(budget_inputs, frozen_tokens=frozen_tokens)
        units: list[BudgetUnit] = []
        unit_tokens = 0
        for prep in prepared:
            bu = dispatcher.budget_unit(prep)
            if bu is not None:
                units.append(bu)
                unit_tokens += bu.original_tokens
            else:
                tokens = tokenizer.count_text(prep.envelope.content)
                units.append(
                    BudgetUnit(
                        key=unit_key(prep.envelope),
                        source=prep.family.value,
                        original_tokens=tokens,
                        min_tokens=tokens,
                        fixed=True,
                    )
                )
                unit_tokens += tokens
        for unit, memo in fixed:
            text = memo.text if memo.text is not None else unit.envelope.content
            tokens = tokenizer.count_text(text)
            units.append(
                BudgetUnit(unit_key(unit.envelope), memo.family, tokens, tokens, fixed=True)
            )
            unit_tokens += tokenizer.count_text(unit.envelope.content)
        live_tokens = tokenizer.count_messages(messages[frozen:])
        other = max(0, live_tokens - unit_tokens)
        allocation = allocate(units, budget, other_live_tokens=other)
        get_arbiter().note_allocation(allocation)
        return allocation

    @staticmethod
    def _prepare_indexes(speculation: Any, store: Any, keys: list[str]) -> None:
        """While the request awaits the model, build CCR span indexes for the
        originals stored this turn (derived data; nothing waits for it)."""
        from headroom.ccr.indexed import get_indexed_ccr

        index = get_indexed_ccr()
        for key in keys[:8]:
            speculation.submit(("index", key), lambda k=key: index.ensure(store, k))
        speculation.detach()

    @staticmethod
    def _has_tool_result(msg: Mapping[str, Any]) -> bool:
        if msg.get("role") == "tool":
            return True
        content = msg.get("content")
        return isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        )


@dataclass
class _Unit:
    msg_idx: int
    block_idx: int | None
    envelope: Any
    request: AdmissionRequest


_HASH_IN_TEXT = re.compile(r"hash=([a-f0-9]{12,24})\b")
_TEST_FAILED = re.compile(r"(?m)^(?:FAILED |ERROR |E\s)|\b\d+ failed\b")


def _admitted_hashes(reps: Mapping[Any, tuple[str, str]]) -> list[str]:
    """CCR hashes in this request's admitted replacements (for index prep)."""
    keys: list[str] = []
    for text, _label in reps.values():
        for match in _HASH_IN_TEXT.findall(text):
            if match not in keys:
                keys.append(match)
    return keys


def cache_mode_of(messages: list[dict[str, Any]]) -> str:
    """``write`` when the client set any prompt-cache breakpoint in messages."""
    for msg in messages:
        if isinstance(msg.get("cache_control"), dict):
            return "write"
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("cache_control"), dict):
                    return "write"
    return "none"


def _latest_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [
                str(b.get("text") or "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            if texts:
                return "\n".join(texts)
            return ""  # a tool-result turn: no new user text
    return ""


def _signal_new_result(feedback: Any, unit: _Unit) -> None:
    """Retention feedback from a newly observed result (#12)."""
    env = unit.envelope
    scope = unit.request.scope
    family = env.tool_family
    if (env.tool_name or "").lower() in EDIT_TOOLS and env.is_error:
        raw = path_of(unit.request.context.tool_input or {})
        if raw:
            feedback.signal_repair(
                scope,
                workspace_relative(raw, root=unit.request.context.root, cwd=unit.request.context.cwd),
            )
        return
    if family == "test" and (env.is_error or env.exit_code not in (None, 0) or _TEST_FAILED.search(env.content)):
        feedback.signal_test_failure(scope)


def _arm(scope: str) -> str:
    from headroom.agent.feedback import transform_arm

    return transform_arm(scope)


def _advisor() -> Any:
    from headroom.agent.hook import _active_runtime

    runtime = _active_runtime()
    return getattr(runtime, "advisor", None) if runtime is not None else None
