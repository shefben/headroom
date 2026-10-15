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
from typing import Any

from headroom.agent.interfaces import content_version_hash
from headroom.agent.tool_admission.dispatcher import (
    ADMITTED_TAG,
    AdmissionRequest,
    ToolAdmissionDispatcher,
    get_dispatcher,
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
    ) -> None:
        self.retrieval_available = retrieval_available
        self._dispatcher = dispatcher

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
        applied: list[str] = []
        out = list(messages[:frozen])
        for msg in messages[frozen:]:
            replacement = self._admit_message(
                msg,
                calls=calls,
                dispatcher=dispatcher,
                base=AdmissionRequest(
                    scope=scope,
                    context=ViewContext(query=query, entities=entities, working_set=ws, root=root, cwd=root),
                    task=task,
                    policy=policy,
                    store=store,
                    tokenizer=tokenizer,
                    model=str(kwargs.get("model") or ""),
                    retrieval_available=self.retrieval_available,
                    source="proxy",
                    base_visible=lambda record, _v=visible: _v.get(record.call_id)
                    == record.version_hash,
                ),
                applyable=bool(_APPLYABLE.search(query or "")),
                applied=applied,
            )
            out.append(replacement)
        if not applied:
            return messages, []
        return out, applied

    @staticmethod
    def _has_tool_result(msg: Mapping[str, Any]) -> bool:
        if msg.get("role") == "tool":
            return True
        content = msg.get("content")
        return isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        )

    def _admit_one(
        self,
        text: str,
        call_id: str,
        calls: Mapping[str, tuple[str, Mapping[str, Any]]],
        dispatcher: ToolAdmissionDispatcher,
        base: AdmissionRequest,
        *,
        is_error: bool,
        applyable: bool,
    ) -> tuple[str | None, str]:
        if not text or is_admitted(text):
            return None, ""
        name, args = calls.get(call_id, ("", {}))
        if not name:
            return None, ""  # orphaned result: no tool identity, never guessed
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
        outcome = dispatcher.admit(envelope, request)
        if outcome.text is None:
            return None, ""
        return outcome.text, f"tool_admission:{outcome.family}:{outcome.label}"

    def _admit_message(
        self,
        msg: dict[str, Any],
        *,
        calls: Mapping[str, tuple[str, Mapping[str, Any]]],
        dispatcher: ToolAdmissionDispatcher,
        base: AdmissionRequest,
        applyable: bool,
        applied: list[str],
    ) -> dict[str, Any]:
        if msg.get("role") == "tool":
            text = _block_text(msg.get("content"))
            call_id = msg.get("tool_call_id")
            if text is None or not isinstance(call_id, str):
                return msg
            new, label = self._admit_one(
                text, call_id, calls, dispatcher, base, is_error=False, applyable=applyable
            )
            if new is None:
                return msg
            applied.append(label)
            clone = dict(msg)
            clone["content"] = new if isinstance(msg.get("content"), str) else [
                {"type": "text", "text": new}
            ]
            return clone
        content = msg.get("content")
        if not isinstance(content, list):
            return msg
        clone: dict[str, Any] | None = None
        for idx, block in enumerate(content):
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            text = _block_text(block.get("content"))
            call_id = block.get("tool_use_id")
            if text is None or not isinstance(call_id, str):
                continue
            new, label = self._admit_one(
                text,
                call_id,
                calls,
                dispatcher,
                base,
                is_error=bool(block.get("is_error")),
                applyable=applyable,
            )
            if new is None:
                continue
            if clone is None:
                clone = copy.deepcopy(msg)
            _set_block_text(clone["content"][idx], new)
            applied.append(label)
        return clone if clone is not None else msg
