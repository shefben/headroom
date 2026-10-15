"""The first-party agent TurnHook (priority 80, stream-safe, request fold only).

Runs after routers (10-49) and before tool-catalog shaping (200+). It never
re-drives the model and has no ``on_response``, so it is safe on streamed
turns. Per request it:

1. resolves the session/workspace identity from the verified wrap channel
   (``project_context``), never from model-supplied paths or headers;
2. normalizes the provider messages into digest-only events (read-only);
3. builds the request's :class:`TaskContext` and runs registered request
   stages (``agent.registry``) in order, each isolated;
4. hands the events to the runtime's background worker (non-blocking).

Any failure is logged once per kind and the request proceeds untouched.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from headroom.agent import registry
from headroom.agent.identity import WorkspaceIdentity, resolve_identity
from headroom.agent.interfaces import TASK_CONTEXT_VERSION, RelevanceQuery, TaskContext
from headroom.agent.normalize import normalize_messages
from headroom.agent.query import build_task_context
from headroom.log_safety import describe_exception

if TYPE_CHECKING:
    from headroom.agent.runtime import AgentRuntime
    from headroom.proxy.turn_hooks import TurnContext

log = logging.getLogger(__name__)

HOOK_NAME = "headroom_agent_runtime"
HOOK_PRIORITY = 80


def task_context_for(ctx: Any) -> TaskContext | None:
    """The TaskContext the agent hook attached to a TurnContext, if any."""
    value = getattr(ctx, "agent_task_context", None)
    return value if isinstance(value, TaskContext) else None


def identity_for(ctx: Any) -> WorkspaceIdentity | None:
    value = getattr(ctx, "agent_identity", None)
    return value if isinstance(value, WorkspaceIdentity) else None


# Request-scoped TaskContexts built ahead of compression, keyed by request id
# and consumed by ``on_request`` (bounded; entries never outlive a request).
_EARLY_LIMIT = 256
_early_lock = threading.Lock()
_early: OrderedDict[str, tuple[WorkspaceIdentity, Any, TaskContext]] = OrderedDict()


def _active_runtime() -> AgentRuntime | None:
    from headroom.agent.install import _installed

    hook = _installed.get("hook")
    return getattr(hook, "runtime", None) if hook is not None else None


def compression_relevance_query(
    provider: str,
    model: str,
    messages: Any,
    *,
    config: Any = None,
    request_id: str | None = None,
) -> RelevanceQuery | None:
    """The request's task query, built BEFORE compression runs (unified-21 #1).

    The agent TurnHook runs after the compression pipeline, so the handlers
    call this first and pass the query to the compressors. It runs the same
    identity resolution, read-only normalization and ``build_task_context`` as
    the hook (authentic user text only, entities first, 512-token cap) and
    parks the result for ``on_request`` under ``request_id`` so the hook does
    not rebuild it. ``None`` when the agent runtime is not installed or the
    build fails: callers keep the legacy latest-user-text context.
    """
    runtime = _active_runtime()
    if runtime is None:
        return None
    from headroom.proxy.project_context import (
        get_current_project,
        get_registered_cwd,
        get_registered_session,
    )

    try:
        identity = resolve_identity(
            session_token=get_registered_session(),
            registered_cwd=get_registered_cwd(),
            attribution=get_current_project(),
        )
        turn = normalize_messages(provider, messages, identity)
        rid = request_id or uuid.uuid4().hex
        shim = SimpleNamespace(provider=provider, model=model, config=config)
        task = build_task_context(
            ctx=shim, identity=identity, turn=turn, runtime=runtime, request_id=rid
        )
    except Exception as exc:  # noqa: BLE001 - compression falls back to the legacy context
        log.warning("agent compression query build failed: %s", describe_exception(exc))
        return None
    if request_id:
        with _early_lock:
            _early[request_id] = (identity, turn, task)
            while len(_early) > _EARLY_LIMIT:
                _early.popitem(last=False)
    _remember_session_query(identity.session_id, task.relevance_query)
    return task.relevance_query


_SESSION_QUERY_LIMIT = 512
_session_queries: OrderedDict[str, RelevanceQuery] = OrderedDict()


def _remember_session_query(session_id: str, query: RelevanceQuery | None) -> None:
    if query is None or not session_id:
        return
    with _early_lock:
        _session_queries[session_id] = query
        _session_queries.move_to_end(session_id)
        while len(_session_queries) > _SESSION_QUERY_LIMIT:
            _session_queries.popitem(last=False)


def latest_session_query(session_id: str) -> RelevanceQuery | None:
    """The most recent request's RelevanceQuery for a session (host-hook admission).

    A host tool hook runs between model requests and has no request of its
    own; it conditions on the task the session's latest request carried.
    """
    with _early_lock:
        return _session_queries.get(session_id)


def _take_early(request_id: str) -> tuple[WorkspaceIdentity, Any, TaskContext] | None:
    with _early_lock:
        return _early.pop(request_id, None)


class AgentRuntimeHook:
    name = HOOK_NAME
    priority = HOOK_PRIORITY
    stream_safe = True
    savings_source = "agent_runtime"

    def __init__(self, runtime: AgentRuntime | None = None) -> None:
        self.runtime = runtime

    def bind(self, runtime: AgentRuntime | None) -> None:
        self.runtime = runtime

    def on_request(self, ctx: TurnContext) -> None:
        runtime = self.runtime
        if runtime is None:
            return
        from headroom.proxy.project_context import (
            get_current_project,
            get_registered_cwd,
            get_registered_session,
        )

        request_id = str(
            getattr(ctx, "request_id", None) or ctx.tags.get("request_id") or uuid.uuid4().hex
        )
        early = _take_early(request_id)
        if early is not None:
            # Built pre-compression from the original (uncompressed) request.
            identity, turn, task = early
            runtime.note_request(identity)
            self._finish(ctx, runtime, identity, turn, task)
            return
        try:
            identity = resolve_identity(
                session_token=get_registered_session(),
                registered_cwd=get_registered_cwd(),
                attribution=get_current_project(),
            )
            turn = normalize_messages(ctx.provider, ctx.messages, identity)
        except Exception as exc:  # noqa: BLE001 - observation must never break a request
            log.warning("agent hook normalization failed: %s", describe_exception(exc))
            return
        runtime.note_request(identity)
        try:
            task = build_task_context(
                ctx=ctx, identity=identity, turn=turn, runtime=runtime, request_id=request_id
            )
        except Exception as exc:  # noqa: BLE001 - fall back to the minimal context
            log.warning("agent task context build failed: %s", describe_exception(exc))
            task = TaskContext(
                v=TASK_CONTEXT_VERSION,
                request_id=request_id,
                workspace_id=identity.workspace_id,
                session_id=identity.session_id,
                trusted=identity.trusted,
                provider=ctx.provider,
                model=ctx.model,
                tool_names=turn.tool_names,
                event_ids=tuple(ev.id for ev in turn.events),
                latest_user_digest=turn.latest_user_digest,
            )
        self._finish(ctx, runtime, identity, turn, task)

    @staticmethod
    def _finish(
        ctx: TurnContext,
        runtime: AgentRuntime,
        identity: WorkspaceIdentity,
        turn: Any,
        task: TaskContext,
    ) -> None:
        # Request-scoped, attached to this TurnContext only (never a global).
        ctx.agent_task_context = task  # type: ignore[attr-defined]
        ctx.agent_identity = identity  # type: ignore[attr-defined]
        _remember_session_query(identity.session_id, task.relevance_query)
        for name, stage in registry.request_stages():
            try:
                stage(ctx, task, runtime)
            except Exception as exc:  # noqa: BLE001 - isolate stages
                log.warning("agent request stage %s failed: %s", name, describe_exception(exc))
        runtime.submit(turn.envelopes)
