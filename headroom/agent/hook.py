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
import uuid
from typing import TYPE_CHECKING, Any

from headroom.agent import registry
from headroom.agent.identity import WorkspaceIdentity, resolve_identity
from headroom.agent.interfaces import TASK_CONTEXT_VERSION, TaskContext
from headroom.agent.normalize import normalize_messages
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
        request_id = str(ctx.tags.get("request_id") or uuid.uuid4().hex)
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
        # Request-scoped, attached to this TurnContext only (never a global).
        ctx.agent_task_context = task  # type: ignore[attr-defined]
        ctx.agent_identity = identity  # type: ignore[attr-defined]
        for name, stage in registry.request_stages():
            try:
                stage(ctx, task, runtime)
            except Exception as exc:  # noqa: BLE001 - isolate stages
                log.warning("agent request stage %s failed: %s", name, describe_exception(exc))
        runtime.submit(turn.envelopes)
