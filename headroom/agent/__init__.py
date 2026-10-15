"""Shared agent runtime: session-scoped events, storage, identity and extension points.

Later phases consume these primitives rather than inventing their own:

* events — :mod:`headroom.agent.events` (``AgentEvent``, ``EventKind``,
  ``register_event_kind``)
* interfaces — :mod:`headroom.agent.interfaces` (``TaskContext``,
  ``DecisionHints``, ``ToolResultEnvelope``, ``ActionAuthorization``)
* extension points — :mod:`headroom.agent.registry`
  (``register_request_stage``, ``register_projector``,
  ``register_commit_listener``, ``register_component``)
* runtime — :mod:`headroom.agent.runtime` (``AgentRuntime``)
"""

from headroom.agent.events import AgentEvent, EventEnvelope, EventKind, Origin, register_event_kind
from headroom.agent.identity import WorkspaceIdentity, resolve_identity
from headroom.agent.interfaces import (
    ActionAuthorization,
    AuthorizationDecision,
    DecisionHints,
    TaskContext,
    ToolResultEnvelope,
)
from headroom.agent.registry import (
    register_commit_listener,
    register_component,
    register_projector,
    register_request_stage,
)
from headroom.agent.runtime import AgentRuntime

__all__ = [
    "ActionAuthorization",
    "AgentEvent",
    "AgentRuntime",
    "AuthorizationDecision",
    "DecisionHints",
    "EventEnvelope",
    "EventKind",
    "Origin",
    "TaskContext",
    "ToolResultEnvelope",
    "WorkspaceIdentity",
    "register_commit_listener",
    "register_component",
    "register_event_kind",
    "register_projector",
    "register_request_stage",
    "resolve_identity",
]
