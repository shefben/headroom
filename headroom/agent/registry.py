"""Extension points later agent phases register through (never replace).

* :func:`register_request_stage` — a synchronous, request-scoped step run by
  the first-party agent TurnHook (priority 80) after events are normalized.
  It receives the live :class:`~headroom.proxy.turn_hooks.TurnContext`, the
  request's :class:`~headroom.agent.interfaces.TaskContext` and the
  :class:`~headroom.agent.runtime.AgentRuntime`. A stage may fold
  ``ctx.messages`` (it runs on streamed turns, so it must never need a
  re-drive) and must be fast: no SQLite, no JevK5, no network.
* :func:`register_projector` — materializes state from events on the runtime's
  worker thread, inside the same transaction as the event insert.
* :func:`register_commit_listener` — called on the worker thread after a batch
  commits, with the newly committed events.
* :func:`register_component` — a one-shot installer invoked once per runtime
  when the proxy installs the agent runtime (``create_app``).

Registrations are code, not user state: they are keyed by name, idempotent,
and hold no per-session data.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from headroom.agent.events import AgentEvent
    from headroom.agent.interfaces import TaskContext
    from headroom.agent.runtime import AgentRuntime
    from headroom.agent.store import Projector
    from headroom.proxy.turn_hooks import TurnContext

RequestStage = Callable[["TurnContext", "TaskContext", "AgentRuntime"], None]
CommitListener = Callable[[list["AgentEvent"], "AgentRuntime"], None]
ComponentInstaller = Callable[["AgentRuntime"], None]


@dataclass(frozen=True)
class _Stage:
    name: str
    order: int
    fn: RequestStage


_lock = threading.Lock()
_stages: dict[str, _Stage] = {}
_projectors: dict[str, Projector] = {}
_listeners: dict[str, CommitListener] = {}
_components: dict[str, ComponentInstaller] = {}


def register_request_stage(name: str, stage: RequestStage, *, order: int = 100) -> None:
    with _lock:
        _stages[name] = _Stage(name, int(order), stage)


def register_projector(projector: Projector) -> None:
    name = getattr(projector, "name", None)
    if not name:
        raise ValueError("projector must define a non-empty name")
    with _lock:
        _projectors[str(name)] = projector


def register_commit_listener(name: str, listener: CommitListener) -> None:
    with _lock:
        _listeners[name] = listener


def register_component(name: str, installer: ComponentInstaller) -> None:
    with _lock:
        _components[name] = installer


def request_stages() -> list[tuple[str, RequestStage]]:
    with _lock:
        ordered = sorted(_stages.values(), key=lambda s: (s.order, s.name))
    return [(stage.name, stage.fn) for stage in ordered]


def projectors() -> list[Projector]:
    with _lock:
        return [_projectors[name] for name in sorted(_projectors)]


def commit_listeners() -> list[tuple[str, CommitListener]]:
    with _lock:
        return sorted(_listeners.items())


def components() -> list[tuple[str, ComponentInstaller]]:
    with _lock:
        return sorted(_components.items())


def unregister(name: str) -> None:
    """Remove ``name`` from every registry (used when a component is torn down)."""
    with _lock:
        _stages.pop(name, None)
        _projectors.pop(name, None)
        _listeners.pop(name, None)
        _components.pop(name, None)


def clear_registry() -> None:
    """Test/reset helper."""
    with _lock:
        _stages.clear()
        _projectors.clear()
        _listeners.clear()
        _components.clear()


def snapshot() -> dict[str, Any]:
    with _lock:
        return {
            "request_stages": sorted(_stages),
            "projectors": sorted(_projectors),
            "commit_listeners": sorted(_listeners),
            "components": sorted(_components),
        }
