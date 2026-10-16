"""Central first-party installer for the agent runtime (called once from create_app).

Wrapped sessions get the runtime without any third-party extension opt-in:
``create_app`` calls :func:`install_agent_runtime`, which — only when the
``agent_runtime`` rollout feature is enabled — builds the runtime and its
store, attaches the JevK5 :class:`IntelligenceManager`, registers the single
agent TurnHook (priority 80, stream-safe), and runs registered component
installers. With the feature off nothing is registered, so the request path is
byte-for-byte what it was without this module.

Re-creating the app in one process (tests, embedders) replaces the hook
rather than duplicating it, and the previous runtime is stopped.
"""

from __future__ import annotations

import logging
from typing import Any

from headroom.agent import registry
from headroom.agent.hook import HOOK_NAME, AgentRuntimeHook
from headroom.agent.runtime import AgentRuntime, create_store
from headroom.log_safety import describe_exception

log = logging.getLogger(__name__)

FEATURE = "agent_runtime"

#: Rollout names reported by /health.agent_runtime.rollout (wrap defaults).
_AGENT_FEATURES = (
    "agent_task_state",
    "agent_evidence",
    "agent_code_graph",
    "agent_tool_admission",
    "agent_arbiter",
    "agent_tool_catalog",
    "agent_complexity_routing",
    "agent_feedback",
    "agent_speculation",
    "agent_change_scope",
    "agent_tool_contracts",
    "agent_test_impact",
    "agent_workflow_macros",
)

_installed: dict[str, Any] = {
    "hook": None,
    "runtime": None,
    "intelligence": None,
    "builtin_teardown": None,
}


def feature_enabled(config: Any) -> bool:
    rollout = getattr(config, "rollout", None)
    if rollout is None:
        return False
    try:
        return bool(rollout.is_enabled(FEATURE))
    except KeyError:
        return False


def _teardown_previous() -> None:
    from headroom.proxy.turn_hooks import unregister_turn_hook

    unregister_turn_hook(HOOK_NAME)
    previous_runtime = _installed.get("runtime")
    previous_intel = _installed.get("intelligence")
    previous_teardown = _installed.get("builtin_teardown")
    _installed.update(hook=None, runtime=None, intelligence=None, builtin_teardown=None)
    if previous_intel is not None:
        previous_intel.shutdown(timeout=0.5)
    if previous_runtime is not None:
        previous_runtime.stop(timeout=1.0)
    if previous_teardown is not None:
        previous_teardown()


def install_agent_runtime(app: Any, config: Any) -> AgentRuntime | None:
    """Install (or, with the feature off, guarantee the absence of) the runtime."""
    _teardown_previous()
    app.state.agent_runtime = None
    app.state.agent_intelligence = None
    if not feature_enabled(config):
        return None
    from headroom.intelligence.manager import IntelligenceManager
    from headroom.proxy.turn_hooks import register_turn_hook

    store = create_store(stateless=bool(getattr(config, "stateless", False)))
    intelligence = IntelligenceManager()
    runtime = AgentRuntime(store, advisor=intelligence.advisor)
    hook = AgentRuntimeHook(runtime)
    register_turn_hook(hook)
    from headroom.agent.builtin import install_builtin_components

    try:
        teardown = install_builtin_components(runtime, config)
    except Exception as exc:  # noqa: BLE001 - phase components must not block the runtime
        log.warning("agent built-in components failed to install: %s", describe_exception(exc))
        teardown = None
    _installed.update(
        hook=hook, runtime=runtime, intelligence=intelligence, builtin_teardown=teardown
    )
    for name, installer in registry.components():
        try:
            installer(runtime)
        except Exception as exc:  # noqa: BLE001 - one component must not block the rest
            log.warning("agent component %s failed to install: %s", name, describe_exception(exc))
    app.state.agent_runtime = runtime
    app.state.agent_intelligence = intelligence
    return runtime


def start_agent_runtime(app: Any) -> None:
    """Lifespan startup: start the worker and the non-blocking JevK5 activation."""
    runtime = getattr(app.state, "agent_runtime", None)
    intelligence = getattr(app.state, "agent_intelligence", None)
    if runtime is not None:
        runtime.start()
    if intelligence is not None:
        intelligence.start_background()


def stop_agent_runtime(app: Any) -> None:
    """Lifespan shutdown: stop owned background work (the shared llama-server stays)."""
    intelligence = getattr(app.state, "agent_intelligence", None)
    runtime = getattr(app.state, "agent_runtime", None)
    if intelligence is not None:
        intelligence.shutdown()
    if runtime is not None:
        runtime.stop()
        from headroom.proxy.turn_hooks import unregister_turn_hook

        if _installed.get("runtime") is runtime:
            unregister_turn_hook(HOOK_NAME)
            teardown = _installed.get("builtin_teardown")
            _installed.update(hook=None, runtime=None, intelligence=None, builtin_teardown=None)
            if teardown is not None:
                teardown()


def agent_runtime_health(app: Any, config: Any) -> dict[str, Any]:
    """The /health block: effective rollout, hook installation, trust mapping, service."""
    from headroom.proxy.turn_hooks import registered_turn_hooks

    runtime = getattr(app.state, "agent_runtime", None)
    intelligence = getattr(app.state, "agent_intelligence", None)
    hooks = [h for h in registered_turn_hooks() if getattr(h, "name", None) == HOOK_NAME]
    block: dict[str, Any] = {
        "feature_enabled": feature_enabled(config),
        "features": {
            name: _feature(config, name)
            for name in ("agent_task_state", "agent_evidence", "agent_code_graph")
        },
        # Effective rollout of every agent feature (Phase 8 `headroom agent status`).
        "rollout": {name: _feature(config, name) for name in _AGENT_FEATURES if name != FEATURE},
        "hook_installed": len(hooks) == 1,
        "hook_priority": getattr(hooks[0], "priority", None) if hooks else None,
        "stateless": bool(getattr(config, "stateless", False)),
        "trusted_workspace_markers": _live_marker_count(getattr(config, "port", None)),
    }
    if runtime is not None:
        block["runtime"] = runtime.status()
    if intelligence is not None:
        block["intelligence"] = intelligence.health()
    return block


def _feature(config: Any, name: str) -> bool:
    rollout = getattr(config, "rollout", None)
    try:
        return bool(rollout is not None and rollout.is_enabled(name))
    except KeyError:
        return False


def _live_marker_count(port: Any) -> int:
    """How many live wrap sessions registered a trusted workspace on this port."""
    if not isinstance(port, int):
        return 0
    import json

    from headroom import fsutil, paths
    from headroom.proxy_client_liveness import pid_alive

    directory = paths.proxy_clients_dir(port)
    if not directory.exists():
        return 0
    count = 0
    for marker in directory.glob("*.json"):
        try:
            payload = json.loads(fsutil.read_text(marker))
            pid = int(marker.stem)
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get("session_token") and pid_alive(pid):
            count += 1
    return count
