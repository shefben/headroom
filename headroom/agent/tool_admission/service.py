"""Proxy-side wiring for tool admission: host-hook endpoint, persistence, health."""

from __future__ import annotations

import logging
from typing import Any

from headroom.agent.tool_admission import FEATURE_TOOL_ADMISSION
from headroom.agent.tool_admission import host as host_adapter
from headroom.agent.tool_admission.dispatcher import get_dispatcher
from headroom.log_safety import describe_exception

log = logging.getLogger(__name__)


def feature_enabled(config: Any, name: str = FEATURE_TOOL_ADMISSION) -> bool:
    rollout = getattr(config, "rollout", None)
    try:
        return bool(rollout is not None and rollout.is_enabled(name))
    except KeyError:
        return False


def phase5_request_parts(config: Any) -> dict[str, Any]:
    """Arbiter / feedback objects for a single (host-hook) admission."""
    parts: dict[str, Any] = {}
    if feature_enabled(config, "agent_arbiter"):
        from headroom.agent.arbiter import RequestLimits, get_arbiter

        parts["arbiter"] = get_arbiter()
        parts["limits"] = RequestLimits()
    if feature_enabled(config, "agent_feedback"):
        from headroom.agent.feedback import get_feedback

        parts["feedback"] = get_feedback()
    return parts


def handle_host_hook_request(data: dict[str, Any], config: Any) -> dict[str, Any]:
    """Body of ``POST /v1/agent/host-hook``: ``{"client": ..., "payload": {...}}``.

    Returns ``{"output": <hook stdout JSON or None>, "stage": ..., "capability": ...}``.
    Every failure answers ``output: None`` (the host proceeds unchanged).
    """
    client = str(data.get("client") or "claude").lower()
    capability = host_adapter.host_capability(client)
    admission_on = feature_enabled(config)
    enforcement_on = feature_enabled(config, "agent_change_scope") or feature_enabled(
        config, "agent_tool_contracts"
    )
    workflows_on = feature_enabled(config, "agent_workflow_macros")
    if not (admission_on or enforcement_on or workflows_on):
        return {"output": None, "stage": "feature_disabled", "capability": capability}
    if capability != host_adapter.HOST_PRE_CONTEXT:
        return {"output": None, "stage": "unsupported_client", "capability": capability}
    payload = data.get("payload")
    if not isinstance(payload, dict):
        return {"output": None, "stage": "bad_payload", "capability": capability}
    try:
        from headroom.agent.hook import latest_session_query
        from headroom.agent.identity import resolve_identity
        from headroom.proxy.project_context import (
            get_current_project,
            get_registered_cwd,
            get_registered_session,
        )

        identity = resolve_identity(
            session_token=get_registered_session(),
            registered_cwd=get_registered_cwd(),
            attribution=get_current_project(),
        )
        if workflows_on and identity.trusted:
            from headroom.agent.enforcement import active_runtime
            from headroom.agent.interfaces import WORKFLOW_TOOL_NAMES

            runtime = active_runtime()
            macros = runtime.services.get("workflow_macros") if runtime is not None else None
            if macros is not None:
                macros.note_hook(identity.session_id, client)  # this session has the hook
                if (
                    payload.get("hook_event_name") == "PreToolUse"
                    and str(payload.get("tool_name") or "") in WORKFLOW_TOOL_NAMES
                ):
                    # Unified-21 #20: the host adapter (workflow_hosts) reads the
                    # hook; the client-independent registry issues the
                    # single-use, revision-pinned execution ticket.
                    host = data.get("host") if isinstance(data.get("host"), dict) else None
                    return macros.pretool(payload, identity=identity, client=client, host=host)
        if payload.get("hook_event_name") == "PreToolUse" and enforcement_on:
            # Unified-21 Phase 6: the trusted pre-tool boundary. Admission's
            # edit-target observation still runs (it never prints anything).
            from headroom.agent.enforcement import host_pretool

            if admission_on:
                host_adapter.handle_hook(payload, identity=identity)
            return host_pretool(payload, identity=identity, client=client)
        if not admission_on:
            return {"output": None, "stage": "admission_disabled", "capability": capability}
        relevance = latest_session_query(identity.session_id)
        query = relevance.text if relevance is not None else ""
        return host_adapter.handle_hook(
            payload,
            identity=identity,
            task=relevance,
            query=query,
            retrieval_available=bool(
                getattr(config, "ccr_inject_tool", True)
                or getattr(config, "ccr_handle_responses", True)
            ),
            **phase5_request_parts(config),
        )
    except Exception as exc:  # noqa: BLE001 - the host must never see an error from us
        log.warning("host hook admission failed: %s", describe_exception(exc))
        return {"output": None, "stage": "error", "capability": capability}


def bind_persistence(runtime: Any) -> bool:
    """Persist resource versions through the agent ledger on the runtime worker."""
    ledger = runtime.services.get("ledger") if runtime is not None else None
    if ledger is None:
        return False
    try:
        get_dispatcher().versions.bind(ledger, runtime.submit_task)
    except Exception as exc:  # noqa: BLE001 - memory-only versions still work
        log.warning("tool admission persistence unavailable: %s", describe_exception(exc))
        return False
    runtime.services["tool_admission"] = _StatusAdapter()
    return True


def unbind_persistence(runtime: Any) -> None:
    get_dispatcher().versions.unbind()
    if runtime is not None:
        runtime.services.pop("tool_admission", None)


class _StatusAdapter:
    def status(self) -> dict[str, Any]:
        return get_dispatcher().status()


def tool_admission_health(config: Any) -> dict[str, Any]:
    enabled = feature_enabled(config)
    block: dict[str, Any] = {"enabled": enabled, "host": host_adapter.status()}
    if enabled:
        block["dispatcher"] = get_dispatcher().status()
    # Unified-21 Phase 5: realized savings are reported apart from estimated
    # avoided turns/time; every block is counts only (no payloads).
    if feature_enabled(config, "agent_arbiter"):
        from headroom.agent.arbiter import get_arbiter

        block["arbiter"] = get_arbiter().status()
    if feature_enabled(config, "agent_feedback"):
        from headroom.agent.feedback import get_feedback

        block["feedback"] = get_feedback().status()
    if feature_enabled(config, "agent_speculation"):
        from headroom.agent.speculation import speculation_status

        block["speculation"] = speculation_status()
    return block


def verification_health(config: Any) -> dict[str, Any]:
    """``/health.agent_verification``: planner + workflow counters (no payloads)."""
    from headroom.agent.enforcement import active_runtime
    from headroom.agent.workflow_macros import workflow_status

    runtime = active_runtime()
    planner = runtime.services.get("test_impact") if runtime is not None else None
    return {
        "test_impact_enabled": feature_enabled(config, "agent_test_impact"),
        "test_impact": planner.status() if planner is not None else None,
        "workflows": workflow_status(config),
    }
