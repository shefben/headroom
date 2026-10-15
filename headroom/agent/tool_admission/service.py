"""Proxy-side wiring for tool admission: host-hook endpoint, persistence, health."""

from __future__ import annotations

import logging
from typing import Any

from headroom.agent.tool_admission import FEATURE_TOOL_ADMISSION
from headroom.agent.tool_admission import host as host_adapter
from headroom.agent.tool_admission.dispatcher import get_dispatcher
from headroom.log_safety import describe_exception

log = logging.getLogger(__name__)


def feature_enabled(config: Any) -> bool:
    rollout = getattr(config, "rollout", None)
    try:
        return bool(rollout is not None and rollout.is_enabled(FEATURE_TOOL_ADMISSION))
    except KeyError:
        return False


def handle_host_hook_request(data: dict[str, Any], config: Any) -> dict[str, Any]:
    """Body of ``POST /v1/agent/host-hook``: ``{"client": ..., "payload": {...}}``.

    Returns ``{"output": <hook stdout JSON or None>, "stage": ..., "capability": ...}``.
    Every failure answers ``output: None`` (the host proceeds unchanged).
    """
    client = str(data.get("client") or "claude").lower()
    capability = host_adapter.host_capability(client)
    if not feature_enabled(config):
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
    return block
