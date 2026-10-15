"""Agent runtime wired through create_app: install-once, trust mapping, idempotent replay,
stateless zero-writes, health, and inertness with the rollout feature off."""

from __future__ import annotations

import json
import os

import httpx
import pytest
from fastapi.testclient import TestClient

from headroom import paths
from headroom.agent import registry
from headroom.agent.hook import HOOK_NAME, AgentRuntimeHook
from headroom.agent.runtime import AgentRuntime
from headroom.agent.store import MemoryAgentStore, SQLiteAgentStore
from headroom.proxy.server import ProxyConfig, create_app
from headroom.proxy.turn_hooks import (
    TurnContext,
    clear_turn_hooks,
    register_turn_hook,
    registered_turn_hooks,
    run_request_hooks,
)
from headroom.rollout import resolve_rollout

PORT = 18977
LOOPBACK = ("127.0.0.1", 50000)
UPSTREAM_BODY = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-6",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 1, "cache_read_input_tokens": 0,
              "cache_creation_input_tokens": 0},
}


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "ws"))
    monkeypatch.setenv("HEADROOM_JEVK5", "off")  # no background activation in tests
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    clear_turn_hooks()
    registry.clear_registry()
    yield
    clear_turn_hooks()
    registry.clear_registry()
    paths.set_process_stateless(False)


def _config(*, enabled: bool, stateless: bool) -> ProxyConfig:
    env = {"HEADROOM_FEATURES": "agent_runtime"} if enabled else {}
    return ProxyConfig(
        port=PORT,
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        memory_enabled=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        stateless=stateless,
        rollout=resolve_rollout(env),
    )


def _register_marker(tmp_path, token: str) -> str:
    cwd = tmp_path / "project"
    cwd.mkdir(exist_ok=True)
    directory = paths.proxy_clients_dir(PORT)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{os.getpid()}.json").write_text(
        json.dumps({"session_token": token, "cwd": str(cwd)}), encoding="utf-8"
    )
    return str(cwd)


def _post(client: TestClient, messages: list, token: str | None) -> httpx.Response:
    headers = {"x-api-key": "k", "anthropic-version": "2023-06-01"}
    if token:
        headers["x-headroom-session-token"] = token
    return client.post(
        "/v1/messages",
        json={"model": "claude-sonnet-4-6", "max_tokens": 16, "messages": messages},
        headers=headers,
    )


def _fake_upstream(client: TestClient) -> None:
    async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
        return httpx.Response(200, json=UPSTREAM_BODY)

    client.app.state.proxy._retry_request = _fake_retry  # type: ignore[assignment]


def test_feature_off_registers_nothing_and_health_reports_it():
    app = create_app(_config(enabled=False, stateless=True))
    assert app.state.agent_runtime is None
    assert not [h for h in registered_turn_hooks() if h.name == HOOK_NAME]
    with TestClient(app, base_url="http://127.0.0.1", client=LOOPBACK) as client:
        body = client.get("/health").json()
    assert body["agent_runtime"]["feature_enabled"] is False
    assert body["agent_runtime"]["hook_installed"] is False
    assert body["config"]["agent_runtime"] is False


def test_install_once_across_app_recreation():
    create_app(_config(enabled=True, stateless=True))
    app = create_app(_config(enabled=True, stateless=True))
    hooks = [h for h in registered_turn_hooks() if h.name == HOOK_NAME]
    assert len(hooks) == 1 and hooks[0].runtime is app.state.agent_runtime
    assert hooks[0].priority == 80 and hooks[0].stream_safe is True


def test_trusted_session_events_idempotent_and_stateless(tmp_path):
    token = "wrap-token-abc"
    _register_marker(tmp_path, token)
    app = create_app(_config(enabled=True, stateless=True))
    messages = [{"role": "user", "content": "please refactor module x"}]
    with TestClient(app, base_url="http://127.0.0.1", client=LOOPBACK) as client:
        _fake_upstream(client)
        runtime: AgentRuntime = client.app.state.agent_runtime
        assert isinstance(runtime.store, MemoryAgentStore)
        assert _post(client, messages, token).status_code == 200
        assert runtime.flush(5)
        events = runtime.store.events()
        assert [e.kind for e in events] == ["USER_INTENT"]
        assert events[0].workspace_id.startswith("ws_")  # trusted mapping
        trusted_session = events[0].session_id
        # Replay of the same history: processed at most once.
        assert _post(client, messages, token).status_code == 200
        assert runtime.flush(5)
        assert len(runtime.store.events()) == 1
        # No token: untrusted, separate scope, still observed in memory.
        assert _post(client, messages, None).status_code == 200
        assert runtime.flush(5)
        untrusted = [e for e in runtime.store.events() if e.session_id != trusted_session]
        assert untrusted and untrusted[0].workspace_id.startswith("wu_")
        health = client.get("/health").json()["agent_runtime"]
        assert health["hook_installed"] and health["trusted_workspace_markers"] == 1
        assert health["runtime"]["counters"]["trusted_requests"] == 2
        assert health["runtime"]["counters"]["untrusted_requests"] == 1
        assert health["intelligence"]["advisor_state"] == "disabled"
    assert not (paths.workspace_dir() / "agent").exists()  # stateless: zero agent writes


def test_persistent_store_under_workspace(tmp_path):
    app = create_app(_config(enabled=True, stateless=False))
    with TestClient(app, base_url="http://127.0.0.1", client=LOOPBACK) as client:
        _fake_upstream(client)
        runtime = client.app.state.agent_runtime
        assert isinstance(runtime.store, SQLiteAgentStore)
        assert _post(client, [{"role": "user", "content": "hello"}], None).status_code == 200
        assert runtime.flush(5)
        assert len(runtime.store.events()) == 1
    assert paths.agent_state_db_path().exists()
    # Lifespan shutdown stopped the worker and removed the hook.
    assert not runtime.running
    assert not [h for h in registered_turn_hooks() if h.name == HOOK_NAME]


def test_hook_runs_on_streaming_turns_and_never_raises():
    runtime = AgentRuntime(MemoryAgentStore())
    register_turn_hook(AgentRuntimeHook(runtime))
    stage_calls = []
    registry.register_request_stage("probe", lambda ctx, task, rt: stage_calls.append(task))
    registry.register_request_stage("broken", lambda ctx, task, rt: 1 / 0, order=50)
    messages = [{"role": "user", "content": "go"}]
    ctx = TurnContext(provider="anthropic", model="m", messages=messages, tags={})
    run_request_hooks(ctx, stream_safe_only=True)
    assert ctx.messages == [{"role": "user", "content": "go"}]
    assert stage_calls and stage_calls[0].trusted is False
    assert ctx.agent_task_context is stage_calls[0]  # type: ignore[attr-defined]
    assert runtime.flush(5)
    assert len(runtime.store.events()) == 1
    # Garbage input is tolerated.
    run_request_hooks(TurnContext(provider="openai", model="m", messages=[None, 3], tags={}))
    runtime.stop()
