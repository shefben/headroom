"""Phase-2 gate through the real proxy app: a normal trusted wrap request uses the
task-state fold; sessions stay isolated; stateless mode writes nothing; features
can be disabled; wrap enables the safe defaults."""

from __future__ import annotations

import json
import os

import httpx
import pytest
from fastapi.testclient import TestClient

from headroom import paths
from headroom.agent import registry
from headroom.agent.task_state import BLOCK_TAG
from headroom.proxy.server import ProxyConfig, create_app
from headroom.proxy.turn_hooks import clear_turn_hooks
from headroom.rollout import apply_wrap_feature_defaults, resolve_rollout

PORT = 18978
LOOPBACK = ("127.0.0.1", 50000)
MODEL = "claude-sonnet-5"
UPSTREAM_BODY = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": MODEL,
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 5,
        "output_tokens": 1,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    },
}


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "ws"))
    monkeypatch.setenv("HEADROOM_JEVK5", "off")
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    clear_turn_hooks()
    registry.clear_registry()
    yield
    clear_turn_hooks()
    registry.clear_registry()
    paths.set_process_stateless(False)


def _config(env: dict[str, str], *, stateless: bool = True) -> ProxyConfig:
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


def _wrap_env(**extra: str) -> dict[str, str]:
    return dict(apply_wrap_feature_defaults(dict(extra)))


def _marker(tmp_path, token: str, name: str) -> str:
    cwd = tmp_path / name
    cwd.mkdir(exist_ok=True)
    (cwd / "parser.py").write_text("def decode_escape(s):\n    return s\n", encoding="utf-8")
    directory = paths.proxy_clients_dir(PORT)
    directory.mkdir(parents=True, exist_ok=True)
    pid = os.getpid() if name == "project" else os.getppid()  # both alive
    marker = directory / f"{pid}.json"
    marker.write_text(json.dumps({"session_token": token, "cwd": str(cwd)}), encoding="utf-8")
    return str(cwd)


def _client(app) -> tuple[TestClient, list[dict]]:
    sent: list[dict] = []
    client = TestClient(app, base_url="http://127.0.0.1", client=LOOPBACK)

    async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
        payload = body if isinstance(body, dict) else json.loads(body)
        sent.append(payload)
        return httpx.Response(200, json=UPSTREAM_BODY)

    return client, sent


def _post(client, messages, token):
    headers = {"x-api-key": "k", "anthropic-version": "2023-06-01"}
    if token:
        headers["x-headroom-session-token"] = token
    return client.post(
        "/v1/messages",
        headers=headers,
        json={"model": MODEL, "max_tokens": 16, "messages": messages},
    )


FIRST = "Fix decode_escape in parser.py. Tests must pass. Don't change the public API."


def _drain(runtime) -> None:
    for _ in range(3):
        assert runtime.flush(5)


def test_wrapped_request_uses_state_fold_and_sessions_are_isolated(tmp_path):
    _marker(tmp_path, "tok-a", "project")
    app = create_app(_config(_wrap_env()))
    client, sent = _client(app)
    with client:
        client.app.state.proxy._retry_request = client.app.state.proxy._retry_request
        proxy = client.app.state.proxy

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            sent.append(body if isinstance(body, dict) else json.loads(body))
            return httpx.Response(200, json=UPSTREAM_BODY)

        proxy._retry_request = _fake_retry
        runtime = client.app.state.agent_runtime
        assert set(runtime.services) >= {"task_state", "evidence", "code_graph", "ledger"}

        msgs = [{"role": "user", "content": FIRST}]
        assert _post(client, msgs, "tok-a").status_code == 200
        _drain(runtime)
        # Turn 1: no compiled state existed yet, so nothing was injected.
        assert not any(BLOCK_TAG in json.dumps(m) for m in sent[-1]["messages"])

        msgs += [
            {"role": "assistant", "content": "Looking at it."},
            {"role": "user", "content": "please continue"},
        ]
        assert _post(client, msgs, "tok-a").status_code == 200
        _drain(runtime)
        forwarded = sent[-1]
        last = forwarded["messages"][-1]
        assert last["role"] == "system" and f"<{BLOCK_TAG}" in last["content"]
        assert "public API" in last["content"] and "Tests must pass" in last["content"]
        # Cache-hot prefix and the latest user message are byte-identical.
        assert forwarded["messages"][:-1] == msgs
        assert forwarded.get("system") in (None, "", [])

        # A second trusted session in the same proxy sees none of it.
        _marker(tmp_path, "tok-b", "other")
        other = [{"role": "user", "content": "Write release notes for v2."}]
        assert _post(client, other, "tok-b").status_code == 200
        _drain(runtime)
        other += [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "go on"}]
        assert _post(client, other, "tok-b").status_code == 200
        block = sent[-1]["messages"][-1]["content"]
        assert "release notes" in block and "public API" not in block

        health = client.get("/health").json()["agent_runtime"]
        assert health["features"] == {
            "agent_task_state": True,
            "agent_evidence": True,
            "agent_code_graph": True,
        }
        services = health["runtime"]["services"]
        assert services["task_state"]["counters"]["injected"] >= 2
        assert "code_graph" in services
    assert not (paths.workspace_dir() / "agent").exists()  # stateless: zero writes


def test_disabled_feature_means_no_fold(tmp_path):
    _marker(tmp_path, "tok-a", "project")
    app = create_app(_config(_wrap_env(HEADROOM_DISABLE_FEATURES="agent_task_state")))
    client, sent = _client(app)
    with client:
        proxy = client.app.state.proxy

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            sent.append(body if isinstance(body, dict) else json.loads(body))
            return httpx.Response(200, json=UPSTREAM_BODY)

        proxy._retry_request = _fake_retry
        runtime = client.app.state.agent_runtime
        assert "task_state" not in runtime.services and "evidence" in runtime.services
        msgs = [{"role": "user", "content": FIRST}]
        _post(client, msgs, "tok-a")
        _drain(runtime)
        msgs += [{"role": "assistant", "content": "x"}, {"role": "user", "content": "go"}]
        _post(client, msgs, "tok-a")
        assert sent[-1]["messages"] == msgs
        assert "task_state" not in registry.snapshot()["request_stages"]


def test_persistent_ledger_written_beside_event_store(tmp_path):
    _marker(tmp_path, "tok-a", "project")
    app = create_app(_config(_wrap_env(), stateless=False))
    client, sent = _client(app)
    with client:
        proxy = client.app.state.proxy

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            return httpx.Response(200, json=UPSTREAM_BODY)

        proxy._retry_request = _fake_retry
        runtime = client.app.state.agent_runtime
        _post(client, [{"role": "user", "content": FIRST}], "tok-a")
        _drain(runtime)
    ledger = paths.agent_state_db_path().with_name("agent_ledger.db")
    assert ledger.exists()


def test_wrap_feature_defaults_respect_disable():
    env = apply_wrap_feature_defaults({"HEADROOM_FEATURES": "read_maturation"})
    assert env["HEADROOM_FEATURES"].split(",") == [
        "read_maturation",
        "agent_runtime",
        "agent_task_state",
        "agent_evidence",
        "agent_code_graph",
        "agent_tool_admission",
        "agent_arbiter",
        "agent_tool_catalog",
        "agent_complexity_routing",
        "agent_feedback",
        "agent_speculation",
    ]
    env = apply_wrap_feature_defaults({"HEADROOM_DISABLE_FEATURES": "agent_runtime,agent_evidence"})
    snapshot = resolve_rollout(env)
    assert not snapshot.is_enabled("agent_runtime")
    assert not snapshot.is_enabled("agent_evidence")
    assert snapshot.is_enabled("agent_task_state")
    # A bare proxy (no wrap defaults) keeps every agent feature opt-in.
    bare = resolve_rollout({})
    assert not any(
        bare.is_enabled(n)
        for n in (
            "agent_task_state",
            "agent_evidence",
            "agent_code_graph",
            "agent_tool_admission",
            "agent_arbiter",
            "agent_tool_catalog",
            "agent_complexity_routing",
            "agent_feedback",
            "agent_speculation",
        )
    )
