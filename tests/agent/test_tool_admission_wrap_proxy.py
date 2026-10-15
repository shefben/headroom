"""Gate 4: wrap hook installation never clobbers user settings, and the proxy
live-zone fallback works through the real /v1/messages path without a host hook."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from headroom import paths
from headroom.agent import registry
from headroom.cli import wrap as wrap_cli
from headroom.proxy.server import ProxyConfig, create_app
from headroom.proxy.turn_hooks import clear_turn_hooks
from headroom.rollout import apply_wrap_feature_defaults, resolve_rollout
from tests.agent.test_tool_admission import pytest_output, search_output

USER_SETTINGS = {
    "permissions": {"allow": ["Read"]},
    "env": {"KEEP": "1"},
    "hooks": {
        "PostToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "echo mine"}]}],
        "Stop": [{"hooks": [{"type": "command", "command": "notify"}]}],
    },
}


def test_project_settings_hooks_install_idempotent_and_remove_cleanly(tmp_path):
    path = tmp_path / ".claude" / "settings.local.json"
    path.parent.mkdir()
    path.write_text(json.dumps(USER_SETTINGS), encoding="utf-8")
    cmd = wrap_cli._claude_tool_hook_command()
    wrap_cli._ensure_claude_tool_hooks(path, cmd)
    wrap_cli._ensure_claude_tool_hooks(path, cmd)  # idempotent
    data = json.loads(path.read_text(encoding="utf-8"))
    post = data["hooks"]["PostToolUse"]
    assert post[0] == USER_SETTINGS["hooks"]["PostToolUse"][0]  # user hook first, untouched
    ours = [h for e in post for h in e["hooks"] if "headroom.host_hook" in h["command"]]
    assert len(ours) == 1
    assert data["hooks"]["PreToolUse"][0]["matcher"] == "Edit|Write|MultiEdit|NotebookEdit"
    assert data["permissions"] == USER_SETTINGS["permissions"] and data["env"] == USER_SETTINGS["env"]
    assert wrap_cli._remove_claude_tool_hooks(path)
    assert json.loads(path.read_text(encoding="utf-8")) == USER_SETTINGS


def test_settings_flag_respects_user_settings_flag():
    cmd = wrap_cli._claude_tool_hook_command("http://127.0.0.1:8787")
    args, installed = wrap_cli._claude_args_with_tool_hooks(("--resume", "x"), cmd)
    assert installed and args[0] == "--settings" and args[2:] == ("--resume", "x")
    hooks = json.loads(args[1])["hooks"]
    assert hooks["PostToolUse"][0]["matcher"] == "Bash|PowerShell|Grep|Glob|mcp__.*"
    user = ("--settings", "my.json")
    assert wrap_cli._claude_args_with_tool_hooks(user, cmd) == (user, False)


def test_corrupt_settings_are_never_overwritten(tmp_path):
    import click

    path = tmp_path / ".claude" / "settings.local.json"
    path.parent.mkdir()
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(click.ClickException):
        wrap_cli._ensure_claude_tool_hooks(path, "x headroom.host_hook")
    assert path.read_text(encoding="utf-8") == "{broken"


# ---------------------------------------------------------------------------
# Proxy fallback through the real request path
# ---------------------------------------------------------------------------

UPSTREAM = {
    "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5",
    "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 1, "cache_read_input_tokens": 0,
              "cache_creation_input_tokens": 0},
}


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "ws"))
    monkeypatch.setenv("HEADROOM_JEVK5", "off")
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    import headroom.agent.tool_admission.dispatcher as disp
    import headroom.cache.compression_store as cs

    monkeypatch.setattr(cs, "_compression_store", None)
    monkeypatch.setattr(disp, "_DISPATCHER", None)
    clear_turn_hooks()
    registry.clear_registry()

    def make(env):
        config = ProxyConfig(
            port=18979, optimize=True, disable_kompress=True, cache_enabled=False,
            rate_limit_enabled=False,
            memory_enabled=False, image_optimize=False, stateless=True,
            rollout=resolve_rollout(env),
        )
        return TestClient(create_app(config), base_url="http://127.0.0.1", client=("127.0.0.1", 50000))

    yield make
    clear_turn_hooks()
    registry.clear_registry()
    paths.set_process_stateless(False)


def _messages():
    return [
        {"role": "user", "content": "Which tests failed? Also where is TokenBudgetExceeded raised?"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "pytest -q"}},
            {"type": "tool_use", "id": "toolu_2", "name": "Grep", "input": {"pattern": "Budget"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": pytest_output()},
            {"type": "tool_result", "tool_use_id": "toolu_2", "content": search_output()},
        ]},
    ]


def _send(client):
    sent: list[dict] = []
    proxy = client.app.state.proxy

    async def fake(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
        sent.append(body if isinstance(body, dict) else json.loads(body))
        return httpx.Response(200, json=UPSTREAM)

    proxy._retry_request = fake
    resp = client.post(
        "/v1/messages",
        headers={"x-api-key": "k", "anthropic-version": "2023-06-01"},
        json={"model": "claude-sonnet-5", "max_tokens": 16, "messages": _messages()},
    )
    assert resp.status_code == 200, resp.text
    return sent[-1]


def test_proxy_fallback_admits_without_host_hook_and_recovers_exactly(app_client):
    with app_client(apply_wrap_feature_defaults({})) as client:
        forwarded = _send(client)
        blocks = forwarded["messages"][2]["content"]
        texts = [b["content"] if isinstance(b["content"], str) else b["content"][0]["text"] for b in blocks]
        assert all("Retrieve more: hash=" in t for t in texts)
        assert "FAILED tests/test_parser.py::test_unicode_escape" in texts[0]
        assert "src/core/budget.py:212:    raise TokenBudgetExceeded(limit)" in texts[1]
        import re

        key = re.search(r"Retrieve more: hash=([a-f0-9]{12,24})", texts[0]).group(1)
        got = client.post("/v1/retrieve", json={"hash": key}).json()
        assert got["original_content"] == pytest_output()
        health = client.get("/health").json()["tool_admission"]
        assert health["dispatcher"]["sources"].get("proxy", 0) >= 2
        assert forwarded["messages"][:2] == _messages()[:2]  # untouched prefix


def test_disabled_feature_leaves_tool_results_to_the_existing_pipeline(app_client):
    env = apply_wrap_feature_defaults({"HEADROOM_DISABLE_FEATURES": "agent_tool_admission"})
    with app_client(env) as client:
        forwarded = _send(client)
        assert "tool_admission" not in json.dumps(client.get("/health").json()["tool_admission"].get("dispatcher", {}))
        joined = json.dumps(forwarded)
        assert "[headroom: test run" not in joined
