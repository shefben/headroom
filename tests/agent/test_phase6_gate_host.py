"""Gate 6 live fixture: the real Claude Code PreToolUse hook against a real proxy.

A ``create_app`` proxy (wrap rollout defaults) is served by uvicorn; the
user's instructions reach it through the normal ``/v1/messages`` path with
the wrap session token (upstream faked), and the hook command wrap installs
(``python -m headroom.host_hook claude --proxy <url>``) runs as a subprocess
with Claude Code's PreToolUse JSON on stdin. Its stdout is exactly what
Claude Code acts on.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

from headroom import paths
from headroom.agent import enforcement, registry
from headroom.cli import wrap as wrap_cli
from headroom.proxy.server import ProxyConfig, create_app
from headroom.proxy.turn_hooks import clear_turn_hooks
from headroom.rollout import apply_wrap_feature_defaults, resolve_rollout

uvicorn = pytest.importorskip("uvicorn")

TOKEN = "gate6-session-token"
MODEL = "claude-sonnet-5"
UPSTREAM = {
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


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def gate(tmp_path, monkeypatch):
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
    enforcement.reset()
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "config").mkdir()
    (project / "src" / "app.py").write_text(
        "from src.util import helper\n\ndef run():\n    return helper()\n", encoding="utf-8"
    )
    (project / "src" / "util.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    secret = project / "config" / "secrets.yaml"
    secret.write_text("token: keep-me\n", encoding="utf-8")
    port = _free_port()
    marker_dir = paths.proxy_clients_dir(port)
    marker_dir.mkdir(parents=True, exist_ok=True)
    (marker_dir / f"{os.getpid()}.json").write_text(
        json.dumps({"session_token": TOKEN, "cwd": str(project)}), encoding="utf-8"
    )
    config = ProxyConfig(
        port=port,
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        memory_enabled=False,
        image_optimize=False,
        stateless=True,
        rollout=resolve_rollout(apply_wrap_feature_defaults({})),
    )
    app = create_app(config)

    async def fake(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
        return httpx.Response(200, json=UPSTREAM)

    app.state.proxy._retry_request = fake
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "proxy did not start"
    try:
        yield {"url": f"http://127.0.0.1:{port}", "app": app, "project": project, "secret": secret}
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        clear_turn_hooks()
        registry.clear_registry()
        enforcement.reset()
        paths.set_process_stateless(False)


def _say(gate, *user_turns: str) -> None:
    """Send the conversation so far through the real /v1/messages path."""
    messages: list[dict] = []
    for i, text in enumerate(user_turns):
        if i:
            messages.append({"role": "assistant", "content": "ok"})
        messages.append({"role": "user", "content": text})
    resp = httpx.post(
        f"{gate['url']}/v1/messages",
        headers={
            "x-api-key": "k",
            "anthropic-version": "2023-06-01",
            "x-headroom-session-token": TOKEN,
        },
        json={"model": MODEL, "max_tokens": 16, "messages": messages},
        timeout=30,
    )
    assert resp.status_code == 200, resp.text
    runtime = gate["app"].state.agent_runtime
    for _ in range(4):
        assert runtime.flush(5)


def _hook_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("HEADROOM_HOOK")}
    env["ANTHROPIC_CUSTOM_HEADERS"] = f"X-Headroom-Session-Token: {TOKEN}"
    env.pop("ANTHROPIC_BASE_URL", None)
    env.update(extra)
    return env


def _pretool(gate, tool: str, tool_input: dict, use_id: str, *, client: str = "claude"):
    payload = {
        "session_id": "gate6-conv",
        "transcript_path": str(gate["project"] / "t.jsonl"),
        "cwd": str(gate["project"]),
        "permission_mode": "default",
        "hook_event_name": "PreToolUse",
        "tool_name": tool,
        "tool_input": tool_input,
        "tool_use_id": use_id,
    }
    proc = subprocess.run(
        [sys.executable, "-m", "headroom.host_hook", client, "--proxy", gate["url"]],
        input=json.dumps(payload).encode("utf-8"),
        capture_output=True,
        env=_hook_env(),
        timeout=60,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert proc.returncode == 0, proc.stderr  # never exit 2: the host decides from stdout
    return json.loads(proc.stdout)["hookSpecificOutput"] if proc.stdout else None


FIRST = "Fix the crash in src/app.py. Do not modify config/secrets.yaml."


def test_gate6_forbidden_write_blocked_at_genuine_host_pretool(gate):
    _say(gate, FIRST)
    secret = gate["secret"]
    spec = _pretool(
        gate, "Write", {"file_path": str(secret), "content": "token: leaked\n"}, "toolu_g1"
    )
    assert spec["hookEventName"] == "PreToolUse"
    assert spec["permissionDecision"] == "deny"
    assert "scope.user_forbidden" in spec["permissionDecisionReason"]
    assert "updatedInput" not in spec
    assert secret.read_text(encoding="utf-8") == "token: keep-me\n"
    # A shell write to the same path is denied too.
    shell = _pretool(gate, "Bash", {"command": "echo x > config/secrets.yaml"}, "toolu_g2")
    assert shell["permissionDecision"] == "deny"
    # Hard-protected internals, regardless of instructions.
    git = _pretool(
        gate,
        "Edit",
        {
            "file_path": str(gate["project"] / ".git" / "config"),
            "old_string": "a",
            "new_string": "b",
        },
        "toolu_g3",
    )
    assert git["permissionDecision"] == "deny"
    assert "scope.hard_protected.git_internal" in git["permissionDecisionReason"]
    health = httpx.get(f"{gate['url']}/health", timeout=10).json()["agent_enforcement"]
    assert health["counters"]["host_prevented"] >= 2


def test_gate6_legitimate_edits_and_dependency_expansion_allowed(gate):
    _say(gate, FIRST)
    project = gate["project"]
    edit = _pretool(
        gate,
        "Edit",
        {
            "file_path": str(project / "src" / "app.py"),
            "old_string": "helper()",
            "new_string": "helper() + 0",
        },
        "toolu_g4",
    )
    assert edit is None  # prints nothing: Claude Code's own permission flow is untouched
    dep = _pretool(
        gate,
        "Edit",
        {"file_path": str(project / "src" / "util.py"), "old_string": "1", "new_string": "2"},
        "toolu_g5",
    )
    assert dep is None  # src/app.py imports src.util: traceable dependency
    status = httpx.get(f"{gate['url']}/health", timeout=10).json()["agent_enforcement"]
    assert status["change_scope"]["counters"]["dependency_expansions"] >= 1
    assert status["counters"]["host_prevented"] == 0


def test_gate6_typo_repair_proposed_not_executed(gate):
    _say(gate, FIRST)
    project = gate["project"]
    typo = project / "src" / "ap.py"
    spec = _pretool(
        gate, "Edit", {"file_path": str(typo), "old_string": "a", "new_string": "b"}, "toolu_g6"
    )
    assert spec is not None and "permissionDecision" not in spec and "updatedInput" not in spec
    assert "suggested (not run)" in spec["additionalContext"]
    assert "src/app.py" in spec["additionalContext"].replace("\\", "/")
    assert not typo.exists()  # nothing was executed or created on the agent's behalf
    command = _pretool(gate, "Bash", {"command": "gti status"}, "toolu_g7")
    if command is not None:  # `git` on PATH: the repair names it, the call is untouched
        assert "permissionDecision" not in command


def test_gate6_newer_revision_cannot_be_authorized_by_stale_state(gate):
    _say(gate, FIRST)
    project = gate["project"]
    first = _pretool(
        gate,
        "Edit",
        {"file_path": str(project / "src" / "util.py"), "old_string": "1", "new_string": "2"},
        "toolu_g8",
    )
    assert first is None
    # The user narrows the scope: the earlier authorization does not carry over.
    _say(gate, FIRST, "Actually, do not modify src/util.py at all.")
    second = _pretool(
        gate,
        "Edit",
        {"file_path": str(project / "src" / "util.py"), "old_string": "1", "new_string": "2"},
        "toolu_g9",
    )
    assert second is not None and second["permissionDecision"] == "deny"
    # Concurrent hook invocations all see the same, current revision.
    results: list = []

    def run(i: int) -> None:
        results.append(
            _pretool(
                gate,
                "Write",
                {"file_path": str(project / "src" / "util.py"), "content": "x"},
                f"toolu_c{i}",
            )
        )

    threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 4 and all(r and r["permissionDecision"] == "deny" for r in results)


def test_gate6_proxy_only_clients_report_observe_mode(gate, capsys):
    _say(gate, FIRST)
    out = _pretool(
        gate,
        "Write",
        {"file_path": str(gate["secret"]), "content": "x"},
        "toolu_g10",
        client="codex",
    )
    assert out is None  # no fabricated "blocked" status for a client without a host hook
    health = httpx.get(f"{gate['url']}/health", timeout=10).json()["agent_enforcement"]
    assert health["proxy_path"] == "observe_only"
    assert {health["capabilities"][c] for c in ("codex", "opencode", "gemini")} == {"observe_only"}
    wrap_cli._report_tool_enforcement("codex", hook_installed=False)
    wrap_cli._report_tool_enforcement("claude", hook_installed=False)  # user kept own --settings
    wrap_cli._report_tool_enforcement("claude", hook_installed=True)
    lines = capsys.readouterr().out.splitlines()
    assert "observe-only" in lines[0] and "nothing is blocked" in lines[0]
    assert "observe-only" in lines[1]
    assert "host pre-tool hook" in lines[2] and "scope=warn" in lines[2]


def test_gate6_existing_permissions_unaffected(tmp_path, monkeypatch):
    monkeypatch.delenv("HEADROOM_DISABLE_FEATURES", raising=False)
    monkeypatch.delenv("HEADROOM_FEATURES", raising=False)
    user = {
        "permissions": {"allow": ["Read", "Bash(git status)"], "deny": ["Write(./secrets/**)"]},
        "hooks": {
            "PreToolUse": [
                {"matcher": "Bash", "hooks": [{"type": "command", "command": "my-guard"}]}
            ]
        },
    }
    path = tmp_path / ".claude" / "settings.local.json"
    path.parent.mkdir()
    path.write_text(json.dumps(user), encoding="utf-8")
    command = wrap_cli._claude_tool_hook_command()
    wrap_cli._ensure_claude_tool_hooks(path, command)
    wrap_cli._ensure_claude_tool_hooks(path, command)  # idempotent
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["permissions"] == user["permissions"]
    pre = data["hooks"]["PreToolUse"]
    assert pre[0] == user["hooks"]["PreToolUse"][0]  # the user's own guard first, untouched
    ours = [
        (e["matcher"], h["command"])
        for e in pre
        for h in e["hooks"]
        if "headroom.host_hook" in h["command"]
    ]
    assert sorted(m for m, _ in ours) == [
        "Bash|PowerShell",
        "Edit|Write|MultiEdit|NotebookEdit",
        "mcp__headroom__headroom_workflow",  # Phase 7 workflow tool boundary
    ]
    assert wrap_cli._remove_claude_tool_hooks(path)
    assert json.loads(path.read_text(encoding="utf-8")) == user
    # The --settings flag carries hooks only, never permissions.
    args, installed = wrap_cli._claude_args_with_tool_hooks((), command)
    flag = json.loads(args[1])
    assert installed and set(flag) == {"hooks"}
    assert [e["matcher"] for e in flag["hooks"]["PreToolUse"]] == [
        "Edit|Write|MultiEdit|NotebookEdit",
        "Bash|PowerShell",
        "mcp__headroom__headroom_workflow",
    ]
