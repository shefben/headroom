"""Gate 7 live fixture: learned workflow through the real wrap proxy + host hook + MCP.

A ``create_app`` proxy with the wrap rollout defaults is served by uvicorn
(upstream faked, request bodies captured). A real Git + pytest project is
the wrap-registered workspace. The agent's repeated tool rounds reach the
proxy through ``/v1/messages``; Claude Code's ``PreToolUse`` hook runs as the
real ``python -m headroom.host_hook`` subprocess; the workflow call goes
through the real Headroom MCP server handler to ``/v1/agent/workflow``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
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
from headroom.agent.workflow_macros import WORKFLOW_MCP_TOOL, WORKFLOW_TOOL_SCHEMA
from headroom.proxy.server import ProxyConfig, create_app
from headroom.proxy.turn_hooks import clear_turn_hooks
from headroom.rollout import apply_wrap_feature_defaults, resolve_rollout

uvicorn = pytest.importorskip("uvicorn")
pytest.importorskip("mcp")

TOKEN = "gate7-session-token"
MODEL = "claude-sonnet-5"
PY = sys.executable.replace("\\", "/")
REPO = Path(__file__).resolve().parents[2]
TOOLS = [
    {
        "name": "Bash",
        "description": "Run a command",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "Read",
        "description": "Read a file",
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    },
    {
        "name": WORKFLOW_MCP_TOOL,
        "description": WORKFLOW_TOOL_SCHEMA["description"],
        "input_schema": WORKFLOW_TOOL_SCHEMA["input_schema"],
    },
]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def live(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "ws"))
    monkeypatch.setenv("HEADROOM_JEVK5", "off")
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", f"X-Headroom-Session-Token: {TOKEN}")
    import headroom.agent.tool_admission.dispatcher as disp
    import headroom.cache.compression_store as cs

    monkeypatch.setattr(cs, "_compression_store", None)
    monkeypatch.setattr(disp, "_DISPATCHER", None)
    clear_turn_hooks()
    registry.clear_registry()
    enforcement.reset()
    project = tmp_path / "project"
    for rel, text in {
        "pytest.ini": "[pytest]\n",
        ".gitignore": ".pytest_cache/\n__pycache__/\n",
        "src/__init__.py": "",
        "src/calc.py": "def add(a, b):\n    return a + b\n",
        "tests/__init__.py": "",
        "tests/test_calc.py": "from src.calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
    }.items():
        path = project / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(project, "init", "-q")
    _git(project, "config", "user.email", "g7@example.invalid")
    _git(project, "config", "user.name", "g7")
    _git(project, "add", "-A")
    _git(project, "commit", "-qm", "init")
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
    upstream: list[dict] = []

    async def fake(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
        upstream.append(json.loads(json.dumps(body, default=str)))
        return httpx.Response(
            200,
            json={
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
            },
        )

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
        yield {
            "url": f"http://127.0.0.1:{port}",
            "app": app,
            "project": project,
            "upstream": upstream,
        }
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        clear_turn_hooks()
        registry.clear_registry()
        enforcement.reset()
        paths.set_process_stateless(False)


def _flush(live) -> None:
    runtime = live["app"].state.agent_runtime
    for _ in range(4):
        assert runtime.flush(10)


def _send(live, messages: list[dict]) -> None:
    resp = httpx.post(
        f"{live['url']}/v1/messages",
        headers={
            "x-api-key": "k",
            "anthropic-version": "2023-06-01",
            "x-headroom-session-token": TOKEN,
        },
        json={"model": MODEL, "max_tokens": 16, "messages": messages, "tools": TOOLS},
        timeout=60,
    )
    assert resp.status_code == 200, resp.text
    _flush(live)


def _round(root: Path, i: int) -> list[dict]:
    """The agent's ordinary flow: three model-mediated tool turns per ask."""
    steps = [
        (
            "Bash",
            {"command": "git status"},
            "On branch main\nnothing to commit, working tree clean",
        ),
        ("Read", {"file_path": str(root / "src" / "calc.py")}, "     1\tdef add(a, b):"),
        ("Bash", {"command": f"{PY} -m pytest -q tests/test_calc.py"}, "1 passed in 0.02s"),
    ]
    out: list[dict] = [
        {"role": "user", "content": f"Round {i}: check the repo and run the calc tests"}
    ]
    for j, (tool, args, result) in enumerate(steps):
        call = f"toolu_live{i}_{j}"
        out.append(
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": call, "name": tool, "input": args}],
            }
        )
        out.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": call, "content": result}],
            }
        )
    out.append({"role": "assistant", "content": [{"type": "text", "text": "All checks pass."}]})
    return out


def _tool_turns(messages: list[dict]) -> int:
    return sum(
        1
        for m in messages
        if m["role"] == "assistant"
        and any(isinstance(b, dict) and b.get("type") == "tool_use" for b in m["content"])
    )


def _hook(live, tool: str, tool_input: dict, use_id: str):
    payload = {
        "session_id": "gate7-conv",
        "transcript_path": str(live["project"] / "t.jsonl"),
        "cwd": str(live["project"]),
        "permission_mode": "default",
        "hook_event_name": "PreToolUse",
        "tool_name": tool,
        "tool_input": tool_input,
        "tool_use_id": use_id,
    }
    env = {k: v for k, v in os.environ.items() if not k.startswith("HEADROOM_HOOK")}
    env["ANTHROPIC_CUSTOM_HEADERS"] = f"X-Headroom-Session-Token: {TOKEN}"
    env.pop("ANTHROPIC_BASE_URL", None)
    proc = subprocess.run(
        [sys.executable, "-m", "headroom.host_hook", "claude", "--proxy", live["url"]],
        input=json.dumps(payload).encode("utf-8"),
        capture_output=True,
        env=env,
        timeout=90,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)["hookSpecificOutput"] if proc.stdout else None


def _mcp(live):
    from headroom.ccr.mcp_server import HeadroomMCPServer

    return HeadroomMCPServer(proxy_url=live["url"], check_proxy=False)


def _workflow_call(live, arguments: dict) -> dict:
    server = _mcp(live)
    (content,) = asyncio.run(server._handle_workflow(arguments))
    return json.loads(content.text)


def _health(live) -> dict:
    return httpx.get(f"{live['url']}/health", timeout=10).json()["agent_verification"]


def _learn(live) -> tuple[list[dict], str]:
    """Three ordinary rounds → one promoted workflow, advertised on the next request."""
    root = live["project"]
    history: list[dict] = []
    for i in range(3):
        history += _round(root, i)
        history.append({"role": "user", "content": f"Thanks, that was round {i}."})
        _send(live, history)
        history.pop()
    # Claude Code ran the Bash calls through the installed pre-tool hook.
    assert _hook(live, "Bash", {"command": "git status"}, "toolu_hookprobe") is None
    history.append({"role": "user", "content": "Round 3: check the repo and run the calc tests"})
    _send(live, history)
    workflows = _health(live)["workflows"]
    assert workflows["enabled"] and workflows["macros"].get("active") == 1
    body = json.dumps(live["upstream"][-1])
    match = re.search(r"- (wf_[0-9a-f]{20}) \(verification, 3 steps, saves ~2 tool turns\)", body)
    assert match, "the eligible workflow was not advertised to the model"
    return history, match.group(1)


def test_gate7_live_macro_reduces_model_tool_turns(live, monkeypatch):
    history, macro_id = _learn(live)
    baseline_turns = _tool_turns(_round(live["project"], 9))
    assert baseline_turns == 3
    # Observe-only clients (no Headroom pre-tool boundary) are never offered it.
    monkeypatch.setenv("HEADROOM_MCP_CLIENT", "codex-mcp-client")
    assert asyncio.run(_mcp(live)._workflow_tool()) is None
    # Claude Code's MCP client gets the tool (schema from the shared registry).
    monkeypatch.setenv("HEADROOM_MCP_CLIENT", "claude-code")
    tool = asyncio.run(_mcp(live)._workflow_tool())
    assert tool is not None and tool.name == "headroom_workflow"
    # The model calls the workflow once: host pre-tool hook → MCP → proxy.
    call_id = "toolu_wf_live"
    args = {"macro_id": macro_id, "typed_inputs": {}}
    assert _hook(live, WORKFLOW_MCP_TOOL, args, call_id) is None  # ticket, no permission field
    runs_before = live["app"].state.agent_runtime.services["test_impact"].counters["runs_recorded"]
    result = _workflow_call(live, args)
    assert result["status"] == "completed", result
    assert [s["status"] for s in result["steps"]] == ["ok", "ok", "ok"]
    assert result["steps"][2]["tests"]["passed"] == 1
    assert result["estimated_turns_avoided"] == 2
    planner = live["app"].state.agent_runtime.services["test_impact"]
    assert planner.counters["runs_recorded"] == runs_before + 1  # ingested once per step
    # The round finishes with ONE model-mediated tool turn instead of three.
    macro_round = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": call_id, "name": WORKFLOW_MCP_TOOL, "input": args}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": call_id, "content": json.dumps(result)}
            ],
        },
    ]
    history += macro_round
    _send(live, history)
    assert _tool_turns(macro_round) == 1 < baseline_turns
    assert planner.counters["runs_recorded"] == runs_before + 1  # no double count via the proxy
    health = _health(live)["workflows"]["counters"]
    assert health["completed"] == 1 and health["estimated_turns_avoided"] == 2


def test_gate7_live_stale_scope_cancels_before_execution(live):
    history, macro_id = _learn(live)
    args = {"macro_id": macro_id, "typed_inputs": {}}
    assert _hook(live, WORKFLOW_MCP_TOOL, args, "toolu_wf_stale") is None
    # Before the MCP call arrives, the user narrows the scope.
    history += [
        {"role": "assistant", "content": [{"type": "text", "text": "Running the checks."}]},
        {"role": "user", "content": "Actually, only touch src/calc.py and do not modify tests/."},
    ]
    _send(live, history)
    result = _workflow_call(live, args)
    assert result["status"] == "cancelled" and result["reason"] == "workflow.stale_revision"
    assert result["fallback"]["run_with_normal_tools"]
    assert _health(live)["workflows"]["counters"]["steps_run"] == 0  # nothing executed


def test_gate7_live_no_host_ticket_falls_back(live):
    _history, macro_id = _learn(live)
    result = _workflow_call(live, {"macro_id": macro_id, "typed_inputs": {}})
    assert result["status"] == "fallback" and result["reason"] == "workflow.no_host_authorization"
    listing = _workflow_call(live, {"macro_id": "", "dry_run": True})
    assert [w["macro_id"] for w in listing["workflows"]] == [macro_id]
    # An invalid input is denied at the host boundary (the only permission field used).
    spec = _hook(
        live,
        WORKFLOW_MCP_TOOL,
        {"macro_id": macro_id, "typed_inputs": {"tests": ["../x.py"]}},
        "toolu_bad",
    )
    assert spec["permissionDecision"] == "deny"
    assert _health(live)["workflows"]["counters"]["steps_run"] == 0
