"""Gate 4 live fixture: the real Claude Code hook command against a real proxy.

A real ``create_app`` proxy (wrap rollout defaults) is served by uvicorn on a
loopback port; the hook command wrap installs (``python -m headroom.host_hook
claude --proxy <url>``) runs as a subprocess with Claude Code's PostToolUse /
PreToolUse JSON on stdin, exactly as Claude Code invokes it. The bytes it
prints are what Claude Code puts in context instead of the tool output, so
"bytes eliminated before host context" is measured on that stdout. Failure
paths must print nothing and exit 0 (the host proceeds unchanged).
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

import pytest

from headroom import paths
from headroom.agent import registry
from headroom.proxy.server import ProxyConfig, create_app
from headroom.proxy.turn_hooks import clear_turn_hooks
from headroom.rollout import apply_wrap_feature_defaults, resolve_rollout
from tests.agent.test_tool_admission import file_list, pytest_output

uvicorn = pytest.importorskip("uvicorn")

TOKEN = "live-hook-session-token"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def live_proxy(tmp_path, monkeypatch):
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
    port = _free_port()
    project = tmp_path / "project"
    project.mkdir()
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
        yield {"url": f"http://127.0.0.1:{port}", "app": app, "project": project}
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        clear_turn_hooks()
        registry.clear_registry()
        paths.set_process_stateless(False)


def _hook_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("HEADROOM_HOOK")}
    env["ANTHROPIC_CUSTOM_HEADERS"] = f"X-Headroom-Session-Token: {TOKEN}"
    env.pop("ANTHROPIC_BASE_URL", None)
    env.update(extra)
    return env


def _run_hook(payload: dict, *, proxy: str | None, env: dict[str, str] | None = None):
    from headroom.cli.wrap import _claude_tool_hook_command

    args = [sys.executable, "-m", "headroom.host_hook", "claude"]
    if proxy:
        args += ["--proxy", proxy]
    # The installed command line is exactly this argv (quoted for the shell).
    assert "headroom.host_hook" in _claude_tool_hook_command(proxy)
    return subprocess.run(
        args,
        input=json.dumps(payload).encode("utf-8"),
        capture_output=True,
        env=env or _hook_env(),
        timeout=60,
        cwd=str(Path(__file__).resolve().parents[2]),
    )


def _post_payload(project: Path, tool: str, tool_input: dict, response, tool_use_id: str) -> dict:
    return {
        "session_id": "live-conv-1",
        "transcript_path": str(project / "t.jsonl"),
        "cwd": str(project),
        "permission_mode": "default",
        "hook_event_name": "PostToolUse",
        "tool_name": tool,
        "tool_input": tool_input,
        "tool_response": response,
        "tool_use_id": tool_use_id,
    }


def test_live_post_tool_hook_eliminates_bytes_before_host_context(live_proxy):
    project = live_proxy["project"]
    original = pytest_output(("tests/test_parser.py::test_unicode_escape",))
    response = {"stdout": original, "stderr": "", "interrupted": False, "isImage": False}
    proc = _run_hook(
        _post_payload(project, "Bash", {"command": "pytest -q"}, response, "toolu_live_1"),
        proxy=live_proxy["url"],
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    spec = out["hookSpecificOutput"]
    assert spec["hookEventName"] == "PostToolUse"
    updated = spec["updatedToolOutput"]
    # Same output shape as Claude Code's Bash tool; only stdout changed.
    assert set(updated) == set(response) and updated["interrupted"] is False
    host_bytes = len(json.dumps(updated))
    assert host_bytes < len(json.dumps(response)) / 3
    assert "FAILED tests/test_parser.py::test_unicode_escape" in updated["stdout"]
    assert "1 failed, 299 passed" in updated["stdout"]
    # Exact recovery: the marker resolves to the byte-exact original through
    # the proxy's own retrieval endpoint.
    import re

    import httpx

    key = re.search(r"Retrieve more: hash=([a-f0-9]{12,24})", updated["stdout"]).group(1)
    got = httpx.post(f"{live_proxy['url']}/v1/retrieve", json={"hash": key}, timeout=10).json()
    assert got["original_content"] == original
    health = httpx.get(f"{live_proxy['url']}/health", timeout=10).json()["tool_admission"]
    assert health["enabled"] and health["host"]["counters"]["rewritten"] >= 1


def test_live_glob_partial_and_pre_tool_observation(live_proxy):
    project = live_proxy["project"]
    files = file_list().split("\n")
    response = {"filenames": files, "durationMs": 4, "numFiles": len(files), "truncated": True}
    proc = _run_hook(
        _post_payload(project, "Glob", {"pattern": "**/*.py"}, response, "toolu_live_2"),
        proxy=live_proxy["url"],
    )
    assert proc.returncode == 0, proc.stderr
    updated = json.loads(proc.stdout)["hookSpecificOutput"]["updatedToolOutput"]
    assert updated["truncated"] is True and updated["numFiles"] == len(files)
    assert any("PARTIAL" in line for line in updated["filenames"])  # never "all files"
    (project / "a.py").write_text("x = 1\n", encoding="utf-8")
    pre = {
        "session_id": "live-conv-1",
        "cwd": str(project),
        "hook_event_name": "PreToolUse",
        "tool_name": "Edit",
        "tool_input": {"file_path": str(project / "a.py"), "old_string": "x", "new_string": "y"},
        "tool_use_id": "toolu_live_3",
    }
    proc = _run_hook(pre, proxy=live_proxy["url"])
    # Observation only: no permission decision, no output, exit 0.
    assert proc.returncode == 0 and proc.stdout == b""
    # An edit of a missing file gets compact advice, never a permission decision.
    missing = {**pre, "tool_use_id": "toolu_live_3b"}
    missing["tool_input"] = {**pre["tool_input"], "file_path": str(project / "nope.py")}
    proc = _run_hook(missing, proxy=live_proxy["url"])
    spec = json.loads(proc.stdout)["hookSpecificOutput"]
    assert proc.returncode == 0 and "permissionDecision" not in spec
    assert "contract.input_missing" in spec["additionalContext"]


def test_live_hook_fails_open(live_proxy, tmp_path):
    project = live_proxy["project"]
    response = {"stdout": pytest_output(), "stderr": "", "interrupted": False}
    payload = _post_payload(project, "Bash", {"command": "pytest -q"}, response, "toolu_live_4")
    dead = f"http://127.0.0.1:{_free_port()}"
    for proc in (
        _run_hook(payload, proxy=dead),  # proxy down
        _run_hook(payload, proxy="http://example.com:80"),  # non-loopback: never sent
        _run_hook(payload, proxy=None),  # no proxy configured
    ):
        assert proc.returncode == 0 and proc.stdout == b""
    bad = subprocess.run(
        [sys.executable, "-m", "headroom.host_hook", "claude", "--proxy", live_proxy["url"]],
        input=b"{not json",
        capture_output=True,
        env=_hook_env(),
        timeout=60,
    )
    assert bad.returncode == 0 and bad.stdout == b""
    # Without the wrap session token the hook still never fails the host.
    env = _hook_env(ANTHROPIC_CUSTOM_HEADERS="")
    proc = _run_hook({**payload, "tool_use_id": "toolu_live_5"}, proxy=live_proxy["url"], env=env)
    assert proc.returncode == 0
