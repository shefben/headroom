"""Workflow host adapters (unified-21 #20): one client-independent engine.

The macro engine learns from every client's tool segments, but only a host
adapter with a genuine pre-tool permission boundary can start a workflow.
Steps learned from a host's built-in Grep/Glob re-execute through that host's
step dialect (ripgrep, same flags), and only on host versions whose semantics
were verified.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from headroom.agent import workflow_hosts as wh
from headroom.agent import workflow_macros as wm

from .test_phase7_workflow_macros import Env, _active, _conversation, project  # noqa: F401

VERIFIED_AGENT = {"agent": "claude-code_2-1-300_agent"}


@pytest.fixture
def env(project: Path):  # noqa: F811 - the shared fixture project
    e = Env(project)
    yield e
    e.close()


def _payload(macro_id: str, call_id: str = "toolu_wf", tool: str = wm.WORKFLOW_MCP_TOOL) -> dict:
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": tool,
        "tool_input": {"macro_id": macro_id, "typed_inputs": {}},
        "tool_use_id": call_id,
    }


def _search_steps(root: Path) -> list:
    return [
        ("Bash", {"command": "git status"}, "nothing to commit, working tree clean"),
        ("Grep", {"pattern": "def add", "path": str(root)}, "Found 1 file\nsrc/calc.py"),
        ("Read", {"file_path": str(root / "src" / "calc.py")}, "     1\tdef add(a, b):"),
    ]


def test_capability_table_is_honest():
    table = wh.capability_table()
    assert table["claude"]["capability"] == wh.TICKETED and table["claude"]["executes_workflows"]
    for client in ("codex", "opencode", "copilot", "cursor", "aider", "gemini"):
        assert table[client]["capability"] == wh.OBSERVE_ONLY
        assert not table[client]["executes_workflows"] and table[client]["reason"]
    assert wh.host_for_mcp_client("claude-code") is wh.HOSTS["claude"]
    assert wh.host_for_mcp_client("codex-mcp-client") is None
    assert wh.host_for_mcp_client("opencode") is None


def test_observe_only_clients_learn_but_never_execute(env: Env):
    env.feed(_conversation(env.root, 3))
    (macro,) = _active(env)  # learned from the proxy path, whoever the client is
    for client in ("codex", "opencode", "some-new-client"):
        out = env.macros.pretool(_payload(macro.macro_id), identity=env.identity, client=client)
        assert out == {
            "output": None,
            "stage": "workflow:observe_only",
            "capability": wh.OBSERVE_ONLY,
        }
    assert env.macros.counters["tickets_issued"] == 0
    result = env.macros.handle(env.identity, {"macro_id": macro.macro_id})
    assert result["status"] == "fallback" and result["fallback"]["run_with_normal_tools"]
    # An observe-only hook never makes the session look ticket-capable.
    env.macros.note_hook(env.identity.session_id, "codex")
    from types import SimpleNamespace

    ctx = SimpleNamespace(
        tools=[
            {"name": wm.WORKFLOW_MCP_TOOL, "input_schema": wm.WORKFLOW_TOOL_SCHEMA["input_schema"]}
        ]
    )
    assert env.macros.advertisement_lines(env.identity, ctx) == []


@pytest.mark.skipif(wh.search_unsupported_reason() is not None, reason="ripgrep unavailable")
def test_grep_step_runs_through_the_claude_dialect_on_a_verified_host(env: Env):
    env.feed(_conversation(env.root, 3, steps=_search_steps(env.root)))
    (macro,) = _active(env)
    grep = macro.steps[1]
    assert grep.kind == "grep" and grep.dialect == wh.CLAUDE_DIALECT
    # No host version (or an unverified one): the ripgrep step's semantics are
    # not assumed, so the host boundary refuses the call.
    for host in (None, {"agent": "claude-code_9-0-0_agent"}):
        deny = env.macros.pretool(_payload(macro.macro_id), identity=env.identity, host=host)
        assert deny["reason_code"] == "workflow.host_semantics_unverified"
        assert deny["output"]["hookSpecificOutput"]["permissionDecision"] == "deny"
    ok = env.macros.pretool(_payload(macro.macro_id), identity=env.identity, host=VERIFIED_AGENT)
    assert ok["stage"] == "workflow:ticket" and ok["output"] is None
    result = env.macros.handle(env.identity, {"macro_id": macro.macro_id})
    assert result["status"] == "completed", result
    step = result["steps"][1]
    assert step["status"] == "ok" and step["output"] == "Found 1 file\nsrc/calc.py"


def test_dialectless_search_step_is_never_compiled(env: Env):
    legacy = wm.StepTemplate(
        "grep",
        "search.grep",
        "Grep",
        (),
        (wm.SlotSpec("pattern", "rg_pattern"),),
        wm.Safety.READ_ONLY.value,
    )
    macro = wm.Macro("wf_legacy", env.identity.workspace_id, (legacy, legacy))
    with pytest.raises(wm.WorkflowError) as exc:
        env.macros.compile(env.identity, macro, {"s1.pattern": "x", "s2.pattern": "y"})
    assert exc.value.code == "workflow.step_dialect_unavailable"
