"""Gate 7B: safe learned Workflow Macro Compiler (#20).

A real AgentRuntime with the wrap-default built-in components (TaskState,
evidence, change scope, tool contracts, verification planner, macros) over a
real Git + pytest fixture project. Conversations are fed through the normal
normalization path; macros are observed from closed tool segments, promoted,
ticketed at the host pre-tool boundary and executed as direct argv.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from headroom import _subprocess
from headroom.agent import enforcement, registry
from headroom.agent import workflow_macros as wm
from headroom.agent.builtin import install_builtin_components
from headroom.agent.exec_runner import run_argv
from headroom.agent.identity import resolve_identity
from headroom.agent.normalize import normalize_messages
from headroom.agent.runtime import AgentRuntime
from headroom.agent.store import MemoryAgentStore
from headroom.agent.tool_catalog import REGISTRY
from headroom.agent.workflow_macros import (
    MacroStatus,
    Safety,
    extract_segments,
    normalize_call,
)
from headroom.rollout import apply_wrap_feature_defaults, resolve_rollout

PY = sys.executable.replace("\\", "/")


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "wfproj"
    files = {
        "pytest.ini": "[pytest]\n",
        ".gitignore": ".pytest_cache/\n__pycache__/\n",
        "src/__init__.py": "",
        "src/calc.py": "def add(a, b):\n    return a + b\n",
        "tests/__init__.py": "",
        "tests/test_calc.py": "from src.calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        "tests/test_slow.py": "import time\n\n\ndef test_slow():\n    time.sleep(30)\n",
        "tests/test_writer.py": (
            "from pathlib import Path\n\n\ndef test_writes():\n"
            "    Path(__file__).with_name('touched.txt').write_text('x')\n"
        ),
    }
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "g7@example.invalid")
    _git(root, "config", "user.name", "g7")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")
    return root


class Env:
    def __init__(self, root: Path, token: str = "tok-7b") -> None:
        registry.clear_registry()
        enforcement.reset()
        self.root = root
        self.runtime = AgentRuntime(MemoryAgentStore())
        config = SimpleNamespace(rollout=resolve_rollout(apply_wrap_feature_defaults({})))
        self.teardown = install_builtin_components(self.runtime, config)
        self.identity = resolve_identity(session_token=token, registered_cwd=str(root))
        self.macros: wm.WorkflowMacroRegistry = self.runtime.services["workflow_macros"]
        self.planner = self.runtime.services["test_impact"]
        self.history: list[dict] = []

    def close(self) -> None:
        self.teardown()
        self.runtime.stop()
        registry.clear_registry()
        enforcement.reset()

    def feed(self, messages: list[dict], identity=None) -> None:
        ident = identity or self.identity
        turn = normalize_messages("anthropic", messages, ident)
        self.planner.note_identity(ident)
        self.runtime.submit(turn.envelopes)
        assert self.runtime.flush(10)
        self.macros.observe(ident, extract_segments("anthropic", messages))
        assert self.runtime.flush(10)


@pytest.fixture
def env(project: Path):
    e = Env(project)
    yield e
    e.close()


def _round(root: Path, i: int, *, tag: str = "r", steps=None, fail: bool = False) -> list[dict]:
    """One user ask + one tool call per assistant turn + a closing answer."""
    steps = steps or [
        (
            "Bash",
            {"command": "git status"},
            "On branch main\nnothing to commit, working tree clean",
        ),
        ("Read", {"file_path": str(root / "src" / "calc.py")}, "     1\tdef add(a, b):"),
        (
            "Bash",
            {"command": f"{PY} -m pytest -q tests/test_calc.py"},
            "1 failed in 0.02s" if fail else "1 passed in 0.02s",
        ),
    ]
    out: list[dict] = [
        {"role": "user", "content": f"Round {i}: check the repo and run the calc tests"}
    ]
    for j, (tool, args, result) in enumerate(steps):
        call_id = f"toolu_{tag}{i}_{j}"
        out.append(
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": call_id, "name": tool, "input": args}],
            }
        )
        block = {"type": "tool_result", "tool_use_id": call_id, "content": result}
        if fail and j == len(steps) - 1:
            block["is_error"] = True
        out.append({"role": "user", "content": [block]})
    out.append({"role": "assistant", "content": [{"type": "text", "text": "Checked."}]})
    return out


def _conversation(root: Path, rounds: int, **kw) -> list[dict]:
    messages: list[dict] = []
    for i in range(rounds):
        messages += _round(root, i, **kw)
    messages.append({"role": "user", "content": "Thanks. Keep going."})
    return messages


def _active(env: Env) -> list[wm.Macro]:
    return env.macros.active(env.identity.workspace_id)


def _pretool(env: Env, macro_id: str, inputs: dict | None = None, call_id: str = "toolu_wf"):
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": wm.WORKFLOW_MCP_TOOL,
        "tool_input": {"macro_id": macro_id, "typed_inputs": inputs or {}},
        "tool_use_id": call_id,
    }
    return env.macros.pretool(payload, identity=env.identity)


# ---------------------------------------------------------------------------
# Normalization: typed families + slots, nothing untrusted generalized
# ---------------------------------------------------------------------------


def test_normalization_families_slots_and_safety(project: Path):
    root = str(project)
    read = normalize_call("Read", {"file_path": str(project / "src" / "calc.py")}, root)
    assert (read.template.family, read.template.safety, read.values) == (
        "read.file",
        "read_only",
        {"path": "src/calc.py"},
    )
    status = normalize_call("Bash", {"command": "git status"}, root)
    assert status.template.family == "git.status" and status.template.safety == "read_only"
    test = normalize_call("Bash", {"command": f"{PY} -m pytest -q tests/test_calc.py"}, root)
    assert test.template.kind == "verify" and test.template.argv[-1] == "{tests}"
    assert test.values == {"tests": ["tests/test_calc.py"]}
    cases = {
        "curl https://example.com/x": Safety.NETWORK,
        "git push origin main": Safety.NETWORK,
        "deploy --api-key=abc123": Safety.CREDENTIAL,
        "rm -rf build": Safety.DELETION,
        "pip install requests": Safety.INSTALL,
        "kubectl apply -f k.yaml": Safety.DEPLOY,
        "sudo ls": Safety.ELEVATION,
        "echo hi > notes.txt": Safety.MUTATING,
        "git commit -m x": Safety.MUTATING,
        "ruff check --fix .": Safety.MUTATING,
        "ls | head": Safety.UNSUPPORTED,
        "echo $(whoami)": Safety.UNSUPPORTED,
    }
    for command, safety in cases.items():
        step = normalize_call("Bash", {"command": command}, root)
        assert step.template.safety == safety.value, command
        if safety not in (Safety.READ_ONLY, Safety.VERIFICATION):
            assert step.template.argv == () and step.values == {}, command  # never generalized
    assert (
        normalize_call("PowerShell", {"command": "Remove-Item x"}, root).template.safety
        == "deletion"
    )
    assert (
        normalize_call(
            "Edit", {"file_path": "a", "old_string": "a", "new_string": "b"}, root
        ).template.safety
        == "mutating"
    )
    assert normalize_call("WebFetch", {"url": "https://x"}, root).template.safety == "network"


def test_tool_registry_owns_the_stable_schema():
    wm.register_local_tool()  # what the built-in installer does when the feature is on
    tool = REGISTRY.get(wm.WORKFLOW_MCP_TOOL)
    assert tool is not None and tool.resident and tool.schema is wm.WORKFLOW_TOOL_SCHEMA
    assert REGISTRY.is_resident(wm.WORKFLOW_TOOL)
    assert tool.schema["input_schema"]["required"] == ["macro_id"]
    assert set(tool.schema["input_schema"]["properties"]) == {"macro_id", "typed_inputs", "dry_run"}


# ---------------------------------------------------------------------------
# Promotion rules
# ---------------------------------------------------------------------------


def test_three_clean_observations_promote_one_safe_workflow(env: Env):
    env.feed(_conversation(env.root, 2))
    assert _active(env) == []
    candidates = env.macros.macros(env.identity.workspace_id)
    assert len(candidates) == 1 and candidates[0].status == MacroStatus.CANDIDATE.value
    env.feed(_conversation(env.root, 3))  # same history replayed + one more round
    active = _active(env)
    assert len(active) == 1
    macro = active[0]
    stats = macro.stats()
    assert stats["occurrences"] == 3 and stats["successes"] == 3  # replay not double-counted
    assert stats["turns_saved_estimate"] == 2 and macro.safety == "verification"
    assert macro.macro_id == wm.macro_id_for(env.identity.workspace_id, macro.steps)
    assert [s.family for s in macro.steps] == ["git.status", "read.file", "test.pytest"]


def test_failures_parallel_calls_and_sessions_gate_promotion(project: Path):
    e = Env(project)
    try:
        # 3 clean + 1 failing occurrence: 75% < 90% success → never promoted.
        rounds = (
            _round(project, 0)
            + _round(project, 1)
            + _round(project, 2, fail=True)
            + _round(project, 3)
        )
        e.feed(rounds + [{"role": "user", "content": "next"}])
        macro = e.macros.macros(e.identity.workspace_id)[0]
        assert macro.stats()["occurrences"] == 4 and macro.status == MacroStatus.CANDIDATE.value
    finally:
        e.close()
    e = Env(project, token="tok-par")
    try:
        # Parallel calls in one assistant turn save no model turns.
        msgs: list[dict] = []
        for i in range(3):
            msgs.append({"role": "user", "content": f"Round {i}: status and read"})
            msgs.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": f"p{i}a",
                            "name": "Bash",
                            "input": {"command": "git status"},
                        },
                        {
                            "type": "tool_use",
                            "id": f"p{i}b",
                            "name": "Read",
                            "input": {"file_path": str(project / "src" / "calc.py")},
                        },
                    ],
                }
            )
            msgs.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": f"p{i}a", "content": "clean"},
                        {"type": "tool_result", "tool_use_id": f"p{i}b", "content": "1\tx"},
                    ],
                }
            )
        e.feed(msgs + [{"role": "user", "content": "next"}])
        assert _active(e) == []
        assert e.macros.macros(e.identity.workspace_id)[0].stats()["turns_saved_estimate"] == 0
    finally:
        e.close()
    # Two sessions also satisfy the repeat rule (2 + 1 clean occurrences).
    a = Env(project, token="tok-s1")
    try:
        a.feed(_conversation(project, 2, tag="a"))
        other = resolve_identity(session_token="tok-s2", registered_cwd=str(project))
        a.feed(_conversation(project, 1, tag="b"), identity=other)
        assert len(_active(a)) == 1 and _active(a)[0].stats()["sessions"] == 2
    finally:
        a.close()


def test_mutating_workflows_never_auto_promote_or_execute(env: Env):
    steps = [
        (
            "Edit",
            {
                "file_path": str(env.root / "src" / "calc.py"),
                "old_string": "a + b",
                "new_string": "b + a",
            },
            "ok",
        ),
        ("Bash", {"command": f"{PY} -m pytest -q tests/test_calc.py"}, "1 passed"),
        ("Bash", {"command": "git status"}, "modified: src/calc.py"),
    ]
    env.feed(_conversation(env.root, 3, steps=steps))
    macro = env.macros.macros(env.identity.workspace_id)[0]
    assert (
        macro.status == MacroStatus.MANUAL_ONLY.value
        and macro.reason == "never_auto_promote:mutating"
    )
    assert _active(env) == []
    assert env.macros.handle(env.identity, {"macro_id": macro.macro_id})["status"] == "unavailable"
    with pytest.raises(wm.WorkflowError, match="never auto-execute"):
        env.macros.compile(env.identity, macro, {})
    assert "Edit" not in json.dumps(macro.steps[0].argv)  # no argv stored for the write


def test_caps_32_active_8_steps(env: Env):
    root = env.root
    for n in range(33):
        steps = [
            ("Bash", {"command": f"git log -{n + 1}"}, "commit abc"),
            ("Read", {"file_path": str(root / "src" / "calc.py")}, "1\tx"),
            ("Bash", {"command": "git status"}, "clean"),
        ]
        env.feed(_conversation(root, 3, steps=steps, tag=f"c{n}_"))
    active = _active(env)
    assert len(active) == wm.MAX_ACTIVE_PER_WORKSPACE == 32
    assert env.macros.counters["retired_cap"] == 1
    long_steps = [("Bash", {"command": "git status"}, "clean")] * 9
    before = env.macros.counters["segments_skipped_unsupported"]
    env.feed(_conversation(root, 3, steps=long_steps, tag="long"))
    assert env.macros.counters["segments_skipped_unsupported"] >= before + 3
    assert wm.MAX_STEPS == 8 and wm.MAX_MACRO_SECONDS == 120.0


# ---------------------------------------------------------------------------
# Execution: ticket, argv, plan consumption, stale/violation cancellation
# ---------------------------------------------------------------------------


def _promote(env: Env) -> wm.Macro:
    env.feed(_conversation(env.root, 3))
    (macro,) = _active(env)
    return macro


def test_execution_needs_the_host_ticket_and_runs_shell_free_argv(env: Env, monkeypatch):
    macro = _promote(env)
    # Without the pre-tool hook's ticket nothing runs; ordinary steps are offered.
    fallback = env.macros.handle(env.identity, {"macro_id": macro.macro_id})
    assert fallback["status"] == "fallback" and fallback["fallback"]["run_with_normal_tools"]
    dry = env.macros.handle(env.identity, {"macro_id": macro.macro_id, "dry_run": True})
    assert dry["status"] == "dry_run" and len(dry["steps"]) == 3
    seen: list[tuple] = []
    real = _subprocess.Popen

    def spy(*args, **kwargs):
        seen.append((args[0], kwargs.get("shell")))
        return real(*args, **kwargs)

    monkeypatch.setattr(_subprocess, "Popen", spy)
    assert _pretool(env, macro.macro_id)["output"] is None  # ticket issued, nothing printed
    result = env.macros.handle(env.identity, {"macro_id": macro.macro_id})
    assert result["status"] == "completed", result
    assert [s["status"] for s in result["steps"]] == ["ok", "ok", "ok"]
    assert result["steps"][2]["tests"]["passed"] == 1
    assert result["estimated_turns_avoided"] == 2
    assert seen and all(isinstance(argv, list) and shell is False for argv, shell in seen)
    # The ticket is single use.
    assert env.macros.handle(env.identity, {"macro_id": macro.macro_id})["status"] == "fallback"
    # Each step was ingested once through the normal event path → evidence.
    evidence = env.runtime.services["evidence"]
    facts = evidence.facts(env.identity.workspace_id, env.identity.session_id)
    assert any(f.predicate == "test.status" and f.polarity == "positive" for f in facts)
    assert env.planner.counters["runs_recorded"] >= 1


def test_verification_steps_consume_the_current_plan(env: Env):
    macro = _promote(env)
    (env.root / "tests" / "test_more.py").write_text(
        "from src.calc import add\n\n\ndef test_more():\n    assert add(2, 2) == 4\n",
        encoding="utf-8",
    )
    (env.root / "src" / "calc.py").write_text(
        "def add(a, b):\n    return b + a\n", encoding="utf-8"
    )
    _pretool(env, macro.macro_id)
    result = env.macros.handle(env.identity, {"macro_id": macro.macro_id})
    assert result["status"] == "completed", result
    test_step = result["steps"][2]
    assert "verification plan" in test_step["note"] and "tier 1" in test_step["note"]
    assert "tests/test_calc.py::test_add" in test_step["run"]
    assert "tests/test_more.py::test_more" in test_step["run"]
    assert result["verification"]["checks"].get("tier1") == "passed"


def test_stale_scope_or_task_cancels_before_execution(env: Env):
    macro = _promote(env)
    assert _pretool(env, macro.macro_id)["output"] is None
    # The user narrows the task after the host approved the call.
    env.feed(
        _conversation(env.root, 3)
        + [
            {"role": "assistant", "content": "ok"},
            {
                "role": "user",
                "content": "Actually, only touch src/calc.py and do not modify tests/.",
            },
        ]
    )
    before = env.macros.counters["steps_run"]
    result = env.macros.handle(env.identity, {"macro_id": macro.macro_id})
    assert result["status"] == "cancelled" and result["reason"] == "workflow.stale_revision"
    assert env.macros.counters["steps_run"] == before  # nothing ran
    assert result["fallback"]["run_with_normal_tools"]


def test_mid_run_user_change_cancels_the_next_step(env: Env, monkeypatch):
    macro = _promote(env)
    _pretool(env, macro.macro_id)
    real = wm.WorkflowMacroRegistry._run_step
    calls = {"n": 0}

    def step_then_user_turn(self, identity, step, deadline_s, cancel):
        out = real(self, identity, step, deadline_s, cancel)
        calls["n"] += 1
        if calls["n"] == 1:
            env.feed(
                _conversation(env.root, 3)
                + [
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "Stop, also check the docs folder first."},
                ]
            )
        return out

    monkeypatch.setattr(wm.WorkflowMacroRegistry, "_run_step", step_then_user_turn)
    result = env.macros.handle(env.identity, {"macro_id": macro.macro_id})
    assert result["status"] == "cancelled" and result["reason"] == "workflow.stale_revision"
    assert [s["status"] for s in result["steps"]] == ["ok", "not_run"]
    assert calls["n"] == 1


def test_step_violation_cancels_before_any_step(env: Env):
    macro = _promote(env)
    _pretool(env, macro.macro_id)
    (env.root / "src" / "calc.py").unlink()  # the Read step's input no longer exists
    before = env.macros.counters["steps_run"]
    result = env.macros.handle(env.identity, {"macro_id": macro.macro_id})
    assert result["status"] == "denied" and result["reason"] == "workflow.input_missing"
    assert env.macros.counters["steps_run"] == before
    # The pre-tool boundary denies invalid inputs outright (shell metacharacters).
    deny = _pretool(env, macro.macro_id, {"tests": ["tests/test_calc.py; rm -rf /"]})
    spec = deny["output"]["hookSpecificOutput"]
    assert (
        spec["permissionDecision"] == "deny"
        and "workflow.input" in spec["permissionDecisionReason"]
    )


def test_environment_change_invalidates_and_side_effects_disable(env: Env):
    macro = _promote(env)
    (env.root / "pytest.ini").write_text("[pytest]\nmarkers = slow\n", encoding="utf-8")
    deny = _pretool(env, macro.macro_id)
    assert (
        "workflow.invalidated.environment_changed"
        in deny["output"]["hookSpecificOutput"]["permissionDecisionReason"]
    )
    assert env.macros.get(env.identity.workspace_id, macro.macro_id).status == "invalidated"
    _git(env.root, "checkout", "--", "pytest.ini")
    # A "verification" macro whose test writes into the worktree is disabled.
    steps = [
        ("Bash", {"command": "git status"}, "clean"),
        ("Read", {"file_path": str(env.root / "src" / "calc.py")}, "1\tx"),
        ("Bash", {"command": f"{PY} -m pytest -q tests/test_writer.py"}, "1 passed"),
    ]
    env.feed(_conversation(env.root, 3, steps=steps, tag="w"))
    # Same template as the invalidated macro: re-observed in a new epoch, re-promoted.
    writer = next(m for m in _active(env) if "tests/test_writer.py" in json.dumps(m.defaults()))
    assert writer.macro_id == macro.macro_id and writer.version == macro.version
    _pretool(env, writer.macro_id)
    result = env.macros.handle(env.identity, {"macro_id": writer.macro_id})
    assert result["steps"][2]["status"] == "side_effect"
    assert env.macros.get(env.identity.workspace_id, writer.macro_id).status == "invalidated"


def test_step_deadline_kills_the_process_tree(env: Env, monkeypatch):
    steps = [
        ("Bash", {"command": "git status"}, "clean"),
        ("Read", {"file_path": str(env.root / "src" / "calc.py")}, "1\tx"),
        ("Bash", {"command": f"{PY} -m pytest -q tests/test_slow.py"}, "1 passed"),
    ]
    env.feed(_conversation(env.root, 3, steps=steps, tag="s"))
    (macro,) = [m for m in _active(env) if "tests/test_slow.py" in json.dumps(m.defaults())]
    monkeypatch.setattr(wm, "MAX_STEP_SECONDS", 3.0)
    _pretool(env, macro.macro_id)
    t0 = time.monotonic()
    result = env.macros.handle(env.identity, {"macro_id": macro.macro_id})
    assert time.monotonic() - t0 < 25
    assert result["status"] == "timeout" and result["steps"][2]["status"] == "timeout"
    assert result["steps"][2].get("process_tree_killed") is True


def test_process_group_cleanup_kills_grandchildren(tmp_path: Path):
    pid_file = tmp_path / "child.pid"
    script = tmp_path / "spawner.py"  # argv tokens never carry newlines
    script.write_text(
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(120)\n",
        encoding="utf-8",
    )
    run = run_argv([sys.executable, str(script)], cwd=str(tmp_path), timeout_s=4)
    assert run.timed_out and run.tree_killed and run.exit_code is None
    child = int(pid_file.read_text())
    deadline = time.time() + 10
    while _running(child) and time.time() < deadline:
        time.sleep(0.2)
    assert not _running(child), "grandchild survived the process-group kill"


def _running(pid: int) -> bool:
    """A process table lookup (an exited PID's kernel object may linger on Windows)."""
    if sys.platform == "win32":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        return f'"{pid}"' in out
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as handle:
            return handle.read().rpartition(")")[2].split()[0] != "Z"
    except OSError:
        return _subprocess.pid_alive(pid)


def test_cancellation_stops_a_running_step(tmp_path: Path):
    cancel = threading.Event()
    timer = threading.Timer(1.0, cancel.set)
    timer.start()
    run = run_argv(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        cwd=str(tmp_path),
        timeout_s=30,
        cancel=cancel,
    )
    timer.cancel()
    assert run.cancelled and run.duration_s < 15 and run.exit_code is None


def test_advertisement_needs_tool_and_hook(env: Env):
    macro = _promote(env)
    ctx = SimpleNamespace(tools=[{"name": "Bash", "input_schema": {"type": "object"}}])
    assert env.macros.advertisement_lines(env.identity, ctx) == []  # tool missing → fallback
    ctx.tools.append(
        {"name": wm.WORKFLOW_MCP_TOOL, "input_schema": wm.WORKFLOW_TOOL_SCHEMA["input_schema"]}
    )
    assert env.macros.advertisement_lines(env.identity, ctx) == []  # no host hook seen yet
    env.macros.note_hook(env.identity.session_id)
    lines = [text for _, text in env.macros.advertisement_lines(env.identity, ctx)]
    assert any(macro.macro_id in line and "saves ~2 tool turns" in line for line in lines)
    assert env.macros.counters["fallback_tool_missing"] >= 1
