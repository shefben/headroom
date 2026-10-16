"""Gate 6B: Tool Contract + Argument Validator (#19)."""

from __future__ import annotations

from pathlib import Path

import pytest

from headroom.agent.change_scope import ChangeScopeFirewall, Enforcement, EnforcementPolicy
from headroom.agent.identity import resolve_identity
from headroom.agent.runtime import AgentRuntime
from headroom.agent.store import MemoryAgentStore
from headroom.agent.task_state import StateAtom, TaskSnapshot, TaskState
from headroom.agent.tool_contracts import (
    PROMOTION_THRESHOLD,
    RECOVERY_TTL_S,
    DecisionKind,
    Family,
    ToolContractValidator,
    check_ref_format,
    family_of,
    validate_schema,
)

HOST = "host_pretool"


class _Compiler:
    def __init__(self) -> None:
        self.snapshots: dict = {}

    def snapshot(self, ws, ss):
        return self.snapshots.get((ws, ss))

    def request_warm(self, ws, ss):
        return None


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def setup(tmp_path: Path):
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "parser.py").write_text("def parse():\n    return 1\n", encoding="utf-8")
    (root / "tests" / "test_parser.py").write_text("def test_p():\n    pass\n", encoding="utf-8")
    runtime = AgentRuntime(MemoryAgentStore())
    compiler = _Compiler()
    runtime.services["task_state"] = compiler
    identity = resolve_identity(session_token="tok-6b", registered_cwd=str(root))
    policy = EnforcementPolicy()
    firewall = ChangeScopeFirewall(runtime, policy=policy)
    clock = _Clock()
    available = {"git", "python", "pytest", "ls", "grep", "cargo", "npm"}
    validator = ToolContractValidator(
        runtime,
        firewall=firewall,
        policy=policy,
        clock=clock,
        which=lambda name: f"/usr/bin/{name}" if name in available else None,
    )
    validator.recoveries._clock = clock

    def instruct(*texts: str, revision: int = 1) -> None:
        atoms = tuple(
            StateAtom(f"at_{i}", "goal", t, "user", 0.95, "ACTIVE", updated_at=float(i))
            for i, t in enumerate(texts)
        )
        state = TaskState(
            id="tk_b",
            workspace_id=identity.workspace_id,
            session_id=identity.session_id,
            revision=revision,
            atoms=atoms,
        )
        compiler.snapshots[(identity.workspace_id, identity.session_id)] = TaskSnapshot(state=state)

    instruct("Fix parse() in src/parser.py so tests pass")
    yield {
        "root": root,
        "validator": validator,
        "identity": identity,
        "instruct": instruct,
        "clock": clock,
        "runtime": runtime,
    }
    runtime.stop(timeout=1.0)


def _eval(s, tool, args, **kw):
    kw.setdefault("cwd", str(s["root"]))
    kw.setdefault("enforcement", HOST)
    return s["validator"].evaluate(tool, args, identity=s["identity"], **kw)


@pytest.mark.parametrize(
    ("tool", "args", "family"),
    [
        ("Read", {"file_path": "a"}, Family.READ),
        ("read_file", {"path": "a"}, Family.READ),  # Codex / OpenCode
        ("Edit", {}, Family.EDIT),
        ("apply_patch", {}, Family.EDIT),
        ("write", {}, Family.EDIT),  # OpenCode
        ("Bash", {"command": "ls"}, Family.EXEC),
        ("shell", {"command": ["bash", "-lc", "ls"]}, Family.EXEC),  # Codex argv
        ("Bash", {"command": "git status"}, Family.GIT),
        ("Bash", {"command": "cd x && python -m pytest -q"}, Family.TEST),
        ("Grep", {"pattern": "x"}, Family.SEARCH),
        ("glob", {"pattern": "x"}, Family.SEARCH),
        ("WebFetch", {"url": "https://x"}, Family.HTTP),
        ("mcp__weird__thing", {}, Family.UNKNOWN),
    ],
)
def test_family_mapping(tool, args, family):
    assert family_of(tool, args) is family


def test_unknown_tools_observed_never_certified(setup):
    decision = _eval(setup, "mcp__x__y", {"a": 1})
    assert decision.kind is DecisionKind.ALLOW and decision.reason_code == "contract.unknown_tool"
    assert decision.contract.certified is False
    assert setup["validator"].counters["unknown_tools_observed"] == 1


def test_legitimate_calls_allow_and_carry_revisions(setup):
    edit = _eval(
        setup,
        "Edit",
        {"file_path": str(setup["root"] / "src/parser.py"), "old_string": "1", "new_string": "2"},
    )
    assert edit.kind is DecisionKind.ALLOW and edit.reason_code == "contract.ok"
    assert edit.task_revision == 1 and edit.scope_revision >= 1
    assert edit.contract.tool_family is Family.EDIT and edit.message() == ""
    read = _eval(setup, "Read", {"file_path": "src/parser.py"})
    assert read.kind is DecisionKind.ALLOW
    test = _eval(setup, "Bash", {"command": "python -m pytest tests/test_parser.py -q"})
    assert test.kind is DecisionKind.ALLOW, test.message()
    git = _eval(setup, "Bash", {"command": "git checkout -b fix/parse-1 && git status"})
    assert git.kind is DecisionKind.ALLOW, git.message()


def test_schema_types_required_enum_exclusive(setup):
    missing = _eval(setup, "Edit", {"file_path": str(setup["root"] / "src/parser.py")})
    assert (
        missing.kind is DecisionKind.WARN and missing.reason_code == "contract.missing_required_arg"
    )
    bad_type = _eval(setup, "Read", {"file_path": "src/parser.py", "limit": "ten"})
    assert bad_type.reason_code == "contract.arg_type"
    enum = _eval(setup, "Grep", {"pattern": "x", "output_mode": "lines"})
    assert enum.reason_code == "contract.arg_enum"
    excl = _eval(setup, "Grep", {"pattern": "x", "glob": "*.py", "type": "py"})
    assert excl.reason_code == "contract.mutually_exclusive"
    noop = _eval(
        setup,
        "Edit",
        {"file_path": str(setup["root"] / "src/parser.py"), "old_string": "a", "new_string": "a"},
    )
    assert noop.reason_code == "contract.noop_edit"
    # The client's own JSON schema wins when the proxy has it.
    schema = {
        "type": "object",
        "required": ["q"],
        "properties": {"q": {"type": "string"}, "n": {"type": "integer"}},
    }
    assert [f.code for f in validate_schema({"n": True}, schema)] == [
        "contract.missing_required_arg",
        "contract.arg_type",
    ]


def test_typo_path_repair_is_suggested_never_applied(setup):
    args = {"file_path": str(setup["root"] / "src" / "parsr.py")}
    decision = _eval(setup, "Read", args)
    assert (
        decision.kind is DecisionKind.REPAIR_SUGGEST
        and decision.reason_code == "contract.input_missing"
    )
    assert decision.repairs and decision.repairs[0].endswith("src/parser.py")
    assert "suggested (not run)" in decision.message()
    assert decision.validated_arguments == args  # the call itself is unchanged


def test_cwd_command_and_platform_checks(setup):
    cwd = _eval(setup, "Bash", {"command": "cd nowhere && ls"})
    assert cwd.reason_code == "contract.cwd_missing"
    typo = _eval(setup, "Bash", {"command": "gti status"})
    assert typo.kind is DecisionKind.REPAIR_SUGGEST and typo.repairs[0] == "git status"
    missing = _eval(setup, "Bash", {"command": "frobnicate --now"})
    assert (
        missing.kind is DecisionKind.WARN and missing.reason_code == "contract.command_unavailable"
    )
    builtins = _eval(
        setup, "Bash", {"command": "export X=1 && echo $X && type ls && [ -f a ] || true"}
    )
    assert builtins.kind is DecisionKind.ALLOW, builtins.message()
    ps = _eval(setup, "PowerShell", {"command": "Get-ChildItem -Recurse | Select-Object -First 3"})
    assert ps.kind is DecisionKind.ALLOW, ps.message()
    switch = _eval(setup, "Bash", {"command": "dir /s"})
    assert "contract.platform_switch" in [f.code for f in switch.findings]
    psrm = _eval(setup, "PowerShell", {"command": "rm -rf build"})
    assert "contract.platform_switch" in [f.code for f in psrm.findings]


def test_git_ref_syntax():
    for good in ("main", "feature/x-1", "HEAD~2", "origin/main", "v1.2.3", "HEAD^", "@{u}", "a@b"):
        assert check_ref_format(good), good
    for bad in ("bad..ref", "has space", "x.lock", "-oops", "a:b", "end/", "a//b", ".hidden", "q?"):
        assert not check_ref_format(bad), bad


def test_git_and_test_invocation_findings(setup):
    ref = _eval(setup, "Bash", {"command": "git checkout 'bad..ref'"})
    assert ref.reason_code == "contract.git_ref_syntax" and ref.kind is DecisionKind.WARN
    k = _eval(setup, "Bash", {"command": "python -m pytest -k"})
    assert k.reason_code == "contract.test_invocation"
    path = _eval(setup, "Bash", {"command": "pytest tests/test_parsr.py::test_p"})
    assert "contract.test_path_missing" in [f.code for f in path.findings]
    npm = _eval(setup, "Bash", {"command": "npm test"})
    assert npm.reason_code == "contract.test_invocation"


def test_privilege_escalation_blocked_only_at_host_and_unless_asked(setup):
    esc = _eval(setup, "Bash", {"command": "sudo pip install requests"})
    assert esc.kind is DecisionKind.BLOCK and esc.reason_code == "contract.privilege_escalation"
    advisory = _eval(
        setup, "Bash", {"command": "sudo pip install requests"}, enforcement="observe_only"
    )
    assert advisory.kind is DecisionKind.WARN
    setup["instruct"]("Install the package system-wide with sudo", revision=2)
    asked = _eval(setup, "Bash", {"command": "sudo pip install requests"})
    assert asked.kind is not DecisionKind.BLOCK


def test_scope_forbidden_flows_through_contract(setup):
    setup["instruct"]("Fix src/parser.py but do not touch tests/", revision=3)
    blocked = _eval(
        setup, "Write", {"file_path": str(setup["root"] / "tests/test_parser.py"), "content": "x"}
    )
    assert blocked.kind is DecisionKind.BLOCK and blocked.reason_code == "scope.user_forbidden"
    assert blocked.scope and blocked.scope[0].authorization is None
    shell = _eval(setup, "Bash", {"command": "echo x > tests/test_parser.py"})
    assert shell.kind is DecisionKind.BLOCK


def test_policy_observe_and_protect(setup):
    v = setup["validator"]
    v.policy = EnforcementPolicy(tool_contracts=Enforcement.OBSERVE)
    quiet = _eval(setup, "Bash", {"command": "frobnicate"})
    assert quiet.kind is DecisionKind.ALLOW and quiet.reason_code == "contract.command_unavailable"
    v.policy = EnforcementPolicy(tool_contracts=Enforcement.PROTECT)
    hard = _eval(setup, "Bash", {"command": "git checkout 'bad..ref'"})
    assert hard.kind is DecisionKind.BLOCK
    v.policy = EnforcementPolicy()


def _recover(s, failed: str, success: str, n: int, *, start: int = 0) -> None:
    v, ident = s["validator"], s["identity"]
    for i in range(start, start + n):
        v.observe(
            "Bash",
            {"command": failed},
            identity=ident,
            call_id=f"f{i}",
            output="ERROR: file or directory not found: tests/test_parsr.py",
            is_error=True,
        )
        v.observe(
            "Bash",
            {"command": success},
            identity=ident,
            call_id=f"s{i}",
            output="3 passed",
            is_error=False,
        )


def test_learned_recovery_promoted_after_three_corroborations(setup):
    _recover(
        setup,
        "python -m pytest tests/test_parsr.py -q",
        "python -m pytest tests/test_parser.py -q",
        PROMOTION_THRESHOLD - 1,
    )
    before = _eval(setup, "Bash", {"command": "python -m pytest tests/test_parsr.py -q"})
    assert "python -m pytest tests/test_parser.py -q" not in before.contract.learned_recovery
    # A replay of the same call ids is not new corroboration.
    _recover(
        setup,
        "python -m pytest tests/test_parsr.py -q",
        "python -m pytest tests/test_parser.py -q",
        PROMOTION_THRESHOLD - 1,
    )
    assert setup["validator"].recoveries.status()["promoted_live"] == 0
    _recover(
        setup,
        "python -m pytest tests/test_parsr.py -q",
        "python -m pytest tests/test_parser.py -q",
        1,
        start=PROMOTION_THRESHOLD - 1,
    )
    after = _eval(setup, "Bash", {"command": "python -m pytest tests/test_parsr.py -q"})
    assert after.kind is DecisionKind.REPAIR_SUGGEST
    assert (
        "python -m pytest tests/test_parser.py -q" in after.repairs
        and after.contract.learned_recovery
    )
    assert after.validated_arguments == {
        "command": "python -m pytest tests/test_parsr.py -q"
    }  # never rewritten


def test_learned_recovery_ttl_and_contradiction(setup):
    _recover(
        setup,
        "python -m pytest tests/test_parsr.py -q",
        "python -m pytest tests/test_parser.py -q",
        PROMOTION_THRESHOLD,
    )
    assert setup["validator"].recoveries.suggest(
        setup["identity"].workspace_id,
        "Bash",
        {"command": "python -m pytest tests/test_parsr.py -q"},
    )
    setup["clock"].now += RECOVERY_TTL_S + 1
    assert not setup["validator"].recoveries.suggest(
        setup["identity"].workspace_id,
        "Bash",
        {"command": "python -m pytest tests/test_parsr.py -q"},
    )
    _recover(
        setup,
        "python -m pytest tests/test_parsr.py -x",
        "python -m pytest tests/test_parser.py -x",
        PROMOTION_THRESHOLD,
        start=10,
    )
    ws = setup["identity"].workspace_id
    assert setup["validator"].recoveries.suggest(
        ws, "Bash", {"command": "python -m pytest tests/test_parsr.py -x"}
    )
    # Contradicting evidence: the suggested recovery itself now fails.
    setup["validator"].observe(
        "Bash",
        {"command": "python -m pytest tests/test_parser.py -x"},
        identity=setup["identity"],
        call_id="c1",
        output="Error: boom",
        is_error=True,
    )
    assert not setup["validator"].recoveries.suggest(
        ws, "Bash", {"command": "python -m pytest tests/test_parsr.py -x"}
    )
    assert setup["validator"].recoveries.counters["invalidated"] >= 1


def test_recovery_environment_fingerprint(setup, monkeypatch):
    _recover(
        setup,
        "python -m pytest tests/test_parsr.py -q",
        "python -m pytest tests/test_parser.py -q",
        PROMOTION_THRESHOLD,
    )
    ws = setup["identity"].workspace_id
    assert setup["validator"].recoveries.suggest(
        ws, "Bash", {"command": "python -m pytest tests/test_parsr.py -q"}
    )
    import headroom.agent.tool_contracts as tc

    monkeypatch.setattr(tc, "environment_fingerprint", lambda shell="posix": "other-env")
    assert not setup["validator"].recoveries.suggest(
        ws, "Bash", {"command": "python -m pytest tests/test_parsr.py -q"}
    )


class _Advisor:
    def __init__(self, pick: str | None) -> None:
        self.pick = pick
        self.calls = 0

    def request_budget(self):
        return object()

    def choose(self, family, state, instructions, options, *, budget):
        self.calls += 1
        if self.pick is None:
            return None
        from types import SimpleNamespace

        return SimpleNamespace(value=self.pick, confidence=0.9)


def test_jevk5_only_reorders_safe_repairs(setup, tmp_path):
    root = setup["root"]
    (root / "src" / "parse.py").write_text("x\n", encoding="utf-8")
    (root / "src" / "parsers.py").write_text("x\n", encoding="utf-8")
    setup["runtime"].advisor = _Advisor("r1")
    decision = _eval(setup, "Read", {"file_path": str(root / "src" / "parsr.py")})
    safe = set(decision.repairs)
    assert decision.kind is DecisionKind.REPAIR_SUGGEST and len(safe) >= 2
    assert setup["runtime"].advisor.calls == 1 and setup["validator"].counters["jevk5_ranked"] == 1
    # An invented option id can never add or authorize anything.
    setup["runtime"].advisor = _Advisor("r99")
    again = _eval(setup, "Read", {"file_path": str(root / "src" / "parsr.py")})
    assert set(again.repairs) == safe
    # JevK5 cannot override a hard scope decision.
    setup["instruct"]("Fix src/parser.py but leave src/parse.py alone", revision=5)
    setup["runtime"].advisor = _Advisor("r0")
    blocked = _eval(setup, "Write", {"file_path": str(root / "src" / "parse.py"), "content": "y"})
    assert blocked.kind is DecisionKind.BLOCK
