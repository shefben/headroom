"""Gate 7A: Test Impact and Verification Planner (#24).

Real fixture projects (Git repos under the pytest tmp dir) for pytest,
Cargo, CTest and Vitest/Jest: the planner selects the tests tied to the
current diff, escalates risky changes to the full suite, invalidates stale
plans, and never reports a flaky pass as green. Selected commands are run
for real where the framework tool exists locally; otherwise the run is
skipped with an explicit platform reason (never a fake pass).
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from headroom.agent.events import EventEnvelope, EventKind, Origin, make_event
from headroom.agent.evidence import extract_candidates
from headroom.agent.exec_runner import run_argv
from headroom.agent.identity import resolve_identity
from headroom.agent.task_state import (
    StateAtom,
    TaskSnapshot,
    TaskState,
    derive_transitions,
    new_task,
)
from headroom.agent.test_impact import (
    Framework,
    TestImpactPlanner,
    compute_changeset,
    parse_output,
)

REPO = Path(__file__).resolve().parents[2]
JS_RUNNER = REPO / "target" / "tmp" / "js_runner" / "node_modules"


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _repo(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "gate7@example.invalid")
    _git(root, "config", "user.name", "gate7")
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")
    return root


class _Compiler:
    def __init__(self) -> None:
        self.snapshots: dict[tuple[str, str], TaskSnapshot] = {}

    def snapshot(self, ws: str, sess: str) -> TaskSnapshot | None:
        return self.snapshots.get((ws, sess))

    def request_warm(self, ws: str, sess: str) -> None:
        return None

    def active_task_id(self, ws: str, sess: str) -> str | None:
        snap = self.snapshot(ws, sess)
        return snap.state.id if snap else None


class _Runtime:
    def __init__(self) -> None:
        self.services: dict = {"task_state": _Compiler()}
        self.submitted: list = []

    def submit(self, envelopes):  # noqa: ANN001
        self.submitted.extend(envelopes)
        return True

    def submit_task(self, fn):  # noqa: ANN001
        fn()
        return True


def _planner(root: Path, token: str = "tok7a"):
    runtime = _Runtime()
    identity = resolve_identity(session_token=token, registered_cwd=str(root))
    return TestImpactPlanner(runtime), identity, runtime


def _set_state(runtime, identity, *atoms: StateAtom, revision: int = 1) -> None:
    state = TaskState(
        id="tk_7a",
        workspace_id=identity.workspace_id,
        session_id=identity.session_id,
        revision=revision,
        atoms=tuple(atoms),
    )
    runtime.services["task_state"].snapshots[(identity.workspace_id, identity.session_id)] = (
        TaskSnapshot(state=state)
    )


PY_FILES = {
    "pytest.ini": "[pytest]\n",
    "src/__init__.py": "",
    "src/calc.py": "def add(a, b):\n    return a + b\n",
    "src/service.py": "from src.calc import add\n\n\ndef total(x):\n    return add(x, 1)\n",
    "src/auth/__init__.py": "",
    "src/auth/token.py": "def issue():\n    return 'x'\n",
    "tests/__init__.py": "",
    "tests/test_calc.py": (
        "from src.calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n\n\n"
        "class TestMore:\n    def test_zero(self):\n        assert add(0, 0) == 0\n"
    ),
    "tests/test_service.py": (
        "from src.service import total\n\n\ndef test_total():\n    assert total(1) == 2\n"
    ),
    "tests/test_unrelated.py": "def test_other():\n    assert True\n",
}


@pytest.fixture
def pyproj(tmp_path: Path) -> Path:
    return _repo(tmp_path / "py", PY_FILES)


# ---------------------------------------------------------------------------
# pytest
# ---------------------------------------------------------------------------


def test_pytest_selection_is_tied_to_the_diff_and_runs_for_real(pyproj: Path):
    (pyproj / "src" / "calc.py").write_text("def add(a, b):\n    return b + a\n", encoding="utf-8")
    planner, identity, _ = _planner(pyproj)
    plan = planner.plan(identity)
    assert plan is not None and plan.frameworks == ("pytest",)
    assert plan.changed_files == ("src/calc.py",)
    tier1 = [t.id for t in plan.tier1]
    assert tier1 == ["tests/test_calc.py::TestMore::test_zero", "tests/test_calc.py::test_add"]
    assert {t.reason for t in plan.tier1} == {"naming"}
    assert [t.id for t in plan.tier2] == ["tests/test_service.py::test_total"]
    assert "tests/test_unrelated.py::test_other" not in tier1 + [t.id for t in plan.tier2]
    assert plan.required_checks == ("tier1",)
    assert plan.estimate["kind"] == "estimate" and plan.estimate["tests_selected_tier1"] == 2
    # Deterministic: the same inputs give the same plan (and plan id).
    again, _ = TestImpactPlanner(_Runtime()), None
    assert again.plan(identity).to_dict()["tier1"] == plan.to_dict()["tier1"]
    # The selected Tier-1 command runs as direct argv (no shell) and passes.
    cmd = plan.commands_for(1)[0]
    argv = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *cmd.targets]
    run = run_argv(argv, cwd=str(pyproj), timeout_s=120)
    outcome = parse_output("pytest", run.output, run.exit_code, targets=cmd.targets)
    assert run.exit_code == 0 and outcome.passed == 2 and outcome.ok
    # --collect-only validation (only inside an authorized run) keeps real ids.
    planner._commands[identity.workspace_id] = {"pytest": f"{sys.executable} -m pytest"}
    refined = planner.collect_validate(identity, planner.plan(identity, force=True))
    assert "pytest.collected" in refined.reason_codes
    assert [t.id for t in refined.tier1] == tier1


def test_direct_import_coverage_and_failure_regressions_rank_first(pyproj: Path):
    (pyproj / "src" / "service.py").write_text(
        "from src.calc import add\n\n\ndef total(x):\n    return add(x, 2) - 1\n", encoding="utf-8"
    )
    # Existing coverage.py data with per-test contexts (read-only use).
    _write_coverage(
        pyproj, {"tests/test_unrelated.py::test_other|run": [str(pyproj / "src" / "service.py")]}
    )
    planner, identity, _ = _planner(pyproj)
    planner.history.record(
        identity.workspace_id, "pytest", "tests/test_calc.py::test_add", "older", False, 1.0
    )
    plan = planner.plan(identity)
    reasons = {t.id: t.reason for t in plan.tier1}
    assert reasons["tests/test_calc.py::test_add"] == "failure_regression"
    assert reasons["tests/test_service.py::test_total"] in ("naming", "direct_import")
    assert reasons["tests/test_unrelated.py::test_other"] == "coverage"
    assert plan.tier1[0].id == "tests/test_calc.py::test_add"  # exact regression first


def _write_coverage(root: Path, contexts: dict[str, list[str]]) -> Path:
    """Real coverage.py data (``coverage.CoverageData``), one context per test."""
    coverage = pytest.importorskip("coverage")
    data = coverage.CoverageData(basename=str(root / ".coverage"))
    for context, files in contexts.items():
        data.set_context(context)
        data.add_lines({f: [1, 2] for f in files})
    data.write()
    return root / ".coverage"


def _coverage_reasons(plan) -> list[str]:
    return [r for r in plan.reason_codes if r.startswith("coverage.")]


def test_coverage_is_used_only_when_it_belongs_to_this_workspace_and_revision(
    pyproj: Path, tmp_path: Path
):
    service = pyproj / "src" / "service.py"
    calc = pyproj / "src" / "calc.py"
    edge = {"tests/test_unrelated.py::test_other|run": [str(service), str(calc)]}
    service.write_text(service.read_text() + "# touched\n", encoding="utf-8")

    def selected(plan) -> dict[str, str]:
        return {t.id: t.reason for t in plan.tier1}

    # Current data: the coverage edge is evidence.
    cov = _write_coverage(pyproj, edge)
    planner, identity, _ = _planner(pyproj)
    plan = planner.plan(identity)
    assert selected(plan)["tests/test_unrelated.py::test_other"] == "coverage"
    assert _coverage_reasons(plan) == ["coverage.used"]

    def replan() -> object:
        fresh, ident, _ = _planner(pyproj)
        return fresh.plan(ident), fresh

    # A measured file outside the current diff changed after measurement.
    old = cov.stat().st_mtime
    os.utime(calc, (old + 60, old + 60))
    plan, fresh = replan()
    assert "tests/test_unrelated.py::test_other" not in selected(plan)
    assert _coverage_reasons(plan) == ["coverage.unavailable.stale_sources"]
    assert fresh.counters["coverage_rejected"] == 1
    # Fallback still selects by naming/imports for the changed file.
    assert "tests/test_service.py::test_total" in selected(plan)
    os.utime(calc, (old - 60, old - 60))

    # HEAD was committed after the data was written (another revision).
    os.utime(cov, (old - 3600, old - 3600))
    plan, _ = replan()
    assert _coverage_reasons(plan) == ["coverage.unavailable.stale_revision"]
    cov.unlink()

    # Data measured in another checkout: no measured file inside this worktree.
    other = tmp_path / "elsewhere" / "src" / "service.py"
    _write_coverage(pyproj, {"tests/test_unrelated.py::test_other|run": [str(other)]})
    plan, _ = replan()
    assert _coverage_reasons(plan) == ["coverage.unavailable.foreign_workspace"]
    cov.unlink()

    # Measured without per-test contexts: nothing to map tests from.
    _write_coverage(pyproj, {"": [str(service)]})
    plan, _ = replan()
    assert _coverage_reasons(plan) == ["coverage.unavailable.no_test_contexts"]
    cov.unlink()

    # Not coverage.py data at all.
    conn = sqlite3.connect(cov)
    conn.execute("CREATE TABLE file (id INTEGER PRIMARY KEY, path TEXT)")
    conn.commit()
    conn.close()
    plan, _ = replan()
    assert _coverage_reasons(plan) == ["coverage.unavailable.schema"]
    cov.unlink()
    plan, _ = replan()
    assert _coverage_reasons(plan) == []  # no data: nothing claimed either way


def test_risky_changes_escalate_to_the_full_suite(pyproj: Path):
    planner, identity, runtime = _planner(pyproj)
    (pyproj / "src" / "auth" / "token.py").write_text("def issue():\n    return 'y'\n")
    plan = planner.plan(identity)
    assert "risk.security" in plan.reason_codes
    assert "full_suite" in plan.required_checks
    assert plan.estimate["full_suite_required"] is True
    _git(pyproj, "checkout", "--", ".")
    (pyproj / "pytest.ini").write_text("[pytest]\naddopts = -q\n", encoding="utf-8")
    plan = planner.plan(identity)
    assert "risk.build_config" in plan.reason_codes and "full_suite" in plan.required_checks
    _git(pyproj, "checkout", "--", ".")
    # Final completion (agent proposed "done") makes the full suite mandatory.
    (pyproj / "src" / "calc.py").write_text("def add(a, b):\n    return a + b + 0\n")
    plain = planner.plan(identity)
    assert "full_suite" not in plain.required_checks
    _set_state(
        runtime,
        identity,
        StateAtom("at_done", "completed", "Fixed add()", "agent", 0.6, "PROPOSED"),
    )
    final = planner.plan(identity)
    assert final.final_completion and "full_suite" in final.required_checks
    assert final.plan_revision > plain.plan_revision


def test_plan_invalidation_on_files_task_scope_and_runner_config(pyproj: Path):
    planner, identity, runtime = _planner(pyproj)
    (pyproj / "src" / "calc.py").write_text("def add(a, b):\n    return b + a\n")
    plan = planner.plan(identity)
    assert planner.is_current(identity, plan)
    (pyproj / "src" / "service.py").write_text("X = 1\n")  # changed files
    assert not planner.is_current(identity, plan)
    newer = planner.plan(identity)
    assert newer.plan_revision == plan.plan_revision + 1
    _set_state(runtime, identity, StateAtom("at_g", "goal", "fix add", "user", 0.9, "ACTIVE"))
    assert not planner.is_current(identity, newer)  # task revision moved
    newest = planner.plan(identity)
    (pyproj / "pytest.ini").write_text("[pytest]\nmarkers = slow\n")  # runner config
    assert not planner.is_current(identity, newest)
    assert planner.plan(identity).runner_fingerprint != newest.runner_fingerprint


def test_flaky_pass_never_masquerades_as_green(pyproj: Path):
    (pyproj / "src" / "calc.py").write_text("def add(a, b):\n    return b + a\n")
    planner, identity, runtime = _planner(pyproj)
    plan = planner.plan(identity)
    cs = compute_changeset(str(pyproj)).digest
    ws, sess = identity.workspace_id, identity.session_id
    fail = parse_output(
        "pytest",
        "FAILED tests/test_calc.py::test_add - assert 4 == 3\n1 failed, 1 passed in 0.1s\n",
        1,
    )
    first = planner.record_outcome(ws, sess, fail, changeset=cs, plan=plan, tier=1)
    assert first["status"] == "failed" and first["overall"] == "failed"
    ids = [t.id for t in plan.tier1]
    passed = parse_output("pytest", "2 passed in 0.1s\n", 0, targets=ids)
    second = planner.record_outcome(ws, sess, passed, changeset=cs, plan=plan, tier=1)
    assert second["status"] == "flaky" and second["overall"] == "flaky"
    assert second["flaky_ids"] == ["tests/test_calc.py::test_add"]
    assert planner.check_status(plan)["overall"] == "flaky"
    # The VERIFICATION_DECISION it emitted is negative evidence and a TaskState blocker.
    event = runtime.submitted[-1].event
    assert event.kind == EventKind.VERIFICATION_DECISION.value
    facts = extract_candidates(EventEnvelope(event=event))
    assert facts[0].predicate == "verification.status" and facts[0].polarity == "negative"
    state = new_task("tk_f", ws, sess, 1.0)
    _, result, _ = derive_transitions(state, EventEnvelope(event=event), now=2.0)
    assert result.status == "BLOCKED"
    assert any("flaky" in a.text for a in result.blockers if a.live)
    # A real fix (new changeset) that passes is green again, not flaky.
    (pyproj / "src" / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    fixed_plan = planner.plan(identity)
    cs2 = compute_changeset(str(pyproj)).digest
    third = planner.record_outcome(
        ws,
        sess,
        parse_output("pytest", "2 passed\n", 0, targets=ids),
        changeset=cs2,
        plan=fixed_plan,
        tier=1,
    )
    assert third["status"] == "passed"


def test_observed_client_runs_feed_history_through_projection(pyproj: Path):
    planner, identity, runtime = _planner(pyproj)
    planner.note_identity(identity)
    (pyproj / "src" / "calc.py").write_text("def add(a, b):\n    return b + a\n")
    plan = planner.plan(identity)

    def test_event(i: int, output: str, code: int):
        meta = {
            "command_head": "pytest tests/test_calc.py::test_add",
            "runner": "pytest",
            "exit_code": code,
            "is_partial": False,
        }
        event = make_event(
            workspace_id=identity.workspace_id,
            session_id=identity.session_id,
            kind=EventKind.TEST_RESULT,
            origin=Origin.TOOL,
            source_id=f"toolu_{i}",
            index=0,
            payload=output,
            metadata=meta,
        )
        return EventEnvelope(event=event, content=output)

    for i, (out, code) in enumerate(
        [("FAILED tests/test_calc.py::test_add - x\n1 failed\n", 1), ("1 passed\n", 0)]
    ):
        env = test_event(i, out, code)
        planner.project(env, None)
        planner.on_commit([env.event], runtime)
    decisions = [
        e.event for e in runtime.submitted if e.event.kind == EventKind.VERIFICATION_DECISION.value
    ]
    assert [d.redacted_metadata["status"] for d in decisions] == ["failed", "flaky"]
    assert plan.plan_id == decisions[-1].redacted_metadata["plan_id"]
    assert planner.counters["flaky_detected"] == 1


def test_unknown_framework_is_explicit(tmp_path: Path):
    root = _repo(tmp_path / "none", {"README.md": "hi\n", "notes/a.txt": "x\n"})
    (root / "notes" / "a.txt").write_text("y\n")
    planner, identity, _ = _planner(root)
    plan = planner.plan(identity)
    assert plan.frameworks == () and "framework.unknown" in plan.unknown
    assert plan.tier1 == () and plan.commands == ()
    canon = _repo(tmp_path / "mk", {"Makefile": "test:\n\techo ok\n", "lib.c": "int x;\n"})
    (canon / "lib.c").write_text("int y;\n")
    planner2, ident2, _ = _planner(canon)
    plan2 = planner2.plan(ident2)
    assert plan2.frameworks == ("canonical",)
    assert plan2.commands[0].argv == ("make", "test")
    assert "full_suite" in plan2.required_checks


# ---------------------------------------------------------------------------
# Cargo
# ---------------------------------------------------------------------------

CARGO_FILES = {
    "Cargo.toml": '[workspace]\nmembers = ["mathlib", "app"]\nresolver = "2"\n',
    "mathlib/Cargo.toml": '[package]\nname = "mathlib"\nversion = "0.1.0"\nedition = "2021"\n',
    "mathlib/src/lib.rs": "pub mod math;\npub mod text;\n",
    "mathlib/src/math.rs": (
        "pub fn add(a: i32, b: i32) -> i32 { a + b }\n\n#[cfg(test)]\nmod tests {\n"
        "    use super::*;\n    #[test]\n    fn adds() { assert_eq!(add(1, 2), 3); }\n}\n"
    ),
    "mathlib/src/text.rs": (
        "pub fn up(s: &str) -> String { s.to_uppercase() }\n\n#[cfg(test)]\nmod tests {\n"
        '    #[test]\n    fn ups() { assert_eq!(super::up("a"), "A"); }\n}\n'
    ),
    "mathlib/tests/integration.rs": (
        "use mathlib::math::add;\n\n#[test]\nfn integrates() { assert_eq!(add(2, 2), 4); }\n"
    ),
    "app/Cargo.toml": (
        '[package]\nname = "app"\nversion = "0.1.0"\nedition = "2021"\n\n'
        '[dependencies]\nmathlib = { path = "../mathlib" }\n'
    ),
    "app/src/lib.rs": (
        "pub fn twice(x: i32) -> i32 { mathlib::math::add(x, x) }\n\n#[cfg(test)]\nmod tests {\n"
        "    #[test]\n    fn doubles() { assert_eq!(super::twice(2), 4); }\n}\n"
    ),
    ".gitignore": "target/\n",
}


@pytest.mark.skipif(shutil.which("cargo") is None, reason="platform limit: cargo not installed")
def test_cargo_selection_by_crate_module_and_dependents(tmp_path: Path, monkeypatch):
    root = _repo(tmp_path / "rs", CARGO_FILES)
    (root / "mathlib" / "src" / "math.rs").write_text(
        CARGO_FILES["mathlib/src/math.rs"].replace("a + b", "b + a"), encoding="utf-8"
    )
    planner, identity, _ = _planner(root)
    plan = planner.plan(identity)
    assert plan.frameworks == ("cargo",)
    assert [t.id for t in plan.tier1] == ["mathlib:lib:math::", "mathlib:test:integration"]
    assert {t.id for t in plan.tier2} >= {"app:package"}
    tier1 = [c.argv for c in plan.commands_for(1)]
    assert ("cargo", "test", "-p", "mathlib", "--lib", "math::") in tier1
    assert ("cargo", "test", "-p", "mathlib", "--test", "integration") in tier1
    assert plan.commands_for(3)[0].argv == ("cargo", "test", "--workspace")
    target = REPO / "target" / "tmp" / "gate7_cargo_target"
    env = {**os.environ, "CARGO_TARGET_DIR": str(target)}
    run = run_argv(list(tier1[0]), cwd=str(root), timeout_s=280, env=env)
    outcome = parse_output("cargo", run.output, run.exit_code)
    assert run.exit_code == 0, run.output[-2000:]
    assert outcome.passed_ids == ("math::tests::adds",)  # text:: tests were not selected
    assert "text::tests::ups" not in run.output


# ---------------------------------------------------------------------------
# CTest
# ---------------------------------------------------------------------------

CMAKE = (
    "cmake_minimum_required(VERSION 3.16)\nproject(fx LANGUAGES NONE)\nenable_testing()\n"
    "add_test(NAME math_add COMMAND ${CMAKE_COMMAND} -E echo math-ok)\n"
    "add_test(NAME text_upper COMMAND ${CMAKE_COMMAND} -E echo text-ok)\n"
)


@pytest.mark.skipif(
    shutil.which("ctest") is None or shutil.which("cmake") is None,
    reason="platform limit: cmake/ctest not installed",
)
def test_ctest_selection_uses_existing_build_dir_only(tmp_path: Path):
    root = _repo(
        tmp_path / "cpp",
        {
            "CMakeLists.txt": CMAKE,
            "src/math.c": "int add(int a, int b) { return a + b; }\n",
            "src/text.c": "int up(void) { return 0; }\n",
            ".gitignore": "build/\n",
        },
    )
    (root / "src" / "math.c").write_text("int add(int a, int b) { return b + a; }\n")
    planner, identity, _ = _planner(root)
    unconfigured = planner.plan(identity)
    assert "ctest.no_configured_build_dir" in unconfigured.unknown
    assert not unconfigured.commands  # never invents a build dir or configures
    assert not (root / "build").exists()
    # The project's own configure (done by the "user"), then discovery works.
    subprocess.run(
        ["cmake", "-S", str(root), "-B", str(root / "build")], check=True, capture_output=True
    )
    plan = planner.plan(identity, force=True)
    assert [t.id for t in plan.tier1] == ["math_add"]
    cmd = plan.commands_for(1)[0]
    assert cmd.cwd == "build" and "^(math_add)$" in cmd.argv
    run = run_argv(list(cmd.argv), cwd=str(root / cmd.cwd), timeout_s=60)
    outcome = parse_output("ctest", run.output, run.exit_code)
    assert run.exit_code == 0 and outcome.passed_ids == ("math_add",) and outcome.failed == 0


# ---------------------------------------------------------------------------
# Jest / Vitest (package manager respected)
# ---------------------------------------------------------------------------

JS_FILES = {
    "src/math.js": "export function add(a, b) { return a + b; }\n",
    "src/text.js": "export function up(s) { return s.toUpperCase(); }\n",
    "src/math.test.js": (
        "import { add } from './math.js';\nimport { test, expect } from 'vitest';\n"
        "test('adds', () => { expect(add(1, 2)).toBe(3); });\n"
    ),
    "src/text.test.js": (
        "import { up } from './text.js';\nimport { test, expect } from 'vitest';\n"
        "test('ups', () => { expect(up('a')).toBe('A'); });\n"
    ),
    "src/__tests__/calc.spec.js": (
        "import { add } from '../math.js';\nimport { test, expect } from 'vitest';\n"
        "test('calc', () => { expect(add(2, 2)).toBe(4); });\n"
    ),
}


def test_jest_selection_respects_pnpm(tmp_path: Path):
    files = {
        **{
            k: v.replace("import { test, expect } from 'vitest';\n", "")
            for k, v in JS_FILES.items()
        },
        "package.json": json.dumps({"name": "fx", "devDependencies": {"jest": "^29"}}),
        "pnpm-lock.yaml": "lockfileVersion: '9.0'\n",
    }
    root = _repo(tmp_path / "jest", files)
    (root / "src" / "math.js").write_text("export function add(a, b) { return b + a; }\n")
    planner, identity, _ = _planner(root)
    plan = planner.plan(identity)
    assert plan.frameworks == ("jest",)
    assert sorted(t.file for t in plan.tier1) == ["src/__tests__/calc.spec.js", "src/math.test.js"]
    cmd = plan.commands_for(1)[0]
    assert cmd.argv[:5] == ("pnpm", "exec", "jest", "--ci", "--runTestsByPath")
    assert "src/text.test.js" not in cmd.argv


def test_vitest_selection_respects_yarn_and_runs_when_available(tmp_path: Path):
    files = {
        **JS_FILES,
        "package.json": json.dumps(
            {"name": "fx", "type": "module", "devDependencies": {"vitest": "^3"}}
        ),
        "yarn.lock": "# yarn lockfile v1\n",
    }
    root = _repo(tmp_path / "vitest", files)
    (root / "src" / "math.js").write_text("export function add(a, b) { return b + a; }\n")
    planner, identity, _ = _planner(root)
    plan = planner.plan(identity)
    assert plan.frameworks == ("vitest",)
    cmd = plan.commands_for(1)[0]
    assert cmd.argv[:3] == ("yarn", "vitest", "run")
    assert sorted(cmd.targets) == ["src/__tests__/calc.spec.js", "src/math.test.js"]
    vitest_bin = JS_RUNNER / ".bin" / ("vitest.cmd" if sys.platform == "win32" else "vitest")
    if not vitest_bin.exists():
        pytest.skip("platform limit: vitest is not provisioned locally (target/tmp/js_runner)")
    if shutil.which("node") is None:
        pytest.skip("platform limit: no node runtime on PATH for the provisioned vitest")
    # Same selection, executed with the provisioned runner binary (direct argv).
    if sys.platform == "win32":  # a junction needs no symlink privilege
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(root / "node_modules"), str(JS_RUNNER)],
            check=True,
            capture_output=True,
        )
    else:
        os.symlink(JS_RUNNER, root / "node_modules", target_is_directory=True)
    run = run_argv(
        [str(vitest_bin), "run", *cmd.targets],
        cwd=str(root),
        timeout_s=180,
        env={**os.environ, "CI": "1"},
    )
    outcome = parse_output("vitest", run.output, run.exit_code)
    assert run.exit_code == 0, run.output[-2000:]
    assert outcome.passed == 2 and outcome.failed == 0
    assert "ups" not in run.output  # text.test.js was not selected


def test_parsers_cover_each_framework():
    jest = parse_output(
        "jest",
        "FAIL src/a.test.js\n  ● math › adds\n\nTests:       1 failed, 2 passed, 3 total\n",
        1,
    )
    assert (jest.failed, jest.passed, jest.failed_ids) == (1, 2, ("math › adds",))
    vitest = parse_output(
        "vitest",
        " RUN  v3.2.4 /x\n FAIL  src/a.test.js > adds\n      Tests  1 failed | 1 passed (2)\n",
        1,
    )
    assert vitest.failed == 1 and vitest.runner_version == "3.2.4"
    cargo = parse_output(
        "cargo",
        "test a::b ... ok\ntest a::c ... FAILED\ntest result: FAILED. 1 passed; 1 failed; 0 ignored\n",
        101,
    )
    assert cargo.failed_ids == ("a::c",) and cargo.passed == 1
    ctest = parse_output(
        "ctest",
        "1/2 Test #1: a ....   Passed    0.01 sec\n2/2 Test #2: b ....***Failed    0.01 sec\n"
        "50% tests passed, 1 tests failed out of 2\n",
        8,
    )
    assert ctest.failed_ids == ("b",) and ctest.passed == 1
    assert Framework.CTEST.value == "ctest"
