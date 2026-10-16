"""Test Impact and Verification Planner (unified-21 #24).

Owns :class:`VerificationPlan` — which tests prove the *current* change, in
which order, and whether the full suite is mandatory — for a trusted wrap
workspace. A plan is tied to:

* the current Git changeset (tracked diff + untracked files, content-hashed),
* the task (TaskState id/revision, acceptance atoms, proposed completion),
* EvidenceFacts (failing tests / error signatures named in the session),
* affected graph nodes (Phase-2 code graph importers/callers when indexed,
  otherwise a bounded lexical import index), and
* prior runs (exact historical failure associations, flaky tests),

and is pinned to ``(task_revision, scope_revision, changeset_hash,
runner_fingerprint)``. Any change to changed files, the task, the change
scope or the runner configuration produces a new ``plan_revision``; a stale
plan is never reported or run as current (:meth:`TestImpactPlanner.is_current`).

Framework adapters preserve the project-selected runner: pytest (static node
ids, ``--collect-only`` validation, existing coverage.py contexts), Cargo
(``cargo metadata --no-deps``, crate/target/module filters), CTest
(``ctest --show-only=json-v1`` in an *existing* configured build directory —
never an invented one, never a configure step) and Jest/Vitest (with the
detected npm/pnpm/yarn/bun). Unsupported projects fall back to a canonical
project test command (``make test``, ``go test ./...``, ``tox``) or an
explicit ``unknown`` result.

Tier 1 = directly related tests, Tier 2 = dependency/interface neighbours,
Tier 3 = full suite. Targets sort deterministically: exact failure
regressions first, then direct scope, dependency neighbours, project smoke.
High-risk changes (public API, security, build config, provider wire,
platform) and final completion make the full suite mandatory even when
Tier 1 passes.

Test outcomes come from the existing event ingestion (``TEST_RESULT`` events
of client-executed runs and of Headroom workflow steps alike). Each recorded
run updates per-test history; a test that fails and then passes on the same
changeset is **flaky** and is reported separately — a flaky pass never makes a
plan green. Every recorded run emits one ``VERIFICATION_DECISION`` event that
the Evidence Ledger and TaskState consume (so the revision changes before the
agent's next action). Planning never executes project tests; only a genuine
host permission boundary (the workflow tool behind Claude Code's pre-tool
hook) can start a run. The avoided-work figure is an estimate, not a billing
claim.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shlex
import sys
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from headroom.agent.events import EventEnvelope, EventKind, Origin, make_event
from headroom.log_safety import describe_exception

log = logging.getLogger(__name__)

FEATURE_TEST_IMPACT = "agent_test_impact"
PLANNER_PROJECTOR = "test_impact_runs"
LISTENER_PLANNER = "test_impact_refresh"
STAGE_VERIFICATION = "agent_verification"
BLOCK_TAG = "headroom_verification"
PLAN_SCHEMA_VERSION = 1

MAX_FILES_SCANNED = 6000
MAX_SOURCE_BYTES = 256 * 1024
MAX_TARGETS_PER_TIER = 48
DISCOVERY_TIMEOUT_S = 20.0
GIT_TIMEOUT_S = 8.0
REFRESH_MIN_INTERVAL_S = 1.0
BLOCK_MAX_TOKENS = 600
_HISTORY_PER_TEST = 16
_HISTORY_TESTS = 4096
_PLANS_LIMIT = 256

_TABLE = (
    "CREATE TABLE IF NOT EXISTS verification_runs ("
    " workspace_id TEXT NOT NULL, framework TEXT NOT NULL, test_id TEXT NOT NULL,"
    " changeset TEXT NOT NULL, ok INTEGER NOT NULL, ts REAL NOT NULL)"
)
_INDEX = "CREATE INDEX IF NOT EXISTS verification_runs_ws ON verification_runs(workspace_id, test_id, ts)"


class Framework(str, Enum):
    PYTEST = "pytest"
    CARGO = "cargo"
    CTEST = "ctest"
    JEST = "jest"
    VITEST = "vitest"
    CANONICAL = "canonical"
    UNKNOWN = "unknown"


class Reason(str, Enum):
    FAILURE_REGRESSION = "failure_regression"
    COVERAGE = "coverage"
    CHANGED_TEST = "changed_test"
    NAMING = "naming"
    DIRECT_IMPORT = "direct_import"
    MODULE_FILTER = "module_filter"
    ACCEPTANCE = "acceptance"
    DEPENDENCY = "dependency_neighbor"
    INTERFACE = "interface_neighbor"
    PACKAGE = "package_scope"
    SMOKE = "project_smoke"
    FULL_SUITE = "full_suite"


_REASON_RANK = {
    Reason.FAILURE_REGRESSION: 0,
    Reason.CHANGED_TEST: 1,
    Reason.COVERAGE: 1,
    Reason.ACCEPTANCE: 1,
    Reason.NAMING: 2,
    Reason.DIRECT_IMPORT: 2,
    Reason.MODULE_FILTER: 2,
    Reason.DEPENDENCY: 3,
    Reason.INTERFACE: 3,
    Reason.PACKAGE: 4,
    Reason.SMOKE: 5,
    Reason.FULL_SUITE: 6,
}


@dataclass(frozen=True)
class TestTarget:
    """One selectable test (node id, filter, file or named test)."""

    __test__ = False  # not a pytest class

    id: str
    framework: str
    reason: str
    source: str = ""  # the changed file / evidence fact / history that selected it
    file: str | None = None
    group: str = ""  # adapter grouping key (cargo package, ctest build dir, ...)

    def sort_key(self) -> tuple[int, str, str]:
        try:
            rank = _REASON_RANK[Reason(self.reason)]
        except ValueError:
            rank = 9
        return (rank, self.framework, self.id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "framework": self.framework,
            "reason": self.reason,
            "source": self.source,
            "file": self.file,
        }


@dataclass(frozen=True)
class TestCommand:
    """A runnable selection: direct argv (no shell) in a workspace-relative cwd."""

    __test__ = False

    framework: str
    tier: int
    argv: tuple[str, ...]
    cwd: str = "."
    targets: tuple[str, ...] = ()

    def display(self) -> str:
        return " ".join(shlex.quote(a) if re.search(r"[\s'\"]", a) else a for a in self.argv)

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "tier": self.tier,
            "argv": list(self.argv),
            "cwd": self.cwd,
            "targets": list(self.targets),
        }


@dataclass(frozen=True)
class VerificationPlan:
    workspace_id: str
    task_id: str | None
    task_revision: int
    scope_revision: int
    changeset_hash: str
    plan_revision: int
    frameworks: tuple[str, ...]
    tier1: tuple[TestTarget, ...]
    tier2: tuple[TestTarget, ...]
    tier3: tuple[TestTarget, ...]
    required_checks: tuple[str, ...]
    risk_score: float
    reason_codes: tuple[str, ...]
    commands: tuple[TestCommand, ...] = ()
    changed_files: tuple[str, ...] = ()
    runner_fingerprint: str = ""
    session_id: str = ""
    final_completion: bool = False
    estimate: Mapping[str, Any] = field(default_factory=dict)
    unknown: tuple[str, ...] = ()  # explicit "cannot plan" results (framework.reason)
    created_at: float = 0.0

    @property
    def plan_id(self) -> str:
        return (
            "vp_"
            + _digest(
                self.workspace_id,
                self.session_id,
                self.task_id or "",
                self.task_revision,
                self.scope_revision,
                self.changeset_hash,
                self.runner_fingerprint,
            )[:20]
        )

    def pins(self) -> tuple[Any, ...]:
        return (
            self.task_id,
            self.task_revision,
            self.scope_revision,
            self.changeset_hash,
            self.runner_fingerprint,
        )

    def commands_for(self, tier: int, framework: str | None = None) -> list[TestCommand]:
        return [
            c
            for c in self.commands
            if c.tier == tier and (framework is None or c.framework == framework)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "v": PLAN_SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "workspace_id": self.workspace_id,
            "task_id": self.task_id,
            "task_revision": self.task_revision,
            "scope_revision": self.scope_revision,
            "changeset_hash": self.changeset_hash,
            "plan_revision": self.plan_revision,
            "frameworks": list(self.frameworks),
            "tier1": [t.to_dict() for t in self.tier1],
            "tier2": [t.to_dict() for t in self.tier2],
            "tier3": [t.to_dict() for t in self.tier3],
            "required_checks": list(self.required_checks),
            "risk_score": round(self.risk_score, 3),
            "reason_codes": list(self.reason_codes),
            "commands": [c.to_dict() for c in self.commands],
            "changed_files": list(self.changed_files),
            "runner_fingerprint": self.runner_fingerprint,
            "final_completion": self.final_completion,
            "estimate": dict(self.estimate),
            "unknown": list(self.unknown),
        }


@dataclass(frozen=True)
class RunOutcome:
    """Parsed result of one test run."""

    framework: str
    exit_code: int | None
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    failed_ids: tuple[str, ...] = ()
    passed_ids: tuple[str, ...] = ()
    runner_version: str = ""
    partial: bool = False
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return (
            not self.partial
            and not self.timed_out
            and self.exit_code == 0
            and self.failed == 0
            and self.errors == 0
        )

    def summary(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "exit_code": self.exit_code,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "failed_ids": list(self.failed_ids[:16]),
            "runner_version": self.runner_version,
            "partial": self.partial,
            "timed_out": self.timed_out,
        }


def _digest(*parts: Any) -> str:
    raw = "\x1f".join(str(p) for p in parts).encode("utf-8", "surrogatepass")
    return hashlib.sha256(raw).hexdigest()


def _rel(path: str) -> str:
    return path.replace("\\", "/").lstrip("./") if not path.startswith("../") else path


# ---------------------------------------------------------------------------
# Changeset (current Git diff) and runner fingerprint
# ---------------------------------------------------------------------------


def _git(root: str, *args: str, timeout: float = GIT_TIMEOUT_S) -> str | None:
    from headroom._subprocess import run

    try:
        result = run(
            ["git", "-C", root, *args],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - git missing / timeout: no diff source
        log.debug("git %s failed: %s", args[:1], describe_exception(exc))
        return None
    if result.returncode != 0:
        return None
    out = result.stdout
    return out.decode("utf-8", "replace") if isinstance(out, bytes) else str(out)


@dataclass(frozen=True)
class Changeset:
    files: tuple[str, ...]
    deleted: tuple[str, ...]
    lines_changed: int
    digest: str
    source: str  # git | recorded_writes | none


def _file_hash(root: str, rel: str) -> str:
    full = os.path.join(root, rel)
    try:
        h = hashlib.sha256()
        with open(full, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()[:32]
    except OSError:
        return "missing"


def compute_changeset(root: str, fallback: Iterable[str] = ()) -> Changeset:
    """Changed files relative to HEAD (tracked + untracked), content-hashed."""
    files: dict[str, str] = {}
    deleted: list[str] = []
    lines = 0
    source = "none"
    if os.path.exists(os.path.join(root, ".git")):
        status = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
        if status is not None:
            source = "git"
            entries = status.split("\x00")
            i = 0
            while i < len(entries):
                entry = entries[i]
                i += 1
                if len(entry) < 4:
                    continue
                code, path = entry[:2], entry[3:]
                if code[0] in "RC":
                    i += 1  # the original path follows a rename/copy record
                rel = _rel(path)
                if "D" in code:
                    deleted.append(rel)
                    files[rel] = "deleted"
                else:
                    files[rel] = _file_hash(root, rel)
            numstat = _git(root, "diff", "--numstat", "HEAD")
            for line in (numstat or "").splitlines():
                parts = line.split("\t")
                if len(parts) >= 3:
                    for value in parts[:2]:
                        if value.isdigit():
                            lines += int(value)
            head = (_git(root, "rev-parse", "HEAD") or "").strip()
            files.setdefault("\x00HEAD", head)
    if source == "none":
        for raw in fallback:
            rel = _rel(raw)
            if rel:
                files[rel] = _file_hash(root, rel)
                source = "recorded_writes"
    head = files.pop("\x00HEAD", "")
    ordered = tuple(sorted(files))
    digest = _digest(head, *(f"{p}={files[p]}" for p in ordered))[:24]
    return Changeset(ordered, tuple(sorted(deleted)), lines, digest, source)


_RUNNER_CONFIG = (
    "pytest.ini",
    "pyproject.toml",
    "setup.cfg",
    "tox.ini",
    "conftest.py",
    "Cargo.toml",
    "Cargo.lock",
    "CMakeLists.txt",
    "CTestTestfile.cmake",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "bun.lockb",
    "bun.lock",
    "jest.config.js",
    "jest.config.ts",
    "jest.config.mjs",
    "jest.config.cjs",
    "vitest.config.ts",
    "vitest.config.js",
    "vitest.config.mts",
    "vite.config.ts",
    "vite.config.js",
    "Makefile",
    "go.mod",
)


def runner_fingerprint(root: str, extra: Iterable[str] = ()) -> str:
    """Digest of the runner configuration (changes invalidate the plan)."""
    parts: list[str] = []
    for name in (*_RUNNER_CONFIG, *extra):
        full = os.path.join(root, name)
        try:
            st = os.stat(full)
        except OSError:
            continue
        parts.append(f"{name}:{st.st_size}:{st.st_mtime_ns}")
    return _digest(*parts)[:16]


# ---------------------------------------------------------------------------
# Project context (file listing + lexical import index)
# ---------------------------------------------------------------------------

_PY_IMPORT = re.compile(
    r"^\s*(?:from\s+([\w.]+)\s+import\s+([\w\s,()*]+)|import\s+([\w.,\s]+))", re.M
)
_JS_IMPORT = re.compile(
    r"""(?:\bimport\s+(?:[^'"]*?\s+from\s+)?|\brequire\(\s*|\bimport\(\s*|\bexport\s+[^'"]*?\s+from\s+)['"]([^'"]+)['"]"""
)
_C_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)
_RS_USE = re.compile(r"^\s*(?:pub\s+)?(?:use\s+(?:crate|super)::([\w:]+)|mod\s+(\w+)\s*;)", re.M)
_SOURCE_EXT = (
    ".py",
    ".pyi",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",
    ".rs",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
)
_JS_EXT = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts")


class ProjectContext:
    """One planning pass over a worktree (bounded reads, cached per pass)."""

    def __init__(self, root: str, *, history_commands: Mapping[str, str] | None = None) -> None:
        self.root = root
        self.history_commands = dict(history_commands or {})
        self._files: list[str] | None = None
        self._sources: dict[str, str] = {}
        self._importers: dict[str, set[str]] | None = None
        self.cache: dict[str, Any] = {}

    @property
    def files(self) -> list[str]:
        if self._files is None:
            from headroom.agent.code_graph import _git_list_files, _walk_files

            listed = _git_list_files(self.root)
            if listed is None:
                listed = _walk_files(self.root, MAX_FILES_SCANNED)
            self._files = sorted({_rel(p) for p in listed})[:MAX_FILES_SCANNED]
        return self._files

    def exists(self, rel: str) -> bool:
        return os.path.isfile(os.path.join(self.root, rel))

    def read(self, rel: str) -> str:
        if rel in self._sources:
            return self._sources[rel]
        full = os.path.join(self.root, rel)
        text = ""
        try:
            if os.path.getsize(full) <= MAX_SOURCE_BYTES:
                with open(full, encoding="utf-8", errors="replace") as handle:
                    text = handle.read()
        except OSError:
            text = ""
        self._sources[rel] = text
        return text

    # -- import index ----------------------------------------------------

    def _python_modules(self) -> dict[str, str]:
        mods: dict[str, str] = self.cache.setdefault("py_modules", {})
        if mods:
            return mods
        for rel in self.files:
            if not rel.endswith(".py"):
                continue
            stem = rel[:-3]
            if stem.endswith("/__init__"):
                stem = stem[: -len("/__init__")]
            parts = stem.split("/")
            for start in range(len(parts)):
                mods.setdefault(".".join(parts[start:]), rel)
        return mods

    def _resolve(self, importer: str, ref: str, kind: str) -> list[str]:
        if kind == "py":
            mods = self._python_modules()
            hit = mods.get(ref)
            return [hit] if hit else []
        base = os.path.dirname(importer)
        if kind == "js":
            if not ref.startswith("."):
                return []
            target = os.path.normpath(os.path.join(base, ref)).replace("\\", "/")
            candidates = [target, *(target + e for e in _JS_EXT)]
            candidates += [f"{target}/index{e}" for e in _JS_EXT]
            # "./a.js" may name "a.ts" under TS module resolution.
            stem, ext = os.path.splitext(target)
            if ext in _JS_EXT:
                candidates += [stem + e for e in _JS_EXT]
            return [c for c in candidates if c in self._file_set()][:1]
        if kind == "c":
            target = os.path.normpath(os.path.join(base, ref)).replace("\\", "/")
            if target in self._file_set():
                return [target]
            name = ref.rsplit("/", 1)[-1]
            return [f for f in self.files if f.rsplit("/", 1)[-1] == name][:2]
        if kind == "rs":
            crate_src = _crate_src(importer)
            module = ref.split("::")[0]
            if crate_src is None:
                return []
            out = []
            for cand in (f"{crate_src}/{module}.rs", f"{crate_src}/{module}/mod.rs"):
                if cand in self._file_set():
                    out.append(cand)
            return out
        return []

    def _file_set(self) -> set[str]:
        files: set[str] | None = self.cache.get("file_set")
        if files is None:
            files = set(self.files)
            self.cache["file_set"] = files
        return files

    def imports_of(self, rel: str) -> set[str]:
        text = self.read(rel)
        if not text:
            return set()
        out: set[str] = set()
        if rel.endswith((".py", ".pyi")):
            for m in _PY_IMPORT.finditer(text):
                if m.group(1):
                    module = m.group(1)
                    out.update(self._resolve(rel, module, "py"))
                    for name in re.split(r"[\s,()]+", m.group(2) or ""):
                        if name and name != "*":
                            out.update(self._resolve(rel, f"{module}.{name}", "py"))
                else:
                    for module in (m.group(3) or "").split(","):
                        module = module.strip().split(" ")[0]
                        if module:
                            out.update(self._resolve(rel, module, "py"))
        elif rel.endswith(_JS_EXT + (".jsx",)):
            for m in _JS_IMPORT.finditer(text):
                out.update(self._resolve(rel, m.group(1), "js"))
        elif rel.endswith((".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp")):
            for m in _C_INCLUDE.finditer(text):
                out.update(self._resolve(rel, m.group(1), "c"))
        elif rel.endswith(".rs"):
            for m in _RS_USE.finditer(text):
                out.update(self._resolve(rel, m.group(1) or m.group(2), "rs"))
        out.discard(rel)
        return out

    def importers(self) -> dict[str, set[str]]:
        if self._importers is None:
            index: dict[str, set[str]] = {}
            for rel in self.files:
                if not rel.endswith(_SOURCE_EXT):
                    continue
                for target in self.imports_of(rel):
                    index.setdefault(target, set()).add(rel)
            self._importers = index
        return self._importers


def _crate_src(rel: str) -> str | None:
    parts = rel.split("/")
    if "src" in parts:
        idx = len(parts) - 1 - parts[::-1].index("src")
        return "/".join(parts[: idx + 1])
    return None


# ---------------------------------------------------------------------------
# Output parsers (shared by observed runs and Headroom-started runs)
# ---------------------------------------------------------------------------

_PYTEST_COUNTS = re.compile(
    r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?)"
)
_PYTEST_FAILED = re.compile(r"^(?:FAILED|ERROR)\s+(\S+?)(?:\s+-\s.*)?$", re.M)
_PYTEST_VERSION = re.compile(r"\bpytest-(\d+\.\d+(?:\.\d+)?)")
_CARGO_TEST = re.compile(r"^test (\S+) \.\.\. (ok|FAILED|ignored)\s*$", re.M)
_CARGO_RESULT = re.compile(
    r"test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored", re.M
)
_CTEST_LINE = re.compile(
    r"^\s*\d+/\d+ Test\s+#\d+: (\S+) \.+\s*(\*{0,3}\s*(?:Passed|Failed|Not Run|Timeout|Exception)[^\n]*)",
    re.M,
)
_CTEST_SUMMARY = re.compile(r"(\d+)% tests passed, (\d+) tests? failed out of (\d+)")
_JEST_TOTALS = re.compile(r"^Tests:\s+(.+)$", re.M)
_JEST_FAILED = re.compile(r"^\s*●\s+(.+?)\s*$", re.M)
_VITEST_TOTALS = re.compile(r"^\s*Tests\s+(.+)$", re.M)
_VITEST_FAILED = re.compile(r"^\s*(?:FAIL|×|✗)\s+(\S+\s*>\s*.+?)\s*(?:\d+ms)?$", re.M)
_VITEST_VERSION = re.compile(r"\bRUN\s+v(\d+\.\d+\.\d+)")
_JEST_PASSED_FILE = re.compile(r"^PASS\s+(\S+)", re.M)


def _counts(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for number, label in re.findall(r"(\d+)\s+(passed|failed|skipped|todo|errors?)", text):
        key = "errors" if label.startswith("error") else label
        out[key] = out.get(key, 0) + int(number)
    return out


def parse_output(
    framework: str, text: str, exit_code: int | None, *, targets: Sequence[str] = ()
) -> RunOutcome:
    """Parse a runner's output into counts and test identities."""
    tail = text[-200_000:]
    passed_ids: tuple[str, ...] = ()
    version = ""
    if framework == Framework.PYTEST.value:
        counts: dict[str, int] = {}
        summary_lines = [ln for ln in tail.splitlines() if _PYTEST_COUNTS.search(ln)]
        if summary_lines:
            for number, label in _PYTEST_COUNTS.findall(summary_lines[-1]):
                key = "errors" if label.startswith("error") else label
                counts[key] = counts.get(key, 0) + int(number)
        failed_ids = tuple(dict.fromkeys(m.group(1) for m in _PYTEST_FAILED.finditer(tail)))
        m = _PYTEST_VERSION.search(text[:4000])
        version = m.group(1) if m else ""
        failed = counts.get("failed", 0)
        errors = counts.get("errors", 0)
        if exit_code == 0 and targets:
            passed_ids = tuple(targets)
        return RunOutcome(
            framework,
            exit_code,
            passed=counts.get("passed", 0),
            failed=failed,
            skipped=counts.get("skipped", 0),
            errors=errors,
            failed_ids=failed_ids,
            passed_ids=passed_ids,
            runner_version=version,
        )
    if framework == Framework.CARGO.value:
        failed_ids = tuple(m.group(1) for m in _CARGO_TEST.finditer(tail) if m.group(2) == "FAILED")
        passed_ids = tuple(m.group(1) for m in _CARGO_TEST.finditer(tail) if m.group(2) == "ok")
        passed = failed = ignored = 0
        for p, f, i in _CARGO_RESULT.findall(tail):
            passed, failed, ignored = passed + int(p), failed + int(f), ignored + int(i)
        if not _CARGO_RESULT.search(tail) and exit_code not in (0, None):
            failed = max(1, len(failed_ids))
        return RunOutcome(
            framework,
            exit_code,
            passed=passed,
            failed=failed,
            skipped=ignored,
            failed_ids=failed_ids,
            passed_ids=passed_ids,
            runner_version=_tool_version("cargo"),
        )
    if framework == Framework.CTEST.value:
        failed_ids_l: list[str] = []
        passed_l: list[str] = []
        for name, status in _CTEST_LINE.findall(tail):
            (passed_l if "Passed" in status else failed_ids_l).append(name)
        m = _CTEST_SUMMARY.search(tail)
        total = int(m.group(3)) if m else len(passed_l) + len(failed_ids_l)
        failed = int(m.group(2)) if m else len(failed_ids_l)
        return RunOutcome(
            framework,
            exit_code,
            passed=max(0, total - failed),
            failed=failed,
            failed_ids=tuple(failed_ids_l),
            passed_ids=tuple(passed_l),
            runner_version=_tool_version("ctest"),
        )
    if framework == Framework.JEST.value:
        m = _JEST_TOTALS.search(tail)
        counts = _counts(m.group(1)) if m else {}
        failed_ids = tuple(
            dict.fromkeys(
                x.strip() for x in _JEST_FAILED.findall(tail) if "Test suite failed" not in x
            )
        )
        return RunOutcome(
            framework,
            exit_code,
            passed=counts.get("passed", 0),
            failed=counts.get("failed", 0),
            skipped=counts.get("skipped", 0) + counts.get("todo", 0),
            failed_ids=failed_ids,
            passed_ids=tuple(_JEST_PASSED_FILE.findall(tail)),
        )
    if framework == Framework.VITEST.value:
        totals = [ln for ln in _VITEST_TOTALS.findall(tail) if re.search(r"\d", ln)]
        counts = _counts(totals[-1]) if totals else {}
        failed_ids = tuple(dict.fromkeys(x.strip() for x in _VITEST_FAILED.findall(tail)))
        m = _VITEST_VERSION.search(text[:4000])
        return RunOutcome(
            framework,
            exit_code,
            passed=counts.get("passed", 0),
            failed=counts.get("failed", 0),
            skipped=counts.get("skipped", 0) + counts.get("todo", 0),
            failed_ids=failed_ids,
            runner_version=m.group(1) if m else "",
        )
    counts = _counts(tail[-4000:])
    return RunOutcome(
        framework,
        exit_code,
        passed=counts.get("passed", 0),
        failed=counts.get("failed", 0) or (1 if exit_code not in (0, None) else 0),
        skipped=counts.get("skipped", 0),
    )


_VERSIONS: dict[str, str] = {}
_VERSIONS_LOCK = threading.Lock()


def _tool_version(tool: str) -> str:
    """``<tool> --version`` once per process (empty when unavailable)."""
    with _VERSIONS_LOCK:
        if tool in _VERSIONS:
            return _VERSIONS[tool]
    from headroom.agent.exec_runner import resolve_executable

    version = ""
    exe = resolve_executable(tool)
    if exe is not None:
        from headroom._subprocess import run

        try:
            result = run([exe, "--version"], capture_output=True, timeout=10, check=False)
            text = result.stdout.decode("utf-8", "replace") if result.stdout else ""
            m = re.search(r"(\d+\.\d+(?:\.\d+)?)", text)
            version = m.group(1) if m else ""
        except Exception:  # noqa: BLE001 - version is metadata only
            version = ""
    with _VERSIONS_LOCK:
        _VERSIONS[tool] = version
    return version


def framework_of_command(command: str) -> str | None:
    """Which framework a client-issued test command used (observed runs)."""
    text = command.lower()
    if re.search(r"\b(?:py\.test|pytest)\b", text):
        return Framework.PYTEST.value
    if re.search(r"\bcargo\s+(?:test|nextest)\b", text):
        return Framework.CARGO.value
    if re.search(r"\bctest\b", text):
        return Framework.CTEST.value
    if re.search(r"\bvitest\b", text):
        return Framework.VITEST.value
    if re.search(r"\bjest\b", text):
        return Framework.JEST.value
    if re.search(r"\b(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?test\b", text):
        return Framework.CANONICAL.value
    return None


# ---------------------------------------------------------------------------
# Framework adapters
# ---------------------------------------------------------------------------

_SAFE_TOKEN = re.compile(r"^[\w./\\:+=@%,\[\]-]+$")


class Adapter:
    framework: Framework

    def detect(self, ctx: ProjectContext) -> bool:
        raise NotImplementedError

    def config_files(self, ctx: ProjectContext) -> list[str]:
        return []

    def is_test_file(self, rel: str) -> bool:
        return False

    def test_files(self, ctx: ProjectContext) -> list[str]:
        return [f for f in ctx.files if self.is_test_file(f)]

    def targets_in_file(
        self, ctx: ProjectContext, rel: str, reason: str, source: str
    ) -> list[TestTarget]:
        return [TestTarget(rel, self.framework.value, reason, source, rel)]

    def source_targets(self, ctx: ProjectContext, changed: str) -> list[TestTarget]:
        """Framework-native mapping of a changed source file (module filters etc.)."""
        return []

    def neighbor_targets(self, ctx: ProjectContext, changed: str) -> list[TestTarget]:
        return []

    def commands(
        self, ctx: ProjectContext, tier: int, targets: Sequence[TestTarget]
    ) -> list[TestCommand]:
        raise NotImplementedError

    def full_command(self, ctx: ProjectContext) -> TestCommand | None:
        raise NotImplementedError

    def collect_argv(self, ctx: ProjectContext, targets: Sequence[str]) -> list[str] | None:
        return None

    def count_tests(self, ctx: ProjectContext) -> int:
        return len(self.test_files(ctx))


def _history_prefix(
    ctx: ProjectContext, framework: str, marker: re.Pattern[str]
) -> list[str] | None:
    """The project's own runner prefix from an observed command (up to the runner word)."""
    command = ctx.history_commands.get(framework)
    if not command:
        return None
    try:
        words = shlex.split(command, posix=sys.platform != "win32")
    except ValueError:
        return None
    for idx, word in enumerate(words):
        if marker.search(os.path.basename(word).lower()):
            prefix = words[: idx + 1]
            if prefix and all(_SAFE_TOKEN.match(w) for w in prefix):
                if idx + 1 < len(words) and words[idx] in ("cargo",):
                    return None
                return prefix
    return None


class PytestAdapter(Adapter):
    framework = Framework.PYTEST
    _FN = re.compile(r"^(?:async\s+)?def\s+(test\w*)\s*\(", re.M)
    _CLASS = re.compile(r"^class\s+(Test\w*)\b")
    _METHOD = re.compile(r"^\s+(?:async\s+)?def\s+(test\w*)\s*\(")

    def detect(self, ctx: ProjectContext) -> bool:
        for name in ("pytest.ini", "conftest.py", "tox.ini", "setup.cfg", "pyproject.toml"):
            if ctx.exists(name):
                text = ctx.read(name)
                if name in ("pytest.ini", "conftest.py") or re.search(
                    r"\[(?:tool:pytest|pytest|tool\.pytest[^\]]*)\]", text
                ):
                    return True
        return any(self.is_test_file(f) for f in ctx.files)

    def config_files(self, ctx: ProjectContext) -> list[str]:
        return [f for f in ctx.files if f.rsplit("/", 1)[-1] == "conftest.py"][:32]

    def is_test_file(self, rel: str) -> bool:
        name = rel.rsplit("/", 1)[-1]
        return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))

    def node_ids(self, ctx: ProjectContext, rel: str) -> list[str]:
        ids: list[str] = []
        current_class: str | None = None
        for line in ctx.read(rel).splitlines():
            if not line.strip():
                continue
            cls = self._CLASS.match(line)
            if cls:
                current_class = cls.group(1)
                continue
            if not line[0].isspace():
                current_class = None
                fn = self._FN.match(line)
                if fn:
                    ids.append(f"{rel}::{fn.group(1)}")
                continue
            if current_class is not None:
                meth = self._METHOD.match(line)
                if meth and len(line) - len(line.lstrip()) == 4:
                    ids.append(f"{rel}::{current_class}::{meth.group(1)}")
        return ids

    def targets_in_file(
        self, ctx: ProjectContext, rel: str, reason: str, source: str
    ) -> list[TestTarget]:
        ids = self.node_ids(ctx, rel)
        if not ids:
            return [TestTarget(rel, self.framework.value, reason, source, rel)]
        return [TestTarget(i, self.framework.value, reason, source, rel) for i in ids]

    def coverage_targets(self, ctx: ProjectContext, changed: Sequence[str]) -> list[TestTarget]:
        """Tests whose recorded coverage context touched a changed file (coverage.py)."""
        path = os.path.join(ctx.root, ".coverage")
        if not os.path.isfile(path) or not changed:
            return []
        import sqlite3

        out: list[TestTarget] = []
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1)
        except sqlite3.Error:
            return []
        try:
            tables = {
                r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if not {"file", "context", "line_bits"} <= tables:
                return []
            files = conn.execute("SELECT id, path FROM file").fetchall()
            wanted = {}
            for fid, fpath in files:
                norm = str(fpath).replace("\\", "/")
                for rel in changed:
                    if norm.lower().endswith("/" + rel.lower()) or norm.lower() == rel.lower():
                        wanted[int(fid)] = rel
            for fid, rel in sorted(wanted.items()):
                rows = conn.execute(
                    "SELECT DISTINCT c.context FROM line_bits lb JOIN context c "
                    "ON c.id = lb.context_id WHERE lb.file_id=?",
                    (fid,),
                ).fetchall()
                for (context,) in rows:
                    node = str(context or "").split("|", 1)[0].strip()
                    if "::" in node and self.is_test_file(node.split("::", 1)[0]):
                        out.append(
                            TestTarget(
                                node,
                                self.framework.value,
                                Reason.COVERAGE.value,
                                rel,
                                node.split("::", 1)[0],
                            )
                        )
        except sqlite3.Error as exc:
            log.debug("coverage read failed: %s", describe_exception(exc))
            return []
        finally:
            conn.close()
        return out

    def prefix(self, ctx: ProjectContext) -> list[str]:
        found = _history_prefix(
            ctx, self.framework.value, re.compile(r"^(?:py\.test|pytest)(?:\.exe)?$")
        )
        if found:
            return found
        for venv in (".venv", "venv", "env"):
            for rel in ("Scripts/python.exe", "bin/python"):
                candidate = os.path.join(ctx.root, venv, rel)
                if os.path.isfile(candidate):
                    return [candidate, "-m", "pytest"]
        from headroom.agent.exec_runner import resolve_executable

        if resolve_executable("pytest"):
            return ["pytest"]
        return [sys.executable, "-m", "pytest"]

    def commands(
        self, ctx: ProjectContext, tier: int, targets: Sequence[TestTarget]
    ) -> list[TestCommand]:
        ids = list(dict.fromkeys(t.id for t in targets))
        if not ids:
            return []
        return [TestCommand(self.framework.value, tier, (*self.prefix(ctx), *ids), ".", tuple(ids))]

    def full_command(self, ctx: ProjectContext) -> TestCommand | None:
        return TestCommand(self.framework.value, 3, tuple(self.prefix(ctx)), ".", ())

    def collect_argv(self, ctx: ProjectContext, targets: Sequence[str]) -> list[str] | None:
        return [*self.prefix(ctx), "--collect-only", "-q", *targets]

    def count_tests(self, ctx: ProjectContext) -> int:
        return sum(max(1, len(self.node_ids(ctx, f))) for f in self.test_files(ctx))


class CargoAdapter(Adapter):
    framework = Framework.CARGO
    _TEST_FN = re.compile(r"#\[(?:tokio::)?test[^\]]*\]\s*(?:async\s+)?fn\s+(\w+)")

    def detect(self, ctx: ProjectContext) -> bool:
        return ctx.exists("Cargo.toml")

    def config_files(self, ctx: ProjectContext) -> list[str]:
        return [f for f in ctx.files if f.rsplit("/", 1)[-1] in ("Cargo.toml", "build.rs")][:64]

    def is_test_file(self, rel: str) -> bool:
        return rel.endswith(".rs") and ("/tests/" in f"/{rel}")

    def metadata(self, ctx: ProjectContext) -> dict[str, Any]:
        cached = ctx.cache.get("cargo_metadata")
        if cached is not None:
            return cached
        from headroom.agent.exec_runner import resolve_executable, run_argv

        data: dict[str, Any] = {}
        if resolve_executable("cargo") is not None:
            result = run_argv(
                ["cargo", "metadata", "--no-deps", "--format-version", "1", "--offline"],
                cwd=ctx.root,
                timeout_s=DISCOVERY_TIMEOUT_S,
                capture_bytes=8 * 1024 * 1024,
            )
            if result.ok:
                try:
                    data = json.loads(result.output[result.output.find("{") :])
                except ValueError:
                    data = {}
        if not data:
            data = self._static_metadata(ctx)
        ctx.cache["cargo_metadata"] = data
        return data

    def _static_metadata(self, ctx: ProjectContext) -> dict[str, Any]:
        packages = []
        for manifest in self.config_files(ctx):
            if not manifest.endswith("Cargo.toml"):
                continue
            text = ctx.read(manifest)
            m = re.search(r"^\[package\][^\[]*?^name\s*=\s*\"([^\"]+)\"", text, re.M | re.S)
            if m:
                packages.append(
                    {
                        "name": m.group(1),
                        "manifest_path": os.path.join(ctx.root, manifest),
                        "targets": [],
                        "dependencies": [],
                    }
                )
        return {"packages": packages, "workspace_root": ctx.root, "static": True}

    def packages(self, ctx: ProjectContext) -> list[dict[str, Any]]:
        out = []
        for pkg in self.metadata(ctx).get("packages") or ():
            manifest = str(pkg.get("manifest_path") or "")
            rel_dir = os.path.relpath(os.path.dirname(manifest), ctx.root).replace("\\", "/")
            out.append({**pkg, "rel_dir": "" if rel_dir == "." else rel_dir})
        return sorted(out, key=lambda p: (-len(p["rel_dir"]), p["name"]))

    def package_of(self, ctx: ProjectContext, rel: str) -> dict[str, Any] | None:
        for pkg in self.packages(ctx):
            prefix = pkg["rel_dir"]
            if not prefix or rel == prefix or rel.startswith(prefix + "/"):
                return pkg
        return None

    def _module_filter(self, pkg: dict[str, Any], rel: str) -> str | None:
        inner = rel[len(pkg["rel_dir"]) + 1 :] if pkg["rel_dir"] else rel
        if not inner.startswith("src/") or not inner.endswith(".rs"):
            return None
        mod = inner[4:-3]
        if mod.endswith("/mod"):
            mod = mod[:-4]
        if mod in ("lib", "main"):
            return ""
        return mod.replace("/", "::") + "::"

    def source_targets(self, ctx: ProjectContext, changed: str) -> list[TestTarget]:
        pkg = self.package_of(ctx, changed)
        if pkg is None or not changed.endswith(".rs"):
            return []
        name = pkg["name"]
        if self.is_test_file(changed):
            test = changed.rsplit("/", 1)[-1][:-3]
            return [
                TestTarget(
                    f"{name}:test:{test}",
                    self.framework.value,
                    Reason.CHANGED_TEST.value,
                    changed,
                    changed,
                    name,
                )
            ]
        filt = self._module_filter(pkg, changed)
        if filt is None:
            return []
        out = [
            TestTarget(
                f"{name}:lib:{filt}" if filt else f"{name}:lib",
                self.framework.value,
                Reason.MODULE_FILTER.value,
                changed,
                changed,
                name,
            )
        ]
        # Integration tests of this crate that use the changed module.
        crate_ident = name.replace("-", "_")
        module = filt.rstrip(":").split("::")[0] if filt else ""
        for test in self.test_files(ctx):
            if (
                self.package_of(ctx, test) is not pkg
                and (self.package_of(ctx, test) or {}).get("name") != name
            ):
                continue
            text = ctx.read(test)
            if module and re.search(rf"\b{re.escape(crate_ident)}::{re.escape(module)}\b", text):
                tname = test.rsplit("/", 1)[-1][:-3]
                out.append(
                    TestTarget(
                        f"{name}:test:{tname}",
                        self.framework.value,
                        Reason.DIRECT_IMPORT.value,
                        changed,
                        test,
                        name,
                    )
                )
        return out

    def neighbor_targets(self, ctx: ProjectContext, changed: str) -> list[TestTarget]:
        pkg = self.package_of(ctx, changed)
        if pkg is None:
            return []
        out = [
            TestTarget(
                f"{pkg['name']}:package",
                self.framework.value,
                Reason.PACKAGE.value,
                changed,
                None,
                pkg["name"],
            )
        ]
        for other in self.packages(ctx):
            for dep in other.get("dependencies") or ():
                if dep.get("name") == pkg["name"] and dep.get("path"):
                    out.append(
                        TestTarget(
                            f"{other['name']}:package",
                            self.framework.value,
                            Reason.DEPENDENCY.value,
                            changed,
                            None,
                            other["name"],
                        )
                    )
        return out

    def commands(
        self, ctx: ProjectContext, tier: int, targets: Sequence[TestTarget]
    ) -> list[TestCommand]:
        out: list[TestCommand] = []
        seen: set[tuple[str, ...]] = set()
        for t in targets:
            pkg, kind, *rest = t.id.split(":", 2) + [""]
            arg = rest[0] if rest else ""
            if kind == "lib":
                argv = ("cargo", "test", "-p", pkg, "--lib", *((arg,) if arg else ()))
            elif kind == "test":
                argv = ("cargo", "test", "-p", pkg, "--test", arg)
            elif kind == "package":
                argv = ("cargo", "test", "-p", pkg)
            else:
                continue
            if argv not in seen:
                seen.add(argv)
                out.append(TestCommand(self.framework.value, tier, argv, ".", (t.id,)))
        return out

    def full_command(self, ctx: ProjectContext) -> TestCommand | None:
        workspace = "[workspace]" in ctx.read("Cargo.toml")
        argv = ("cargo", "test", "--workspace") if workspace else ("cargo", "test")
        return TestCommand(self.framework.value, 3, argv, ".", ())

    def count_tests(self, ctx: ProjectContext) -> int:
        return (
            sum(len(self._TEST_FN.findall(ctx.read(f))) for f in ctx.files if f.endswith(".rs"))
            or 1
        )


class CTestAdapter(Adapter):
    framework = Framework.CTEST
    _ADD_TEST = re.compile(r"add_test\s*\(\s*(?:NAME\s+)?([\w.:-]+)", re.I)

    def detect(self, ctx: ProjectContext) -> bool:
        text = ctx.read("CMakeLists.txt") if ctx.exists("CMakeLists.txt") else ""
        return bool(text) and bool(
            re.search(r"\b(enable_testing|add_test|include\s*\(\s*CTest)", text, re.I)
        )

    def build_dir(self, ctx: ProjectContext) -> str | None:
        """An *existing* configured build directory (never invented)."""
        if "ctest_build" in ctx.cache:
            return ctx.cache["ctest_build"]
        found: str | None = None
        candidates = [
            "build",
            "_build",
            "builddir",
            "out/build",
            "cmake-build-debug",
            "cmake-build-release",
        ]
        try:
            for name in sorted(os.listdir(ctx.root)):
                if name.startswith("cmake-build-") and name not in candidates:
                    candidates.append(name)
            out_build = os.path.join(ctx.root, "out", "build")
            if os.path.isdir(out_build):
                candidates += [f"out/build/{n}" for n in sorted(os.listdir(out_build))]
        except OSError:
            pass
        for rel in candidates:
            full = os.path.join(ctx.root, rel)
            if os.path.isfile(os.path.join(full, "CTestTestfile.cmake")) and os.path.isfile(
                os.path.join(full, "CMakeCache.txt")
            ):
                found = rel
                break
        ctx.cache["ctest_build"] = found
        return found

    def config_args(self, ctx: ProjectContext) -> list[str]:
        """``-C <config>`` for a multi-config build tree (the configuration already built)."""
        build = self.build_dir(ctx)
        if build is None:
            return []
        cache = ctx.read(f"{build}/CMakeCache.txt")
        m = re.search(r"^CMAKE_CONFIGURATION_TYPES:\w+=(.+)$", cache, re.M)
        if not m:
            return []
        configs = [c for c in m.group(1).strip().split(";") if c]
        built = [c for c in configs if os.path.isdir(os.path.join(ctx.root, build, c))]
        return ["-C", (built or configs or ["Debug"])[0]]

    def config_files(self, ctx: ProjectContext) -> list[str]:
        out = [f for f in ctx.files if f.endswith(("CMakeLists.txt", ".cmake"))][:64]
        build = self.build_dir(ctx)
        if build:
            out += [f"{build}/CMakeCache.txt", f"{build}/CTestTestfile.cmake"]
        return out

    def tests(self, ctx: ProjectContext) -> list[dict[str, Any]]:
        cached = ctx.cache.get("ctest_tests")
        if cached is not None:
            return cached
        build = self.build_dir(ctx)
        tests: list[dict[str, Any]] = []
        if build is not None:
            from headroom.agent.exec_runner import resolve_executable, run_argv

            if resolve_executable("ctest") is not None:
                version = _tool_version("ctest")
                json_ok = _version_at_least(version, (3, 14))
                argv = ["ctest", *self.config_args(ctx)]
                argv += ["--show-only=json-v1"] if json_ok else ["-N"]
                result = run_argv(
                    argv,
                    cwd=os.path.join(ctx.root, build),
                    timeout_s=DISCOVERY_TIMEOUT_S,
                    capture_bytes=8 * 1024 * 1024,
                )
                if result.ok and json_ok:
                    try:
                        data = json.loads(result.output[result.output.find("{") :])
                    except ValueError:
                        data = {}
                    for test in data.get("tests") or ():
                        props = {
                            p.get("name"): p.get("value") for p in test.get("properties") or ()
                        }
                        tests.append(
                            {
                                "name": str(test.get("name") or ""),
                                "command": [str(c) for c in test.get("command") or ()],
                                "workdir": str(props.get("WORKING_DIRECTORY") or ""),
                                "labels": list(props.get("LABELS") or ()),
                            }
                        )
                elif result.ok:
                    for name in re.findall(r"Test\s+#\d+:\s+(\S+)", result.output):
                        tests.append({"name": name, "command": [], "workdir": "", "labels": []})
        ctx.cache["ctest_tests"] = tests
        return tests

    def _declared(self, ctx: ProjectContext) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for rel in ctx.files:
            if rel.endswith("CMakeLists.txt"):
                directory = rel.rsplit("/", 1)[0] if "/" in rel else ""
                out[directory] = self._ADD_TEST.findall(ctx.read(rel))
        return out

    def source_targets(self, ctx: ProjectContext, changed: str) -> list[TestTarget]:
        tests = self.tests(ctx)
        if not tests:
            return []
        stem = changed.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
        stem_core = re.sub(r"^(?:test_|test)|(?:_test|_tests|test)$", "", stem) or stem
        base = changed.rsplit("/", 1)[-1]
        out: list[TestTarget] = []
        for test in tests:
            name = test["name"]
            lname = name.lower()
            if len(stem_core) >= 3 and stem_core in lname:
                out.append(
                    TestTarget(
                        name, self.framework.value, Reason.NAMING.value, changed, None, "ctest"
                    )
                )
            elif any(base in os.path.basename(arg) for arg in test["command"][1:]):
                out.append(
                    TestTarget(
                        name,
                        self.framework.value,
                        Reason.DIRECT_IMPORT.value,
                        changed,
                        None,
                        "ctest",
                    )
                )
        return out

    def neighbor_targets(self, ctx: ProjectContext, changed: str) -> list[TestTarget]:
        names = {t["name"] for t in self.tests(ctx)}
        directory = changed.rsplit("/", 1)[0] if "/" in changed else ""
        declared = self._declared(ctx)
        out: list[TestTarget] = []
        probe = directory
        while True:
            for name in declared.get(probe, ()):
                if name in names:
                    out.append(
                        TestTarget(
                            name,
                            self.framework.value,
                            Reason.DEPENDENCY.value,
                            changed,
                            None,
                            "ctest",
                        )
                    )
            if not probe:
                break
            probe = probe.rsplit("/", 1)[0] if "/" in probe else ""
            if out:
                break
        return out

    def commands(
        self, ctx: ProjectContext, tier: int, targets: Sequence[TestTarget]
    ) -> list[TestCommand]:
        build = self.build_dir(ctx)
        names = list(dict.fromkeys(t.id for t in targets))
        if build is None or not names:
            return []
        regex = "^(" + "|".join(re.escape(n) for n in names) + ")$"
        argv = ("ctest", *self.config_args(ctx), "-R", regex, "--output-on-failure")
        return [TestCommand(self.framework.value, tier, argv, build, tuple(names))]

    def full_command(self, ctx: ProjectContext) -> TestCommand | None:
        build = self.build_dir(ctx)
        if build is None:
            return None
        argv = ("ctest", *self.config_args(ctx), "--output-on-failure")
        return TestCommand(self.framework.value, 3, argv, build, ())

    def count_tests(self, ctx: ProjectContext) -> int:
        return len(self.tests(ctx)) or 1


def _version_at_least(version: str, minimum: tuple[int, ...]) -> bool:
    try:
        parts = tuple(int(p) for p in version.split(".")[: len(minimum)])
    except ValueError:
        return False
    return bool(parts) and parts >= minimum


class JsAdapter(Adapter):
    _TEST_FILE = re.compile(r"(?:^|/)(?:__tests__/.+|[^/]+\.(?:test|spec))\.[cm]?[jt]sx?$")

    def __init__(self, framework: Framework) -> None:
        self.framework = framework

    def package(self, ctx: ProjectContext) -> dict[str, Any]:
        cached = ctx.cache.get("package_json")
        if cached is None:
            try:
                cached = json.loads(ctx.read("package.json") or "{}")
            except ValueError:
                cached = {}
            ctx.cache["package_json"] = cached if isinstance(cached, dict) else {}
        return ctx.cache["package_json"]

    def runner(self, ctx: ProjectContext) -> str | None:
        pkg = self.package(ctx)
        deps = {**(pkg.get("dependencies") or {}), **(pkg.get("devDependencies") or {})}
        script = str((pkg.get("scripts") or {}).get("test") or "")
        for name in ("vitest", "jest"):
            if name in deps or re.search(rf"\b{name}\b", script):
                return name
            if any(f.startswith(f"{name}.config.") for f in ctx.files if "/" not in f):
                return name
        return None

    def detect(self, ctx: ProjectContext) -> bool:
        return ctx.exists("package.json") and self.runner(ctx) == self.framework.value

    def package_manager(self, ctx: ProjectContext) -> str:
        declared = str(self.package(ctx).get("packageManager") or "")
        for name in ("pnpm", "yarn", "bun", "npm"):
            if declared.startswith(name + "@") or declared == name:
                return name
        for lock, name in (
            ("pnpm-lock.yaml", "pnpm"),
            ("yarn.lock", "yarn"),
            ("bun.lockb", "bun"),
            ("bun.lock", "bun"),
            ("package-lock.json", "npm"),
        ):
            if ctx.exists(lock):
                return name
        return "npm"

    def exec_prefix(self, ctx: ProjectContext) -> list[str]:
        return {
            "pnpm": ["pnpm", "exec"],
            "yarn": ["yarn"],
            "bun": ["bunx"],
            "npm": ["npx", "--no-install"],
        }[self.package_manager(ctx)]

    def config_files(self, ctx: ProjectContext) -> list[str]:
        return [
            f
            for f in ctx.files
            if re.match(r"^(?:jest|vitest|vite)\.config\.", f) or f == "tsconfig.json"
        ]

    def is_test_file(self, rel: str) -> bool:
        return "node_modules/" not in rel and bool(self._TEST_FILE.search(rel))

    def commands(
        self, ctx: ProjectContext, tier: int, targets: Sequence[TestTarget]
    ) -> list[TestCommand]:
        files = list(dict.fromkeys(t.file or t.id for t in targets))
        if not files:
            return []
        if self.framework is Framework.JEST:
            argv = (*self.exec_prefix(ctx), "jest", "--ci", "--runTestsByPath", *files)
        else:
            argv = (*self.exec_prefix(ctx), "vitest", "run", *files)
        return [TestCommand(self.framework.value, tier, argv, ".", tuple(files))]

    def full_command(self, ctx: ProjectContext) -> TestCommand | None:
        if self.framework is Framework.JEST:
            argv = (*self.exec_prefix(ctx), "jest", "--ci")
        else:
            argv = (*self.exec_prefix(ctx), "vitest", "run")
        return TestCommand(self.framework.value, 3, argv, ".", ())


class CanonicalAdapter(Adapter):
    """Fallback: the project's canonical test command, else explicit unknown."""

    framework = Framework.CANONICAL

    def detect(self, ctx: ProjectContext) -> bool:
        return self.full_command(ctx) is not None

    def full_command(self, ctx: ProjectContext) -> TestCommand | None:
        if ctx.exists("Makefile") and re.search(r"^test\s*:", ctx.read("Makefile"), re.M):
            return TestCommand(self.framework.value, 3, ("make", "test"), ".", ())
        if ctx.exists("go.mod"):
            return TestCommand(self.framework.value, 3, ("go", "test", "./..."), ".", ())
        if ctx.exists("tox.ini"):
            return TestCommand(self.framework.value, 3, ("tox",), ".", ())
        if ctx.exists("package.json"):
            try:
                pkg = json.loads(ctx.read("package.json") or "{}")
            except ValueError:
                pkg = {}
            if isinstance(pkg, dict) and (pkg.get("scripts") or {}).get("test"):
                pm = JsAdapter(Framework.JEST).package_manager(ctx)
                return TestCommand(self.framework.value, 3, (pm, "test"), ".", ())
        return None

    def commands(
        self, ctx: ProjectContext, tier: int, targets: Sequence[TestTarget]
    ) -> list[TestCommand]:
        return []


def adapters() -> list[Adapter]:
    return [
        PytestAdapter(),
        CargoAdapter(),
        CTestAdapter(),
        JsAdapter(Framework.JEST),
        JsAdapter(Framework.VITEST),
    ]


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

_BUILD_CONFIG = re.compile(
    r"(?:^|/)(?:pyproject\.toml|setup\.py|setup\.cfg|requirements[^/]*\.txt|tox\.ini|pytest\.ini|"
    r"conftest\.py|noxfile\.py|Cargo\.toml|Cargo\.lock|build\.rs|CMakeLists\.txt|[^/]+\.cmake|"
    r"Makefile|package\.json|package-lock\.json|pnpm-lock\.yaml|yarn\.lock|bun\.lockb?|"
    r"tsconfig[^/]*\.json|(?:jest|vitest|vite)\.config\.[^/]+|Dockerfile|go\.mod|go\.sum)$|"
    r"(?:^|/)\.github/workflows/",
    re.I,
)
_SECURITY = re.compile(
    r"(?:^|[/_.-])(?:auth\w*|security|crypto\w*|secrets?|tokens?|permissions?|sandbox|"
    r"credentials?|passwords?|acl|tls|ssl|oauth|jwt)(?:[/_.-]|$)",
    re.I,
)
_WIRE = re.compile(
    r"(?:^|[/_.-])(?:wire|protocol|proto|serializ\w*|codec|schema|openapi|grpc|handlers?)(?:[/_.-]|$)|\.proto$",
    re.I,
)
_PLATFORM = re.compile(
    r"(?:^|[/_.-])(?:windows|win32|win|posix|linux|darwin|macos|platform|nt)(?:[/_.-]|$)|"
    r"\.(?:ps1|bat|cmd|sh)$",
    re.I,
)
_PUBLIC_API = re.compile(
    r"(?:^|/)(?:__init__\.py|lib\.rs|mod\.rs|index\.[cm]?[jt]sx?)$|\.(?:pyi|h|hpp|hh|d\.ts)$"
)

#: Changes whose class makes the full suite mandatory (even when Tier 1 passes).
MANDATORY_FULL = frozenset(
    {
        "risk.public_api",
        "risk.security",
        "risk.build_config",
        "risk.provider_wire",
        "risk.platform",
        "risk.final_completion",
    }
)


def risk_of(
    changed: Sequence[str],
    *,
    lines_changed: int,
    importer_counts: Mapping[str, int],
    frameworks: Sequence[str],
    history_fail_ratio: float,
    uncertain_evidence: bool,
    uncovered: Sequence[str],
    drift: Sequence[str],
    flaky_in_scope: bool,
    final_completion: bool,
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0

    def add(code: str, weight: float) -> None:
        nonlocal score
        if code not in reasons:
            reasons.append(code)
            score += weight

    for rel in changed:
        if _BUILD_CONFIG.search(rel):
            add("risk.build_config", 0.35)
        if _SECURITY.search(rel):
            add("risk.security", 0.35)
        if _WIRE.search(rel):
            add("risk.provider_wire", 0.3)
        if _PLATFORM.search(rel):
            add("risk.platform", 0.25)
        if _PUBLIC_API.search(rel) or importer_counts.get(rel, 0) >= 3:
            add("risk.public_api", 0.3)
    if history_fail_ratio > 0:
        add("risk.history_failures", min(0.2, 0.2 * history_fail_ratio + 0.05))
    if len(frameworks) > 1:
        add("risk.platform_breadth", min(0.2, 0.1 * (len(frameworks) - 1)))
    if uncertain_evidence:
        add("risk.evidence_uncertain", 0.15)
    size = min(0.25, len(changed) / 40 + lines_changed / 2000)
    if size >= 0.05:
        add("risk.change_size", size)
    if drift:
        add("risk.scope_drift", 0.2)
    if uncovered:
        add("risk.uncovered_change", 0.15)
    if flaky_in_scope:
        add("risk.flaky_in_scope", 0.15)
    if final_completion:
        add("risk.final_completion", 0.2)
    return min(1.0, round(score, 4)), reasons


# ---------------------------------------------------------------------------
# History (per workspace, bounded, persisted through the agent ledger)
# ---------------------------------------------------------------------------


class RunHistory:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, OrderedDict[str, deque[tuple[str, bool, float]]]] = {}
        self._db: Any = None
        self._submit: Any = None

    def bind(self, ledger: Any, submit: Any) -> None:
        with ledger.lock:
            ledger.connection.execute(_TABLE)
            ledger.connection.execute(_INDEX)
        with self._lock:
            self._db, self._submit = ledger, submit
        submit(lambda: self._load(ledger))

    def unbind(self) -> None:
        with self._lock:
            self._db = self._submit = None

    def _load(self, db: Any) -> None:
        try:
            rows = db.query(
                "SELECT workspace_id, framework, test_id, changeset, ok, ts FROM verification_runs"
                " ORDER BY ts DESC LIMIT ?",
                (_HISTORY_TESTS * 4,),
            )
        except Exception as exc:  # noqa: BLE001 - memory-only history still works
            log.warning("verification history load failed: %s", describe_exception(exc))
            return
        for ws, fw, test_id, changeset, ok, ts in reversed(rows):
            self._add(
                str(ws), f"{fw}|{test_id}", str(changeset), bool(ok), float(ts), persist=False
            )

    def _add(
        self, ws: str, key: str, changeset: str, ok: bool, ts: float, *, persist: bool
    ) -> None:
        with self._lock:
            tests = self._runs.setdefault(ws, OrderedDict())
            runs = tests.setdefault(key, deque(maxlen=_HISTORY_PER_TEST))
            runs.append((changeset, ok, ts))
            tests.move_to_end(key)
            while len(tests) > _HISTORY_TESTS:
                tests.popitem(last=False)
            db, submit = self._db, self._submit
        if persist and db is not None and submit is not None:
            fw, _, test_id = key.partition("|")

            def write() -> None:
                db.transaction(
                    [
                        (
                            "INSERT INTO verification_runs(workspace_id, framework, test_id, changeset, ok, ts)"
                            " VALUES (?,?,?,?,?,?)",
                            (ws, fw, test_id[:300], changeset, int(ok), ts),
                        ),
                        (
                            "DELETE FROM verification_runs WHERE workspace_id=? AND rowid NOT IN ("
                            "SELECT rowid FROM verification_runs WHERE workspace_id=? ORDER BY ts DESC LIMIT ?)",
                            (ws, ws, _HISTORY_TESTS * 4),
                        ),
                    ]
                )

            submit(write)

    def record(
        self, ws: str, framework: str, test_id: str, changeset: str, ok: bool, ts: float
    ) -> None:
        self._add(ws, f"{framework}|{test_id}", changeset, ok, ts, persist=True)

    def flaky(self, ws: str, framework: str, test_id: str, changeset: str | None = None) -> bool:
        """Failed and passed on one unchanged changeset."""
        with self._lock:
            runs = list(self._runs.get(ws, {}).get(f"{framework}|{test_id}", ()))
        by_change: dict[str, set[bool]] = {}
        for cs, ok, _ in runs:
            if changeset is None or cs == changeset:
                by_change.setdefault(cs, set()).add(ok)
        return any(len(v) == 2 for v in by_change.values())

    def failing(self, ws: str) -> list[tuple[str, str, float]]:
        """(framework, test_id, ts) whose most recent run failed."""
        with self._lock:
            tests = dict(self._runs.get(ws, {}))
        out = []
        for key, runs in tests.items():
            if runs and not runs[-1][1]:
                fw, _, test_id = key.partition("|")
                out.append((fw, test_id, runs[-1][2]))
        return sorted(out, key=lambda r: (-r[2], r[1]))

    def failure_ratio(self, ws: str, keys: Iterable[tuple[str, str]]) -> float:
        with self._lock:
            tests = self._runs.get(ws, {})
            total = failed = 0
            for fw, test_id in keys:
                for _, ok, _ in tests.get(f"{fw}|{test_id}", ()):
                    total += 1
                    failed += 0 if ok else 1
        return failed / total if total else 0.0

    def flaky_ids(self, ws: str) -> list[tuple[str, str]]:
        with self._lock:
            keys = list(self._runs.get(ws, {}))
        out = []
        for key in keys:
            fw, _, test_id = key.partition("|")
            if self.flaky(ws, fw, test_id):
                out.append((fw, test_id))
        return sorted(out)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


@dataclass
class _CheckState:
    status: dict[str, str] = field(default_factory=dict)  # check -> passed|failed|flaky
    flaky: set[str] = field(default_factory=set)
    failed: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class _PendingRun:
    workspace_id: str
    session_id: str
    event_id: str
    framework: str
    command: str
    outcome: RunOutcome
    ts: float
    tier: int | None = None


class TestImpactPlanner:
    """Owner of verification plans and run history (see module docstring)."""

    __test__ = False

    def __init__(self, runtime: Any = None, *, clock: Any = time.time) -> None:
        self.runtime = runtime
        self._clock = clock
        self._lock = threading.RLock()
        self._plans: OrderedDict[tuple[str, str], VerificationPlan] = OrderedDict()
        self._checks: OrderedDict[str, _CheckState] = OrderedDict()
        self._roots: dict[str, str] = {}
        self._identities: dict[tuple[str, str], Any] = {}
        self._commands: dict[str, dict[str, str]] = {}
        self._last_refresh: dict[tuple[str, str], float] = {}
        self._dirty: set[tuple[str, str]] = set()
        self._pending: deque[_PendingRun] = deque(maxlen=256)
        self._expected: OrderedDict[str, tuple[int, tuple[str, ...]]] = OrderedDict()
        self.history = RunHistory()
        self.name = PLANNER_PROJECTOR
        self.counters: dict[str, int] = {
            "plans_built": 0,
            "plans_reused": 0,
            "plans_invalidated": 0,
            "runs_recorded": 0,
            "flaky_detected": 0,
            "full_suite_required": 0,
            "unknown_framework": 0,
            "stale_plan_refused": 0,
        }

    # -- identity -----------------------------------------------------------

    def note_identity(self, identity: Any) -> None:
        if identity is None or not identity.can_authorize_paths():
            return
        with self._lock:
            self._roots[identity.workspace_id] = identity.root
            self._identities[(identity.workspace_id, identity.session_id)] = identity
            while len(self._identities) > _PLANS_LIMIT:
                self._identities.pop(next(iter(self._identities)))

    # -- inputs -------------------------------------------------------------

    def _snapshot(self, identity: Any) -> Any:
        compiler = self.runtime.services.get("task_state") if self.runtime is not None else None
        if compiler is None:
            return None
        return compiler.snapshot(identity.workspace_id, identity.session_id)

    def _scope(self, identity: Any) -> Any:
        firewall = self.runtime.services.get("change_scope") if self.runtime is not None else None
        if firewall is None:
            return None
        try:
            return firewall.contract(identity)
        except Exception as exc:  # noqa: BLE001 - planning works without a contract
            log.debug("change contract unavailable: %s", describe_exception(exc))
            return None

    def _graph_dependents(
        self, identity: Any, changed: Sequence[str]
    ) -> tuple[set[str], set[str]] | None:
        adapter = self.runtime.services.get("code_graph") if self.runtime is not None else None
        if adapter is None:
            return None
        graph = adapter.ensure_workspace(identity)
        if graph is None or not graph.ready:
            return None
        direct = {
            r[5:] for r in adapter.affected_resources(identity.workspace_id, changed, radius=1)
        }
        wider = {
            r[5:] for r in adapter.affected_resources(identity.workspace_id, changed, radius=2)
        }
        return direct, wider - direct

    def revisions(self, identity: Any) -> tuple[str | None, int, int]:
        snap = self._snapshot(identity)
        task_id = snap.state.id if snap is not None else None
        task_rev = snap.state.revision if snap is not None else 0
        contract = self._scope(identity)
        return task_id, task_rev, int(contract.revision) if contract is not None else 0

    # -- planning -----------------------------------------------------------

    def current(self, identity: Any) -> VerificationPlan | None:
        """The cached plan for a session (may be stale; see :meth:`is_current`)."""
        with self._lock:
            return self._plans.get((identity.workspace_id, identity.session_id))

    def is_current(self, identity: Any, plan: VerificationPlan) -> bool:
        """True only when every pin (task, scope, changeset, runner config) still holds."""
        if not identity.can_authorize_paths():
            return False
        task_id, task_rev, scope_rev = self.revisions(identity)
        if (task_id, task_rev, scope_rev) != (
            plan.task_id,
            plan.task_revision,
            plan.scope_revision,
        ):
            return False
        root = identity.root
        writes = self._recorded_writes(identity)
        changeset = compute_changeset(root, writes)
        fingerprint = runner_fingerprint(root, self._extra_config(root))
        return (changeset.digest, fingerprint) == (plan.changeset_hash, plan.runner_fingerprint)

    def _recorded_writes(self, identity: Any) -> list[str]:
        snap = self._snapshot(identity)
        if snap is None:
            return []
        return [r[5:] for r in snap.state.resources if r.startswith("path:")]

    def _extra_config(self, root: str) -> list[str]:
        ctx = ProjectContext(root)
        build = CTestAdapter().build_dir(ctx) if ctx.exists("CMakeLists.txt") else None
        return [f"{build}/CMakeCache.txt", f"{build}/CTestTestfile.cmake"] if build else []

    def plan(
        self, identity: Any, *, final: bool | None = None, force: bool = False
    ) -> VerificationPlan | None:
        """Build (or reuse) the current plan. Worker / endpoint threads only."""
        if identity is None or not identity.can_authorize_paths():
            return None
        self.note_identity(identity)
        root = identity.root
        snap = self._snapshot(identity)
        task_id, task_rev, scope_rev = self.revisions(identity)
        changeset = compute_changeset(root, self._recorded_writes(identity))
        fingerprint = runner_fingerprint(root, self._extra_config(root))
        if final is None:
            final = self._final_completion(snap)
        key = (identity.workspace_id, identity.session_id)
        with self._lock:
            previous = self._plans.get(key)
        if (
            previous is not None
            and not force
            and previous.pins() == (task_id, task_rev, scope_rev, changeset.digest, fingerprint)
            and previous.final_completion == final
        ):
            self.counters["plans_reused"] += 1
            return previous
        plan = self._build(
            identity,
            snap=snap,
            task=(task_id, task_rev, scope_rev),
            changeset=changeset,
            fingerprint=fingerprint,
            final=final,
            revision=(previous.plan_revision + 1) if previous is not None else 1,
        )
        with self._lock:
            if previous is not None:
                self.counters["plans_invalidated"] += 1
            self._plans[key] = plan
            self._plans.move_to_end(key)
            while len(self._plans) > _PLANS_LIMIT:
                self._plans.popitem(last=False)
            self._dirty.discard(key)
            self._last_refresh[key] = time.monotonic()
        self.counters["plans_built"] += 1
        if "full_suite" in plan.required_checks:
            self.counters["full_suite_required"] += 1
        return plan

    @staticmethod
    def _final_completion(snap: Any) -> bool:
        if snap is None:
            return False
        state = snap.state
        return any(a.kind == "completed" and a.state == "PROPOSED" for a in state.atoms if a.live)

    def _build(
        self,
        identity: Any,
        *,
        snap: Any,
        task: tuple[str | None, int, int],
        changeset: Changeset,
        fingerprint: str,
        final: bool,
        revision: int,
    ) -> VerificationPlan:
        ws = identity.workspace_id
        ctx = ProjectContext(identity.root, history_commands=self._commands.get(ws, {}))
        active = [a for a in adapters() if a.detect(ctx)]
        unknown: list[str] = []
        if not active:
            canonical = CanonicalAdapter()
            if canonical.detect(ctx):
                active = [canonical]
            else:
                unknown.append("framework.unknown")
                self.counters["unknown_framework"] += 1
        for adapter in active:
            if isinstance(adapter, CTestAdapter) and adapter.build_dir(ctx) is None:
                unknown.append("ctest.no_configured_build_dir")
        changed = [f for f in changeset.files if f not in changeset.deleted]
        tier1: dict[tuple[str, str], TestTarget] = {}
        tier2: dict[tuple[str, str], TestTarget] = {}

        def put(bucket: dict[tuple[str, str], TestTarget], target: TestTarget) -> None:
            key = (target.framework, target.id)
            current = bucket.get(key)
            if current is None or target.sort_key() < current.sort_key():
                bucket[key] = target

        # (1) exact historical failure regressions (prior runs + evidence).
        failing = self.history.failing(ws)
        for fw, test_id, _ in failing:
            adapter = next((a for a in active if a.framework.value == fw), None)
            if adapter is not None and self._target_exists(ctx, adapter, test_id):
                put(
                    tier1,
                    TestTarget(
                        test_id, fw, Reason.FAILURE_REGRESSION.value, "history", _file_of(test_id)
                    ),
                )
        for fact in snap.evidence if snap is not None else ():
            if (
                fact.predicate == "test.status"
                and fact.polarity == "negative"
                and fact.status == "ACTIVE"
            ):
                for adapter in active:
                    for test_id in _ids_in(fact.subject, adapter):
                        if self._target_exists(ctx, adapter, test_id):
                            put(
                                tier1,
                                TestTarget(
                                    test_id,
                                    adapter.framework.value,
                                    Reason.FAILURE_REGRESSION.value,
                                    fact.id,
                                    _file_of(test_id),
                                ),
                            )

        # (2) direct scope: changed tests, coverage, naming, direct imports.
        graph = self._graph_dependents(identity, changed)
        importers = ctx.importers() if graph is None else {}
        importer_counts: dict[str, int] = {}
        uncovered: list[str] = []
        for rel in changed:
            selected_before = len(tier1)
            direct_deps = graph[0] if graph is not None else set(importers.get(rel, ()))
            importer_counts[rel] = len(direct_deps)
            for adapter in active:
                if adapter.is_test_file(rel):
                    for t in adapter.targets_in_file(ctx, rel, Reason.CHANGED_TEST.value, rel):
                        put(tier1, t)
                    continue
                for t in adapter.source_targets(ctx, rel):
                    put(tier1, t)
                tests = adapter.test_files(ctx)
                stem = _stem(rel)
                for test in tests:
                    if stem and _stem(test) == stem and len(stem) >= 2:
                        for t in adapter.targets_in_file(ctx, test, Reason.NAMING.value, rel):
                            put(tier1, t)
                for test in sorted(set(direct_deps) & set(tests)):
                    for t in adapter.targets_in_file(ctx, test, Reason.DIRECT_IMPORT.value, rel):
                        put(tier1, t)
                if isinstance(adapter, PytestAdapter):
                    for t in adapter.coverage_targets(ctx, [rel]):
                        put(tier1, t)
            if len(tier1) == selected_before and rel.endswith(_SOURCE_EXT) and active:
                uncovered.append(rel)
        # Acceptance atoms naming a test explicitly.
        if snap is not None:
            for atom in snap.state.atoms:
                if atom.kind != "acceptance" or not atom.live:
                    continue
                for adapter in active:
                    for test_id in _ids_in(atom.text, adapter):
                        if self._target_exists(ctx, adapter, test_id):
                            put(
                                tier1,
                                TestTarget(
                                    test_id,
                                    adapter.framework.value,
                                    Reason.ACCEPTANCE.value,
                                    atom.id,
                                    _file_of(test_id),
                                ),
                            )

        # (3) dependency / interface neighbours (second hop) and package scope.
        tier1_files = {t.file for t in tier1.values() if t.file}
        for rel in changed:
            if graph is not None:
                second = graph[1]
            else:
                second = set()
                for dep in importers.get(rel, ()):
                    second |= importers.get(dep, set())
                second -= set(importers.get(rel, ()))
            for adapter in active:
                tests = set(adapter.test_files(ctx))
                for test in sorted((second & tests) - tier1_files):
                    for t in adapter.targets_in_file(ctx, test, Reason.DEPENDENCY.value, rel):
                        put(tier2, t)
                for t in adapter.neighbor_targets(ctx, rel):
                    if (t.framework, t.id) not in tier1:
                        put(tier2, t)
        for key in list(tier2):
            if key in tier1:
                tier2.pop(key)

        frameworks = tuple(a.framework.value for a in active)
        tier3: list[TestTarget] = []
        commands: list[TestCommand] = []
        t1 = sorted(tier1.values(), key=TestTarget.sort_key)[:MAX_TARGETS_PER_TIER]
        t2 = sorted(tier2.values(), key=TestTarget.sort_key)[:MAX_TARGETS_PER_TIER]
        for adapter in active:
            fw = adapter.framework.value
            commands += adapter.commands(ctx, 1, [t for t in t1 if t.framework == fw])
            commands += adapter.commands(ctx, 2, [t for t in t2 if t.framework == fw])
            full = adapter.full_command(ctx)
            if full is not None:
                commands.append(full)
                tier3.append(TestTarget(f"{fw}:full", fw, Reason.FULL_SUITE.value, "project"))
        if not t1 and not t2 and changed and not tier3:
            unknown.append("plan.no_runnable_selection")

        # (4) risk and staged escalation.
        uncertain = bool(
            snap is not None and any(f.status in ("CONTRADICTED", "STALE") for f in snap.evidence)
        )
        flaky_keys = set(self.history.flaky_ids(ws))
        flaky_in_scope = any((t.framework, t.id) in flaky_keys for t in t1 + t2)
        contract = self._scope(identity)
        drift = _drift(contract, changed)
        hist = self.history.failure_ratio(ws, [(t.framework, t.id) for t in t1])
        risk, reasons = risk_of(
            changed,
            lines_changed=changeset.lines_changed,
            importer_counts=importer_counts,
            frameworks=frameworks,
            history_fail_ratio=hist,
            uncertain_evidence=uncertain,
            uncovered=uncovered,
            drift=drift,
            flaky_in_scope=flaky_in_scope,
            final_completion=final,
        )
        required: list[str] = []
        if t1:
            required.append("tier1")
        mandatory = bool(MANDATORY_FULL & set(reasons))
        if t2 and (risk >= 0.35 or not t1):
            required.append("tier2")
        if tier3 and (mandatory or risk >= 0.7 or (changed and not t1 and not t2)):
            required.append("full_suite")
        if not changed and not t1:
            reasons.append("plan.no_changes")
        reasons += unknown
        total = sum(a.count_tests(ctx) for a in active) if active else 0
        selected = len(t1)
        estimate = {
            "kind": "estimate",
            "tests_total_estimate": total,
            "tests_selected_tier1": selected,
            "fraction_avoided_estimate": round(1 - selected / total, 3)
            if total and selected and "full_suite" not in required
            else 0.0,
            "full_suite_required": "full_suite" in required,
        }
        return VerificationPlan(
            workspace_id=ws,
            task_id=task[0],
            task_revision=task[1],
            scope_revision=task[2],
            changeset_hash=changeset.digest,
            plan_revision=revision,
            frameworks=frameworks,
            tier1=tuple(t1),
            tier2=tuple(t2),
            tier3=tuple(tier3),
            required_checks=tuple(required),
            risk_score=risk,
            reason_codes=tuple(dict.fromkeys(reasons)),
            commands=tuple(commands),
            changed_files=tuple(changeset.files),
            runner_fingerprint=fingerprint,
            session_id=identity.session_id,
            final_completion=final,
            estimate=estimate,
            unknown=tuple(unknown),
            created_at=self._clock(),
        )

    @staticmethod
    def _target_exists(ctx: ProjectContext, adapter: Adapter, test_id: str) -> bool:
        if adapter.framework is Framework.PYTEST:
            return ctx.exists(test_id.split("::", 1)[0])
        if adapter.framework in (Framework.JEST, Framework.VITEST):
            return ctx.exists(test_id.split(" > ", 1)[0].strip())
        return True

    # -- run recording (worker) ---------------------------------------------

    def expect_run(self, call_id: str, *, tier: int, targets: Sequence[str]) -> None:
        """A Headroom-started run: its tier and exact targets (command heads are clipped)."""
        with self._lock:
            self._expected[call_id] = (int(tier), tuple(targets))
            while len(self._expected) > 256:
                self._expected.popitem(last=False)

    def project(self, envelope: EventEnvelope, tx: Any) -> None:
        """Projector: parse TEST_RESULT content (cheap); the listener records it."""
        event = envelope.event
        if event.kind != EventKind.TEST_RESULT.value or envelope.content is None:
            return
        if not event.workspace_id.startswith("ws_"):
            return
        meta = event.redacted_metadata
        command = str(meta.get("command_head") or "")
        framework = framework_of_command(command)
        if framework is None:
            return
        with self._lock:
            expected = self._expected.pop(event.source_id, None)
        targets = (
            list(expected[1]) if expected is not None else _targets_of_command(command, framework)
        )
        if framework == Framework.PYTEST.value:
            targets = [t for t in targets if "::" in t or t.endswith(".py")]
        else:
            targets = []
        outcome = parse_output(framework, envelope.content, meta.get("exit_code"), targets=targets)
        if meta.get("is_partial"):
            outcome = replace(outcome, partial=True)
        self._pending.append(
            _PendingRun(
                event.workspace_id,
                event.session_id,
                event.id,
                framework,
                command,
                outcome,
                event.ts,
                expected[0] if expected is not None else None,
            )
        )

    def on_commit(self, events: list[Any], _runtime: Any) -> None:
        """Commit listener: refresh plans for changed sessions, then record parsed runs."""
        runs: list[_PendingRun] = []
        while self._pending:
            runs.append(self._pending.popleft())
        touched: set[tuple[str, str]] = {(r.workspace_id, r.session_id) for r in runs}
        for event in events:
            if event.kind in (
                EventKind.FILE_CHANGE.value,
                EventKind.GIT_CHANGE.value,
                EventKind.TASK_STATE_CHANGE.value,
                EventKind.USER_INTENT.value,
                EventKind.SCOPE_DECISION.value,
            ) and event.workspace_id.startswith("ws_"):
                touched.add((event.workspace_id, event.session_id))
        with_runs = {(r.workspace_id, r.session_id) for r in runs}
        for key in sorted(touched):
            with self._lock:
                self._dirty.add(key)
                identity = self._identities.get(key)
                last = self._last_refresh.get(key, 0.0)
            due = key in with_runs or time.monotonic() - last >= REFRESH_MIN_INTERVAL_S
            if identity is not None and due:
                try:
                    self.plan(identity)  # a run is judged against the current plan
                except Exception as exc:  # noqa: BLE001
                    log.warning("verification plan refresh failed: %s", describe_exception(exc))
        for run in runs:
            try:
                self._record_pending(run)
            except Exception as exc:  # noqa: BLE001 - never kill the worker
                log.warning("verification run recording failed: %s", describe_exception(exc))

    def _record_pending(self, run: _PendingRun) -> None:
        with self._lock:
            commands = self._commands.setdefault(run.workspace_id, {})
            if run.outcome.ok or run.framework not in commands:
                commands[run.framework] = run.command
            identity = self._identities.get((run.workspace_id, run.session_id))
        root = identity.root if identity is not None else self._roots.get(run.workspace_id)
        if root is None:
            return
        changeset = compute_changeset(root, self._recorded_writes(identity) if identity else ())
        plan = self.current(identity) if identity is not None else None
        tier = (
            run.tier if run.tier is not None else (_tier_of(plan, run) if plan is not None else 0)
        )
        self.record_outcome(
            run.workspace_id,
            run.session_id,
            run.outcome,
            changeset=changeset.digest,
            plan=plan,
            tier=tier,
            source_event_id=run.event_id,
            ts=run.ts,
        )

    def record_outcome(
        self,
        workspace_id: str,
        session_id: str,
        outcome: RunOutcome,
        *,
        changeset: str,
        plan: VerificationPlan | None,
        tier: int,
        source_event_id: str = "",
        ts: float | None = None,
    ) -> dict[str, Any]:
        """Record one run: history, flaky tracking, check status, VERIFICATION_DECISION."""
        now = self._clock() if ts is None else ts
        fw = outcome.framework
        self.counters["runs_recorded"] += 1
        conclusive = not outcome.partial and not outcome.timed_out
        if conclusive and outcome.ok:
            # A clean run of a file / the whole suite also passed every test of
            # it whose most recent run failed (pytest -q lists failures only).
            covered = set(outcome.passed_ids)
            extra = [
                test_id
                for f, test_id, _ in self.history.failing(workspace_id)
                if f == fw
                and (not covered or test_id in covered or (_file_of(test_id) or "") in covered)
            ]
            if extra:
                outcome = replace(
                    outcome, passed_ids=tuple(dict.fromkeys((*outcome.passed_ids, *extra)))
                )
        if conclusive:
            for test_id in outcome.failed_ids:
                self.history.record(workspace_id, fw, test_id, changeset, False, now)
            for test_id in outcome.passed_ids:
                if test_id not in outcome.failed_ids:
                    self.history.record(workspace_id, fw, test_id, changeset, True, now)
        flaky = sorted(
            t
            for t in set(outcome.failed_ids) | set(outcome.passed_ids)
            if self.history.flaky(workspace_id, fw, t, changeset)
        )
        if outcome.partial or outcome.timed_out:
            status = "inconclusive"
        elif flaky:
            status = "flaky"
        elif outcome.ok:
            status = "passed"
        else:
            status = "failed"
        check = {1: "tier1", 2: "tier2", 3: "full_suite"}.get(tier, "observed")
        overall = status
        if plan is not None and plan.changeset_hash == changeset:
            with self._lock:
                state = self._checks.setdefault(plan.plan_id, _CheckState())
                self._checks.move_to_end(plan.plan_id)
                while len(self._checks) > _PLANS_LIMIT:
                    self._checks.popitem(last=False)
                if check != "observed":
                    state.status[check] = status
                state.flaky |= set(flaky)
                if status == "failed":
                    state.failed |= set(outcome.failed_ids)
                overall = self._overall(plan, state)
        elif plan is not None:
            overall = "stale_plan" if status == "passed" else status
        if flaky:
            self.counters["flaky_detected"] += 1
        meta = {
            "plan_id": plan.plan_id if plan is not None else "",
            "plan_revision": plan.plan_revision if plan is not None else 0,
            "changeset": changeset,
            "framework": fw,
            "tier": tier,
            "check": check,
            "status": status,
            "overall": overall,
            "exit_code": outcome.exit_code,
            "passed": outcome.passed,
            "failed": outcome.failed + outcome.errors,
            "flaky": len(flaky),
            "flaky_ids": flaky[:8],
            "failed_ids": list(outcome.failed_ids[:8]),
            "required": list(plan.required_checks) if plan is not None else [],
            "risk": plan.risk_score if plan is not None else None,
            "platform": sys.platform,
            "runner_version": outcome.runner_version,
            "test_event": source_event_id,
        }
        self._emit(workspace_id, session_id, meta, now)
        return meta

    @staticmethod
    def _overall(plan: VerificationPlan, state: _CheckState) -> str:
        if state.flaky:
            return "flaky"
        if any(v == "failed" for v in state.status.values()):
            return "failed"
        required = plan.required_checks or ()
        if required and all(state.status.get(c) == "passed" for c in required):
            return "verified"
        return "incomplete"

    def check_status(self, plan: VerificationPlan) -> dict[str, Any]:
        with self._lock:
            state = self._checks.get(plan.plan_id) or _CheckState()
            return {
                "checks": dict(state.status),
                "flaky": sorted(state.flaky),
                "overall": self._overall(plan, state),
            }

    def _emit(self, ws: str, session: str, meta: Mapping[str, Any], ts: float) -> None:
        if self.runtime is None:
            return
        try:
            event = make_event(
                workspace_id=ws,
                session_id=session,
                kind=EventKind.VERIFICATION_DECISION,
                origin=Origin.SYSTEM,
                source_id=f"verify:{meta.get('test_event') or meta.get('plan_id')}",
                index=int(meta.get("plan_revision") or 0),
                payload=dict(meta),
                metadata=meta,
                ts=ts,
            )
            self.runtime.submit([EventEnvelope(event=event)])
        except Exception as exc:  # noqa: BLE001 - audit is best effort
            log.debug("verification decision event failed: %s", describe_exception(exc))

    # -- request side -------------------------------------------------------

    def request_refresh(self, identity: Any) -> None:
        """Ask the worker to (re)build a session's plan if it may be stale."""
        if self.runtime is None or not identity.can_authorize_paths():
            return
        self.note_identity(identity)
        key = (identity.workspace_id, identity.session_id)
        with self._lock:
            have = key in self._plans
            dirty = key in self._dirty
            last = self._last_refresh.get(key, 0.0)
        if have and not dirty and time.monotonic() - last < 30.0:
            return
        with self._lock:
            self._last_refresh[key] = time.monotonic()
        self.runtime.submit_task(lambda: self.plan(identity))

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "plans": len(self._plans),
                "workspaces": len(self._roots),
                "counters": dict(self.counters),
            }


def _stem(rel: str) -> str:
    name = rel.rsplit("/", 1)[-1]
    name = re.sub(r"\.(?:test|spec)(?=\.)", "", name)
    name = name.split(".", 1)[0]
    name = re.sub(r"^test_|_test$|_tests$", "", name)
    return name.lower()


def _file_of(test_id: str) -> str | None:
    if "::" in test_id:
        return test_id.split("::", 1)[0]
    if " > " in test_id:
        return test_id.split(" > ", 1)[0].strip()
    return None


_NODE_ID = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*[\w.-]+\.py::[\w:\[\]-]+)")


def _ids_in(text: str, adapter: Adapter) -> list[str]:
    if adapter.framework is Framework.PYTEST:
        return list(dict.fromkeys(m.replace("\\", "/") for m in _NODE_ID.findall(text or "")))
    return []


def _targets_of_command(command: str, framework: str) -> list[str]:
    try:
        words = shlex.split(command, posix=True)
    except ValueError:
        words = command.split()
    if framework == Framework.PYTEST.value:
        return [w.replace("\\", "/") for w in words if "::" in w or w.endswith(".py")]
    return []


def _tier_of(plan: VerificationPlan, run: _PendingRun) -> int:
    """Which plan tier a run covered (exact command match, else target subset)."""
    try:
        words = tuple(shlex.split(run.command, posix=True))
    except ValueError:
        words = tuple(run.command.split())
    for cmd in plan.commands:
        if cmd.framework != run.framework:
            continue
        if words and tuple(
            os.path.basename(w) if i == 0 else w for i, w in enumerate(words)
        ) == tuple(os.path.basename(a) if i == 0 else a for i, a in enumerate(cmd.argv)):
            return cmd.tier
    if run.framework == Framework.PYTEST.value:
        targets = set(_targets_of_command(run.command, run.framework))
        if not targets:
            return 3  # bare runner invocation = the project's full suite
        tier1 = {t.id for t in plan.tier1 if t.framework == run.framework}
        tier1_files = {t.file for t in plan.tier1 if t.framework == run.framework}
        if tier1 and (tier1 <= targets or tier1_files <= targets):
            return 1
    return 0


def _drift(contract: Any, changed: Sequence[str]) -> list[str]:
    if contract is None or getattr(contract, "mode", None) is None:
        return []
    if str(getattr(contract.mode, "value", contract.mode)) == "LEARNING":
        return []
    from headroom.agent.change_scope import matches_any

    allowed = (
        tuple(contract.expected_paths)
        + tuple(contract.dependency_paths)
        + tuple(contract.recent_writes)
        + tuple(contract.generated_paths)
    )
    return [rel for rel in changed if matches_any(allowed, rel) is None]


# ---------------------------------------------------------------------------
# Request stage: one compact advisory block (rationale + avoided-work estimate)
# ---------------------------------------------------------------------------


def _plan_lines(plan: VerificationPlan, check: Mapping[str, Any]) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    reasons = [r for r in plan.reason_codes if r.startswith("risk.")]
    head = (
        f"Verification plan r{plan.plan_revision} for {len(plan.changed_files)} changed file(s); "
        f"risk {plan.risk_score:.2f}"
        + (f" ({', '.join(r[5:] for r in reasons[:4])})" if reasons else "")
    )
    lines.append((0, head + ". Headroom selected these from the diff; it has not run them."))
    for tier, label in ((1, "Tier 1 (direct)"), (2, "Tier 2 (neighbours)"), (3, "Full suite")):
        for cmd in plan.commands_for(tier)[:3]:
            cwd = f" (in {cmd.cwd})" if cmd.cwd not in (".", "") else ""
            lines.append((tier, f"{label}: `{cmd.display()}`{cwd}"))
    why = list(plan.tier1[:4])
    if why:
        lines.append(
            (
                2,
                "Why: "
                + "; ".join(
                    f"{t.id} ← {t.reason}{(' ' + t.source) if t.source and t.source != t.id else ''}"
                    for t in why
                ),
            )
        )
    if plan.required_checks:
        req = ", ".join("full suite" if c == "full_suite" else c for c in plan.required_checks)
        lines.append((0, f"Required before claiming done: {req}."))
    if check.get("flaky"):
        lines.append(
            (0, "Flaky (failed then passed unchanged; NOT green): " + ", ".join(check["flaky"][:4]))
        )
    if check.get("overall") in ("failed", "verified", "flaky"):
        lines.append((0, f"Status: {check['overall']}."))
    est = plan.estimate
    if (
        est.get("tests_total_estimate")
        and est.get("tests_selected_tier1")
        and not est.get("full_suite_required")
    ):
        lines.append(
            (
                4,
                f"Estimated avoided: ~{est['tests_total_estimate'] - est['tests_selected_tier1']} of "
                f"{est['tests_total_estimate']} tests (estimate, not measured).",
            )
        )
    for code in plan.unknown:
        lines.append((3, f"Unknown: {code}"))
    return lines


def render_block(
    lines: Sequence[tuple[int, str]], *, max_tokens: int = BLOCK_MAX_TOKENS
) -> str | None:
    from headroom.agent.task_state import estimate_tokens

    work = list(lines)
    while work:
        body = "\n".join(text for _, text in work)
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]
        block = f'<{BLOCK_TAG} hash="{digest}">\n{body}\n</{BLOCK_TAG}>'
        if estimate_tokens(block) <= max_tokens:
            return block
        worst = max(range(len(work)), key=lambda i: (work[i][0], i))
        work.pop(worst)
    return None


def _latest_hash(messages: Sequence[Any]) -> str | None:
    from headroom.agent.task_state import _text_of

    pattern = re.compile(rf'<{BLOCK_TAG} hash="([0-9a-f]+)">')
    for message in reversed(messages):
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        text = _text_of(content)
        m = pattern.search(text)
        if m:
            return m.group(1)
    return None


def inject_after_headroom_blocks(
    provider: str, model: str, messages: list[Any], block: str
) -> tuple[list[Any] | None, str]:
    """Place a block after the live turn, alongside other trailing Headroom blocks."""
    from headroom.agent.task_state import _text_of, plan_injection

    k = len(messages)
    while k > 0:
        item = messages[k - 1]
        if not isinstance(item, Mapping) or item.get("role") not in ("system", "developer"):
            break
        if not re.search(r"<headroom_[a-z_]+ hash=", _text_of(item.get("content"))):
            break
        k -= 1
    planned, placement = plan_injection(provider, model, messages[:k], block)
    if planned is None:
        return None, placement
    return [*planned, *messages[k:]], placement


def verification_stage(ctx: Any, task: Any, runtime: Any) -> None:
    """Request stage (order 65): refresh the plan off-loop; fold one advisory block."""
    from headroom.agent.hook import identity_for

    planner: TestImpactPlanner | None = runtime.services.get("test_impact")
    identity = identity_for(ctx)
    if identity is None or not identity.can_authorize_paths():
        return
    plan = None
    lines: list[tuple[int, str]] = []
    if planner is not None:
        planner.request_refresh(identity)
        plan = planner.current(identity)
        if plan is not None and (plan.changed_files or plan.tier1):
            lines += _plan_lines(plan, planner.check_status(plan))
    macros = runtime.services.get("workflow_macros")
    if macros is not None:
        try:
            lines += macros.advertisement_lines(identity, ctx)
        except Exception as exc:  # noqa: BLE001 - advertisement is optional
            log.debug("workflow advertisement failed: %s", describe_exception(exc))
    if not lines:
        return
    block = render_block(lines)
    if block is None:
        return
    current = re.search(r'hash="([0-9a-f]+)"', block)
    if current and _latest_hash(ctx.messages) == current.group(1):
        ctx.tags["agent_verification"] = {"injected": False, "reason": "unchanged_in_context"}
        return
    new_messages, placement = inject_after_headroom_blocks(
        ctx.provider, ctx.model, ctx.messages, block
    )
    if new_messages is None:
        ctx.tags["agent_verification"] = {"injected": False, "reason": placement}
        return
    ctx.messages = new_messages
    ctx.tags["agent_verification"] = {
        "injected": True,
        "placement": placement,
        "plan_revision": plan.plan_revision if plan is not None else None,
    }


__all__ = [
    "Adapter",
    "CTestAdapter",
    "CanonicalAdapter",
    "CargoAdapter",
    "Changeset",
    "FEATURE_TEST_IMPACT",
    "Framework",
    "JsAdapter",
    "ProjectContext",
    "PytestAdapter",
    "Reason",
    "RunHistory",
    "RunOutcome",
    "TestCommand",
    "TestImpactPlanner",
    "TestTarget",
    "VerificationPlan",
    "adapters",
    "compute_changeset",
    "framework_of_command",
    "parse_output",
    "risk_of",
    "runner_fingerprint",
    "verification_stage",
]
