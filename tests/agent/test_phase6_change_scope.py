"""Gate 6A: Change-Scope / Drift Firewall (#23).

Unit coverage of contract compilation, classification, hard protection,
traceable dependency expansion, revision-pinned authorization under
concurrency and Windows/UNC/case path handling, plus the proxy-side wiring
(host-hook endpoint, observe-only proxy path, /health).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from headroom import paths
from headroom.agent import enforcement, registry
from headroom.agent.change_scope import (
    ChangeScopeFirewall,
    ContractMode,
    Enforcement,
    EnforcementPolicy,
    PlannedChange,
    ScopeClass,
    Verdict,
    canonicalize,
    hard_protection,
    path_mentions,
    planned_changes,
)
from headroom.agent.evidence import EvidenceFact
from headroom.agent.identity import resolve_identity
from headroom.agent.runtime import AgentRuntime
from headroom.agent.store import MemoryAgentStore
from headroom.agent.task_state import StateAtom, TaskSnapshot, TaskState
from headroom.proxy.server import ProxyConfig, create_app
from headroom.proxy.turn_hooks import clear_turn_hooks
from headroom.rollout import apply_wrap_feature_defaults, resolve_rollout

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _Compiler:
    """The TaskStateCompiler surface the firewall reads (published snapshots)."""

    def __init__(self) -> None:
        self.snapshots: dict[tuple[str, str], TaskSnapshot] = {}

    def snapshot(self, workspace_id: str, session_id: str) -> TaskSnapshot | None:
        return self.snapshots.get((workspace_id, session_id))

    def request_warm(self, workspace_id: str, session_id: str) -> None:
        return None


def _atom(text: str, *, n: int, kind: str = "goal", ts: float | None = None) -> StateAtom:
    return StateAtom(
        id=f"at_{n}",
        kind=kind,
        text=text,
        origin="user",
        confidence=0.95,
        state="ACTIVE",
        updated_at=ts if ts is not None else float(n),
    )


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "docs").mkdir()
    (root / "config").mkdir()
    (root / ".git").mkdir()
    (root / "src" / "app.py").write_text(
        "from src.util import helper\n\ndef run():\n    return helper()\n", encoding="utf-8"
    )
    (root / "src" / "util.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (root / "src" / "other.py").write_text("X = 1\n", encoding="utf-8")
    (root / "tests" / "test_app.py").write_text(
        "from src.app import run\n\ndef test_run():\n    assert run() == 1\n", encoding="utf-8"
    )
    (root / "docs" / "notes.md").write_text("notes\n", encoding="utf-8")
    (root / "config" / "secrets.yaml").write_text("k: v\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='p'\n", encoding="utf-8")
    return root


class _Env:
    def __init__(self, root: Path, policy: EnforcementPolicy | None = None) -> None:
        self.root = root
        self.runtime = AgentRuntime(MemoryAgentStore())
        self.compiler = _Compiler()
        self.runtime.services["task_state"] = self.compiler
        self.identity = resolve_identity(session_token="tok-6a", registered_cwd=str(root))
        self.firewall = ChangeScopeFirewall(self.runtime, policy=policy or EnforcementPolicy())
        self.revision = 0

    def instruct(self, *texts: str, evidence: tuple[EvidenceFact, ...] = (), resources=()) -> None:
        self.revision += 1
        atoms = tuple(_atom(t, n=i + 1) for i, t in enumerate(texts))
        state = TaskState(
            id="tk_1",
            workspace_id=self.identity.workspace_id,
            session_id=self.identity.session_id,
            revision=self.revision,
            atoms=atoms,
            resources=tuple(resources),
        )
        key = (self.identity.workspace_id, self.identity.session_id)
        self.compiler.snapshots[key] = TaskSnapshot(state=state, evidence=evidence)

    def path(self, rel: str) -> str:
        return str(self.root / rel)

    def authorize(self, rel_or_abs: str, effect: str = "edit", *, enforceable: bool = True):
        return self.firewall.authorize(
            self.identity,
            PlannedChange(effect, rel_or_abs),
            cwd=str(self.root),
            enforceable=enforceable,
        )


@pytest.fixture
def env(project: Path) -> _Env:
    e = _Env(project)
    yield e
    e.runtime.stop(timeout=1.0)


# ---------------------------------------------------------------------------
# Contract compilation from explicit user instructions
# ---------------------------------------------------------------------------


def test_contract_from_user_instructions(env: _Env):
    env.instruct(
        "Fix the crash in src/app.py, do not modify config/secrets.yaml or vendor/",
        "Commit when done but don't push",
    )
    contract = env.firewall.contract(env.identity)
    assert contract.mode is ContractMode.ACTIVE
    assert contract.expected_paths == ("src/app.py",)
    assert set(contract.forbidden_paths) == {"config/secrets.yaml", "vendor/"}
    assert "git_commit" in contract.permitted_effects
    assert "git_push" in contract.forbidden_effects and "git_push" not in contract.permitted_effects
    assert contract.task_id == "tk_1" and contract.task_revision == env.revision
    # No inference from reads: a later TOOL atom naming a path adds nothing.
    assert contract.to_dict()["v"] == 1


def test_learning_then_active_then_frozen(env: _Env):
    env.instruct("Make the parser faster")
    assert env.firewall.contract(env.identity).mode is ContractMode.LEARNING
    decision = env.authorize(env.path("docs/notes.md"))
    assert decision.verdict is Verdict.ALLOW and decision.reason_code == "scope.learning"
    env.instruct("Make the parser faster in src/app.py")
    assert env.firewall.contract(env.identity).mode is ContractMode.ACTIVE
    env.instruct("Only edit src/app.py and nothing else")
    assert env.firewall.contract(env.identity).mode is ContractMode.FROZEN


def test_latest_user_instruction_supersedes_older(env: _Env):
    env.instruct("Leave config/ alone")
    assert env.authorize(env.path("config/secrets.yaml")).verdict is Verdict.BLOCK
    # A newer explicit approval (TaskState revision) expands the boundary.
    env.revision += 1
    state = TaskState(
        id="tk_1",
        workspace_id=env.identity.workspace_id,
        session_id=env.identity.session_id,
        revision=env.revision,
        atoms=(
            _atom("Leave config/ alone", n=1, ts=1.0),
            _atom("You may now edit config/ to add the flag", n=2, ts=2.0),
        ),
    )
    env.compiler.snapshots[(env.identity.workspace_id, env.identity.session_id)] = TaskSnapshot(
        state=state
    )
    assert env.authorize(env.path("config/secrets.yaml")).verdict is Verdict.ALLOW


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_forbidden_blocks_only_when_enforceable(env: _Env):
    env.instruct("Fix src/app.py but do not touch config/secrets.yaml")
    blocked = env.authorize(env.path("config/secrets.yaml"))
    assert blocked.verdict is Verdict.BLOCK and blocked.scope is ScopeClass.FORBIDDEN
    assert blocked.reason_code == "scope.user_forbidden" and blocked.authorization is None
    advisory = env.authorize(env.path("config/secrets.yaml"), enforceable=False)
    assert advisory.verdict is Verdict.WARN and advisory.scope is ScopeClass.FORBIDDEN
    observe = _Env(env.root, EnforcementPolicy(change_scope=Enforcement.OBSERVE))
    observe.instruct("Fix src/app.py but do not touch config/secrets.yaml")
    assert observe.authorize(observe.path("config/secrets.yaml")).verdict is Verdict.WARN
    observe.runtime.stop(timeout=1.0)


def test_legitimate_edits_and_traceable_dependency_expansion(env: _Env):
    env.instruct("Fix the crash in src/app.py")
    expected = env.authorize(env.path("src/app.py"))
    assert expected.verdict is Verdict.ALLOW and expected.scope is ScopeClass.IN_SCOPE
    assert expected.authorization is not None
    # Direct import link (src/app.py imports src.util): dependency scope.
    dep = env.authorize(env.path("src/util.py"))
    assert dep.verdict is Verdict.ALLOW and dep.scope is ScopeClass.DEPENDENCY_SCOPE
    assert dep.reason_code == "scope.dependency.import" and dep.trace == "src/app.py"
    contract = env.firewall.contract(env.identity)
    assert "src/util.py" in contract.dependency_paths  # recorded with its reason
    assert contract.dependency_reasons["src/util.py"].startswith("scope.dependency.import")
    # Related test and required config.
    test = env.authorize(env.path("tests/test_app.py"))
    assert test.scope is ScopeClass.DEPENDENCY_SCOPE and "related_test" in test.reason_code
    cfg = env.authorize(env.path("pyproject.toml"))
    assert cfg.scope is ScopeClass.DEPENDENCY_SCOPE and cfg.reason_code == "scope.dependency.config"
    # Generated effect: the lockfile of an in-scope manifest.
    env.instruct("Fix the crash in src/app.py and bump pyproject.toml")
    lock = env.authorize(env.path("uv.lock"))
    assert lock.scope is ScopeClass.GENERATED_EFFECT and lock.verdict is Verdict.ALLOW
    # No link at all: unexplained -> warn (never block by default).
    other = env.authorize(env.path("docs/notes.md"))
    assert other.verdict is Verdict.WARN and other.scope is ScopeClass.UNRELATED
    assert env.firewall.counters["dependency_expansions"] >= 3


def test_failing_evidence_justifies_dependency(env: _Env):
    fact = EvidenceFact(
        id="ef_1",
        claim_key="error.signature|pytest",
        subject="pytest",
        predicate="error.signature",
        value_json=json.dumps('File "src/other.py", line 1, in <module> NameError'),
        polarity="negative",
        source_kind="tool_text",
        source_event_id="ev_1",
        source_hash="h",
        workspace_id=env.identity.workspace_id,
        session_id=env.identity.session_id,
        task_id="tk_1",
        confidence=0.75,
        status="ACTIVE",
        valid_from=time.time(),
        valid_until=None,
        supersedes_id=None,
    )
    env.instruct("Fix the crash in src/app.py", evidence=(fact,))
    decision = env.authorize(env.path("src/other.py"))
    assert decision.scope is ScopeClass.DEPENDENCY_SCOPE
    assert decision.reason_code == "scope.dependency.failing_evidence" and decision.trace == "ef_1"


def test_frozen_scope_protect_mode_blocks_unrelated(project: Path):
    e = _Env(project, EnforcementPolicy(change_scope=Enforcement.PROTECT))
    e.instruct("Only edit src/app.py, nothing else")
    assert e.authorize(e.path("src/app.py")).verdict is Verdict.ALLOW
    blocked = e.authorize(e.path("docs/notes.md"))
    assert blocked.verdict is Verdict.BLOCK and blocked.reason_code == "scope.frozen_unrelated"
    # A dependency of a locked scope is allowed but flagged.
    assert e.authorize(e.path("src/util.py")).verdict is Verdict.WARN
    e.runtime.stop(timeout=1.0)


@pytest.mark.parametrize(
    ("rel", "kind"),
    [
        (".git/config", "git_internal"),
        (".git/hooks/pre-commit", "git_internal"),
        (".env", "credentials"),
        ("deploy/.env.production", "credentials"),
        ("keys/server.pem", "private_key"),
        ("home/.ssh/id_ed25519", "private_key"),
        (".aws/credentials", "credentials"),
        (".claude/settings.json", "provider_config"),
        (".codex/auth.json", "provider_config"),
    ],
)
def test_hard_protection_blocks(env: _Env, rel: str, kind: str):
    env.instruct("Fix the crash in src/app.py")
    assert hard_protection(rel) == kind
    decision = env.authorize(env.path(rel))
    assert (
        decision.verdict is Verdict.BLOCK and decision.reason_code == f"scope.hard_protected.{kind}"
    )


def test_hard_protection_negatives_and_user_approval(env: _Env):
    assert hard_protection(".env.example") is None
    assert hard_protection(".github/workflows/ci.yml") is None
    assert hard_protection(".gitignore") is None
    env.instruct("Add the new API key to .env")
    assert env.authorize(env.path(".env")).verdict is Verdict.ALLOW  # explicitly approved
    env.instruct("Edit .git/config to add the remote")
    # .git internals are never approvable through instructions.
    assert env.authorize(env.path(".git/config")).verdict is Verdict.BLOCK


def test_untrusted_workspace_never_gets_write_authority(project: Path):
    runtime = AgentRuntime(MemoryAgentStore())
    firewall = ChangeScopeFirewall(runtime, policy=EnforcementPolicy())
    untrusted = resolve_identity(session_token=None, registered_cwd=None, attribution="proj")
    decision = firewall.authorize(
        untrusted, PlannedChange("edit", str(project / "src/app.py")), enforceable=True
    )
    assert decision.authorization is None and decision.verdict is Verdict.WARN
    assert decision.reason_code == "scope.untrusted_workspace"
    # Even a forbidden-looking path is not "blocked" without a trusted workspace.
    git = firewall.authorize(
        untrusted, PlannedChange("edit", str(project / ".git/config")), enforceable=True
    )
    assert git.verdict is not Verdict.BLOCK
    runtime.stop(timeout=1.0)


def test_outside_worktree_and_user_named_external(env: _Env, tmp_path: Path):
    outside = tmp_path / "elsewhere" / "file.txt"
    outside.parent.mkdir()
    env.instruct("Fix the crash in src/app.py")
    decision = env.authorize(str(outside))
    assert decision.verdict is Verdict.WARN and decision.reason_code == "scope.outside_worktree"
    env.instruct(f"Fix src/app.py and also update {outside.as_posix()}")
    assert env.authorize(str(outside)).verdict is Verdict.ALLOW  # user intent is authoritative


def test_effects_from_shell_and_git(env: _Env):
    env.instruct("Fix src/app.py. Don't push and do not delete any files.")
    changes = planned_changes("Bash", {"command": "git push origin main"})
    assert changes == [PlannedChange("git_push")]
    push = env.firewall.authorize(env.identity, changes[0], enforceable=True)
    assert push.verdict is Verdict.BLOCK and push.reason_code == "scope.forbidden_effect.git_push"
    rm = env.authorize(env.path("src/app.py"), "delete")
    assert rm.verdict is Verdict.BLOCK and rm.reason_code == "scope.forbidden_effect.delete"
    env.instruct("Fix src/app.py")
    force = env.firewall.authorize(
        env.identity, PlannedChange("git_history_rewrite"), enforceable=True
    )
    assert force.verdict is Verdict.WARN
    assert force.reason_code == "scope.effect_not_permitted.git_history_rewrite"
    preview = planned_changes("Bash", {"command": "git clean -n -d"})
    assert preview[0].preview
    assert (
        env.firewall.authorize(env.identity, preview[0], enforceable=True).verdict is Verdict.ALLOW
    )
    clean = planned_changes("Bash", {"command": "git clean -fd"})[0]
    assert not clean.preview  # never default-delete untracked files silently
    assert env.firewall.authorize(env.identity, clean, enforceable=True).verdict is Verdict.WARN


def test_planned_changes_cover_shell_writes():
    cmd = "cd src && sed -i 's/a/b/' app.py && echo x > ../out.txt && rm -f old.py | cat"
    got = planned_changes("Bash", {"command": cmd})
    assert PlannedChange("edit", "app.py") in got
    assert PlannedChange("edit", "../out.txt") in got
    assert PlannedChange("delete", "old.py") in got
    ps = planned_changes(
        "PowerShell",
        {"command": "Set-Content -Path notes.txt -Value hi; Remove-Item -Recurse build"},
    )
    assert PlannedChange("edit", "notes.txt") in ps and PlannedChange("delete", "build") in ps
    patch = "*** Begin Patch\n*** Update File: src/a.py\n*** Add File: src/b.py\n*** End Patch"
    assert planned_changes("apply_patch", {"input": patch}) == [
        PlannedChange("edit", "src/a.py"),
        PlannedChange("create", "src/b.py"),
    ]
    assert planned_changes("Bash", {"command": "ls -la > /dev/null 2>&1"}) == []
    assert planned_changes("Read", {"file_path": "x"}) == []


# ---------------------------------------------------------------------------
# Revisions and concurrency
# ---------------------------------------------------------------------------


def test_authorization_pinned_and_stale_after_concurrent_write(env: _Env):
    env.instruct("Fix the crash in src/app.py")
    decision = env.firewall.evaluate(
        env.identity, PlannedChange("edit", env.path("src/app.py")), enforceable=True
    )
    assert decision.authorization is not None and env.firewall.verify(env.identity, decision)
    # Another session/thread executes a write to the same resource.
    env.firewall.record_write(env.identity.workspace_id, env.identity.session_id, "src/app.py")
    assert not env.firewall.verify(env.identity, decision)
    # A TaskState revision change also invalidates an issued authorization.
    fresh = env.firewall.evaluate(
        env.identity, PlannedChange("edit", env.path("src/app.py")), enforceable=True
    )
    env.instruct("Fix the crash in src/app.py")
    assert not env.firewall.verify(env.identity, fresh)


def test_reevaluation_under_concurrent_scope_change(env: _Env, monkeypatch):
    """A stale ALLOW is re-evaluated once; a concurrently forbidden path is then denied."""
    env.instruct("Fix src/app.py and src/util.py")
    original = env.firewall.evaluate
    calls = {"n": 0}

    def racing(identity, change, **kw):
        result = original(identity, change, **kw)
        calls["n"] += 1
        if calls["n"] == 1:  # the user forbids it while the first check is in flight
            env.instruct("Fix src/app.py, do not modify src/util.py")
        return result

    monkeypatch.setattr(env.firewall, "evaluate", racing)
    decision = env.authorize(env.path("src/util.py"))
    assert calls["n"] == 2 and decision.reevaluated
    assert decision.verdict is Verdict.BLOCK and decision.reason_code == "scope.user_forbidden"


def test_persistent_revision_churn_denies_with_stable_code(env: _Env, monkeypatch):
    env.instruct("Fix src/app.py")
    counter = {"n": 0}

    def churn(identity, rel):
        counter["n"] += 1
        return counter["n"]

    monkeypatch.setattr(env.firewall, "resource_revision", churn)
    decision = env.authorize(env.path("src/app.py"))
    assert decision.verdict is Verdict.BLOCK and decision.reason_code == "scope.stale_revision"
    assert env.firewall.counters["stale_denied"] == 1
    advisory = env.authorize(env.path("src/app.py"), enforceable=False)
    assert advisory.verdict is Verdict.WARN and advisory.reason_code == "scope.stale_revision"


def test_concurrent_writers_never_hold_valid_stale_authorizations(env: _Env):
    env.instruct("Fix src/app.py")
    issued: list = []
    errors: list = []

    def worker(i: int) -> None:
        try:
            for _ in range(20):
                d = env.firewall.evaluate(
                    env.identity, PlannedChange("edit", env.path("src/app.py")), enforceable=True
                )
                issued.append(d)
                env.firewall.record_write(
                    env.identity.workspace_id, env.identity.session_id, "src/app.py"
                )
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    # Every authorization issued before the last write is now stale.
    assert issued and not any(env.firewall.verify(env.identity, d) for d in issued)


# ---------------------------------------------------------------------------
# Paths: Windows drive/UNC/MSYS/case, symlinks
# ---------------------------------------------------------------------------


def test_canonicalize_relative_and_case(project: Path):
    root = str(project)
    rel = canonicalize("src/../src/app.py", root=root, cwd=root)
    assert rel.rel == "src/app.py" and rel.inside and rel.exists
    folded = canonicalize(str(project / "SRC" / "APP.PY"), root=root, case_insensitive=True)
    assert folded.inside
    strict = canonicalize("/elsewhere/SRC/app.py", root="/proj", case_insensitive=False)
    assert not strict.inside


@pytest.mark.skipif(os.name != "nt", reason="Windows path forms")
def test_windows_drive_msys_extended_and_unc(project: Path):
    root = str(project)
    drive = root[0].lower()
    msys = "/" + drive + root[2:].replace("\\", "/") + "/src/app.py"
    assert canonicalize(msys, root=root).rel == "src/app.py"
    extended = "\\\\?\\" + root + "\\src\\app.py"
    assert canonicalize(extended, root=root).rel == "src/app.py"
    upper = canonicalize(root.upper() + "\\SRC\\app.py", root=root)
    assert upper.inside and upper.rel is not None
    unc = canonicalize("\\\\fileserver\\share\\proj\\src\\app.py", root=root)
    assert unc.unc and not unc.inside
    ext_unc = canonicalize("\\\\?\\UNC\\fileserver\\share\\x.py", root=root)
    assert ext_unc.unc and not ext_unc.inside


def test_symlink_escape_is_not_in_scope(env: _Env, tmp_path: Path):
    target = tmp_path / "outside_dir"
    target.mkdir()
    link = env.root / "src" / "linked"
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable for this user")
    env.instruct("Fix src/app.py")
    canon = canonicalize(str(link / "f.py"), root=str(env.root))
    assert canon.symlink_escape and not canon.inside
    decision = env.authorize(str(link / "f.py"))
    assert decision.verdict is Verdict.WARN and decision.reason_code == "scope.symlink_escape"
    # A symlink resolving into .git is hard-protected on its resolved path.
    git_link = env.root / "gitlink"
    os.symlink(env.root / ".git", git_link, target_is_directory=True)
    assert env.authorize(str(git_link / "config")).verdict is Verdict.BLOCK


def test_path_mentions_ignore_urls_versions_and_flags():
    text = "See https://x.io/a.py, v1.2.3, e.g. --flag=1; edit src/a.py and README.md"
    assert path_mentions(text) == ["src/a.py", "README.md"]


# ---------------------------------------------------------------------------
# Proxy wiring: host-hook endpoint, observe-only proxy path, /health
# ---------------------------------------------------------------------------

PORT = 18986
UPSTREAM = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5",
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


@pytest.fixture
def wired(tmp_path, monkeypatch, project):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "ws"))
    monkeypatch.setenv("HEADROOM_JEVK5", "off")
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    clear_turn_hooks()
    registry.clear_registry()
    enforcement.reset()
    directory = paths.proxy_clients_dir(PORT)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{os.getpid()}.json").write_text(
        json.dumps({"session_token": "tok-wired", "cwd": str(project)}), encoding="utf-8"
    )
    config = ProxyConfig(
        port=PORT,
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        memory_enabled=False,
        image_optimize=False,
        stateless=True,
        rollout=resolve_rollout(apply_wrap_feature_defaults({})),
    )
    app = create_app(config)
    sent: list[dict] = []

    async def fake(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
        sent.append(body if isinstance(body, dict) else json.loads(body))
        return httpx.Response(200, json=UPSTREAM)

    app.state.proxy._retry_request = fake
    client = TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000))
    yield {"client": client, "app": app, "sent": sent, "project": project}
    clear_turn_hooks()
    registry.clear_registry()
    enforcement.reset()
    paths.set_process_stateless(False)


def _messages(project: Path, *extra):
    return [
        {
            "role": "user",
            "content": "Fix the crash in src/app.py. Do not modify config/secrets.yaml.",
        },
        *extra,
    ]


def _post(client, messages):
    resp = client.post(
        "/v1/messages",
        headers={
            "x-api-key": "k",
            "anthropic-version": "2023-06-01",
            "x-headroom-session-token": "tok-wired",
        },
        json={"model": "claude-sonnet-5", "max_tokens": 16, "messages": messages},
    )
    assert resp.status_code == 200, resp.text
    return resp


def _drain(app) -> None:
    runtime = app.state.agent_runtime
    for _ in range(4):
        assert runtime.flush(5)


def _hook(client, payload):
    resp = client.post(
        "/v1/agent/host-hook",
        headers={"x-headroom-session-token": "tok-wired"},
        json={"client": "claude", "payload": payload},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_host_hook_endpoint_denies_forbidden_and_allows_scope(wired):
    client, app, project = wired["client"], wired["app"], wired["project"]
    with client:
        _post(client, _messages(project))
        _drain(app)
        base = {"session_id": "c1", "cwd": str(project), "hook_event_name": "PreToolUse"}
        denied = _hook(
            client,
            {
                **base,
                "tool_name": "Write",
                "tool_use_id": "t1",
                "tool_input": {
                    "file_path": str(project / "config" / "secrets.yaml"),
                    "content": "x",
                },
            },
        )
        spec = denied["output"]["hookSpecificOutput"]
        assert (
            spec["permissionDecision"] == "deny"
            and "scope.user_forbidden" in spec["permissionDecisionReason"]
        )
        allowed = _hook(
            client,
            {
                **base,
                "tool_name": "Edit",
                "tool_use_id": "t2",
                "tool_input": {
                    "file_path": str(project / "src" / "app.py"),
                    "old_string": "a",
                    "new_string": "b",
                },
            },
        )
        assert allowed["output"] is None  # prints nothing: permissions untouched
        dep = _hook(
            client,
            {
                **base,
                "tool_name": "Edit",
                "tool_use_id": "t3",
                "tool_input": {
                    "file_path": str(project / "src" / "util.py"),
                    "old_string": "1",
                    "new_string": "2",
                },
            },
        )
        assert dep["output"] is None
        warned = _hook(
            client,
            {
                **base,
                "tool_name": "Bash",
                "tool_use_id": "t4",
                "tool_input": {"command": "echo hi > docs/notes.md"},
            },
        )
        ctx = warned["output"]["hookSpecificOutput"]
        assert "permissionDecision" not in ctx and "additionalContext" in ctx
        health = client.get("/health").json()["agent_enforcement"]
        assert health["capabilities"]["claude"] == "host_pretool"
        assert (
            health["capabilities"]["codex"] == "observe_only"
            and health["proxy_path"] == "observe_only"
        )
        assert health["counters"]["host_prevented"] == 1
        assert health["change_scope"]["counters"]["blocked"] == 1


def test_proxy_only_path_observes_without_claiming_blocks(wired):
    client, app, project = wired["client"], wired["app"], wired["project"]
    with client:
        _post(client, _messages(project))
        _drain(app)
        executed = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_w1",
                        "name": "write_file",
                        "input": {"path": str(project / "config" / "secrets.yaml"), "content": "x"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_w1", "content": "wrote 1 file"}
                ],
            },
        ]
        _post(client, _messages(project, *executed))
        _drain(app)
        forwarded = json.dumps(wired["sent"][-1])
        assert "blocked" not in forwarded.lower() and "Headroom tool check" not in forwarded
        counters = client.get("/health").json()["agent_enforcement"]["counters"]
        assert counters["observed_calls"] >= 1
        assert counters["observed_would_block"] == 1 and counters["host_prevented"] == 0


def test_codex_client_hook_reports_unsupported(wired):
    client, project = wired["client"], wired["project"]
    with client:
        got = client.post(
            "/v1/agent/host-hook",
            headers={"x-headroom-session-token": "tok-wired"},
            json={
                "client": "codex",
                "payload": {
                    "hook_event_name": "PreToolUse",
                    "tool_name": "Write",
                    "tool_input": {"file_path": str(project / ".git/config")},
                },
            },
        ).json()
        assert got["output"] is None and got["capability"] == "observational_or_proxy_only"


def test_relative_mentions_from_a_subdirectory_anchor_on_components(project: Path):
    """A user in ``<repo>/proj`` names ``src/x.py``; the worktree root is the repo."""
    repo = project.parent
    (repo / ".git").mkdir(exist_ok=True)
    e = _Env(project)
    e.identity = resolve_identity(session_token="tok-sub", registered_cwd=str(repo))
    e.instruct("Fix src/app.py but do not modify src/other.py")
    contract = e.firewall.contract(e.identity)
    assert contract.expected_paths == ("**/src/app.py",)
    blocked = e.authorize(e.path("src/other.py"))
    assert blocked.verdict is Verdict.BLOCK and blocked.reason_code == "scope.user_forbidden"
    assert e.authorize(e.path("src/app.py")).verdict is Verdict.ALLOW
    from headroom.agent.change_scope import matches_any

    assert matches_any(("**/src/app.py",), "proj/src/app.py") is not None
    assert matches_any(("**/src/app.py",), "proj/xsrc/app.py") is None
    e.runtime.stop(timeout=1.0)


# ---------------------------------------------------------------------------
# Case folding is a property of the workspace filesystem, not of the OS
# ---------------------------------------------------------------------------


def test_case_folding_follows_the_workspace_filesystem(env: _Env, monkeypatch):
    from headroom.agent import change_scope as cs

    root = str(env.root)
    probed = cs.case_insensitive_at(root)
    # The probe agrees with the filesystem itself.
    upper = env.root / "SRC"
    assert probed == (upper.exists() and os.path.samefile(upper, env.root / "src"))
    # A worktree on a case-insensitive filesystem (e.g. a Windows drive under
    # WSL, drvfs) must fold: "CONFIG/Secrets.yaml" is the forbidden file there.
    monkeypatch.setitem(cs._FOLD_CACHE, os.path.realpath(root), True)
    env.instruct("Fix src/app.py but do not touch config/secrets.yaml")
    blocked = env.authorize("CONFIG/Secrets.yaml")
    assert blocked.scope is ScopeClass.FORBIDDEN and blocked.verdict is Verdict.BLOCK
    assert cs.hard_protection(".GIT/config", fold=cs.case_insensitive_at(root)) == "git_internal"
    if sys.platform in ("win32", "cygwin"):
        return  # Windows filesystems always fold (and the probe is not consulted)
    # On a case-sensitive filesystem the differently-cased name is another file.
    monkeypatch.setitem(cs._FOLD_CACHE, os.path.realpath(root), False)
    fresh = ChangeScopeFirewall(env.runtime, policy=EnforcementPolicy())
    other = fresh.authorize(
        env.identity, PlannedChange("edit", "CONFIG/Secrets.yaml"), cwd=root, enforceable=True
    )
    assert other.scope is not ScopeClass.FORBIDDEN
