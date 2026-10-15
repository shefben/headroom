"""Identity isolation, SQLite/memory store semantics, runtime worker behavior."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from headroom.agent import registry
from headroom.agent.events import EventEnvelope, EventKind, Origin, make_event
from headroom.agent.identity import resolve_identity
from headroom.agent.interfaces import AuthorizationDecision
from headroom.agent.runtime import AgentRuntime, create_store
from headroom.agent.store import MemoryAgentStore, SQLiteAgentStore


@pytest.fixture(autouse=True)
def _clean_registry():
    registry.clear_registry()
    yield
    registry.clear_registry()


def _env(ws="w", sess="s", source="c1", index=0, payload="p", kind=EventKind.TOOL_RESULT):
    return EventEnvelope(
        event=make_event(
            workspace_id=ws, session_id=sess, kind=kind, origin=Origin.TOOL,
            source_id=source, index=index, payload=payload,
        ),
        content=str(payload),
    )


class CountingProjector:
    name = "count"

    def __init__(self):
        self.seen_content = []

    def project(self, envelope, tx):
        self.seen_content.append(envelope.content)
        current = tx.get("n")
        revision, value = current if current else (0, 0)
        assert tx.compare_and_set("n", revision, value + 1)


# -- identity ------------------------------------------------------------------


def test_trusted_identity_requires_token_and_existing_cwd(tmp_path):
    trusted = resolve_identity(session_token="tok", registered_cwd=str(tmp_path), principal="u")
    assert trusted.trusted and trusted.can_authorize_paths()
    assert trusted.session_id.startswith("ss_") and trusted.workspace_id.startswith("ws_")
    assert "tok" not in trusted.session_id
    missing = resolve_identity(session_token="tok", registered_cwd=str(tmp_path / "gone"))
    assert not missing.trusted and not missing.can_authorize_paths()
    header_only = resolve_identity(session_token=None, registered_cwd=None, attribution="proj")
    assert not header_only.trusted


def test_identities_isolate_sessions_and_principals(tmp_path):
    a = resolve_identity(session_token="a", registered_cwd=str(tmp_path), principal="u1")
    b = resolve_identity(session_token="b", registered_cwd=str(tmp_path), principal="u1")
    c = resolve_identity(session_token="a", registered_cwd=str(tmp_path), principal="u2")
    assert a.workspace_id == b.workspace_id and a.session_id != b.session_id
    assert a.workspace_id != c.workspace_id
    untrusted = resolve_identity(session_token=None, registered_cwd=None, principal="u1")
    assert untrusted.workspace_id != a.workspace_id


def test_linked_worktree_is_distinct_mutation_scope(tmp_path):
    main = tmp_path / "repo"
    (main / ".git" / "worktrees" / "feat").mkdir(parents=True)
    (main / ".git" / "worktrees" / "feat" / "commondir").write_text("../..")
    linked = tmp_path / "feat"
    linked.mkdir()
    (linked / ".git").write_text(f"gitdir: {main / '.git' / 'worktrees' / 'feat'}")
    a = resolve_identity(session_token="t", registered_cwd=str(main / ""), principal="u")
    b = resolve_identity(session_token="t", registered_cwd=str(linked), principal="u")
    assert a.workspace_id != b.workspace_id
    assert b.linked_worktree and not a.linked_worktree
    assert a.repository_id == b.repository_id is not None


def test_runtime_authorization_only_for_trusted(tmp_path):
    runtime = AgentRuntime(MemoryAgentStore())
    trusted = resolve_identity(session_token="t", registered_cwd=str(tmp_path))
    untrusted = resolve_identity(session_token=None, registered_cwd=None)
    kwargs = dict(task_revision=1, scope_revision=1, resource_revision=1,
                  decision=AuthorizationDecision.ALLOW)
    assert runtime.authorize_action(untrusted, **kwargs) is None
    auth = runtime.authorize_action(trusted, **kwargs)
    assert auth is not None and auth.workspace_id == trusted.workspace_id


# -- store -------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_store_idempotent_projection_and_cas(kind, tmp_path):
    store = MemoryAgentStore() if kind == "memory" else SQLiteAgentStore(tmp_path / "a.db")
    projector = CountingProjector()
    first = store.commit([_env(source="c1"), _env(source="c2")], [projector])
    again = store.commit([_env(source="c1"), _env(source="c2")], [projector])
    assert len(first) == 2 and again == []
    assert store.get_projection("count", "w", "s", "n") == (2, 2)
    assert projector.seen_content == ["p", "p"]
    # stale revision is rejected, current one accepted
    assert not store.compare_and_set("count", "w", "s", "n", 1, 99)
    assert store.compare_and_set("count", "w", "s", "n", 2, 10)
    # replay rebuilds projections from events alone (no content on replay)
    projector.seen_content.clear()
    assert store.rebuild([projector]) == 2
    assert store.get_projection("count", "w", "s", "n") == (2, 2)
    assert projector.seen_content == [None, None]
    # other sessions are isolated
    store.commit([_env(sess="other", source="c1")], [projector])
    assert store.get_projection("count", "w", "other", "n") == (1, 1)
    assert len(store.events(session_id="s")) == 2
    store.close()


def test_sqlite_wal_schema_and_bounded_retention(tmp_path):
    store = SQLiteAgentStore(tmp_path / "a.db", max_events=5)
    assert store.journal_mode == "wal"
    store.commit([_env(source=f"c{i}") for i in range(8)])
    assert len(store.events()) == 5
    assert [e.source_id for e in store.events(limit=2)] == ["c6", "c7"]
    store.close()
    conn = sqlite3.connect(tmp_path / "a.db")
    assert conn.execute("SELECT value FROM agent_meta WHERE key='schema_version'").fetchone()[0] == "1"
    conn.close()


def test_sqlite_projector_bug_is_isolated(tmp_path):
    store = SQLiteAgentStore(tmp_path / "a.db")

    class Boom:
        name = "boom"

        def project(self, envelope, tx):
            raise RuntimeError("projector bug")

    # A projector bug is isolated (event still recorded), never a lost batch.
    assert len(store.commit([_env()], [Boom()])) == 1
    store.close()


def test_sqlite_newer_schema_is_never_downgraded(tmp_path):
    path = tmp_path / "a.db"
    SQLiteAgentStore(path).close()
    conn = sqlite3.connect(path)
    conn.execute("UPDATE agent_meta SET value='99' WHERE key='schema_version'")
    conn.commit()
    conn.close()
    from headroom.agent.store import SchemaTooNewError

    with pytest.raises(SchemaTooNewError):
        SQLiteAgentStore(path)


def test_sqlite_corruption_recovery_salvages_and_rebuilds(tmp_path):
    path = tmp_path / "a.db"
    store = SQLiteAgentStore(path)
    store.commit([_env(source="c1"), _env(source="c2")], [CountingProjector()])
    store.close()
    for suffix in ("-wal", "-shm"):
        Path(str(path) + suffix).unlink(missing_ok=True)
    data = bytearray(path.read_bytes())
    data[:16] = b"\x00" * 16  # destroy the header magic
    path.write_bytes(bytes(data))
    recovered = SQLiteAgentStore(path)
    assert recovered.recovered
    assert list(tmp_path.glob("a.db.corrupt-*"))
    recovered.commit([_env(source="c3")])
    assert any(e.source_id == "c3" for e in recovered.events())
    recovered.close()


# -- runtime ------------------------------------------------------------------------


def test_runtime_worker_commits_off_thread_and_skips_replays():
    store = MemoryAgentStore()
    runtime = AgentRuntime(store)
    projector = CountingProjector()
    registry.register_projector(projector)
    seen_threads = []
    registry.register_commit_listener(
        "t", lambda events, rt: seen_threads.append(threading.current_thread().name)
    )
    batch = [_env(source="c1"), _env(source="c2")]
    assert runtime.submit(batch)
    assert runtime.submit(batch)  # replay: filtered before queueing
    assert runtime.flush(5)
    assert runtime.counters["committed_events"] == 2
    assert runtime.counters["skipped_replays"] == 2
    assert seen_threads == ["headroom-agent-runtime"]
    assert store.get_projection("count", "w", "s", "n") == (2, 2)
    runtime.stop()
    assert not runtime.running
    assert runtime.submit([_env(source="c9")]) is False


def test_runtime_bounded_queue_drops_without_blocking():
    gate = threading.Event()

    class Slow:
        name = "slow"

        def project(self, envelope, tx):
            gate.wait(5)

    registry.register_projector(Slow())
    runtime = AgentRuntime(MemoryAgentStore(), queue_size=1)
    results = [runtime.submit([_env(source=f"c{i}")]) for i in range(6)]
    assert False in results
    assert runtime.counters["dropped_batches"] >= 1
    gate.set()
    assert runtime.flush(5)
    runtime.stop()


def test_create_store_stateless_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "ws"))
    store = create_store(stateless=True)
    assert isinstance(store, MemoryAgentStore)
    monkeypatch.setenv("HEADROOM_STATELESS", "1")
    assert isinstance(create_store(stateless=False), MemoryAgentStore)
    assert not (tmp_path / "ws").exists()


def test_create_store_persistent_under_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "ws"))
    store = create_store(stateless=False)
    assert isinstance(store, SQLiteAgentStore)
    assert store.path == tmp_path / "ws" / "agent" / "agent_state.db"
    store.close()


def test_create_store_refuses_symlink_escape(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (ws / "agent").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(ws))
    assert isinstance(create_store(stateless=False), MemoryAgentStore)
    assert not list(outside.iterdir())
