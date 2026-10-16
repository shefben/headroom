"""Unified-21 Phase 8 retention: bounded windows over the real agent SQLite
stores, background-only scheduling, and a workspace-scoped operator purge."""

from __future__ import annotations

import subprocess
import threading
import time

import pytest
from click.testing import CliRunner

from headroom.agent import registry
from headroom.agent.events import EventEnvelope, EventKind, Origin, make_event
from headroom.agent.feedback import FeedbackKey, workspace_hash
from headroom.agent.ledger_db import open_ledger
from headroom.agent.retention import (
    DAY,
    RetentionPolicy,
    RetentionScheduler,
    prune,
    purge_workspace,
    workspace_id_for,
)
from headroom.agent.runtime import AgentRuntime
from headroom.agent.store import MemoryAgentStore, SQLiteAgentStore
from headroom.cli.main import main

NOW = 1_800_000_000.0


@pytest.fixture(autouse=True)
def _clean_registry():
    registry.clear_registry()
    yield
    registry.clear_registry()


def _env(ws: str, sess: str, index: int, ts: float) -> EventEnvelope:
    return EventEnvelope(
        event=make_event(
            workspace_id=ws,
            session_id=sess,
            kind=EventKind.TOOL_RESULT,
            origin=Origin.TOOL,
            source_id="c",
            index=index,
            payload=f"p{index}",
            ts=ts,
        ),
        content="x",
    )


#: Ledger tables other components create on bind (same DDL as their modules).
_EXTRA_TABLES = (
    "CREATE TABLE IF NOT EXISTS verification_runs (workspace_id TEXT NOT NULL,"
    " framework TEXT NOT NULL, test_id TEXT NOT NULL, changeset TEXT NOT NULL,"
    " ok INTEGER NOT NULL, ts REAL NOT NULL)",
    "CREATE TABLE IF NOT EXISTS agent_feedback_priors (key TEXT PRIMARY KEY,"
    " prior REAL NOT NULL, observations INTEGER NOT NULL, buffered TEXT NOT NULL,"
    " updated_at REAL NOT NULL)",
    "CREATE TABLE IF NOT EXISTS tool_resource_versions (scope TEXT NOT NULL,"
    " resource_id TEXT NOT NULL, version_hash TEXT NOT NULL, ccr_hash TEXT NOT NULL,"
    " complete INTEGER NOT NULL, family TEXT NOT NULL, unit_kind TEXT NOT NULL,"
    " units TEXT NOT NULL, total_units INTEGER NOT NULL, call_id TEXT NOT NULL,"
    " updated_at REAL NOT NULL, presented_exact INTEGER NOT NULL,"
    " PRIMARY KEY (scope, resource_id))",
)


def _with_tables(ledger):
    for stmt in _EXTRA_TABLES:
        ledger.execute(stmt)
    return ledger


def _stores(tmp_path):
    (tmp_path / "agent").mkdir(parents=True, exist_ok=True)
    store = SQLiteAgentStore(tmp_path / "agent" / "agent_state.db")
    return store, _with_tables(open_ledger(store))


def _count(db, table: str) -> int:
    return int(db.query(f"SELECT COUNT(*) FROM {table}")[0][0])


def _store_count(store, table: str) -> int:
    with store._lock:
        return int(store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def test_policy_defaults_and_env_overrides():
    assert RetentionPolicy.from_env({}) == RetentionPolicy(30.0, 30.0, 90.0)
    p = RetentionPolicy.from_env(
        {
            "HEADROOM_AGENT_EVENT_RETENTION_DAYS": "7",
            "HEADROOM_AGENT_SESSION_RETENTION_DAYS": "junk",
            "HEADROOM_AGENT_WORKSPACE_RETENTION_DAYS": "-3",
        }
    )
    assert p == RetentionPolicy(7.0, 30.0, 90.0)


def test_prune_applies_the_three_windows(tmp_path):
    store, ledger = _stores(tmp_path)
    store.commit([_env("ws_a", "s1", 0, NOW - 31 * DAY), _env("ws_a", "s1", 1, NOW - DAY)])
    assert store.compare_and_set("task_state", "ws_a", "old", "k", 0, {"atom": "d_old"})
    assert store.compare_and_set("task_state", "ws_a", "live", "k", 0, {"atom": "d_live"})
    with store._lock:  # backdate one session's derived state
        store._conn.execute(
            "UPDATE agent_projections SET updated_at=? WHERE session_id='old'", (NOW - 40 * DAY,)
        )
        store._conn.execute(
            "UPDATE agent_projections SET updated_at=? WHERE session_id='live'", (NOW - DAY,)
        )
    ledger.put_text("d_old", "old atom", NOW - 40 * DAY)
    ledger.put_text("d_live", "live atom", NOW - 40 * DAY)
    ledger.execute(
        "INSERT INTO verification_runs VALUES ('ws_a','pytest','t','c',1,?),('ws_a','pytest','t','c',1,?)",
        (NOW - 91 * DAY, NOW - 60 * DAY),
    )
    ledger.execute(
        "INSERT INTO tool_resource_versions VALUES ('ws_a|s1','r','h','c',1,'read','line','[]',1,'id',?,1)",
        (NOW - 31 * DAY,),
    )
    removed = prune(store, ledger, now=NOW)
    assert removed == {
        "agent_events": 1,
        "agent_projections": 1,
        "verification_runs": 1,
        "tool_resource_versions": 1,
        "atom_texts": 1,
    }
    assert _store_count(store, "agent_events") == 1
    assert store.get_projection("task_state", "ws_a", "live", "k") is not None
    assert {r[0] for r in ledger.query("SELECT digest FROM atom_texts")} == {"d_live"}
    assert _count(ledger, "verification_runs") == 1  # 60 days < 90-day workspace window
    assert prune(store, ledger, now=NOW) == {}  # idempotent
    store.close()


def test_stateless_memory_stores_are_left_to_their_count_bounds():
    store = MemoryAgentStore()
    store.commit([_env("w", "s", 0, NOW - 400 * DAY)])
    assert prune(store, open_ledger(store), now=NOW) == {}


def _wait_runs(sched: RetentionScheduler, n: int) -> None:
    deadline = time.monotonic() + 5
    while sched.runs < n and time.monotonic() < deadline:
        time.sleep(0.01)
    thread = sched._thread
    if thread is not None:
        thread.join(5)


def test_scheduler_prunes_in_background_at_most_once_per_interval(tmp_path):
    store, _ = _stores(tmp_path)
    runtime = AgentRuntime(store)
    clock = [0.0]
    sched = RetentionScheduler(runtime, open_ledger(store), interval=100.0, clock=lambda: clock[0])
    registry.register_commit_listener("agent.retention", sched.on_commit)
    runtime.start()
    try:
        old = time.time() - 45 * DAY
        assert runtime.submit([_env("w", "s", 0, old)])
        assert runtime.flush()
        _wait_runs(sched, 1)
        assert sched.runs == 1 and sched.last_removed == {"agent_events": 1}
        assert sched._thread is not threading.current_thread()
        assert runtime.submit([_env("w", "s", 1, old)])
        assert runtime.flush()
        _wait_runs(sched, 2)
        assert sched.runs == 1  # within the interval: not re-run
        clock[0] = 150.0
        assert runtime.submit([_env("w", "s", 2, old)])
        assert runtime.flush()
        _wait_runs(sched, 2)
        assert sched.runs == 2
        assert sched.status()["session_days"] == 30.0
    finally:
        sched.close()
        runtime.stop()


def test_a_slow_prune_never_blocks_the_worker_or_flush(tmp_path, monkeypatch):
    import headroom.agent.retention as retention_mod

    store, _ = _stores(tmp_path)
    runtime = AgentRuntime(store)
    release = threading.Event()
    started = threading.Event()

    def slow_prune(*_a, **_k):
        started.set()
        release.wait(10)
        return {}

    monkeypatch.setattr(retention_mod, "prune", slow_prune)
    sched = RetentionScheduler(runtime, open_ledger(store), interval=0.0)
    registry.register_commit_listener("agent.retention", sched.on_commit)
    runtime.start()
    try:
        assert runtime.submit([_env("w", "s", 0, NOW)])
        assert started.wait(5)  # the prune is in progress ...
        t0 = time.monotonic()
        for i in range(1, 20):  # ... while the worker keeps committing
            assert runtime.submit([_env("w", "s", i, NOW)])
        assert runtime.flush(2)
        assert time.monotonic() - t0 < 2
        assert runtime.store.stats()["events"] == 20
    finally:
        release.set()
        sched.close()
        runtime.stop()


def test_purge_removes_only_the_named_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "hr"))
    from headroom import paths

    project = tmp_path / "proj"
    other_project = tmp_path / "other"
    for d in (project, other_project):  # two separate worktrees
        d.mkdir()
        subprocess.run(["git", "init", "-q", str(d)], check=True)
    ws, other = workspace_id_for(project), workspace_id_for(other_project)
    assert ws.startswith("ws_") and ws != other
    paths.agent_state_db_path().parent.mkdir(parents=True)
    store = SQLiteAgentStore(paths.agent_state_db_path())
    ledger = _with_tables(open_ledger(store))
    # Distinct sessions: event ids are (session, source, kind, index, digest).
    store.commit([_env(ws, "s_mine", 0, NOW), _env(other, "s_other", 0, NOW)])
    key = FeedbackKey("read", "read", "code", "type_specific", "sonnet", "", workspace_hash(ws))
    other_key = FeedbackKey(
        "read", "read", "code", "type_specific", "sonnet", "", workspace_hash(other)
    )
    ledger.execute(
        "INSERT INTO agent_feedback_priors VALUES (?,0.4,6,'[]',?),(?,0.4,6,'[]',?),(?,0.4,6,'[]',?)",
        (key.ident(), NOW, other_key.ident(), NOW, key.global_key().ident(), NOW),
    )
    ledger.execute(
        "INSERT INTO tool_resource_versions VALUES (?,'r','h','c',1,'read','line','[]',1,'id',?,1),"
        "(?,'r','h','c',1,'read','line','[]',1,'id',?,1)",
        (f"{ws}|s", NOW, f"{other}|s", NOW),
    )
    store.close()
    ledger.close()
    result = CliRunner().invoke(main, ["agent", "purge", "--workspace", str(project), "--yes"])
    assert result.exit_code == 0, result.output
    assert "removed 3 rows" in result.output  # 1 event, 1 resource version, 1 prior
    store = SQLiteAgentStore(paths.agent_state_db_path())
    ledger = open_ledger(store)
    with store._lock:
        left = {r[0] for r in store._conn.execute("SELECT workspace_id FROM agent_events")}
    assert left == {other}
    priors = {r[0] for r in ledger.query("SELECT key FROM agent_feedback_priors")}
    assert priors == {other_key.ident(), key.global_key().ident()}  # global prior kept
    assert {r[0] for r in ledger.query("SELECT scope FROM tool_resource_versions")} == {
        f"{other}|s"
    }
    assert purge_workspace(ws) == {}
    store.close()
    ledger.close()


def test_purge_cli_rejects_missing_paths_and_needs_confirmation(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "hr"))
    missing = CliRunner().invoke(main, ["agent", "purge", "--workspace", str(tmp_path / "nope")])
    assert missing.exit_code != 0
    aborted = CliRunner().invoke(
        main, ["agent", "purge", "--workspace", str(tmp_path)], input="n\n"
    )
    assert aborted.exit_code != 0 and "Aborted" in aborted.output
    assert not (tmp_path / "hr" / "agent").exists()  # nothing created by a refused purge
