"""Local SQLite ledger for agent intelligence that is too relational for projections.

Holds three things the key/value projections in :mod:`headroom.agent.store`
cannot index well:

* ``evidence_facts`` — the provenance-aware Evidence Ledger (#17), indexed by
  ``(workspace, claim_key, status)``, ``(task, status)``, ``(source_event_id)``
  and timestamps;
* ``atom_texts`` — content-addressed TaskState atom texts, so replaying the
  digest-only ``TASK_STATE_CHANGE`` events can rebuild the full state;
* ``graph_files`` — per-worktree incremental code-graph file records (hash,
  mtime, size, extracted symbols/edges) so a restart only re-parses changed
  files.

It is a sibling file of the agent event store (``agent_ledger.db``) so the
event store's corruption recovery never renames it underneath a live
connection. In stateless mode (or when persistence is refused) it is an
in-memory SQLite database with identical semantics and zero filesystem writes.

Only background threads use it (the runtime worker and the graph indexer);
request handlers never do. Every call is serialized by one lock.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from headroom import fileperms
from headroom.log_safety import describe_exception

log = logging.getLogger(__name__)

LEDGER_SCHEMA_VERSION = 1
_BUSY_TIMEOUT_MS = 2000

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS ledger_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS evidence_facts ("
    " id TEXT PRIMARY KEY, claim_key TEXT NOT NULL, subject TEXT NOT NULL,"
    " predicate TEXT NOT NULL, value_json TEXT NOT NULL, polarity TEXT NOT NULL,"
    " source_kind TEXT NOT NULL, source_event_id TEXT NOT NULL, source_hash TEXT NOT NULL,"
    " workspace_id TEXT NOT NULL, session_id TEXT NOT NULL, task_id TEXT,"
    " confidence REAL NOT NULL, status TEXT NOT NULL, valid_from REAL NOT NULL,"
    " valid_until REAL, supersedes_id TEXT, metadata TEXT NOT NULL,"
    " revision INTEGER NOT NULL DEFAULT 1, updated_at REAL NOT NULL)",
    "CREATE INDEX IF NOT EXISTS evidence_ws_claim_status"
    " ON evidence_facts(workspace_id, claim_key, status)",
    "CREATE INDEX IF NOT EXISTS evidence_task_status ON evidence_facts(task_id, status)",
    "CREATE INDEX IF NOT EXISTS evidence_source ON evidence_facts(source_event_id)",
    "CREATE INDEX IF NOT EXISTS evidence_time ON evidence_facts(valid_from, updated_at)",
    "CREATE INDEX IF NOT EXISTS evidence_scope"
    " ON evidence_facts(workspace_id, session_id, status, updated_at)",
    "CREATE TABLE IF NOT EXISTS evidence_processed ( event_id TEXT PRIMARY KEY, ts REAL NOT NULL)",
    "CREATE INDEX IF NOT EXISTS evidence_processed_ts ON evidence_processed(ts)",
    "CREATE TABLE IF NOT EXISTS atom_texts ("
    " digest TEXT PRIMARY KEY, text TEXT NOT NULL, created_at REAL NOT NULL)",
    "CREATE TABLE IF NOT EXISTS graph_files ("
    " workspace_id TEXT NOT NULL, path TEXT NOT NULL, hash TEXT NOT NULL,"
    " mtime_ns INTEGER NOT NULL, size INTEGER NOT NULL, lang TEXT NOT NULL,"
    " parser TEXT NOT NULL, payload TEXT NOT NULL, indexed_at REAL NOT NULL,"
    " PRIMARY KEY (workspace_id, path))",
)


class LedgerSchemaTooNewError(RuntimeError):
    """The ledger was written by a newer schema; never downgrade it."""


class AgentLedgerDB:
    """See module docstring. ``path=None`` means in-memory."""

    def __init__(self, path: Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self.persistent = self.path is not None
        self._lock = threading.RLock()
        self._conn = self._open()

    def _open(self) -> sqlite3.Connection:
        if self.path is None:
            conn = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
        else:
            conn = fileperms.connect_private_sqlite(
                self.path,
                what="agent ledger database",
                timeout=_BUSY_TIMEOUT_MS / 1000,
                check_same_thread=False,
                isolation_level=None,
            )
            conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()
            if not mode or str(mode[0]).lower() != "wal":
                conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("BEGIN IMMEDIATE")
        try:
            for statement in _SCHEMA:
                conn.execute(statement)
            row = conn.execute(
                "SELECT value FROM ledger_meta WHERE key='schema_version'"
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO ledger_meta(key, value) VALUES ('schema_version', ?)",
                    (str(LEDGER_SCHEMA_VERSION),),
                )
            elif int(row[0]) > LEDGER_SCHEMA_VERSION:
                raise LedgerSchemaTooNewError(
                    f"agent ledger schema {row[0]} is newer than supported {LEDGER_SCHEMA_VERSION}"
                )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            conn.close()
            raise
        return conn

    # -- generic helpers ----------------------------------------------------

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
        with self._lock:
            return list(self._conn.execute(sql, tuple(params)).fetchall())

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        with self._lock:
            return int(self._conn.execute(sql, tuple(params)).rowcount)

    def transaction(self, statements: Iterable[tuple[str, Sequence[Any]]]) -> None:
        """Run ``statements`` atomically."""
        with self._lock:
            conn = self._conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                for sql, params in statements:
                    conn.execute(sql, tuple(params))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    @property
    def connection(self) -> sqlite3.Connection:
        """Raw connection; callers must hold :attr:`lock`."""
        return self._conn

    # -- atom texts ---------------------------------------------------------

    def put_text(self, digest: str, text: str, now: float) -> None:
        self.execute(
            "INSERT OR IGNORE INTO atom_texts(digest, text, created_at) VALUES (?,?,?)",
            (digest, text, now),
        )

    def get_text(self, digest: str) -> str | None:
        rows = self.query("SELECT text FROM atom_texts WHERE digest=?", (digest,))
        return str(rows[0][0]) if rows else None

    def stats(self) -> dict[str, Any]:
        with self._lock:
            facts = self._conn.execute("SELECT COUNT(*) FROM evidence_facts").fetchone()[0]
            files = self._conn.execute("SELECT COUNT(*) FROM graph_files").fetchone()[0]
        return {
            "backend": "sqlite" if self.persistent else "memory",
            "evidence_facts": int(facts),
            "graph_files": int(files),
            "schema_version": LEDGER_SCHEMA_VERSION,
        }

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass


def open_ledger(store: Any) -> AgentLedgerDB:
    """Ledger beside a persistent event store, else in memory.

    A ledger that cannot be opened safely degrades to memory rather than
    failing the proxy (same rule as the event store).
    """
    store_path = getattr(store, "path", None)
    if not getattr(store, "persistent", False) or store_path is None:
        return AgentLedgerDB(None)
    ledger_path = Path(store_path).with_name("agent_ledger.db")
    try:
        if ledger_path.is_symlink():
            raise OSError(f"agent ledger database is a symlink: {ledger_path}")
        return AgentLedgerDB(ledger_path)
    except (OSError, sqlite3.Error, LedgerSchemaTooNewError) as exc:
        log.warning(
            "agent ledger unavailable (%s); keeping ledger state in memory",
            describe_exception(exc),
        )
        return AgentLedgerDB(None)
