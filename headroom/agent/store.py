"""Agent event store: append-only bounded events plus materialized projections.

Two implementations share one contract:

* :class:`SQLiteAgentStore` — WAL journal (falls back to the default rollback
  journal where WAL is unavailable), a ``meta.schema_version`` row,
  transactional event-insert + projection update, idempotent inserts keyed on
  the event id, revision compare-and-swap on projections, replay-to-rebuild,
  and read-only salvage when the database is corrupt.
* :class:`MemoryAgentStore` — the same semantics in bounded memory, used in
  stateless mode (zero filesystem writes) and when persistence is refused.

Only the runtime's worker thread writes; every method is guarded by a bounded
lock so a diagnostic read from another thread cannot deadlock the worker.

Projectors receive :class:`~headroom.agent.events.EventEnvelope` objects. On
live commit ``envelope.content`` may carry the transient source text; on
replay it is always ``None`` (content is never persisted), so a projector must
derive persisted state from the event fields alone or tolerate the absence.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from headroom import fileperms
from headroom.agent.events import AgentEvent, EventEnvelope
from headroom.log_safety import describe_exception

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_MAX_EVENTS = 50_000
_LOCK_TIMEOUT_S = 5.0
_BUSY_TIMEOUT_MS = 2000

_EVENT_COLUMNS = (
    "id",
    "v",
    "workspace_id",
    "session_id",
    "task_id",
    "kind",
    "origin",
    "source_id",
    "resource_id",
    "resource_hash",
    "revision",
    "ts",
    "payload_digest",
    "redacted_metadata",
)


class StoreLockTimeout(RuntimeError):
    """The store lock could not be acquired within the bounded timeout."""


class ProjectionTx(Protocol):
    """Projection access scoped to one projector and one (workspace, session)."""

    def get(self, key: str) -> tuple[int, Any] | None: ...

    def compare_and_set(self, key: str, expected_revision: int, value: Any) -> bool: ...


@runtime_checkable
class Projector(Protocol):
    name: str

    def project(self, envelope: EventEnvelope, tx: ProjectionTx) -> None: ...


class _Lock:
    def __init__(self) -> None:
        self._lock = threading.RLock()

    def __enter__(self) -> None:
        if not self._lock.acquire(timeout=_LOCK_TIMEOUT_S):
            raise StoreLockTimeout("agent store lock timed out")

    def __exit__(self, *exc: object) -> None:
        self._lock.release()


def _run_projectors(
    projectors: Sequence[Projector], envelope: EventEnvelope, make_tx: Any
) -> None:
    for projector in projectors:
        try:
            projector.project(envelope, make_tx(projector.name, envelope.event))
        except Exception as exc:  # a projector bug must not drop the event
            log.warning(
                "agent projector %s failed: %s",
                getattr(projector, "name", type(projector).__name__),
                describe_exception(exc),
            )


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------


class _MemoryTx:
    def __init__(self, store: MemoryAgentStore, name: str, ws: str, sess: str) -> None:
        self._store, self._name, self._ws, self._sess = store, name, ws, sess

    def get(self, key: str) -> tuple[int, Any] | None:
        return self._store._projections.get((self._name, self._ws, self._sess, key))

    def compare_and_set(self, key: str, expected_revision: int, value: Any) -> bool:
        return self._store._cas((self._name, self._ws, self._sess, key), expected_revision, value)


class MemoryAgentStore:
    """Bounded in-memory store with the SQLite store's semantics."""

    persistent = False
    journal_mode = "memory"

    def __init__(self, *, max_events: int = DEFAULT_MAX_EVENTS, max_projections: int = 20_000):
        self._max_events = max(1, int(max_events))
        self._max_projections = max(1, int(max_projections))
        self._events: deque[AgentEvent] = deque()
        self._ids: set[str] = set()
        self._projections: OrderedDict[tuple[str, str, str, str], tuple[int, Any]] = OrderedDict()
        self._lock = _Lock()
        self.recovered = False

    def _cas(self, key: tuple[str, str, str, str], expected: int, value: Any) -> bool:
        current = self._projections.get(key)
        revision = current[0] if current else 0
        if revision != expected:
            return False
        # Round-trip through JSON so memory and SQLite stores behave identically.
        self._projections[key] = (revision + 1, json.loads(json.dumps(value, default=str)))
        self._projections.move_to_end(key)
        while len(self._projections) > self._max_projections:
            self._projections.popitem(last=False)
        return True

    def commit(
        self, envelopes: Iterable[EventEnvelope], projectors: Sequence[Projector] = ()
    ) -> list[AgentEvent]:
        committed: list[AgentEvent] = []
        with self._lock:
            for envelope in envelopes:
                event = envelope.event
                if event.id in self._ids:
                    continue
                self._events.append(event)
                self._ids.add(event.id)
                committed.append(event)
                _run_projectors(
                    projectors,
                    envelope,
                    lambda name, ev: _MemoryTx(self, name, ev.workspace_id, ev.session_id),
                )
            while len(self._events) > self._max_events:
                self._ids.discard(self._events.popleft().id)
        return committed

    def contains(self, event_id: str) -> bool:
        with self._lock:
            return event_id in self._ids

    def events(
        self,
        *,
        workspace_id: str | None = None,
        session_id: str | None = None,
        limit: int | None = None,
    ) -> list[AgentEvent]:
        with self._lock:
            selected = [
                ev
                for ev in self._events
                if (workspace_id is None or ev.workspace_id == workspace_id)
                and (session_id is None or ev.session_id == session_id)
            ]
        return selected[-limit:] if limit else selected

    def get_projection(self, name: str, ws: str, sess: str, key: str) -> tuple[int, Any] | None:
        with self._lock:
            return self._projections.get((name, ws, sess, key))

    def compare_and_set(
        self, name: str, ws: str, sess: str, key: str, expected_revision: int, value: Any
    ) -> bool:
        with self._lock:
            return self._cas((name, ws, sess, key), expected_revision, value)

    def rebuild(self, projectors: Sequence[Projector]) -> int:
        with self._lock:
            self._projections.clear()
            events = list(self._events)
            for event in events:
                _run_projectors(
                    projectors,
                    EventEnvelope(event=event),
                    lambda name, ev: _MemoryTx(self, name, ev.workspace_id, ev.session_id),
                )
        return len(events)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "backend": "memory",
                "journal_mode": self.journal_mode,
                "events": len(self._events),
                "projections": len(self._projections),
                "schema_version": SCHEMA_VERSION,
            }

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# SQLite store
# ---------------------------------------------------------------------------


class SchemaTooNewError(RuntimeError):
    """The database was written by a newer schema; never downgrade it."""


class _SQLiteTx:
    def __init__(self, conn: sqlite3.Connection, name: str, ws: str, sess: str) -> None:
        self._conn, self._name, self._ws, self._sess = conn, name, ws, sess

    def get(self, key: str) -> tuple[int, Any] | None:
        return _sql_get(self._conn, self._name, self._ws, self._sess, key)

    def compare_and_set(self, key: str, expected_revision: int, value: Any) -> bool:
        return _sql_cas(self._conn, self._name, self._ws, self._sess, key, expected_revision, value)


def _sql_get(
    conn: sqlite3.Connection, name: str, ws: str, sess: str, key: str
) -> tuple[int, Any] | None:
    row = conn.execute(
        "SELECT revision, value FROM agent_projections "
        "WHERE name=? AND workspace_id=? AND session_id=? AND key=?",
        (name, ws, sess, key),
    ).fetchone()
    return (int(row[0]), json.loads(row[1])) if row else None


def _sql_cas(
    conn: sqlite3.Connection,
    name: str,
    ws: str,
    sess: str,
    key: str,
    expected: int,
    value: Any,
) -> bool:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    now = time.time()
    if expected == 0:
        cur = conn.execute(
            "INSERT OR IGNORE INTO agent_projections"
            "(name, workspace_id, session_id, key, revision, value, updated_at) "
            "VALUES (?,?,?,?,1,?,?)",
            (name, ws, sess, key, encoded, now),
        )
        return cur.rowcount == 1
    cur = conn.execute(
        "UPDATE agent_projections SET revision=revision+1, value=?, updated_at=? "
        "WHERE name=? AND workspace_id=? AND session_id=? AND key=? AND revision=?",
        (encoded, now, name, ws, sess, key, expected),
    )
    return cur.rowcount == 1


_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS agent_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS agent_events ("
    " seq INTEGER PRIMARY KEY AUTOINCREMENT,"
    " id TEXT NOT NULL UNIQUE, v INTEGER NOT NULL,"
    " workspace_id TEXT NOT NULL, session_id TEXT NOT NULL, task_id TEXT,"
    " kind TEXT NOT NULL, origin TEXT NOT NULL, source_id TEXT NOT NULL,"
    " resource_id TEXT, resource_hash TEXT, revision INTEGER NOT NULL DEFAULT 0,"
    " ts REAL NOT NULL, payload_digest TEXT NOT NULL, redacted_metadata TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS agent_events_scope ON agent_events(workspace_id, session_id, seq)",
    "CREATE TABLE IF NOT EXISTS agent_projections ("
    " name TEXT NOT NULL, workspace_id TEXT NOT NULL, session_id TEXT NOT NULL,"
    " key TEXT NOT NULL, revision INTEGER NOT NULL, value TEXT NOT NULL,"
    " updated_at REAL NOT NULL, PRIMARY KEY (name, workspace_id, session_id, key))",
)


class SQLiteAgentStore:
    """Durable store under ``paths.agent_state_db_path()``."""

    persistent = True

    def __init__(self, path: Path, *, max_events: int = DEFAULT_MAX_EVENTS) -> None:
        self.path = Path(path)
        self._max_events = max(1, int(max_events))
        self._lock = _Lock()
        self.recovered = False
        self.salvaged_events = 0
        self.journal_mode = "unknown"
        self._conn = self._open()

    # -- opening / recovery ---------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = fileperms.connect_private_sqlite(
            self.path,
            what="agent state database",
            timeout=_BUSY_TIMEOUT_MS / 1000,
            check_same_thread=False,
            isolation_level=None,
        )
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        return conn

    def _open(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = None
        try:
            conn = self._connect()
            check = conn.execute("PRAGMA quick_check").fetchone()
            if not check or check[0] != "ok":
                raise sqlite3.DatabaseError(f"quick_check failed: {check[0] if check else None}")
            self._init_schema(conn)
            return conn
        except SchemaTooNewError:
            if conn is not None:
                conn.close()
            raise
        except sqlite3.DatabaseError as exc:
            if conn is not None:
                conn.close()
            log.warning(
                "agent state database unusable (%s); salvaging read-only and recreating",
                describe_exception(exc),
            )
            return self._recover()

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        mode_row = conn.execute("PRAGMA journal_mode=WAL").fetchone()
        self.journal_mode = str(mode_row[0]).lower() if mode_row else "unknown"
        if self.journal_mode != "wal":
            # Network filesystems and some sandboxes refuse WAL; the rollback
            # journal keeps the same transactional guarantees, just slower.
            fallback = conn.execute("PRAGMA journal_mode=DELETE").fetchone()
            self.journal_mode = str(fallback[0]).lower() if fallback else "delete"
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("BEGIN IMMEDIATE")
        try:
            for statement in _SCHEMA:
                conn.execute(statement)
            row = conn.execute(
                "SELECT value FROM agent_meta WHERE key='schema_version'"
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO agent_meta(key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            elif int(row[0]) > SCHEMA_VERSION:
                raise SchemaTooNewError(
                    f"agent state schema {row[0]} is newer than supported {SCHEMA_VERSION}"
                )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    def _salvage(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not self.path.exists():
            return rows
        uri = "file:" + self.path.resolve().as_posix() + "?mode=ro"
        try:
            ro = sqlite3.connect(uri, uri=True, timeout=1.0)
        except sqlite3.Error:
            return rows
        try:
            cursor = ro.execute(
                f"SELECT {', '.join(_EVENT_COLUMNS)} FROM agent_events ORDER BY seq"
            )
            while True:
                try:
                    row = cursor.fetchone()
                except sqlite3.Error:
                    break
                if row is None:
                    break
                rows.append(dict(zip(_EVENT_COLUMNS, row, strict=True)))
        except sqlite3.Error:
            pass
        finally:
            ro.close()
        return rows

    def _recover(self) -> sqlite3.Connection:
        salvaged = self._salvage()
        stamp = time.strftime("%Y%m%dT%H%M%S")
        for suffix in ("", "-wal", "-shm", "-journal"):
            source = Path(str(self.path) + suffix)
            if source.exists():
                try:
                    os.replace(source, Path(f"{self.path}.corrupt-{stamp}{suffix}"))
                except OSError:
                    source.unlink(missing_ok=True)
        conn = self._connect()
        self._init_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        try:
            for row in salvaged:
                conn.execute(
                    f"INSERT OR IGNORE INTO agent_events({', '.join(_EVENT_COLUMNS)}) "
                    f"VALUES ({', '.join('?' for _ in _EVENT_COLUMNS)})",
                    tuple(row[c] for c in _EVENT_COLUMNS),
                )
            conn.execute("COMMIT")
        except sqlite3.Error:
            conn.execute("ROLLBACK")
            salvaged = []
        self.recovered = True
        self.salvaged_events = len(salvaged)
        return conn

    # -- writes -----------------------------------------------------------

    def commit(
        self, envelopes: Iterable[EventEnvelope], projectors: Sequence[Projector] = ()
    ) -> list[AgentEvent]:
        """Insert new events and run projectors for them in ONE transaction."""
        batch = list(envelopes)
        if not batch:
            return []
        committed: list[AgentEvent] = []
        with self._lock:
            conn = self._conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                for envelope in batch:
                    row = envelope.event.to_row()
                    cur = conn.execute(
                        f"INSERT OR IGNORE INTO agent_events({', '.join(_EVENT_COLUMNS)}) "
                        f"VALUES ({', '.join('?' for _ in _EVENT_COLUMNS)})",
                        tuple(row[c] for c in _EVENT_COLUMNS),
                    )
                    if cur.rowcount != 1:
                        continue  # already processed: at-most-once
                    committed.append(envelope.event)
                    _run_projectors(
                        projectors,
                        envelope,
                        lambda name, ev: _SQLiteTx(conn, name, ev.workspace_id, ev.session_id),
                    )
                self._prune(conn)
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return committed

    def _prune(self, conn: sqlite3.Connection) -> None:
        row = conn.execute("SELECT COUNT(*), MIN(seq), MAX(seq) FROM agent_events").fetchone()
        count = int(row[0] or 0)
        if count > self._max_events:
            excess = count - self._max_events
            conn.execute(
                "DELETE FROM agent_events WHERE seq IN "
                "(SELECT seq FROM agent_events ORDER BY seq LIMIT ?)",
                (excess,),
            )

    def compare_and_set(
        self, name: str, ws: str, sess: str, key: str, expected_revision: int, value: Any
    ) -> bool:
        with self._lock:
            conn = self._conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                ok = _sql_cas(conn, name, ws, sess, key, expected_revision, value)
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return ok

    def rebuild(self, projectors: Sequence[Projector]) -> int:
        """Drop projections and replay every retained event through ``projectors``."""
        with self._lock:
            conn = self._conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute("DELETE FROM agent_projections")
                rows = conn.execute(
                    f"SELECT {', '.join(_EVENT_COLUMNS)} FROM agent_events ORDER BY seq"
                ).fetchall()
                for raw in rows:
                    event = AgentEvent.from_row(dict(zip(_EVENT_COLUMNS, raw, strict=True)))
                    _run_projectors(
                        projectors,
                        EventEnvelope(event=event),
                        lambda name, ev: _SQLiteTx(conn, name, ev.workspace_id, ev.session_id),
                    )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return len(rows)

    # -- reads ------------------------------------------------------------

    def contains(self, event_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM agent_events WHERE id=?", (event_id,)
            ).fetchone()
        return row is not None

    def events(
        self,
        *,
        workspace_id: str | None = None,
        session_id: str | None = None,
        limit: int | None = None,
    ) -> list[AgentEvent]:
        clauses, params = [], []
        if workspace_id is not None:
            clauses.append("workspace_id=?")
            params.append(workspace_id)
        if session_id is not None:
            clauses.append("session_id=?")
            params.append(session_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT {', '.join(_EVENT_COLUMNS)} FROM agent_events{where} ORDER BY seq"
        if limit:
            sql = (
                f"SELECT {', '.join(_EVENT_COLUMNS)} FROM (SELECT seq, "
                f"{', '.join(_EVENT_COLUMNS)} FROM agent_events{where} "
                f"ORDER BY seq DESC LIMIT {int(limit)}) ORDER BY seq"
            )
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [AgentEvent.from_row(dict(zip(_EVENT_COLUMNS, r, strict=True))) for r in rows]

    def get_projection(self, name: str, ws: str, sess: str, key: str) -> tuple[int, Any] | None:
        with self._lock:
            return _sql_get(self._conn, name, ws, sess, key)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            events = self._conn.execute("SELECT COUNT(*) FROM agent_events").fetchone()[0]
            projections = self._conn.execute("SELECT COUNT(*) FROM agent_projections").fetchone()[0]
        return {
            "backend": "sqlite",
            "journal_mode": self.journal_mode,
            "events": int(events),
            "projections": int(projections),
            "schema_version": SCHEMA_VERSION,
            "recovered": self.recovered,
            "salvaged_events": self.salvaged_events,
        }

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
