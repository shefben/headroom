"""Bounded retention and scoped purge for agent state (unified-21 Phase 8).

Defaults (each overridable through the existing env/settings mechanism):

* events (``agent_events``): 30 days — ``HEADROOM_AGENT_EVENT_RETENTION_DAYS``;
* derived per-session state (projections, evidence facts, resource versions,
  processed-event marks): 30 days after the session's last activity —
  ``HEADROOM_AGENT_SESSION_RETENTION_DAYS``;
* workspace aggregates (verification history, workflow observations, learned
  macros, feedback priors, graph index): 90 days — ``HEADROOM_AGENT_WORKSPACE_RETENTION_DAYS``.

Exact CCR keeps its own TTL and is not touched here; learned macros keep their
32-per-workspace cap. :func:`prune` runs on its own background thread at most every
:data:`PRUNE_INTERVAL_S` (the first commit after start triggers it), never on a
request path or the runtime worker. :func:`purge_workspace` removes one
workspace's rows from the two agent databases and nothing else (no memory,
CCR, savings or settings data).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from headroom.log_safety import describe_exception

log = logging.getLogger(__name__)

DAY = 86_400.0
PRUNE_INTERVAL_S = 6 * 3600.0
LISTENER = "agent.retention"

EVENTS_ENV = "HEADROOM_AGENT_EVENT_RETENTION_DAYS"
SESSION_ENV = "HEADROOM_AGENT_SESSION_RETENTION_DAYS"
WORKSPACE_ENV = "HEADROOM_AGENT_WORKSPACE_RETENTION_DAYS"


def _days(env: Mapping[str, str], name: str, default: float) -> float:
    try:
        value = float(env.get(name) or default)
    except ValueError:
        return default
    return default if value != value or value <= 0 else min(value, 3650.0)


@dataclass(frozen=True)
class RetentionPolicy:
    event_days: float = 30.0
    session_days: float = 30.0
    workspace_days: float = 90.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> RetentionPolicy:
        source = os.environ if env is None else env
        return cls(
            _days(source, EVENTS_ENV, 30.0),
            _days(source, SESSION_ENV, 30.0),
            _days(source, WORKSPACE_ENV, 90.0),
        )


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _session_stale(table: str, ts: str) -> str:
    return (
        f"DELETE FROM {table} WHERE (workspace_id, session_id) IN ("
        f"SELECT workspace_id, session_id FROM {table} GROUP BY workspace_id, session_id "
        f"HAVING MAX({ts}) < ?)"
    )


#: (database, table, statement, cutoff kind). Only tables that exist are touched.
_STATE_RULES: tuple[tuple[str, str, str], ...] = (
    ("agent_events", "DELETE FROM agent_events WHERE ts < ?", "event"),
    ("agent_projections", _session_stale("agent_projections", "updated_at"), "session"),
)
_LEDGER_RULES: tuple[tuple[str, str, str], ...] = (
    ("evidence_facts", _session_stale("evidence_facts", "updated_at"), "session"),
    ("evidence_processed", "DELETE FROM evidence_processed WHERE ts < ?", "session"),
    (
        "tool_resource_versions",
        "DELETE FROM tool_resource_versions WHERE updated_at < ?",
        "session",
    ),
    ("verification_runs", "DELETE FROM verification_runs WHERE ts < ?", "workspace"),
    ("workflow_observations", "DELETE FROM workflow_observations WHERE ts < ?", "workspace"),
    ("workflow_macros", "DELETE FROM workflow_macros WHERE updated_at < ?", "workspace"),
    (
        "agent_feedback_priors",
        "DELETE FROM agent_feedback_priors WHERE updated_at < ?",
        "workspace",
    ),
    ("graph_files", "DELETE FROM graph_files WHERE indexed_at < ?", "workspace"),
)


def _apply(
    conn: sqlite3.Connection,
    rules: tuple[tuple[str, str, str], ...],
    cutoffs: Mapping[str, float],
) -> dict[str, int]:
    present = _tables(conn)
    removed: dict[str, int] = {}
    conn.execute("BEGIN IMMEDIATE")
    try:
        for table, sql, kind in rules:
            if table in present:
                removed[table] = int(conn.execute(sql, (cutoffs[kind],)).rowcount)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return {k: v for k, v in removed.items() if v}


def _prune_atom_texts(ledger_conn: sqlite3.Connection, live_text: str, cutoff: float) -> int:
    """Content-addressed atom texts past the session window are kept only while a
    retained projection still names their digest."""
    if "atom_texts" not in _tables(ledger_conn):
        return 0
    old = [
        r[0]
        for r in ledger_conn.execute(
            "SELECT digest FROM atom_texts WHERE created_at < ?", (cutoff,)
        )
    ]
    dead = [(d,) for d in old if d not in live_text]
    if dead:
        ledger_conn.executemany("DELETE FROM atom_texts WHERE digest = ?", dead)
    return len(dead)


def prune(
    store: Any, ledger: Any, *, policy: RetentionPolicy | None = None, now: float | None = None
) -> dict[str, int]:
    """Apply the retention windows to the persistent agent databases."""
    policy = policy or RetentionPolicy.from_env()
    now = time.time() if now is None else now
    cutoffs = {
        "event": now - policy.event_days * DAY,
        "session": now - policy.session_days * DAY,
        "workspace": now - policy.workspace_days * DAY,
    }
    removed: dict[str, int] = {}
    live_text = ""
    if getattr(store, "persistent", False) and getattr(store, "_conn", None) is not None:
        with store._lock:
            removed.update(_apply(store._conn, _STATE_RULES, cutoffs))
            live_text = "\n".join(
                str(r[0]) for r in store._conn.execute("SELECT value FROM agent_projections")
            )
    if ledger is not None and getattr(ledger, "persistent", False):
        with ledger.lock:
            conn = ledger.connection
            removed.update(_apply(conn, _LEDGER_RULES, cutoffs))
            if getattr(store, "persistent", False):
                n = _prune_atom_texts(conn, live_text, cutoffs["session"])
                if n:
                    removed["atom_texts"] = n
    return removed


class RetentionScheduler:
    """Starts :func:`prune` on its own daemon thread at most every ``interval`` seconds.

    It is triggered from a commit listener but never prunes there: the runtime
    worker (event commits, ``flush``) only checks the cadence and returns. One
    prune runs at a time; the SQLite connections' own locks serialize it with
    commits, one short transaction per database.
    """

    def __init__(
        self,
        runtime: Any,
        ledger: Any,
        *,
        interval: float = PRUNE_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.runtime, self.ledger = runtime, ledger
        self.interval, self._clock = interval, clock
        self._lock = threading.Lock()
        self._last: float | None = None
        self._thread: threading.Thread | None = None
        self._closed = False
        self.runs = 0
        self.last_removed: dict[str, int] = {}
        self.last_error: str | None = None

    def _claim(self) -> bool:
        """True when a prune is due and none is running (records the start time)."""
        with self._lock:
            if self._closed or (self._thread is not None and self._thread.is_alive()):
                return False
            now = self._clock()
            if self._last is not None and now - self._last < self.interval:
                return False
            self._last = now
            return True

    def run(self) -> None:
        """Prune body (background thread); never raises."""
        try:
            removed = prune(self.runtime.store, self.ledger)
        except sqlite3.ProgrammingError as exc:  # store closed during shutdown
            log.debug("agent retention skipped: %s", describe_exception(exc))
            return
        except Exception as exc:  # noqa: BLE001 - retention must not disturb the runtime
            self.last_error = describe_exception(exc)
            log.warning("agent retention prune failed: %s", self.last_error)
            return
        self.runs += 1
        self.last_removed = removed
        if removed:
            log.info("agent retention pruned %s", removed)

    def on_commit(self, _events: list[Any], _runtime: Any) -> None:
        """Commit listener: only starts the background prune when it is due."""
        if not self._claim():
            return
        thread = threading.Thread(target=self.run, name="headroom-agent-retention", daemon=True)
        with self._lock:
            self._thread = thread
        thread.start()

    def close(self, timeout: float = 2.0) -> None:
        """Stop scheduling and wait (bounded) for a running prune."""
        with self._lock:
            self._closed = True
            thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def status(self) -> dict[str, Any]:
        policy = RetentionPolicy.from_env()
        return {
            "event_days": policy.event_days,
            "session_days": policy.session_days,
            "workspace_days": policy.workspace_days,
            "runs": self.runs,
            "last_removed": dict(self.last_removed),
            "last_error": self.last_error,
        }


# -- operator purge ---------------------------------------------------------------


def workspace_id_for(path: str | os.PathLike[str]) -> str:
    """The trusted ``workspace_id`` wrap would derive for ``path`` (same principal)."""
    from headroom.agent.identity import resolve_identity

    identity = resolve_identity(session_token="purge", registered_cwd=str(path))
    if not identity.trusted:
        raise ValueError(f"not an existing directory: {path}")
    return identity.workspace_id


_PURGE_STATE = ("agent_events", "agent_projections")
_PURGE_LEDGER = (
    "evidence_facts",
    "graph_files",
    "verification_runs",
    "workflow_macros",
    "workflow_observations",
)


def _purge_db(
    path: Path, tables: tuple[str, ...], workspace_id: str, extra: Callable[..., int]
) -> dict[str, int]:
    from headroom import fileperms

    if not path.exists() or path.is_symlink():
        return {}
    conn = fileperms.connect_private_sqlite(
        path, what="agent database", timeout=2.0, check_same_thread=False, isolation_level=None
    )
    try:
        conn.execute("PRAGMA busy_timeout=2000")
        present = _tables(conn)
        removed: dict[str, int] = {}
        conn.execute("BEGIN IMMEDIATE")
        try:
            for table in tables:
                if table in present:
                    removed[table] = int(
                        conn.execute(
                            f"DELETE FROM {table} WHERE workspace_id = ?", (workspace_id,)
                        ).rowcount
                    )
            removed.update(extra(conn, present))
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        return {k: v for k, v in removed.items() if v}
    finally:
        conn.close()


def purge_workspace(workspace_id: str, *, state_db: Path | None = None) -> dict[str, int]:
    """Delete one workspace's persisted agent rows. Other workspaces and all
    non-agent Headroom data are untouched. A running proxy keeps its in-memory
    caches until it restarts."""
    from headroom import paths
    from headroom.agent.feedback import workspace_hash

    state = Path(state_db) if state_db is not None else paths.agent_state_db_path()
    ledger = state.with_name("agent_ledger.db")
    removed = _purge_db(state, _PURGE_STATE, workspace_id, lambda conn, present: {})
    ws_hash = workspace_hash(workspace_id)

    def ledger_extra(conn: sqlite3.Connection, present: set[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        if "tool_resource_versions" in present:
            out["tool_resource_versions"] = int(
                conn.execute(
                    "DELETE FROM tool_resource_versions WHERE substr(scope, 1, ?) = ?",
                    (len(workspace_id) + 1, workspace_id + "|"),
                ).rowcount
            )
        if "agent_feedback_priors" in present and ws_hash:
            out["agent_feedback_priors"] = int(
                conn.execute(
                    "DELETE FROM agent_feedback_priors WHERE substr(key, -?) = ?",
                    (len(ws_hash) + 1, "|" + ws_hash),
                ).rowcount
            )
        return out

    removed.update(_purge_db(ledger, _PURGE_LEDGER, workspace_id, ledger_extra))
    return removed


__all__ = [
    "RetentionPolicy",
    "RetentionScheduler",
    "prune",
    "purge_workspace",
    "workspace_id_for",
]
