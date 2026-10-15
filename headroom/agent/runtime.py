"""AgentRuntime: the single owner of session events and their storage.

Request handlers never touch SQLite. The TurnHook normalizes events on the
event loop (pure CPU, digests only) and hands them to :meth:`submit`, which is
non-blocking: a bounded queue feeds one background worker thread that commits
batches (event insert + projections in one transaction) and then notifies
commit listeners. A full queue drops the batch and counts it — observation is
best-effort and can never stall a request.

Replays of the same history (every turn resends it) are filtered twice: a
bounded in-memory LRU of recently submitted event ids skips them before they
are queued, and the store's unique event id makes the insert at-most-once even
after a restart.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from headroom import paths
from headroom.agent import registry
from headroom.agent.events import EventEnvelope
from headroom.agent.identity import WorkspaceIdentity
from headroom.agent.interfaces import (
    ACTION_AUTHORIZATION_VERSION,
    ActionAuthorization,
    AuthorizationDecision,
)
from headroom.agent.private_fs import UnsafePathError, ensure_within, secure_dir
from headroom.agent.store import MemoryAgentStore, SchemaTooNewError, SQLiteAgentStore
from headroom.log_safety import describe_exception

log = logging.getLogger(__name__)

DEFAULT_QUEUE_SIZE = 1024
DEFAULT_RECENT_IDS = 16_384
DEFAULT_AUTHORIZATION_TTL_S = 60.0
_STOP = object()


class AgentRuntime:
    """Session-safe event runtime (see module docstring)."""

    def __init__(
        self,
        store: MemoryAgentStore | SQLiteAgentStore,
        *,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        recent_ids: int = DEFAULT_RECENT_IDS,
        advisor: Any = None,
    ) -> None:
        self.store = store
        self.advisor = advisor
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, int(queue_size)))
        self._recent: OrderedDict[str, None] = OrderedDict()
        self._recent_cap = max(1, int(recent_ids))
        self._recent_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()
        self._stopped = False
        self._idle = threading.Condition()
        self._pending = 0
        self.counters: dict[str, int] = {
            "submitted_batches": 0,
            "submitted_events": 0,
            "skipped_replays": 0,
            "dropped_batches": 0,
            "committed_events": 0,
            "commit_errors": 0,
            "trusted_requests": 0,
            "untrusted_requests": 0,
        }
        self.last_error: str | None = None

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        with self._start_lock:
            if self._thread is not None or self._stopped:
                return
            self._thread = threading.Thread(
                target=self._run, name="headroom-agent-runtime", daemon=True
            )
            self._thread.start()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self, timeout: float = 3.0) -> None:
        """Drain the queue (bounded by ``timeout``), stop the worker, close the store."""
        with self._start_lock:
            if self._stopped:
                return
            self._stopped = True
            thread = self._thread
        if thread is not None:
            try:
                self._queue.put(_STOP, timeout=timeout)
            except queue.Full:
                pass
            thread.join(timeout)
        self.store.close()

    # -- submission (event loop side) -------------------------------------

    def _filter_recent(self, envelopes: Sequence[EventEnvelope]) -> list[EventEnvelope]:
        fresh: list[EventEnvelope] = []
        with self._recent_lock:
            for envelope in envelopes:
                event_id = envelope.event.id
                if event_id in self._recent:
                    self._recent.move_to_end(event_id)
                    self.counters["skipped_replays"] += 1
                    continue
                self._recent[event_id] = None
                fresh.append(envelope)
            while len(self._recent) > self._recent_cap:
                self._recent.popitem(last=False)
        return fresh

    def _forget(self, envelopes: Sequence[EventEnvelope]) -> None:
        with self._recent_lock:
            for envelope in envelopes:
                self._recent.pop(envelope.event.id, None)

    def submit(self, envelopes: Sequence[EventEnvelope]) -> bool:
        """Queue events for the worker. Never blocks; returns False when dropped."""
        if self._stopped:
            return False
        fresh = self._filter_recent(envelopes)
        if not fresh:
            return True
        self.start()
        with self._idle:
            self._pending += 1
        try:
            self._queue.put_nowait(fresh)
        except queue.Full:
            with self._idle:
                self._pending -= 1
                self._idle.notify_all()
            # Forget them so a later replay of the same history can retry.
            self._forget(fresh)
            self.counters["dropped_batches"] += 1
            return False
        self.counters["submitted_batches"] += 1
        self.counters["submitted_events"] += len(fresh)
        return True

    def note_request(self, identity: WorkspaceIdentity) -> None:
        key = "trusted_requests" if identity.trusted else "untrusted_requests"
        self.counters[key] += 1

    # -- worker -------------------------------------------------------------

    def _run(self) -> None:
        if getattr(self.store, "recovered", False):
            try:
                self.store.rebuild(registry.projectors())
            except Exception as exc:  # noqa: BLE001 - rebuild failure leaves projections empty
                self.last_error = describe_exception(exc)
        while True:
            item = self._queue.get()
            if item is _STOP:
                break
            try:
                committed = self.store.commit(item, registry.projectors())
                self.counters["committed_events"] += len(committed)
                if committed:
                    for name, listener in registry.commit_listeners():
                        try:
                            listener(committed, self)
                        except Exception as exc:  # noqa: BLE001 - isolate listeners
                            log.warning(
                                "agent commit listener %s failed: %s",
                                name,
                                describe_exception(exc),
                            )
            except Exception as exc:  # noqa: BLE001 - the worker must survive
                self.counters["commit_errors"] += 1
                self.last_error = describe_exception(exc)
                self._forget(item)
                log.warning("agent runtime commit failed: %s", self.last_error)
            finally:
                with self._idle:
                    self._pending -= 1
                    self._idle.notify_all()

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until every submitted batch has been processed."""
        deadline = time.monotonic() + timeout
        with self._idle:
            while self._pending > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._idle.wait(remaining)
        return True

    # -- authorization ------------------------------------------------------

    def authorize_action(
        self,
        identity: WorkspaceIdentity,
        *,
        task_revision: int,
        scope_revision: int,
        resource_revision: int,
        decision: AuthorizationDecision,
        ttl_s: float = DEFAULT_AUTHORIZATION_TTL_S,
    ) -> ActionAuthorization | None:
        """Issue a revision-pinned authorization; ``None`` for an untrusted identity.

        Only a trusted wrap-registered identity can ever receive one. Advisory
        confidence is not an input here by design.
        """
        if not identity.can_authorize_paths():
            return None
        return ActionAuthorization(
            v=ACTION_AUTHORIZATION_VERSION,
            workspace_id=identity.workspace_id,
            session_id=identity.session_id,
            task_revision=int(task_revision),
            scope_revision=int(scope_revision),
            resource_revision=int(resource_revision),
            expires_at=time.time() + max(1.0, float(ttl_s)),
            decision=decision,
        )

    # -- diagnostics --------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Cheap status for /health (no SQLite access)."""
        return {
            "running": self.running,
            "store": {
                "backend": "sqlite" if self.store.persistent else "memory",
                "journal_mode": self.store.journal_mode,
                "recovered": bool(getattr(self.store, "recovered", False)),
            },
            "queue_depth": self._queue.qsize(),
            "counters": dict(self.counters),
            "last_error": self.last_error,
            "extensions": registry.snapshot(),
        }


def create_store(
    *, stateless: bool, max_events: int | None = None
) -> MemoryAgentStore | SQLiteAgentStore:
    """Durable SQLite store when persistence is allowed, else bounded memory.

    Stateless mode returns the memory store before touching the filesystem.
    A store that cannot be opened safely (symlink escape, newer schema,
    unwritable directory) degrades to memory rather than failing the proxy.
    """
    limit = {"max_events": max_events} if max_events else {}
    if stateless or not paths.persistence_allowed("agent runtime state"):
        return MemoryAgentStore(**limit)
    db_path = paths.agent_state_db_path()
    try:
        default_location = not paths._env(paths.HEADROOM_AGENT_STATE_DB_ENV)
        root = paths.workspace_dir()
        if default_location:
            root.mkdir(parents=True, exist_ok=True)
        secure_dir(db_path.parent)
        if default_location:
            # A symlink anywhere between the workspace and the db directory
            # must not redirect agent state outside the workspace.
            ensure_within(db_path.parent, root)
        if db_path.is_symlink():
            raise UnsafePathError(f"agent state database is a symlink: {db_path}")
        return SQLiteAgentStore(Path(db_path), **limit)
    except (OSError, SchemaTooNewError) as exc:
        log.warning(
            "agent state database unavailable (%s); keeping agent state in memory",
            describe_exception(exc),
        )
        return MemoryAgentStore(**limit)
