"""Bounded speculative candidate preparation (unified-21 #15).

While one tool result is being gated/selected, or while a forwarded request
awaits the model, *pure* preparation work — content hashes, bounded chunks,
family-view candidate previews and derived CCR span indexes — runs ahead on
one worker per request, under a process-wide concurrency cap.

Rules:

* exactly one in-flight task per request (:class:`RequestSpeculation`), and
  at most :data:`PROCESS_CAP` across the process (a shared semaphore). A task
  that cannot get a slot is simply not run;
* every task has a deadline; :meth:`RequestSpeculation.take` never blocks the
  baseline path beyond ``wait_s`` (0 by default). A result that is not ready
  is cancelled/discarded and the caller computes it synchronously, so the
  winner never depends on scheduling (the work is a pure function of the
  complete authoritative input);
* the final guard, task context, selection and every CCR/store *write* stay
  on the authoritative path; speculation only prepares;
* :meth:`RequestSpeculation.close` cancels what has not started and never
  waits for running work; results arriving after close are discarded.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Hashable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from headroom.log_safety import describe_exception

log = logging.getLogger(__name__)

FEATURE_SPECULATION = "agent_speculation"

PROCESS_CAP = max(1, min(4, (os.cpu_count() or 2) // 2))
DEFAULT_DEADLINE_S = 2.0
MAX_QUEUED_PER_REQUEST = 8

_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()
_slots = threading.BoundedSemaphore(PROCESS_CAP)


def _pool() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(
                    max_workers=PROCESS_CAP, thread_name_prefix="headroom-speculate"
                )
    return _executor


class _Stats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.counters: dict[str, int] = {
            "submitted": 0,
            "completed": 0,
            "hits": 0,
            "misses": 0,
            "discarded_late": 0,
            "cancelled": 0,
            "expired": 0,
            "no_slot": 0,
            "dropped_queue_full": 0,
            "errors": 0,
        }
        self.saved_ms: deque[float] = deque(maxlen=512)

    def bump(self, name: str, n: int = 1) -> None:
        with self.lock:
            self.counters[name] = self.counters.get(name, 0) + n

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            saved = sum(self.saved_ms)
            return {**self.counters, "process_cap": PROCESS_CAP, "avoided_ms": round(saved, 2)}


STATS = _Stats()


class RequestSpeculation:
    """One request's speculation lane (1 worker, bounded queue)."""

    def __init__(self, *, enabled: bool = True, deadline_s: float = DEFAULT_DEADLINE_S) -> None:
        self.enabled = enabled
        self.deadline_s = deadline_s
        self._lock = threading.Lock()
        self._queue: deque[tuple[Hashable, Callable[[], Any], float]] = deque()
        self._running: tuple[Hashable, Future[Any]] | None = None
        self._results: dict[Hashable, tuple[Any, float]] = {}
        self._abandoned: set[Hashable] = set()
        self._closed = False
        self._cancelled = False

    # -- producer -------------------------------------------------------------

    def submit(self, key: Hashable, fn: Callable[[], Any]) -> bool:
        """Queue pure work for ``key``; returns False when not accepted."""
        if not self.enabled:
            return False
        with self._lock:
            if self._closed or key in self._results:
                return False
            if any(k == key for k, _, _ in self._queue) or (
                self._running is not None and self._running[0] == key
            ):
                return False
            if len(self._queue) >= MAX_QUEUED_PER_REQUEST:
                STATS.bump("dropped_queue_full")
                return False
            self._queue.append((key, fn, time.monotonic() + self.deadline_s))
        STATS.bump("submitted")
        self._pump()
        return True

    def _pump(self) -> None:
        with self._lock:
            if self._cancelled or self._running is not None or not self._queue:
                return
            now = time.monotonic()
            while self._queue and self._queue[0][2] <= now:
                self._queue.popleft()
                STATS.bump("expired")
            if not self._queue:
                return
            if not _slots.acquire(blocking=False):
                STATS.bump("no_slot")
                return  # retried on the next submit/take/completion
            key, fn, deadline = self._queue.popleft()
            try:
                future = _pool().submit(self._run, key, fn, deadline)
            except RuntimeError:  # interpreter shutting down
                _slots.release()
                return
            self._running = (key, future)

    def _run(self, key: Hashable, fn: Callable[[], Any], deadline: float) -> None:
        started = time.monotonic()
        try:
            if started > deadline:
                STATS.bump("expired")
                return
            try:
                value = fn()
            except Exception as exc:  # noqa: BLE001 - speculation never fails a request
                STATS.bump("errors")
                log.debug("speculative prep failed: %s", describe_exception(exc))
                return
            elapsed = (time.monotonic() - started) * 1000.0
            with self._lock:
                if self._cancelled or key in self._abandoned or time.monotonic() > deadline:
                    self._abandoned.discard(key)
                    STATS.bump("discarded_late")
                    return
                self._results[key] = (value, elapsed)
            STATS.bump("completed")
        finally:
            _slots.release()
            with self._lock:
                if self._running is not None and self._running[0] == key:
                    self._running = None
            self._pump()

    # -- consumer -------------------------------------------------------------

    def take(self, key: Hashable, wait_s: float = 0.0) -> tuple[bool, Any]:
        """``(True, value)`` if prepared in time; otherwise ``(False, None)``.

        A queued-but-unstarted task for ``key`` is cancelled (the caller is
        about to compute it itself); a running one is left to finish and its
        late result is discarded.
        """
        with self._lock:
            hit = self._results.pop(key, None)
            running = self._running if self._running and self._running[0] == key else None
        if hit is None and running is not None and wait_s > 0:
            try:
                running[1].result(timeout=wait_s)
            except Exception:  # noqa: BLE001 - timeout or task error: compute inline
                pass
            with self._lock:
                hit = self._results.pop(key, None)
        if hit is not None:
            STATS.bump("hits")
            with STATS.lock:
                STATS.saved_ms.append(hit[1])
            return True, hit[0]
        with self._lock:
            before = len(self._queue)
            self._queue = deque(item for item in self._queue if item[0] != key)
            cancelled = before - len(self._queue)
            if running is not None:
                # The caller computes it now; the eventual result is stale.
                self._abandoned.add(key)
        if cancelled:
            STATS.bump("cancelled", cancelled)
        STATS.bump("misses")
        self._pump()
        return False, None

    def close(self) -> None:
        """Cancel queued work; never wait for running work."""
        with self._lock:
            if self._cancelled:
                return
            self._closed = True
            self._cancelled = True
            cancelled = len(self._queue)
            self._queue.clear()
            self._results.clear()
        if cancelled:
            STATS.bump("cancelled", cancelled)

    def detach(self) -> None:
        """No further submissions; queued work still runs until its deadline.

        Used for prep whose value is a derived side effect (a CCR span index)
        once the request has been forwarded and awaits the model: nothing
        waits for it, and a task past its deadline is dropped unstarted.
        """
        with self._lock:
            self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed


def speculation_status() -> dict[str, Any]:
    return STATS.snapshot()


__all__ = [
    "FEATURE_SPECULATION",
    "PROCESS_CAP",
    "RequestSpeculation",
    "speculation_status",
]
