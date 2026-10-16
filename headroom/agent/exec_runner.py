"""Bounded, cancellable argv execution for verification runs and workflow steps.

Every process Headroom itself starts on behalf of an agent (test discovery,
targeted test runs, compiled workflow-macro steps) goes through
:func:`run_argv`:

* **argv only, ``shell=False``** — there is no command-string form and no
  shell interpolation; callers pass a validated token list;
* **own process group** — POSIX ``start_new_session``; Windows
  ``CREATE_NEW_PROCESS_GROUP`` — so a timeout or cancellation kills the whole
  tree (``killpg``; ``taskkill /T /F`` on Windows), never just the parent;
* **strict deadline** and a cooperative ``cancel`` event polled while waiting;
* **bounded output** — stdout+stderr are merged (or, with
  ``merge_stderr=False``, kept apart: a tool whose stdout is its *result*,
  such as ripgrep, must not have diagnostics mixed into it) and kept up to
  ``capture_bytes``; anything beyond is counted and flagged as truncated so a
  partial output is never presented as complete;
* **scrubbed environment** — provider credentials and the wrap session token
  are removed from the child environment.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from headroom import _subprocess

DEFAULT_CAPTURE_BYTES = 4 * 1024 * 1024
_POLL_S = 0.05
_KILL_GRACE_S = 2.0
_SECRET_ENV = re.compile(
    r"(?:API_KEY|AUTH_TOKEN|ACCESS_TOKEN|SECRET|PASSWORD|SESSION_TOKEN|CUSTOM_HEADERS)", re.I
)
# Characters a Windows batch shim (npm.cmd, pnpm.cmd, ...) would interpret.
_BATCH_META = re.compile(r'[&|<>^%"\r\n]')


class ArgvError(ValueError):
    """The argv is not a safe, direct token list."""


@dataclass
class RunResult:
    argv: tuple[str, ...]
    exit_code: int | None
    output: str
    duration_s: float
    timed_out: bool = False
    cancelled: bool = False
    truncated: bool = False
    total_bytes: int = 0
    error: str | None = None
    started: bool = True
    tree_killed: bool = False
    meta: dict[str, Any] = field(default_factory=dict)
    #: Separate stderr (only with ``merge_stderr=False``; else merged into output).
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return (
            self.started
            and self.exit_code == 0
            and not self.timed_out
            and not self.cancelled
            and self.error is None
        )


def child_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment a child gets: the proxy's, minus credentials/tokens."""
    source = os.environ if base is None else base
    return {k: v for k, v in source.items() if not _SECRET_ENV.search(k)}


def validate_argv(argv: Sequence[str]) -> tuple[str, ...]:
    if not argv or not all(isinstance(a, str) for a in argv):
        raise ArgvError("argv must be a non-empty list of strings")
    out = tuple(argv)
    for token in out:
        if "\x00" in token or "\n" in token or "\r" in token:
            raise ArgvError("argv tokens may not contain NUL or newlines")
    exe = out[0].lower()
    if sys.platform == "win32" and exe.endswith((".cmd", ".bat")):
        # cmd.exe parses a batch file's arguments; refuse anything it would
        # interpret instead of passing through literally.
        for token in out[1:]:
            if _BATCH_META.search(token):
                raise ArgvError("argument not safe for a Windows batch launcher")
    return out


def resolve_executable(name: str) -> str | None:
    """Absolute path of an executable on PATH (``.cmd``/``.exe`` aware on Windows)."""
    import shutil

    if os.path.isabs(name):
        return name if os.path.exists(name) else None
    return shutil.which(name)


def _popen_kwargs() -> dict[str, Any]:
    if sys.platform == "win32":
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200) | getattr(
            subprocess, "CREATE_NO_WINDOW", 0x08000000
        )
        return {"creationflags": flags}
    return {"start_new_session": True}


def kill_tree(proc: subprocess.Popen[bytes]) -> bool:
    """Kill ``proc`` and every descendant in its process group/tree."""
    if proc.poll() is not None and sys.platform != "win32":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        return True
    if sys.platform == "win32":
        try:
            _subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=10,
                check=False,
            )
        except Exception:  # noqa: BLE001 - fall back to killing the parent
            pass
        try:
            proc.kill()
        except OSError:
            pass
        return True
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        pass
    deadline = time.monotonic() + _KILL_GRACE_S
    while time.monotonic() < deadline and proc.poll() is None:
        time.sleep(_POLL_S)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass
    return True


class _Collector:
    def __init__(self, cap: int) -> None:
        self.cap = max(1024, int(cap))
        self.chunks: list[bytes] = []
        self.kept = 0
        self.total = 0
        self.truncated = False

    def feed(self, stream: Any) -> None:
        try:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                self.total += len(chunk)
                room = self.cap - self.kept
                if room > 0:
                    piece = chunk[:room]
                    self.chunks.append(piece)
                    self.kept += len(piece)
                if len(chunk) > max(room, 0):
                    self.truncated = True
        except (OSError, ValueError):
            pass

    def text(self) -> str:
        return b"".join(self.chunks).decode("utf-8", "replace")


def run_argv(
    argv: Sequence[str],
    *,
    cwd: str,
    timeout_s: float,
    cancel: threading.Event | None = None,
    capture_bytes: int = DEFAULT_CAPTURE_BYTES,
    env: Mapping[str, str] | None = None,
    merge_stderr: bool = True,
) -> RunResult:
    """Run ``argv`` directly (``shell=False``) under a strict deadline (see module doc)."""
    started_at = time.monotonic()
    try:
        tokens = validate_argv(argv)
    except ArgvError as exc:
        return RunResult(tuple(argv), None, "", 0.0, error=str(exc), started=False)
    exe = resolve_executable(tokens[0])
    if exe is None:
        return RunResult(
            tokens, None, "", 0.0, error=f"executable not found: {tokens[0]}", started=False
        )
    if cancel is not None and cancel.is_set():
        return RunResult(tokens, None, "", 0.0, cancelled=True, started=False)
    try:
        proc = _subprocess.Popen(
            [exe, *tokens[1:]],
            cwd=cwd,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
            env=dict(env) if env is not None else child_env(),
            **_popen_kwargs(),
        )
    except OSError as exc:
        return RunResult(tokens, None, "", 0.0, error=f"could not start: {exc}", started=False)
    collector = _Collector(capture_bytes)
    reader = threading.Thread(target=collector.feed, args=(proc.stdout,), daemon=True)
    reader.start()
    errors = _Collector(64 * 1024)
    err_reader: threading.Thread | None = None
    if not merge_stderr:
        err_reader = threading.Thread(target=errors.feed, args=(proc.stderr,), daemon=True)
        err_reader.start()
    deadline = started_at + max(0.1, float(timeout_s))
    timed_out = cancelled = killed = False
    while proc.poll() is None:
        if cancel is not None and cancel.is_set():
            cancelled = True
            break
        if time.monotonic() >= deadline:
            timed_out = True
            break
        time.sleep(_POLL_S)
    if timed_out or cancelled:
        killed = kill_tree(proc)
    try:
        proc.wait(timeout=_KILL_GRACE_S * 2)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass
    reader.join(timeout=_KILL_GRACE_S * 2)
    if err_reader is not None:
        err_reader.join(timeout=_KILL_GRACE_S * 2)
    for stream in (proc.stdout, proc.stderr):
        try:
            if stream is not None:
                stream.close()
        except OSError:
            pass
    return RunResult(
        argv=tokens,
        exit_code=None if (timed_out or cancelled) else proc.returncode,
        output=collector.text(),
        duration_s=time.monotonic() - started_at,
        timed_out=timed_out,
        cancelled=cancelled,
        truncated=collector.truncated,
        total_bytes=collector.total,
        tree_killed=killed,
        meta={"pid": proc.pid},
        stderr=errors.text() if not merge_stderr else "",
    )


__all__ = [
    "ArgvError",
    "DEFAULT_CAPTURE_BYTES",
    "RunResult",
    "child_env",
    "kill_tree",
    "resolve_executable",
    "run_argv",
    "validate_argv",
]
