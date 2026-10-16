"""Cross-platform process identity for owned-service validation.

Before Headroom stops a process it must prove the PID still belongs to the
llama-server it started: same executable, same start time, and a command line
naming the same model. PIDs are reused, ports are shared — neither alone is
proof. Uses ``psutil`` when installed, else ``/proc`` on Linux and the Win32
API (``QueryFullProcessImageNameW`` / ``GetProcessTimes``) on Windows.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

from headroom._subprocess import pid_alive

_STILL_ACTIVE = 259


def _windows_running(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
    if not handle:
        return bool(kernel32.GetLastError() == 5)  # access denied => exists
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return code.value == _STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _posix_running(pid: int) -> bool:
    """POSIX liveness that does not count a zombie as running.

    A terminated child of this process stays a zombie (and ``kill(pid, 0)``
    still succeeds) until it is reaped, so a service Headroom started and
    stopped in the same process would look alive forever. Reap it when it is
    our child; otherwise read its state from ``/proc`` where available.
    """
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)
        if reaped == pid:
            return False
    except ChildProcessError:
        pass  # not our child: nothing to reap here
    except OSError:
        pass
    if not pid_alive(pid):
        return False
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii", errors="replace") as handle:
            state = handle.read().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return True
    return state not in ("Z", "X")


def is_running(pid: int) -> bool:
    """True while ``pid`` is a running process.

    On Windows an exited process stays openable while any handle to it is
    open, so liveness is decided by its exit code, not by OpenProcess alone.
    On POSIX an exited but unreaped child (zombie) is not running either.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            return _windows_running(pid)
        except (OSError, AttributeError, ValueError):
            return pid_alive(pid)
    return _posix_running(pid)


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    executable: str | None
    cmdline: tuple[str, ...] | None
    create_time: float | None


def _norm(path: str | None) -> str | None:
    if not path:
        return None
    return os.path.normcase(os.path.realpath(path))


def _windows_info(pid: int) -> ProcessInfo | None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        size = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        exe = None
        if kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            exe = buffer.value
        creation, exit_t, kernel_t, user_t = (wintypes.FILETIME() for _ in range(4))
        created = None
        if kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_t),
            ctypes.byref(kernel_t),
            ctypes.byref(user_t),
        ):
            ticks = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
            created = ticks / 1e7 - 11644473600.0
        return ProcessInfo(pid, exe, None, created)
    finally:
        kernel32.CloseHandle(handle)


def _linux_info(pid: int) -> ProcessInfo | None:
    try:
        exe = os.readlink(f"/proc/{pid}/exe")
    except OSError:
        exe = None
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            raw = handle.read()
        cmdline = tuple(part.decode("utf-8", "replace") for part in raw.split(b"\0") if part)
    except OSError:
        cmdline = None
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            fields = handle.read().rpartition(b")")[2].split()
        ticks = os.sysconf("SC_CLK_TCK")
        with open("/proc/stat", "rb") as handle:
            btime = next(float(line.split()[1]) for line in handle if line.startswith(b"btime"))
        created = btime + float(fields[19]) / ticks
    except (OSError, IndexError, ValueError, StopIteration):
        created = None
    if exe is None and cmdline is None:
        return None
    return ProcessInfo(pid, exe, cmdline, created)


def process_info(pid: int) -> ProcessInfo | None:
    if not is_running(pid):
        return None
    try:
        import psutil  # type: ignore[import-untyped]

        proc = psutil.Process(pid)
        return ProcessInfo(pid, proc.exe(), tuple(proc.cmdline()), proc.create_time())
    except Exception:  # noqa: BLE001 - psutil optional / access denied
        pass
    if sys.platform == "win32":
        try:
            return _windows_info(pid)
        except (OSError, AttributeError, ValueError):
            return None
    if sys.platform.startswith("linux"):
        return _linux_info(pid)
    return ProcessInfo(pid, None, None, None)


def matches_owned(
    info: ProcessInfo | None,
    *,
    executable: str,
    create_time: float | None,
    model_path: str | None,
) -> tuple[bool, str]:
    """Whether ``info`` is provably the owned process. Returns (ok, reason)."""
    if info is None:
        return False, "process not running"
    if info.executable is None:
        return False, "process executable unknown; refusing to treat as owned"
    if _norm(info.executable) != _norm(executable):
        return False, "executable differs from the recorded owned process"
    if create_time is None or info.create_time is None:
        return False, "process start time unavailable; refusing to treat as owned"
    if abs(info.create_time - create_time) > 2.0:
        return False, "start time differs (PID reused)"
    if info.cmdline is not None and model_path:
        joined = " ".join(info.cmdline)
        if os.path.normcase(model_path) not in os.path.normcase(joined):
            return False, "command line does not reference the recorded model"
    return True, "owned"


def terminate(pid: int, *, timeout: float = 10.0) -> bool:
    """Terminate an owned process (and on POSIX its process group)."""
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                timeout=timeout,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (OSError, AttributeError):
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_running(pid):
            return True
        time.sleep(0.1)
    if sys.platform != "win32":
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (OSError, AttributeError):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(0.2)
    return not is_running(pid)
