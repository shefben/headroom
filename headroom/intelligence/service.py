"""Managed ``llama-server`` lifecycle: start, reuse, stop — only what Headroom owns.

* A managed server binds ``127.0.0.1`` on a free port (prefers 8091, falls
  back to an OS-assigned port, retries on a bind race) with
  ``--ctx-size 8192 --parallel 1`` so the whole context is available to one
  decision, and ``--n-gpu-layers auto`` when the binary supports it.
* Ownership is recorded in an owner-only manifest (PID, executable, start
  time, port, model path + hash). Reuse and stop both re-validate PID +
  executable + start time (+ command line where the OS exposes it); a process
  that fails validation is never signalled, and another application occupying
  a port is never touched.
* ``HEADROOM_JEVK5_LLAMA_URL`` / ``HEADROOM_JEVK5_SYSTEMONE_URL`` endpoints are
  operator-managed: never started or stopped here.
* Windows: no console window, separate process group. POSIX: its own session
  (process group) so a stop terminates the server and only the server.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from headroom import fileperms
from headroom.agent.private_fs import atomic_write_json, read_json, secure_dir
from headroom.intelligence.discovery import LlamaBinary
from headroom.intelligence.process import is_running, matches_owned, process_info, terminate
from headroom.intelligence.settings import IntelligenceSettings
from headroom.intelligence.transport import LlamaProbe, llama_health, probe_llama_server

MANIFEST_SCHEMA = 1
LogFn = Callable[[str], None]


class ServiceError(RuntimeError):
    pass


class BinaryLoadError(ServiceError):
    """This llama-server binary could not serve the model (load, health or probe).

    Specific to the binary: the manager may try the next discovered candidate
    or a managed build. Port/lock problems are plain :class:`ServiceError`.
    """


@dataclass(frozen=True)
class ServiceInfo:
    kind: str  # "managed" | "operator-llama" | "systemone" | "none"
    url: str | None
    owned: bool
    healthy: bool
    detail: str
    pid: int | None = None
    port: int | None = None
    executable: str | None = None
    model_path: str | None = None
    model_sha256: str | None = None
    started_at: float | None = None
    n_ctx: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _intelligence_dir() -> Path:
    from headroom import paths

    return paths.intelligence_dir()


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def pick_port(host: str, preferred: int) -> int:
    if port_is_free(host, preferred):
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


class ServiceManager:
    def __init__(
        self,
        settings: IntelligenceSettings,
        *,
        spawn: Callable[..., Any] = subprocess.Popen,
        probe: Callable[..., LlamaProbe] = probe_llama_server,
        health: Callable[..., tuple[bool, str]] = llama_health,
    ) -> None:
        self.settings = settings
        self._spawn = spawn
        self._probe = probe
        self._health = health

    # -- paths -------------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return _intelligence_dir() / "service.json"

    @property
    def lock_path(self) -> Path:
        return _intelligence_dir() / "service.lock"

    def log_path(self, port: int) -> Path:
        return _intelligence_dir() / "logs" / f"llama-server-{port}.log"

    # -- manifest ------------------------------------------------------------

    def read_manifest(self) -> dict[str, Any] | None:
        data = read_json(self.manifest_path)
        if not isinstance(data, dict) or data.get("schema") != MANIFEST_SCHEMA:
            return None
        if data.get("owner") != "headroom":
            return None
        return data

    def _owned_process_ok(self, manifest: dict[str, Any]) -> tuple[bool, str]:
        try:
            pid = int(manifest["pid"])
        except (KeyError, TypeError, ValueError):
            return False, "manifest has no pid"
        return matches_owned(
            process_info(pid),
            executable=str(manifest.get("executable") or ""),
            create_time=manifest.get("create_time"),
            model_path=manifest.get("model_path"),
            port=_int_or_none(manifest.get("port")),
        )

    # -- status ----------------------------------------------------------------

    def configured_endpoint(self) -> tuple[str, str] | None:
        """The endpoint the advisor should use, cheaply (no HTTP)."""
        if self.settings.systemone_url:
            return ("systemone", self.settings.systemone_url)
        if self.settings.llama_url:
            return ("llama-server", self.settings.llama_url)
        manifest = self.read_manifest()
        if manifest and is_running(int(manifest.get("pid") or 0)):
            return ("llama-server", str(manifest["url"]))
        return None

    def current(self, *, probe: bool = True) -> ServiceInfo:
        settings = self.settings
        if settings.systemone_url:
            return ServiceInfo("systemone", settings.systemone_url, False, False,
                               "operator-managed /v1/systemone (probe via doctor)")
        if settings.llama_url:
            if not probe:
                return ServiceInfo("operator-llama", settings.llama_url, False, False, "not probed")
            result = self._probe(settings.llama_url)
            return ServiceInfo("operator-llama", settings.llama_url, False, result.ok,
                               result.detail, n_ctx=result.n_ctx)
        manifest = self.read_manifest()
        if manifest is None:
            return ServiceInfo("none", None, False, False, "no managed llama-server")
        owned, reason = self._owned_process_ok(manifest)
        base = dict(
            url=manifest.get("url"),
            pid=manifest.get("pid"),
            port=manifest.get("port"),
            executable=manifest.get("executable"),
            model_path=manifest.get("model_path"),
            model_sha256=manifest.get("model_sha256"),
            started_at=manifest.get("started_at"),
            n_ctx=manifest.get("n_ctx"),
        )
        if not owned:
            return ServiceInfo("managed", owned=False, healthy=False, detail=reason, **base)
        healthy, detail = (self._health(str(manifest["url"])) if probe else (True, "not probed"))
        return ServiceInfo("managed", owned=True, healthy=healthy, detail=detail, **base)

    # -- start -------------------------------------------------------------------

    @contextmanager
    def _start_lock(self, timeout: float) -> Iterator[None]:
        secure_dir(self.lock_path.parent)
        deadline = time.monotonic() + timeout
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                break
            except FileExistsError:
                try:
                    holder = int(self.lock_path.read_text(encoding="utf-8").strip() or 0)
                except (OSError, ValueError):
                    holder = 0
                if not is_running(holder):
                    self.lock_path.unlink(missing_ok=True)
                    continue
                if time.monotonic() > deadline:
                    raise ServiceError("another Headroom process is starting llama-server") from None
                time.sleep(0.25)
        try:
            yield
        finally:
            self.lock_path.unlink(missing_ok=True)

    def ensure_started(
        self,
        binary: LlamaBinary,
        model_path: Path,
        model_sha256: str | None,
        *,
        log: LogFn = lambda _m: None,
    ) -> ServiceInfo:
        """Reuse a healthy owned server for this model or start a new one."""
        if self.settings.llama_url or self.settings.systemone_url:
            raise ServiceError("an operator-managed endpoint is configured; not starting one")
        with self._start_lock(self.settings.load_timeout_s):
            existing = self.current(probe=True)
            if (
                existing.kind == "managed"
                and existing.owned
                and existing.healthy
                and existing.model_sha256 == model_sha256
                and os.path.normcase(str(existing.model_path)) == os.path.normcase(str(model_path))
            ):
                log(f"reusing managed llama-server pid {existing.pid} on {existing.url}")
                return existing
            if existing.kind == "managed" and existing.owned:
                log("stopping stale managed llama-server (different model or unhealthy)")
                self._terminate_owned()
            elif existing.kind == "managed":
                self.manifest_path.unlink(missing_ok=True)  # stale record; never signal
            last: ServiceError = ServiceError("llama-server did not start")
            for _attempt in range(3):
                port = pick_port(self.settings.host, self.settings.preferred_port)
                try:
                    return self._spawn_and_wait(binary, model_path, model_sha256, port, log)
                except ServiceError as exc:
                    last = exc
                    text = str(exc).lower()
                    if "port" not in text and "bind" not in text:
                        break
                    # Lost a bind race for that port: retry on a fresh one.
            raise last

    def _command(self, binary: LlamaBinary, model_path: Path, port: int) -> list[str]:
        settings = self.settings
        if binary.supports_gpu_layers_auto or settings.gpu_layers != "auto":
            ngl = settings.gpu_layers
        else:
            has_gpu = any(not d.upper().startswith("CPU") for d in binary.devices)
            ngl = "999" if has_gpu else "0"
        return [
            binary.path,
            "-m",
            str(model_path),
            "--host",
            settings.host,
            "--port",
            str(port),
            "--ctx-size",
            str(settings.ctx_size),
            "--parallel",
            "1",
            "--n-gpu-layers",
            ngl,
        ]

    def _spawn_and_wait(
        self,
        binary: LlamaBinary,
        model_path: Path,
        model_sha256: str | None,
        port: int,
        log: LogFn,
    ) -> ServiceInfo:
        command = self._command(binary, model_path, port)
        log_file = self.log_path(port)
        secure_dir(log_file.parent)
        handle = fileperms.open_owner_only(log_file, "w", encoding="utf-8", errors="replace")
        kwargs: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": handle,
            "stderr": subprocess.STDOUT,
            "close_fds": True,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
        else:
            kwargs["start_new_session"] = True
        log("starting " + " ".join(command))
        try:
            proc = self._spawn(command, **kwargs)
        except OSError as exc:
            handle.close()
            raise BinaryLoadError(
                f"could not launch llama-server: {type(exc).__name__}"
            ) from exc
        handle.close()
        url = f"http://{self.settings.host}:{port}"
        deadline = time.monotonic() + self.settings.load_timeout_s
        info = None
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                tail = self._log_tail(port)
                error_type = ServiceError if "bind" in tail.lower() else BinaryLoadError
                raise error_type(
                    f"llama-server exited with {proc.returncode} during load: {tail}"
                )
            if info is None:
                info = process_info(proc.pid)
            healthy, _detail = self._health(url, timeout=2.0)
            if healthy:
                break
            time.sleep(0.5)
        else:
            terminate(proc.pid)
            raise BinaryLoadError("llama-server did not become healthy before the load timeout")
        probe = self._probe(url, timeout=30.0)
        if not probe.ok:
            terminate(proc.pid)
            raise BinaryLoadError(f"llama-server failed capability probe: {probe.detail}")
        if probe.n_ctx is not None and probe.n_ctx < self.settings.ctx_size:
            terminate(proc.pid)
            raise BinaryLoadError(
                f"llama-server runs with n_ctx={probe.n_ctx}, below the required "
                f"{self.settings.ctx_size}"
            )
        info = info or process_info(proc.pid)
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "owner": "headroom",
            "owner_pid": os.getpid(),
            "pid": proc.pid,
            "executable": (info.executable if info and info.executable else binary.path),
            "create_time": info.create_time if info else None,
            "command": command,
            "port": port,
            "url": url,
            "model_path": str(model_path),
            "model_sha256": model_sha256,
            "n_ctx": probe.n_ctx or self.settings.ctx_size,
            "started_at": time.time(),
            "llama_server_version": binary.version,
        }
        atomic_write_json(self.manifest_path, manifest)
        log(f"llama-server ready on {url} (pid {proc.pid})")
        return self.current(probe=False)

    def _log_tail(self, port: int) -> str:
        try:
            lines = self.log_path(port).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        return " | ".join(line.strip() for line in lines[-3:])[:400]

    # -- stop -----------------------------------------------------------------

    def _terminate_owned(self) -> tuple[bool, str]:
        manifest = self.read_manifest()
        if manifest is None:
            return False, "no managed llama-server recorded"
        owned, reason = self._owned_process_ok(manifest)
        pid = int(manifest.get("pid") or 0)
        if not owned:
            if not is_running(pid):
                self.manifest_path.unlink(missing_ok=True)
                return False, "recorded llama-server is not running (stale record removed)"
            # Alive but not provably ours: never signal it. Drop the stale record.
            self.manifest_path.unlink(missing_ok=True)
            return False, f"refusing to stop pid {pid}: {reason}"
        stopped = terminate(pid)
        if stopped:
            self.manifest_path.unlink(missing_ok=True)
            return True, f"stopped managed llama-server pid {pid}"
        return False, f"pid {pid} did not exit"

    def stop(self) -> tuple[bool, str]:
        if self.settings.llama_url or self.settings.systemone_url:
            return False, "operator-managed endpoint configured; Headroom does not stop it"
        return self._terminate_owned()
