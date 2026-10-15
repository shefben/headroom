"""IntelligenceManager: setup, status, doctor, stop and the proxy's background activation.

The CLI (``headroom intelligence setup|status|doctor|stop``) and the proxy
share this one orchestrator, so both see the same discovery, model
verification, service ownership and advisor state.

Proxy activation (``start_background``) never blocks startup or traffic: it
runs on a daemon thread, honours ``HEADROOM_JEVK5`` (``off`` → nothing),
``HEADROOM_JEVK5_AUTOSTART`` and ``HEADROOM_OFFLINE`` (verified cache only, no
git/pip/HF), and marks the advisor ready only after conformance passes within
the hot-path deadline. Until then — and forever, if anything fails — the
advisor returns no advice and deterministic behavior continues.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from headroom import paths
from headroom.agent.private_fs import atomic_write_json, read_json, secure_dir
from headroom.intelligence import build as build_mod
from headroom.intelligence import discovery
from headroom.intelligence import model as model_mod
from headroom.intelligence.advisor import DecisionAdvisor
from headroom.intelligence.conformance import (
    ConformanceReport,
    run_llama_conformance,
    run_systemone_conformance,
)
from headroom.intelligence.service import ServiceError, ServiceManager
from headroom.intelligence.settings import (
    JEVK5_RUNTIME_VERSION,
    IntelligenceSettings,
    ServicePolicy,
)
from headroom.log_safety import describe_exception
from headroom.offline import is_offline

log = logging.getLogger(__name__)

LogFn = Callable[[str], None]


@dataclass
class SetupStep:
    name: str
    ok: bool
    detail: str


@dataclass
class SetupReport:
    steps: list[SetupStep] = field(default_factory=list)
    conformance: ConformanceReport | None = None

    @property
    def ok(self) -> bool:
        return bool(self.steps) and all(step.ok for step in self.steps)

    def add(self, name: str, ok: bool, detail: str) -> bool:
        self.steps.append(SetupStep(name, ok, detail))
        return ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "steps": [step.__dict__ for step in self.steps],
            "conformance": self.conformance.to_dict() if self.conformance else None,
        }


class IntelligenceManager:
    def __init__(
        self,
        settings: IntelligenceSettings | None = None,
        *,
        service: ServiceManager | None = None,
    ) -> None:
        self.settings = settings or IntelligenceSettings.from_env()
        self.service = service or ServiceManager(self.settings)
        self._lock = threading.Lock()
        self._state = "disabled" if self.settings.policy is ServicePolicy.OFF else "idle"
        self._state_detail = ""
        self._endpoint: tuple[str, str] | None = None
        self._n_ctx: int | None = None
        self._conformance: ConformanceReport | None = None
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self.advisor = DecisionAdvisor(
            self.settings, self.ready_endpoint, effective_ctx=lambda: self._n_ctx
        )

    # -- state ------------------------------------------------------------------

    @property
    def setup_manifest_path(self) -> Path:
        return paths.intelligence_dir() / "setup.json"

    def ready_endpoint(self) -> tuple[str, str] | None:
        with self._lock:
            return self._endpoint if self._state == "ready" else None

    def _set_state(self, state: str, detail: str = "") -> None:
        with self._lock:
            self._state = state
            self._state_detail = detail

    # -- pieces -------------------------------------------------------------------

    def discover(self) -> tuple[discovery.LlamaBinary | None, list[discovery.LlamaBinary]]:
        return discovery.discover_llama_server(
            self.settings.llama_server_override, self.setup_manifest_path
        )

    def model(self, *, force_hash: bool = False) -> model_mod.ModelStatus:
        return model_mod.model_status(self.settings, force_hash=force_hash)

    def _conform(self, kind: str, url: str, timeout: float) -> ConformanceReport:
        if kind == "systemone":
            return run_systemone_conformance(url, self.settings, timeout=timeout)
        return run_llama_conformance(url, self.settings, timeout=timeout)

    def _accept(self, kind: str, url: str, report: ConformanceReport) -> None:
        with self._lock:
            self._conformance = report
            self._n_ctx = report.details.get("n_ctx") or self.settings.ctx_size
            if report.passed:
                self._endpoint = (kind, url)
                self._state, self._state_detail = "ready", "conformance passed"
            elif all(v for k, v in report.checks.items() if k != "latency_within_deadline"):
                self._endpoint = None
                self._state = "too_slow"
                self._state_detail = "decisions exceed the hot-path deadline; advisor disabled"
            else:
                self._endpoint = None
                failed = sorted(k for k, v in report.checks.items() if not v)
                self._state, self._state_detail = "unavailable", "conformance failed: " + ",".join(
                    failed
                )

    # -- setup (CLI) ------------------------------------------------------------------

    def setup(
        self,
        *,
        log_fn: LogFn = lambda _m: None,
        allow_build: bool = True,
        allow_download: bool = True,
        start_service: bool = True,
    ) -> SetupReport:
        report = SetupReport()
        settings = self.settings
        if settings.policy is ServicePolicy.OFF:
            report.add("policy", False, "HEADROOM_JEVK5=off: setup skipped")
            return report
        if not report.add(
            "python", sys.version_info >= (3, 10), f"Python {sys.version.split()[0]} (>=3.10 required)"
        ):
            return report
        if settings.systemone_url or settings.llama_url:
            kind = "systemone" if settings.systemone_url else "llama-server"
            url = settings.systemone_url or settings.llama_url or ""
            if kind == "llama-server" and not self._ensure_client(report, log_fn):
                return report
            conformance = self._conform(kind, url, settings.load_timeout_s)
            report.conformance = conformance
            self._accept(kind, url, conformance)
            report.add("conformance", conformance.passed, f"{kind} {url}")
            return report
        if not self._ensure_client(report, log_fn):
            return report
        binary, probed = self.discover()
        if binary is None and allow_build:
            if is_offline():
                report.add("llama-server", False, "not found; HEADROOM_OFFLINE forbids building")
                return report
            log_fn("no compatible llama-server found; building a managed one from source")
            try:
                built = build_mod.build_llama_server(settings, log=log_fn)
            except (build_mod.BuildError, OSError) as exc:
                report.add("llama-server", False, f"build failed: {exc}")
                return report
            binary = discovery.probe_binary(str(built.binary), "managed-build")
        if binary is None or not binary.compatible:
            detail = "; ".join(f"{b.path}: {','.join(b.problems)}" for b in probed) or "not found"
            report.add("llama-server", False, detail)
            return report
        report.add("llama-server", True, f"{binary.path} ({binary.source}, {binary.version})")
        status = self.model()
        if not status.verified:
            if status.expected_sha256 is None or settings.model_path_override:
                # Custom/operator files are never fetched or re-fetched by Headroom.
                report.add("model", False, status.detail)
                return report
            if not allow_download or is_offline():
                report.add("model", False, f"{status.detail}; offline/verified-cache-only mode")
                return report
            log_fn(f"downloading {settings.model_file} from {settings.model_repo}")
            try:
                status = model_mod.download_model(
                    settings, progress=_progress_printer(log_fn)
                )
            except model_mod.ModelError as exc:
                report.add("model", False, str(exc))
                return report
        if not report.add("model", status.verified, f"{status.path} ({status.detail})"):
            return report
        self._write_setup_manifest(binary, status)
        if not start_service:
            return report
        try:
            info = self.service.ensure_started(
                binary, Path(status.path), status.sha256 or status.expected_sha256, log=log_fn
            )
        except (ServiceError, OSError) as exc:
            report.add("service", False, str(exc))
            return report
        report.add("service", info.healthy or info.owned, f"{info.url} pid {info.pid}")
        conformance = self._conform("llama-server", str(info.url), settings.load_timeout_s)
        report.conformance = conformance
        self._accept("llama-server", str(info.url), conformance)
        report.add(
            "conformance",
            conformance.passed,
            "passed" if conformance.passed else self._state_detail,
        )
        return report

    def _ensure_client(self, report: SetupReport, log_fn: LogFn) -> bool:
        if discovery.jevk5_client_compatible():
            return report.add("jevk5-client", True, f"jevk5 {JEVK5_RUNTIME_VERSION}")
        log_fn(f"installing JevK5 GGUF client {JEVK5_RUNTIME_VERSION} (--no-deps)")
        ok, detail = discovery.install_jevk5_client()
        return report.add("jevk5-client", ok, detail)

    def _write_setup_manifest(
        self, binary: discovery.LlamaBinary, status: model_mod.ModelStatus
    ) -> None:
        secure_dir(self.setup_manifest_path.parent)
        atomic_write_json(
            self.setup_manifest_path,
            {
                "schema": 1,
                "llama_server": binary.path,
                "llama_server_version": binary.version,
                "llama_server_source": binary.source,
                "model_path": status.path,
                "model_sha256": status.sha256 or status.expected_sha256,
                "jevk5_version": discovery.jevk5_client_version(),
                "completed_at": time.time(),
            },
        )

    # -- status / doctor -----------------------------------------------------------

    def status(self, *, probe: bool = True) -> dict[str, Any]:
        info = self.service.current(probe=probe)
        with self._lock:
            state, detail = self._state, self._state_detail
        manifest = read_json(self.setup_manifest_path)
        return {
            "policy": self.settings.policy.value,
            "autostart": self.settings.autostart,
            "offline": is_offline(),
            "advisor_state": state,
            "advisor_detail": detail,
            "service": info.to_dict(),
            "setup_completed": isinstance(manifest, dict),
            "jevk5_client": discovery.jevk5_client_version(),
            "advisor": self.advisor.status(),
        }

    def doctor(self, *, verify_hash: bool = False) -> dict[str, Any]:
        settings = self.settings
        binary, probed = (None, [])
        if not (settings.llama_url or settings.systemone_url):
            binary, probed = self.discover()
        model = self.model(force_hash=verify_hash)
        info = self.service.current(probe=True)
        conformance = None
        endpoint = self.service.configured_endpoint()
        if endpoint is not None and (info.healthy or info.kind == "systemone"):
            conformance = self._conform(endpoint[0], endpoint[1], settings.load_timeout_s)
            self._accept(endpoint[0], endpoint[1], conformance)
        return {
            "policy": settings.policy.value,
            "python": sys.version.split()[0],
            "jevk5_runtime": discovery.jevk5_client_version(),
            "jevk5_runtime_expected": JEVK5_RUNTIME_VERSION,
            "llama_server": binary.to_dict() if binary else None,
            "llama_server_candidates": [b.to_dict() for b in probed],
            "model": model.to_dict(),
            "endpoint_type": info.kind,
            "endpoint": info.url,
            "effective_context": min(info.n_ctx or settings.ctx_size, settings.ctx_size),
            "configured_context": settings.ctx_size,
            "calibration": {
                "temperature": settings.temperature,
                "knockout_temperature": settings.knockout_temperature,
                "top_k": settings.top_k,
                "missing_retry_n_probs": settings.missing_retry_n_probs,
            },
            "missing_letter_health": (
                conformance.checks.get("answer_letters_present") if conformance else None
            ),
            "process_ownership": {
                "owned": info.owned,
                "pid": info.pid,
                "executable": info.executable,
                "detail": info.detail,
            },
            "conformance": conformance.to_dict() if conformance else None,
            "advisor": self.advisor.status(),
        }

    def stop(self) -> tuple[bool, str]:
        self._set_state("idle" if self.settings.policy is not ServicePolicy.OFF else "disabled")
        with self._lock:
            self._endpoint = None
        return self.service.stop()

    # -- proxy background activation ------------------------------------------------

    def activate(self) -> None:
        """Bring the advisor up without ever raising (runs on the background thread)."""
        settings = self.settings
        if settings.policy is ServicePolicy.OFF:
            self._set_state("disabled", "HEADROOM_JEVK5=off")
            return
        self._set_state("starting")
        try:
            if settings.systemone_url or settings.llama_url:
                kind = "systemone" if settings.systemone_url else "llama-server"
                url = settings.systemone_url or settings.llama_url or ""
                if kind == "llama-server" and not discovery.jevk5_client_compatible():
                    self._set_state("unavailable", "jevk5 client not installed")
                    return
                self._accept(kind, url, self._conform(kind, url, settings.load_timeout_s))
                return
            if not settings.autostart:
                endpoint = self.service.configured_endpoint()
                if endpoint is None:
                    self._set_state("unavailable", "autostart disabled and no service running")
                    return
                self._accept(endpoint[0], endpoint[1],
                             self._conform(endpoint[0], endpoint[1], settings.load_timeout_s))
                return
            binary, _ = self.discover()
            status = self.model()
            ready = binary is not None and status.verified and discovery.jevk5_client_compatible()
            if not ready:
                if is_offline():
                    self._set_state(
                        "unavailable", "offline: needs an installed llama-server and verified model"
                    )
                    return
                if self._stopping.is_set():
                    return
                report = self.setup(log_fn=lambda m: log.info("jevk5 setup: %s", m))
                if not report.ok:
                    failed = [s for s in report.steps if not s.ok]
                    detail = f"{failed[0].name}: {failed[0].detail}" if failed else "setup failed"
                    self._set_state("unavailable", detail)
                    if settings.policy is ServicePolicy.ON:
                        log.warning("JevK5 advisor setup failed (HEADROOM_JEVK5=on): %s", detail)
                return
            if self._stopping.is_set():
                return
            info = self.service.ensure_started(
                binary, Path(status.path), status.sha256 or status.expected_sha256
            )
            self._accept(
                "llama-server", str(info.url),
                self._conform("llama-server", str(info.url), settings.load_timeout_s),
            )
        except Exception as exc:  # noqa: BLE001 - background activation must fail open
            self._set_state("unavailable", describe_exception(exc))
            if settings.policy is ServicePolicy.ON:
                log.warning("JevK5 advisor unavailable (HEADROOM_JEVK5=on): %s",
                            describe_exception(exc))

    def start_background(self) -> None:
        if self.settings.policy is ServicePolicy.OFF or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self.activate, name="headroom-jevk5-activate", daemon=True
        )
        self._thread.start()

    def shutdown(self, timeout: float = 2.0) -> None:
        """Stop background work. The shared llama-server keeps running (``stop`` ends it)."""
        self._stopping.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        self.advisor.shutdown()

    def health(self) -> dict[str, Any]:
        """Cheap, non-probing status for /health."""
        with self._lock:
            state, detail, endpoint = self._state, self._state_detail, self._endpoint
        return {
            "policy": self.settings.policy.value,
            "advisor_state": state,
            "advisor_detail": detail,
            "endpoint_type": endpoint[0] if endpoint else None,
            "breaker": self.advisor.status()["breaker"],
        }


def _progress_printer(log_fn: LogFn) -> Callable[[int, int | None], None]:
    last = [0.0]

    def report(done: int, total: int | None) -> None:
        now = time.monotonic()
        if now - last[0] < 5.0 and (total is None or done < total):
            return
        last[0] = now
        if total:
            log_fn(f"  {done / 2**30:.2f} / {total / 2**30:.2f} GiB ({100 * done / total:.0f}%)")
        else:
            log_fn(f"  {done / 2**30:.2f} GiB")

    return report
