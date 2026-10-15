"""Model cache verification, managed service ownership, manager flows and the CLI."""

from __future__ import annotations

import hashlib
import io
import json
import os
import socket
import struct
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from headroom.intelligence import discovery
from headroom.intelligence import model as model_mod
from headroom.intelligence.discovery import LlamaBinary
from headroom.intelligence.manager import IntelligenceManager
from headroom.intelligence.service import ServiceManager, pick_port
from headroom.intelligence.settings import IntelligenceSettings, ServicePolicy

from .fake_llama import FakeLlamaServer

FAKE_SCRIPT = str(Path(__file__).with_name("fake_llama.py"))


@pytest.fixture(autouse=True)
def _workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "ws"))
    for name in ("HEADROOM_OFFLINE", "HEADROOM_JEVK5_LLAMA_URL", "HEADROOM_JEVK5_SYSTEMONE_URL"):
        monkeypatch.delenv(name, raising=False)


def _gguf(path: Path, arch: str = "qwen35") -> Path:
    def s(text: str) -> bytes:
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    kvs = [
        s("general.architecture") + struct.pack("<I", 8) + s(arch),
        s("general.name") + struct.pack("<I", 8) + s("test"),
        s(f"{arch}.context_length") + struct.pack("<I", 4) + struct.pack("<I", 262144),
        s("tokenizer.ggml.scores") + struct.pack("<I", 9) + struct.pack("<IQ", 6, 3)
        + struct.pack("<3f", 0.1, 0.2, 0.3),
        s("tokenizer.ggml.tokens") + struct.pack("<I", 9) + struct.pack("<IQ", 8, 2) + s("a") + s("b"),
    ]
    body = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 1) + struct.pack("<Q", len(kvs))
    path.write_bytes(body + b"".join(kvs) + b"\x00" * 64)
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# -- model ---------------------------------------------------------------------


def test_gguf_metadata_and_checksum_verification(tmp_path, monkeypatch):
    path = _gguf(tmp_path / "m.gguf")
    info = model_mod.read_gguf_info(path)
    assert info.architecture == "qwen35" and info.context_length == 262144
    settings = IntelligenceSettings(model_path_override=str(path), model_sha256=_sha(path))
    status = model_mod.model_status(settings)
    assert status.verified and status.sha256 == _sha(path)
    # Second call trusts the size+mtime sidecar instead of rehashing.
    monkeypatch.setattr(model_mod, "sha256_file", lambda *a, **k: pytest.fail("rehashed"))
    assert model_mod.model_status(settings).verified


def test_checksum_mismatch_and_custom_model_rules(tmp_path):
    path = _gguf(tmp_path / "m.gguf")
    wrong = IntelligenceSettings(model_path_override=str(path), model_sha256="0" * 64)
    assert model_mod.model_status(wrong).detail == "SHA-256 mismatch"
    no_hash = IntelligenceSettings(model_path_override=str(path), model_sha256=None)
    assert not model_mod.model_status(no_hash).verified
    operator = IntelligenceSettings(model_path_override=str(path), model_sha256=None,
                                    model_operator_managed=True)
    assert model_mod.model_status(operator).verified
    bad_arch = _gguf(tmp_path / "llama.gguf", arch="llama")
    other = IntelligenceSettings(model_path_override=str(bad_arch), model_sha256=_sha(bad_arch))
    assert "unsupported architecture" in model_mod.model_status(other).detail
    junk = tmp_path / "junk.gguf"
    junk.write_bytes(b"NOPE" + b"\x00" * 32)
    assert not model_mod.model_status(
        IntelligenceSettings(model_path_override=str(junk), model_sha256="0" * 64)
    ).verified


def test_download_refused_offline_and_for_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(model_mod.urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("network used"))
    monkeypatch.setenv("HEADROOM_OFFLINE", "1")
    with pytest.raises(model_mod.ModelError, match="OFFLINE"):
        model_mod.download_model(IntelligenceSettings())
    monkeypatch.delenv("HEADROOM_OFFLINE")
    with pytest.raises(model_mod.ModelError, match="operator-managed"):
        model_mod.download_model(IntelligenceSettings(model_path_override=str(tmp_path / "x")))


def test_resumable_download_verifies_then_renames(tmp_path, monkeypatch):
    payload = _gguf(tmp_path / "src.gguf").read_bytes()
    settings = IntelligenceSettings(model_file="custom.gguf", model_sha256=hashlib.sha256(
        payload).hexdigest())
    part = settings.model_path.with_name("custom.gguf.part")
    part.parent.mkdir(parents=True)
    part.write_bytes(payload[:10])  # an interrupted earlier download
    seen = {}

    class Response(io.BytesIO):
        status = 206

        def __init__(self, data):
            super().__init__(data)
            self.headers = {"Content-Length": str(len(data))}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout):
        seen["range"] = request.headers.get("Range")
        return Response(payload[10:])

    monkeypatch.setattr(model_mod.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(model_mod.shutil, "disk_usage",
                        lambda p: type("U", (), {"free": 10**12})())
    status = model_mod.download_model(settings)
    assert seen["range"] == "bytes=10-"
    assert status.verified and settings.model_path.read_bytes() == payload
    assert not part.exists()


# -- service --------------------------------------------------------------------


def _spawn_fake(command, **kwargs):
    return subprocess.Popen([sys.executable, FAKE_SCRIPT, *command[1:]], **kwargs)


def _binary() -> LlamaBinary:
    return LlamaBinary(path=sys.executable, source="test", version="fake",
                       supports_gpu_layers_auto=True, supports_hf=True, supports_ctx_size=True)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    assert pick_port("127.0.0.1", port) == port
    return port


def test_managed_service_start_reuse_stop(tmp_path):
    settings = IntelligenceSettings(preferred_port=_free_port(), load_timeout_s=30)
    manager = ServiceManager(settings, spawn=_spawn_fake)
    model = _gguf(tmp_path / "m.gguf")
    info = manager.ensure_started(_binary(), model, "abc")
    try:
        assert info.owned and info.kind == "managed" and info.port == settings.preferred_port
        manifest = json.loads(manager.manifest_path.read_text())
        assert manifest["owner"] == "headroom" and manifest["model_sha256"] == "abc"
        assert "--ctx-size" in manifest["command"] and "8192" in manifest["command"]
        assert manager.current().healthy
        again = manager.ensure_started(_binary(), model, "abc")
        assert again.pid == info.pid  # reused, not restarted
    finally:
        stopped, detail = manager.stop()
    assert stopped, detail
    assert not manager.manifest_path.exists()
    from headroom.intelligence.process import is_running

    assert not is_running(info.pid)


def test_occupied_port_is_never_touched(tmp_path):
    occupant = socket.socket()
    occupant.bind(("127.0.0.1", 0))
    occupant.listen(1)
    busy = occupant.getsockname()[1]
    try:
        settings = IntelligenceSettings(preferred_port=busy, load_timeout_s=30)
        manager = ServiceManager(settings, spawn=_spawn_fake)
        info = manager.ensure_started(_binary(), _gguf(tmp_path / "m.gguf"), "abc")
        assert info.port != busy
        manager.stop()
        occupant.getsockname()  # still ours, still open
    finally:
        occupant.close()


def test_foreign_process_is_never_stopped(tmp_path):
    settings = IntelligenceSettings()
    manager = ServiceManager(settings)
    manager.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manager.manifest_path.write_text(json.dumps({
        "schema": 1, "owner": "headroom", "pid": os.getpid(),
        "executable": str(tmp_path / "llama-server.exe"), "create_time": 1.0,
        "url": "http://127.0.0.1:1", "model_path": "m",
    }))
    stopped, detail = manager.stop()
    assert not stopped and "refusing" in detail
    assert not manager.manifest_path.exists()  # stale record dropped, process untouched


def test_operator_endpoints_are_never_managed():
    settings = IntelligenceSettings(llama_url="http://127.0.0.1:1")
    manager = ServiceManager(settings)
    stopped, detail = manager.stop()
    assert not stopped and "operator" in detail
    from headroom.intelligence.service import ServiceError

    with pytest.raises(ServiceError):
        manager.ensure_started(_binary(), Path("m"), None)


# -- discovery ----------------------------------------------------------------


def test_probe_binary_parses_flags(monkeypatch):
    outputs = {
        "--version": (0, "version: 3 (44da65e)\nbuilt with MSVC"),
        "--help": (0, "-c, --ctx-size N\n-ngl, --n-gpu-layers N  'auto', or 'all'\n"
                      "-hf, --hf-repo\n--list-devices"),
        "--list-devices": (0, "Available devices:\n  CUDA0: NVIDIA RTX (16000 MiB)\n"),
    }
    monkeypatch.setattr(discovery, "_run", lambda args, timeout=30: outputs[args[1]])
    info = discovery.probe_binary("llama-server", "path")
    assert info.compatible and info.supports_gpu_layers_auto
    assert info.version == "3 (44da65e)" and info.devices == ("CUDA0: NVIDIA RTX (16000 MiB)",)


def test_override_is_authoritative(tmp_path, monkeypatch):
    fake = tmp_path / ("llama-server.exe" if sys.platform == "win32" else "llama-server")
    fake.write_text("x")
    monkeypatch.setattr(discovery, "probe_binary", lambda p, s: LlamaBinary(
        p, s, None, False, False, False, problems=("no --ctx-size flag",)))
    found, probed = discovery.discover_llama_server(str(fake), None)
    assert found is None and len(probed) == 1 and probed[0].source == "override"


def test_client_install_refused_offline(monkeypatch):
    monkeypatch.setenv("HEADROOM_OFFLINE", "1")
    monkeypatch.setattr(discovery, "_run", lambda *a, **k: pytest.fail("pip ran offline"))
    ok, detail = discovery.install_jevk5_client()
    assert not ok and "OFFLINE" in detail


# -- manager + CLI ----------------------------------------------------------------


def test_manager_policy_off_and_offline_fail_open(monkeypatch):
    off = IntelligenceManager(IntelligenceSettings(policy=ServicePolicy.OFF))
    assert not off.setup().ok
    off.activate()
    assert off.health()["advisor_state"] == "disabled"
    monkeypatch.setenv("HEADROOM_OFFLINE", "1")
    monkeypatch.setattr(IntelligenceManager, "discover", lambda self: (None, []))
    manager = IntelligenceManager(IntelligenceSettings())
    manager.activate()
    assert manager.health()["advisor_state"] == "unavailable"
    budget = manager.advisor.request_budget()
    assert manager.advisor.noul("f", "s", "q?", budget=budget) is None


def test_manager_operator_endpoint_ready_and_advises():
    with FakeLlamaServer() as fake:
        manager = IntelligenceManager(IntelligenceSettings(llama_url=fake.url))
        report = manager.setup()
        assert report.ok, report.to_dict()
        assert manager.health()["advisor_state"] == "ready"
        advice = manager.advisor.choose(
            "parcel", "No. 17 vs No. 71", "What happened?",
            ["delivered", "misdelivered", "unknown"], budget=manager.advisor.request_budget(),
        )
        assert advice is not None and advice.value == "misdelivered"
        manager.shutdown()


def test_manager_too_slow_endpoint_disables_hot_path():
    with FakeLlamaServer(delay_s=0.3) as fake:
        manager = IntelligenceManager(
            IntelligenceSettings(llama_url=fake.url, decision_timeout_ms=200)
        )
        manager.activate()
        assert manager.health()["advisor_state"] == "too_slow"
        assert manager.ready_endpoint() is None


def test_cli_status_doctor_stop_with_operator_endpoint(monkeypatch):
    from headroom.cli.main import main

    with FakeLlamaServer() as fake:
        monkeypatch.setenv("HEADROOM_JEVK5_LLAMA_URL", fake.url)
        runner = CliRunner()
        status = runner.invoke(main, ["intelligence", "status", "--json"])
        assert status.exit_code == 0, status.output
        payload = json.loads(status.output)
        assert payload["service"]["kind"] == "operator-llama" and payload["service"]["healthy"]
        doctor = runner.invoke(main, ["intelligence", "doctor", "--json"])
        assert doctor.exit_code == 0, doctor.output
        data = json.loads(doctor.output)
        assert data["endpoint_type"] == "operator-llama"
        assert data["effective_context"] == 8192
        assert data["conformance"]["passed"] and data["missing_letter_health"] is True
        assert data["calibration"]["temperature"] == 1.22
        stop = runner.invoke(main, ["intelligence", "stop"])
        assert stop.exit_code == 0 and "operator" in stop.output
