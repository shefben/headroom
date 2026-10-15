"""Discover and probe a user-installed ``llama-server``; check the JevK5 client.

Search order: ``HEADROOM_JEVK5_LLAMA_SERVER`` override → the binary recorded
by a previous managed setup → ``PATH`` → common build directories under
``LLAMA_CPP_HOME``. Each candidate is probed (``--version``, ``--help``,
``--list-devices``) without modifying it; the first one that exposes the flags
Headroom needs is reused. Model loading itself (Qwen3.5 hybrid layers) is
proven later by starting the server and passing conformance.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from headroom.agent.private_fs import read_json
from headroom.intelligence.settings import JEVK5_PIP_SPEC, JEVK5_RUNTIME_VERSION
from headroom.offline import is_offline

_EXE = "llama-server.exe" if sys.platform == "win32" else "llama-server"
_PROBE_TIMEOUT_S = 30.0
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass(frozen=True)
class LlamaBinary:
    path: str
    source: str
    version: str | None
    supports_gpu_layers_auto: bool
    supports_hf: bool
    supports_ctx_size: bool
    devices: tuple[str, ...] = ()
    problems: tuple[str, ...] = field(default_factory=tuple)

    @property
    def compatible(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "source": self.source,
            "version": self.version,
            "gpu_layers_auto": self.supports_gpu_layers_auto,
            "devices": list(self.devices),
            "compatible": self.compatible,
            "problems": list(self.problems),
        }


def _run(args: list[str], timeout: float = _PROBE_TIMEOUT_S) -> tuple[int, str]:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=_NO_WINDOW,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return -1, f"{type(exc).__name__}"
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def probe_binary(path: str, source: str) -> LlamaBinary:
    problems: list[str] = []
    code, version_out = _run([path, "--version"])
    version = None
    match = re.search(r"version:\s*([^\n]+)", version_out)
    if match:
        version = match.group(1).strip()
    elif code != 0:
        problems.append("--version failed")
    code, help_out = _run([path, "--help"])
    if code != 0 and not help_out:
        problems.append("--help failed")
    supports_ctx = "--ctx-size" in help_out
    supports_ngl = "--n-gpu-layers" in help_out or "--gpu-layers" in help_out
    supports_auto = supports_ngl and "'auto'" in help_out
    supports_hf = "--hf-repo" in help_out
    if not supports_ctx:
        problems.append("no --ctx-size flag")
    if not supports_ngl:
        problems.append("no --n-gpu-layers flag")
    devices: list[str] = []
    if "--list-devices" in help_out:
        _, devices_out = _run([path, "--list-devices"])
        for line in devices_out.splitlines():
            line = line.strip()
            if re.match(r"^[A-Za-z]+\d*:", line) and "Available" not in line:
                devices.append(line[:120])
    return LlamaBinary(
        path=path,
        source=source,
        version=version,
        supports_gpu_layers_auto=supports_auto,
        supports_hf=supports_hf,
        supports_ctx_size=supports_ctx,
        devices=tuple(devices),
        problems=tuple(problems),
    )


def candidate_binaries(
    override: str | None, manifest_path: Path | None
) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out: list[tuple[str, str]] = []

    def add(path: str | None, source: str) -> None:
        if not path:
            return
        candidate = Path(path).expanduser()
        if candidate.is_dir():
            candidate = candidate / _EXE
        if not candidate.is_file():
            return
        key = os.path.normcase(str(candidate.resolve()))
        if key not in seen:
            seen.add(key)
            out.append((str(candidate), source))

    add(override, "override")
    if manifest_path is not None:
        manifest = read_json(manifest_path)
        if isinstance(manifest, dict):
            add(manifest.get("llama_server"), "managed-manifest")
    add(shutil.which("llama-server"), "path")
    home = os.environ.get("LLAMA_CPP_HOME")
    if home:
        base = Path(home).expanduser()
        for rel in ("", "bin", "build/bin", "build/bin/Release", "build/bin/RelWithDebInfo",
                    "build-cuda/bin", "build-cuda/bin/Release", "build/Release"):
            add(str(base / rel / _EXE), "LLAMA_CPP_HOME")
    return out


def discover_llama_server(
    override: str | None, manifest_path: Path | None
) -> tuple[LlamaBinary | None, list[LlamaBinary]]:
    """Return (first compatible binary, every probed candidate)."""
    probed: list[LlamaBinary] = []
    for path, source in candidate_binaries(override, manifest_path):
        info = probe_binary(path, source)
        probed.append(info)
        if info.compatible:
            return info, probed
        if source == "override":
            break  # an explicit override is authoritative: do not silently swap it
    return None, probed


# -- JevK5 GGUF client -------------------------------------------------------


def jevk5_client_version() -> str | None:
    try:
        import importlib.metadata

        return importlib.metadata.version("jevk5")
    except Exception:  # noqa: BLE001 - not installed / broken metadata
        return None


def jevk5_client_compatible() -> bool:
    if jevk5_client_version() != JEVK5_RUNTIME_VERSION:
        return False
    try:
        from jevk5.gguf import JevK5GGUF  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def install_jevk5_client(*, timeout: float = 600.0) -> tuple[bool, str]:
    """Install the pinned lightweight GGUF client into the active interpreter.

    ``--no-deps`` keeps JevK5's torch runtime out; the GGUF path needs only the
    standard library. Refused offline.
    """
    if sys.version_info < (3, 10):
        return False, "Python >= 3.10 is required"
    if is_offline():
        return False, "HEADROOM_OFFLINE is set: not installing the JevK5 client"
    if jevk5_client_compatible():
        return True, "already installed"
    from headroom.offline import guard_egress

    guard_egress("JevK5 client install", "github.com/allebee/jevk5")
    commands = [[sys.executable, "-m", "pip", "install", "--no-deps", JEVK5_PIP_SPEC]]
    uv = shutil.which("uv")
    if uv:
        commands.append(
            [uv, "pip", "install", "--python", sys.executable, "--no-deps", JEVK5_PIP_SPEC]
        )
    detail = ""
    for command in commands:
        code, output = _run(command, timeout=timeout)
        if code == 0:
            import importlib

            importlib.invalidate_caches()
            if jevk5_client_compatible():
                return True, "installed"
        detail = output.strip().splitlines()[-1][:200] if output.strip() else f"exit {code}"
    return False, f"pip install failed: {detail}"
