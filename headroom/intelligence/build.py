"""Build a separate, Headroom-managed ``llama-server`` from official source.

Used only when no compatible user-installed binary exists. The build lives
under ``<workspace>/intelligence/llama.cpp`` and never touches the user's own
installation. Requires ``git`` and ``cmake`` (plus a C++ toolchain); CUDA is
enabled when ``nvcc`` is available, otherwise a CPU build is produced. No
root/admin is needed. Refused under ``HEADROOM_OFFLINE``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from headroom.agent.private_fs import secure_dir
from headroom.intelligence.settings import IntelligenceSettings
from headroom.offline import guard_egress, is_offline

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
LogFn = Callable[[str], None]


class BuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class BuildResult:
    binary: Path
    cuda: bool
    ref: str


def build_root() -> Path:
    from headroom import paths

    return paths.intelligence_dir() / "llama.cpp"


def _cuda_available() -> bool:
    return bool(shutil.which("nvcc") or os.environ.get("CUDA_PATH"))


def _step(args: list[str], cwd: Path | None, log: LogFn, timeout: float) -> None:
    log("$ " + " ".join(args))
    try:
        result = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=_NO_WINDOW,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BuildError(f"{args[0]} failed to run: {type(exc).__name__}") from exc
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-5:]
        raise BuildError(f"{' '.join(args[:3])} exited {result.returncode}: " + " | ".join(tail))


def find_built_binary(root: Path) -> Path | None:
    exe = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    for rel in ("build/bin/Release", "build/bin", "build/Release/bin"):
        candidate = root / rel / exe
        if candidate.is_file():
            return candidate
    return None


def build_llama_server(settings: IntelligenceSettings, *, log: LogFn = print) -> BuildResult:
    if is_offline():
        raise BuildError("HEADROOM_OFFLINE is set: cannot fetch llama.cpp source")
    git = shutil.which("git")
    cmake = shutil.which("cmake")
    if not git or not cmake:
        raise BuildError(
            "building llama-server needs git and cmake on PATH (or install llama.cpp yourself "
            "and set HEADROOM_JEVK5_LLAMA_SERVER)"
        )
    root = build_root()
    secure_dir(root.parent)
    guard_egress("llama.cpp source checkout", settings.llama_cpp_repo)
    if not (root / ".git").is_dir():
        _step([git, "clone", "--filter=blob:none", settings.llama_cpp_repo, str(root)], None, log,
              1800)
    else:
        _step([git, "fetch", "--tags", "origin"], root, log, 900)
    _step([git, "checkout", "--detach", settings.llama_cpp_ref], root, log, 300)
    cuda = _cuda_available()
    configure = [
        cmake,
        "-S",
        str(root),
        "-B",
        str(root / "build"),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_CURL=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        f"-DGGML_CUDA={'ON' if cuda else 'OFF'}",
    ]
    _step(configure, root, log, 1800)
    jobs = str(max(1, (os.cpu_count() or 2) - 1))
    _step(
        [cmake, "--build", str(root / "build"), "--config", "Release", "--target", "llama-server",
         "-j", jobs],
        root,
        log,
        7200,
    )
    binary = find_built_binary(root)
    if binary is None:
        raise BuildError("build finished but llama-server was not found in the build tree")
    return BuildResult(binary=binary, cuda=cuda, ref=settings.llama_cpp_ref)
