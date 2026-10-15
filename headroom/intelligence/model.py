"""JevK5 GGUF model cache: verify, validate metadata, download once.

A model file is usable only after (a) its GGUF header and metadata parse and
name a supported architecture, and (b) its SHA-256 equals the trusted checksum
for *that* file — the publisher's hash for the default
``jevk5-4b-v0.3-Q8_0.gguf``, or the operator-supplied hash for a custom file.
A custom file without a checksum is usable only when explicitly marked
operator-managed. A successful verification is remembered in a sidecar keyed
by size + mtime so ``status`` does not rehash 4.5 GB each time.

Downloads (online only) resume into ``<file>.part`` with HTTP Range requests
and are atomically renamed into place only after the checksum matches.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import struct
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from headroom.agent.private_fs import atomic_write_json, read_json, secure_dir
from headroom.intelligence.settings import DEFAULT_MODEL_BYTES_APPROX, IntelligenceSettings
from headroom.offline import guard_egress, is_offline

GGUF_MAGIC = b"GGUF"
SUPPORTED_ARCH_PREFIXES = ("qwen3", "qwen35", "qwen3.5")
_SIDECAR_SUFFIX = ".headroom-verified.json"
_CHUNK = 8 * 1024 * 1024

ProgressFn = Callable[[int, int | None], None]


class ModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class GGUFInfo:
    version: int
    tensor_count: int
    architecture: str | None
    name: str | None
    context_length: int | None


@dataclass(frozen=True)
class ModelStatus:
    path: str
    present: bool
    verified: bool
    sha256: str | None
    expected_sha256: str | None
    operator_managed: bool
    gguf: GGUFInfo | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "present": self.present,
            "verified": self.verified,
            "sha256": self.sha256,
            "expected_sha256": self.expected_sha256,
            "operator_managed": self.operator_managed,
            "architecture": self.gguf.architecture if self.gguf else None,
            "model_context_length": self.gguf.context_length if self.gguf else None,
            "detail": self.detail,
        }


# -- GGUF metadata -----------------------------------------------------------

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q",
           11: "<q", 12: "<d"}


def _read(handle: BinaryIO, fmt: str) -> Any:
    size = struct.calcsize(fmt)
    data = handle.read(size)
    if len(data) != size:
        raise ModelError("truncated GGUF header")
    return struct.unpack(fmt, data)[0]


def _read_str(handle: BinaryIO) -> str:
    length = _read(handle, "<Q")
    if length > 1 << 20:
        raise ModelError("implausible GGUF string length")
    data = handle.read(length)
    if len(data) != length:
        raise ModelError("truncated GGUF string")
    return data.decode("utf-8", "replace")


def _read_value(handle: BinaryIO, vtype: int, *, keep: bool) -> Any:
    if vtype in _SCALAR:
        return _read(handle, _SCALAR[vtype])
    if vtype == 8:
        return _read_str(handle)
    if vtype == 9:
        item_type = _read(handle, "<I")
        count = _read(handle, "<Q")
        if item_type in _SCALAR:
            handle.seek(struct.calcsize(_SCALAR[item_type]) * count, os.SEEK_CUR)
        else:
            for _ in range(count):
                _read_value(handle, item_type, keep=False)
        return None
    raise ModelError(f"unknown GGUF value type {vtype}")


def read_gguf_info(path: Path) -> GGUFInfo:
    """Parse the GGUF header and the metadata keys Headroom validates."""
    with open(path, "rb") as handle:
        if handle.read(4) != GGUF_MAGIC:
            raise ModelError("not a GGUF file")
        version = _read(handle, "<I")
        if version < 2 or version > 3:
            raise ModelError(f"unsupported GGUF version {version}")
        tensor_count = _read(handle, "<Q")
        kv_count = _read(handle, "<Q")
        if kv_count > 100_000:
            raise ModelError("implausible GGUF metadata count")
        meta: dict[str, Any] = {}
        for _ in range(kv_count):
            key = _read_str(handle)
            vtype = _read(handle, "<I")
            value = _read_value(handle, vtype, keep=True)
            if key.startswith("general.") or key.endswith(".context_length"):
                meta[key] = value
    arch = meta.get("general.architecture")
    ctx = meta.get(f"{arch}.context_length") if arch else None
    return GGUFInfo(
        version=version,
        tensor_count=int(tensor_count),
        architecture=str(arch) if arch else None,
        name=str(meta["general.name"]) if meta.get("general.name") else None,
        context_length=int(ctx) if isinstance(ctx, int) else None,
    )


# -- hashing / verification -------------------------------------------------


def sha256_file(path: Path, *, progress: ProgressFn | None = None) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)
    return digest.hexdigest()


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + _SIDECAR_SUFFIX)


def _stat_key(path: Path) -> dict[str, int]:
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def model_status(
    settings: IntelligenceSettings, *, force_hash: bool = False, progress: ProgressFn | None = None
) -> ModelStatus:
    """Verify the configured model (hash cached by size+mtime unless forced)."""
    path = settings.model_path
    expected = settings.expected_sha256
    operator = settings.model_operator_managed and expected is None
    if not path.is_file():
        return ModelStatus(str(path), False, False, None, expected, operator, None, "not cached")
    try:
        info = read_gguf_info(path)
    except (OSError, ModelError) as exc:
        return ModelStatus(str(path), True, False, None, expected, operator, None, str(exc))
    arch = (info.architecture or "").lower()
    if not arch.startswith(SUPPORTED_ARCH_PREFIXES):
        return ModelStatus(
            str(path), True, False, None, expected, operator, info,
            f"unsupported architecture {info.architecture!r}",
        )
    if expected is None:
        if operator:
            return ModelStatus(
                str(path), True, True, None, None, True, info, "operator-managed (no checksum)"
            )
        return ModelStatus(
            str(path), True, False, None, None, False, info,
            "custom model has no trusted checksum (set HEADROOM_JEVK5_MODEL_SHA256 or "
            "HEADROOM_JEVK5_MODEL_OPERATOR_MANAGED=1)",
        )
    key = _stat_key(path)
    sidecar = read_json(_sidecar(path))
    if (
        not force_hash
        and isinstance(sidecar, dict)
        and sidecar.get("sha256") == expected
        and sidecar.get("size") == key["size"]
        and sidecar.get("mtime_ns") == key["mtime_ns"]
        and sidecar.get("schema") == 1
    ):
        return ModelStatus(str(path), True, True, expected, expected, False, info, "verified")
    actual = sha256_file(path, progress=progress)
    if actual != expected:
        return ModelStatus(
            str(path), True, False, actual, expected, False, info, "SHA-256 mismatch"
        )
    try:
        atomic_write_json(_sidecar(path), {"schema": 1, "sha256": actual, **key,
                                           "verified_at": time.time()})
    except OSError:
        pass
    return ModelStatus(str(path), True, True, actual, expected, False, info, "verified")


def download_model(
    settings: IntelligenceSettings, *, progress: ProgressFn | None = None, timeout: float = 60.0
) -> ModelStatus:
    """Resumable Hugging Face download of the default (or checksummed custom) file."""
    if settings.model_path_override:
        raise ModelError("a model path override is operator-managed; Headroom will not download")
    if settings.expected_sha256 is None:
        raise ModelError("refusing to download a model without a trusted checksum")
    if is_offline():
        raise ModelError("HEADROOM_OFFLINE is set: using the verified cache only")
    target = settings.model_path
    secure_dir(target.parent)
    part = target.with_name(target.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    free = shutil.disk_usage(target.parent).free
    needed = max(0, DEFAULT_MODEL_BYTES_APPROX - have) + 256 * 1024 * 1024
    if free < needed:
        raise ModelError(
            f"not enough free disk space in {target.parent} ({free // 2**20} MiB free, "
            f"~{needed // 2**20} MiB needed); set HEADROOM_WORKSPACE_DIR to a larger volume"
        )
    url = f"https://huggingface.co/{settings.model_repo}/resolve/main/{settings.model_file}"
    guard_egress("JevK5 model download", url)
    headers = {"User-Agent": "headroom-intelligence"}
    if have:
        headers["Range"] = f"bytes={have}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if have and response.status != 206:
                have = 0  # server ignored Range: restart cleanly
            length = response.headers.get("Content-Length")
            total = (have + int(length)) if length and length.isdigit() else None
            mode = "ab" if have else "wb"
            done = have
            with open(part, mode) as handle:
                while chunk := response.read(_CHUNK):
                    handle.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
                handle.flush()
                os.fsync(handle.fileno())
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise ModelError(f"model download interrupted ({type(exc).__name__}); rerun to resume") from exc
    actual = sha256_file(part)
    if actual != settings.expected_sha256:
        part.unlink(missing_ok=True)
        raise ModelError("downloaded model failed SHA-256 verification; partial file removed")
    os.replace(part, target)
    return model_status(settings)
