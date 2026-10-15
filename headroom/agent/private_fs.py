"""Owner-only directories, symlink-escape checks and atomic writes for agent state.

POSIX: directories ``0700`` and files ``0600`` via :mod:`headroom.fileperms`.
Windows: the leaf directory's ACL is replaced with a single user-scoped grant
(inheritance removed) using the built-in ``icacls`` tool, so files created
inside inherit an ACL readable only by the current user. Python has no ACL
API; failure to apply the ACL is reported (``False``), never claimed.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from headroom import fileperms

log = logging.getLogger(__name__)

_ACL_APPLIED: set[str] = set()


class UnsafePathError(PermissionError):
    """A state path resolves outside its permitted root (symlink escape)."""


def ensure_within(path: Path, root: Path) -> Path:
    """Resolve ``path`` and require it to stay inside ``root`` (both resolved).

    Defeats a symlink planted anywhere along the path that would redirect
    state writes outside the workspace.
    """
    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise UnsafePathError(f"state path escapes its root: {path} -> {resolved}")
    return resolved


def _windows_user_acl(path: Path) -> bool:
    key = str(path)
    if key in _ACL_APPLIED:
        return True
    icacls = shutil.which("icacls")
    user = os.environ.get("USERNAME")
    domain = os.environ.get("USERDOMAIN")
    if not icacls or not user:
        return False
    principal = f"{domain}\\{user}" if domain else user
    try:
        result = subprocess.run(
            [icacls, key, "/inheritance:r", "/grant:r", f"{principal}:(OI)(CI)F"],
            capture_output=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if result.returncode == 0:
        _ACL_APPLIED.add(key)
        return True
    return False


def secure_dir(path: Path) -> bool:
    """Create ``path`` and restrict it to the current user. Returns whether enforced."""
    if path.is_symlink():
        raise UnsafePathError(f"refusing to use a symlinked state directory: {path}")
    fileperms.private_dir(path)
    if fileperms.OWNER_ONLY_SUPPORTED:
        return True
    if sys.platform == "win32":
        return _windows_user_acl(path)
    return False


def atomic_write_json(path: Path, value: Any) -> None:
    """Write JSON atomically (temp file in the same dir + ``os.replace``), owner-only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise UnsafePathError(f"refusing to replace a symlink: {path}")
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        fileperms.restrict_fd_to_owner(fd)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: Path) -> Any | None:
    """Read a JSON file; ``None`` when missing, a symlink, or unparsable."""
    try:
        if path.is_symlink() or not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
