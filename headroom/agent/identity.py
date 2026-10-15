"""Session and workspace identity from the wrap trust channel.

A request is *trusted* only when its ``x-headroom-session-token`` matches a
live marker that ``headroom wrap`` registered (``workspace_registry``) and the
registered cwd still resolves to an existing directory. From that:

* ``session_id`` — digest of the registered token (the token itself is never
  stored or logged);
* ``workspace_id`` — digest of the OS principal and the canonical root of the
  working tree the cwd belongs to. A linked Git worktree has its own working
  tree root, so it is a distinct mutation scope; ``repository_id`` (digest of
  the shared Git common dir) links worktrees of one repository for read-only
  correlation.

Anything else — no token, a stale marker, a vanished directory, or only an
``X-Headroom-Project`` attribution header — yields an *untrusted* identity:
in-memory observation is allowed, path-sensitive protection or execution is
not (:meth:`WorkspaceIdentity.can_authorize_paths`).
"""

from __future__ import annotations

import getpass
import hashlib
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

# Untrusted identities are salted per process so their ids never collide with,
# or can be precomputed to impersonate, a trusted workspace.
_UNTRUSTED_SALT = secrets.token_hex(16)


def _digest(*parts: str) -> str:
    raw = "\x1f".join(parts).encode("utf-8", "surrogatepass")
    return hashlib.sha256(raw).hexdigest()[:32]


def current_principal() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - getuser raises varied errors without a login name
        return f"uid:{os.getuid()}" if hasattr(os, "getuid") else "unknown"


def _canonical(path: str) -> Path:
    resolved = Path(os.path.realpath(path))
    return Path(os.path.normcase(str(resolved)))


@dataclass(frozen=True)
class GitLayout:
    worktree_root: Path
    common_dir: Path | None
    linked_worktree: bool


def git_layout(start: Path) -> GitLayout:
    """Find the working-tree root for ``start`` and whether it is a linked worktree.

    Reads ``.git`` directly (no subprocess). A ``.git`` *file* containing
    ``gitdir: <path>`` marks a linked worktree (or submodule); its
    ``commondir`` file points at the shared repository.
    """
    current = start
    for candidate in (current, *current.parents):
        dot_git = candidate / ".git"
        try:
            if dot_git.is_dir():
                return GitLayout(candidate, _canonical(str(dot_git)), False)
            if dot_git.is_file():
                text = dot_git.read_text(encoding="utf-8", errors="replace").strip()
                if text.startswith("gitdir:"):
                    gitdir = Path(text.split(":", 1)[1].strip())
                    if not gitdir.is_absolute():
                        gitdir = candidate / gitdir
                    common = gitdir
                    commondir_file = gitdir / "commondir"
                    if commondir_file.is_file():
                        rel = commondir_file.read_text(encoding="utf-8").strip()
                        common = Path(rel) if Path(rel).is_absolute() else gitdir / rel
                    return GitLayout(candidate, _canonical(str(common)), True)
        except OSError:
            break
    return GitLayout(start, None, False)


@dataclass(frozen=True)
class WorkspaceIdentity:
    workspace_id: str
    session_id: str
    trusted: bool
    principal: str
    root: str | None = None
    repository_id: str | None = None
    linked_worktree: bool = False

    def can_authorize_paths(self) -> bool:
        """Path-sensitive protection/execution needs a trusted, rooted identity."""
        return self.trusted and self.root is not None


def resolve_identity(
    *,
    session_token: str | None,
    registered_cwd: str | None,
    attribution: str | None = None,
    principal: str | None = None,
) -> WorkspaceIdentity:
    """Derive the identity for one request (see module docstring)."""
    who = principal or current_principal()
    if session_token and registered_cwd:
        try:
            root = _canonical(registered_cwd)
            if root.is_dir():
                layout = git_layout(root)
                tree_root = _canonical(str(layout.worktree_root))
                return WorkspaceIdentity(
                    workspace_id="ws_" + _digest("workspace", who, str(tree_root)),
                    session_id="ss_" + _digest("session", session_token),
                    trusted=True,
                    principal=who,
                    root=str(tree_root),
                    repository_id=(
                        "repo_" + _digest("repository", who, str(layout.common_dir))
                        if layout.common_dir is not None
                        else None
                    ),
                    linked_worktree=layout.linked_worktree,
                )
        except OSError:
            pass
    label = attribution or ""
    return WorkspaceIdentity(
        workspace_id="wu_" + _digest("untrusted-workspace", _UNTRUSTED_SALT, who, label),
        session_id="su_"
        + _digest("untrusted-session", _UNTRUSTED_SALT, who, label, session_token or ""),
        trusted=False,
        principal=who,
    )
