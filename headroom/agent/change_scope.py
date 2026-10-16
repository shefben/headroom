"""Change-Scope / Drift Firewall (unified-21 #23).

The firewall answers one question for every potentially mutating action:
*is this change part of what the user asked for?* It compiles a
:class:`ChangeContract` per (workspace, session) from

* **explicit user instructions** — live USER-origin TaskState atoms: named
  files/directories/globs become expected scope, "do not touch X" /
  "leave X alone" become forbidden paths, "only edit …" / "nothing else"
  freezes the contract, "don't commit/push/delete" forbids an effect;
* **TaskState** — the task id and revision, and its recorded write resources;
* **validated EvidenceFacts** — live negative test/error facts that name a
  file justify touching that file (``DEPENDENCY_SCOPE``), and the
  ``git.changed_paths`` fact is the after-edit diff check;
* **recent authorized writes** — writes that were classified in scope.

It never infers scope from unrelated reads, and a JevK5 vote is never an
input: dependency expansion needs a traceable reason (direct import/build
link, a necessary interface, failing compiler/test evidence, required
platform/config file, or a related test), recorded on the contract.

Classification: ``IN_SCOPE | DEPENDENCY_SCOPE | GENERATED_EFFECT | UNRELATED
| FORBIDDEN``. Hard protection (credentials, private keys, ``.git``
internals, AI-client provider config, user-excluded paths) is FORBIDDEN.
Enforcement is a separate typed policy (``observe|warn|protect``, default
``warn``): FORBIDDEN is blocked only at a trusted host pre-tool boundary;
unexplained (UNRELATED) changes warn; ``protect`` additionally blocks
unrelated changes under a FROZEN contract. An untrusted/unknown workspace
never receives write authority.

Every decision that would let a mutating action proceed is pinned by an
:class:`~headroom.agent.interfaces.ActionAuthorization` to the task, scope
and resource revision it was computed from, and :meth:`authorize` verifies
it again immediately before returning to the host: a mismatch re-evaluates
once, otherwise the protected action is denied (``scope.stale_revision``).

Paths are canonicalized before any comparison: relative paths join the
call's cwd, MSYS (``/c/...``) and extended (``\\\\?\\``) forms are unwrapped,
symlinks are resolved, a lexical-inside/resolved-outside path is a symlink
escape, UNC shares are flagged, and comparison folds case only on
case-insensitive platforms. Only the registered worktree is ever in scope.
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
import re
import shlex
import sys
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

from headroom.agent.events import EventEnvelope, EventKind, Origin, make_event
from headroom.agent.interfaces import ActionAuthorization, AuthorizationDecision
from headroom.log_safety import describe_exception

if TYPE_CHECKING:
    from headroom.agent.identity import WorkspaceIdentity
    from headroom.agent.task_state import TaskSnapshot

log = logging.getLogger(__name__)

FEATURE_CHANGE_SCOPE = "agent_change_scope"
CHANGE_CONTRACT_VERSION = 1
MODE_ENV = "HEADROOM_CHANGE_SCOPE_MODE"
TOOL_CONTRACTS_MODE_ENV = "HEADROOM_TOOL_CONTRACTS_MODE"
AUTHORIZATION_TTL_S = 30.0
_CONTRACTS_LIMIT = 512
_RESOURCES_LIMIT = 8192
_WRITES_LIMIT = 64
_CLASSIFIED_LIMIT = 8192
_SOURCE_READ_LIMIT = 256 * 1024


class ScopeClass(str, Enum):
    IN_SCOPE = "IN_SCOPE"
    DEPENDENCY_SCOPE = "DEPENDENCY_SCOPE"
    GENERATED_EFFECT = "GENERATED_EFFECT"
    UNRELATED = "UNRELATED"
    FORBIDDEN = "FORBIDDEN"


class ContractMode(str, Enum):
    LEARNING = "LEARNING"  # new goal: no expected scope yet, observe
    ACTIVE = "ACTIVE"  # expected scope established
    FROZEN = "FROZEN"  # the user explicitly locked the scope


class Enforcement(str, Enum):
    OBSERVE = "observe"
    WARN = "warn"
    PROTECT = "protect"


class Verdict(str, Enum):
    ALLOW = "ALLOW"
    WARN = "WARN"
    BLOCK = "BLOCK"


def _mode_from(value: str | None, name: str) -> Enforcement:
    raw = (value or "").strip().lower()
    if not raw:
        return Enforcement.WARN
    try:
        return Enforcement(raw)
    except ValueError:
        log.warning("%s=%r is not observe|warn|protect; using warn", name, value)
        return Enforcement.WARN


@dataclass(frozen=True)
class EnforcementPolicy:
    """Per-feature enforcement behavior (separate from rollout on/off)."""

    change_scope: Enforcement = Enforcement.WARN
    tool_contracts: Enforcement = Enforcement.WARN

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> EnforcementPolicy:
        source = os.environ if env is None else env
        return cls(
            change_scope=_mode_from(source.get(MODE_ENV), MODE_ENV),
            tool_contracts=_mode_from(source.get(TOOL_CONTRACTS_MODE_ENV), TOOL_CONTRACTS_MODE_ENV),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "change_scope": self.change_scope.value,
            "tool_contracts": self.tool_contracts.value,
        }


# ---------------------------------------------------------------------------
# Path canonicalization
# ---------------------------------------------------------------------------


def platform_case_insensitive() -> bool:
    return sys.platform in ("win32", "cygwin", "darwin")


_MSYS_DRIVE = re.compile(r"^/(?:cygdrive/)?([A-Za-z])(?:/(.*))?$")


@dataclass(frozen=True)
class CanonicalPath:
    raw: str
    resolved: str  # real path, forward slashes, original case
    rel: str | None  # workspace-relative ('/' separated) or None when outside
    lexical_rel: str | None  # relative form before symlink resolution
    inside: bool
    symlink_escape: bool
    unc: bool
    exists: bool
    is_dir: bool


def _unwrap(raw: str) -> str:
    text = raw.strip().strip("`'\"")
    if text.startswith(("\\\\?\\UNC\\", "//?/UNC/")):
        text = "\\\\" + text[8:]
    elif text.startswith(("\\\\?\\", "//?/", "\\\\.\\")):
        text = text[4:]
    if os.name == "nt":
        match = _MSYS_DRIVE.match(text.replace("\\", "/"))
        if match and not text.startswith("//"):
            text = f"{match.group(1).upper()}:/" + (match.group(2) or "")
    return os.path.expanduser(text)


def _cmp(path: str, fold: bool) -> str:
    norm = path.replace("\\", "/").rstrip("/") or "/"
    return norm.lower() if fold else norm


def _relative(path: str, root: str, fold: bool) -> str | None:
    p, r = _cmp(path, fold), _cmp(root, fold)
    if p == r:
        return "."
    if p.startswith(r.rstrip("/") + "/"):
        return path.replace("\\", "/")[len(r.rstrip("/")) + 1 :].rstrip("/")
    return None


def canonicalize(
    raw: str,
    *,
    root: str | None,
    cwd: str | None = None,
    case_insensitive: bool | None = None,
) -> CanonicalPath:
    """Canonical form of a model-supplied path (see module docstring)."""
    fold = platform_case_insensitive() if case_insensitive is None else case_insensitive
    text = _unwrap(raw)
    base = cwd or root or os.getcwd()
    if root and cwd:
        real_cwd = os.path.realpath(_unwrap(cwd))
        if _relative(real_cwd, os.path.realpath(root), fold) is None and not os.path.isdir(
            real_cwd
        ):
            base = root  # a vanished/unknown cwd never moves relative paths outside the root
    joined = (
        text if os.path.isabs(text) or text.startswith(("\\\\", "//")) else os.path.join(base, text)
    )
    lexical = os.path.normpath(joined)
    try:
        resolved = os.path.realpath(lexical)
    except (OSError, ValueError):
        resolved = lexical
    real_root = os.path.realpath(root) if root else None
    rel = _relative(resolved, real_root, fold) if real_root else None
    lexical_rel = _relative(lexical, real_root, fold) if real_root else None
    if real_root and lexical_rel is None:
        lexical_rel = _relative(lexical, root or "", fold)
    try:
        exists = os.path.lexists(resolved)
        is_dir = os.path.isdir(resolved)
    except (OSError, ValueError):
        exists, is_dir = False, False
    forward = resolved.replace("\\", "/")
    return CanonicalPath(
        raw=raw,
        resolved=forward,
        rel=rel,
        lexical_rel=lexical_rel,
        inside=rel is not None,
        symlink_escape=lexical_rel is not None and rel is None,
        unc=forward.startswith("//"),
        exists=exists,
        is_dir=is_dir,
    )


def path_key(rel: str, fold: bool | None = None) -> str:
    folded = platform_case_insensitive() if fold is None else fold
    norm = rel.replace("\\", "/").strip("/")
    norm = re.sub(r"^(?:\./)+", "", norm)
    return norm.lower() if folded else norm


# ---------------------------------------------------------------------------
# Hard protection
# ---------------------------------------------------------------------------

_KEY_NAMES = frozenset(
    {"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "id_ecdsa_sk", "id_ed25519_sk"}
)
_KEY_EXTS = frozenset({".pem", ".key", ".p12", ".pfx", ".jks", ".keystore", ".ppk"})
_CREDENTIAL_NAMES = frozenset(
    {
        ".netrc",
        "_netrc",
        ".npmrc",
        ".pypirc",
        ".git-credentials",
        ".pgpass",
        ".dockercfg",
        "credentials",
        "credentials.json",
        ".htpasswd",
    }
)
_CREDENTIAL_DIRS = frozenset({".ssh", ".aws", ".gnupg", ".azure", ".kube", ".docker"})
_ENV_TEMPLATES = frozenset({"example", "sample", "template", "dist", "defaults", "schema"})
_PROVIDER_CONFIG = (
    (".claude", "settings.json"),
    (".claude", "settings.local.json"),
    ("", ".claude.json"),
    (".codex", "config.toml"),
    (".codex", "auth.json"),
    (".gemini", "settings.json"),
)


def hard_protection(path: str, *, fold: bool | None = None) -> str | None:
    """``git_internal | private_key | credentials | provider_config`` or None."""
    folded = platform_case_insensitive() if fold is None else fold
    norm = path.replace("\\", "/")
    parts = [p for p in norm.split("/") if p and p != "."]
    if not parts:
        return None
    cmp = [p.lower() for p in parts] if folded else parts
    if ".git" in cmp:
        return "git_internal"
    name = cmp[-1]
    if name in _KEY_NAMES or os.path.splitext(name)[1].lower() in _KEY_EXTS:
        return "private_key"
    if name in _CREDENTIAL_NAMES or any(d in _CREDENTIAL_DIRS for d in cmp[:-1]):
        return "credentials"
    if name == ".env" or (
        name.startswith(".env.") and name.split(".", 2)[2].lower() not in _ENV_TEMPLATES
    ):
        return "credentials"
    parent = cmp[-2] if len(cmp) >= 2 else ""
    for directory, filename in _PROVIDER_CONFIG:
        if name == filename and (not directory or parent == directory):
            return "provider_config"
    if ".headroom" in cmp[:-1]:
        return "provider_config"
    return None


# ---------------------------------------------------------------------------
# User-instruction parsing (TaskState atoms, USER origin only)
# ---------------------------------------------------------------------------

_FORBID = re.compile(
    r"\b(?:do\s+not|don'?t|never|must\s+not|mustn'?t|should\s+not|shouldn'?t|avoid|without|no)\s+"
    r"(?:\w+\s+){0,2}?(?:touch(?:ing)?|modify(?:ing)?|edit(?:ing)?|chang(?:e|ing)|writ(?:e|ing)|"
    r"delet(?:e|ing)|remov(?:e|ing)|alter(?:ing)?|rewrit(?:e|ing)|updat(?:e|ing)|chang(?:es))\b"
    r"|\bleave\b|\b(?:is|are)\s+off[- ]limits\b|\bexclud(?:e|ing)\b|\bhands\s+off\b",
    re.IGNORECASE,
)
_LEAVE_TAIL = re.compile(r"\b(?:alone|untouched|unchanged|as\s+is)\b", re.IGNORECASE)
_LOCK = re.compile(
    r"\bonly\s+(?:touch|modify|edit|change|update|work\s+on|write)\b|\b(?:limit|restrict|confine)\s+"
    r"(?:the\s+|your\s+)?(?:changes|edits|yourself|work)\s+to\b|\bnothing\s+else\b|"
    r"\bno\s+other\s+(?:files|changes)\b|\b(?:lock|freeze)\s+(?:the\s+)?scope\b|\bscope\s+is\s+locked\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(?:do\s+not|don'?t|never|must\s+not|mustn'?t|should\s+not|shouldn'?t|avoid|no)\b", re.I
)
_EFFECT_WORDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("git_push", re.compile(r"\b(?:push|pushing)\b", re.I)),
    ("git_commit", re.compile(r"\b(?:commit|committing)\b", re.I)),
    (
        "git_history_rewrite",
        re.compile(
            r"\b(?:force[- ]push|rebase|reset\s+--hard|hard\s+reset|amend|rewrite\s+history)\b",
            re.I,
        ),
    ),
    ("delete_untracked", re.compile(r"\b(?:git\s+clean|untracked)\b", re.I)),
    (
        "delete",
        re.compile(r"\b(?:delete|deleting|remove|removing|rm)\s+(?:any\s+|the\s+)?files?\b", re.I),
    ),
    (
        "package_install",
        re.compile(
            r"\b(?:install|add)\s+(?:any\s+|new\s+)?(?:packages?|dependenc\w+|deps)\b", re.I
        ),
    ),
)
DEFAULT_PERMITTED_EFFECTS = (
    "edit",
    "create",
    "delete",
    "git_stage",
    "git_commit",
    "package_install",
    "discard",
)
#: Effects that only the user can permit (warned when not permitted).
EXPLICIT_EFFECTS = ("git_push", "git_history_rewrite", "delete_untracked")

_TOKEN_SPLIT = re.compile(r"[\s,;()\[\]{}<>\"'`]+")
_EXT = re.compile(r"\.[A-Za-z][A-Za-z0-9_+-]{0,9}$")
_NOT_PATH = re.compile(r"^(?:e\.g|i\.e|etc|vs|a\.k\.a|n\.b)\.?$", re.IGNORECASE)


def path_mentions(text: str) -> list[str]:
    """Path-like tokens a user wrote (files, directories, globs)."""
    out: list[str] = []
    for token in _TOKEN_SPLIT.split(text):
        tok = token.strip().rstrip(".,:;!?")
        if len(tok) < 2 or "://" in tok or tok.startswith(("-", "$", "@")) or _NOT_PATH.match(tok):
            continue
        has_sep = "/" in tok or "\\" in tok
        if has_sep:
            if not re.search(r"[A-Za-z0-9_*]", tok):
                continue
        elif not (_EXT.search(tok) and re.search(r"[A-Za-z_]", tok.rsplit(".", 1)[0] or "x")):
            continue
        if re.fullmatch(r"v?\d+(?:\.\d+)+", tok):
            continue
        if tok not in out:
            out.append(tok)
    return out


@dataclass(frozen=True)
class Mention:
    path: str
    kind: str  # "expect" | "forbid"
    ts: float
    atom_id: str


_CLAUSE = re.compile(r"\s*(?:[,;]|\.\s|\bbut\b|\band\b|\bwhile\b|\bexcept\b|\bhowever\b)\s*", re.I)


def _clause_mentions(clause: str, ts: float, atom_id: str) -> list[Mention]:
    forbid = _FORBID.search(clause)
    if (
        forbid is not None
        and forbid.group(0).lower().startswith("leave")
        and not _LEAVE_TAIL.search(clause)
    ):
        forbid = None
    if forbid is None:
        return [Mention(p, "expect", ts, atom_id) for p in path_mentions(clause)]
    phrase = forbid.group(0).lower()
    head, tail = clause[: forbid.start()], clause[forbid.start() :]
    if re.search(r"off[- ]limits|hands\s+off", phrase):
        # "X is off-limits": the named paths are the object, wherever they are.
        return [Mention(p, "forbid", ts, atom_id) for p in path_mentions(clause)]
    out = [Mention(p, "expect", ts, atom_id) for p in path_mentions(head)]
    out.extend(Mention(p, "forbid", ts, atom_id) for p in path_mentions(tail))
    return out


def _atom_mentions(atom: Any) -> tuple[list[Mention], bool]:
    """(mentions, locks) of one user atom, clause by clause."""
    text = str(getattr(atom, "text", "") or "")
    ts = float(getattr(atom, "updated_at", 0.0) or 0.0)
    atom_id = str(getattr(atom, "id", ""))
    out: list[Mention] = []
    for clause in _CLAUSE.split(text):
        if clause and clause.strip():
            out.extend(_clause_mentions(clause, ts, atom_id))
    return out, bool(_LOCK.search(text))


def _effect_mentions(text: str) -> tuple[set[str], set[str]]:
    """(permitted, forbidden) effects a user sentence names."""
    permitted: set[str] = set()
    forbidden: set[str] = set()
    for effect, pattern in _EFFECT_WORDS:
        for match in pattern.finditer(text):
            window = _CLAUSE.split(text[max(0, match.start() - 60) : match.start()])[-1]
            (forbidden if _NEGATION.search(window) else permitted).add(effect)
    return permitted, forbidden


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChangeContract:
    workspace_id: str
    task_id: str | None
    revision: int
    goal: str
    expected_paths: tuple[str, ...]
    dependency_paths: tuple[str, ...]
    generated_paths: tuple[str, ...]
    forbidden_paths: tuple[str, ...]
    permitted_effects: tuple[str, ...]
    mode: ContractMode
    created_at: float
    v: int = CHANGE_CONTRACT_VERSION
    session_id: str = ""
    task_revision: int = 0
    forbidden_effects: tuple[str, ...] = ()
    approved_protected: tuple[str, ...] = ()
    recent_writes: tuple[str, ...] = ()
    dependency_reasons: Mapping[str, str] = field(default_factory=dict)
    fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "v": self.v,
            "workspace_id": self.workspace_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "task_revision": self.task_revision,
            "revision": self.revision,
            "goal": self.goal,
            "expected_paths": list(self.expected_paths),
            "dependency_paths": list(self.dependency_paths),
            "generated_paths": list(self.generated_paths),
            "forbidden_paths": list(self.forbidden_paths),
            "permitted_effects": list(self.permitted_effects),
            "forbidden_effects": list(self.forbidden_effects),
            "mode": self.mode.value,
            "created_at": self.created_at,
            "recent_writes": list(self.recent_writes),
            "dependency_reasons": dict(self.dependency_reasons),
        }


def _pattern_matches(pattern: str, rel: str, fold: bool) -> bool:
    """A contract pattern (file, ``dir/``, glob or bare basename) covers ``rel``."""
    p = path_key(pattern, fold)
    r = path_key(rel, fold)
    if not p:
        return False
    if any(ch in p for ch in "*?["):
        return fnmatch.fnmatchcase(r, p) or fnmatch.fnmatchcase(r.rsplit("/", 1)[-1], p)
    if r == p or r.startswith(p + "/"):
        return True
    if "/" not in p:
        return r.rsplit("/", 1)[-1] == p
    return False


def matches_any(patterns: Iterable[str], rel: str, fold: bool | None = None) -> str | None:
    folded = platform_case_insensitive() if fold is None else fold
    for pattern in patterns:
        if _pattern_matches(pattern, rel, folded):
            return pattern
    return None


_LOCKFILES: dict[str, tuple[str, ...]] = {
    "package.json": (
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "bun.lockb",
        "npm-shrinkwrap.json",
    ),
    "cargo.toml": ("Cargo.lock",),
    "pyproject.toml": ("uv.lock", "poetry.lock", "pdm.lock"),
    "requirements.in": ("requirements.txt",),
    "go.mod": ("go.sum",),
    "gemfile": ("Gemfile.lock",),
    "composer.json": ("composer.lock",),
}
_CONFIG_FILES = frozenset(
    {
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
        "tox.ini",
        "pytest.ini",
        "conftest.py",
        "requirements.txt",
        "requirements-dev.txt",
        "cargo.toml",
        "package.json",
        "tsconfig.json",
        "jsconfig.json",
        "go.mod",
        "makefile",
        "cmakelists.txt",
        "build.gradle",
        "pom.xml",
        "meson.build",
        "build.rs",
        "__init__.py",
        "mod.rs",
        "lib.rs",
    }
)
_TEST_NAME = re.compile(
    r"(?:^|/)(?:test_[^/]+|[^/]+_test\.[^/.]+|[^/]+\.(?:test|spec)\.[^/]+|tests?/.+|__tests__/.+)$",
    re.IGNORECASE,
)


def _stem(rel: str) -> str:
    name = rel.rsplit("/", 1)[-1]
    name = name.split(".", 1)[0]
    for prefix in ("test_",):
        if name.lower().startswith(prefix):
            name = name[len(prefix) :]
    for suffix in ("_test", "_spec", "_tests"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    return name.lower()


@dataclass(frozen=True)
class ScopeVerdict:
    scope: ScopeClass
    reason_code: str
    rel: str | None
    trace: str = ""
    protection: str | None = None


@dataclass(frozen=True)
class PlannedChange:
    effect: str  # edit | create | delete | discard | git_* | package_install | delete_untracked
    path: str | None = None
    preview: bool = False


@dataclass(frozen=True)
class ScopeDecision:
    verdict: Verdict
    reason_code: str
    scope: ScopeClass
    rel: str | None
    trace: str
    enforceable: bool
    mode: ContractMode
    task_revision: int
    scope_revision: int
    resource_revision: int
    authorization: ActionAuthorization | None = None
    effect: str = "edit"
    reevaluated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "reason_code": self.reason_code,
            "scope": self.scope.value,
            "rel": self.rel,
            "trace": self.trace,
            "enforceable": self.enforceable,
            "mode": self.mode.value,
            "task_revision": self.task_revision,
            "scope_revision": self.scope_revision,
            "resource_revision": self.resource_revision,
            "authorized": self.authorization is not None,
            "effect": self.effect,
            "reevaluated": self.reevaluated,
        }


# ---------------------------------------------------------------------------
# Planned changes of a tool call (edit tools, apply_patch, shell)
# ---------------------------------------------------------------------------

_PATCH_FILE = re.compile(r"^\*\*\* (Update|Add|Delete) File: (.+)$", re.MULTILINE)
_EDIT_TOOLS = frozenset(
    {
        "edit",
        "write",
        "multiedit",
        "notebookedit",
        "str_replace_based_edit_tool",
        "str_replace_editor",
        "write_file",
        "edit_file",
        "create_file",
        "replace_in_file",
        "patch",
    }
)
_SHELL_TOOLS = frozenset(
    {
        "bash",
        "powershell",
        "shell",
        "local_shell",
        "exec_command",
        "run_terminal_cmd",
        "terminal",
        "run_command",
        "shell_command",
        "run_shell_command",
        "execute_command",
    }
)
_PATH_KEYS = ("file_path", "path", "filePath", "notebook_path", "filename", "target_file", "file")
_SEGMENTS = re.compile(r"\s*(?:&&|\|\||;|\n|(?<![|>])\|(?!\|))\s*")
_REDIRECT = re.compile(r"(?<![0-9&<])(?:\d?>>?|&>)\s*(?!&)([^\s|;&<>]+)")
_NULL_SINKS = frozenset({"/dev/null", "nul", "$null", "/dev/stderr", "/dev/stdout"})
_DRY_RUN = re.compile(r"(?:^|\s)(?:--dry-run|--dryrun|-WhatIf|--what-if|--check)\b", re.I)


def shell_of(tool_name: str) -> str:
    return "powershell" if tool_name.lower() == "powershell" else "posix"


def split_command(segment: str, shell: str = "posix") -> list[str]:
    try:
        if shell == "posix":
            return shlex.split(segment, posix=True)
        return [t.strip("'\"") for t in shlex.split(segment, posix=False)]
    except ValueError:
        return segment.split()


def command_segments(command: str) -> list[str]:
    return [s for s in _SEGMENTS.split(command.strip()) if s and s.strip()]


def _strip_prefix_words(words: list[str]) -> list[str]:
    out = list(words)
    while out and (
        re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", out[0])
        or out[0] in ("sudo", "doas", "time", "nohup", "env", "command", "exec", "builtin", "xargs")
    ):
        out = out[1:]
    return out


def _git_changes(words: list[str]) -> list[PlannedChange]:
    args = words[1:]
    while args and args[0] in ("-C", "-c", "--git-dir", "--work-tree"):
        args = args[2:]
    while args and args[0].startswith("--") and "=" in args[0]:
        args = args[1:]
    if not args:
        return []
    sub, rest = args[0], args[1:]
    flags = {a for a in rest if a.startswith("-")}
    preview = bool({"-n", "--dry-run"} & flags)
    if sub == "commit":
        effect = "git_history_rewrite" if "--amend" in flags else "git_commit"
        return [PlannedChange(effect, preview=preview)]
    if sub == "push":
        forced = bool(
            {"-f", "--force", "--force-with-lease", "--mirror", "--delete", "-d"} & flags
        ) or any(a.startswith("+") or a.startswith("--force") for a in rest)
        return [PlannedChange("git_history_rewrite" if forced else "git_push", preview=preview)]
    if sub in ("rebase", "filter-branch", "filter-repo") or (sub == "reset" and "--hard" in flags):
        return [PlannedChange("git_history_rewrite", preview=preview)]
    if sub == "clean":
        return [
            PlannedChange(
                "delete_untracked",
                preview=preview
                or not (
                    {"-f", "--force"} & flags or any(re.match(r"^-[a-zA-Z]*f", a) for a in rest)
                ),
            )
        ]
    if sub in ("checkout", "restore") and ("--" in rest or sub == "restore"):
        paths = (
            rest[rest.index("--") + 1 :]
            if "--" in rest
            else [a for a in rest if not a.startswith("-")]
        )
        if sub == "restore" and "--staged" in flags and "--worktree" not in flags:
            return [PlannedChange("git_stage")]
        return [PlannedChange("discard", p) for p in paths] or [PlannedChange("discard")]
    if sub == "stash" and rest[:1] and rest[0] in ("drop", "clear"):
        return [PlannedChange("git_history_rewrite")]
    if sub == "branch" and ({"-D", "--delete"} & flags or "-d" in flags):
        return [PlannedChange("git_history_rewrite" if "-D" in flags else "git_commit")]
    if sub in ("add", "rm", "mv"):
        targets = [a for a in rest if not a.startswith("-")]
        if sub == "add":
            return [PlannedChange("git_stage")]
        if sub == "rm":
            return [PlannedChange("delete", t) for t in targets]
        return (
            [PlannedChange("delete", targets[0]), PlannedChange("create", targets[-1])]
            if len(targets) >= 2
            else []
        )
    return []


def _shell_segment_changes(segment: str, shell: str) -> list[PlannedChange]:
    out: list[PlannedChange] = []
    for match in _REDIRECT.finditer(segment):
        target = match.group(1).strip("'\"")
        if target.lower() not in _NULL_SINKS and not target.startswith("&"):
            out.append(PlannedChange("edit", target))
    words = _strip_prefix_words(split_command(_REDIRECT.sub(" ", segment), shell))
    if not words:
        return out
    head = words[0].lower().rsplit("/", 1)[-1].removesuffix(".exe")
    operands = [w for w in words[1:] if not w.startswith("-")]
    preview = bool(_DRY_RUN.search(segment))
    if head == "git":
        return out + _git_changes(words)
    if head in ("rm", "del", "erase", "rmdir", "rd", "unlink", "remove-item", "ri"):
        named = [w for w in words[1:] if not w.startswith("-") and not w.startswith("/")]
        if head == "remove-item":
            named = _ps_operands(words[1:], ("-path", "-literalpath"))
        return out + [PlannedChange("delete", p, preview=preview) for p in named]
    if head in ("mv", "move", "move-item", "mi", "ren", "rename", "rename-item"):
        named = (
            operands
            if not head.endswith("-item")
            else _ps_operands(words[1:], ("-path", "-destination"))
        )
        if len(named) >= 2:
            return out + [
                PlannedChange("delete", named[0], preview=preview),
                PlannedChange("create", named[-1], preview=preview),
            ]
        return out
    if head in ("cp", "copy", "copy-item", "cpi", "install"):
        named = (
            operands if head != "copy-item" else _ps_operands(words[1:], ("-path", "-destination"))
        )
        return out + (
            [PlannedChange("create", named[-1], preview=preview)] if len(named) >= 2 else []
        )
    if head in ("touch", "mkdir", "md", "new-item", "ni"):
        named = operands if head not in ("new-item", "ni") else _ps_operands(words[1:], ("-path",))
        return out + [PlannedChange("create", p, preview=preview) for p in named]
    if head in ("tee", "out-file", "set-content", "add-content", "sc", "ac"):
        named = (
            operands
            if head == "tee"
            else _ps_operands(words[1:], ("-path", "-filepath", "-literalpath"))
        )
        return out + [PlannedChange("edit", p) for p in named]
    if head in ("sed", "perl") and any(
        w == "-i" or w.startswith("-i") or w == "-pi" for w in words[1:]
    ):
        return out + [PlannedChange("edit", p) for p in operands[1:]]
    if (
        head in ("pip", "pip3", "uv", "npm", "pnpm", "yarn", "cargo", "poetry", "gem", "go", "bun")
        and len(words) > 1
    ):
        verb = words[1].lower()
        if verb in ("install", "add", "i", "get") or (
            head == "uv" and verb == "pip" and words[2:3] == ["install"]
        ):
            return out + [PlannedChange("package_install", preview=preview)]
    return out


def _ps_operands(args: list[str], named: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("-"):
            if arg.lower() in named and i + 1 < len(args):
                out.append(args[i + 1])
                i += 2
                continue
            i += 1
            continue
        out.append(arg)
        i += 1
    return out


def planned_changes(tool_name: str, args: Mapping[str, Any]) -> list[PlannedChange]:
    """Every mutating effect a call would have (empty for read-only calls)."""
    lname = (tool_name or "").lower()
    if lname == "apply_patch":
        patch = args.get("input") or args.get("patch") or ""
        mapping = {"Update": "edit", "Add": "create", "Delete": "delete"}
        return [
            PlannedChange(mapping[k], p.strip())
            for k, p in _PATCH_FILE.findall(patch if isinstance(patch, str) else "")
        ]
    if lname in _EDIT_TOOLS:
        for key in _PATH_KEYS:
            value = args.get(key)
            if isinstance(value, str) and value:
                effect = "create" if lname in ("write", "write_file", "create_file") else "edit"
                return [PlannedChange(effect, value)]
        return []
    if lname in _SHELL_TOOLS:
        command = args.get("command") or args.get("cmd")
        if isinstance(command, list):
            command = " ".join(str(c) for c in command)
        if not isinstance(command, str) or not command.strip():
            return []
        shell = shell_of(lname)
        out: list[PlannedChange] = []
        for segment in command_segments(command):
            out.extend(_shell_segment_changes(segment, shell))
        return out
    return []


def shell_cwd(command: str, cwd: str | None, shell: str = "posix") -> str | None:
    """The directory a leading ``cd X &&`` moves to (relative to ``cwd``)."""
    segments = command_segments(command)
    current = cwd
    for segment in segments:
        words = split_command(segment, shell)
        if not words or words[0].lower() not in ("cd", "pushd", "set-location", "sl", "chdir"):
            break
        target = next((w for w in words[1:] if not w.startswith("-")), None)
        if target is None:
            break
        target = _unwrap(target)
        current = target if os.path.isabs(target) else os.path.join(current or ".", target)
    return current


# ---------------------------------------------------------------------------
# Firewall service
# ---------------------------------------------------------------------------


def _digest(*parts: Any) -> str:
    raw = "\x1f".join(str(p) for p in parts).encode("utf-8", "surrogatepass")
    return hashlib.sha256(raw).hexdigest()[:24]


class ChangeScopeFirewall:
    """Owner of change contracts and write authorizations (see module docstring)."""

    def __init__(
        self,
        runtime: Any = None,
        *,
        policy: EnforcementPolicy | None = None,
        clock: Callable[[], float] = time.time,
        case_insensitive: bool | None = None,
    ) -> None:
        self.runtime = runtime
        self.policy = policy or EnforcementPolicy.from_env()
        self._clock = clock
        self.fold = platform_case_insensitive() if case_insensitive is None else case_insensitive
        self._lock = threading.RLock()
        self._contracts: OrderedDict[tuple[str, str], ChangeContract] = OrderedDict()
        self._dependencies: OrderedDict[tuple[str, str, str], dict[str, str]] = OrderedDict()
        self._resources: OrderedDict[tuple[str, str], tuple[int, tuple[Any, ...]]] = OrderedDict()
        self._writes: OrderedDict[tuple[str, str], deque[str]] = OrderedDict()
        self._classified: OrderedDict[tuple[str, str, str], ScopeClass] = OrderedDict()
        self._seen_diff_facts: OrderedDict[str, None] = OrderedDict()
        self.counters: dict[str, int] = {
            "decisions": 0,
            "allowed": 0,
            "warned": 0,
            "blocked": 0,
            "would_block_observe": 0,
            "stale_reevaluated": 0,
            "stale_denied": 0,
            "dependency_expansions": 0,
            "writes_recorded": 0,
            "diff_unrelated": 0,
            "contracts_built": 0,
        }

    # -- inputs --------------------------------------------------------------

    def _snapshot(self, identity: WorkspaceIdentity) -> TaskSnapshot | None:
        runtime = self.runtime
        compiler = runtime.services.get("task_state") if runtime is not None else None
        if compiler is None:
            return None
        snap = compiler.snapshot(identity.workspace_id, identity.session_id)
        if snap is None:
            compiler.request_warm(identity.workspace_id, identity.session_id)
        return snap

    def _graph(self, identity: WorkspaceIdentity) -> Any:
        runtime = self.runtime
        adapter = runtime.services.get("code_graph") if runtime is not None else None
        if adapter is None:
            return None
        graph = adapter.ensure_workspace(identity)
        return graph if graph is not None and graph.ready else None

    # -- contract ------------------------------------------------------------

    def contract(
        self, identity: WorkspaceIdentity, snapshot: TaskSnapshot | None = None
    ) -> ChangeContract:
        """The current contract for an identity (rebuilt when its inputs changed)."""
        snap = snapshot if snapshot is not None else self._snapshot(identity)
        state = snap.state if snap is not None else None
        task_id = state.id if state is not None else None
        user_atoms = [
            a for a in (state.atoms if state is not None else ()) if a.origin == "user" and a.live
        ]
        mentions: list[Mention] = []
        locked = False
        permitted = set(DEFAULT_PERMITTED_EFFECTS)
        forbidden_effects: set[str] = set()
        for atom in user_atoms:
            found, locks = _atom_mentions(atom)
            mentions.extend(found)
            locked = locked or locks
            allow, deny = _effect_mentions(atom.text)
            permitted |= allow
            forbidden_effects |= deny
        permitted -= forbidden_effects
        latest: dict[str, Mention] = {}
        for mention in sorted(mentions, key=lambda m: m.ts):
            rel = self._mention_rel(mention.path, identity)
            if rel is None:
                continue
            latest[path_key(rel, self.fold)] = Mention(
                rel, mention.kind, mention.ts, mention.atom_id
            )
        expected = tuple(sorted({m.path for m in latest.values() if m.kind == "expect"}))
        forbidden = tuple(sorted({m.path for m in latest.values() if m.kind == "forbid"}))
        writes = self._recent_writes(identity, state)
        failing = self._evidence_paths(snap, identity.root)
        generated = self._generated(expected + writes + tuple(failing))
        key = (identity.workspace_id, identity.session_id)
        with self._lock:
            deps = dict(
                self._dependencies.get(
                    (identity.workspace_id, identity.session_id, task_id or ""), {}
                )
            )
        for rel, reason in failing.items():
            deps.setdefault(rel, reason)
        if locked:
            mode = ContractMode.FROZEN
        elif expected or failing:
            mode = ContractMode.ACTIVE
        else:
            mode = ContractMode.LEARNING
        goal_atom = state.goal if state is not None else None
        fingerprint = _digest(
            task_id,
            state.revision if state is not None else 0,
            expected,
            forbidden,
            sorted(deps.items()),
            generated,
            sorted(permitted),
            sorted(forbidden_effects),
            mode.value,
            writes,
        )
        now = self._clock()
        with self._lock:
            current = self._contracts.get(key)
            if current is not None and current.fingerprint == fingerprint:
                self._contracts.move_to_end(key)
                return current
            same_task = current is not None and current.task_id == task_id
            contract = ChangeContract(
                workspace_id=identity.workspace_id,
                session_id=identity.session_id,
                task_id=task_id,
                task_revision=state.revision if state is not None else 0,
                revision=(current.revision + 1) if current is not None else 1,
                goal=(goal_atom.text if goal_atom is not None else "")[:280],
                expected_paths=expected,
                dependency_paths=tuple(sorted(deps)),
                generated_paths=generated,
                forbidden_paths=forbidden,
                permitted_effects=tuple(sorted(permitted)),
                forbidden_effects=tuple(sorted(forbidden_effects)),
                mode=mode,
                created_at=current.created_at if same_task and current is not None else now,
                approved_protected=tuple(p for p in expected if hard_protection(p, fold=self.fold)),
                recent_writes=writes,
                dependency_reasons=deps,
                fingerprint=fingerprint,
            )
            self._contracts[key] = contract
            self._contracts.move_to_end(key)
            while len(self._contracts) > _CONTRACTS_LIMIT:
                self._contracts.popitem(last=False)
            self.counters["contracts_built"] += 1
        return contract

    def _mention_rel(self, raw: str, identity: WorkspaceIdentity) -> str | None:
        if any(ch in raw for ch in "*?["):
            return raw.replace("\\", "/").lstrip("./") or None
        if "/" not in raw and "\\" not in raw:
            return raw  # bare basename pattern
        root = identity.root
        if root is None:
            return None
        canon = canonicalize(raw, root=root, cwd=root, case_insensitive=self.fold)
        if canon.rel is not None:
            return canon.rel + ("/" if canon.is_dir or raw.endswith(("/", "\\")) else "")
        return canon.resolved  # an explicitly named external path

    def _recent_writes(self, identity: WorkspaceIdentity, state: Any) -> tuple[str, ...]:
        out: list[str] = []
        with self._lock:
            recorded = list(self._writes.get((identity.workspace_id, identity.session_id), ()))
            task_id = state.id if state is not None else ""
            for resource in list(state.resources if state is not None else ()) + recorded:
                rel = resource[5:] if resource.startswith("path:") else None
                if not rel:
                    continue
                cls = self._classified.get(
                    (identity.workspace_id, task_id, path_key(rel, self.fold))
                )
                if cls in (ScopeClass.UNRELATED, ScopeClass.FORBIDDEN):
                    continue  # a drifted write never legitimizes itself
                if rel not in out:
                    out.append(rel)
        return tuple(out[-_WRITES_LIMIT:])

    def _evidence_paths(self, snapshot: TaskSnapshot | None, root: str | None) -> dict[str, str]:
        """Files named by live failing test/compiler evidence (traceable reason)."""
        out: dict[str, str] = {}
        if snapshot is None:
            return out
        for fact in snapshot.evidence:
            if fact.polarity != "negative" or fact.status not in ("ACTIVE", "CONTRADICTED"):
                continue
            if fact.predicate not in ("error.signature", "test.status", "command.exit"):
                continue
            text = f"{fact.subject} {fact.value_json}"
            for mention in path_mentions(text.replace("\\\\", "/")):
                rel = mention.replace("\\", "/").lstrip("./")
                drive = re.match(r"^[A-Za-z]:/", rel)
                if ":" in rel[2 if drive else 0 :]:
                    rel = rel[: rel.index(":", 2 if drive else 0)]
                if root is not None and (drive or rel.startswith("/")):
                    inside = _relative(os.path.realpath(rel), os.path.realpath(root), self.fold)
                    if inside is None:
                        continue
                    rel = inside
                if "/" in rel or _EXT.search(rel):
                    out.setdefault(rel, f"scope.dependency.failing_evidence:{fact.id}")
        return out

    def _generated(self, anchors: Iterable[str]) -> tuple[str, ...]:
        out: set[str] = set()
        for rel in anchors:
            directory, _, name = rel.replace("\\", "/").rpartition("/")
            prefix = f"{directory}/" if directory else ""
            for lock in _LOCKFILES.get(name.lower(), ()):
                out.add(prefix + lock)
            if _TEST_NAME.search(rel):
                out.add(prefix + "__snapshots__/")
                out.add(prefix + "snapshots/")
        return tuple(sorted(out))

    # -- classification ------------------------------------------------------

    def classify(
        self,
        identity: WorkspaceIdentity,
        raw_path: str,
        *,
        cwd: str | None = None,
        contract: ChangeContract | None = None,
        effect: str = "edit",
    ) -> ScopeVerdict:
        """Scope class of a write to ``raw_path`` (no side effects except expansions)."""
        root = identity.root if identity.can_authorize_paths() else None
        canon = canonicalize(raw_path, root=root, cwd=cwd or root, case_insensitive=self.fold)
        target_for_protection = canon.rel if canon.rel is not None else canon.resolved
        lexical_target = canon.lexical_rel or raw_path
        protection = hard_protection(target_for_protection, fold=self.fold) or hard_protection(
            lexical_target, fold=self.fold
        )
        if root is None:
            return ScopeVerdict(
                ScopeClass.UNRELATED, "scope.untrusted_workspace", canon.rel, protection=protection
            )
        current = contract or self.contract(identity)
        rel = canon.rel
        if protection is not None:
            approved = (
                protection != "git_internal"
                and rel is not None
                and matches_any(current.approved_protected, rel, self.fold) is not None
            )
            if not approved:
                return ScopeVerdict(
                    ScopeClass.FORBIDDEN,
                    f"scope.hard_protected.{protection}",
                    rel,
                    trace=protection,
                    protection=protection,
                )
        if rel is not None:
            hit = matches_any(current.forbidden_paths, rel, self.fold)
            if hit is not None:
                return ScopeVerdict(ScopeClass.FORBIDDEN, "scope.user_forbidden", rel, trace=hit)
        if effect in current.forbidden_effects:
            return ScopeVerdict(
                ScopeClass.FORBIDDEN, f"scope.forbidden_effect.{effect}", rel, trace=effect
            )
        if rel is None:
            named = canon.resolved
            if any(_cmp(named, self.fold) == _cmp(p, self.fold) for p in current.expected_paths):
                return ScopeVerdict(
                    ScopeClass.IN_SCOPE, "scope.outside_worktree.user_named", None, trace=named
                )
            code = "scope.symlink_escape" if canon.symlink_escape else "scope.outside_worktree"
            return ScopeVerdict(ScopeClass.UNRELATED, code, None, trace="unc" if canon.unc else "")
        hit = matches_any(current.expected_paths, rel, self.fold)
        if hit is not None:
            return ScopeVerdict(ScopeClass.IN_SCOPE, "scope.expected", rel, trace=hit)
        hit = matches_any(current.recent_writes, rel, self.fold)
        if hit is not None:
            return ScopeVerdict(ScopeClass.IN_SCOPE, "scope.recent_write", rel, trace=hit)
        hit = matches_any(current.generated_paths, rel, self.fold)
        if hit is not None:
            return ScopeVerdict(ScopeClass.GENERATED_EFFECT, "scope.generated", rel, trace=hit)
        hit = matches_any(current.dependency_paths, rel, self.fold)
        if hit is not None:
            reason = current.dependency_reasons.get(hit, "scope.dependency")
            code, _, trace = reason.partition(":")
            return ScopeVerdict(ScopeClass.DEPENDENCY_SCOPE, code, rel, trace=trace or hit)
        anchors = tuple(
            p for p in current.expected_paths + current.recent_writes if not p.endswith("/")
        )
        expansion = self._dependency_reason(identity, rel, anchors)
        if expansion is not None:
            code, trace = expansion
            self._expand(identity, current, rel, f"{code}:{trace}")
            return ScopeVerdict(ScopeClass.DEPENDENCY_SCOPE, code, rel, trace=trace)
        if current.mode is ContractMode.LEARNING:
            return ScopeVerdict(ScopeClass.IN_SCOPE, "scope.learning", rel)
        code = (
            "scope.frozen_unrelated" if current.mode is ContractMode.FROZEN else "scope.unrelated"
        )
        return ScopeVerdict(ScopeClass.UNRELATED, code, rel)

    def _expand(
        self, identity: WorkspaceIdentity, contract: ChangeContract, rel: str, reason: str
    ) -> None:
        key = (identity.workspace_id, identity.session_id, contract.task_id or "")
        with self._lock:
            deps = self._dependencies.setdefault(key, {})
            if rel not in deps:
                deps[rel] = reason
                self.counters["dependency_expansions"] += 1
            self._dependencies.move_to_end(key)
            while len(self._dependencies) > _CONTRACTS_LIMIT:
                self._dependencies.popitem(last=False)

    def _dependency_reason(
        self, identity: WorkspaceIdentity, rel: str, anchors: Sequence[str]
    ) -> tuple[str, str] | None:
        """A traceable reason ``rel`` belongs to the change, or None."""
        if not anchors:
            return None
        fold = self.fold
        target_key = path_key(rel, fold)
        anchor_keys = {path_key(a, fold): a for a in anchors}
        name = target_key.rsplit("/", 1)[-1]
        directory = target_key.rsplit("/", 1)[0] if "/" in target_key else ""
        # Required platform/config: a build/package file on an anchor's path.
        if name in _CONFIG_FILES:
            for key, anchor in anchor_keys.items():
                anchor_dir = key.rsplit("/", 1)[0] if "/" in key else ""
                if (
                    not directory
                    or anchor_dir == directory
                    or anchor_dir.startswith(directory + "/")
                ):
                    return "scope.dependency.config", anchor
        # Related test: test file named after an anchor.
        if _TEST_NAME.search(target_key):
            stem = _stem(target_key)
            for key, anchor in anchor_keys.items():
                if not _TEST_NAME.search(key) and _stem(key) == stem and stem:
                    return "scope.dependency.related_test", anchor
        graph = self._graph(identity)
        if graph is not None:
            found = self._graph_link(graph, rel, anchors)
            if found is not None:
                return found
        return self._lexical_link(identity, rel, anchors)

    def _graph_link(self, graph: Any, rel: str, anchors: Sequence[str]) -> tuple[str, str] | None:
        with graph.lock:
            target = graph.normalize_path(rel)
            if target is None:
                return None
            record = graph.files.get(target)
            imports = graph.resolve_imports(record) if record is not None else set()
            importers = graph.importers.get(target, set())
            target_defs = {d.qualname.rsplit(".", 1)[-1] for d in record.defs} if record else set()
            for anchor in anchors:
                a = graph.normalize_path(anchor)
                if a is None:
                    continue
                if a in imports or a in importers:
                    kind = (
                        "scope.dependency.related_test"
                        if _TEST_NAME.search(target)
                        else "scope.dependency.import"
                    )
                    return kind, a
                arecord = graph.files.get(a)
                if arecord is None:
                    continue
                anchor_defs = {d.qualname.rsplit(".", 1)[-1] for d in arecord.defs}
                for name in target_defs:
                    if any(p == a for p, _ in graph.callers.get(name, ())):
                        return "scope.dependency.interface", f"{a}->{name}"
                for name in anchor_defs:
                    if any(p == target for p, _ in graph.callers.get(name, ())) or any(
                        p == target for p, _ in graph.subclasses.get(name, ())
                    ):
                        return "scope.dependency.interface", f"{target}->{name}"
        return None

    def _lexical_link(
        self, identity: WorkspaceIdentity, rel: str, anchors: Sequence[str]
    ) -> tuple[str, str] | None:
        """Direct import/include link read from source (graph not indexed yet)."""
        root = identity.root
        if root is None:
            return None

        def read(path: str) -> str:
            full = os.path.join(root, path)
            try:
                if os.path.getsize(full) > _SOURCE_READ_LIMIT:
                    return ""
                with open(full, encoding="utf-8", errors="replace") as handle:
                    return handle.read(_SOURCE_READ_LIMIT)
            except OSError:
                return ""

        def imports_of(source: str, module: str) -> bool:
            if not module:
                return False
            pattern = re.compile(
                r"^\s*(?:from\s+[\w.]*\b"
                + re.escape(module)
                + r"\b|import\s+[\w.,\s]*\b"
                + re.escape(module)
                + r"\b|#\s*include\s+[\"<][^\">]*\b"
                + re.escape(module)
                + r"\b|use\s+[\w:]*\b"
                + re.escape(module)
                + r"\b|.*\brequire\(\s*['\"][^'\"]*\b"
                + re.escape(module)
                + r"\b|.*\bfrom\s+['\"][^'\"]*\b"
                + re.escape(module)
                + r"\b|mod\s+"
                + re.escape(module)
                + r"\b)",
                re.MULTILINE,
            )
            return bool(pattern.search(source))

        target_module = rel.rsplit("/", 1)[-1].split(".", 1)[0]
        target_source = read(rel)
        for anchor in anchors:
            anchor_module = anchor.rsplit("/", 1)[-1].split(".", 1)[0]
            if target_source and imports_of(target_source, anchor_module):
                kind = (
                    "scope.dependency.related_test"
                    if _TEST_NAME.search(rel)
                    else "scope.dependency.import"
                )
                return kind, anchor
            anchor_source = read(anchor)
            if anchor_source and imports_of(anchor_source, target_module):
                return "scope.dependency.import", anchor
        return None

    # -- revisions -----------------------------------------------------------

    def _stat(self, identity: WorkspaceIdentity, rel: str | None) -> tuple[Any, ...]:
        if rel is None or identity.root is None:
            return ()
        try:
            st = os.lstat(os.path.join(identity.root, rel))
        except OSError:
            return ("missing",)
        return (st.st_mtime_ns, st.st_size, st.st_ino)

    def resource_revision(self, identity: WorkspaceIdentity, rel: str | None) -> int:
        """Per-resource revision: bumped on recorded writes and observed fs changes."""
        if rel is None:
            return 0
        key = (identity.workspace_id, path_key(rel, self.fold))
        stat = self._stat(identity, rel)
        with self._lock:
            revision, last = self._resources.get(key, (0, stat))
            if last != stat:
                revision += 1
            self._resources[key] = (revision, stat)
            self._resources.move_to_end(key)
            while len(self._resources) > _RESOURCES_LIMIT:
                self._resources.popitem(last=False)
            return revision

    def task_revision(self, identity: WorkspaceIdentity) -> int:
        snap = self._snapshot(identity)
        return snap.state.revision if snap is not None else 0

    def record_write(self, workspace_id: str, session_id: str, rel: str) -> None:
        """An executed write (FILE_CHANGE): bumps the resource revision."""
        clean = rel[5:] if rel.startswith("path:") else rel
        key = (workspace_id, path_key(clean, self.fold))
        with self._lock:
            revision, stat = self._resources.get(key, (0, ()))
            self._resources[key] = (revision + 1, ("written", time.monotonic_ns()))
            writes = self._writes.setdefault(
                (workspace_id, session_id), deque(maxlen=_WRITES_LIMIT)
            )
            if clean not in writes:
                writes.append(clean)
            self._writes.move_to_end((workspace_id, session_id))
            while len(self._writes) > _CONTRACTS_LIMIT:
                self._writes.popitem(last=False)
            self.counters["writes_recorded"] += 1

    def _remember_class(
        self, identity: WorkspaceIdentity, task_id: str | None, rel: str | None, cls: ScopeClass
    ) -> None:
        if rel is None:
            return
        key = (identity.workspace_id, task_id or "", path_key(rel, self.fold))
        with self._lock:
            previous = self._classified.get(key)
            if (
                previous
                in (ScopeClass.IN_SCOPE, ScopeClass.DEPENDENCY_SCOPE, ScopeClass.GENERATED_EFFECT)
                and cls is ScopeClass.UNRELATED
            ):
                return
            self._classified[key] = cls
            self._classified.move_to_end(key)
            while len(self._classified) > _CLASSIFIED_LIMIT:
                self._classified.popitem(last=False)

    # -- authorization -------------------------------------------------------

    def _verdict_for(
        self, scope: ScopeVerdict, contract: ChangeContract, *, enforceable: bool, effect: str
    ) -> tuple[Verdict, str]:
        mode = self.policy.change_scope
        if scope.scope is ScopeClass.FORBIDDEN:
            if enforceable and mode is not Enforcement.OBSERVE:
                return Verdict.BLOCK, scope.reason_code
            return Verdict.WARN, scope.reason_code
        if scope.reason_code == "scope.untrusted_workspace":
            return Verdict.WARN, scope.reason_code
        if effect in EXPLICIT_EFFECTS and effect not in contract.permitted_effects:
            return Verdict.WARN, f"scope.effect_not_permitted.{effect}"
        if scope.scope is ScopeClass.UNRELATED:
            if mode is Enforcement.OBSERVE:
                return Verdict.ALLOW, scope.reason_code
            if mode is Enforcement.PROTECT and enforceable and contract.mode is ContractMode.FROZEN:
                return Verdict.BLOCK, scope.reason_code
            return Verdict.WARN, scope.reason_code
        if contract.mode is ContractMode.FROZEN and scope.scope is ScopeClass.DEPENDENCY_SCOPE:
            return Verdict.WARN, f"{scope.reason_code}.frozen"
        return Verdict.ALLOW, scope.reason_code

    def evaluate(
        self,
        identity: WorkspaceIdentity,
        change: PlannedChange,
        *,
        cwd: str | None = None,
        enforceable: bool = False,
    ) -> ScopeDecision:
        """One scope decision with a revision-pinned authorization (no verification)."""
        snap = self._snapshot(identity)
        contract = self.contract(identity, snap)
        if change.path is not None:
            scope = self.classify(
                identity, change.path, cwd=cwd, contract=contract, effect=change.effect
            )
        elif change.effect in contract.forbidden_effects:
            scope = ScopeVerdict(
                ScopeClass.FORBIDDEN,
                f"scope.forbidden_effect.{change.effect}",
                None,
                trace=change.effect,
            )
        elif not identity.can_authorize_paths():
            scope = ScopeVerdict(ScopeClass.UNRELATED, "scope.untrusted_workspace", None)
        else:
            scope = ScopeVerdict(ScopeClass.IN_SCOPE, f"scope.effect.{change.effect}", None)
        if change.preview and scope.scope is not ScopeClass.FORBIDDEN:
            scope = ScopeVerdict(
                ScopeClass.IN_SCOPE, "scope.preview", scope.rel, trace=change.effect
            )
        # A dependency expansion just revised the contract: pin the new revision.
        contract = self.contract(identity, snap)
        verdict, reason = (
            (Verdict.ALLOW, "scope.preview")
            if scope.reason_code == "scope.preview"
            else self._verdict_for(scope, contract, enforceable=enforceable, effect=change.effect)
        )
        task_revision = snap.state.revision if snap is not None else 0
        resource_revision = self.resource_revision(identity, scope.rel)
        authorization = None
        runtime = self.runtime
        if verdict is not Verdict.BLOCK and identity.can_authorize_paths() and runtime is not None:
            authorization = runtime.authorize_action(
                identity,
                task_revision=task_revision,
                scope_revision=contract.revision,
                resource_revision=resource_revision,
                decision=AuthorizationDecision.ALLOW
                if verdict is Verdict.ALLOW
                else AuthorizationDecision.WARN,
                ttl_s=AUTHORIZATION_TTL_S,
            )
        self._remember_class(identity, contract.task_id, scope.rel, scope.scope)
        return ScopeDecision(
            verdict=verdict,
            reason_code=reason,
            scope=scope.scope,
            rel=scope.rel,
            trace=scope.trace,
            enforceable=enforceable,
            mode=contract.mode,
            task_revision=task_revision,
            scope_revision=contract.revision,
            resource_revision=resource_revision,
            authorization=authorization,
            effect=change.effect,
        )

    def verify(self, identity: WorkspaceIdentity, decision: ScopeDecision) -> bool:
        """True when the decision's authorization still matches every current revision."""
        auth = decision.authorization
        if auth is None:
            return False
        contract = self.contract(identity)
        return auth.is_valid(
            workspace_id=identity.workspace_id,
            session_id=identity.session_id,
            task_revision=self.task_revision(identity),
            scope_revision=contract.revision,
            resource_revision=self.resource_revision(identity, decision.rel),
        )

    def authorize(
        self,
        identity: WorkspaceIdentity,
        change: PlannedChange,
        *,
        cwd: str | None = None,
        enforceable: bool = False,
    ) -> ScopeDecision:
        """Evaluate, then verify immediately before returning (see module docstring)."""
        decision = self.evaluate(identity, change, cwd=cwd, enforceable=enforceable)
        if decision.verdict is Verdict.ALLOW and decision.authorization is not None:
            if not self.verify(identity, decision):
                with self._lock:
                    self.counters["stale_reevaluated"] += 1
                retry = self.evaluate(identity, change, cwd=cwd, enforceable=enforceable)
                if retry.verdict is Verdict.ALLOW and not self.verify(identity, retry):
                    with self._lock:
                        self.counters["stale_denied"] += 1
                    stale = (
                        Verdict.BLOCK
                        if enforceable and self.policy.change_scope is not Enforcement.OBSERVE
                        else Verdict.WARN
                    )
                    decision = ScopeDecision(
                        verdict=stale,
                        reason_code="scope.stale_revision",
                        scope=retry.scope,
                        rel=retry.rel,
                        trace="revision changed during authorization",
                        enforceable=enforceable,
                        mode=retry.mode,
                        task_revision=retry.task_revision,
                        scope_revision=retry.scope_revision,
                        resource_revision=retry.resource_revision,
                        authorization=None,
                        effect=change.effect,
                        reevaluated=True,
                    )
                else:
                    decision = replace(retry, reevaluated=True)
        self._count(decision)
        self._emit(identity, decision)
        return decision

    def _count(self, decision: ScopeDecision) -> None:
        with self._lock:
            self.counters["decisions"] += 1
            if decision.verdict is Verdict.BLOCK:
                self.counters["blocked"] += 1
            elif decision.verdict is Verdict.WARN:
                self.counters["warned"] += 1
                if decision.scope is ScopeClass.FORBIDDEN:
                    self.counters["would_block_observe"] += 1
            else:
                self.counters["allowed"] += 1

    def _emit(
        self, identity: WorkspaceIdentity, decision: ScopeDecision, *, source: str = "authorize"
    ) -> None:
        runtime = self.runtime
        if runtime is None:
            return
        meta = {
            "source": source,
            "verdict": decision.verdict.value,
            "scope": decision.scope.value,
            "reason_code": decision.reason_code,
            "effect": decision.effect,
            "enforceable": decision.enforceable,
            "mode": decision.mode.value,
            "task_revision": decision.task_revision,
            "scope_revision": decision.scope_revision,
            "resource_revision": decision.resource_revision,
        }
        try:
            event = make_event(
                workspace_id=identity.workspace_id,
                session_id=identity.session_id,
                kind=EventKind.SCOPE_DECISION,
                origin=Origin.SYSTEM,
                source_id=f"scope:{_digest(decision.rel or decision.effect, source)}",
                index=decision.scope_revision * 100_000 + decision.resource_revision,
                payload={**meta, "rel": decision.rel},
                resource_id=f"path:{decision.rel}" if decision.rel else None,
                revision=decision.scope_revision,
                metadata=meta,
            )
            runtime.submit([EventEnvelope(event=event)])
        except Exception as exc:  # noqa: BLE001 - audit is best-effort
            log.debug("scope decision event failed: %s", describe_exception(exc))

    # -- after-edit checks ---------------------------------------------------

    def observe_change(
        self, identity: WorkspaceIdentity, change: PlannedChange, *, cwd: str | None = None
    ) -> ScopeDecision:
        """Classify an already-executed change (proxy path: observe only)."""
        decision = self.evaluate(identity, change, cwd=cwd, enforceable=False)
        self._count(decision)
        self._emit(identity, decision, source="observed")
        return decision

    def check_diff(self, identity: WorkspaceIdentity) -> list[ScopeDecision]:
        """Compare the latest ``git.changed_paths`` evidence against the contract."""
        snap = self._snapshot(identity)
        if snap is None or not identity.can_authorize_paths():
            return []
        out: list[ScopeDecision] = []
        for fact in snap.evidence:
            if fact.predicate != "git.changed_paths" or fact.status != "ACTIVE":
                continue
            with self._lock:
                if fact.id in self._seen_diff_facts:
                    continue
                self._seen_diff_facts[fact.id] = None
                while len(self._seen_diff_facts) > _CLASSIFIED_LIMIT:
                    self._seen_diff_facts.popitem(last=False)
            value = fact.value if isinstance(fact.value, dict) else {}
            for raw in value.get("paths") or ():
                decision = self.evaluate(
                    identity, PlannedChange("edit", str(raw)), cwd=identity.root
                )
                if decision.scope in (ScopeClass.UNRELATED, ScopeClass.FORBIDDEN):
                    with self._lock:
                        self.counters["diff_unrelated"] += 1
                    self._emit(identity, decision, source="git_diff")
                    out.append(decision)
        return out

    def on_commit(self, events: list[Any], _runtime: Any) -> None:
        """Commit listener: executed writes (FILE_CHANGE) bump resource revisions."""
        for event in events:
            if event.kind == EventKind.FILE_CHANGE.value and (event.resource_id or "").startswith(
                "path:"
            ):
                self.record_write(event.workspace_id, event.session_id, str(event.resource_id))

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "policy": self.policy.to_dict(),
                "contracts": len(self._contracts),
                "case_insensitive": self.fold,
                "counters": dict(self.counters),
            }


def describe(decision: ScopeDecision) -> str:
    """Compact, reasoned diagnostic for the agent's normal tool context."""
    target = decision.rel or "this path"
    code = decision.reason_code
    if code.startswith("scope.hard_protected."):
        kind = code.rsplit(".", 1)[-1].replace("_", " ")
        return (
            f"{target} is hard-protected ({kind}); writes there need the user's explicit approval"
        )
    if code == "scope.user_forbidden":
        return f"{target} is excluded by the user's instruction ({decision.trace})"
    if code.startswith("scope.forbidden_effect."):
        return f"the user ruled out this operation ({decision.trace or decision.effect})"
    if code.startswith("scope.effect_not_permitted."):
        effect = code.rsplit(".", 1)[-1].replace("_", " ")
        return f"{effect} was not requested by the user; confirm before running it"
    if code == "scope.stale_revision":
        return f"the task/scope changed while this write to {target} was being checked; re-plan and retry"
    if code in ("scope.unrelated", "scope.frozen_unrelated"):
        lock = " (the user locked the scope)" if code == "scope.frozen_unrelated" else ""
        return (
            f"{target} is outside the task's change scope{lock}: no import, interface, failing-test, "
            "config or related-test link to the files in scope"
        )
    if code in ("scope.outside_worktree", "scope.symlink_escape"):
        how = "resolves through a symlink to" if code == "scope.symlink_escape" else "is"
        return f"this path {how} a location outside the registered worktree"
    if code.endswith(".frozen"):
        return (
            f"{target} is a dependency of the locked scope ({decision.trace}); confirm it is needed"
        )
    return f"{target}: {code}"
