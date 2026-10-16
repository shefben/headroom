"""Safe learned Workflow Macro Compiler (unified-21 #20).

Observation (proxy path, every client; observe-only):
    Each request's history is cut into *segments* — the tool calls between two
    authentic user turns. A closed segment of 2–8 successful, normalizable
    calls with no failure, no host contract/scope verdict, no following user
    correction is one clean observation of a template. Calls normalize to
    typed families (``read.file``, ``search.grep``, ``git.status``,
    ``test.pytest`` ...) plus typed parameter slots; credentials, URLs and
    untrusted script strings are never generalized (such steps keep no argv at
    all). Observations persist per workspace (segment id is the key, so a
    replayed history never double-counts).

Promotion:
    ≥3 occurrences, ≥90% success, ≥2 sessions or 3 clean repeats in one,
    no contract/scope violation, and an estimated reduction of ≥2
    model-mediated tool turns. Only **read-only or verification-only** macros
    auto-promote; mutating, deletion, network, package-install, deployment,
    credential and elevation steps never promote and never execute. Macro ids
    are deterministic (workspace + template). At most 32 active macros per
    workspace, 8 steps per macro, 120 s per macro (smaller per-step
    deadlines). A macro is versioned and is invalidated when a step's
    executable, the runner configuration, the enforcement policy or the
    host tool schemas change; it must then be re-observed.

Execution (``headroom_workflow(macro_id, typed_inputs, dry_run)``):
    A local tool registered in the existing :data:`tool_catalog.REGISTRY`
    and exposed by the Headroom MCP server. Only a genuine host permission
    boundary can start it: Claude Code's pre-tool hook validates the call and
    issues a revision-pinned single-use ticket; the MCP call (which runs only
    after the host's own permission check) must present a matching ticket.
    Every step is re-checked immediately before it runs — fresh task/scope
    revision compare, current ToolContract decision, ChangeScope
    authorization — and runs as direct argv with ``shell=False`` in its own
    process group under a deadline, with bounded output, exit and side-effect
    verification, stopping at the first failure. Verification steps take
    their targets from the current :class:`~headroom.agent.test_impact.VerificationPlan`
    unless explicit tests are passed. Each step is ingested once through the
    normal event path (so evidence/TaskState update before the next action).
    A missing tool, hook or executable disables the macro and the result lists
    the ordinary steps to run instead.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
import re
import shlex
import statistics
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from headroom.agent.interfaces import WORKFLOW_TOOL_NAMES, ToolResultEnvelope
from headroom.log_safety import describe_exception

log = logging.getLogger(__name__)

FEATURE_WORKFLOW_MACROS = "agent_workflow_macros"
WORKFLOW_TOOL = "headroom_workflow"
WORKFLOW_MCP_TOOL = "mcp__headroom__headroom_workflow"
STAGE_OBSERVE = "agent_workflow_observe"

MIN_STEPS = 2
MAX_STEPS = 8
MIN_OCCURRENCES = 3
MIN_SUCCESS_RATE = 0.90
MIN_TURNS_SAVED = 2
MAX_ACTIVE_PER_WORKSPACE = 32
MAX_MACRO_SECONDS = 120.0
MAX_STEP_SECONDS = 90.0
TICKET_TTL_S = 300.0
STEP_OUTPUT_PREVIEW = 1500
STEP_OUTPUT_EXTERNALIZE = 4000
MAX_READ_BYTES = 2 * 1024 * 1024
MAX_GREP_FILES = 2000
MAX_GLOB_RESULTS = 500
_SEGMENT_MESSAGES = 400
_SEEN_LIMIT = 16384
_OBS_PER_MACRO = 32
_MACROS_PER_WORKSPACE = 256

_TABLES = (
    "CREATE TABLE IF NOT EXISTS workflow_macros ("
    " workspace_id TEXT NOT NULL, macro_id TEXT NOT NULL, spec TEXT NOT NULL,"
    " updated_at REAL NOT NULL, PRIMARY KEY (workspace_id, macro_id))",
    "CREATE TABLE IF NOT EXISTS workflow_observations ("
    " segment_id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL, macro_id TEXT NOT NULL,"
    " ts REAL NOT NULL)",
)

WORKFLOW_TOOL_SCHEMA: dict[str, Any] = {
    "name": WORKFLOW_TOOL,
    "description": (
        "Run one of Headroom's learned workflows for this workspace (read-only or "
        "verification-only steps, e.g. git status → read → targeted tests) in a single "
        "call instead of several tool turns. Eligible workflow ids are listed in the "
        "<headroom_verification> context block. Every step is re-validated right before "
        "it runs and the workflow stops at the first failure. Call with dry_run=true "
        "(and optionally no macro_id) to list or preview without running anything. Tests "
        "default to the current verification plan unless typed_inputs.tests is given."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "macro_id": {"type": "string", "description": "Workflow id (wf_...)."},
            "typed_inputs": {
                "type": "object",
                "description": "Values for the workflow's typed inputs (see its listing).",
            },
            "dry_run": {
                "type": "boolean",
                "description": "Validate and show the compiled steps without running them.",
            },
        },
        "required": ["macro_id"],
    },
}


class Safety(str, Enum):
    READ_ONLY = "read_only"
    VERIFICATION = "verification"
    MUTATING = "mutating"
    DELETION = "deletion"
    NETWORK = "network"
    INSTALL = "package_install"
    DEPLOY = "deployment"
    CREDENTIAL = "credential"
    ELEVATION = "elevation"
    UNSUPPORTED = "unsupported"


AUTO_PROMOTABLE = frozenset({Safety.READ_ONLY.value, Safety.VERIFICATION.value})
_SEVERITY = {
    Safety.READ_ONLY.value: 0,
    Safety.VERIFICATION.value: 1,
    Safety.MUTATING.value: 2,
    Safety.DELETION.value: 3,
    Safety.INSTALL.value: 3,
    Safety.NETWORK.value: 4,
    Safety.DEPLOY.value: 5,
    Safety.CREDENTIAL.value: 6,
    Safety.ELEVATION.value: 6,
    Safety.UNSUPPORTED.value: 7,
}


class MacroStatus(str, Enum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    MANUAL_ONLY = "manual_only"  # observed, but its safety class never auto-promotes
    INVALIDATED = "invalidated"
    RETIRED = "retired"
    DISABLED = "disabled"


def _digest(*parts: Any) -> str:
    raw = "\x1f".join(str(p) for p in parts).encode("utf-8", "surrogatepass")
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Normalization: tool call -> typed step template + slot values
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotSpec:
    name: str
    type: str  # path | paths | pattern | glob | tests | int
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.type, "required": self.required}


@dataclass(frozen=True)
class StepTemplate:
    kind: str  # read | glob | grep | exec | verify | edit | other
    family: str
    tool: str
    argv: tuple[str, ...] = ()
    slots: tuple[SlotSpec, ...] = ()
    safety: str = Safety.UNSUPPORTED.value
    framework: str = ""

    def key(self) -> str:
        return json.dumps(
            [self.kind, self.family, list(self.argv), [s.to_dict() for s in self.slots]],
            separators=(",", ":"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "family": self.family,
            "tool": self.tool,
            "argv": list(self.argv),
            "slots": [s.to_dict() for s in self.slots],
            "safety": self.safety,
            "framework": self.framework,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StepTemplate:
        return cls(
            kind=str(value.get("kind") or "other"),
            family=str(value.get("family") or ""),
            tool=str(value.get("tool") or ""),
            argv=tuple(str(a) for a in value.get("argv") or ()),
            slots=tuple(
                SlotSpec(str(s["name"]), str(s["type"]), bool(s.get("required", True)))
                for s in value.get("slots") or ()
            ),
            safety=str(value.get("safety") or Safety.UNSUPPORTED.value),
            framework=str(value.get("framework") or ""),
        )

    def describe(self, defaults: Mapping[str, Any] | None = None) -> str:
        defaults = defaults or {}
        if self.kind == "read":
            return f"read {defaults.get('path') or '{path}'}"
        if self.kind == "glob":
            return f"glob {defaults.get('pattern') or '{pattern}'}"
        if self.kind == "grep":
            return f"grep {defaults.get('pattern') or '{pattern}'}"
        if self.kind in ("exec", "verify"):
            words = [_fill_display(a, defaults) for a in self.argv]
            if words and (os.path.isabs(words[0]) or "/" in words[0]):
                words[0] = os.path.basename(words[0])
            return " ".join(words)
        return self.family


def _fill_display(token: str, values: Mapping[str, Any]) -> str:
    m = re.fullmatch(r"\{(\w+)\}", token)
    if not m:
        return token
    value = values.get(m.group(1))
    if value is None:
        return token
    if isinstance(value, list | tuple):
        return " ".join(str(v) for v in value)
    return str(value)


@dataclass(frozen=True)
class NormalizedStep:
    template: StepTemplate
    values: Mapping[str, Any]


_URL = re.compile(r"(?i)^(?:[a-z][\w+.-]*://|[\w.-]+@[\w.-]+:)|://")
_CREDENTIAL = re.compile(
    r"(?i)(?:token|passw(?:or)?d|secret|api[_-]?key|bearer|authorization|credential|private[_-]?key)"
)
_SHELL_META = re.compile(r"[|&;<>`$*?~\n\r]|\$\(")
_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_ELEVATION = frozenset({"sudo", "doas", "runas", "su", "pkexec", "gsudo"})
_GIT_READ = frozenset(
    {"status", "diff", "log", "show", "rev-parse", "ls-files", "blame", "describe", "shortlog"}
)
_GIT_NETWORK = frozenset({"fetch", "pull", "push", "clone", "ls-remote", "remote", "submodule"})
_DELETE = frozenset({"rm", "del", "rmdir", "erase", "remove-item", "rd", "shred", "unlink"})
_MUTATE = frozenset(
    {
        "mv",
        "cp",
        "mkdir",
        "touch",
        "chmod",
        "chown",
        "ln",
        "tee",
        "set-content",
        "add-content",
        "new-item",
        "copy-item",
        "move-item",
        "rename-item",
        "patch",
        "dd",
        "truncate",
    }
)
_NETWORK = frozenset(
    {
        "curl",
        "wget",
        "ssh",
        "scp",
        "sftp",
        "rsync",
        "nc",
        "ncat",
        "telnet",
        "ftp",
        "invoke-webrequest",
        "iwr",
        "invoke-restmethod",
        "http",
        "https",
    }
)
_INSTALL_TOOLS = {
    "pip": {"install", "uninstall", "download"},
    "pip3": {"install", "uninstall", "download"},
    "uv": {"add", "remove", "sync", "pip", "tool"},
    "npm": {"install", "i", "ci", "add", "uninstall", "update"},
    "pnpm": {"add", "install", "i", "remove", "update"},
    "yarn": {"add", "install", "remove", "upgrade"},
    "bun": {"add", "install", "remove"},
    "cargo": {"add", "install", "remove", "update"},
    "poetry": {"add", "install", "remove", "update"},
    "apt": None,
    "apt-get": None,
    "brew": None,
    "choco": None,
    "winget": None,
    "conda": None,
    "gem": None,
    "go": {"get", "install"},
}
_DEPLOY = {
    "kubectl": None,
    "helm": None,
    "terraform": {"apply", "destroy", "import"},
    "docker": {"push", "login"},
    "gh": {"release", "pr", "repo"},
    "npm": {"publish"},
    "cargo": {"publish"},
    "twine": None,
    "firebase": None,
    "vercel": None,
    "flyctl": None,
}
_READ_SHELL = frozenset(
    {"ls", "cat", "head", "tail", "wc", "rg", "grep", "find", "tree", "stat", "file"}
)
_CHECK_TOOLS = {
    "ruff": {"check", "format"},
    "mypy": None,
    "pyright": None,
    "flake8": None,
    "black": None,
    "eslint": None,
    "tsc": None,
    "cargo": {"check", "clippy", "fmt"},
}
_FIX_FLAGS = re.compile(r"^--?(?:fix|write|in-place|i)$|^--fix=")
_VALUE_OPTS = {
    "pytest": frozenset(
        {
            "-k",
            "-m",
            "-p",
            "-c",
            "-o",
            "-W",
            "-n",
            "--rootdir",
            "--tb",
            "--deselect",
            "--ignore",
            "--maxfail",
            "--timeout",
            "--durations",
            "--basetemp",
            "-r",
        }
    ),
    "cargo": frozenset(
        {
            "-p",
            "--package",
            "--test",
            "--bench",
            "--bin",
            "--example",
            "--features",
            "-F",
            "--target",
            "--manifest-path",
            "-j",
            "--jobs",
            "--profile",
            "--exclude",
        }
    ),
    "ctest": frozenset(
        {"-R", "-E", "-L", "-LE", "-C", "--test-dir", "-j", "--parallel", "--timeout", "-I"}
    ),
    "js": frozenset(
        {
            "-t",
            "--testNamePattern",
            "--config",
            "-c",
            "--project",
            "--reporter",
            "--maxWorkers",
            "-w",
            "--environment",
            "--dir",
        }
    ),
}


def _basename(word: str) -> str:
    name = os.path.basename(word.replace("\\", "/")).lower()
    for ext in (".exe", ".cmd", ".bat"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def _rel_value(value: Any, root: str | None) -> str | None:
    """Workspace-relative form of a path argument (None when outside / invalid)."""
    if not isinstance(value, str) or not value or "\x00" in value or "\n" in value:
        return None
    if root is None:
        return None
    from headroom.agent.change_scope import canonicalize

    canon = canonicalize(value, root=root, cwd=root)
    return canon.rel if canon.rel is not None else None


def _unsupported(tool: str, family: str = "unsupported") -> NormalizedStep:
    return NormalizedStep(StepTemplate("other", family, tool, safety=Safety.UNSUPPORTED.value), {})


def _no_argv(tool: str, family: str, safety: Safety) -> NormalizedStep:
    """A step whose arguments are never generalized or stored (secrets, URLs, writes)."""
    return NormalizedStep(StepTemplate("other", family, tool, safety=safety.value), {})


def normalize_call(tool_name: str, args: Mapping[str, Any], root: str | None) -> NormalizedStep:
    """Typed template + slot values of one executed tool call."""
    from headroom.agent.tool_contracts import Family, family_of

    name = tool_name or ""
    lname = name.lower()
    if name in WORKFLOW_TOOL_NAMES:
        return _unsupported(name, "headroom.workflow")
    args = args if isinstance(args, Mapping) else {}
    family = family_of(name, args)
    if lname == "read":
        rel = _rel_value(args.get("file_path") or args.get("path"), root)
        if rel is None:
            return _unsupported(name, "read.outside")
        values: dict[str, Any] = {"path": rel}
        slots = [SlotSpec("path", "path")]
        for key in ("offset", "limit"):
            if isinstance(args.get(key), int) and not isinstance(args.get(key), bool):
                values[key] = int(args[key])
                slots.append(SlotSpec(key, "int", required=False))
        return NormalizedStep(
            StepTemplate("read", "read.file", "Read", (), tuple(slots), Safety.READ_ONLY.value),
            values,
        )
    if lname == "glob":
        pattern = args.get("pattern")
        if not isinstance(pattern, str) or not _glob_ok(pattern):
            return _unsupported(name, "search.glob")
        values = {"pattern": pattern}
        slots = [SlotSpec("pattern", "glob")]
        if args.get("path"):
            rel = _rel_value(args.get("path"), root)
            if rel is None:
                return _unsupported(name, "search.glob")
            values["path"] = rel or "."
            slots.append(SlotSpec("path", "path", required=False))
        return NormalizedStep(
            StepTemplate("glob", "search.glob", "Glob", (), tuple(slots), Safety.READ_ONLY.value),
            values,
        )
    if lname == "grep":
        pattern = args.get("pattern")
        if not isinstance(pattern, str) or not _pattern_ok(pattern):
            return _unsupported(name, "search.grep")
        values = {"pattern": pattern}
        slots = [SlotSpec("pattern", "pattern")]
        constants: list[str] = []
        for key in sorted(args):
            value = args[key]
            if key == "pattern":
                continue
            if key == "path":
                rel = _rel_value(value, root)
                if rel is None:
                    return _unsupported(name, "search.grep")
                values["path"] = rel or "."
                slots.append(SlotSpec("path", "path", required=False))
            elif key == "glob" and isinstance(value, str) and _glob_ok(value):
                values["glob"] = value
                slots.append(SlotSpec("glob", "glob", required=False))
            elif key in (
                "output_mode",
                "-i",
                "-n",
                "multiline",
                "type",
                "head_limit",
            ) and isinstance(value, str | bool | int):
                constants.append(f"{key}={json.dumps(value)}")
            else:
                return _unsupported(name, "search.grep")
        return NormalizedStep(
            StepTemplate(
                "grep",
                "search.grep",
                "Grep",
                tuple(constants),
                tuple(slots),
                Safety.READ_ONLY.value,
            ),
            values,
        )
    if family is Family.EDIT:
        return _no_argv(name, f"edit.{lname}", Safety.MUTATING)
    if family is Family.HTTP:
        return _no_argv(name, f"http.{lname}", Safety.NETWORK)
    if family in (Family.EXEC, Family.GIT, Family.TEST):
        command = args.get("command") if isinstance(args.get("command"), str) else None
        if command is None:
            return _unsupported(name, "exec")
        return _normalize_command(name, command, root)
    return _unsupported(name)


def _glob_ok(pattern: str) -> bool:
    return (
        0 < len(pattern) <= 200
        and "\x00" not in pattern
        and "\n" not in pattern
        and not os.path.isabs(pattern)
        and not pattern.startswith(("/", "\\"))
        and ".." not in re.split(r"[\\/]", pattern)
    )


def _pattern_ok(pattern: str) -> bool:
    if not (0 < len(pattern) <= 300) or "\x00" in pattern or "\n" in pattern:
        return False
    try:
        re.compile(pattern)
    except re.error:
        return False
    return True


def _normalize_command(tool: str, command: str, root: str | None) -> NormalizedStep:
    from headroom.agent.change_scope import planned_changes

    text = command.strip()
    if not text:
        return _unsupported(tool, "exec")
    if _CREDENTIAL.search(text):
        return _no_argv(tool, "exec.credential", Safety.CREDENTIAL)
    if re.search(r"(?i)https?://|\bgit@|ssh://", text):
        return _no_argv(tool, "exec.network", Safety.NETWORK)
    if tool.lower() == "powershell":
        # Cmdlets need the PowerShell host to run: there is no direct argv form.
        words = text.split()
        head = words[0].lower() if words else ""
        if head in _DELETE:
            return _no_argv(tool, "exec.delete", Safety.DELETION)
        if head in _MUTATE:
            return _no_argv(tool, "exec.mutate", Safety.MUTATING)
        if head in _NETWORK:
            return _no_argv(tool, "exec.network", Safety.NETWORK)
    if _SHELL_META.search(text):
        if planned_changes(tool, {"command": text}):
            return _no_argv(tool, "exec.mutate", Safety.MUTATING)
        return _unsupported(tool, "exec.compound")
    try:
        words = shlex.split(text, posix=True)
    except ValueError:
        return _unsupported(tool, "exec")
    if not words or any(_ENV_ASSIGN.match(w) for w in words[:1]):
        return _unsupported(tool, "exec.env")
    head = _basename(words[0])
    sub = words[1].lower() if len(words) > 1 else ""
    if head in _ELEVATION:
        return _no_argv(tool, "exec.elevation", Safety.ELEVATION)
    if head in _DELETE:
        return _no_argv(tool, "exec.delete", Safety.DELETION)
    if head in _NETWORK:
        return _no_argv(tool, "exec.network", Safety.NETWORK)
    for table, safety in ((_INSTALL_TOOLS, Safety.INSTALL), (_DEPLOY, Safety.DEPLOY)):
        subs = table.get(head, ()) if head in table else ()
        if head in table and (subs is None or sub in subs):
            return _no_argv(tool, f"exec.{head}", safety)
    if head == "git" and sub in _GIT_NETWORK:
        return _no_argv(tool, "git.network", Safety.NETWORK)
    if planned_changes(tool, {"command": text}):
        return _no_argv(tool, "exec.mutate", Safety.MUTATING)
    if head == "git":
        if sub in _GIT_NETWORK:
            return _no_argv(tool, "git.network", Safety.NETWORK)
        if sub == "clean":
            return _no_argv(tool, "git.clean", Safety.DELETION)
        branch_listing = sub == "branch" and all(
            w in ("--list", "-a", "-v", "-vv", "--show-current", "-r") for w in words[2:]
        )
        if sub not in _GIT_READ and not branch_listing:
            return _no_argv(tool, f"git.{sub or 'other'}", Safety.MUTATING)
        if any(
            w in ("-C", "-c", "--git-dir", "--work-tree", "--output", "-o", "--ext-diff")
            for w in words[1:]
        ):
            return _unsupported(tool, f"git.{sub}")
        return _generic_exec(tool, words, root, f"git.{sub}", Safety.READ_ONLY, kind="exec")
    from headroom.agent.test_impact import framework_of_command

    framework = framework_of_command(text)
    if framework is not None and framework != "canonical":
        return _test_step(tool, words, root, framework)
    if head in ("npm", "pnpm", "yarn", "bun") and re.match(
        r"^(?:run\s+)?test$", " ".join(w.lower() for w in words[1:3]).strip()
    ):
        return NormalizedStep(
            StepTemplate(
                "verify",
                f"test.{head}",
                tool,
                tuple(words),
                (),
                Safety.VERIFICATION.value,
                "canonical",
            ),
            {},
        )
    if head in _CHECK_TOOLS:
        subs = _CHECK_TOOLS[head]
        if (subs is None or sub in subs) and not any(_FIX_FLAGS.match(w) for w in words):
            if head == "black" and "--check" not in words:
                return _no_argv(tool, "check.black", Safety.MUTATING)
            if head == "ruff" and sub == "format" and "--check" not in words:
                return _no_argv(tool, "check.ruff", Safety.MUTATING)
            if head == "cargo" and sub == "fmt" and "--check" not in words:
                return _no_argv(tool, "check.cargo_fmt", Safety.MUTATING)
            return _generic_exec(
                tool, words, root, f"check.{head}", Safety.VERIFICATION, kind="exec"
            )
        return _no_argv(tool, f"check.{head}", Safety.MUTATING)
    if head in _READ_SHELL:
        if head == "find" and any(
            w in ("-exec", "-execdir", "-delete", "-ok", "-fprint", "-fprintf", "-fls")
            for w in words
        ):
            return _no_argv(tool, "exec.find", Safety.MUTATING)
        return _generic_exec(tool, words, root, f"shell.{head}", Safety.READ_ONLY, kind="exec")
    if (
        head in ("make", "cmake", "ninja", "msbuild")
        or (head == "cargo" and sub in ("build", "run"))
        or (head in ("npm", "pnpm", "yarn") and sub in ("run", "build"))
    ):
        return _no_argv(tool, f"build.{head}", Safety.MUTATING)
    return _unsupported(tool, f"exec.{head}")


def _generic_exec(
    tool: str, words: Sequence[str], root: str | None, family: str, safety: Safety, *, kind: str
) -> NormalizedStep:
    """Options are template constants; path operands become ``paths`` slot values."""
    exe = _absolute_exe([words[0]], root)
    if exe is None:
        return _unsupported(tool, family)
    argv: list[str] = list(exe)
    paths: list[str] = []
    seen_paths = False
    for word in words[1:]:
        if word.startswith("-") or not _looks_like_path(word, root):
            if seen_paths and not word.startswith("-"):
                return _unsupported(tool, family)
            argv.append(word)
            continue
        rel = _rel_value(word, root)
        if rel is None:
            return _unsupported(tool, family)
        paths.append(rel or ".")
        if not seen_paths:
            argv.append("{paths}")
            seen_paths = True
    slots = (SlotSpec("paths", "paths", required=False),) if seen_paths else ()
    return NormalizedStep(
        StepTemplate(kind, family, tool, tuple(argv), slots, safety.value),
        {"paths": paths} if seen_paths else {},
    )


def _absolute_exe(prefix: Sequence[str], root: str | None) -> list[str] | None:
    """A relative executable path (``.venv/bin/python``) is anchored at the root."""
    out = list(prefix)
    head = out[0] if out else ""
    if ("/" in head or "\\" in head) and not os.path.isabs(head):
        if root is None:
            return None
        candidate = os.path.normpath(os.path.join(root, head))
        if _rel_value(candidate, root) is None or not os.path.exists(candidate):
            return None
        out[0] = candidate
    return out


def _looks_like_path(word: str, root: str | None) -> bool:
    if "/" in word or "\\" in word or re.search(r"\.\w{1,6}$", word):
        return True
    return bool(root) and os.path.exists(os.path.join(root, word))


def _test_step(tool: str, words: Sequence[str], root: str | None, framework: str) -> NormalizedStep:
    """Runner prefix + options as constants; selected tests become the ``tests`` slot."""
    marker = {
        "pytest": re.compile(r"^(?:py\.test|pytest)$"),
        "cargo": re.compile(r"^cargo$"),
        "ctest": re.compile(r"^ctest$"),
        "jest": re.compile(r"^jest$"),
        "vitest": re.compile(r"^vitest$"),
    }[framework]
    idx = next((i for i, w in enumerate(words) if marker.match(_basename(w))), None)
    if idx is None:
        return _unsupported(tool, f"test.{framework}")
    prefix = list(words[: idx + 1])
    rest = list(words[idx + 1 :])
    if framework == "cargo":
        if not rest or rest[0] not in ("test", "nextest"):
            return _unsupported(tool, "test.cargo")
        prefix.append(rest.pop(0))
        if prefix[-1] == "nextest" and rest and rest[0] == "run":
            prefix.append(rest.pop(0))
    if framework == "vitest" and rest and rest[0] in ("run", "related"):
        prefix.append(rest.pop(0))
    value_opts = _VALUE_OPTS.get(framework if framework in _VALUE_OPTS else "js", frozenset())
    constants: list[str] = []
    tests: list[str] = []
    passthrough = False
    i = 0
    while i < len(rest):
        word = rest[i]
        if passthrough:
            constants.append(word)
        elif word == "--":
            passthrough = True
            constants.append(word)
        elif word.startswith("-"):
            constants.append(word)
            if word in value_opts and i + 1 < len(rest):
                constants.append(rest[i + 1])
                i += 1
        else:
            if framework in ("pytest", "jest", "vitest"):
                file_part = word.split("::", 1)[0]
                rel = _rel_value(file_part, root)
                if rel is None:
                    return _unsupported(tool, f"test.{framework}")
                tests.append(rel + word[len(file_part) :])
            else:
                if not re.fullmatch(r"[\w:.\-]+", word):
                    return _unsupported(tool, f"test.{framework}")
                tests.append(word)
        i += 1
    if (
        any(_FIX_FLAGS.match(w) for w in constants)
        or "-u" in constants
        or "--updateSnapshot" in constants
    ):
        return _no_argv(tool, f"test.{framework}", Safety.MUTATING)
    split = constants.index("--") if "--" in constants else len(constants)
    anchored = _absolute_exe(prefix, root)
    if anchored is None:
        return _unsupported(tool, f"test.{framework}")
    argv = (*anchored, *constants[:split], "{tests}", *constants[split:])
    return NormalizedStep(
        StepTemplate(
            "verify",
            f"test.{framework}",
            tool,
            tuple(argv),
            (SlotSpec("tests", "tests", required=False),),
            Safety.VERIFICATION.value,
            framework,
        ),
        {"tests": tests},
    )


# ---------------------------------------------------------------------------
# Segments of executed tool calls (proxy path)
# ---------------------------------------------------------------------------


@dataclass
class SegmentCall:
    call_id: str
    tool_name: str
    args: Mapping[str, Any]
    turn: int
    ok: bool | None = None
    output_chars: int = 0


@dataclass
class Segment:
    calls: list[SegmentCall] = field(default_factory=list)
    closed: bool = False
    corrected: bool = False

    def segment_id(self, session_id: str) -> str:
        return "sg_" + _digest(session_id, *(c.call_id for c in self.calls))[:32]

    @property
    def turns(self) -> int:
        return len({c.turn for c in self.calls})


def _authored(content: Any) -> str:
    from headroom.agent.normalize import authored_user_text

    if isinstance(content, str):
        return authored_user_text(content)
    if isinstance(content, list):
        texts = [
            str(b.get("text") or "")
            for b in content
            if isinstance(b, Mapping) and b.get("type") in (None, "text", "input_text")
        ]
        return authored_user_text("\n".join(texts))
    return ""


def _result_ok(content: Any, is_error: bool) -> tuple[bool, int]:
    from headroom.agent.normalize import _block_text

    text = content if isinstance(content, str) else _block_text(content)
    env = ToolResultEnvelope.from_text(
        tool_call_id="", tool_name="", text=text or "", is_error=is_error
    )
    return (not env.is_error and not env.is_partial), len(text or "")


def extract_segments(provider: str, messages: Any) -> list[Segment]:
    """Tool-call segments between authentic user turns (Anthropic / OpenAI chat)."""
    from headroom.agent.task_state import classify_user_turn

    if not isinstance(messages, list):
        return []
    items = messages[-_SEGMENT_MESSAGES:]
    segments: list[Segment] = []
    current = Segment()
    by_id: dict[str, SegmentCall] = {}
    turn = 0

    def boundary(text: str) -> None:
        nonlocal current
        if current.calls:
            current.closed = True
            current.corrected = classify_user_turn(text) == "correction"
            segments.append(current)
        current = Segment()

    for message in items:
        if not isinstance(message, Mapping):
            continue
        role = message.get("role")
        content = message.get("content")
        if provider == "anthropic":
            if role == "user":
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, Mapping) and block.get("type") == "tool_result":
                            call = by_id.get(str(block.get("tool_use_id") or ""))
                            if call is not None:
                                call.ok, call.output_chars = _result_ok(
                                    block.get("content"), bool(block.get("is_error"))
                                )
                text = _authored(content)
                if text:
                    boundary(text)
            elif role == "assistant":
                uses = [
                    b
                    for b in (content if isinstance(content, list) else ())
                    if isinstance(b, Mapping) and b.get("type") == "tool_use"
                ]
                if uses:
                    turn += 1
                    for b in uses:
                        call = SegmentCall(
                            str(b.get("id") or ""),
                            str(b.get("name") or ""),
                            b.get("input") if isinstance(b.get("input"), Mapping) else {},
                            turn,
                        )
                        by_id[call.call_id] = call
                        current.calls.append(call)
        elif provider == "openai":
            if role == "user":
                text = _authored(content)
                if text:
                    boundary(text)
            elif role == "assistant":
                calls = message.get("tool_calls") or ()
                if calls:
                    turn += 1
                    for c in calls:
                        if not isinstance(c, Mapping):
                            continue
                        fn = c.get("function") or {}
                        try:
                            parsed = json.loads(fn.get("arguments") or "{}")
                        except (TypeError, ValueError):
                            parsed = {}
                        call = SegmentCall(
                            str(c.get("id") or ""),
                            str(fn.get("name") or ""),
                            parsed if isinstance(parsed, Mapping) else {},
                            turn,
                        )
                        by_id[call.call_id] = call
                        current.calls.append(call)
            elif role == "tool":
                call = by_id.get(str(message.get("tool_call_id") or ""))
                if call is not None:
                    call.ok, call.output_chars = _result_ok(content, False)
    return segments  # the open (latest) segment is observed once the user speaks again


# ---------------------------------------------------------------------------
# Macro records
# ---------------------------------------------------------------------------


@dataclass
class Observation:
    segment_id: str
    session_id: str
    ok: bool
    clean: bool
    turns: int
    ts: float
    epoch: int
    values: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "session_id": self.session_id,
            "ok": self.ok,
            "clean": self.clean,
            "turns": self.turns,
            "ts": self.ts,
            "epoch": self.epoch,
            "values": self.values,
        }

    @classmethod
    def from_dict(cls, v: Mapping[str, Any]) -> Observation:
        return cls(
            str(v["segment_id"]),
            str(v["session_id"]),
            bool(v["ok"]),
            bool(v["clean"]),
            int(v["turns"]),
            float(v["ts"]),
            int(v.get("epoch") or 0),
            [dict(x) for x in v.get("values") or ()],
        )


@dataclass
class Macro:
    macro_id: str
    workspace_id: str
    steps: tuple[StepTemplate, ...]
    status: str = MacroStatus.CANDIDATE.value
    version: int = 1
    epoch: int = 0
    observations: list[Observation] = field(default_factory=list)
    violation: bool = False
    env_fingerprint: str = ""
    tool_digest: str = ""
    created_at: float = 0.0
    promoted_at: float = 0.0
    last_used: float = 0.0
    uses: int = 0
    reason: str = ""

    @property
    def safety(self) -> str:
        return max((s.safety for s in self.steps), key=lambda v: _SEVERITY.get(v, 9))

    def current(self) -> list[Observation]:
        return [o for o in self.observations if o.epoch == self.epoch]

    def stats(self) -> dict[str, Any]:
        obs = self.current()
        ok = [o for o in obs if o.ok and o.clean]
        sessions: dict[str, int] = {}
        for o in ok:
            sessions[o.session_id] = sessions.get(o.session_id, 0) + 1
        saved = [o.turns - 1 for o in ok]
        return {
            "occurrences": len(obs),
            "successes": len(ok),
            "success_rate": (len(ok) / len(obs)) if obs else 0.0,
            "sessions": len(sessions),
            "max_per_session": max(sessions.values(), default=0),
            "turns_saved_estimate": int(statistics.median(saved)) if saved else 0,
        }

    def defaults(self) -> list[dict[str, Any]]:
        """Per-step slot values identical across every clean observation (else unbound)."""
        ok = [o for o in self.current() if o.ok and o.clean] or self.current()
        out: list[dict[str, Any]] = []
        for i, step in enumerate(self.steps):
            bound: dict[str, Any] = {}
            for slot in step.slots:
                values = [
                    json.dumps(o.values[i].get(slot.name), sort_keys=True)
                    for o in ok
                    if i < len(o.values)
                ]
                if values and len(set(values)) == 1 and json.loads(values[0]) is not None:
                    bound[slot.name] = json.loads(values[0])
            out.append(bound)
        return out

    def inputs(self) -> list[dict[str, Any]]:
        """Typed inputs the caller must/may pass (``s<i>.<slot>``; shared ``tests``)."""
        defaults = self.defaults()
        out: list[dict[str, Any]] = []
        for i, step in enumerate(self.steps):
            for slot in step.slots:
                if slot.type == "tests":
                    if not any(x["name"] == "tests" for x in out):
                        out.append(
                            {
                                "name": "tests",
                                "type": "tests",
                                "required": False,
                                "default": "verification plan",
                            }
                        )
                    continue
                if slot.name in defaults[i]:
                    out.append(
                        {
                            "name": f"s{i + 1}.{slot.name}",
                            "type": slot.type,
                            "required": False,
                            "default": defaults[i][slot.name],
                        }
                    )
                elif slot.required:
                    out.append(
                        {"name": f"s{i + 1}.{slot.name}", "type": slot.type, "required": True}
                    )
        return out

    def to_spec(self) -> dict[str, Any]:
        return {
            "macro_id": self.macro_id,
            "workspace_id": self.workspace_id,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status,
            "version": self.version,
            "epoch": self.epoch,
            "observations": [o.to_dict() for o in self.observations[-_OBS_PER_MACRO:]],
            "violation": self.violation,
            "env_fingerprint": self.env_fingerprint,
            "tool_digest": self.tool_digest,
            "created_at": self.created_at,
            "promoted_at": self.promoted_at,
            "last_used": self.last_used,
            "uses": self.uses,
            "reason": self.reason,
        }

    @classmethod
    def from_spec(cls, v: Mapping[str, Any]) -> Macro:
        return cls(
            macro_id=str(v["macro_id"]),
            workspace_id=str(v["workspace_id"]),
            steps=tuple(StepTemplate.from_dict(s) for s in v.get("steps") or ()),
            status=str(v.get("status") or MacroStatus.CANDIDATE.value),
            version=int(v.get("version") or 1),
            epoch=int(v.get("epoch") or 0),
            observations=[Observation.from_dict(o) for o in v.get("observations") or ()],
            violation=bool(v.get("violation")),
            env_fingerprint=str(v.get("env_fingerprint") or ""),
            tool_digest=str(v.get("tool_digest") or ""),
            created_at=float(v.get("created_at") or 0.0),
            promoted_at=float(v.get("promoted_at") or 0.0),
            last_used=float(v.get("last_used") or 0.0),
            uses=int(v.get("uses") or 0),
            reason=str(v.get("reason") or ""),
        )

    def summary(self) -> dict[str, Any]:
        defaults = self.defaults()
        return {
            "macro_id": self.macro_id,
            "status": self.status,
            "version": self.version,
            "safety": self.safety,
            "steps": [s.describe(defaults[i]) for i, s in enumerate(self.steps)],
            "inputs": self.inputs(),
            **self.stats(),
            "uses": self.uses,
            "reason": self.reason,
        }


def macro_id_for(workspace_id: str, steps: Sequence[StepTemplate]) -> str:
    return "wf_" + _digest(workspace_id, *(s.key() for s in steps))[:20]


def _tools_digest(tools: Any, names: Sequence[str]) -> str:
    from headroom.agent.tool_contracts import _schemas_from_tools

    schemas = _schemas_from_tools(tools)
    wanted = sorted({n for n in names if n})
    return _digest(
        *(f"{n}:{json.dumps(schemas.get(n), sort_keys=True, default=str)}" for n in wanted)
    )[:16]


# ---------------------------------------------------------------------------
# Compiled steps and their execution
# ---------------------------------------------------------------------------


class WorkflowError(Exception):
    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code
        self.message = message or code


@dataclass
class CompiledStep:
    index: int
    template: StepTemplate
    tool_name: str
    tool_input: dict[str, Any]
    argv: tuple[str, ...] = ()
    cwd: str = ""
    values: dict[str, Any] = field(default_factory=dict)
    plan_tier: int = 0
    plan_targets: tuple[str, ...] = ()
    note: str = ""

    @property
    def step_id(self) -> str:
        return f"s{self.index + 1}"

    def display(self) -> str:
        if self.argv:
            return " ".join(shlex.quote(a) for a in self.argv)
        return f"{self.tool_name} {json.dumps(self.tool_input, sort_keys=True)}"


@dataclass
class Ticket:
    session_id: str
    macro_id: str
    inputs_digest: str
    version: int
    pins: tuple[Any, ...]
    issued_at: float
    call_id: str


def _inputs_digest(inputs: Any) -> str:
    return _digest(
        json.dumps(inputs if isinstance(inputs, Mapping) else {}, sort_keys=True, default=str)
    )[:24]


_CACHE_PARTS = frozenset(
    {
        ".pytest_cache",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        "target",
        ".vite",
        "coverage",
        ".coverage",
        ".hypothesis",
        ".cache",
        "node_modules",
        "Testing",
    }
)


def worktree_state(root: str) -> str | None:
    """Digest of the Git-visible worktree changes (tool caches excluded)."""
    from headroom.agent.test_impact import _git

    status = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if status is None:
        return None
    entries = []
    for entry in sorted(e for e in status.split("\x00") if len(e) > 3):
        path = entry[3:].replace("\\", "/")
        parts = set(path.split("/"))
        if parts & _CACHE_PARTS or path.endswith((".pyc", ".pyo")):
            continue
        full = os.path.join(root, path)
        try:
            st = os.stat(full)
            stamp = f"{st.st_size}:{st.st_mtime_ns}"
        except OSError:
            stamp = "missing"
        entries.append(f"{entry[:2]}|{path}|{stamp}")
    return _digest(*entries)[:24]


class WorkflowMacroRegistry:
    """Owner of learned workflow macros (see module docstring)."""

    def __init__(self, runtime: Any = None, *, clock: Any = time.time) -> None:
        self.runtime = runtime
        self._clock = clock
        self._lock = threading.RLock()
        self._macros: dict[str, OrderedDict[str, Macro]] = {}
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._tickets: OrderedDict[tuple[str, str, str], Ticket] = OrderedDict()
        self._runs: dict[str, threading.Event] = {}
        self._busy: set[str] = set()
        self._hook_seen: OrderedDict[str, float] = OrderedDict()
        self._db: Any = None
        self._submit: Any = None
        self.counters: dict[str, int] = {
            "segments_observed": 0,
            "segments_skipped_unsupported": 0,
            "segments_unclean": 0,
            "promoted": 0,
            "manual_only": 0,
            "retired_cap": 0,
            "invalidated": 0,
            "advertised": 0,
            "fallback_tool_missing": 0,
            "fallback_hook_missing": 0,
            "tickets_issued": 0,
            "pretool_denied": 0,
            "executions": 0,
            "completed": 0,
            "failed": 0,
            "cancelled_stale": 0,
            "cancelled_step_violation": 0,
            "cancelled": 0,
            "timeouts": 0,
            "dry_runs": 0,
            "steps_run": 0,
            "estimated_turns_avoided": 0,
        }

    # -- persistence --------------------------------------------------------

    def bind(self, ledger: Any, submit: Any) -> None:
        with ledger.lock:
            for statement in _TABLES:
                ledger.connection.execute(statement)
        with self._lock:
            self._db, self._submit = ledger, submit
        submit(lambda: self._load(ledger))

    def unbind(self) -> None:
        with self._lock:
            self._db = self._submit = None

    def _load(self, db: Any) -> None:
        try:
            rows = db.query("SELECT spec FROM workflow_macros ORDER BY updated_at")
            seen = db.query(
                "SELECT segment_id FROM workflow_observations ORDER BY ts DESC LIMIT ?",
                (_SEEN_LIMIT,),
            )
        except Exception as exc:  # noqa: BLE001 - memory-only macros still work
            log.warning("workflow macros load failed: %s", describe_exception(exc))
            return
        with self._lock:
            for (spec,) in rows:
                try:
                    macro = Macro.from_spec(json.loads(spec))
                except (ValueError, KeyError, TypeError):
                    continue
                self._macros.setdefault(macro.workspace_id, OrderedDict()).setdefault(
                    macro.macro_id, macro
                )
            for (segment_id,) in reversed(seen):
                self._seen[str(segment_id)] = None

    def _persist(self, macro: Macro, segment_id: str | None = None) -> None:
        with self._lock:
            db, submit = self._db, self._submit
            spec = json.dumps(macro.to_spec(), sort_keys=True, default=str)
        if db is None or submit is None:
            return
        now = self._clock()

        def write() -> None:
            statements: list[tuple[str, Sequence[Any]]] = [
                (
                    "INSERT OR REPLACE INTO workflow_macros(workspace_id, macro_id, spec, updated_at) VALUES (?,?,?,?)",
                    (macro.workspace_id, macro.macro_id, spec, now),
                )
            ]
            if segment_id:
                statements.append(
                    (
                        "INSERT OR IGNORE INTO workflow_observations(segment_id, workspace_id, macro_id, ts) VALUES (?,?,?,?)",
                        (segment_id, macro.workspace_id, macro.macro_id, now),
                    )
                )
            db.transaction(statements)

        submit(write)

    # -- observation (worker) -----------------------------------------------

    def observe(
        self, identity: Any, segments: Sequence[Segment], *, tools: Any = None
    ) -> list[str]:
        """Record closed segments once each; returns the macro ids that changed."""
        from headroom.agent.enforcement import host_verdict

        changed: list[str] = []
        if not identity.can_authorize_paths():
            return changed
        for segment in segments:
            if not segment.closed:
                continue
            seg_id = segment.segment_id(identity.session_id)
            with self._lock:
                if seg_id in self._seen:
                    continue
                self._seen[seg_id] = None
                while len(self._seen) > _SEEN_LIMIT:
                    self._seen.popitem(last=False)
            calls = segment.calls
            if any(c.tool_name in WORKFLOW_TOOL_NAMES for c in calls):
                continue  # an executed macro is a use, not a new observation
            if not (MIN_STEPS <= len(calls) <= MAX_STEPS):
                self.counters["segments_skipped_unsupported"] += 1
                continue
            normalized = [normalize_call(c.tool_name, c.args, identity.root) for c in calls]
            if any(n.template.safety == Safety.UNSUPPORTED.value for n in normalized):
                self.counters["segments_skipped_unsupported"] += 1
                continue
            steps = tuple(n.template for n in normalized)
            ok = all(c.ok is True for c in calls)
            verdicts = [host_verdict(c.call_id) for c in calls]
            violation = any(
                v in ("BLOCK", "WARN", "REPAIR_SUGGEST") for v in verdicts
            ) or self._scope_violation(identity, calls)
            clean = ok and not violation and not segment.corrected
            if not clean:
                self.counters["segments_unclean"] += 1
            macro_id = macro_id_for(identity.workspace_id, steps)
            now = self._clock()
            with self._lock:
                bucket = self._macros.setdefault(identity.workspace_id, OrderedDict())
                macro = bucket.get(macro_id)
                if macro is None:
                    macro = Macro(macro_id, identity.workspace_id, steps, created_at=now)
                    bucket[macro_id] = macro
                    while len(bucket) > _MACROS_PER_WORKSPACE:
                        victim = next(
                            (k for k, m in bucket.items() if m.status != MacroStatus.ACTIVE.value),
                            None,
                        )
                        if victim is None:
                            break
                        bucket.pop(victim)
                macro.observations.append(
                    Observation(
                        seg_id,
                        identity.session_id,
                        ok,
                        clean,
                        segment.turns,
                        now,
                        macro.epoch,
                        [dict(n.values) for n in normalized],
                    )
                )
                macro.observations = macro.observations[-_OBS_PER_MACRO:]
                if violation:
                    macro.violation = True
                if tools is not None:
                    macro.tool_digest = _tools_digest(tools, [s.tool for s in steps])
            self.counters["segments_observed"] += 1
            self._evaluate(identity, macro)
            self._persist(macro, seg_id)
            changed.append(macro_id)
        return changed

    def _scope_violation(self, identity: Any, calls: Sequence[SegmentCall]) -> bool:
        firewall = self.runtime.services.get("change_scope") if self.runtime is not None else None
        if firewall is None:
            return False
        from headroom.agent.change_scope import ScopeClass, planned_changes

        for call in calls:
            for change in planned_changes(call.tool_name, call.args)[:8]:
                if change.path is None:
                    continue
                verdict = firewall.classify(
                    identity, change.path, cwd=identity.root, effect=change.effect
                )
                if verdict.scope in (ScopeClass.UNRELATED, ScopeClass.FORBIDDEN):
                    return True
        return False

    def _evaluate(self, identity: Any, macro: Macro) -> None:
        """Promotion rules (see module docstring)."""
        stats = macro.stats()
        with self._lock:
            if macro.status in (
                MacroStatus.ACTIVE.value,
                MacroStatus.RETIRED.value,
                MacroStatus.DISABLED.value,
            ):
                return
            eligible = (
                stats["occurrences"] >= MIN_OCCURRENCES
                and stats["success_rate"] >= MIN_SUCCESS_RATE
                and stats["successes"] >= MIN_OCCURRENCES
                and (stats["sessions"] >= 2 or stats["max_per_session"] >= MIN_OCCURRENCES)
                and not macro.violation
                and stats["turns_saved_estimate"] >= MIN_TURNS_SAVED
            )
            if not eligible:
                return
            if macro.safety not in AUTO_PROMOTABLE:
                if macro.status != MacroStatus.MANUAL_ONLY.value:
                    macro.status = MacroStatus.MANUAL_ONLY.value
                    macro.reason = f"never_auto_promote:{macro.safety}"
                    self.counters["manual_only"] += 1
                return
            fingerprint = self._env_fingerprint(identity, macro)
            if fingerprint is None:
                macro.status = MacroStatus.DISABLED.value
                macro.reason = "step_executable_missing"
                return
            bucket = self._macros.get(macro.workspace_id, OrderedDict())
            active = [m for m in bucket.values() if m.status == MacroStatus.ACTIVE.value]
            if len(active) >= MAX_ACTIVE_PER_WORKSPACE:
                victim = min(active, key=lambda m: (m.last_used or m.promoted_at, m.macro_id))
                victim.status = MacroStatus.RETIRED.value
                victim.reason = "active_cap"
                self.counters["retired_cap"] += 1
                self._persist(victim)
            macro.status = MacroStatus.ACTIVE.value
            macro.promoted_at = self._clock()
            macro.env_fingerprint = fingerprint
            macro.reason = ""
            if macro.version < 1:
                macro.version = 1
            self.counters["promoted"] += 1

    def _env_fingerprint(self, identity: Any, macro: Macro) -> str | None:
        """Executables, runner config and enforcement policy a macro depends on."""
        from headroom.agent.change_scope import EnforcementPolicy
        from headroom.agent.exec_runner import resolve_executable

        parts: list[str] = [json.dumps(EnforcementPolicy.from_env().to_dict(), sort_keys=True)]
        root = identity.root
        for step in macro.steps:
            if step.kind in ("exec", "verify") and step.argv:
                path = resolve_executable(step.argv[0])
                if path is None:
                    return None
                try:
                    st = os.stat(path)
                    parts.append(f"{os.path.normcase(path)}:{st.st_size}:{st.st_mtime_ns}")
                except OSError:
                    return None
            if step.kind == "verify" and root:
                from headroom.agent.test_impact import runner_fingerprint

                parts.append(runner_fingerprint(root))
        return _digest(*parts)[:16]

    def invalidate(self, workspace_id: str, macro_id: str, reason: str) -> None:
        with self._lock:
            macro = self._macros.get(workspace_id, {}).get(macro_id)
            if macro is None or macro.status not in (
                MacroStatus.ACTIVE.value,
                MacroStatus.CANDIDATE.value,
                MacroStatus.MANUAL_ONLY.value,
            ):
                return
            macro.status = MacroStatus.INVALIDATED.value
            macro.epoch += 1
            macro.version += 1
            macro.reason = reason
            macro.violation = False
            self.counters["invalidated"] += 1
            for key in [k for k in self._tickets if k[1] == macro_id]:
                self._tickets.pop(key, None)
        self._persist(macro)

    def macros(self, workspace_id: str) -> list[Macro]:
        with self._lock:
            return list(self._macros.get(workspace_id, {}).values())

    def get(self, workspace_id: str, macro_id: str) -> Macro | None:
        with self._lock:
            return self._macros.get(workspace_id, {}).get(macro_id)

    def active(self, workspace_id: str) -> list[Macro]:
        out = [m for m in self.macros(workspace_id) if m.status == MacroStatus.ACTIVE.value]
        return sorted(out, key=lambda m: (-m.uses, m.macro_id))

    def _revalidate(self, identity: Any, macro: Macro, *, tools: Any = None) -> str | None:
        """Reason the macro is no longer valid (and invalidate it), else None."""
        fingerprint = self._env_fingerprint(identity, macro)
        reason = None
        if fingerprint is None:
            reason = "step_executable_missing"
        elif fingerprint != macro.env_fingerprint:
            reason = "environment_changed"
        elif (
            tools is not None
            and macro.tool_digest
            and _tools_digest(tools, [s.tool for s in macro.steps]) != macro.tool_digest
        ):
            reason = "tool_schema_changed"
        if reason is not None:
            self.invalidate(macro.workspace_id, macro.macro_id, reason)
        return reason

    # -- advertisement (request path; cheap) --------------------------------

    def note_hook(self, session_id: str) -> None:
        with self._lock:
            self._hook_seen[session_id] = self._clock()
            self._hook_seen.move_to_end(session_id)
            while len(self._hook_seen) > 4096:
                self._hook_seen.popitem(last=False)

    def advertisement_lines(self, identity: Any, ctx: Any) -> list[tuple[int, str]]:
        if not identity.can_authorize_paths():
            return []
        active = self.active(identity.workspace_id)
        if not active:
            return []
        from headroom.agent.tool_contracts import _schemas_from_tools

        names = set(_schemas_from_tools(getattr(ctx, "tools", None)))
        tool = next((n for n in (WORKFLOW_MCP_TOOL, WORKFLOW_TOOL) if n in names), None)
        if tool is None:
            self.counters["fallback_tool_missing"] += 1
            return []
        with self._lock:
            hook = identity.session_id in self._hook_seen
        if not hook:
            self.counters["fallback_hook_missing"] += 1
            return []
        lines: list[tuple[int, str]] = [
            (
                5,
                f"Learned workflows (one {tool} call replaces several tool turns; each step is "
                "re-validated before it runs; dry_run=true previews):",
            )
        ]
        shown = 0
        for macro in active:
            if shown >= 6:
                break
            if self._revalidate(identity, macro, tools=getattr(ctx, "tools", None)) is not None:
                continue
            summary = macro.summary()
            inputs = [i for i in summary["inputs"] if i.get("required")]
            needs = ", ".join(f"{i['name']}:{i['type']}" for i in inputs) or "none required"
            tests = (
                " tests default to the verification plan;"
                if any(s.kind == "verify" for s in macro.steps)
                else ""
            )
            lines.append(
                (
                    5,
                    f"- {macro.macro_id} ({macro.safety}, {len(macro.steps)} steps, saves ~"
                    f"{summary['turns_saved_estimate']} tool turns): {' → '.join(summary['steps'])};"
                    f"{tests} inputs: {needs}",
                )
            )
            shown += 1
        if shown == 0:
            return []
        self.counters["advertised"] += 1
        return lines

    # -- compile / validate -------------------------------------------------

    def _validate_inputs(
        self, identity: Any, macro: Macro, typed_inputs: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        defaults = macro.defaults()
        allowed = {i["name"] for i in macro.inputs()} | {"tests"}
        for key in typed_inputs:
            if key not in allowed:
                raise WorkflowError("workflow.unknown_input", f"unknown input {key!r}")
        out: list[dict[str, Any]] = []
        for i, step in enumerate(macro.steps):
            values: dict[str, Any] = {}
            for slot in step.slots:
                key = "tests" if slot.type == "tests" else f"s{i + 1}.{slot.name}"
                if key in typed_inputs:
                    raw = typed_inputs[key]
                elif slot.name in defaults[i] and slot.type != "tests":
                    raw = defaults[i][slot.name]
                elif slot.type == "tests":
                    raw = None  # filled from the verification plan
                elif slot.required:
                    raise WorkflowError("workflow.missing_input", f"missing input {key}")
                else:
                    continue
                if raw is None:
                    continue
                values[slot.name] = _check_value(slot.type, raw, identity.root)
            out.append(values)
        return out

    def compile(
        self, identity: Any, macro: Macro, typed_inputs: Mapping[str, Any], *, plan: Any = None
    ) -> list[CompiledStep]:
        if macro.safety not in AUTO_PROMOTABLE:
            raise WorkflowError(
                "workflow.not_executable", f"{macro.safety} workflows never auto-execute"
            )
        if len(macro.steps) > MAX_STEPS:
            raise WorkflowError("workflow.too_many_steps")
        root = identity.root
        values = self._validate_inputs(identity, macro, typed_inputs)
        defaults = macro.defaults()
        compiled: list[CompiledStep] = []
        for i, step in enumerate(macro.steps):
            v = values[i]
            if step.kind == "read":
                rel = v["path"]
                tool_input: dict[str, Any] = {
                    "file_path": os.path.join(root, rel).replace("\\", "/")
                }
                for key in ("offset", "limit"):
                    if key in v:
                        tool_input[key] = v[key]
                compiled.append(CompiledStep(i, step, "Read", tool_input, cwd=root, values=v))
            elif step.kind in ("glob", "grep"):
                tool_input = {"pattern": v["pattern"]}
                if "path" in v:
                    tool_input["path"] = os.path.join(root, v["path"]).replace("\\", "/")
                if "glob" in v:
                    tool_input["glob"] = v["glob"]
                for const in step.argv:
                    key, _, raw = const.partition("=")
                    tool_input[key] = json.loads(raw)
                compiled.append(
                    CompiledStep(
                        i,
                        step,
                        step.tool.capitalize()
                        if step.tool.lower() in ("glob", "grep")
                        else step.tool,
                        tool_input,
                        cwd=root,
                        values=v,
                    )
                )
            elif step.kind in ("exec", "verify"):
                tier, targets, argv, cwd, note = 0, (), None, root, ""
                if step.kind == "verify" and "tests" not in v and plan is not None:
                    cmds = [
                        c
                        for c in plan.commands_for(1)
                        if c.framework == (step.framework or c.framework)
                    ]
                    if cmds:
                        cmd = cmds[0]
                        tier, targets = 1, tuple(cmd.targets)
                        if step.framework in ("pytest", "jest", "vitest"):
                            # The workflow's own runner and options, the plan's tests.
                            argv = _fill_argv(
                                step.argv, {**defaults[i], "tests": list(cmd.targets)}
                            )
                        else:
                            argv = tuple(cmd.argv)
                            cwd = os.path.join(root, cmd.cwd) if cmd.cwd not in ("", ".") else root
                        note = f"verification plan r{plan.plan_revision} tier 1"
                if argv is None:
                    fill = dict(defaults[i])
                    fill.update(v)
                    if step.kind == "verify" and "tests" not in v and "tests" not in defaults[i]:
                        fill["tests"] = []
                        note = note or "no plan selection: observed runner without targets"
                    argv = _fill_argv(step.argv, fill)
                if (
                    step.kind == "verify"
                    and step.framework
                    and argv
                    and plan is not None
                    and tier == 0
                ):
                    tier = _tier_for_argv(plan, step.framework, argv)
                command = shlex.join(argv)
                compiled.append(
                    CompiledStep(
                        i,
                        step,
                        "Bash",
                        {"command": command},
                        argv=argv,
                        cwd=cwd,
                        values=v,
                        plan_tier=tier,
                        plan_targets=targets,
                        note=note,
                    )
                )
            else:
                raise WorkflowError("workflow.unsupported_step", step.family)
        return compiled

    # -- revisions ----------------------------------------------------------

    def _pins(self, identity: Any) -> tuple[Any, ...]:
        """(task id, task revision, scope revision, live user atom ids digest)."""
        runtime = self.runtime
        compiler = runtime.services.get("task_state") if runtime is not None else None
        firewall = runtime.services.get("change_scope") if runtime is not None else None
        snap = (
            compiler.snapshot(identity.workspace_id, identity.session_id)
            if compiler is not None
            else None
        )
        task_id = snap.state.id if snap is not None else None
        task_rev = snap.state.revision if snap is not None else 0
        users = (
            _digest(*sorted(a.id for a in snap.state.atoms if a.origin == "user" and a.live))[:12]
            if snap is not None
            else ""
        )
        scope_rev = int(firewall.contract(identity).revision) if firewall is not None else 0
        return (task_id, task_rev, scope_rev, users)

    def _flush(self, timeout: float = 3.0) -> None:
        if self.runtime is not None:
            self.runtime.flush(timeout)

    # -- host pre-tool boundary ---------------------------------------------

    def pretool(self, payload: Mapping[str, Any], *, identity: Any) -> dict[str, Any]:
        """``PreToolUse`` for the workflow tool: validate, then issue a single-use ticket."""
        from headroom.agent.enforcement import render_pretool

        tool_input = (
            payload.get("tool_input") if isinstance(payload.get("tool_input"), Mapping) else {}
        )
        macro_id = str(tool_input.get("macro_id") or "")
        typed = (
            tool_input.get("typed_inputs")
            if isinstance(tool_input.get("typed_inputs"), Mapping)
            else {}
        )
        call_id = str(payload.get("tool_use_id") or "")
        self.note_hook(identity.session_id)
        if bool(tool_input.get("dry_run")) or not macro_id:
            return {"output": None, "stage": "workflow:dry_run", "capability": "host_pretool"}

        def deny(code: str, message: str) -> dict[str, Any]:
            self.counters["pretool_denied"] += 1
            return {
                "output": render_pretool(
                    "BLOCK", f"{message} [{code}]; run the steps with your normal tools instead"
                ),
                "stage": "workflow:denied",
                "capability": "host_pretool",
                "reason_code": code,
            }

        if not identity.can_authorize_paths():
            return {"output": None, "stage": "workflow:untrusted", "capability": "host_pretool"}
        macro = self.get(identity.workspace_id, macro_id)
        if macro is None or macro.status != MacroStatus.ACTIVE.value:
            return deny(
                "workflow.not_active", f"workflow {macro_id} is not an active workflow here"
            )
        reason = self._revalidate(identity, macro)
        if reason is not None:
            return deny(
                f"workflow.invalidated.{reason}", f"workflow {macro_id} was disabled ({reason})"
            )
        self._flush()
        planner = self.runtime.services.get("test_impact") if self.runtime is not None else None
        plan = planner.plan(identity) if planner is not None else None
        try:
            steps = self.compile(identity, macro, typed, plan=plan)
            for step in steps:
                self._authorize_step(
                    identity, step, call_id=f"{call_id}:{step.step_id}:pre", dry=True
                )
        except WorkflowError as exc:
            return deny(exc.code, exc.message)
        ticket = Ticket(
            identity.session_id,
            macro_id,
            _inputs_digest(typed),
            macro.version,
            self._pins(identity),
            self._clock(),
            call_id,
        )
        with self._lock:
            self._tickets[(identity.session_id, macro_id, ticket.inputs_digest)] = ticket
            while len(self._tickets) > 1024:
                self._tickets.popitem(last=False)
        self.counters["tickets_issued"] += 1
        return {"output": None, "stage": "workflow:ticket", "capability": "host_pretool"}

    def _authorize_step(
        self, identity: Any, step: CompiledStep, *, call_id: str, dry: bool = False
    ) -> None:
        """Current ToolContract + ChangeScope decision for one step (raises on refusal)."""
        from headroom.agent.change_scope import Verdict, planned_changes
        from headroom.agent.enforcement import HOST_PRETOOL
        from headroom.agent.tool_contracts import DecisionKind

        if step.template.safety not in AUTO_PROMOTABLE:
            raise WorkflowError(
                "workflow.step_not_safe", f"{step.step_id} is {step.template.safety}"
            )
        if step.argv:
            from headroom.agent.exec_runner import ArgvError, resolve_executable, validate_argv

            try:
                validate_argv(step.argv)
            except ArgvError as exc:
                raise WorkflowError("workflow.argv_invalid", str(exc)) from None
            if resolve_executable(step.argv[0]) is None:
                raise WorkflowError(
                    "workflow.executable_missing", f"{step.argv[0]} is not available"
                )
        validator = (
            self.runtime.services.get("tool_contracts") if self.runtime is not None else None
        )
        firewall = self.runtime.services.get("change_scope") if self.runtime is not None else None
        if validator is not None and not dry:
            decision = validator.evaluate(
                step.tool_name,
                step.tool_input,
                identity=identity,
                source="headroom_workflow",
                cwd=step.cwd or identity.root,
                enforcement=HOST_PRETOOL,
                call_id=call_id,
            )
            if decision.kind is not DecisionKind.ALLOW:
                raise WorkflowError(
                    f"workflow.contract.{decision.reason_code}",
                    f"{step.step_id} failed its tool contract ({decision.reason_code})",
                )
        changes = planned_changes(step.tool_name, step.tool_input)
        if changes and firewall is None:
            raise WorkflowError("workflow.scope_unavailable", f"{step.step_id} would change files")
        for change in changes:
            decision = firewall.authorize(
                identity, change, cwd=step.cwd or identity.root, enforceable=True
            )
            if decision.verdict is not Verdict.ALLOW or not firewall.verify(identity, decision):
                raise WorkflowError(
                    f"workflow.{decision.reason_code}", f"{step.step_id}: change scope refused it"
                )

    # -- execution ----------------------------------------------------------

    def handle(self, identity: Any, request: Mapping[str, Any]) -> dict[str, Any]:
        """Body of ``POST /v1/agent/workflow`` (dry run, list, or ticketed execution)."""
        macro_id = str(request.get("macro_id") or "")
        typed = (
            request.get("typed_inputs") if isinstance(request.get("typed_inputs"), Mapping) else {}
        )
        dry = bool(request.get("dry_run"))
        run_id = str(request.get("run_id") or uuid.uuid4().hex)
        if not identity.can_authorize_paths():
            return {"status": "unavailable", "reason": "workflow.untrusted_workspace"}
        if not macro_id:
            self.counters["dry_runs"] += 1
            return {
                "status": "listing",
                "workflows": [m.summary() for m in self.active(identity.workspace_id)],
            }
        macro = self.get(identity.workspace_id, macro_id)
        if macro is None:
            return {"status": "unavailable", "reason": "workflow.unknown", "macro_id": macro_id}
        fallback = {
            "run_with_normal_tools": [s.describe(d) for s, d in zip(macro.steps, macro.defaults())]
        }
        if macro.status != MacroStatus.ACTIVE.value:
            return {
                "status": "unavailable",
                "reason": f"workflow.{macro.status}",
                "macro_id": macro_id,
                "fallback": fallback,
            }
        reason = self._revalidate(identity, macro)
        if reason is not None:
            return {
                "status": "disabled",
                "reason": f"workflow.invalidated.{reason}",
                "macro_id": macro_id,
                "fallback": fallback,
            }
        planner = self.runtime.services.get("test_impact") if self.runtime is not None else None
        if dry:
            self.counters["dry_runs"] += 1
            plan = planner.plan(identity) if planner is not None else None
            try:
                steps = self.compile(identity, macro, typed, plan=plan)
                for step in steps:
                    self._authorize_step(
                        identity, step, call_id=f"{run_id}:{step.step_id}:dry", dry=True
                    )
            except WorkflowError as exc:
                return {
                    "status": "invalid",
                    "reason": exc.code,
                    "message": exc.message,
                    "macro_id": macro_id,
                }
            return {
                "status": "dry_run",
                "macro_id": macro_id,
                "version": macro.version,
                "steps": [
                    {
                        "id": s.step_id,
                        "family": s.template.family,
                        "run": s.display(),
                        "note": s.note,
                    }
                    for s in steps
                ],
                "verification_plan": plan.plan_id if plan is not None else None,
            }
        with self._lock:
            ticket = self._tickets.pop((identity.session_id, macro_id, _inputs_digest(typed)), None)
        if ticket is None or self._clock() - ticket.issued_at > TICKET_TTL_S:
            self.counters["fallback_hook_missing"] += 1
            return {
                "status": "fallback",
                "reason": "workflow.no_host_authorization",
                "message": "this workflow can only start behind the host's pre-tool permission hook",
                "macro_id": macro_id,
                "fallback": fallback,
            }
        if ticket.version != macro.version:
            self.counters["cancelled_stale"] += 1
            return {
                "status": "cancelled",
                "reason": "workflow.stale_version",
                "macro_id": macro_id,
                "fallback": fallback,
            }
        with self._lock:
            if identity.session_id in self._busy:
                return {
                    "status": "busy",
                    "reason": "workflow.already_running",
                    "macro_id": macro_id,
                }
            self._busy.add(identity.session_id)
            cancel = threading.Event()
            self._runs[run_id] = cancel
        try:
            return self._execute(identity, macro, typed, ticket, run_id, cancel, planner, fallback)
        finally:
            with self._lock:
                self._busy.discard(identity.session_id)
                self._runs.pop(run_id, None)

    def cancel(self, run_id: str) -> bool:
        with self._lock:
            event = self._runs.get(run_id)
        if event is None:
            return False
        event.set()
        return True

    def _execute(
        self,
        identity: Any,
        macro: Macro,
        typed: Mapping[str, Any],
        ticket: Ticket,
        run_id: str,
        cancel: threading.Event,
        planner: Any,
        fallback: Mapping[str, Any],
    ) -> dict[str, Any]:
        started = time.monotonic()
        deadline = started + MAX_MACRO_SECONDS
        self.counters["executions"] += 1
        self._flush()
        result: dict[str, Any] = {
            "run_id": run_id,
            "macro_id": macro.macro_id,
            "version": macro.version,
            "status": "completed",
            "reason": "",
            "steps": [],
        }
        baseline = ticket.pins
        if self._pins(identity) != baseline:
            self.counters["cancelled_stale"] += 1
            return {
                **result,
                "status": "cancelled",
                "reason": "workflow.stale_revision",
                "message": "the task or change scope changed after the host approved this call; nothing ran",
                "fallback": dict(fallback),
            }
        plan = planner.plan(identity) if planner is not None else None
        if (
            plan is not None
            and "tests" not in typed
            and any(s.kind == "verify" and s.framework == "pytest" for s in macro.steps)
        ):
            plan = planner.collect_validate(
                identity, plan, timeout_s=min(20.0, MAX_MACRO_SECONDS / 4), cancel=cancel
            )
        try:
            steps = self.compile(identity, macro, typed, plan=plan)
        except WorkflowError as exc:
            self.counters["cancelled_step_violation"] += 1
            return {
                **result,
                "status": "denied",
                "reason": exc.code,
                "message": exc.message,
                "fallback": dict(fallback),
            }
        verification: list[dict[str, Any]] = []
        for step in steps:
            entry: dict[str, Any] = {
                "id": step.step_id,
                "family": step.template.family,
                "run": step.display(),
            }
            if step.note:
                entry["note"] = step.note
            result["steps"].append(entry)
            if cancel.is_set():
                entry["status"] = "cancelled"
                result.update(status="cancelled", reason="workflow.cancelled")
                self.counters["cancelled"] += 1
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0.5:
                entry["status"] = "not_run"
                result.update(status="timeout", reason="workflow.deadline")
                self.counters["timeouts"] += 1
                break
            # Fresh revision compare immediately before the step: the user's
            # task (id + live user atoms) must be exactly what the host approved,
            # and nothing else may have moved since our previous step settled.
            pins = self._pins(identity)
            if (pins[0], pins[3]) != (ticket.pins[0], ticket.pins[3]) or pins != baseline:
                entry["status"] = "not_run"
                result.update(status="cancelled", reason="workflow.stale_revision")
                self.counters["cancelled_stale"] += 1
                break
            if (
                plan is not None
                and step.plan_tier
                and planner is not None
                and not planner.is_current(identity, plan)
            ):
                entry["status"] = "not_run"
                result.update(status="cancelled", reason="workflow.stale_verification_plan")
                planner.counters["stale_plan_refused"] += 1
                self.counters["cancelled_stale"] += 1
                break
            call_id = f"wf_{run_id[:12]}_{step.step_id}"
            try:
                self._authorize_step(identity, step, call_id=call_id)
            except WorkflowError as exc:
                entry.update(status="denied", reason=exc.code)
                result.update(status="denied", reason=exc.code, message=exc.message)
                self.counters["cancelled_step_violation"] += 1
                break
            step_deadline = min(MAX_STEP_SECONDS, remaining)
            outcome = self._run_step(identity, step, step_deadline, cancel)
            self.counters["steps_run"] += 1
            entry.update({k: v for k, v in outcome.items() if not k.startswith("_")})
            if outcome.get("_side_effect"):
                self.invalidate(macro.workspace_id, macro.macro_id, "side_effect_detected")
            if step.template.kind == "verify" and planner is not None and step.template.framework:
                planner_fw = step.template.framework
                if planner_fw != "canonical":
                    planner.expect_run(
                        call_id,
                        tier=step.plan_tier,
                        targets=step.plan_targets or tuple(step.values.get("tests") or ()),
                    )
            self._ingest(identity, step, call_id, outcome)
            self._flush(5.0)  # evidence / TaskState / verification update before the next step
            if step.template.kind == "verify" and planner is not None and plan is not None:
                fresh = planner.current(identity)
                if fresh is not None:
                    status = planner.check_status(fresh)
                    verification.append(
                        {
                            "plan_revision": fresh.plan_revision,
                            **status,
                            "required": list(fresh.required_checks),
                        }
                    )
            baseline = self._pins(identity)  # our own step's effects are the new baseline
            if outcome["status"] != "ok":
                result.update(
                    status="failed" if outcome["status"] != "timeout" else "timeout",
                    reason=f"workflow.step_{outcome['status']}",
                )
                break
        if result["status"] == "completed":
            self.counters["completed"] += 1
            saved = macro.stats()["turns_saved_estimate"]
            result["estimated_turns_avoided"] = saved
            self.counters["estimated_turns_avoided"] += saved
            with self._lock:
                macro.uses += 1
                macro.last_used = self._clock()
            self._persist(macro)
        elif result["status"] == "failed":
            self.counters["failed"] += 1
        if verification:
            result["verification"] = verification[-1]
        result["timing_ms"] = int((time.monotonic() - started) * 1000)
        if result["status"] != "completed":
            result["fallback"] = dict(fallback)
        return result

    def _run_step(
        self, identity: Any, step: CompiledStep, deadline_s: float, cancel: threading.Event
    ) -> dict[str, Any]:
        t0 = time.monotonic()
        kind = step.template.kind
        if kind == "read":
            text, ok, err = _read_step(identity.root, step.values)
            return self._step_result(step, text, 0 if ok else 1, t0, error=err)
        if kind == "glob":
            text, ok, err = _glob_step(identity.root, step.tool_input, deadline_s)
            return self._step_result(step, text, 0 if ok else 1, t0, error=err)
        if kind == "grep":
            text, ok, err = _grep_step(identity.root, step.tool_input, deadline_s)
            return self._step_result(step, text, 0 if ok else 1, t0, error=err)
        from headroom.agent.exec_runner import run_argv

        before = worktree_state(identity.root)
        run = run_argv(
            step.argv, cwd=step.cwd or identity.root, timeout_s=deadline_s, cancel=cancel
        )
        after = worktree_state(identity.root)
        side_effect = before is not None and after is not None and before != after
        status_override = None
        if run.timed_out:
            status_override = "timeout"
        elif run.cancelled:
            status_override = "cancelled"
        elif run.error:
            status_override = "error"
        out = self._step_result(
            step,
            run.output,
            run.exit_code,
            t0,
            error=run.error,
            status=status_override,
            truncated=run.truncated,
        )
        if side_effect:
            out["status"] = "side_effect"
            out["error"] = "a read-only/verification step changed the worktree; workflow disabled"
            out["_side_effect"] = True
        if run.tree_killed:
            out["process_tree_killed"] = True
        if kind == "verify" and step.template.framework and step.template.framework != "canonical":
            from headroom.agent.test_impact import parse_output

            outcome = parse_output(
                step.template.framework, run.output, run.exit_code, targets=step.plan_targets
            )
            out["tests"] = outcome.summary()
        return out

    def _step_result(
        self,
        step: CompiledStep,
        text: str,
        exit_code: int | None,
        t0: float,
        *,
        error: str | None = None,
        status: str | None = None,
        truncated: bool = False,
    ) -> dict[str, Any]:
        from headroom.agent.evidence import _ERROR_LINE

        ok = exit_code == 0 and error is None and status is None
        entry: dict[str, Any] = {
            "status": status or ("ok" if ok else "failed"),
            "exit_code": exit_code,
            "duration_ms": int((time.monotonic() - t0) * 1000),
            "_text": text,
        }
        if error:
            entry["error"] = error
        if truncated:
            entry["output_truncated"] = True
        if not ok:
            errors = [m.group(1).strip()[:200] for m in _ERROR_LINE.finditer(text[-20000:])]
            if errors:
                entry["errors"] = list(dict.fromkeys(errors))[:5]
        if len(text) > STEP_OUTPUT_EXTERNALIZE:
            entry["output"] = (
                text[:STEP_OUTPUT_PREVIEW]
                + f"\n… [{len(text) - 2 * STEP_OUTPUT_PREVIEW} chars elided] …\n"
                + text[-STEP_OUTPUT_PREVIEW:]
            )
            ccr = _externalize(text, step)
            if ccr:
                entry["ccr_hash"] = ccr
                entry["retrieve"] = f'headroom_retrieve(hash="{ccr}") returns the exact full output'
        else:
            entry["output"] = text
        return entry

    def _ingest(
        self, identity: Any, step: CompiledStep, call_id: str, outcome: Mapping[str, Any]
    ) -> None:
        """One TOOL_CALL + TOOL_RESULT (+ derived TEST_RESULT) per step via the normal path."""
        if self.runtime is None:
            return
        from headroom.agent.normalize import _Builder

        compiler = self.runtime.services.get("task_state")
        task_id = (
            compiler.active_task_id(identity.workspace_id, identity.session_id)
            if compiler is not None
            else None
        )
        builder = _Builder(identity, task_id)
        builder.tool_call(call_id, step.tool_name, step.tool_input)
        text = str(outcome.get("_text") or "")
        if step.argv:
            exit_code = outcome.get("exit_code")
            text = f"{text.rstrip()}\nExit code: {exit_code if exit_code is not None else -1}"
        builder.tool_result(call_id, text, is_error=outcome.get("status") != "ok")
        self.runtime.submit(builder.turn.envelopes)

    # -- status -------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._lock:
            by_status: dict[str, int] = {}
            for bucket in self._macros.values():
                for macro in bucket.values():
                    by_status[macro.status] = by_status.get(macro.status, 0) + 1
            return {
                "workspaces": len(self._macros),
                "macros": by_status,
                "limits": {
                    "max_active_per_workspace": MAX_ACTIVE_PER_WORKSPACE,
                    "max_steps": MAX_STEPS,
                    "max_seconds": MAX_MACRO_SECONDS,
                },
                "counters": dict(self.counters),
            }


def _fill_argv(template: Sequence[str], values: Mapping[str, Any]) -> tuple[str, ...]:
    out: list[str] = []
    for token in template:
        m = re.fullmatch(r"\{(\w+)\}", token)
        if not m:
            out.append(token)
            continue
        value = values.get(m.group(1))
        if value is None:
            continue
        if isinstance(value, list | tuple):
            out.extend(str(v) for v in value)
        else:
            out.append(str(value))
    return tuple(out)


def _tier_for_argv(plan: Any, framework: str, argv: Sequence[str]) -> int:
    for cmd in plan.commands:
        if cmd.framework == framework and tuple(cmd.argv) == tuple(argv):
            return cmd.tier
    return 0


_TEST_ID = re.compile(r"^[\w./:\[\]\-+=,@ ]+$")


def _check_value(kind: str, raw: Any, root: str) -> Any:
    from headroom.agent.change_scope import canonicalize, hard_protection

    def path(value: Any) -> str:
        if not isinstance(value, str) or not value or len(value) > 512 or value.startswith("-"):
            raise WorkflowError("workflow.input_invalid", "path inputs must be workspace paths")
        if "\x00" in value or "\n" in value:
            raise WorkflowError("workflow.input_invalid", "path contains control characters")
        canon = canonicalize(value, root=root, cwd=root)
        if canon.rel is None:
            raise WorkflowError(
                "workflow.input_outside_workspace", f"{value} is outside the workspace"
            )
        if hard_protection(canon.rel):
            raise WorkflowError("workflow.input_protected", f"{canon.rel} is a protected path")
        if not canon.exists:
            raise WorkflowError("workflow.input_missing", f"{canon.rel} does not exist")
        return canon.rel or "."

    if kind == "path":
        return path(raw)
    if kind == "paths":
        if not isinstance(raw, list | tuple) or len(raw) > 32:
            raise WorkflowError("workflow.input_invalid", "paths must be a list (≤32)")
        return [path(v) for v in raw]
    if kind == "int":
        if not isinstance(raw, int) or isinstance(raw, bool) or not (0 <= raw <= 10_000_000):
            raise WorkflowError("workflow.input_invalid", "integer input out of range")
        return raw
    if kind == "glob":
        if not isinstance(raw, str) or not _glob_ok(raw):
            raise WorkflowError("workflow.input_invalid", "glob must be a relative pattern")
        return raw
    if kind == "pattern":
        if not isinstance(raw, str) or not _pattern_ok(raw):
            raise WorkflowError(
                "workflow.input_invalid", "pattern must be a valid single-line regex"
            )
        return raw
    if kind == "tests":
        values = [raw] if isinstance(raw, str) else raw
        if not isinstance(values, list | tuple) or len(values) > 64:
            raise WorkflowError("workflow.input_invalid", "tests must be a list (≤64)")
        out = []
        for value in values:
            if (
                not isinstance(value, str)
                or not value
                or value.startswith("-")
                or not _TEST_ID.match(value)
            ):
                raise WorkflowError(
                    "workflow.input_invalid", "test ids must be plain node ids / names"
                )
            file_part = value.split("::", 1)[0]
            if "/" in file_part or file_part.endswith((".py", ".ts", ".js", ".tsx", ".jsx", ".rs")):
                rel = path(file_part)
                value = rel + value[len(file_part) :]
            out.append(value)
        return out
    raise WorkflowError("workflow.input_invalid", f"unsupported input type {kind}")


def _read_step(root: str, values: Mapping[str, Any]) -> tuple[str, bool, str | None]:
    full = os.path.join(root, values["path"])
    try:
        size = os.path.getsize(full)
        with open(full, "rb") as handle:
            data = handle.read(MAX_READ_BYTES)
    except OSError as exc:
        return "", False, f"read failed: {exc.strerror or exc}"
    text = data.decode("utf-8", "replace")
    lines = text.splitlines()
    offset = int(values.get("offset") or 1)
    start = max(0, offset - 1)
    limit = values.get("limit")
    end = start + int(limit) if isinstance(limit, int) and limit > 0 else len(lines)
    body = "\n".join(f"{i + 1:6}\t{line}" for i, line in enumerate(lines[start:end], start=start))
    if size > MAX_READ_BYTES:
        body += f"\n[file truncated at {MAX_READ_BYTES} bytes of {size}]"
    return body, True, None


def _walk(root: str, base: str, deadline: float) -> list[str]:
    ignore = {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "target",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
    }
    out: list[str] = []
    for current, dirs, files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d not in ignore)
        for name in sorted(files):
            out.append(os.path.relpath(os.path.join(current, name), root).replace("\\", "/"))
        if time.monotonic() > deadline or len(out) > 50_000:
            break
    return out


def _search_base(root: str, tool_input: Mapping[str, Any]) -> str:
    path = tool_input.get("path")
    return path if isinstance(path, str) and path else root


def _glob_step(
    root: str, tool_input: Mapping[str, Any], deadline_s: float
) -> tuple[str, bool, str | None]:
    deadline = time.monotonic() + deadline_s
    base = _search_base(root, tool_input)
    pattern = str(tool_input["pattern"])
    base_rel = os.path.relpath(base, root).replace("\\", "/")
    hits = []
    for rel in _walk(root, base, deadline):
        inner = rel if base_rel in (".", "") else rel[len(base_rel) + 1 :]
        if fnmatch.fnmatch(inner, pattern) or fnmatch.fnmatch(inner, pattern.replace("**/", "")):
            hits.append(os.path.join(root, rel).replace("\\", "/"))
    hits = hits[:MAX_GLOB_RESULTS]
    return ("\n".join(hits) if hits else "No files found"), True, None


def _grep_step(
    root: str, tool_input: Mapping[str, Any], deadline_s: float
) -> tuple[str, bool, str | None]:
    deadline = time.monotonic() + deadline_s
    base = _search_base(root, tool_input)
    flags = re.IGNORECASE if tool_input.get("-i") else 0
    try:
        regex = re.compile(str(tool_input["pattern"]), flags)
    except re.error as exc:
        return "", False, f"invalid pattern: {exc}"
    glob = tool_input.get("glob")
    mode = str(tool_input.get("output_mode") or "files_with_matches")
    out: list[str] = []
    counts: list[str] = []
    files = []
    if os.path.isfile(base):
        files = [os.path.relpath(base, root).replace("\\", "/")]
    else:
        files = _walk(root, base, deadline)
    for rel in files[:MAX_GREP_FILES]:
        if time.monotonic() > deadline:
            return "\n".join(out), False, "grep deadline exceeded"
        if (
            glob
            and not fnmatch.fnmatch(rel.rsplit("/", 1)[-1], str(glob))
            and not fnmatch.fnmatch(rel, str(glob))
        ):
            continue
        full = os.path.join(root, rel)
        try:
            if os.path.getsize(full) > 1024 * 1024:
                continue
            with open(full, encoding="utf-8", errors="strict") as handle:
                text = handle.read()
        except (OSError, UnicodeDecodeError):
            continue
        matched = 0
        for n, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                matched += 1
                if mode == "content":
                    out.append(
                        f"{full.replace(os.sep, '/')}:{n}:{line}"
                        if tool_input.get("-n")
                        else f"{full.replace(os.sep, '/')}:{line}"
                    )
        if matched:
            if mode == "files_with_matches":
                out.append(full.replace(os.sep, "/"))
            elif mode == "count":
                counts.append(f"{full.replace(os.sep, '/')}:{matched}")
    result = counts if mode == "count" else out
    limit = tool_input.get("head_limit")
    if isinstance(limit, int) and limit > 0:
        result = result[:limit]
    return ("\n".join(result) if result else "No matches found"), True, None


def _externalize(text: str, step: CompiledStep) -> str | None:
    """Exact original into CCR (read back verified) + span index for selective retrieval."""
    try:
        from headroom.cache.compression_store import get_compression_store

        store = get_compression_store()
        key = store.store(text, "", tool_name=step.tool_name, compression_strategy="workflow_step")
        if not store.verify_exact(key, text):
            return None
        try:
            from headroom.ccr.indexed import get_indexed_ccr

            get_indexed_ccr().ensure(store, key)
        except Exception as exc:  # noqa: BLE001 - selective retrieval is optional
            log.debug("workflow output index failed: %s", describe_exception(exc))
        return str(key)
    except Exception as exc:  # noqa: BLE001 - the inline preview still stands
        log.debug("workflow output externalization failed: %s", describe_exception(exc))
        return None


# ---------------------------------------------------------------------------
# Request stage (observation) and proxy endpoints
# ---------------------------------------------------------------------------


def observe_stage(ctx: Any, task: Any, runtime: Any) -> None:
    """Request stage (order 62): cut closed segments here, record them on the worker."""
    from headroom.agent.hook import identity_for

    registry: WorkflowMacroRegistry | None = runtime.services.get("workflow_macros")
    if registry is None:
        return
    identity = identity_for(ctx)
    if identity is None or not identity.can_authorize_paths():
        return
    provider = str(ctx.provider or "")
    if provider not in ("anthropic", "openai"):
        return
    segments = [s for s in extract_segments(provider, ctx.messages) if s.closed]
    if not segments:
        return
    tools = ctx.tools
    runtime.submit_task(lambda: registry.observe(identity, segments, tools=tools))


def _feature(config: Any, name: str) -> bool:
    rollout = getattr(config, "rollout", None)
    try:
        return bool(rollout is not None and rollout.is_enabled(name))
    except KeyError:
        return False


def _request_identity() -> Any:
    from headroom.agent.identity import resolve_identity
    from headroom.proxy.project_context import (
        get_current_project,
        get_registered_cwd,
        get_registered_session,
    )

    return resolve_identity(
        session_token=get_registered_session(),
        registered_cwd=get_registered_cwd(),
        attribution=get_current_project(),
    )


def _registry() -> WorkflowMacroRegistry | None:
    from headroom.agent.enforcement import active_runtime

    runtime = active_runtime()
    return runtime.services.get("workflow_macros") if runtime is not None else None


def handle_workflow_request(data: Mapping[str, Any], config: Any) -> dict[str, Any]:
    """``POST /v1/agent/workflow`` (loopback; identity from the wrap session token)."""
    if not _feature(config, FEATURE_WORKFLOW_MACROS):
        return {"status": "unavailable", "reason": "workflow.feature_disabled"}
    registry = _registry()
    if registry is None:
        return {"status": "unavailable", "reason": "workflow.runtime_unavailable"}
    try:
        return registry.handle(_request_identity(), data)
    except Exception as exc:  # noqa: BLE001 - surfaced as a result, never raised
        log.warning("workflow request failed: %s", describe_exception(exc))
        return {"status": "error", "reason": "workflow.internal_error"}


def handle_workflow_cancel(data: Mapping[str, Any], config: Any) -> dict[str, Any]:
    registry = _registry()
    run_id = str(data.get("run_id") or "")
    return {"cancelled": bool(registry is not None and run_id and registry.cancel(run_id))}


def workflow_status(config: Any) -> dict[str, Any]:
    enabled = _feature(config, FEATURE_WORKFLOW_MACROS)
    registry = _registry() if enabled else None
    block: dict[str, Any] = {"enabled": enabled, "tool": WORKFLOW_TOOL}
    if registry is not None:
        block.update(registry.status())
    return block


def register_local_tool() -> None:
    """``headroom_workflow`` in the shared tool registry (resident, stable schema)."""
    from headroom.agent.tool_catalog import REGISTRY, LocalTool

    REGISTRY.register(LocalTool(WORKFLOW_TOOL, resident=True, schema=WORKFLOW_TOOL_SCHEMA))


__all__ = [
    "AUTO_PROMOTABLE",
    "FEATURE_WORKFLOW_MACROS",
    "MAX_ACTIVE_PER_WORKSPACE",
    "MAX_MACRO_SECONDS",
    "MAX_STEPS",
    "Macro",
    "MacroStatus",
    "Safety",
    "Segment",
    "StepTemplate",
    "WORKFLOW_MCP_TOOL",
    "WORKFLOW_TOOL",
    "WORKFLOW_TOOL_SCHEMA",
    "WorkflowError",
    "WorkflowMacroRegistry",
    "extract_segments",
    "handle_workflow_cancel",
    "handle_workflow_request",
    "macro_id_for",
    "normalize_call",
    "observe_stage",
    "register_local_tool",
    "workflow_status",
    "worktree_state",
]
