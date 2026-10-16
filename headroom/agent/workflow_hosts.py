"""Host adapters for learned workflows (unified-21 #20).

The macro engine (:mod:`headroom.agent.workflow_macros`: observation,
promotion, tickets, compilation, ``exec_runner`` execution and the
``/v1/agent/workflow*`` endpoints) is client-independent. Everything that
depends on *which* wrapped client is driving the session lives here, in two
small interfaces:

* :class:`StepDialect` — how one host spells its built-in read/search tools
  and what they *mean*. A learned step that came from a host tool re-executes
  only through that host's dialect, which reproduces the host tool's
  semantics (for Claude Code's ripgrep-based Grep/Glob: ripgrep itself, with
  the flags the host was observed to use) or refuses the step. A dialect never
  approximates: a parameter, pattern class or environment whose host
  semantics it cannot reproduce makes the step ineligible for promotion, and
  an unavailable ``rg`` makes Grep/Glob steps ineligible rather than falling
  back to a different matcher.
* :class:`WorkflowHost` — whether the client has a genuine, Headroom-installed
  pre-tool permission boundary that can approve one ``headroom_workflow``
  call (single-use, revision-pinned ticket), how that hook's payload is read
  and how a refusal is rendered in the host's own hook protocol. Clients
  without one are observe-only: their tool segments are still learned from
  (proxy path), but Headroom never executes a workflow for them and never
  claims enforcement it does not have.

Capability table (see :func:`capability_table`; also on
``/health.agent_verification.workflows.hosts``):

=========  ====================  ==================================================
client     capability            why
=========  ====================  ==================================================
claude     host_pretool_ticket   ``wrap claude`` installs a PreToolUse command hook
                                 for ``mcp__headroom__headroom_workflow``; the MCP
                                 call itself runs only after Claude Code's own
                                 permission check.
codex      observe_only          ``wrap codex`` registers the MCP server and base
                                 URL only; no pre-tool hook reaches Headroom.
opencode   observe_only          the wrap plugin registers ``headroom_retrieve``,
                                 the transport and ``shell.env`` only; it installs
                                 no pre-tool permission callback.
others     observe_only          no host hook integration in ``wrap``.
=========  ====================  ==================================================
"""

from __future__ import annotations

import os
import re
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

TICKETED = "host_pretool_ticket"
OBSERVE_ONLY = "observe_only"

WORKFLOW_TOOL = "headroom_workflow"
#: Claude Code's name for the Headroom MCP server's workflow tool.
CLAUDE_WORKFLOW_TOOL = "mcp__headroom__headroom_workflow"

CLAUDE_DIALECT = "claude_code"

# ---------------------------------------------------------------------------
# Claude Code built-in tool semantics (observed against Claude Code 2.1.301 on
# Windows and re-checked on Linux; see tests/agent/test_workflow_search_parity.py)
# ---------------------------------------------------------------------------

#: Major.minor series of Claude Code whose Grep/Glob behaviour was verified.
CLAUDE_VERIFIED_SERIES = frozenset({"2.1"})
#: Grep searches hidden files but never VCS metadata directories.
RG_VCS_EXCLUDES = (".git", ".svn", ".hg", ".bzr", ".jj", ".sl")
#: Grep replaces matching lines longer than this with "[Omitted long matching line]".
GREP_MAX_COLUMNS = 500
#: Grep's default ``head_limit`` (``0`` means unlimited).
GREP_DEFAULT_HEAD_LIMIT = 250
#: Glob lists at most this many files (oldest first) and says it truncated.
GLOB_MAX_RESULTS = 100
#: Both tools give ripgrep this long before reporting a timeout.
SEARCH_TIMEOUT_S = 20.0
#: Read returns this many lines when no ``limit`` is given.
READ_DEFAULT_LIMIT = 2000
READ_MAX_BYTES = 2 * 1024 * 1024

_GREP_CONSTANT_KEYS = frozenset(
    {"output_mode", "-i", "-n", "multiline", "type", "head_limit", "offset"}
)
_OUTPUT_MODES = frozenset({"content", "files_with_matches", "count"})
_RG_TYPE = re.compile(r"^[A-Za-z0-9_+\-]{1,32}$")
_AGENT_VERSION = re.compile(r"claude-code[_-](\d+)[-.](\d+)[-.](\d+)", re.I)


def ripgrep() -> str | None:
    """The ``rg`` a Grep/Glob step would run, or None (steps are then ineligible)."""
    from headroom.agent.exec_runner import resolve_executable

    return resolve_executable("rg")


def search_unsupported_reason() -> str | None:
    """Why ripgrep-backed steps cannot reproduce the host here (None when they can)."""
    if ripgrep() is None:
        return "ripgrep_unavailable"
    if os.environ.get("RIPGREP_CONFIG_PATH"):
        # The host may or may not read the operator's ripgrep config; Headroom
        # always passes --no-config, so the two could differ.
        return "ripgrep_config_present"
    return None


def rg_pattern_ok(pattern: Any) -> bool:
    """A ripgrep regex Headroom will pass through unchanged (ripgrep validates it)."""
    return (
        isinstance(pattern, str)
        and 0 < len(pattern) <= 300
        and "\x00" not in pattern
        and "\n" not in pattern
        and "\r" not in pattern
        and not pattern.startswith("-")
    )


def rg_glob_ok(pattern: Any) -> bool:
    """A glob whose ripgrep match does not depend on how the host roots the search.

    ripgrep matches a glob containing ``/`` against the *displayed* path, which
    starts with however the host spelled the search root (observed: Claude
    Code's Grep/Glob find nothing for ``src/*.py`` with a ``path`` argument).
    Only slash-free globs (basename at any depth) and globs that start with
    ``**/`` are prefix-independent; any other slash glob is ineligible.
    """
    from headroom.agent.workflow_macros import _glob_ok

    if not isinstance(pattern, str) or not _glob_ok(pattern):
        return False
    if any(c.isspace() for c in pattern) or "\\" in pattern:
        return False
    if pattern.startswith("!"):
        return False
    return "/" not in pattern or pattern.startswith("**/")


@dataclass(frozen=True)
class StepRun:
    text: str
    ok: bool
    error: str | None = None
    partial: bool = False


class StepDialect(ABC):
    """One host's built-in read/search tools (see module docstring)."""

    name = ""

    @abstractmethod
    def normalize(self, tool_name: str, args: Mapping[str, Any], root: str | None) -> Any:
        """``NormalizedStep`` for this host's tool, else None (not this dialect's tool)."""

    @abstractmethod
    def compile_input(
        self, template: Any, values: Mapping[str, Any], root: str
    ) -> tuple[str, dict[str, Any]]:
        """``(host tool name, host tool input)`` of a compiled step."""

    @abstractmethod
    def run(
        self,
        kind: str,
        root: str,
        tool_input: Mapping[str, Any],
        deadline_s: float,
        cancel: threading.Event | None = None,
    ) -> StepRun:
        """Execute one compiled step with the host tool's semantics."""

    @abstractmethod
    def fingerprint(self, template: Any) -> str | None:
        """Executables/semantics a step depends on (None: the step cannot run here)."""

    def host_semantics_ok(self, template: Any, host_version: str | None) -> bool:
        return True


class ClaudeCodeDialect(StepDialect):
    """Claude Code ``Read`` / ``Grep`` / ``Glob`` (exact, case-sensitive names)."""

    name = CLAUDE_DIALECT

    def normalize(self, tool_name: str, args: Mapping[str, Any], root: str | None) -> Any:
        from headroom.agent.workflow_macros import (
            NormalizedStep,
            Safety,
            SlotSpec,
            StepTemplate,
            _rel_value,
            _unsupported,
        )

        if tool_name not in ("Read", "Grep", "Glob"):
            return None
        args = args if isinstance(args, Mapping) else {}
        if tool_name == "Read":
            rel = _rel_value(args.get("file_path"), root)
            if rel is None:
                return _unsupported(tool_name, "read.outside")
            values: dict[str, Any] = {"path": rel}
            slots = [SlotSpec("path", "path")]
            for key in ("offset", "limit"):
                value = args.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    values[key] = int(value)
                    slots.append(SlotSpec(key, "int", required=False))
            extra = set(args) - {"file_path", "offset", "limit"}
            if extra:
                return _unsupported(tool_name, "read.options")
            return NormalizedStep(
                StepTemplate(
                    "read",
                    "read.file",
                    "Read",
                    (),
                    tuple(slots),
                    Safety.READ_ONLY.value,
                    dialect=self.name,
                ),
                values,
            )
        family = "search.grep" if tool_name == "Grep" else "search.glob"
        blocked = search_unsupported_reason()
        if blocked is not None:
            return _unsupported(tool_name, f"{family}.{blocked}")
        pattern = args.get("pattern")
        if tool_name == "Glob":
            if not rg_glob_ok(pattern) or set(args) - {"pattern", "path"}:
                return _unsupported(tool_name, family)
            values = {"pattern": pattern}
            slots = [SlotSpec("pattern", "rg_glob")]
        else:
            if not rg_pattern_ok(pattern):
                return _unsupported(tool_name, family)
            values = {"pattern": pattern}
            slots = [SlotSpec("pattern", "rg_pattern")]
        constants: list[str] = []
        for key in sorted(args):
            value = args[key]
            if key == "pattern":
                continue
            if key == "path":
                if value in (None, ""):
                    continue
                rel = _rel_value(value, root)
                if rel is None:
                    return _unsupported(tool_name, f"{family}.outside")
                values["path"] = rel or "."
                slots.append(SlotSpec("path", "path", required=False))
            elif tool_name == "Grep" and key == "glob":
                if not rg_glob_ok(value):
                    return _unsupported(tool_name, f"{family}.glob")
                values["glob"] = value
                slots.append(SlotSpec("glob", "rg_glob", required=False))
            elif tool_name == "Grep" and key in _GREP_CONSTANT_KEYS and _constant_ok(key, value):
                constants.append(f"{key}={_json(value)}")
            else:
                # Context lines (-A/-B/-C), pagination forms and anything new
                # were not verified against the host: never generalized.
                return _unsupported(tool_name, f"{family}.option")
        return NormalizedStep(
            StepTemplate(
                "grep" if tool_name == "Grep" else "glob",
                family,
                tool_name,
                tuple(constants),
                tuple(slots),
                Safety.READ_ONLY.value,
                dialect=self.name,
            ),
            values,
        )

    def compile_input(
        self, template: Any, values: Mapping[str, Any], root: str
    ) -> tuple[str, dict[str, Any]]:
        import json

        def absolute(rel: str) -> str:
            return os.path.join(root, rel).replace("\\", "/") if rel not in ("", ".") else root

        if template.kind == "read":
            tool_input: dict[str, Any] = {"file_path": absolute(values["path"])}
            for key in ("offset", "limit"):
                if key in values:
                    tool_input[key] = values[key]
            return "Read", tool_input
        tool_input = {"pattern": values["pattern"]}
        if "path" in values:
            tool_input["path"] = absolute(values["path"])
        if "glob" in values:
            tool_input["glob"] = values["glob"]
        for const in template.argv:
            key, _, raw = const.partition("=")
            tool_input[key] = json.loads(raw)
        return ("Grep" if template.kind == "grep" else "Glob"), tool_input

    def fingerprint(self, template: Any) -> str | None:
        if template.kind == "read":
            return f"{self.name}:read:v1"
        if search_unsupported_reason() is not None:
            return None
        path = ripgrep()
        try:
            st = os.stat(path or "")
        except OSError:
            return None
        return f"{self.name}:{template.kind}:v1:{os.path.normcase(str(path))}:{st.st_size}:{st.st_mtime_ns}"

    def host_semantics_ok(self, template: Any, host_version: str | None) -> bool:
        if template.kind not in ("grep", "glob"):
            return True
        series = ".".join((host_version or "").split(".")[:2])
        return series in CLAUDE_VERIFIED_SERIES

    def run(
        self,
        kind: str,
        root: str,
        tool_input: Mapping[str, Any],
        deadline_s: float,
        cancel: threading.Event | None = None,
    ) -> StepRun:
        if kind == "read":
            return _claude_read(tool_input)
        if kind == "grep":
            return _claude_grep(root, tool_input, deadline_s, cancel)
        if kind == "glob":
            return _claude_glob(root, tool_input, deadline_s, cancel)
        return StepRun("", False, f"unsupported step kind {kind}")


def _json(value: Any) -> str:
    import json

    return json.dumps(value)


def _constant_ok(key: str, value: Any) -> bool:
    if key == "output_mode":
        return value in _OUTPUT_MODES
    if key in ("-i", "-n", "multiline"):
        return isinstance(value, bool)
    if key in ("head_limit", "offset"):
        return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 100_000
    if key == "type":
        return isinstance(value, str) and bool(_RG_TYPE.match(value))
    return False


def _claude_read(tool_input: Mapping[str, Any]) -> StepRun:
    """Claude Code ``Read``: ``cat -n`` numbering, 1-based offset, 2000 lines by default."""
    full = str(tool_input["file_path"])
    try:
        size = os.path.getsize(full)
        with open(full, "rb") as handle:
            data = handle.read(READ_MAX_BYTES)
    except OSError as exc:
        return StepRun("", False, f"read failed: {exc.strerror or exc}")
    lines = data.decode("utf-8", "replace").splitlines()
    start = max(0, int(tool_input.get("offset") or 1) - 1)
    limit = tool_input.get("limit")
    count = int(limit) if isinstance(limit, int) and limit > 0 else READ_DEFAULT_LIMIT
    end = min(len(lines), start + count)
    body = "\n".join(f"{i + 1:6}\t{line}" for i, line in enumerate(lines[start:end], start=start))
    partial = False
    if end < len(lines):
        body += f"\n[showing lines {start + 1}-{end} of {len(lines)}]"
        partial = True
    if size > READ_MAX_BYTES:
        body += f"\n[file truncated at {READ_MAX_BYTES} bytes of {size}]"
        partial = True
    return StepRun(body, True, None, partial)


def _rg_rel(root: str, tool_input: Mapping[str, Any]) -> str | None:
    path = tool_input.get("path")
    if not isinstance(path, str) or not path:
        return "."
    rel = os.path.relpath(path, root).replace("\\", "/")
    if rel == ".." or rel.startswith("../") or os.path.isabs(rel):
        return None
    return rel


def _rg(
    root: str, argv: list[str], deadline_s: float, cancel: threading.Event | None
) -> tuple[Any, str | None]:
    from headroom.agent.exec_runner import run_argv

    timeout = max(0.5, min(float(deadline_s), SEARCH_TIMEOUT_S))
    run = run_argv(argv, cwd=root, timeout_s=timeout, cancel=cancel, merge_stderr=False)
    if run.timed_out:
        return run, f"ripgrep search timed out after {int(timeout)} seconds"
    if run.cancelled:
        return run, "cancelled"
    if run.error:
        return run, run.error
    if run.exit_code not in (0, 1):
        detail = " ".join(
            ln.strip() for ln in (run.stderr or run.output).splitlines() if ln.strip()
        )
        return run, "ripgrep failed: " + (detail[:600] or f"exit {run.exit_code}")
    if run.truncated:
        return run, "ripgrep output exceeded the capture bound"
    return run, None


def _mtime_ns(root: str, rel: str) -> int:
    try:
        return os.stat(os.path.join(root, rel)).st_mtime_ns
    except OSError:
        return 0


def _paginate(items: list[str], tool_input: Mapping[str, Any]) -> tuple[list[str], bool]:
    head = tool_input.get("head_limit", GREP_DEFAULT_HEAD_LIMIT)
    head = GREP_DEFAULT_HEAD_LIMIT if not isinstance(head, int) else head
    offset = tool_input.get("offset") or 0
    offset = offset if isinstance(offset, int) and offset > 0 else 0
    window = items[offset:] if head == 0 else items[offset : offset + head]
    return window, len(items) > offset + len(window)


def grep_argv(rg: str, root: str, tool_input: Mapping[str, Any]) -> list[str] | None:
    """The ripgrep argv equivalent to one Claude Code Grep call (None: not reproducible)."""
    rel = _rg_rel(root, tool_input)
    if rel is None:
        return None
    mode = str(tool_input.get("output_mode") or "files_with_matches")
    argv = [rg, "--no-config", "--hidden", "--path-separator", "/"]
    for vcs in RG_VCS_EXCLUDES:
        argv += ["--glob", f"!{vcs}"]
    argv += ["--max-columns", str(GREP_MAX_COLUMNS)]
    if mode == "files_with_matches":
        argv.append("--files-with-matches")
    elif mode == "count":
        argv += ["--count", "--sort", "path"]
    else:
        argv += ["--no-heading", "--sort", "path"]
        if tool_input.get("-n", True):
            argv.append("--line-number")
    if tool_input.get("-i"):
        argv.append("--ignore-case")
    if tool_input.get("type"):
        argv += ["--type", str(tool_input["type"])]
    if tool_input.get("glob"):
        argv += ["--glob", str(tool_input["glob"])]
    if tool_input.get("multiline"):
        argv += ["--multiline", "--multiline-dotall"]
    argv += ["--regexp", str(tool_input["pattern"])]
    if rel != ".":
        argv += ["--", rel]
    return argv


def glob_argv(rg: str, root: str, tool_input: Mapping[str, Any]) -> list[str] | None:
    """The ripgrep argv equivalent to one Claude Code Glob call (None: not reproducible)."""
    rel = _rg_rel(root, tool_input)
    if rel is None:
        return None
    argv = [
        rg,
        "--no-config",
        "--files",
        "--hidden",
        "--no-ignore",
        "--path-separator",
        "/",
        "--glob",
        str(tool_input["pattern"]),
    ]
    if rel != ".":
        argv += ["--", rel]
    return argv


def _claude_grep(
    root: str, tool_input: Mapping[str, Any], deadline_s: float, cancel: threading.Event | None
) -> StepRun:
    blocked = search_unsupported_reason()
    if blocked is not None:
        return StepRun("", False, f"grep step unavailable: {blocked}")
    if not rg_pattern_ok(tool_input.get("pattern")):
        return StepRun("", False, "pattern not eligible")
    if tool_input.get("glob") and not rg_glob_ok(tool_input.get("glob")):
        return StepRun("", False, "glob not eligible")
    argv = grep_argv(str(ripgrep()), root, tool_input)
    if argv is None:
        return StepRun("", False, "path outside the workspace")
    run, error = _rg(root, argv, deadline_s, cancel)
    if error is not None:
        return StepRun("", False, error)
    mode = str(tool_input.get("output_mode") or "files_with_matches")
    lines = [ln for ln in run.output.splitlines() if ln]
    if not lines:
        return StepRun(
            "No files found" if mode == "files_with_matches" else "No matches found", True
        )
    if mode == "files_with_matches":
        lines.sort(key=lambda rel: (-_mtime_ns(root, rel), rel))  # newest first
    window, more = _paginate(lines, tool_input)
    if mode == "files_with_matches":
        noun = "file" if len(window) == 1 else "files"
        text = f"Found {len(window)} {noun}\n" + "\n".join(window)
    elif mode == "count":
        total = 0
        for line in window:
            tail = line.rsplit(":", 1)[-1]
            total += int(tail) if tail.isdigit() else 0
        text = "\n".join(window) + (
            f"\n\nFound {total} total occurrence{'s' if total != 1 else ''} across "
            f"{len(window)} file{'s' if len(window) != 1 else ''}."
        )
    else:
        text = "\n".join(window)
    if more:
        text += f"\n\n[results truncated: {len(window)} of {len(lines)} shown]"
    return StepRun(text, True, None, more)


def _claude_glob(
    root: str, tool_input: Mapping[str, Any], deadline_s: float, cancel: threading.Event | None
) -> StepRun:
    blocked = search_unsupported_reason()
    if blocked is not None:
        return StepRun("", False, f"glob step unavailable: {blocked}")
    if not rg_glob_ok(tool_input.get("pattern")):
        return StepRun("", False, "glob not eligible")
    argv = glob_argv(str(ripgrep()), root, tool_input)
    if argv is None:
        return StepRun("", False, "path outside the workspace")
    run, error = _rg(root, argv, deadline_s, cancel)
    if error is not None:
        return StepRun("", False, error)
    files = sorted(
        (ln for ln in run.output.splitlines() if ln),
        key=lambda rel: (_mtime_ns(root, rel), rel),  # oldest first
    )
    if not files:
        return StepRun("No files found", True)
    shown = files[:GLOB_MAX_RESULTS]
    text = "\n".join(shown)
    more = len(files) > len(shown)
    if more:
        text += f"\n(Showing {len(shown)} of {len(files)} matching files)"
    return StepRun(text, True, None, more)


DIALECTS: dict[str, StepDialect] = {CLAUDE_DIALECT: ClaudeCodeDialect()}


def dialect(name: str) -> StepDialect | None:
    return DIALECTS.get(name)


def normalize_host_tool(tool_name: str, args: Mapping[str, Any], root: str | None) -> Any:
    """The dialect normalization of a host built-in tool call, else None."""
    for item in DIALECTS.values():
        step = item.normalize(tool_name, args, root)
        if step is not None:
            return step
    return None


# ---------------------------------------------------------------------------
# Host permission boundaries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PretoolCall:
    """One ``headroom_workflow`` call as a host pre-tool hook saw it."""

    macro_id: str
    typed_inputs: Mapping[str, Any]
    dry_run: bool
    call_id: str
    host_version: str | None = None


@dataclass(frozen=True)
class WorkflowHost:
    client: str
    capability: str
    reason: str
    workflow_tool_names: frozenset[str] = frozenset()
    #: MCP ``clientInfo.name`` values this host identifies itself with.
    mcp_client_names: frozenset[str] = frozenset()
    dialect: str = ""

    @property
    def ticketed(self) -> bool:
        return self.capability == TICKETED

    def parse_pretool(
        self, payload: Mapping[str, Any], host: Mapping[str, Any] | None = None
    ) -> PretoolCall | None:
        return None

    def render_refusal(self, message: str) -> Any:
        return None

    def describe(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "executes_workflows": self.ticketed,
            "reason": self.reason,
            "tool_names": sorted(self.workflow_tool_names),
            "step_dialect": self.dialect or None,
        }


def host_version_of(host: Mapping[str, Any] | None) -> str | None:
    """``2.1.300`` from the ``AI_AGENT`` value the host hook forwards, else None."""
    agent = str((host or {}).get("agent") or "")
    m = _AGENT_VERSION.search(agent)
    return ".".join(m.groups()) if m else None


@dataclass(frozen=True)
class ClaudeCodeHost(WorkflowHost):
    """Claude Code: the PreToolUse command hook ``wrap claude`` installs."""

    def parse_pretool(
        self, payload: Mapping[str, Any], host: Mapping[str, Any] | None = None
    ) -> PretoolCall | None:
        if payload.get("hook_event_name") not in (None, "PreToolUse"):
            return None
        if str(payload.get("tool_name") or "") not in self.workflow_tool_names:
            return None
        tool_input = payload.get("tool_input")
        tool_input = tool_input if isinstance(tool_input, Mapping) else {}
        typed = tool_input.get("typed_inputs")
        return PretoolCall(
            macro_id=str(tool_input.get("macro_id") or ""),
            typed_inputs=typed if isinstance(typed, Mapping) else {},
            dry_run=bool(tool_input.get("dry_run")),
            call_id=str(payload.get("tool_use_id") or ""),
            host_version=host_version_of(host),
        )

    def render_refusal(self, message: str) -> Any:
        from headroom.agent.enforcement import render_pretool

        return render_pretool("BLOCK", message)


def _observe(client: str, reason: str) -> WorkflowHost:
    return WorkflowHost(client, OBSERVE_ONLY, reason)


HOSTS: dict[str, WorkflowHost] = {
    "claude": ClaudeCodeHost(
        "claude",
        TICKETED,
        "wrap installs a PreToolUse command hook for the workflow tool that validates each "
        "call and issues a single-use, revision-pinned ticket; the MCP call itself runs only "
        "after Claude Code's own permission check",
        frozenset({CLAUDE_WORKFLOW_TOOL}),
        frozenset({"claude-code"}),
        CLAUDE_DIALECT,
    ),
    "codex": _observe(
        "codex",
        "wrap codex registers the MCP server and base URL only; no pre-tool hook reaches "
        "Headroom, so no host boundary can approve a workflow call",
    ),
    "opencode": _observe(
        "opencode",
        "the wrap plugin registers headroom_retrieve, the transport and shell.env only; it "
        "installs no pre-tool permission callback, so workflows are observed only",
    ),
}
_DEFAULT_REASON = "wrap installs no host pre-tool hook for this client"


def workflow_host(client: str | None) -> WorkflowHost:
    name = (client or "").lower()
    return HOSTS.get(name) or _observe(name or "unknown", _DEFAULT_REASON)


def host_for_mcp_client(name: str | None) -> WorkflowHost | None:
    """The ticketing host an MCP client identifies as, else None."""
    lname = (name or "").lower()
    for host in HOSTS.values():
        if lname and lname in host.mcp_client_names:
            return host
    return None


def capability_table() -> dict[str, dict[str, Any]]:
    clients = ("claude", "codex", "opencode", "copilot", "cursor", "aider", "gemini")
    return {c: workflow_host(c).describe() for c in clients}


__all__ = [
    "CLAUDE_DIALECT",
    "CLAUDE_VERIFIED_SERIES",
    "CLAUDE_WORKFLOW_TOOL",
    "DIALECTS",
    "HOSTS",
    "OBSERVE_ONLY",
    "PretoolCall",
    "StepDialect",
    "StepRun",
    "TICKETED",
    "WorkflowHost",
    "capability_table",
    "dialect",
    "glob_argv",
    "grep_argv",
    "host_for_mcp_client",
    "host_version_of",
    "normalize_host_tool",
    "rg_glob_ok",
    "rg_pattern_ok",
    "ripgrep",
    "search_unsupported_reason",
    "workflow_host",
]
