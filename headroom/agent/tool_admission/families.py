"""Tool families, stable resource identity and envelope construction (unified-21 #4/#14).

Every observed tool result, from a host hook or from the proxy live zone, is
normalized here into one :class:`~headroom.agent.interfaces.ToolResultEnvelope`.

``resource_id`` names *what* was observed, so two calls that read the same
thing share it and the delta engine can compare versions:

* Read — canonical workspace-relative path plus the requested range;
* shell/test/git — the normalized command plus the workspace-relative cwd;
* search — query plus scope (path, glob, type, mode, flags);
* listing — pattern plus scope;
* JSON/API — tool name plus the canonical JSON of its arguments.

The tool-call id is **never** part of resource identity. Ids are a digest of
the canonical key (bounded, no raw paths in storage); the readable key is kept
in ``source_metadata["resource_key"]`` for diagnostics only.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
from collections.abc import Mapping
from enum import Enum
from typing import Any

from headroom.agent.interfaces import ToolResultEnvelope


class ToolFamily(str, Enum):
    READ = "read"
    SEARCH = "search"
    LISTING = "listing"
    SHELL = "shell"
    TEST = "test"
    GIT = "git"
    JSON = "json"
    OTHER = "other"


#: The seven intercepted families (OTHER is observed, never rewritten).
INTERCEPTED_FAMILIES: tuple[ToolFamily, ...] = (
    ToolFamily.READ,
    ToolFamily.SEARCH,
    ToolFamily.LISTING,
    ToolFamily.SHELL,
    ToolFamily.TEST,
    ToolFamily.GIT,
    ToolFamily.JSON,
)

READ_TOOLS = frozenset(
    {"read", "read_file", "view", "cat", "readfile", "open_file", "view_file", "read_text_file"}
)
SEARCH_TOOLS = frozenset(
    {
        "grep",
        "search",
        "rg",
        "ripgrep",
        "grep_search",
        "search_files",
        "codebase_search",
        "find_in_files",
        "search_for_pattern",
    }
)
LISTING_TOOLS = frozenset(
    {
        "glob",
        "ls",
        "list_directory",
        "list_dir",
        "list_files",
        "find_files",
        "file_search",
        "glob_files",
        "find_file",
    }
)
SHELL_TOOLS = frozenset(
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
    }
)
EDIT_TOOLS = frozenset(
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
        "apply_patch",
    }
)
#: Read-range keys (Claude Code offset/limit, Cursor line_range, Aider
#: start/end, Codex ranges, PDF pages). Same provenance as the ast-grep
#: interceptor's list, plus ``pages``.
RANGE_KEYS = ("offset", "limit", "line_range", "start_line", "end_line", "ranges", "pages")
PATH_KEYS = ("file_path", "path", "filePath", "filename", "notebook_path", "target_file", "file")

_TEST_COMMAND = re.compile(
    r"(?:^|[\s;&|(])(?:python\d?(?:\.\d+)?\s+-m\s+)?(pytest|py\.test|cargo\s+(?:test|nextest)|"
    r"npm\s+(?:run\s+)?test|yarn\s+test|pnpm\s+(?:run\s+)?test|bun\s+test|go\s+test|jest|"
    r"vitest|mvn\s+test|gradle\s+test|ctest|tox|nox|unittest|rspec|phpunit|dotnet\s+test)\b"
)
_GIT_VIEW = re.compile(
    r"^git\s+(?:-C\s+\S+\s+)?(?:-c\s+\S+\s+)*(?:--no-pager\s+)?(diff|status|show|log|stash\s+show)\b"
)
_SEARCH_CMD = re.compile(
    r"^(?:rg|grep|egrep|fgrep|ag|ack|git\s+grep|findstr|select-string|sls)\b", re.IGNORECASE
)
_LIST_CMD = re.compile(
    r"^(?:ls|dir|tree|find|fd|fdfind|get-childitem|gci|git\s+ls-files)\b", re.IGNORECASE
)
_READ_CMD = re.compile(
    r"^(?:cat|type|get-content|gc|bat)\s+(?P<path>[^\s|;&<>]+)\s*$", re.IGNORECASE
)
#: A pipe into one of these means the visible output is a cut of the real one.
_PIPE_CUT = re.compile(
    r"\|\s*(?:head|tail|less|more|select-object\s+-(?:first|last)|sed\s+-n|awk\s+'NR)",
    re.IGNORECASE,
)
_PRELUDE = re.compile(
    r"^(?:cd|pushd|popd|set|export|source|\.|chdir|set-location|sl)\b", re.IGNORECASE
)
_SEGMENT_SPLIT = re.compile(r"\s*(?:&&|\|\||;)\s*")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def digest(text: str, n: int = 24) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:n]


def parse_args(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, Mapping) else {}
    return {}


def command_of(tool_name: str, args: Mapping[str, Any]) -> str | None:
    """The shell command a shell-family tool ran (string or argv list)."""
    if (tool_name or "").lower() not in SHELL_TOOLS:
        return None
    command = args.get("command") or args.get("cmd")
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    if isinstance(command, str):
        return command
    action = args.get("action")
    if isinstance(action, Mapping):
        return command_of(tool_name, action)
    return None


def path_of(args: Mapping[str, Any]) -> str | None:
    for key in PATH_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def normalize_command(command: str) -> str:
    return " ".join(command.split())


def primary_command(command: str) -> str:
    """First meaningful segment: skips ``cd``/``export`` preludes, keeps the pipe head."""
    for segment in _SEGMENT_SPLIT.split(command.strip()):
        seg = segment.strip()
        if not seg or _PRELUDE.match(seg):
            continue
        return seg.split("|", 1)[0].strip()
    return command.strip()


def workspace_relative(path: str, *, root: str | None, cwd: str | None = None) -> str:
    """Canonical workspace-relative form of ``path`` (``ext:<digest>`` outside it).

    Separators are normalized and case is folded on drive-letter paths, so the
    same file named two ways shares one identity.
    """
    norm = path.replace("\\", "/").strip()
    absolute = bool(re.match(r"^(?:/|[A-Za-z]:/|//)", norm))
    if not absolute and cwd:
        base = cwd.replace("\\", "/").rstrip("/")
        norm = f"{base}/{norm}"
        absolute = True
    if absolute:
        windows = bool(re.match(r"^[A-Za-z]:/", norm)) or norm.startswith("//")
        collapsed = posixpath.normpath(norm)
        if root:
            root_norm = posixpath.normpath(root.replace("\\", "/").rstrip("/"))
            a, b = (collapsed.lower(), root_norm.lower()) if windows else (collapsed, root_norm)
            if a == b:
                return "."
            if a.startswith(b + "/"):
                return collapsed[len(root_norm) + 1 :]
        return "ext:" + digest(collapsed.lower() if windows else collapsed)
    rel = posixpath.normpath(norm)
    return re.sub(r"^(?:\./)+", "", rel)


def _range_of(args: Mapping[str, Any]) -> dict[str, Any]:
    return {k: args[k] for k in RANGE_KEYS if k in args and args[k] not in (None, "")}


def classify(tool_name: str | None, args: Mapping[str, Any], content: str) -> ToolFamily:
    """Map a tool call to its family (provider-neutral; unknown tools observed only)."""
    lname = (tool_name or "").lower()
    if lname.startswith("mcp__headroom") or lname in {"headroom_retrieve", "headroom_compress"}:
        return ToolFamily.OTHER  # never re-admit Headroom's own retrieval output
    if lname in READ_TOOLS and path_of(args):
        return ToolFamily.READ
    if lname in SEARCH_TOOLS:
        return ToolFamily.SEARCH
    if lname in LISTING_TOOLS:
        return ToolFamily.LISTING
    command = command_of(lname, args)
    if command is not None:
        if _TEST_COMMAND.search(command):
            return ToolFamily.TEST
        head = primary_command(command)
        if _GIT_VIEW.match(head):
            return ToolFamily.GIT
        if _SEARCH_CMD.match(head):
            return ToolFamily.SEARCH
        if _LIST_CMD.match(head):
            return ToolFamily.LISTING
        if _READ_CMD.match(head):
            return ToolFamily.READ
        if looks_like_json(content):
            return ToolFamily.JSON
        return ToolFamily.SHELL
    if looks_like_json(content):
        return ToolFamily.JSON
    return ToolFamily.OTHER


def looks_like_json(content: str) -> bool:
    text = content.strip()
    if len(text) < 2 or text[0] not in "[{" or text[-1] not in "]}":
        return False
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def resource_key(
    family: ToolFamily,
    tool_name: str,
    args: Mapping[str, Any],
    *,
    root: str | None,
    cwd: str | None,
) -> str:
    """Readable canonical key (hashed into ``resource_id``)."""
    lname = (tool_name or "").lower()
    rel_cwd = workspace_relative(cwd, root=root) if cwd else "."
    command = command_of(lname, args)
    if command is not None:
        head = primary_command(command)
        match = _READ_CMD.match(head) if family is ToolFamily.READ else None
        if match:
            rel = workspace_relative(match.group("path"), root=root, cwd=cwd)
            return f"read:{rel}"
        return f"cmd:{normalize_command(command)}|cwd={rel_cwd}"
    if family is ToolFamily.READ:
        rel = workspace_relative(path_of(args) or "", root=root, cwd=cwd)
        rng = _range_of(args)
        return f"read:{rel}" + (f"#{_canonical_json(rng)}" if rng else "")
    if family in (ToolFamily.SEARCH, ToolFamily.LISTING):
        scope = {k: v for k, v in args.items() if k not in PATH_KEYS}
        target = path_of(args)
        if target:
            scope["path"] = workspace_relative(target, root=root, cwd=cwd)
        kind = "search" if family is ToolFamily.SEARCH else "list"
        return f"{kind}:{lname}:{_canonical_json(scope)}|cwd={rel_cwd}"
    return f"api:{lname}:{_canonical_json(dict(args))}"


def resource_id_for(
    family: ToolFamily,
    tool_name: str,
    args: Mapping[str, Any],
    *,
    root: str | None,
    cwd: str | None,
) -> tuple[str, str]:
    key = resource_key(family, tool_name, args, root=root, cwd=cwd)
    prefix = key.split(":", 1)[0]
    return f"{prefix}:{digest(key)}", key


def explicit_partial_reason(
    family: ToolFamily, tool_name: str, args: Mapping[str, Any]
) -> str | None:
    """Partiality visible from the call itself (never inferred as complete)."""
    if family is ToolFamily.READ and _range_of(args):
        return "range_read"
    command = command_of((tool_name or "").lower(), args)
    if command and _PIPE_CUT.search(command):
        return "pipe_truncated"
    if family is ToolFamily.SEARCH:
        for key in ("head_limit", "max_results", "max_count", "limit"):
            if args.get(key) not in (None, "", 0):
                return f"result_limit:{key}"
    return None


def build_envelope(
    *,
    tool_name: str,
    tool_input: Any,
    content: str,
    stderr: str = "",
    exit_code: int | None = None,
    call_id: str = "",
    session: str = "",
    workspace: str = "",
    root: str | None = None,
    cwd: str | None = None,
    is_error: bool = False,
    partial_reason: str | None = None,
    source_byte_length: int | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> ToolResultEnvelope:
    """Build the admission envelope (content kept) for one observed result."""
    args = parse_args(tool_input)
    family = classify(tool_name, args, content)
    resource_id, key = resource_id_for(family, tool_name, args, root=root, cwd=cwd)
    reason = partial_reason or explicit_partial_reason(family, tool_name, args)
    meta = dict(source_metadata or {})
    meta["resource_key"] = key[:200]
    return ToolResultEnvelope.from_text(
        tool_call_id=call_id,
        tool_name=tool_name or "",
        text=content,
        is_error=is_error,
        source_byte_length=source_byte_length,
        stderr=stderr,
        exit_code=exit_code,
        partial_reason=reason,
        tool_family=family.value,
        session=session,
        workspace=workspace,
        resource_id=resource_id,
        source_metadata=meta,
        keep_content=True,
    )
