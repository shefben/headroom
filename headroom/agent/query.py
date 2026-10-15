"""Per-request TaskContext and relevance query (2D) — shared by every consumer.

:func:`build_task_context` runs inside the agent TurnHook on the event loop,
so it is pure CPU over the already-normalized turn plus in-memory snapshots
(TaskState, code graph). It extracts:

* the explicit latest user goal — the authentic user text left after harness
  wrappers are stripped by the provider walkers (never harness boilerplate);
* tool intent / name / call id of the latest tool exchange;
* exact paths, symbols, error ids and identifiers the user named (these are
  prioritized before prose in the relevance query);
* workspace, content type, completeness/truncation, resource id + version
  hash of the latest tool result, cache policy;
* graph neighbors (code tasks only) and the current TaskState revision.

The resulting :class:`~headroom.agent.interfaces.RelevanceQuery` is capped at
512 tokens and is the typed object later phases pass to compressors and
across the Python/Rust boundary (``RelevanceQuery.to_dict()``).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from headroom.agent.events import EventKind, Origin
from headroom.agent.interfaces import (
    TASK_CONTEXT_VERSION,
    DecisionHints,
    RelevanceQuery,
    TaskContext,
)
from headroom.agent.text_safety import clip

if TYPE_CHECKING:
    from headroom.agent.identity import WorkspaceIdentity
    from headroom.agent.normalize import NormalizedTurn
    from headroom.agent.runtime import AgentRuntime

LATEST_GOAL_LIMIT = 4000
MAX_ENTITIES = 24
GRAPH_NEIGHBOR_LIMIT = 8

_PATH = re.compile(
    r"(?<![\w@])((?:[A-Za-z]:[\\/])?(?:\.{1,2}[\\/])?(?:[\w.\-]+[\\/])*[\w\-]+\."
    r"(?:py|pyi|js|jsx|ts|tsx|go|rs|java|kt|rb|c|h|cc|cpp|hpp|cs|swift|php|lua|zig|sql|md|rst|"
    r"toml|ya?ml|json|ini|cfg|sh|ps1|txt|html|css|scss|vue|svelte|ex|exs|scala|m|mm|r|jl))\b"
)
_DIR_PATH = re.compile(r"(?<![\w@])((?:[\w.\-]+/){1,}[\w.\-]+)")
_BACKTICK = re.compile(r"`([^`\n]{2,120})`")
_CAMEL = re.compile(r"\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+|[a-z]+(?:[A-Z][a-z0-9]+)+)\b")
_SNAKE = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+|[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\b")
_CALL = re.compile(r"\b([A-Za-z_][\w.]*)\(\)")
_DOTTED = re.compile(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*){1,4})\b")
_ERROR_ID = re.compile(
    r"\b(\w*(?:Error|Exception|Warning)\b|E\d{3,5}\b|TS\d{4}\b|E[A-Z]{3,12}\b|"
    r"HTTP\s?[1-5]\d\d\b|exit (?:code|status) -?\d+)"
)
_IDENTIFIER = re.compile(
    r"\b([0-9a-f]{7,40}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}|[A-Z][A-Z0-9]{1,9}-\d{1,6})\b|(#\d{1,7})\b"
)
_CODE_EXT = re.compile(
    r"\.(?:py|pyi|js|jsx|ts|tsx|go|rs|java|kt|rb|c|h|cc|cpp|hpp|cs|swift|php|scala)$", re.I
)
_FILE_EXT_ONLY = re.compile(r"\.[A-Za-z0-9]{1,8}$")

_READ_TOOLS = {"read", "read_file", "view", "cat", "open_file", "readfile", "notebookread"}
_SEARCH_TOOLS = {
    "grep",
    "glob",
    "search",
    "find",
    "ripgrep",
    "codebase_search",
    "file_search",
    "list_dir",
    "ls",
    "websearch",
    "web_search",
}
_EDIT_TOOLS = {
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
_SHELL_TOOLS = {"bash", "shell", "local_shell", "exec_command", "run_terminal_cmd", "terminal"}


def _unique(values: Sequence[str], limit: int) -> tuple[str, ...]:
    out: dict[str, None] = {}
    for value in values:
        cleaned = value.strip().strip(".,;:'\"()[]{}")
        if len(cleaned) >= 2 and cleaned not in out:
            out[cleaned] = None
        if len(out) >= limit:
            break
    return tuple(out)


def extract_entities(text: str) -> dict[str, tuple[str, ...]]:
    """Exact paths, symbols, error ids and identifiers a user named."""
    paths: list[str] = []
    symbols: list[str] = []
    for quoted in _BACKTICK.findall(text):
        q = quoted.strip()
        if _PATH.fullmatch(q) or ("/" in q and " " not in q):
            paths.append(q)
        elif " " not in q:
            symbols.append(q.rstrip("()"))
    paths.extend(m.group(1) for m in _PATH.finditer(text))
    paths.extend(
        m.group(1) for m in _DIR_PATH.finditer(text) if not m.group(1).startswith(("http", "www."))
    )
    path_set = set(paths)
    symbols.extend(m.group(1) for m in _CALL.finditer(text))
    symbols.extend(m.group(1) for m in _CAMEL.finditer(text))
    symbols.extend(m.group(1) for m in _SNAKE.finditer(text))
    symbols.extend(
        m.group(1)
        for m in _DOTTED.finditer(text)
        if m.group(1) not in path_set
        and not _FILE_EXT_ONLY.search(m.group(1))
        and not any(m.group(1) in p for p in path_set)
    )
    errors = [m.group(1) for m in _ERROR_ID.finditer(text)]
    error_set = set(errors)
    identifiers = [m.group(1) or m.group(2) for m in _IDENTIFIER.finditer(text)]
    return {
        "paths": _unique(paths, MAX_ENTITIES),
        "symbols": _unique([s for s in symbols if s not in error_set], MAX_ENTITIES),
        "errors": _unique(errors, MAX_ENTITIES),
        "identifiers": _unique(identifiers, MAX_ENTITIES),
    }


def tool_intent_of(tool_name: str, command: str | None, path: str | None) -> str:
    lname = tool_name.lower()
    if lname in _SHELL_TOOLS and command:
        from headroom.agent.normalize import _GIT_MUTATION, _TEST_COMMAND

        if _TEST_COMMAND.search(command):
            kind = "test"
        elif command.strip().startswith("git") or _GIT_MUTATION.search(command):
            kind = "vcs"
        elif re.match(r"\s*(?:rg|grep|find|fd|ag|ls|tree)\b", command):
            kind = "search"
        elif re.match(r"\s*(?:cat|head|tail|less|more|sed -n|type|bat)\b", command):
            kind = "read"
        else:
            kind = "shell"
        return clip(f"{kind}: {command}", 200)
    if lname in _READ_TOOLS:
        return clip(f"read: {path or ''}", 200)
    if lname in _SEARCH_TOOLS:
        return clip(f"search: {path or ''}", 200)
    if lname in _EDIT_TOOLS:
        return clip(f"edit: {path or ''}", 200)
    return clip(f"tool: {tool_name} {path or command or ''}", 200)


def _content_type(text: str) -> str | None:
    if not text:
        return None
    from headroom.transforms.content_detector import detect_content_type

    try:
        return detect_content_type(text[:4000]).content_type.value
    except Exception:  # noqa: BLE001 - detection is advisory
        return None


def build_task_context(
    *,
    ctx: Any,
    identity: WorkspaceIdentity,
    turn: NormalizedTurn,
    runtime: AgentRuntime | None,
    request_id: str,
) -> TaskContext:
    """Assemble the request's TaskContext (event loop: in-memory reads only)."""
    envelopes = turn.envelopes
    latest_user = ""
    latest_user_index = -1
    for index, env in enumerate(envelopes):
        if env.event.kind == EventKind.USER_INTENT.value and env.event.origin is Origin.USER:
            latest_user = env.content or ""
            latest_user_index = index
    tail = envelopes[latest_user_index + 1 :] if latest_user_index >= 0 else envelopes

    last_call = next(
        (e.event for e in reversed(envelopes) if e.event.kind == EventKind.TOOL_CALL.value), None
    )
    last_result = next(
        (e for e in reversed(envelopes) if e.event.kind == EventKind.TOOL_RESULT.value), None
    )
    tool_name = str(last_call.redacted_metadata.get("tool_name") or "") if last_call else None
    command = last_call.redacted_metadata.get("command_head") if last_call else None
    call_path = last_call.redacted_metadata.get("path") if last_call else None
    tool_intent = tool_intent_of(tool_name or "", command, call_path) if last_call else None

    entities = (
        extract_entities(latest_user)
        if latest_user
        else {"paths": (), "symbols": (), "errors": (), "identifiers": ()}
    )

    active_error = ""
    from headroom.agent.evidence import _error_signature

    for env in reversed(tail):
        meta = env.event.redacted_metadata
        if env.event.kind == EventKind.TOOL_RESULT.value and meta.get("is_error") and env.content:
            active_error = _error_signature(env.content) or ""
            if active_error:
                break

    snapshot = None
    services = runtime.services if runtime is not None else {}
    compiler = services.get("task_state")
    if compiler is not None and identity.trusted:
        snapshot = compiler.snapshot(identity.workspace_id, identity.session_id)
    if not active_error and snapshot is not None:
        blockers = [a for a in snapshot.state.blockers if a.live and a.state == "ACTIVE"]
        if blockers:
            active_error = blockers[-1].text

    result_meta = last_result.event.redacted_metadata if last_result else {}
    result_text = last_result.content if last_result else None
    resource_id = last_result.event.resource_id if last_result else None
    content_type = _content_type(result_text or "")
    path_candidates = [*entities["paths"]]
    if resource_id:
        path_candidates.append(resource_id)
    elif call_path:
        path_candidates.append(str(call_path))
    is_code = bool(
        content_type == "source_code"
        or any(_CODE_EXT.search(p.removeprefix("path:")) for p in path_candidates)
        or entities["symbols"]
        and (tool_name or "").lower() in (_READ_TOOLS | _EDIT_TOOLS | _SEARCH_TOOLS)
    )

    neighbors: tuple[str, ...] = ()
    graph = services.get("code_graph")
    if graph is not None and identity.can_authorize_paths():
        graph.ensure_workspace(identity)
        if is_code:
            active_path = resource_id or (str(call_path) if call_path else None)
            labels: dict[str, None] = {}
            targets = [*entities["symbols"][:6], *entities["paths"][:6]]
            if active_path:
                targets.append(active_path)
            for target in targets:
                for neighbor in graph.neighbors(
                    identity.workspace_id, target, radius=2, limit=12, active_path=active_path
                ):
                    labels.setdefault(neighbor.label, None)
                if len(labels) >= GRAPH_NEIGHBOR_LIMIT * 2:
                    break
            neighbors = tuple(list(labels)[:GRAPH_NEIGHBOR_LIMIT])

    exact_entities = (
        *entities["paths"],
        *entities["symbols"],
        *entities["errors"],
        *entities["identifiers"],
    )
    query = RelevanceQuery(
        explicit_user_goal=clip(latest_user, 1500) if latest_user else "",
        tool_intent=tool_intent or "",
        exact_entities=tuple(dict.fromkeys(exact_entities)),
        active_error=active_error,
        graph_neighbors=neighbors if is_code else (),
        is_code=is_code,
    )
    hints = DecisionHints(family="relevance_query", query_terms=query.terms)
    config = getattr(ctx, "config", None)
    cache_policy = str(getattr(config, "mode", "") or "") or None
    return TaskContext(
        v=TASK_CONTEXT_VERSION,
        request_id=request_id,
        workspace_id=identity.workspace_id,
        session_id=identity.session_id,
        trusted=identity.trusted,
        provider=ctx.provider,
        model=ctx.model,
        tool_names=turn.tool_names,
        event_ids=tuple(ev.id for ev in turn.events),
        latest_user_digest=turn.latest_user_digest,
        hints=hints,
        latest_user_goal=latest_user[:LATEST_GOAL_LIMIT] if latest_user else None,
        tool_intent=tool_intent,
        tool_name=tool_name,
        tool_call_id=last_call.source_id if last_call else None,
        exact_paths=entities["paths"],
        exact_symbols=entities["symbols"],
        exact_errors=entities["errors"],
        exact_identifiers=entities["identifiers"],
        content_type=content_type,
        result_complete=(not result_meta.get("is_partial")) if last_result else None,
        result_truncated=bool(result_meta.get("is_partial")),
        resource_id=resource_id,
        resource_hash=last_result.event.resource_hash if last_result else None,
        cache_policy=cache_policy,
        graph_neighbors=neighbors,
        task_id=snapshot.state.id if snapshot else None,
        task_revision=snapshot.state.revision if snapshot else 0,
        relevance_query=query,
    )
