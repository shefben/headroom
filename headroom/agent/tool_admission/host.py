"""Host pre-/post-tool adapter: Claude Code hook protocol on the proxy side (unified-21 #4).

``headroom wrap claude`` installs a ``PreToolUse`` and a ``PostToolUse``
command hook (:mod:`headroom.host_hook`). The hook process forwards Claude
Code's hook JSON to the local proxy (``POST /v1/agent/host-hook``), which runs
:func:`handle_hook` here and returns the JSON the hook prints on stdout.

* ``PreToolUse`` is observation only: it records edit targets (the active
  working set, kept exact on later reads). It prints nothing, so it can never
  change a permission decision, cancel a tool or alter its input.
* ``PostToolUse`` admits the result through the one
  :class:`~headroom.agent.tool_admission.dispatcher.ToolAdmissionDispatcher`
  and, only when a candidate is admitted, answers with
  ``hookSpecificOutput.updatedToolOutput`` in exactly the tool's own output
  shape, so the bytes never enter the host's context. Tools whose rendered
  form would be misrepresented by a rewrite (Read: Claude Code re-numbers
  the content lines) are left to the proxy live-zone fallback.

Clients without such a hook surface report ``observational_or_proxy_only``:
the proxy compresses their tool results only when they appear in the live
zone, and nothing claims the bytes stayed out of client history.
"""

from __future__ import annotations

import copy
import logging
import threading
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from typing import Any

from headroom.agent.tool_admission.dispatcher import AdmissionRequest, get_dispatcher
from headroom.agent.tool_admission.families import (
    EDIT_TOOLS,
    build_envelope,
    path_of,
    workspace_relative,
)
from headroom.agent.tool_admission.views import ViewContext

log = logging.getLogger(__name__)

HOST_PRE_CONTEXT = "host_pre_context"
OBSERVATIONAL = "observational_or_proxy_only"

#: What each wrapped client can do with tool results. Only a host that lets a
#: hook replace a tool result before it enters history may claim pre-context.
HOST_CAPABILITIES: dict[str, str] = {
    "claude": HOST_PRE_CONTEXT,
    "codex": OBSERVATIONAL,
    "opencode": OBSERVATIONAL,
    "copilot": OBSERVATIONAL,
    "cursor": OBSERVATIONAL,
    "aider": OBSERVATIONAL,
    "gemini": OBSERVATIONAL,
}

#: Claude Code tools whose output shape the adapter rewrites.
_SHAPED_TOOLS = frozenset({"Bash", "PowerShell", "Grep", "Glob"})
_WORKING_SET_LIMIT = 256
_SCOPES_LIMIT = 512

_lock = threading.Lock()
_working_sets: OrderedDict[str, OrderedDict[str, None]] = OrderedDict()
_counters: dict[str, int] = defaultdict(int)


def host_capability(client: str, *, hook_installed: bool = True) -> str:
    cap = HOST_CAPABILITIES.get(client.lower(), OBSERVATIONAL)
    if cap == HOST_PRE_CONTEXT and not hook_installed:
        return OBSERVATIONAL
    return cap


def status() -> dict[str, Any]:
    with _lock:
        return {
            "capabilities": dict(HOST_CAPABILITIES),
            "counters": dict(_counters),
            "scopes": len(_working_sets),
        }


def reset() -> None:
    with _lock:
        _working_sets.clear()
        _counters.clear()


def _count(name: str) -> None:
    with _lock:
        _counters[name] += 1


def _scope(identity: Any, payload: Mapping[str, Any]) -> str:
    conversation = str(payload.get("session_id") or "")[:128]
    return f"{identity.workspace_id}|cc:{conversation or identity.session_id}"


def _remember_edit(scope: str, entries: list[str]) -> None:
    with _lock:
        ws = _working_sets.setdefault(scope, OrderedDict())
        _working_sets.move_to_end(scope)
        for entry in entries:
            ws[entry] = None
            ws.move_to_end(entry)
        while len(ws) > _WORKING_SET_LIMIT:
            ws.popitem(last=False)
        while len(_working_sets) > _SCOPES_LIMIT:
            _working_sets.popitem(last=False)


def _working_set(scope: str) -> frozenset[str]:
    with _lock:
        return frozenset(_working_sets.get(scope, ()))


def _text_of_blocks(value: Any) -> tuple[str, str] | None:
    """``(text, shape)`` for an MCP-style result (string or text blocks)."""
    if isinstance(value, str):
        return value, "str"
    if isinstance(value, list) and value and all(
        isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
        for b in value
    ):
        return "\n".join(b["text"] for b in value), "blocks"
    if isinstance(value, dict) and isinstance(value.get("content"), list):
        inner = _text_of_blocks(value["content"])
        if inner is not None:
            return inner[0], "content_blocks"
    return None


def extract_result(
    tool_name: str, response: Any
) -> tuple[str, str, str | None, str | None] | None:
    """``(text, slot, stderr, partial_reason)`` for a rewritable result, else None.

    ``slot`` names where the replacement goes in the tool's output shape.
    """
    if tool_name in ("Bash", "PowerShell"):
        if not isinstance(response, dict) or not isinstance(response.get("stdout"), str):
            return None
        if response.get("isImage") or response.get("backgroundTaskId"):
            return None
        if response.get("persistedOutputPath") or response.get("rawOutputPath"):
            return None  # the host already externalized it; its preview is partial
        reason = "interrupted" if response.get("interrupted") else None
        if response.get("timedOutAfterMs"):
            reason = "timed_out"
        stderr = response.get("stderr") if isinstance(response.get("stderr"), str) else ""
        return response["stdout"], "stdout", stderr, reason
    if tool_name == "Grep":
        if not isinstance(response, dict):
            return None
        reason = "result_limit" if response.get("appliedLimit") else None
        mode = response.get("mode") or "files_with_matches"
        if mode == "content" and isinstance(response.get("content"), str):
            return response["content"], "content", None, reason
        if mode == "files_with_matches" and isinstance(response.get("filenames"), list):
            return "\n".join(str(f) for f in response["filenames"]), "filenames", None, reason
        return None
    if tool_name == "Glob":
        if not isinstance(response, dict) or not isinstance(response.get("filenames"), list):
            return None
        reason = "client_truncated" if response.get("truncated") else None
        return "\n".join(str(f) for f in response["filenames"]), "filenames", None, reason
    if tool_name.startswith("mcp__") and not tool_name.startswith("mcp__headroom"):
        found = _text_of_blocks(response)
        if found is None:
            return None
        return found[0], f"mcp:{found[1]}", None, None
    return None


def shaped_output(response: Any, slot: str, text: str) -> Any:
    """The tool's own output object with ``slot`` replaced by ``text``."""
    if slot.startswith("mcp:"):
        shape = slot.split(":", 1)[1]
        if shape == "str":
            return text
        if shape == "blocks":
            return [{"type": "text", "text": text}]
        clone = copy.deepcopy(response)
        clone["content"] = [{"type": "text", "text": text}]
        return clone
    clone = copy.deepcopy(response)
    if slot == "filenames":
        clone["filenames"] = text.split("\n")
    else:
        clone[slot] = text
    return clone


def handle_hook(
    payload: Mapping[str, Any],
    *,
    identity: Any,
    task: Any = None,
    query: str = "",
    policy: Any = None,
    store: Any = None,
    retrieval_available: bool = True,
    arbiter: Any = None,
    limits: Any = None,
    feedback: Any = None,
) -> dict[str, Any]:
    """Process one Claude Code hook payload; returns ``{"output": ..., ...}``.

    ``output`` is what the hook prints on stdout (``None`` = print nothing,
    i.e. the host proceeds exactly as without the hook).
    """
    event = str(payload.get("hook_event_name") or "")
    tool_name = str(payload.get("tool_name") or "")
    tool_input = payload.get("tool_input") if isinstance(payload.get("tool_input"), dict) else {}
    scope = _scope(identity, payload)
    root = identity.root or (str(payload.get("cwd")) if payload.get("cwd") else None)
    if event == "PreToolUse":
        _count("pre_tool")
        if tool_name.lower() in EDIT_TOOLS:
            raw = path_of(tool_input)
            if raw:
                _remember_edit(scope, [raw, workspace_relative(raw, root=root, cwd=root)])
                _count("edit_targets")
        return {"output": None, "stage": "observed", "capability": HOST_PRE_CONTEXT}
    if event != "PostToolUse":
        _count("ignored_event")
        return {"output": None, "stage": "ignored_event", "capability": HOST_PRE_CONTEXT}
    _count("post_tool")
    if tool_name not in _SHAPED_TOOLS and not tool_name.startswith("mcp__"):
        _count("left_to_proxy")
        return {"output": None, "stage": "left_to_proxy", "capability": HOST_PRE_CONTEXT}
    response = payload.get("tool_response")
    extracted = extract_result(tool_name, response)
    if extracted is None:
        _count("unshaped")
        return {"output": None, "stage": "unshaped", "capability": HOST_PRE_CONTEXT}
    text, slot, stderr, partial_reason = extracted
    from headroom.transforms.safety_gate import task_payload

    gate_task = task_payload(task, fallback_text=query)
    entities = tuple(gate_task.get("exact_entities") or ())
    envelope = build_envelope(
        tool_name=tool_name,
        tool_input=tool_input,
        content=text,
        stderr=stderr or "",
        call_id=str(payload.get("tool_use_id") or ""),
        session=scope,
        workspace=str(getattr(identity, "workspace_id", "")),
        root=root,
        cwd=str(payload.get("cwd") or root or "") or None,
        partial_reason=partial_reason,
        source_metadata={"host": "claude", "slot": slot},
    )
    if policy is None:
        from headroom.transforms.content_router import default_gate_policy

        policy = default_gate_policy()
    if store is None:
        from headroom.cache.compression_store import get_compression_store

        store = get_compression_store()
    ctx = ViewContext(
        query=query,
        entities=entities,
        working_set=_working_set(scope),
        root=root,
        cwd=str(payload.get("cwd") or root or "") or None,
        tool_input=dict(tool_input),
    )
    outcome = get_dispatcher().admit(
        envelope,
        AdmissionRequest(
            scope=scope,
            context=ctx,
            task=gate_task,
            policy=policy,
            store=store,
            retrieval_available=retrieval_available,
            source="host",
            arbiter=arbiter,
            limits=limits,
            feedback=feedback,
        ),
    )
    _count(f"stage:{outcome.stage}")
    result: dict[str, Any] = {
        "output": None,
        "stage": outcome.stage,
        "capability": HOST_PRE_CONTEXT,
        "outcome": outcome.to_dict(),
    }
    if outcome.text is not None:
        result["output"] = {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "updatedToolOutput": shaped_output(response, slot, outcome.text),
            }
        }
        _count("rewritten")
    return result
