"""Normalize provider-native request messages into agent events, read-only.

Supports Anthropic Messages, OpenAI Chat Completions, OpenAI Responses
``input`` items and Google ``contents``. The input list and every nested
object are only read — never mutated or reserialized — so cache-hot prefix
bytes are untouched.

Origin rules: a user-role message counts as an authentic ``USER`` turn only
for the text left after removing harness-injected wrappers (Claude Code
``<system-reminder>``/command blocks, Codex ``<environment_context>`` /
``<user_instructions>``, interruption markers). A user-role message that is
entirely harness content is ``HARNESS``. Tool results are ``TOOL`` evidence.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from headroom.agent.events import (
    AgentEvent,
    EventEnvelope,
    EventKind,
    Origin,
    content_digest,
    make_event,
)
from headroom.agent.identity import WorkspaceIdentity
from headroom.agent.interfaces import ToolResultEnvelope

_HARNESS_BLOCKS = re.compile(
    r"<(system-reminder|command-name|command-message|command-args|local-command-stdout|"
    r"local-command-stderr|local-command-caveat|user-prompt-submit-hook|environment_context|"
    r"user_instructions|turn_aborted|bash-input|bash-stdout|bash-stderr)\b[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_HARNESS_LINES = re.compile(
    r"^\s*(?:\[Request interrupted by user[^\]]*\]|Caveat: The messages below were generated "
    r"by the user while running local commands.*|# AGENTS\.md instructions for .*)$",
    re.MULTILINE,
)

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
_PATH_KEYS = ("file_path", "path", "notebook_path", "filename", "target_file")
_PATCH_FILE = re.compile(r"^\*\*\* (?:Update|Add|Delete) File: (.+)$", re.MULTILINE)
_GIT_MUTATION = re.compile(
    r"\bgit\s+(?:-C\s+\S+\s+)?(commit|checkout|switch|merge|rebase|reset|cherry-pick|revert|"
    r"stash|pull|am|apply|restore|tag|branch\s+-[dDmM])\b"
)
_TEST_COMMAND = re.compile(
    r"(?:^|[\s;&|(])(?:python\d?(?:\.\d+)?\s+-m\s+)?(pytest|cargo\s+(?:test|nextest)|"
    r"npm\s+(?:run\s+)?test|yarn\s+test|pnpm\s+(?:run\s+)?test|go\s+test|jest|vitest|"
    r"mvn\s+test|gradle\s+test|ctest|tox|nox|unittest|rspec|phpunit|dotnet\s+test)\b"
)
_PYTEST_SUMMARY = re.compile(r"(\d+) (passed|failed|error|errors|skipped)")
_CARGO_SUMMARY = re.compile(r"test result: (ok|FAILED)\. (\d+) passed; (\d+) failed")


def authored_user_text(text: str) -> str:
    """Text a human actually authored, with harness wrappers removed."""
    stripped = _HARNESS_BLOCKS.sub("", text)
    stripped = _HARNESS_LINES.sub("", stripped)
    return stripped.strip()


def _block_text(content: Any) -> str:
    """Concatenate text from a provider content value (string or block list)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        for key in ("text", "output", "content"):
            value = content.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                return _block_text(value)
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping):
                btype = block.get("type")
                if btype in (None, "text", "input_text", "output_text"):
                    value = block.get("text")
                    if isinstance(value, str):
                        parts.append(value)
                elif btype == "tool_result":
                    parts.append(_block_text(block.get("content")))
        return "\n".join(p for p in parts if p)
    return ""


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _parse_args(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, Mapping) else {}
    return {}


@dataclass
class _Call:
    call_id: str
    name: str
    args: Mapping[str, Any]


@dataclass
class NormalizedTurn:
    """Events derived from one request plus request-level facts."""

    envelopes: list[EventEnvelope] = field(default_factory=list)
    latest_user_digest: str | None = None
    tool_names: tuple[str, ...] = ()

    @property
    def events(self) -> list[AgentEvent]:
        return [env.event for env in self.envelopes]


class _Builder:
    def __init__(self, identity: WorkspaceIdentity, task_id: str | None) -> None:
        self.identity = identity
        self.task_id = task_id
        self.turn = NormalizedTurn()
        self.calls: dict[str, _Call] = {}
        self._occurrences: Counter[tuple[str, str]] = Counter()
        self._anon = 0

    def _add(
        self,
        kind: EventKind,
        origin: Origin,
        source_id: str,
        payload: Any,
        *,
        index: int = 0,
        content: str | None = None,
        resource_id: str | None = None,
        resource_hash: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        event = make_event(
            workspace_id=self.identity.workspace_id,
            session_id=self.identity.session_id,
            task_id=self.task_id,
            kind=kind,
            origin=origin,
            source_id=source_id,
            index=index,
            payload=payload,
            resource_id=resource_id,
            resource_hash=resource_hash,
            metadata=metadata,
        )
        self.turn.envelopes.append(EventEnvelope(event=event, content=content))

    def _occurrence(self, scope: str, digest: str) -> int:
        key = (scope, digest)
        index = self._occurrences[key]
        self._occurrences[key] += 1
        return index

    def anon_id(self, prefix: str) -> str:
        self._anon += 1
        return f"{prefix}:{self._anon}"

    # -- producers -------------------------------------------------------

    def user_text(self, text: str) -> None:
        if not text.strip():
            return
        authored = authored_user_text(text)
        if authored:
            digest = content_digest(authored)
            self._add(
                EventKind.USER_INTENT,
                Origin.USER,
                "user",
                authored,
                index=self._occurrence("user", digest),
                content=authored,
                metadata={"chars": len(authored)},
            )
            self.turn.latest_user_digest = digest
        else:
            digest = content_digest(text)
            self._add(
                EventKind.USER_INTENT,
                Origin.HARNESS,
                "harness",
                text,
                index=self._occurrence("harness", digest),
                metadata={"chars": len(text), "harness": True},
            )

    def model_text(self, text: str) -> None:
        if not text.strip():
            return
        digest = content_digest(text)
        self._add(
            EventKind.MODEL_OUTPUT,
            Origin.MODEL,
            "assistant",
            text,
            index=self._occurrence("assistant", digest),
            metadata={"chars": len(text)},
        )

    def tool_call(self, call_id: str, name: str, args: Any) -> None:
        parsed = _parse_args(args)
        self.calls[call_id] = _Call(call_id, name, parsed)
        meta: dict[str, Any] = {"tool_name": name}
        command = _command_of(name, parsed)
        if command:
            meta["command_head"] = command.strip().split("\n", 1)[0][:120]
        self._add(
            EventKind.TOOL_CALL,
            Origin.MODEL,
            call_id,
            {"name": name, "args": _json_text(args)},
            metadata=meta,
        )

    def tool_result(self, call_id: str, content: Any, *, is_error: bool = False) -> None:
        call = self.calls.get(call_id)
        name = call.name if call else ""
        text = _block_text(content) if not isinstance(content, str) else content
        envelope = ToolResultEnvelope.from_text(
            tool_call_id=call_id, tool_name=name, text=text, is_error=is_error
        )
        self._add(
            EventKind.TOOL_RESULT,
            Origin.TOOL,
            call_id,
            text,
            content=text,
            metadata=envelope.to_metadata(),
        )
        if call is not None:
            self._derived(call, envelope, text)

    def _derived(self, call: _Call, envelope: ToolResultEnvelope, text: str) -> None:
        lname = call.name.lower()
        if lname in _EDIT_TOOLS and not envelope.is_error:
            for ordinal, path in enumerate(_edited_paths(lname, call.args)):
                resource = self._resource(path)
                self._add(
                    EventKind.FILE_CHANGE,
                    Origin.TOOL,
                    call.call_id,
                    {"path": path, "args": _json_text(call.args)},
                    index=ordinal,
                    resource_id=resource,
                    resource_hash=content_digest(_json_text(call.args)),
                    metadata={"tool_name": call.name, "path": path[-160:]},
                )
        command = _command_of(call.name, call.args)
        if not command:
            return
        git = _GIT_MUTATION.search(command)
        if git and not envelope.is_error:
            self._add(
                EventKind.GIT_CHANGE,
                Origin.TOOL,
                call.call_id,
                {"command": command, "result": text},
                metadata={"operation": git.group(1).split()[0], "tool_name": call.name},
            )
        test = _TEST_COMMAND.search(command)
        if test:
            meta: dict[str, Any] = {
                "runner": test.group(1).split()[0],
                "exit_code": envelope.exit_code,
                "is_error": envelope.is_error,
                "is_partial": envelope.is_partial,
            }
            meta.update(_test_counts(text))
            self._add(
                EventKind.TEST_RESULT,
                Origin.TOOL,
                call.call_id,
                {"command": command, "result": text},
                content=text,
                metadata=meta,
            )

    def _resource(self, path: str) -> str:
        root = self.identity.root
        normalized = path.replace("\\", "/")
        if not re.match(r"^(?:/|[A-Za-z]:/)", normalized):
            return "path:" + re.sub(r"^(?:\./)+", "", normalized)
        if root:
            root_norm = root.replace("\\", "/").rstrip("/").lower()
            if normalized.lower().startswith(root_norm + "/"):
                return "path:" + normalized[len(root_norm) + 1 :]
        return "ext:" + content_digest(normalized)[:24]


def _command_of(name: str, args: Mapping[str, Any]) -> str | None:
    if name.lower() not in _SHELL_TOOLS:
        return None
    command = args.get("command") or args.get("cmd")
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    if isinstance(command, str):
        return command
    action = args.get("action")
    if isinstance(action, Mapping):
        return _command_of(name, action)
    return None


def _edited_paths(lname: str, args: Mapping[str, Any]) -> list[str]:
    if lname == "apply_patch":
        patch = args.get("input") or args.get("patch") or ""
        return _PATCH_FILE.findall(patch) if isinstance(patch, str) else []
    for key in _PATH_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value:
            return [value]
    return []


def _test_counts(text: str) -> dict[str, Any]:
    tail = text[-6000:]
    counts: dict[str, Any] = {}
    cargo = _CARGO_SUMMARY.findall(tail)
    if cargo:
        counts["passed"] = sum(int(p) for _, p, _ in cargo)
        counts["failed"] = sum(int(f) for _, _, f in cargo)
        return counts
    for number, label in _PYTEST_SUMMARY.findall(tail):
        key = "errors" if label.startswith("error") else label
        counts[key] = counts.get(key, 0) + int(number)
    return counts


# -- provider walkers ------------------------------------------------------


def _walk_anthropic(builder: _Builder, messages: Iterable[Any]) -> None:
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, str):
            (builder.user_text if role == "user" else builder.model_text)(content)
            continue
        if not isinstance(content, list):
            continue
        texts: list[str] = []
        for block in content:
            if not isinstance(block, Mapping):
                continue
            btype = block.get("type")
            if btype == "text" and isinstance(block.get("text"), str):
                texts.append(block["text"])
            elif btype in ("tool_use", "server_tool_use") and role == "assistant":
                call_id = str(block.get("id") or builder.anon_id("tu"))
                builder.tool_call(call_id, str(block.get("name") or ""), block.get("input"))
            elif btype == "tool_result":
                call_id = str(block.get("tool_use_id") or builder.anon_id("tr"))
                builder.tool_result(
                    call_id, block.get("content"), is_error=bool(block.get("is_error"))
                )
        if texts:
            joined = "\n".join(texts)
            (builder.user_text if role == "user" else builder.model_text)(joined)


def _walk_openai_chat(builder: _Builder, messages: Iterable[Any]) -> None:
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = message.get("role")
        if role == "user":
            builder.user_text(_block_text(message.get("content")))
        elif role == "assistant":
            builder.model_text(_block_text(message.get("content")))
            for call in message.get("tool_calls") or ():
                if not isinstance(call, Mapping):
                    continue
                function = call.get("function") or {}
                builder.tool_call(
                    str(call.get("id") or builder.anon_id("tc")),
                    str(function.get("name") or ""),
                    function.get("arguments"),
                )
        elif role == "tool":
            builder.tool_result(
                str(message.get("tool_call_id") or builder.anon_id("tr")),
                message.get("content"),
            )


def _walk_openai_responses(builder: _Builder, items: Iterable[Any]) -> None:
    for item in items:
        if not isinstance(item, Mapping):
            continue
        itype = item.get("type")
        if itype in (None, "message"):
            role = item.get("role")
            text = _block_text(item.get("content"))
            if role == "user":
                builder.user_text(text)
            elif role == "assistant":
                builder.model_text(text)
        elif itype in ("function_call", "custom_tool_call"):
            builder.tool_call(
                str(item.get("call_id") or item.get("id") or builder.anon_id("fc")),
                str(item.get("name") or ""),
                item.get("arguments") if itype == "function_call" else {"input": item.get("input")},
            )
        elif itype == "local_shell_call":
            builder.tool_call(
                str(item.get("call_id") or item.get("id") or builder.anon_id("ls")),
                "local_shell",
                {"action": item.get("action") or {}},
            )
        elif itype in (
            "function_call_output",
            "custom_tool_call_output",
            "local_shell_call_output",
        ):
            builder.tool_result(
                str(item.get("call_id") or builder.anon_id("fo")), item.get("output")
            )


def _walk_google(builder: _Builder, contents: Iterable[Any]) -> None:
    for message in contents:
        if not isinstance(message, Mapping):
            continue
        role = message.get("role")
        texts: list[str] = []
        for part in message.get("parts") or ():
            if not isinstance(part, Mapping):
                continue
            if isinstance(part.get("text"), str):
                texts.append(part["text"])
            elif isinstance(part.get("functionCall"), Mapping):
                call = part["functionCall"]
                name = str(call.get("name") or "")
                call_id = str(call.get("id") or f"g:{name}:{len(builder.calls)}")
                builder.tool_call(call_id, name, call.get("args"))
            elif isinstance(part.get("functionResponse"), Mapping):
                resp = part["functionResponse"]
                name = str(resp.get("name") or "")
                call_id = str(resp.get("id") or "")
                if not call_id:
                    match = [c for c in builder.calls.values() if c.name == name]
                    call_id = match[-1].call_id if match else builder.anon_id("gr")
                builder.tool_result(call_id, _json_text(resp.get("response")))
        if texts:
            (builder.user_text if role == "user" else builder.model_text)("\n".join(texts))


def _is_responses_input(messages: list[Any]) -> bool:
    return any(
        isinstance(item, Mapping)
        and (
            item.get("type")
            in (
                "message",
                "function_call",
                "function_call_output",
                "custom_tool_call",
                "custom_tool_call_output",
                "local_shell_call",
                "local_shell_call_output",
                "reasoning",
            )
        )
        for item in messages
    )


def normalize_messages(
    provider: str,
    messages: Any,
    identity: WorkspaceIdentity,
    *,
    task_id: str | None = None,
) -> NormalizedTurn:
    """Derive events from provider-native messages without mutating them."""
    builder = _Builder(identity, task_id)
    if isinstance(messages, str):
        builder.user_text(messages)
    elif isinstance(messages, list):
        if provider == "anthropic":
            _walk_anthropic(builder, messages)
        elif provider in ("google", "gemini", "vertex"):
            _walk_google(builder, messages)
        elif _is_responses_input(messages):
            _walk_openai_responses(builder, messages)
        else:
            _walk_openai_chat(builder, messages)
    builder.turn.tool_names = tuple(sorted({call.name for call in builder.calls.values()}))
    return builder.turn
