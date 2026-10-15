"""Agent events and provider normalization (read-only, origin rules, idempotency)."""

from __future__ import annotations

import copy
import json

import pytest

from headroom.agent.events import (
    EventKind,
    Origin,
    event_kind_version,
    idempotency_key,
    make_event,
    redact_metadata,
    register_event_kind,
)
from headroom.agent.identity import resolve_identity
from headroom.agent.interfaces import (
    ActionAuthorization,
    AuthorizationDecision,
    DecisionHints,
    InterfaceVersionError,
    TaskContext,
    ToolResultEnvelope,
)
from headroom.agent.normalize import authored_user_text, normalize_messages


@pytest.fixture
def identity():
    return resolve_identity(session_token=None, registered_cwd=None)


def test_idempotency_key_is_stable_and_sensitive():
    a = idempotency_key("s", "call_1", "TOOL_RESULT", 0, "d1")
    assert a == idempotency_key("s", "call_1", "TOOL_RESULT", 0, "d1")
    assert a != idempotency_key("s2", "call_1", "TOOL_RESULT", 0, "d1")
    assert a != idempotency_key("s", "call_1", "TOOL_RESULT", 1, "d1")
    assert a != idempotency_key("s", "call_1", "TOOL_RESULT", 0, "d2")


def test_events_carry_digest_not_content():
    event = make_event(
        workspace_id="w",
        session_id="s",
        kind=EventKind.USER_INTENT,
        origin=Origin.USER,
        source_id="user",
        index=0,
        payload="secret password hunter2",
    )
    row = json.dumps(event.to_row())
    assert "hunter2" not in row
    assert len(event.payload_digest) == 64


def test_redacted_metadata_drops_nested_and_truncates():
    meta = redact_metadata({"a": "x" * 500, "nested": {"k": "v"}, "n": 3, "flag": True})
    assert meta["a"] == "x" * 160
    assert "nested" not in meta
    assert meta["n"] == 3 and meta["flag"] is True


def test_register_versioned_event_kind():
    name = register_event_kind("phase_two_custom", 2)
    assert name == "PHASE_TWO_CUSTOM"
    assert event_kind_version(name) == 2
    register_event_kind("phase_two_custom", 1)  # lower version never downgrades
    assert event_kind_version(name) == 2
    event = make_event(
        workspace_id="w",
        session_id="s",
        kind=name,
        origin=Origin.SYSTEM,
        source_id="x",
        index=0,
        payload={},
    )
    assert event.v == 2 and event.kind == name
    with pytest.raises(ValueError):
        register_event_kind("bad kind!")


def test_harness_wrappers_are_not_user_authored():
    text = "<system-reminder>do X</system-reminder>\nPlease fix the bug in foo.py"
    assert authored_user_text(text) == "Please fix the bug in foo.py"
    assert authored_user_text("<system-reminder>only harness</system-reminder>") == ""
    assert authored_user_text("<environment_context>cwd</environment_context>") == ""


ANTHROPIC = [
    {"role": "user", "content": "<system-reminder>ctx</system-reminder>"},
    {"role": "user", "content": [{"type": "text", "text": "Run the tests please"}]},
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Running."},
            {"type": "tool_use", "id": "tu_1", "name": "Bash",
             "input": {"command": "python -m pytest tests/ -q"}},
            {"type": "tool_use", "id": "tu_2", "name": "Edit",
             "input": {"file_path": "src/app.py", "old_string": "a", "new_string": "b"}},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "tu_1",
             "content": "....\n3 passed, 1 failed in 0.5s\nExit code: 1"},
            {"type": "tool_result", "tool_use_id": "tu_2", "content": "File updated"},
        ],
    },
]


def test_anthropic_normalization_is_read_only_and_typed(identity):
    original = copy.deepcopy(ANTHROPIC)
    turn = normalize_messages("anthropic", ANTHROPIC, identity)
    assert ANTHROPIC == original  # never mutated
    kinds = [(e.kind, e.origin) for e in turn.events]
    assert (EventKind.USER_INTENT.value, Origin.HARNESS) in kinds
    assert (EventKind.USER_INTENT.value, Origin.USER) in kinds
    assert kinds.count((EventKind.TOOL_CALL.value, Origin.MODEL)) == 2
    test_events = [e for e in turn.events if e.kind == "TEST_RESULT"]
    assert len(test_events) == 1
    meta = test_events[0].redacted_metadata
    assert meta["runner"] == "pytest" and meta["passed"] == 3 and meta["failed"] == 1
    assert meta["exit_code"] == 1
    file_changes = [e for e in turn.events if e.kind == "FILE_CHANGE"]
    assert len(file_changes) == 1 and file_changes[0].resource_id == "path:src/app.py"
    assert turn.tool_names == ("Bash", "Edit")
    assert turn.latest_user_digest is not None


def test_normalization_is_deterministic_for_replay(identity):
    first = normalize_messages("anthropic", ANTHROPIC, identity)
    second = normalize_messages("anthropic", copy.deepcopy(ANTHROPIC), identity)
    assert [e.id for e in first.events] == [e.id for e in second.events]


def test_openai_chat_normalization(identity):
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [{"type": "text", "text": "commit it"}]},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "bash", "arguments": json.dumps({"command": "git commit -m x"})}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "[main abc] x"},
    ]
    turn = normalize_messages("openai", messages, identity)
    kinds = [e.kind for e in turn.events]
    assert kinds == ["USER_INTENT", "TOOL_CALL", "TOOL_RESULT", "GIT_CHANGE"]
    git = turn.events[-1]
    assert git.redacted_metadata["operation"] == "commit"


def test_openai_responses_normalization(identity):
    items = [
        {"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": "<environment_context>x</environment_context>"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "edit it"}]},
        {"type": "custom_tool_call", "call_id": "p1", "name": "apply_patch",
         "input": "*** Begin Patch\n*** Update File: a/b.py\n@@\n-x\n+y\n*** End Patch"},
        {"type": "custom_tool_call_output", "call_id": "p1", "output": "Done"},
        {"type": "reasoning", "summary": []},
    ]
    turn = normalize_messages("openai", items, identity)
    origins = [(e.kind, e.origin) for e in turn.events]
    assert origins[0] == ("USER_INTENT", Origin.HARNESS)
    assert origins[1] == ("USER_INTENT", Origin.USER)
    changes = [e for e in turn.events if e.kind == "FILE_CHANGE"]
    assert [c.redacted_metadata["path"] for c in changes] == ["a/b.py"]


def test_google_normalization(identity):
    contents = [
        {"role": "user", "parts": [{"text": "list files"}]},
        {"role": "model", "parts": [{"functionCall": {"name": "ls", "args": {"p": "."}}}]},
        {"role": "user", "parts": [{"functionResponse": {"name": "ls", "response": {"o": "a"}}}]},
    ]
    turn = normalize_messages("google", contents, identity)
    assert [e.kind for e in turn.events] == ["USER_INTENT", "TOOL_CALL", "TOOL_RESULT"]
    assert turn.events[1].source_id == turn.events[2].source_id


def test_tool_result_envelope_partial_and_exit_code():
    env = ToolResultEnvelope.from_text(
        tool_call_id="c", tool_name="Bash", text="line\n... (120 more lines)\nexit code: 2"
    )
    assert env.is_partial and env.partial_reason == "truncation_marker"
    assert env.exit_code == 2 and env.is_error
    env2 = ToolResultEnvelope.from_text(
        tool_call_id="c", tool_name="Read", text="abc", source_byte_length=10_000
    )
    assert env2.is_partial and env2.source_truncated
    assert ToolResultEnvelope.from_dict(env2.to_dict()) == env2


def test_interfaces_round_trip_and_version_check():
    hints = DecisionHints(family="f", source="jevk5", advisory={"a": 0.7, "b": 0.3},
                          choice="a", confidence=0.7, weight=0.25)
    task = TaskContext(v=1, request_id="r", workspace_id="w", session_id="s", trusted=True,
                       provider="anthropic", model="m", hints=hints)
    assert TaskContext.from_dict(task.to_dict()) == task
    bad = task.to_dict() | {"v": 99}
    with pytest.raises(InterfaceVersionError):
        TaskContext.from_dict(bad)
    auth = ActionAuthorization(v=1, workspace_id="w", session_id="s", task_revision=1,
                               scope_revision=2, resource_revision=3, expires_at=200.0,
                               decision=AuthorizationDecision.ALLOW)
    assert ActionAuthorization.from_dict(auth.to_dict()) == auth
    common = dict(workspace_id="w", session_id="s", task_revision=1, scope_revision=2,
                  resource_revision=3)
    assert auth.is_valid(**common, now=100.0)
    assert not auth.is_valid(**common, now=300.0)
    assert not auth.is_valid(**{**common, "resource_revision": 4}, now=100.0)
