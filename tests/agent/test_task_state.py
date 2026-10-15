"""Task State Compiler (#16): multi-turn requirements, corrections, evidence-backed
completion, CAS/stale revisions, replay, advisor bounds and the request fold."""

from __future__ import annotations

import pytest

from headroom.agent import registry
from headroom.agent.builtin import install_builtin_components
from headroom.agent.events import EventKind
from headroom.agent.hook import AgentRuntimeHook
from headroom.agent.identity import WorkspaceIdentity
from headroom.agent.normalize import normalize_messages
from headroom.agent.runtime import AgentRuntime
from headroom.agent.store import MemoryAgentStore
from headroom.agent.task_state import (
    BLOCK_TAG,
    HEAD_KEY,
    TASK_STATE_PROJECTOR,
    AtomKind,
    AtomState,
    TaskSnapshot,
    TaskState,
    TaskStateProjector,
    TaskStatus,
    budget_tokens,
    classify_user_turn,
    estimate_tokens,
    plan_injection,
    render_state_block,
)
from headroom.proxy.turn_hooks import (
    TurnContext,
    clear_turn_hooks,
    register_turn_hook,
    run_request_hooks,
)
from headroom.rollout import resolve_rollout

WS, SS = "ws_task", "ss_task"
IDENT = WorkspaceIdentity(WS, SS, True, "me", root=None)
MODEL = "claude-sonnet-5"


class _Config:
    def __init__(self, features: str) -> None:
        self.rollout = resolve_rollout({"HEADROOM_FEATURES": features})


@pytest.fixture()
def rt():
    registry.clear_registry()
    runtime = AgentRuntime(MemoryAgentStore())
    teardown = install_builtin_components(
        runtime, _Config("agent_runtime,agent_task_state,agent_evidence")
    )
    yield runtime
    runtime.stop()
    teardown()
    registry.clear_registry()
    clear_turn_hooks()


def _turn(rt, messages, ident=IDENT):
    turn = normalize_messages("anthropic", messages, ident)
    rt.submit(turn.envelopes)
    assert rt.flush(5)
    assert rt.flush(5)  # the TASK_STATE_CHANGE/EVIDENCE_CHANGE follow-up batch


def _state(rt, ident=IDENT) -> TaskState:
    snap = rt.services["task_state"].snapshot(ident.workspace_id, ident.session_id)
    assert snap is not None
    return snap.state


def _bash(call_id, command, output, is_error=False):
    return [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": call_id, "name": "Bash", "input": {"command": command}}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": output,
                    "is_error": is_error,
                }
            ],
        },
    ]


FIRST = (
    "Fix the unicode escape bug in parser.py so decode_escape handles \\u sequences.\n"
    "- add a regression test\n"
    "Tests must pass. Don't change the public API; never touch vendor/."
)


def test_multi_turn_requirements_and_targeted_correction(rt):
    msgs = [{"role": "user", "content": FIRST}]
    _turn(rt, msgs)
    state = _state(rt)
    assert state.goal is not None and "unicode escape bug" in state.goal.text
    assert [a.text for a in state.subgoals] == ["add a regression test"]
    assert any("Tests must pass" in a.text for a in state.acceptance)
    constraints = [a.text for a in state.hard_constraints]
    assert any("public API" in c for c in constraints)
    assert any("vendor/" in c for c in constraints)
    for atom in state.atoms:
        assert atom.origin == "user" and atom.source_event_ids

    msgs += [
        {"role": "assistant", "content": "On it."},
        {
            "role": "user",
            "content": "Actually, you may change the public API if you keep a deprecated alias.",
        },
    ]
    _turn(rt, msgs)
    state = _state(rt)
    old = next(a for a in state.atoms if "Don't change the public API" in a.text)
    new = next(a for a in state.atoms if a.supersedes_id == old.id)
    assert old.state == AtomState.SUPERSEDED.value  # history kept, not deleted
    assert new.kind == AtomKind.HARD_CONSTRAINT.value and "deprecated alias" in new.text
    # Only the targeted atom changed.
    assert any("vendor/" in a.text and a.live for a in state.hard_constraints)
    assert state.goal is not None and state.goal.live


def test_agent_claim_is_not_proof_and_tests_complete_the_task(rt):
    msgs = [{"role": "user", "content": FIRST}]
    _turn(rt, msgs)
    msgs += [{"role": "assistant", "content": "I have fixed the bug and all tests pass."}]
    _turn(rt, msgs)
    state = _state(rt)
    claim = next(a for a in state.completed if a.origin == "agent")
    assert claim.state == AtomState.PROPOSED.value
    assert state.status == TaskStatus.ACTIVE.value

    msgs += _bash(
        "t1",
        "pytest tests/test_parser.py",
        "FAILED tests/test_parser.py::test_u - AssertionError\n1 failed, 2 passed\nExit code: 1",
        is_error=True,
    )
    _turn(rt, msgs)
    state = _state(rt)
    assert state.status == TaskStatus.BLOCKED.value
    blocker = next(a for a in state.blockers if a.live)
    assert blocker.state == AtomState.ACTIVE.value and blocker.evidence_ids

    # Truncated output is never proof of success.
    msgs += _bash("t2", "pytest tests/test_parser.py", "3 passed\n[output truncated]\nExit code: 0")
    _turn(rt, msgs)
    assert _state(rt).status == TaskStatus.BLOCKED.value

    msgs += _bash("t3", "pytest tests/test_parser.py", "3 passed in 0.2s\nExit code: 0")
    _turn(rt, msgs)
    state = _state(rt)
    acceptance = next(a for a in state.acceptance if "Tests must pass" in a.text)
    assert acceptance.state == AtomState.SATISFIED.value and acceptance.evidence_ids
    assert (
        next(a for a in state.completed if a.origin == "agent").state == AtomState.SATISFIED.value
    )
    assert state.status == TaskStatus.COMPLETE.value
    # Evidence ids link to real facts in the ledger.
    evidence = rt.services["evidence"]
    for fact_id in acceptance.evidence_ids:
        assert evidence.get(WS, SS, fact_id) is not None


def test_user_acceptance_then_new_task(rt):
    msgs = [{"role": "user", "content": "Rename the helper in utils.py to parse_header."}]
    _turn(rt, msgs)
    first_id = _state(rt).id
    msgs += [
        {"role": "assistant", "content": "Renamed."},
        {"role": "user", "content": "thanks, looks good"},
    ]
    _turn(rt, msgs)
    assert _state(rt).status == TaskStatus.COMPLETE.value
    msgs += [
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Now add caching to the HTTP client in client.py."},
    ]
    _turn(rt, msgs)
    state = _state(rt)
    assert state.id != first_id and "caching" in state.goal.text
    archived = rt.store.get_projection(TASK_STATE_PROJECTOR, WS, SS, f"task:{first_id}")
    assert archived is not None and archived[1]["status"] == TaskStatus.COMPLETE.value


def test_trivial_turns_and_duplicates_do_not_churn(rt):
    msgs = [{"role": "user", "content": FIRST}]
    _turn(rt, msgs)
    revision = _state(rt).revision
    msgs += [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "continue"}]
    _turn(rt, msgs)
    assert _state(rt).revision == revision
    msgs += [
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Tests must pass."},
    ]
    _turn(rt, msgs)
    state = _state(rt)
    assert len([a for a in state.acceptance if a.live]) == 1  # duplicate merged
    assert len(next(a for a in state.acceptance).source_event_ids) == 2


def test_state_survives_trimmed_history(rt):
    msgs = [{"role": "user", "content": FIRST}]
    _turn(rt, msgs)
    # The client compacted history: the original request is gone.
    _turn(rt, [{"role": "user", "content": "Please continue with the fix in parser.py."}])
    state = _state(rt)
    assert any("public API" in a.text for a in state.hard_constraints)
    assert "unicode escape bug" in state.goal.text


def test_untrusted_scope_gets_no_state(rt):
    untrusted = WorkspaceIdentity("wu_x", "su_x", False, "me")
    _turn(rt, [{"role": "user", "content": FIRST}], untrusted)
    assert rt.services["task_state"].snapshot("wu_x", "su_x") is None


def test_sessions_are_isolated(rt):
    other = WorkspaceIdentity(WS, "ss_other", True, "me", root=None)
    _turn(rt, [{"role": "user", "content": FIRST}])
    _turn(rt, [{"role": "user", "content": "Write a README for the CLI."}], other)
    assert "unicode" in _state(rt).goal.text
    assert "README" in _state(rt, other).goal.text
    assert not _state(rt, other).hard_constraints


class _RacingTx:
    """A projection tx whose head is bumped by a concurrent writer between get and CAS."""

    def __init__(self, races: int) -> None:
        self.races = races
        self.value: dict | None = None
        self.revision = 0

    def get(self, key):
        if self.value is None and self.revision == 0:
            return None
        return (self.revision, self.value)

    def compare_and_set(self, key, expected, value):
        if key != HEAD_KEY:
            return True
        if self.races > 0:
            self.races -= 1
            self.revision += 1  # a newer writer committed first
            return False
        if expected != self.revision:
            return False
        self.value, self.revision = value, self.revision + 1
        return True


def _envelopes(text):
    return normalize_messages("anthropic", [{"role": "user", "content": text}], IDENT).envelopes


def test_stale_revision_retries_once_then_rejects():
    from headroom.agent.ledger_db import AgentLedgerDB

    db = AgentLedgerDB(None)
    projector = TaskStateProjector(db)
    tx = _RacingTx(races=1)
    projector.project(_envelopes(FIRST)[0], tx)
    assert projector.counters["cas_retries"] == 1 and tx.value is not None

    rejecting = _RacingTx(races=2)
    projector2 = TaskStateProjector(db)
    projector2.project(_envelopes(FIRST)[0], rejecting)
    assert projector2.counters["cas_rejected"] == 1
    assert rejecting.value is None  # never silently committed
    db.close()


def test_replay_rebuilds_state_from_change_events(rt):
    msgs = [{"role": "user", "content": FIRST}]
    _turn(rt, msgs)
    msgs += [
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Actually, you may change the public API."},
    ]
    _turn(rt, msgs)
    live = _state(rt)
    changes = [e for e in rt.store.events() if e.kind == EventKind.TASK_STATE_CHANGE.value]
    assert changes and all("text" not in e.redacted_metadata for e in changes)  # digest-only
    rt.store.rebuild(registry.projectors())
    rebuilt = rt.store.get_projection(TASK_STATE_PROJECTOR, WS, SS, HEAD_KEY)
    state = TaskState.from_dict(rebuilt[1])
    assert state.id == live.id and state.revision == live.revision
    assert [(a.id, a.state, a.text) for a in state.atoms] == [
        (a.id, a.state, a.text) for a in live.atoms
    ]


def test_advisor_may_only_choose_an_existing_atom():
    from headroom.agent.ledger_db import AgentLedgerDB
    from headroom.agent.store import MemoryAgentStore

    calls = []

    def chooser_factory(event):
        def choose(text, options):
            calls.append([o.id for o in options])
            return "invented-goal-id"  # not an option: must be ignored

        return choose

    db = AgentLedgerDB(None)
    store = MemoryAgentStore()
    projector = TaskStateProjector(db, chooser_factory=chooser_factory)
    msgs = [
        {
            "role": "user",
            "content": "Build the cache layer. We will use sqlite for the cache "
            "store. We will use sqlite for the queue.",
        }
    ]
    store.commit(normalize_messages("anthropic", msgs, IDENT).envelopes, [projector])
    msgs += [
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "No, use postgres instead of sqlite."},
    ]
    store.commit(normalize_messages("anthropic", msgs, IDENT).envelopes, [projector])
    head = TaskState.from_dict(store.get_projection(TASK_STATE_PROJECTOR, WS, SS, HEAD_KEY)[1])
    assert calls  # ambiguous target consulted the advisor
    new = next(a for a in head.atoms if "postgres" in a.text)
    assert new.supersedes_id in {a.id for a in head.atoms}  # deterministic fallback target
    assert not any(a.id == "invented-goal-id" for a in head.atoms)
    db.close()


def test_classify_user_turn():
    assert classify_user_turn("ok") == "trivial"
    assert classify_user_turn("No, use the v2 endpoint") == "correction"
    assert classify_user_turn("New task: add docs") == "new_task"
    assert classify_user_turn("thanks!") == "accept"
    assert classify_user_turn("thanks, but the test still fails") == "content"


# -- injection: budget, placement, cache boundary ----------------------------


def _big_snapshot() -> TaskSnapshot:
    from headroom.agent.task_state import StateAtom

    atoms = [StateAtom("g", "goal", "Ship the parser fix", "user", 0.95, "ACTIVE")]
    atoms += [
        StateAtom(
            f"c{i}",
            "hard_constraint",
            f"constraint number {i} " + "x" * 120,
            "user",
            0.95,
            "ACTIVE",
        )
        for i in range(30)
    ]
    atoms += [
        StateAtom(f"s{i}", "subgoal", f"subgoal {i} " + "y" * 150, "user", 0.95, "ACTIVE")
        for i in range(30)
    ]
    return TaskSnapshot(TaskState("tk", WS, SS, 5, "ACTIVE", tuple(atoms)))


def test_block_respects_900_token_ceiling_and_window_fraction():
    snapshot = _big_snapshot()
    block = render_state_block(snapshot, max_tokens=900)
    assert block is not None and estimate_tokens(block) <= 900
    assert "Hard constraints" in block  # absolute user requirements kept first
    assert "subgoal 29" not in block
    assert budget_tokens("claude-sonnet-5", {}) == 900
    assert budget_tokens("gpt-4", {}) == int(8192 * 0.02)  # 2% of a small window
    assert budget_tokens("claude-sonnet-5", {"agent_state_budget_tokens": 200}) == 200
    small = render_state_block(snapshot, max_tokens=200)
    assert small is not None and estimate_tokens(small) <= 200
    # Deterministic: identical state renders identical bytes (stable hash).
    assert render_state_block(snapshot, max_tokens=900) == block


def test_placement_never_touches_system_prefix_or_latest_user_message():
    block = f'<{BLOCK_TAG} hash="abc">\nStatus: ACTIVE\n</{BLOCK_TAG}>'
    user = {
        "role": "user",
        "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}],
    }
    reminder = {"role": "system", "content": "env context"}
    messages = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        user,
        reminder,
    ]
    out, placement = plan_injection("anthropic", MODEL, messages, block)
    assert placement == "anthropic_mid_system"
    assert out[:4] == messages and out[2] is user  # prefix and latest user untouched
    assert out[-1] == {"role": "system", "content": block}
    # Relocation keeps it mid-conversation for models that support it.
    from headroom.proxy.helpers import relocate_system_messages_to_top_level

    kept, system, changed = relocate_system_messages_to_top_level(out, "SYS", MODEL)
    assert system == "SYS" and kept[-1]["content"] == block
    # Models without mid-conversation system support get nothing.
    assert plan_injection("anthropic", "claude-3-5-sonnet", messages, block)[0] is None
    # Assistant-last (prefill) requests are left alone.
    assert plan_injection("anthropic", MODEL, messages[:2], block)[0] is None
    # Responses: trailing developer item; chat completions: unsupported.
    items = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x"}]}]
    out, placement = plan_injection("openai", "gpt-5", items, block)
    assert placement == "responses_developer" and out[-1]["role"] == "developer"
    assert plan_injection("openai", "gpt-5", [{"role": "user", "content": "x"}], block)[0] is None


def _hook_ctx(messages, model=MODEL, tags=None):
    return TurnContext(provider="anthropic", model=model, messages=messages, tags=tags or {})


def test_request_stage_folds_once_and_skips_unchanged(rt, monkeypatch):
    from headroom.agent import hook as hook_mod

    monkeypatch.setattr(hook_mod, "resolve_identity", lambda **kw: IDENT)
    register_turn_hook(AgentRuntimeHook(rt))
    msgs = [{"role": "user", "content": FIRST}]
    _turn(rt, msgs)
    msgs += [
        {"role": "assistant", "content": "Looking."},
        {"role": "user", "content": "go on, fix it"},
    ]
    ctx = _hook_ctx(list(msgs))
    run_request_hooks(ctx, stream_safe_only=True)
    assert ctx.tags["agent_task_state"]["injected"] is True
    assert ctx.messages[:-1] == msgs
    injected = ctx.messages[-1]
    assert injected["role"] == "system" and f"<{BLOCK_TAG}" in injected["content"]
    assert "public API" in injected["content"]
    assert ctx.agent_task_context.task_revision == _state(rt).revision  # type: ignore[attr-defined]

    # Next turn: the cache-mode replay already carries the identical block.
    replay = [
        *ctx.messages,
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "go on"},
    ]
    ctx2 = _hook_ctx(list(replay))
    run_request_hooks(ctx2, stream_safe_only=True)
    assert ctx2.tags["agent_task_state"] == {"injected": False, "reason": "unchanged_in_context"}
    assert ctx2.messages == replay

    # A correction in the current request makes the stored state stale: omit.
    ctx3 = _hook_ctx(
        [
            *msgs,
            {"role": "assistant", "content": "x"},
            {"role": "user", "content": "No, don't fix parser.py, fix lexer.py"},
        ]
    )
    run_request_hooks(ctx3, stream_safe_only=True)
    assert ctx3.tags["agent_task_state"]["reason"] == "contradicts_current_request"


def test_request_stage_never_injects_for_untrusted(rt, monkeypatch):
    from headroom.agent import hook as hook_mod

    untrusted = WorkspaceIdentity("wu_y", "su_y", False, "me")
    monkeypatch.setattr(hook_mod, "resolve_identity", lambda **kw: untrusted)
    register_turn_hook(AgentRuntimeHook(rt))
    ctx = _hook_ctx([{"role": "user", "content": FIRST}])
    run_request_hooks(ctx, stream_safe_only=True)
    assert ctx.tags["agent_task_state"]["reason"] == "untrusted_session"
    assert ctx.messages == [{"role": "user", "content": FIRST}]
