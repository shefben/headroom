"""Evidence ledger (#17): extraction, provenance, contradiction, supersession,
invalidation, retraction, replay and scope isolation."""

from __future__ import annotations

import pytest

from headroom.agent.events import EventEnvelope, EventKind, Origin, make_event
from headroom.agent.evidence import (
    AgentEvidenceStore,
    EvidenceFact,
    EvidenceProjector,
    EvidenceStatus,
    claim_key_for,
    extract_candidates,
    fact_id_for,
)
from headroom.agent.identity import WorkspaceIdentity
from headroom.agent.ledger_db import AgentLedgerDB
from headroom.agent.normalize import normalize_messages
from headroom.agent.store import MemoryAgentStore

WS, SS = "ws_test", "ss_test"
IDENT = WorkspaceIdentity(WS, SS, True, "me", root=None)


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture()
def ledger():
    db = AgentLedgerDB(None)
    yield db
    db.close()


def _store(ledger, clock=None):
    return AgentEvidenceStore(ledger, clock=clock or Clock())


def _bash(call_id: str, command: str, output: str, *, is_error: bool = False) -> list[dict]:
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


def _observe(store, messages, ident=IDENT):
    """Project every event of a request through the evidence projector."""
    emitted: list = []
    projector = EvidenceProjector(store, emit=emitted.extend)
    turn = normalize_messages("anthropic", messages, ident)
    for env in turn.envelopes:
        projector.project(env, None)
    return turn, emitted


def test_extracts_exit_test_error_and_negative_facts(ledger):
    store = _store(ledger)
    messages = [
        {"role": "user", "content": "run the tests"},
        *_bash(
            "t1",
            "pytest tests/test_a.py",
            "E   ModuleNotFoundError: No module named 'foo'\n1 failed in 0.1s\nExit code: 1",
            is_error=True,
        ),
        *_bash(
            "t2",
            "foobar --version",
            "bash: foobar: command not found\nExit code: 127",
            is_error=True,
        ),
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "r1",
                    "name": "Read",
                    "input": {"file_path": "missing.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "r1",
                    "content": "File does not exist.",
                    "is_error": True,
                }
            ],
        },
    ]
    _observe(store, messages)
    facts = {f.claim_key: f for f in store.facts(WS, SS)}
    test = facts[claim_key_for("test.status", "pytest tests/test_a.py")]
    assert test.value["status"] == "failed" and test.polarity == "negative"
    assert facts[claim_key_for("command.exit", "pytest tests/test_a.py")].value == 1
    sig = facts[claim_key_for("error.signature", "pytest tests/test_a.py")]
    assert "ModuleNotFoundError" in sig.value
    assert facts[claim_key_for("tool.available", "foobar")].value is False
    assert facts[claim_key_for("file.exists", "path:missing.py")].value is False
    # Provenance: every fact names its exact source event and digest.
    for fact in facts.values():
        assert fact.source_event_id.startswith("ev_") and len(fact.source_hash) == 64
        assert fact.workspace_id == WS and fact.session_id == SS


def test_newer_observation_supersedes_and_success_stales_error(ledger):
    store = _store(ledger)
    first = [
        {"role": "user", "content": "fix"},
        *_bash("t1", "pytest -q", "AssertionError: boom\n1 failed\nExit code: 1", is_error=True),
    ]
    _observe(store, first)
    second = first + _bash("t2", "pytest -q", "3 passed in 0.1s\nExit code: 0")
    _observe(store, second)
    key = claim_key_for("test.status", "pytest -q")
    heads = store.heads(WS, SS, key)
    assert len(heads) == 1 and heads[0].value["status"] == "passed"
    old = [f for f in store.facts(WS, SS, claim_key=key) if f.id != heads[0].id]
    assert old[0].status == EvidenceStatus.SUPERSEDED.value
    assert heads[0].supersedes_id == old[0].id
    sig = store.facts(WS, SS, claim_key=claim_key_for("error.signature", "pytest -q"))
    assert sig and sig[0].status == EvidenceStatus.STALE.value


def test_user_statement_vs_observation_coexist_as_contradicted(ledger):
    store = _store(ledger)
    msgs = [{"role": "user", "content": "docker is installed on this box"}]
    _observe(store, msgs)
    msgs = msgs + _bash("t1", "docker ps", "docker: command not found", is_error=True)
    _observe(store, msgs)
    user_key = claim_key_for("state.installed", "docker")
    user_fact = store.heads(WS, SS, user_key)[0]
    assert user_fact.source_kind == "user" and user_fact.value is True
    # Different claim keys stay independent; a same-claim conflict is labelled:
    conflict_event = make_event(
        workspace_id=WS,
        session_id=SS,
        kind=EventKind.TOOL_RESULT,
        origin=Origin.TOOL,
        source_id="probe",
        index=0,
        payload="probe",
    )
    from headroom.agent.evidence import FactCandidate, SourceKind

    store.observe(
        conflict_event,
        [FactCandidate("docker", "state.installed", False, "negative", SourceKind.TOOL_TEXT)],
    )
    heads = store.heads(WS, SS, user_key)
    assert {h.status for h in heads} == {EvidenceStatus.CONTRADICTED.value}
    assert len(heads) == 2  # coexist: no averaging of incompatible booleans
    assert heads[0].confidence != heads[1].confidence or heads[0].value != heads[1].value
    # A later observation that agrees with the user resolves the conflict.
    resolve = make_event(
        workspace_id=WS,
        session_id=SS,
        kind=EventKind.TOOL_RESULT,
        origin=Origin.TOOL,
        source_id="probe2",
        index=0,
        payload="probe2",
    )
    store.observe(
        resolve,
        [FactCandidate("docker", "state.installed", True, "positive", SourceKind.TOOL_EXIT)],
    )
    heads = store.heads(WS, SS, user_key)
    assert all(h.status == EvidenceStatus.ACTIVE.value for h in heads)
    assert all(h.value is True for h in heads)
    # Corroboration raised the user's fact confidence and recorded the event.
    corroborated = store.get(WS, SS, user_fact.id)
    assert corroborated.confidence > user_fact.confidence
    assert resolve.id in corroborated.metadata["corroborated_by"]


def test_file_change_stales_content_bound_facts(ledger):
    store = _store(ledger)
    read = [
        {"role": "user", "content": "look at a.py"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "r1", "name": "Read", "input": {"file_path": "a.py"}}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "r1", "content": "def f():\n    return 1\n"}
            ],
        },
    ]
    _observe(store, read)
    hash_key = claim_key_for("file.content_hash", "path:a.py")
    assert store.heads(WS, SS, hash_key)[0].status == EvidenceStatus.ACTIVE.value
    edit = read + [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "e1",
                    "name": "Edit",
                    "input": {"file_path": "a.py", "old_string": "1", "new_string": "2"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "e1", "content": "ok"}],
        },
    ]
    _observe(store, edit)
    assert store.heads(WS, SS, hash_key) == []
    stale = store.facts(WS, SS, claim_key=hash_key)[0]
    assert stale.status == EvidenceStatus.STALE.value
    assert stale.metadata["reason"] == "content_changed"
    assert store.heads(WS, SS, claim_key_for("file.modified", "path:a.py"))[0].value is True


def test_volatile_port_fact_expires(ledger):
    clock = Clock()
    store = _store(ledger, clock)
    _observe(
        store,
        [
            {"role": "user", "content": "start it"},
            *_bash(
                "t1",
                "npm start",
                "Error: listen EADDRINUSE: address already in use :::3000",
                is_error=True,
            ),
        ],
    )
    key = claim_key_for("port.in_use", "port:3000")
    assert store.heads(WS, SS, key)[0].valid_until == pytest.approx(clock.now + 300)
    clock.now += 301
    _observe(
        store, [{"role": "user", "content": "start it"}, *_bash("t2", "ls", "a\nb\nExit code: 0")]
    )
    assert store.facts(WS, SS, claim_key=key)[0].status == EvidenceStatus.STALE.value
    assert store.germane(WS, SS, task_id=None, resources=["port:3000"]) == []


def test_git_mutation_stales_git_state(ledger):
    store = _store(ledger)
    status = "On branch main\nChanges not staged for commit:\n\tmodified:   src/x.py\n"
    msgs = [{"role": "user", "content": "commit"}, *_bash("g1", "git status", status)]
    _observe(store, msgs)
    branch_key = claim_key_for("git.branch", "HEAD")
    assert store.heads(WS, SS, branch_key)[0].value == "main"
    changed = store.heads(WS, SS, claim_key_for("git.changed_paths", "worktree"))[0]
    assert changed.value == {"count": 1, "paths": ["src/x.py"]}
    _observe(store, msgs + _bash("g2", "git commit -am wip", "[main abc1234] wip\nExit code: 0"))
    assert store.heads(WS, SS, branch_key) == []


def test_user_retraction(ledger):
    store = _store(ledger)
    msgs = [{"role": "user", "content": "redis is running locally"}]
    _observe(store, msgs)
    key = claim_key_for("state.running", "redis")
    assert store.heads(WS, SS, key)
    _observe(
        store,
        msgs
        + [
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "ignore what I said about redis"},
        ],
    )
    assert store.heads(WS, SS, key) == []
    assert store.facts(WS, SS, claim_key=key)[0].status == EvidenceStatus.RETRACTED.value


def test_secrets_and_whole_files_are_never_persisted(ledger):
    store = _store(ledger)
    secret = "sk-ant-api03-" + "A" * 40
    big = "x = 1\n" * 5000
    msgs = [
        {"role": "user", "content": "go"},
        *_bash("t1", "deploy", f"Error: auth failed token={secret}\nExit code: 1", is_error=True),
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "r1", "name": "Read", "input": {"file_path": "big.py"}}
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "r1", "content": big}]},
    ]
    _observe(store, msgs)
    dump = repr(ledger.query("SELECT * FROM evidence_facts"))
    assert secret not in dump
    assert "x = 1" not in dump
    assert all(len(f.value_json) <= 160 for f in store.facts(WS, SS))


def test_replay_rebuilds_identical_ledger(ledger):
    store = _store(ledger)
    event_store = MemoryAgentStore()
    emitted: list = []
    projector = EvidenceProjector(store, emit=emitted.extend)
    msgs = [
        {"role": "user", "content": "docker is installed"},
        *_bash("t1", "pytest", "1 failed\nExit code: 1", is_error=True),
        *_bash("t2", "pytest", "2 passed\nExit code: 0"),
    ]
    turn = normalize_messages("anthropic", msgs, IDENT)
    event_store.commit(turn.envelopes, [projector])
    event_store.commit(list(emitted), [projector])  # the EVIDENCE_CHANGE audit trail
    live = {f.id: f for f in store.facts(WS, SS)}
    assert live and all(e.event.kind == EventKind.EVIDENCE_CHANGE.value for e in emitted)

    fresh_db = AgentLedgerDB(None)
    fresh = AgentEvidenceStore(fresh_db)
    fresh.replay(event_store.events())
    replayed = {f.id: f for f in fresh.facts(WS, SS)}
    assert replayed.keys() == live.keys()
    for fact_id, fact in live.items():
        other = replayed[fact_id]
        assert (other.status, other.value_json, other.source_event_id, other.supersedes_id) == (
            fact.status,
            fact.value_json,
            fact.source_event_id,
            fact.supersedes_id,
        )
    # Every fact is traceable to an event that exists in the canonical store.
    event_ids = {e.id for e in event_store.events()}
    assert all(f.source_event_id in event_ids for f in replayed.values())
    # Re-projecting the same raw events (store rebuild) changes nothing.
    before = len(store.facts(WS, SS))
    event_store.rebuild([projector])
    assert len(store.facts(WS, SS)) == before
    fresh_db.close()


def test_scopes_are_isolated(ledger):
    store = _store(ledger)
    other = WorkspaceIdentity("ws_other", "ss_other", True, "me", root=None)
    _observe(
        store,
        [{"role": "user", "content": "x"}, *_bash("t1", "make", "Exit code: 2", is_error=True)],
    )
    _observe(store, [{"role": "user", "content": "y"}, *_bash("t1", "make", "Exit code: 0")], other)
    mine = store.facts(WS, SS)
    theirs = store.facts("ws_other", "ss_other")
    assert {f.value for f in mine if f.predicate == "command.exit"} == {2}
    assert {f.value for f in theirs if f.predicate == "command.exit"} == {0}
    assert store.facts(WS, "ss_other") == []


def test_fact_id_deterministic_and_meta_roundtrip(ledger):
    store = _store(ledger)
    _observe(store, [{"role": "user", "content": "x"}, *_bash("t1", "make", "Exit code: 0")])
    fact = store.facts(WS, SS)[0]
    assert fact.id == fact_id_for(WS, SS, fact.claim_key, fact.source_event_id)
    clone = EvidenceFact.from_meta(WS, SS, fact.to_meta())
    assert clone.to_dict() == fact.to_dict()


def test_extract_ignores_partial_test_output():
    event = make_event(
        workspace_id=WS,
        session_id=SS,
        kind=EventKind.TEST_RESULT,
        origin=Origin.TOOL,
        source_id="t",
        index=0,
        payload="p",
        metadata={
            "runner": "pytest",
            "command_head": "pytest",
            "exit_code": 0,
            "passed": 3,
            "is_partial": True,
        },
    )
    (cand,) = extract_candidates(EventEnvelope(event=event, content="..."))
    assert cand.value["status"] == "inconclusive"
