"""Unified-21 Phase 5D gate: retention feedback (#12) and bounded speculative
preparation (#15)."""

from __future__ import annotations

import json
import re
import threading
import time

import pytest

from headroom.agent import speculation as spec_mod
from headroom.agent.feedback import (
    MIN_OBSERVATIONS,
    FeedbackKey,
    RetentionFeedback,
)
from headroom.agent.speculation import RequestSpeculation, speculation_status
from headroom.agent.tool_admission.delta import ResourceVersionStore
from headroom.agent.tool_admission.dispatcher import ToolAdmissionDispatcher
from headroom.agent.tool_admission.families import ToolFamily
from headroom.agent.tool_admission.transform import ToolAdmissionTransform
from headroom.agent.tool_admission.views import INTERCEPTORS
from headroom.transforms.compression_policy import policy_default_payg
from tests.agent.test_phase5_arbiter import MODEL, Tok, _by_id, _conversation
from tests.agent.test_tool_admission import numbered, python_source

KEY = FeedbackKey("search", "search", "general", "type_specific", "claude-sonnet-4-5", "", "ws1")


# ---------------------------------------------------------------------------
# Feedback: bounded and reversible
# ---------------------------------------------------------------------------


def test_prior_moves_only_after_five_independent_observations():
    fb = RetentionFeedback()
    for _i in range(MIN_OBSERVATIONS - 1):
        assert fb.observe(KEY, 0.0) == 0.5
        assert fb.prior(KEY) == 0.5
    fb.observe(KEY, 0.0)
    expected = 0.5
    for _ in range(MIN_OBSERVATIONS):
        expected = max(0.1, min(0.9, 0.95 * expected + 0.05 * 0.0))
    assert fb.prior(KEY) == pytest.approx(expected)


def test_ewma_is_clamped_and_reversible():
    fb = RetentionFeedback()
    for _ in range(400):
        fb.observe(KEY, 0.0)
    assert fb.prior(KEY) == pytest.approx(0.1)
    for _ in range(400):
        fb.observe(KEY, 1.0)
    assert fb.prior(KEY) == pytest.approx(0.9)
    before = fb.prior(KEY)
    fb.observe(KEY, 0.0)
    after = fb.prior(KEY)
    assert after < before
    assert fb.revert(KEY) == pytest.approx(before)
    fb.reset(KEY)
    assert fb.prior(KEY) == 0.5
    fb.observe(KEY, 1.0)
    fb.enabled = False
    assert fb.prior(KEY) == 0.5  # disabled: neutral, nothing learned is applied


def test_workspace_prior_falls_back_to_global_and_keys_hold_no_raw_text():
    fb = RetentionFeedback()
    for _ in range(MIN_OBSERVATIONS):
        fb.observe(KEY.global_key(), 0.0)
    assert fb.prior(KEY) < 0.5  # workspace has no evidence yet -> global prior
    ident = KEY.ident()
    assert "ws1" in ident and "claude-sonnet-4-5" not in ident  # model generalized to family
    assert ident.split("|")[4] == "sonnet"


def test_signals_resolve_each_observation_once():
    fb = RetentionFeedback()
    fb.register("s", KEY, ccr_hash="h1", resource_id="r1", path="src/a.py")
    fb.register("s", KEY, ccr_hash="h2", resource_id="r2")
    assert fb.signal_retrieval("h1") is True
    assert fb.signal_reread("s", "r1") is False  # already resolved
    assert fb.signal_reread("s", "r2") is True
    fb.register("s", KEY, ccr_hash="h3", resource_id="r3", path="src/b.py")
    assert fb.signal_repair("s", "src/b.py") is True
    fb.register("s", KEY, ccr_hash="h4", resource_id="r4")
    assert fb.signal_correction("s", "digest-1") == 1
    assert fb.signal_correction("s", "digest-1") == 0  # same user message counted once
    fb.register("s", KEY, ccr_hash="h5", resource_id="r5")
    assert fb.signal_test_failure("s") == 1
    fb.register("s", KEY, ccr_hash="h6", resource_id="r6")
    for _ in range(4):
        fb.tick("s")
    st = fb.status()
    assert st["resolved_positive"] == 1 and st["resolved_false_candidate"] == 1
    assert st["resolved_negative"] == 4 and st["pending"] == 0


def test_ccr_retrieval_feeds_back_through_the_real_store(monkeypatch):
    import headroom.agent.feedback as fbm
    import headroom.cache.compression_store as cs

    monkeypatch.setattr(cs, "_compression_store", None)
    fb = RetentionFeedback()
    monkeypatch.setattr(fbm, "_FEEDBACK", fb)
    t = ToolAdmissionTransform(
        dispatcher=ToolAdmissionDispatcher(versions=ResourceVersionStore()),
        arbiter_enabled=True,
        feedback_enabled=True,
    )
    out = t.apply(
        _conversation(),
        Tok(),
        model=MODEL,
        model_limit=200_000,
        compression_policy=policy_default_payg(),
        context=_conversation()[0]["content"],
    )
    assert fb.status()["registered"] >= 1
    key = re.search(r"hash=([a-f0-9]{12,24})", _by_id(out.messages)["t2"]).group(1)
    cs.get_compression_store().retrieve(key)
    assert fb.status()["retrievals"] == 1 and fb.status()["resolved_negative"] == 1
    from headroom.ccr.indexed import get_indexed_ccr

    get_indexed_ccr().retrieve(cs.get_compression_store(), key, mode="search", query="Budget")
    assert fb.status()["resolved_negative"] == 1  # one observation, resolved once


def test_feedback_persists_through_the_agent_ledger(tmp_path):
    from headroom.agent.ledger_db import AgentLedgerDB

    db = AgentLedgerDB(tmp_path / "ledger.db")
    run = lambda fn: (fn(), True)[1]  # noqa: E731 - synchronous worker
    fb = RetentionFeedback()
    fb.bind(db, run)
    for _ in range(MIN_OBSERVATIONS + 2):
        fb.observe(KEY, 0.0)
    learned = fb.prior(KEY)
    fresh = RetentionFeedback()
    fresh.bind(db, run)
    assert fresh.prior(KEY) == pytest.approx(learned)
    rows = db.query("SELECT key FROM agent_feedback_priors")
    assert all("/" not in r[0] and "src" not in r[0] for r in rows)
    db.close()


def test_feedback_never_overrides_invariants(monkeypatch):
    """A maximally favourable prior cannot make a gate-rejected form win."""
    import headroom.agent.feedback as fbm
    import headroom.cache.compression_store as cs

    monkeypatch.setattr(cs, "_compression_store", None)
    fb = RetentionFeedback()
    for kind in ("type_specific", "structural", "indexed_externalization", "query_aware"):
        for _ in range(200):
            fb.observe(FeedbackKey("test", "test", "general", kind, MODEL, "", ""), 1.0)
    monkeypatch.setattr(fbm, "_FEEDBACK", fb)
    t = ToolAdmissionTransform(
        dispatcher=ToolAdmissionDispatcher(versions=ResourceVersionStore()),
        arbiter_enabled=True,
        feedback_enabled=True,
    )
    out = t.apply(
        _conversation(),
        Tok(),
        model=MODEL,
        model_limit=200_000,
        compression_policy=policy_default_payg(),
        context=_conversation()[0]["content"],
    )
    assert "FAILED tests/test_parser.py::test_unicode_escape" in _by_id(out.messages)["t1"]


# ---------------------------------------------------------------------------
# Speculation: never delays the baseline, late work discarded
# ---------------------------------------------------------------------------


def test_take_hits_prepared_work_and_misses_without_waiting():
    lane = RequestSpeculation()
    assert lane.submit("k", lambda: 41 + 1)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        hit, value = lane.take("k", wait_s=0.05)
        if hit:
            break
    assert hit and value == 42
    gate = threading.Event()
    lane.submit("slow", lambda: gate.wait(5) or "late")
    t0 = time.perf_counter()
    hit, value = lane.take("slow")  # baseline computes it itself
    assert not hit and value is None and time.perf_counter() - t0 < 0.05
    before = speculation_status()["discarded_late"]
    gate.set()
    deadline = time.monotonic() + 2
    while speculation_status()["discarded_late"] == before and time.monotonic() < deadline:
        time.sleep(0.01)
    assert speculation_status()["discarded_late"] == before + 1
    assert lane.take("slow") == (False, None)  # the stale result never surfaces


def test_one_worker_per_request_queue_bound_and_close_cancels():
    lane = RequestSpeculation()
    running = []
    lock = threading.Lock()
    gate = threading.Event()

    def job():
        with lock:
            running.append(1)
            assert len(running) == 1
        gate.wait(2)
        with lock:
            running.pop()

    accepted = [lane.submit(i, job) for i in range(spec_mod.MAX_QUEUED_PER_REQUEST + 3)]
    assert accepted.count(False) >= 2  # bounded queue
    cancelled_before = speculation_status()["cancelled"]
    t0 = time.perf_counter()
    lane.close()  # never waits for the running task
    assert time.perf_counter() - t0 < 0.05
    assert speculation_status()["cancelled"] > cancelled_before
    assert not lane.submit("after", job)
    gate.set()


def test_deadline_expired_work_is_not_started(monkeypatch):
    lane = RequestSpeculation(deadline_s=0.0)
    ran = []
    lane.submit("x", lambda: ran.append(1))
    time.sleep(0.05)
    assert ran == []


def _slow_read_interceptor(delay: float):
    real = INTERCEPTORS[ToolFamily.READ]

    class Slow:
        family = ToolFamily.READ

        def view(self, envelope, ctx):
            time.sleep(delay)
            return real.view(envelope, ctx)

    return {**INTERCEPTORS, ToolFamily.READ: Slow()}


def _reads():
    srcs = [numbered(python_source(60 + i)) for i in range(3)]
    uses = [
        {
            "type": "tool_use",
            "id": f"r{i}",
            "name": "Read",
            "input": {"file_path": f"/work/repo/src/m{i}.py"},
        }
        for i in range(3)
    ]
    results = [
        {"type": "tool_result", "tool_use_id": f"r{i}", "content": s} for i, s in enumerate(srcs)
    ]
    return [
        {"role": "user", "content": "What does compute_checksum return?"},
        {"role": "assistant", "content": uses},
        {"role": "user", "content": results},
    ]


@pytest.mark.parametrize("delay", [0.0, 0.3])
def test_speculation_on_or_off_gives_identical_winners(monkeypatch, delay):
    import headroom.cache.compression_store as cs

    monkeypatch.setattr(cs, "_compression_store", None)
    outs = []
    for enabled in (False, True):
        t = ToolAdmissionTransform(
            dispatcher=ToolAdmissionDispatcher(
                interceptors=_slow_read_interceptor(delay), versions=ResourceVersionStore()
            ),
            arbiter_enabled=True,
            speculation_enabled=enabled,
        )
        out = t.apply(
            _reads(),
            Tok(),
            model=MODEL,
            model_limit=200_000,
            compression_policy=policy_default_payg(),
            context="What does compute_checksum return?",
        )
        outs.append(json.dumps(out.messages, sort_keys=True))
    assert outs[0] == outs[1]


def test_request_completion_does_not_wait_for_stale_futures(monkeypatch):
    """A speculative task still running when the request finishes is abandoned."""
    import headroom.cache.compression_store as cs

    monkeypatch.setattr(cs, "_compression_store", None)
    t = ToolAdmissionTransform(
        dispatcher=ToolAdmissionDispatcher(
            interceptors=_slow_read_interceptor(0.4), versions=ResourceVersionStore()
        ),
        arbiter_enabled=True,
        speculation_enabled=True,
    )
    t0 = time.perf_counter()
    t.apply(
        _reads(),
        Tok(),
        model=MODEL,
        model_limit=200_000,
        compression_policy=policy_default_payg(),
        context="What does compute_checksum return?",
    )
    elapsed = time.perf_counter() - t0
    # Three slow views computed inline would take >= 1.2 s; with the lane they
    # overlap, and the request never waits beyond its own inline work.
    assert elapsed < 1.2 + 0.6
    st = speculation_status()
    assert st["hits"] + st["misses"] >= 2
