"""Unified-21 Phase 8 (plan item 8): transform holdout and automatic revert of a
regressing tool family to reversible-only admission, through the real
dispatcher, arbiter and proxy transform."""

from __future__ import annotations

import pytest

from headroom.agent import feedback as fbm
from headroom.agent.arbiter import get_arbiter, reset_arbiter
from headroom.agent.feedback import (
    ARM_CONTROL,
    REVERT_MIN_OBSERVATIONS,
    FeedbackKey,
    RetentionFeedback,
    transform_arm,
    transform_holdout_fraction,
)
from headroom.agent.tool_admission.delta import ResourceVersionStore
from headroom.agent.tool_admission.dispatcher import ToolAdmissionDispatcher
from headroom.agent.tool_admission.transform import ToolAdmissionTransform
from tests.agent.test_phase5_arbiter import _by_id, _conversation, _run
from tests.agent.test_tool_admission import pytest_output, search_output

KEY = FeedbackKey("search", "search", "general", "type_specific", "claude-sonnet-4-5", "", "")


def _observe(fb: RetentionFeedback, arm: str, n: int, negatives: int, family: str = "search"):
    key = FeedbackKey(family, family, "general", "type_specific", "claude-sonnet-4-5", "", "")
    for i in range(n):
        rid = f"{arm}-{family}-{i}"
        fb.register("s", key, resource_id=rid, arm=arm)
        if i < negatives:
            assert fb.signal_reread("s", rid) is True
        else:
            assert fb._resolve(fb._by_resource[("s", rid)], fbm.OUTCOME_POSITIVE) is True


@pytest.fixture
def fresh(monkeypatch):
    import headroom.cache.compression_store as cs

    monkeypatch.setattr(cs, "_compression_store", None)
    fb = RetentionFeedback()
    monkeypatch.setattr(fbm, "_FEEDBACK", fb)
    reset_arbiter()
    yield fb, cs
    reset_arbiter()
    monkeypatch.setattr(cs, "_compression_store", None)


def _transform():
    return ToolAdmissionTransform(
        dispatcher=ToolAdmissionDispatcher(versions=ResourceVersionStore()),
        arbiter_enabled=True,
        feedback_enabled=True,
    )


def test_holdout_fraction_is_bounded_and_assignment_scope_stable(monkeypatch):
    monkeypatch.delenv("HEADROOM_TRANSFORM_HOLDOUT", raising=False)
    assert transform_holdout_fraction() == 0.0
    assert transform_arm("ws|s1") == "treatment"
    for raw, want in (("0.2", 0.2), ("9", 0.5), ("-1", 0.0), ("nan", 0.0), ("junk", 0.0)):
        monkeypatch.setenv("HEADROOM_TRANSFORM_HOLDOUT", raw)
        assert transform_holdout_fraction() == want
    monkeypatch.setenv("HEADROOM_TRANSFORM_HOLDOUT", "0.5")
    arms = {s: transform_arm(f"ws|{s}") for s in range(400)}
    assert arms == {s: transform_arm(f"ws|{s}") for s in range(400)}  # stable per scope
    share = sum(1 for a in arms.values() if a == ARM_CONTROL) / len(arms)
    assert 0.4 < share < 0.6


def test_control_observations_never_move_strategy_priors():
    fb = RetentionFeedback()
    _observe(fb, ARM_CONTROL, 40, 40)
    assert fb.prior(KEY) == 0.5 and fb.observations(KEY) == 0
    assert fb.status()["arms"]["search:control"] == {"resolved": 40, "negative": 40}
    assert fb.status()["control_registered"] == 40


def test_regressing_family_reverts_against_the_control_arm_only():
    fb = RetentionFeedback()
    _observe(fb, ARM_CONTROL, REVERT_MIN_OBSERVATIONS, 2)  # 25% baseline rereads
    _observe(fb, "treatment", REVERT_MIN_OBSERVATIONS, 3)  # 37.5%: within the margin
    assert not fb.family_reverted("search")
    _observe(fb, "treatment", REVERT_MIN_OBSERVATIONS, REVERT_MIN_OBSERVATIONS)
    assert fb.family_reverted("search")
    info = fb.status()["reverted_families"]["search"]
    # Reverted at the first crossing: 5/10 = 50% >= 25% control + 25% margin.
    assert info["control_negative_rate"] == 0.25 and info["treatment_resolved"] == 10
    assert info["treatment_negative_rate"] == 0.5
    assert not fb.family_reverted("read")  # per family
    fb.enabled = False
    assert not fb.family_reverted("search")  # feedback off: no revert applies
    fb.enabled = True
    fb.reinstate("search")
    assert not fb.family_reverted("search")
    assert "search:treatment" not in fb.status()["arms"]  # comparison restarts


def test_without_a_control_arm_an_absolute_rate_reverts():
    fb = RetentionFeedback()
    _observe(fb, "treatment", REVERT_MIN_OBSERVATIONS - 1, REVERT_MIN_OBSERVATIONS - 1)
    assert not fb.family_reverted("search")  # too few observations
    _observe(fb, "treatment", 1, 0, family="search")
    assert fb.family_reverted("search")  # 7/8 >= .5
    assert fb.status()["family_reverts"] == 1
    fb.reset()
    assert not fb.family_reverted("search") and fb.status()["arms"] == {}


def test_reverted_family_admits_only_reversible_forms_and_original_stays_exact(fresh):
    fb, cs = fresh
    baseline = _by_id(_run(_transform(), _conversation()).messages)
    assert "hash=" in baseline["t2"]  # search is normally admitted lossy
    reset_arbiter()
    cs._compression_store = None
    _observe(fb, "treatment", REVERT_MIN_OBSERVATIONS, REVERT_MIN_OBSERVATIONS)
    assert fb.family_reverted("search")
    texts = _by_id(_run(_transform(), _conversation()).messages)
    assert texts["t2"] == search_output()  # no reversible search form: exact original
    assert texts["t1"] == baseline["t1"]  # other families unaffected
    rejections = get_arbiter().status()["candidate_rejections"]
    assert rejections.get("family_reverted", 0) >= 1


def test_holdout_arm_forwards_originals_and_counts_forgone_not_realized(fresh, monkeypatch):
    fb, cs = fresh
    monkeypatch.setenv("HEADROOM_TRANSFORM_HOLDOUT", "0.5")
    monkeypatch.setattr("headroom.agent.tool_admission.transform._arm", lambda scope: ARM_CONTROL)
    t = _transform()
    out = _run(t, _conversation())
    texts = _by_id(out.messages)
    assert texts["t1"] == pytest_output() and texts["t2"] == search_output()
    assert out.messages == _conversation()
    status = get_arbiter().status()
    assert status["holdout"]["units"] >= 2 and status["holdout"]["forgone_tokens"] > 0
    assert status["realized"]["tokens_saved"] == 0
    st = fb.status()
    assert st["control_registered"] >= 2 and st["registered"] == st["control_registered"]
    assert t.dispatcher.status()["stages"].get("holdout_original", 0) >= 2


def test_treatment_arm_is_unchanged_by_the_holdout_machinery(fresh, monkeypatch):
    monkeypatch.delenv("HEADROOM_TRANSFORM_HOLDOUT", raising=False)
    first = _by_id(_run(_transform(), _conversation()).messages)
    reset_arbiter()
    monkeypatch.setattr("headroom.agent.tool_admission.transform._arm", lambda scope: "treatment")
    second = _by_id(_run(_transform(), _conversation()).messages)
    assert first == second
    assert get_arbiter().status()["holdout"]["units"] == 0
