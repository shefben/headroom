"""Unified-21: JevK5 is actually consulted on the Claude/Anthropic paths.

Covers the Scenario A-J gate from the integration brief: the uncertainty
triggers (an obvious decision calls nothing, an ambiguous one calls once), the
single proxy-owned advisor, the shared per-request allowance, the live
``/health`` counters, and every fail-open path.

The advisor here is the real :class:`DecisionAdvisor` driven against the fake
llama-server, so budgets, the TTL cache, the circuit breaker, the confidence
floor and the decision deadline are the production ones.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from headroom import paths
from headroom.agent import complexity, registry
from headroom.agent.arbiter import (
    ArbiterContext,
    CandidateRecord,
    Evaluation,
    TransformArbiter,
    arbiter_uncertainty,
)
from headroom.agent.builtin import _chooser_factory
from headroom.agent.complexity import route_request, should_consult_advisor
from headroom.agent.tool_catalog import (
    CATALOG_FINALISTS,
    CatalogRequest,
    ProgressiveToolCatalog,
    cut_uncertainty,
    get_catalog,
)
from headroom.intelligence import policy
from headroom.intelligence.advisor import DecisionAdvisor, RequestBudget
from headroom.intelligence.settings import IntelligenceSettings
from headroom.proxy.server import ProxyConfig, create_app
from headroom.proxy.turn_hooks import clear_turn_hooks
from headroom.rollout import apply_wrap_feature_defaults, resolve_rollout

from ..intelligence.fake_llama import FakeLlamaServer
from .test_phase5_catalog import anthropic_tools

MODEL = "claude-sonnet-4-5"
PORT = 18981
LOOPBACK = ("127.0.0.1", 50000)
FAMILIES = ("complexity_router", "transform_arbiter", "tool_catalog", "task_state")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("HEADROOM_OUTPUT_HOLDOUT", raising=False)
    monkeypatch.delenv("HEADROOM_OUTPUT_SHAPER", raising=False)
    monkeypatch.delenv("HEADROOM_JEVK5_AMBIGUITY", raising=False)
    complexity.reset()
    policy.reset_consumers()
    get_catalog().reset()
    yield
    complexity.reset()
    policy.reset_consumers()


def config(extra: dict | None = None):
    env = apply_wrap_feature_defaults(dict(extra or {}))
    return SimpleNamespace(rollout=resolve_rollout(env), optimize=True, mode="token")


def advisor_for(url: str, **overrides) -> DecisionAdvisor:
    return DecisionAdvisor(IntelligenceSettings(**overrides), lambda: ("llama-server", url))


def claude_body(result: str, *, effort: str = "xhigh", is_error: bool = False) -> dict:
    """One Anthropic ``/v1/messages`` body mid-tool-loop (a mechanical continuation)."""
    return {
        "model": MODEL,
        "max_tokens": 1000,
        "output_config": {"effort": effort},
        "messages": [
            {"role": "user", "content": "Refactor the parser"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Running the checks."},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Bash",
                        "input": {"command": "make check"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": result,
                        "is_error": is_error,
                    }
                ],
            },
        ],
    }


def route_claude(body: dict, advisor, monkeypatch, *, request_id="req-1", session="s"):
    monkeypatch.setattr(
        complexity, "_snapshot_and_advisor", lambda hint: (None, advisor, f"ws|{hint}")
    )
    return route_request(
        body,
        shape="anthropic",
        config=config(),
        model=MODEL,
        request_id=request_id,
        session_key=session,
    )


class CountingAdvisor:
    """Minimal advisor double that records the options it was shown."""

    readiness = "ready"

    def __init__(self, pick: str | None = None, weight: float = 0.4) -> None:
        self.pick = pick
        self.weight = weight
        self.calls: list[dict] = []
        self._budget = RequestBudget(4)

    def request_budget(self, request_key=None):  # noqa: ANN001
        return self._budget

    def choose(self, family, state, instructions, options, *, budget):  # noqa: ANN001
        self.calls.append(dict(options))
        if not budget.take():
            return None
        key = self.pick if self.pick in options else next(iter(options))
        return SimpleNamespace(
            value=key,
            probabilities={key: 0.95},
            confidence=0.95,
            weight=self.weight,
            latency_ms=5.0,
        )


# ---------------------------------------------------------------------------
# Scenario A/B: Claude complexity routing
# ---------------------------------------------------------------------------


def test_scenario_a_borderline_claude_complexity_reaches_jevk5(monkeypatch):
    """A genuinely borderline Anthropic turn produces a real advisor call."""
    with FakeLlamaServer() as fake:
        advisor = advisor_for(fake.url)
        body = claude_body("src/pkg/alpha.py: ok\nsrc/pkg/beta.py: ok")
        decision = route_claude(body, advisor, monkeypatch, request_id="req-A")
        assert decision is not None and decision.score_source == "jevk5"
        assert advisor.stats.calls == 1 and advisor.stats.successes == 1
        assert advisor.stats.by_family == {"effort_complexity": 1}
        # JevK5 answers one side of one boundary, and may lower but never
        # raise above what the client sent.
        assert decision.score in (0, 1, 2)
        assert body["output_config"]["effort"] in ("medium", "high", "xhigh")
        stats = policy.consumer_stats("complexity_router").snapshot()
        assert stats["eligible"] == 1 and stats["calls_requested"] == 1
        assert stats["advice_received"] == 1 and stats["advice_applied"] == 1
        # And the live advisor exposes the per-family attribution.
        assert advisor.status()["stats"]["by_family"]["effort_complexity"] == 1


def test_scenario_b_obvious_claude_complexity_calls_nothing(monkeypatch):
    with FakeLlamaServer() as fake:
        advisor = advisor_for(fake.url)
        body = claude_body("All 12 checks passed.")
        decision = route_claude(body, advisor, monkeypatch, request_id="req-B")
        assert decision is not None and decision.score_source == "deterministic"
        assert advisor.stats.calls == 0 and fake.cfg.completions == 0
        stats = policy.consumer_stats("complexity_router").snapshot()
        assert stats["calls_requested"] == 0
        assert stats["skipped"]["no_complexity_evidence"] == 1


def test_protected_claude_turn_is_never_lowered_and_asks_nothing(monkeypatch):
    with FakeLlamaServer() as fake:
        advisor = advisor_for(fake.url)
        body = claude_body(
            'Traceback (most recent call last):\n  File "src/pkg/alpha.py", line 3\nError: boom',
            is_error=True,
        )
        decision = route_claude(body, advisor, monkeypatch, request_id="req-P")
        assert decision is not None and decision.protected == "error_continuation"
        assert body["output_config"]["effort"] == "xhigh"  # untouched
        assert advisor.stats.calls == 0


def test_effort_already_at_the_floor_is_never_a_question(monkeypatch):
    with FakeLlamaServer() as fake:
        advisor = advisor_for(fake.url)
        body = claude_body("src/pkg/alpha.py: ok\nsrc/pkg/beta.py: ok", effort="medium")
        route_claude(body, advisor, monkeypatch, request_id="req-F")
        assert advisor.stats.calls == 0
        skipped = policy.consumer_stats("complexity_router").snapshot()["skipped"]
        assert skipped["no_effort_change_possible"] == 1


def test_complexity_uncertainty_is_a_policy_not_one_score_bucket():
    base = {"turn_kind": "mechanical_continuation", "unknown_requirements": True}
    clean = complexity.ComplexitySignals(**base)
    assert should_consult_advisor(clean, clean.score, effort="xhigh").blocked
    loud = complexity.ComplexitySignals(
        **base,
        path_count=9,
        multi_file=True,
        new_symbol_count=40,
        new_symbols=True,
        hedge_count=6,
        uncertainty=True,
    )
    assert not should_consult_advisor(loud, loud.score, effort="xhigh").ambiguous
    borderline = complexity.ComplexitySignals(**base, path_count=2)
    verdict = should_consult_advisor(borderline, borderline.score, effort="xhigh")
    assert verdict.ambiguous and "borderline_threshold" in verdict.reasons
    # An effort whose routed value cannot move is never asked about.
    assert should_consult_advisor(borderline, borderline.score, effort="low").blocked
    assert should_consult_advisor(borderline, borderline.score, effort=None).blocked


# ---------------------------------------------------------------------------
# Scenario C/D: transform arbiter
# ---------------------------------------------------------------------------


def _rec(
    kind: str,
    *,
    tokens: int,
    net_cost: float,
    relevance: float = 1.0,
    soft_recall: float = 1.0,
    recovery: float = 1.0,
    prior: float = 0.5,
    allowed: bool = True,
    reasons: tuple[str, ...] = (),
) -> CandidateRecord:
    return CandidateRecord(
        kind=kind,
        label=kind,
        tier="aggressive",
        tokens=tokens,
        allowed=allowed,
        reasons=reasons,
        soft_recall=soft_recall,
        relevance=relevance,
        recovery=recovery,
        prior=prior,
        net_cost=net_cost,
    )


def _close_records() -> tuple[CandidateRecord, ...]:
    return (
        _rec("original", tokens=100, net_cost=100.0),
        _rec("structural", tokens=50, net_cost=50.0, relevance=0.80, soft_recall=0.85,
             recovery=0.90),
    )


def _decisive_records() -> tuple[CandidateRecord, ...]:
    return (
        _rec("original", tokens=100, net_cost=100.0),
        _rec("structural", tokens=95, net_cost=95.0, relevance=0.20, soft_recall=0.30,
             recovery=0.30),
    )


def _utilities(records) -> TransformArbiter:
    """Run ``choose`` once so every record carries its components and utility."""
    arb = TransformArbiter()
    ctx = ArbiterContext(original="x" * 100)
    arb.choose(Evaluation(100, tuple(records)), ctx)
    return arb


def test_scenario_c_and_d_arbiter_ambiguity_is_real_decision_uncertainty():
    """Missing query terms is not ambiguity; close, unstable utilities are."""
    ctx = ArbiterContext(original="x" * 100, ambiguous=True)  # no task evidence
    decisive = list(_decisive_records())
    _utilities(decisive)
    verdict = arbiter_uncertainty([r for r in decisive if r.allowed], ctx)
    assert not verdict.ambiguous, verdict
    close = list(_close_records())
    _utilities(close)
    verdict = arbiter_uncertainty([r for r in close if r.allowed], ctx)
    assert verdict.ambiguous and "close_utilities" in verdict.reasons
    assert arbiter_uncertainty(close[:1], ctx).blocked == "single_candidate"


def test_scenario_d_clear_transform_never_calls_jevk5():
    adv = CountingAdvisor()
    ctx = ArbiterContext(
        original="x" * 100,
        ambiguous=True,
        advisor=adv,
        advisor_budget=adv.request_budget("req-D"),
    )
    arb = TransformArbiter()
    decision = arb.choose(Evaluation(100, _decisive_records()), ctx)
    assert adv.calls == [] and decision.jev_weight == 0.0
    assert arb.telemetry.snapshot()["jev"] == {
        "eligible": 0,
        "calls": 0,
        "applied": 0,
        "consumer": policy.consumer_stats("transform_arbiter").snapshot(),
    }


def test_scenario_c_ambiguous_transform_blends_within_the_weight_cap():
    adv = CountingAdvisor(weight=0.9)  # the advisor's own cap applies, not this
    ctx = ArbiterContext(
        original="x" * 100,
        ambiguous=True,
        advisor=adv,
        advisor_budget=adv.request_budget("req-C"),
    )
    arb = TransformArbiter()
    decision = arb.choose(Evaluation(100, _close_records()), ctx)
    assert len(adv.calls) == 1 and decision.jev_weight == 0.40
    jev = arb.telemetry.snapshot()["jev"]
    assert jev["eligible"] == 1 and jev["calls"] == 1 and jev["applied"] == 1
    consumer = policy.consumer_stats("transform_arbiter").snapshot()
    assert consumer["eligible"] == 1 and consumer["advice_applied"] == 1


def test_gate_rejected_candidates_are_never_shown_to_jevk5():
    adv = CountingAdvisor()
    records = (
        *_close_records(),
        _rec(
            "delta",
            tokens=5,
            net_cost=5.0,
            allowed=False,
            reasons=("gate_reject",),
        ),
    )
    ctx = ArbiterContext(
        original="x" * 100,
        ambiguous=True,
        advisor=adv,
        advisor_budget=adv.request_budget("req-G"),
    )
    TransformArbiter().choose(Evaluation(100, records), ctx)
    assert len(adv.calls) == 1
    shown = " ".join(adv.calls[0].values())
    assert "delta" not in shown and len(adv.calls[0]) == 2


# ---------------------------------------------------------------------------
# Scenario E/F: tool catalog vs native tool search
# ---------------------------------------------------------------------------


def _catalog_request(
    tools, *, text: str, advisor=None, session: str = "cat", request_id: str = "req-cat"
) -> CatalogRequest:
    return CatalogRequest(
        tools=tools,
        messages=[{"role": "user", "content": text}],
        session_key=session,
        query=text,
        advisor=advisor,
        request_id=request_id,
    )


def test_scenario_e_native_tool_search_bypasses_the_catalog_and_says_so():
    cat = ProgressiveToolCatalog()
    adv = CountingAdvisor()
    tools = anthropic_tools() + [
        {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}
    ]
    out = cat.apply(_catalog_request(tools, text="fix the sentry issue", advisor=adv))
    assert out.reason == "native_search" and out.tools == tools and adv.calls == []
    status = cat.status()
    assert status["native_search"] == 1 and status["bypassed_native_tool_search"] == 1
    assert status["jev_calls"] == 0
    assert status["jevk5"]["skipped"]["native_tool_search"] == 1


def test_scenario_f_ambiguous_cut_consults_a_bounded_shortlist():
    cat = ProgressiveToolCatalog()
    adv = CountingAdvisor(pick=None)
    out = cat.apply(_catalog_request(anthropic_tools(), text="", advisor=adv))
    assert len(adv.calls) == 1 and len(adv.calls[0]) <= CATALOG_FINALISTS
    assert out.jev_weight == 0.40
    status = cat.status()
    assert status["jev_eligible"] == 1 and status["jev_calls"] == 1
    assert status["jev_applied"] == 1 and status["bypassed_native_tool_search"] == 0


def test_catalog_cut_uncertainty_skips_a_decisive_cut():
    decisive = [("a", 0.90), ("b", 0.80), ("c", 0.20), ("d", 0.10)]
    assert not cut_uncertainty(decisive, 2, has_task_terms=True).ambiguous
    crowded = [("a", 0.50), ("b", 0.49), ("c", 0.48), ("d", 0.47)]
    assert cut_uncertainty(crowded, 2, has_task_terms=True).ambiguous
    assert cut_uncertainty(decisive, 9, has_task_terms=True).blocked == "no_cut"


# ---------------------------------------------------------------------------
# Scenario G: ambiguous task-state correction
# ---------------------------------------------------------------------------


def _atoms():
    return [
        SimpleNamespace(id="atom-1", text="keep the retry limit at three attempts"),
        SimpleNamespace(id="atom-2", text="keep the retry budget at three requests"),
    ]


def test_scenario_g_ambiguous_correction_consults_and_cannot_invent_an_atom():
    with FakeLlamaServer(favor=["retry budget"]) as fake:
        advisor = advisor_for(fake.url)
        chooser = _chooser_factory(SimpleNamespace(advisor=advisor))(SimpleNamespace())
        assert chooser is not None
        assert chooser("no, use five requests", _atoms()) == "atom-2"
        assert advisor.stats.by_family == {"task_state_target": 1}
        stats = policy.consumer_stats("task_state").snapshot()
        assert stats["eligible"] == 1 and stats["advice_applied"] == 1
    # A single candidate is not ambiguous, and an invented id is discarded.
    advisor = SimpleNamespace(
        readiness="ready",
        request_budget=lambda key=None: RequestBudget(4),
        choose=lambda *a, **k: SimpleNamespace(
            value="atom-999", probabilities={}, confidence=0.9, weight=0.25, latency_ms=1.0
        ),
    )
    chooser = _chooser_factory(SimpleNamespace(advisor=advisor))(SimpleNamespace())
    assert chooser("no, five", _atoms()[:1]) is None
    assert chooser("no, five", _atoms()) is None
    assert policy.consumer_stats("task_state").snapshot()["skipped"]["unknown_atom"] == 1


def test_task_state_chooser_is_reachable_from_the_projector():
    """The ambiguous-correction path really runs the chooser (unified-21 #5)."""
    from headroom.agent.identity import WorkspaceIdentity
    from headroom.agent.ledger_db import AgentLedgerDB
    from headroom.agent.normalize import normalize_messages
    from headroom.agent.store import MemoryAgentStore
    from headroom.agent.task_state import (
        HEAD_KEY,
        TASK_STATE_PROJECTOR,
        TaskState,
        TaskStateProjector,
    )

    ident = WorkspaceIdentity("ws_jev", "ss_jev", True, "me", root=None)
    seen: list[list[str]] = []

    def chooser_factory(event):  # noqa: ANN001
        def choose(correction: str, options: list) -> str | None:
            seen.append([o.id for o in options])
            return options[-1].id  # a real option: the projector must honour it

        return choose

    db = AgentLedgerDB(None)
    store = MemoryAgentStore()
    projector = TaskStateProjector(db, chooser_factory=chooser_factory)
    messages = [
        {
            "role": "user",
            "content": "Build the cache layer. We will use sqlite for the cache store. "
            "We will use sqlite for the queue.",
        },
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "No, use postgres instead of sqlite."},
    ]
    store.commit(normalize_messages("anthropic", messages[:1], ident).envelopes, [projector])
    store.commit(normalize_messages("anthropic", messages, ident).envelopes, [projector])
    state = TaskState.from_dict(
        store.get_projection(TASK_STATE_PROJECTOR, "ws_jev", "ss_jev", HEAD_KEY)[1]
    )
    db.close()
    assert seen and len(seen[0]) > 1, "the ambiguous correction never reached the chooser"
    chosen = seen[0][-1]
    assert any(a.supersedes_id == chosen for a in state.atoms), "the chooser was ignored"


# ---------------------------------------------------------------------------
# Scenario H/I: fail-open
# ---------------------------------------------------------------------------


def test_scenario_h_advisor_failure_keeps_the_deterministic_path(monkeypatch):
    advisor = DecisionAdvisor(
        IntelligenceSettings(), lambda: ("llama-server", "http://127.0.0.1:9")
    )
    body = claude_body("src/pkg/alpha.py: ok\nsrc/pkg/beta.py: ok")
    decision = route_claude(body, advisor, monkeypatch, request_id="req-H")
    assert decision is not None and decision.score_source == "deterministic"
    assert decision.score == 1 and body["output_config"]["effort"] == "high"
    assert advisor.stats.failures == 1


def test_advisor_absent_or_unavailable_is_skipped_without_a_call(monkeypatch):
    advisor = DecisionAdvisor(IntelligenceSettings(), lambda: None)
    assert advisor.readiness == "unavailable"
    body = claude_body("src/pkg/alpha.py: ok\nsrc/pkg/beta.py: ok")
    assert route_claude(body, advisor, monkeypatch, request_id="req-U") is not None
    assert advisor.stats.calls == 0 and advisor.stats.unavailable == 0
    assert policy.consumer_stats("complexity_router").snapshot()["not_ready"] == 1


def test_scenario_i_timeout_applies_no_stale_advice(monkeypatch):
    with FakeLlamaServer(delay_s=0.4) as fake:
        advisor = advisor_for(fake.url, decision_timeout_ms=150)
        body = claude_body("src/pkg/alpha.py: ok\nsrc/pkg/beta.py: ok")
        started = time.monotonic()
        decision = route_claude(body, advisor, monkeypatch, request_id="req-I")
        assert time.monotonic() - started < 2.0
        assert decision is not None and decision.score_source == "deterministic"
        assert advisor.stats.timeouts == 1 and advisor.stats.successes == 0
        stats = policy.consumer_stats("complexity_router").snapshot()
        assert stats["calls_requested"] == 1 and stats["advice_received"] == 0
        assert stats["deterministic_fallback"] == 1


def test_low_confidence_advice_is_ignored_and_counted(monkeypatch):
    with FakeLlamaServer() as fake:
        advisor = advisor_for(fake.url, min_confidence=0.999)
        body = claude_body("src/pkg/alpha.py: ok\nsrc/pkg/beta.py: ok")
        decision = route_claude(body, advisor, monkeypatch, request_id="req-L")
        assert decision is not None and decision.score_source == "deterministic"
        assert advisor.stats.low_confidence == 1
        stats = policy.consumer_stats("complexity_router").snapshot()
        assert stats["advice_ignored_low_confidence"] == 1


def test_open_breaker_is_visible_and_skips_without_calling():
    advisor = DecisionAdvisor(
        IntelligenceSettings(), lambda: ("llama-server", "http://127.0.0.1:9")
    )
    for i in range(3):
        advisor.noul("effort_complexity", f"s{i}", "q?", budget=RequestBudget(4))
    assert advisor.readiness == "breaker_open" and advisor.status()["breaker"] == "open"
    stats = policy.consumer_stats("complexity_router")
    assert not policy.should_consult(
        policy.FAMILY_COMPLEXITY,
        policy.Uncertainty(score=1.0, reasons=("x",)),
        advisor=advisor,
        budget=advisor.request_budget("r"),
        stats=stats,
    )
    assert stats.snapshot()["skipped"]["breaker_open"] == 1


def test_cached_answer_avoids_a_second_call(monkeypatch):
    with FakeLlamaServer() as fake:
        advisor = advisor_for(fake.url)
        for request_id in ("req-c1", "req-c2"):
            body = claude_body("src/pkg/alpha.py: ok\nsrc/pkg/beta.py: ok")
            route_claude(body, advisor, monkeypatch, request_id=request_id, session="cache")
        assert fake.cfg.completions == 1 and advisor.stats.cache_hits == 1


# ---------------------------------------------------------------------------
# Scenario J: the request-wide allowance
# ---------------------------------------------------------------------------


def test_one_request_key_is_one_shared_allowance():
    advisor = advisor_for("http://127.0.0.1:9")
    shared = advisor.request_budget("req-J")
    assert advisor.request_budget("req-J") is shared
    assert advisor.request_budget("req-other") is not shared
    # No key = a private allowance (an off-request decision such as the
    # task-state projector running after the response).
    assert advisor.request_budget() is not advisor.request_budget()


def test_scenario_j_budget_exhausts_and_the_rest_fall_back():
    with FakeLlamaServer() as fake:
        advisor = advisor_for(fake.url)
        budget = advisor.request_budget("req-J2")
        stats = policy.consumer_stats("complexity_router")
        results = [
            policy.ask_score(
                advisor,
                policy.FAMILY_COMPLEXITY,
                {"signals": {"n": i}},
                "How hard is the next step (0..4)?",
                ["0", "1", "2", "3", "4"],
                budget=budget,
                stats=stats,
            )
            for i in range(6)
        ]
        assert advisor.stats.calls == 4  # calls_per_request
        assert sum(1 for r in results if r is not None) == 4
        assert advisor.stats.budget_exhausted == 2
        snapshot = stats.snapshot()
        assert snapshot["budget_exhausted"] == 2 and snapshot["deterministic_fallback"] == 2


def test_every_consumer_of_one_request_draws_from_the_same_budget(monkeypatch):
    """Complexity + arbiter + catalog together respect calls_per_request."""
    with FakeLlamaServer() as fake:
        advisor = advisor_for(fake.url, calls_per_request=2)
        shared = advisor.request_budget("req-M")
        body = claude_body("src/pkg/alpha.py: ok\nsrc/pkg/beta.py: ok")
        route_claude(body, advisor, monkeypatch, request_id="req-M")
        ctx = ArbiterContext(
            original="x" * 100, ambiguous=True, advisor=advisor, advisor_budget=shared
        )
        TransformArbiter().choose(Evaluation(100, _close_records()), ctx)
        cat = ProgressiveToolCatalog()
        cat.apply(
            _catalog_request(anthropic_tools(), text="", advisor=advisor, request_id="req-M")
        )
        # Two consumers used the whole allowance; the third was refused by the
        # shared gate before a question was even built.
        assert advisor.stats.calls == 2 and shared.remaining == 0
        assert advisor.stats.by_family == {"effort_complexity": 1, "transform_arbiter": 1}
        assert policy.consumer_stats("tool_catalog").snapshot()["budget_exhausted"] == 1


def test_concurrent_requests_keep_independent_allowances():
    with FakeLlamaServer() as fake:
        advisor = advisor_for(fake.url)
        stats = policy.consumer_stats("complexity_router")
        errors: list[BaseException] = []

        def run(i: int) -> None:
            try:
                budget = advisor.request_budget(f"req-{i}")
                for j in range(4):
                    policy.ask_score(
                        advisor,
                        policy.FAMILY_COMPLEXITY,
                        {"signals": {"r": i, "j": j}},
                        "How hard is the next step (0..4)?",
                        ["0", "1", "2", "3", "4"],
                        budget=budget,
                        stats=stats,
                    )
                assert budget.remaining == 0
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert not errors
        assert advisor.stats.calls == 16 and advisor.stats.budget_exhausted == 0


# ---------------------------------------------------------------------------
# Live proxy: the one advisor instance and the /health counters
# ---------------------------------------------------------------------------


def _proxy_config() -> ProxyConfig:
    env = {
        "HEADROOM_FEATURES": (
            "agent_runtime,agent_task_state,agent_complexity_routing,agent_tool_catalog"
        )
    }
    return ProxyConfig(
        port=PORT,
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        memory_enabled=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        stateless=True,
        rollout=resolve_rollout(env),
    )


@pytest.fixture
def _proxy_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "ws"))
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    clear_turn_hooks()
    registry.clear_registry()
    yield
    clear_turn_hooks()
    registry.clear_registry()
    paths.set_process_stateless(False)


def test_live_proxy_owns_exactly_one_advisor(_proxy_env, monkeypatch):
    monkeypatch.setenv("HEADROOM_JEVK5", "off")
    first = create_app(_proxy_config())
    second = create_app(_proxy_config())
    from headroom.agent.hook import _active_runtime

    assert second.state.agent_runtime is not first.state.agent_runtime
    assert _active_runtime() is second.state.agent_runtime
    # Every live consumer resolves to the advisor the proxy's manager owns.
    assert second.state.agent_runtime.advisor is second.state.agent_intelligence.advisor
    from headroom.agent.tool_admission.transform import _advisor as transform_advisor

    assert transform_advisor() is second.state.agent_intelligence.advisor
    snapshot, advisor, _key = complexity._snapshot_and_advisor(None)
    assert advisor is second.state.agent_intelligence.advisor


def test_live_health_exposes_real_advisor_counters(_proxy_env, monkeypatch):
    with FakeLlamaServer() as fake:
        monkeypatch.setenv("HEADROOM_JEVK5", "auto")
        monkeypatch.setenv("HEADROOM_JEVK5_LLAMA_URL", fake.url)
        app = create_app(_proxy_config())
        with TestClient(app, base_url="http://127.0.0.1", client=LOOPBACK) as client:
            manager = app.state.agent_intelligence
            assert app.state.agent_runtime.advisor is manager.advisor
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and not manager.ready:
                time.sleep(0.05)
            assert manager.ready, manager.health()
            advice = manager.advisor.score(
                "effort_complexity",
                {"signals": {"path_count": 2}},
                "How hard is the next step (0..4)?",
                ["0", "1", "2", "3", "4"],
                budget=manager.advisor.request_budget("live-1"),
            )
            assert advice is not None
            block = client.get("/health").json()["agent_runtime"]["intelligence"]
    assert block["advisor_state"] == "ready" and block["ready"] is True
    assert block["endpoint_type"] == "llama-server" and block["breaker"] == "closed"
    stats = block["advisor"]["stats"]
    assert stats["calls"] >= 1 and stats["successes"] >= 1
    assert stats["by_family"]["effort_complexity"] >= 1
    for key in (
        "cache_hits",
        "failures",
        "timeouts",
        "low_confidence",
        "budget_exhausted",
        "breaker_rejections",
        "unavailable",
        "non_english",
        "context_overflow",
        "last_failure",
        "last_latency_ms",
    ):
        assert key in stats, key
    limits = block["advisor"]["limits"]
    assert limits["decision_timeout_ms"] == 1200 and limits["calls_per_request"] == 4
    assert limits["min_confidence"] == 0.60 and limits["weight"] == 0.25
    # Per-consumer attribution is present for every live consumer.
    assert set(FAMILIES) <= set(block["consumers"])
    for name in FAMILIES:
        assert "eligible" in block["consumers"][name]
    # Counters only: no prompt, task text or path ever reaches /health.
    serialized = json.dumps(block)
    for leak in ("Refactor the parser", "src/pkg/alpha.py", "signals", "prompt"):
        assert leak not in serialized


def test_health_reports_a_disabled_advisor_without_counters_lying(_proxy_env, monkeypatch):
    monkeypatch.setenv("HEADROOM_JEVK5", "off")
    app = create_app(_proxy_config())
    with TestClient(app, base_url="http://127.0.0.1", client=LOOPBACK) as client:
        block = client.get("/health").json()["agent_runtime"]["intelligence"]
    assert block["advisor_state"] == "disabled" and block["ready"] is False
    assert block["readiness"] == "disabled"
    assert block["advisor"]["stats"]["calls"] == 0
    assert app.state.agent_intelligence.advisor.readiness == "disabled"


def test_health_exposes_the_subsystem_counters(_proxy_env, monkeypatch):
    monkeypatch.setenv("HEADROOM_JEVK5", "off")
    app = create_app(_proxy_config())
    with TestClient(app, base_url="http://127.0.0.1", client=LOOPBACK) as client:
        block = client.get("/health").json()["agent_runtime"]
    assert "jevk5" in block["complexity_routing"]
    assert "bypassed_native_tool_search" in block["tool_catalog"]


def test_an_upstream_claude_request_still_succeeds_when_the_advisor_is_dead(
    _proxy_env, monkeypatch
):
    """Scenario H end to end: a dead advisor endpoint never fails a request."""
    monkeypatch.setenv("HEADROOM_JEVK5", "auto")
    monkeypatch.setenv("HEADROOM_JEVK5_LLAMA_URL", "http://127.0.0.1:9")
    app = create_app(_proxy_config())
    with TestClient(app, base_url="http://127.0.0.1", client=LOOPBACK) as client:

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            return httpx.Response(
                200,
                json={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": MODEL,
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        client.app.state.proxy._retry_request = _fake_retry  # type: ignore[assignment]
        response = client.post(
            "/v1/messages",
            json=claude_body("src/pkg/alpha.py: ok\nsrc/pkg/beta.py: ok"),
            headers={"x-api-key": "k", "anthropic-version": "2023-06-01"},
        )
        assert response.status_code == 200
        assert app.state.agent_intelligence.advisor.readiness != "ready"
