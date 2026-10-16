"""Unified-21 Phase 5A gate: global budget (#8) and multi-candidate Transform
Arbiter (#6) through the dispatcher, the proxy transform and the real
``headroom wrap`` request path."""

from __future__ import annotations

import itertools
import json
import random
import re
from types import SimpleNamespace

import pytest

from headroom.agent.arbiter import (
    KINDS,
    WEIGHTS,
    ArbiterContext,
    Candidate,
    TransformArbiter,
    cache_multiplier,
)
from headroom.agent.budget import (
    BudgetInputs,
    BudgetUnit,
    allocate,
    compute_budget,
    model_supports_1m,
    request_budget_inputs,
)
from headroom.agent.tool_admission.delta import ResourceVersionStore
from headroom.agent.tool_admission.dispatcher import ToolAdmissionDispatcher
from headroom.agent.tool_admission.transform import ToolAdmissionTransform
from headroom.cache.compression_store import CompressionStore
from headroom.proxy import context_guard
from headroom.transforms.compression_policy import policy_default_payg
from headroom.transforms.safety_gate import Provenance
from tests.agent.test_tool_admission import (
    build_log,
    json_records,
    pytest_output,
    search_output,
)

MODEL = "claude-sonnet-4-5"


class Tok:
    """Deterministic tokenizer (bytes/4) so budgets are exact in tests."""

    def count_text(self, text: str) -> int:
        return max(1, len(text.encode("utf-8")) // 4)

    def count_messages(self, messages) -> int:
        return sum(self.count_text(json.dumps(m, sort_keys=True)) for m in messages)


class FakeAdvisor:
    """Advisor double: prefers the option whose label mentions ``prefer``."""

    def __init__(self, prefer: str, weight: float = 0.25, available: bool = True) -> None:
        self.prefer, self.weight, self.available = prefer, weight, available
        self.calls: list[dict] = []

    def request_budget(self):
        return SimpleNamespace(take=lambda: True)

    def choose(self, family, state, instructions, options, *, budget):
        self.calls.append({"family": family, "options": dict(options)})
        if not self.available:
            return None
        probs = {
            k: (0.9 if self.prefer in v else 0.1 / max(1, len(options) - 1))
            for k, v in options.items()
        }
        best = max(probs, key=probs.get)
        return SimpleNamespace(
            value=best, probabilities=probs, confidence=probs[best], weight=self.weight
        )


@pytest.fixture
def store():
    return CompressionStore(max_entries=500)


# ---------------------------------------------------------------------------
# 5A budget
# ---------------------------------------------------------------------------


def test_budget_effective_limit_never_trusts_a_bare_1m_claim(monkeypatch):
    monkeypatch.delenv("HEADROOM_CONTEXT_GUARD", raising=False)
    context_guard.reset_learned_limits()
    beta = "context-1m-2025-08-07"
    claimed = BudgetInputs(model="claude-haiku-4-5", model_limit=200_000, beta_header=beta)
    assert not model_supports_1m("claude-haiku-4-5")
    b = compute_budget(claimed, frozen_tokens=0)
    assert b.effective_limit == 200_000 and not b.one_m_honoured
    real = BudgetInputs(model=MODEL, model_limit=200_000, beta_header=beta)
    b1m = compute_budget(real, frozen_tokens=0)
    assert b1m.effective_limit == 1_000_000 and b1m.one_m_honoured
    # A limit learned from a real prompt-too-long error caps it.
    context_guard.note_prompt_too_long(
        MODEL, beta, "prompt is too long: 300000 tokens > 250000 maximum"
    )
    assert compute_budget(real, frozen_tokens=0).effective_limit == 250_000
    context_guard.reset_learned_limits()
    # HEADROOM_CONTEXT_GUARD=0 keeps the registry window.
    monkeypatch.setenv("HEADROOM_CONTEXT_GUARD", "0")
    off = compute_budget(real, frozen_tokens=0)
    assert off.effective_limit == 200_000 and not off.context_guard


def test_budget_subtracts_frozen_overhead_reserve_and_margin():
    body = {
        "system": "s" * 4000,
        "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
        "max_tokens": 8000,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}],
            }
        ],
    }
    inputs = request_budget_inputs(
        body,
        model=MODEL,
        model_limit=200_000,
        effective_limit=200_000,
        tokenizer=Tok(),
        keep_last_turns="4",
        cache_ttl_seconds=300,
    )
    assert inputs.cache_mode == "write" and inputs.keep_last_turns == 4
    b = compute_budget(inputs, frozen_tokens=10_000)
    assert b.reserved_generation == 8000 and b.safety_margin == 4000
    assert (
        b.available == 200_000 - 10_000 - inputs.system_tokens - inputs.tools_tokens - 8000 - 4000
    )
    assert b.keep_last_turns == 4


def _units(n: int = 6):
    out = []
    for i in range(n):
        out.append(
            BudgetUnit(
                key=f"r{i}|c{i}",
                source=("test", "search", "shell")[i % 3],
                original_tokens=1000 + 100 * i,
                min_tokens=100 + 10 * i,
                relevance=(i % 4) / 3,
                novelty=1.0,
                anomaly=1.0 if i == 2 else 0.0,
                recoverability=1.0,
            )
        )
    return out


def test_allocation_mandatory_first_diversity_and_order_independence():
    b = compute_budget(
        BudgetInputs(model=MODEL, model_limit=10_000, max_output_tokens=1000), frozen_tokens=0
    )
    units = _units()
    first = allocate(units, b, other_live_tokens=500)
    for u in units:
        assert first.allowances[u.key] >= u.min_tokens  # mandatory floor always given
    assert sum(first.allowances.values()) + 500 <= b.available
    assert first.pressure and not first.insufficient
    assert {u.source for u in units if first.allowances[u.key] > u.min_tokens} == {
        "test",
        "search",
        "shell",
    }
    rng = random.Random(7)
    for _ in range(5):
        shuffled = units[:]
        rng.shuffle(shuffled)
        assert allocate(shuffled, b, other_live_tokens=500).allowances == first.allowances


def test_allocation_ample_budget_and_insufficient_floor():
    big = compute_budget(BudgetInputs(model=MODEL, model_limit=1_000_000), frozen_tokens=0)
    ample = allocate(_units(), big)
    assert not ample.pressure and all(
        ample.allowances[u.key] == u.original_tokens for u in _units()
    )
    tiny = compute_budget(
        BudgetInputs(model=MODEL, model_limit=3000, max_output_tokens=100), frozen_tokens=1500
    )
    short = allocate(_units(), tiny)
    assert short.insufficient and all(v == 1.0 for v in short.pressure.values())


# ---------------------------------------------------------------------------
# 5A arbiter
# ---------------------------------------------------------------------------


def _prov(original: str, key: str = "", reversible: bool = False) -> Provenance:
    return Provenance(
        content_type="plain_text",
        strategy="test",
        reversible=reversible,
        retrieval_available=True,
        ccr_payloads={key: original} if key else {},
    )


def test_weights_kinds_and_original_always_present():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-12
    assert WEIGHTS == {
        "relevance": 0.28,
        "soft_recall": 0.20,
        "recovery": 0.16,
        "net_savings": 0.14,
        "cache": 0.10,
        "prior": 0.07,
        "latency": 0.05,
    }
    assert KINDS == (
        "original",
        "lossless",
        "type_specific",
        "query_aware",
        "structural",
        "delta",
        "indexed_externalization",
    )
    arb = TransformArbiter()
    ctx = ArbiterContext(original="x" * 4000, policy=policy_default_payg(), tokenizer=Tok())
    ev = arb.evaluate([], ctx)
    assert [r.kind for r in ev.records] == ["original"]
    assert arb.choose(ev, ctx).winner.kind == "original"


def test_hard_filters_run_before_scoring():
    original = pytest_output()
    arb = TransformArbiter()
    ctx = ArbiterContext(
        original=original,
        policy=policy_default_payg(),
        tokenizer=Tok(),
        task={"v": 1, "text": "", "exact_entities": []},
    )
    # Drops the failing test id (a hard invariant) and is not recoverable.
    lossy = "\n".join(
        ln for ln in original.splitlines() if "FAILED" not in ln and "Error" not in ln
    )[:500]
    bigger = original + "\n" + "padding " * 50  # costs more than the original
    ev = arb.evaluate(
        [
            Candidate("type_specific", lossy, "aggressive", "drop_failures", _prov(original)),
            Candidate("structural", bigger, "conservative", "bigger", _prov(original)),
        ],
        ctx,
    )
    recs = {r.label: r for r in ev.records}
    assert not recs["drop_failures"].allowed and recs["drop_failures"].utility == 0.0
    assert not recs["bigger"].allowed
    assert arb.choose(ev, ctx).winner.kind == "original"
    assert (
        arb.evaluate(
            [Candidate("type_specific", lossy, "aggressive", "x", _prov(original))],
            ArbiterContext(original=original, policy=None, tokenizer=Tok()),
        )
        .survivors[-1]
        .kind
        == "original"
    )


def test_billed_cost_includes_cache_and_retrieval_overhead():
    assert cache_multiplier("none", None) == 1.0
    assert cache_multiplier("write", 300) == 1.25 and cache_multiplier("write", 3600) == 2.0
    original = "\n".join(f"line {i} value={i}" for i in range(400))
    key = "a" * 24
    stub = f"line 0 value=0\n[... exact bytes via headroom_retrieve(hash={key}) ...]\n<<ccr:{key}>>"
    arb = TransformArbiter()
    ctx = ArbiterContext(original=original, policy=policy_default_payg(), tokenizer=Tok())
    rec = arb.evaluate(
        [Candidate("indexed_externalization", stub, "ccr_exact", "ccr", _prov(original, key))], ctx
    ).records[1]
    assert rec.retrieval_p == 0.5 and rec.retrieval_overhead > 0
    assert rec.net_cost == pytest.approx(rec.tokens + rec.retrieval_overhead)


def test_tie_break_prefers_less_loss_then_cost_then_original():
    from headroom.agent.arbiter import CandidateRecord, Evaluation

    arb = TransformArbiter()
    ctx = ArbiterContext(original="x", tokenizer=Tok())

    def rec(kind, label, tokens, loss=0.0, cost=None, recovery=1.0):
        return CandidateRecord(
            kind,
            label,
            "aggressive",
            tokens,
            True,
            irreversible_loss=loss,
            net_cost=float(tokens if cost is None else cost),
            recovery=recovery,
        )

    original = rec("original", "original", 1000)
    # Same utility (both fit the allowance => full savings), different loss.
    ev = Evaluation(
        1000,
        (
            original,
            rec("type_specific", "lossy", 100, loss=0.3),
            rec("structural", "less_lossy", 100, loss=0.1),
        ),
    )
    assert arb.choose(ev, ctx, allowance=200, pressure=1.0).winner.label == "less_lossy"
    # Same utility and loss: lower net cost wins.
    ev = Evaluation(
        1000,
        (
            original,
            rec("type_specific", "dear", 150, cost=150),
            rec("structural", "cheap", 120, cost=120),
        ),
    )
    assert arb.choose(ev, ctx, allowance=200, pressure=1.0).winner.label == "cheap"
    # An exact tie with the original: the original wins.
    ev = Evaluation(1000, (original, rec("lossless", "same", 1000)))
    assert arb.choose(ev, ctx).winner.kind == "original"


def test_deterministic_winner_under_candidate_permutations(store):
    d = ToolAdmissionDispatcher(versions=ResourceVersionStore())
    from headroom.agent.arbiter import get_arbiter
    from headroom.agent.tool_admission.families import build_envelope
    from tests.agent.test_tool_admission import request

    env = build_envelope(
        tool_name="Grep",
        tool_input={"pattern": "Budget"},
        content=search_output(),
        call_id="p1",
        root="/work/repo",
    )
    req = request(store)
    req.arbiter = get_arbiter()
    prepared = d.prepare(env, req)
    recs = prepared.evaluation.records
    cands = [r.candidate for r in recs if r.candidate is not None]
    arb = TransformArbiter()
    winners = set()
    for perm in itertools.permutations(cands):
        ev = arb.evaluate(list(perm), prepared.arbiter_ctx)
        winners.add(
            (
                arb.choose(ev, prepared.arbiter_ctx).winner.kind,
                arb.choose(ev, prepared.arbiter_ctx).winner.label,
            )
        )
    assert len(winners) == 1


def test_jevk5_blends_only_when_close_and_is_capped():
    original = "\n".join(f"row {i}: k={i % 7} payload={'x' * 30}" for i in range(300))
    key = "b" * 24
    a = "\n".join(original.splitlines()[:150]) + f"\n[headroom: hash={key}]"
    b = "\n".join(original.splitlines()[::2]) + f"\n[headroom: hash={key}]"
    cands = [
        Candidate("type_specific", a, "aggressive", "head", _prov(original, key)),
        Candidate("structural", b, "conservative", "alternate", _prov(original, key)),
    ]
    arb = TransformArbiter()
    base = ArbiterContext(original=original, policy=policy_default_payg(), tokenizer=Tok())
    plain = arb.choose(arb.evaluate(cands, base), base)
    # Unavailable advisor: w = 0, deterministic utility decides.
    off = FakeAdvisor("structural", available=False)
    ctx_off = ArbiterContext(
        original=original,
        policy=policy_default_payg(),
        tokenizer=Tok(),
        advisor=off,
        advisor_budget=off.request_budget(),
        ambiguous=True,
    )
    d_off = arb.choose(arb.evaluate(cands, ctx_off), ctx_off)
    assert d_off.jev_weight == 0.0 and d_off.winner.kind == plain.winner.kind
    # Available + ambiguous: the blend uses the capped advisor weight.
    loser = "structural" if plain.winner.kind != "structural" else "type_specific"
    adv = FakeAdvisor(loser, weight=0.9)
    ctx = ArbiterContext(
        original=original,
        policy=policy_default_payg(),
        tokenizer=Tok(),
        advisor=adv,
        advisor_budget=adv.request_budget(),
        ambiguous=True,
    )
    d = arb.choose(arb.evaluate(cands, ctx), ctx)
    assert d.jev_weight == 0.40  # capped
    assert len(adv.calls) == 1 and len(adv.calls[0]["options"]) <= 16
    for rec in d.records:
        if rec.allowed:
            assert 0.0 <= rec.blended <= 1.0


def test_jevk5_reorders_only_the_top_16_finalists():
    """A survivor ranked below the 16 finalists was never shown to JevK5.

    Blending scales every finalist by ``(1-w)``; the 17th+ survivors keep their
    raw utility and must still rank after every finalist, never overtake one.
    """
    original = "\n".join(f"row {i}: k={i % 7} payload={'x' * 30}" for i in range(400))
    key = "c" * 24
    lines = original.splitlines()
    cands = [
        Candidate(
            "structural",
            "\n".join(lines[: 100 + 12 * n]) + f"\n[headroom: hash={key}]",
            "conservative",
            f"head{n}",
            _prov(original, key),
        )
        for n in range(20)
    ]
    arb = TransformArbiter()
    base = ArbiterContext(original=original, policy=policy_default_payg(), tokenizer=Tok())
    plain_eval = arb.evaluate(cands, base)
    ranked = sorted(
        (r for r in plain_eval.records if r.allowed),
        key=lambda r: (-round(r.utility, 9), r.label),
    )
    assert len(ranked) > 16, "fixture needs more than 16 safe survivors"
    finalists = {r.label for r in ranked[:16]}
    # The advisor favors none of the shown options (uniform low probability).
    adv = FakeAdvisor("no-such-option", weight=0.40)
    ctx = ArbiterContext(
        original=original,
        policy=policy_default_payg(),
        tokenizer=Tok(),
        advisor=adv,
        advisor_budget=adv.request_budget(),
        ambiguous=True,
    )
    d = arb.choose(arb.evaluate(cands, ctx), ctx)
    assert d.jev_weight == 0.40
    assert len(adv.calls) == 1 and len(adv.calls[0]["options"]) == 16
    assert d.winner.label in finalists


# ---------------------------------------------------------------------------
# Dispatcher + transform (the real tool-admission path)
# ---------------------------------------------------------------------------


def _conversation(order=(0, 1, 2, 3)):
    tools = [
        ("t1", "Bash", {"command": "pytest -q"}, pytest_output()),
        ("t2", "Grep", {"pattern": "Budget"}, search_output()),
        ("t3", "mcp__db__query", {"sql": "select *"}, json_records()),
        ("t4", "Bash", {"command": "cargo build"}, build_log() + "\nexit code: 101"),
    ]
    uses = [{"type": "tool_use", "id": t[0], "name": t[1], "input": t[2]} for t in tools]
    results = [{"type": "tool_result", "tool_use_id": t[0], "content": t[3]} for t in tools]
    return [
        {
            "role": "user",
            "content": "Which tests failed, where is TokenBudgetExceeded raised, "
            "which record has E_QUOTA_4291 and why does netclient fail?",
        },
        {"role": "assistant", "content": [uses[i] for i in order]},
        {"role": "user", "content": [results[i] for i in order]},
    ]


def _transform(**kw):
    return ToolAdmissionTransform(
        dispatcher=ToolAdmissionDispatcher(versions=ResourceVersionStore()),
        arbiter_enabled=True,
        feedback_enabled=False,
        speculation_enabled=kw.pop("speculation", False),
        **kw,
    )


def _run(t, messages, model_limit=200_000, **kw):
    return t.apply(
        messages,
        Tok(),
        model=MODEL,
        model_limit=model_limit,
        compression_policy=policy_default_payg(),
        context=messages[0]["content"],
        **kw,
    )


def _by_id(messages):
    return {
        b["tool_use_id"]: (
            b["content"] if isinstance(b["content"], str) else b["content"][0]["text"]
        )
        for b in messages[2]["content"]
    }


@pytest.fixture
def fresh_store(monkeypatch):
    import headroom.cache.compression_store as cs

    monkeypatch.setattr(cs, "_compression_store", None)
    yield cs.get_compression_store()
    monkeypatch.setattr(cs, "_compression_store", None)


def test_transform_candidate_selection_preserves_invariants_and_recovers(fresh_store):
    out = _run(_transform(), _conversation())
    texts = _by_id(out.messages)
    assert out.tokens_after < out.tokens_before / 3
    assert "FAILED tests/test_parser.py::test_unicode_escape" in texts["t1"]
    assert "src/core/budget.py:212:    raise TokenBudgetExceeded(limit)" in texts["t2"]
    assert "E_QUOTA_4291" in texts["t3"]
    assert "error[E0308]: mismatched types" in texts["t4"] and "exit code: 101" in texts["t4"]
    for tid, original in (("t1", pytest_output()), ("t2", search_output())):
        key = re.search(r"hash=([a-f0-9]{12,24})", texts[tid]).group(1)
        assert fresh_store.retrieve(key).original_content == original


def test_arbitrary_tool_ordering_gives_identical_per_tool_bytes(fresh_store):
    orders = [(0, 1, 2, 3), (3, 2, 1, 0), (2, 0, 3, 1), (1, 3, 0, 2)]
    for limit in (200_000, 12_000):
        results = [
            _by_id(_run(_transform(), _conversation(order), model_limit=limit).messages)
            for order in orders
        ]
        assert all(r == results[0] for r in results), limit


def test_budget_pressure_selects_smaller_safe_forms(fresh_store):
    roomy = _run(_transform(), _conversation(), model_limit=200_000)
    tight = _run(_transform(), _conversation(), model_limit=6_000)
    assert tight.tokens_after <= roomy.tokens_after
    texts = _by_id(tight.messages)
    # Hard invariants never yield to the budget.
    assert (
        "FAILED tests/test_parser.py::test_unicode_escape" in texts["t1"] or "hash=" in texts["t1"]
    )
    from headroom.agent.arbiter import get_arbiter

    last = get_arbiter().status()["last_allocation"]
    assert last["budget"]["effective_limit"] == 6_000


def test_cache_decisions_memoized_bytes_are_stable_and_prefix_untouched(fresh_store):
    t = _transform()
    msgs = _conversation()
    first = _run(t, msgs)
    # Next request: the same results now sit in the frozen prefix plus one more turn,
    # and the budget is much tighter. Already-rendered results must not change.
    follow = msgs + [
        {"role": "assistant", "content": [{"type": "text", "text": "Looking."}]},
        {"role": "user", "content": "thanks, continue"},
    ]
    second = _run(t, follow, model_limit=6_000, frozen_message_count=0)
    assert _by_id(second.messages) == _by_id(first.messages)
    frozen = _run(t, follow, frozen_message_count=3)
    assert frozen.messages[:3] == follow[:3]


def test_cache_write_mix_is_recorded(fresh_store):
    msgs = _conversation()
    msgs[2]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    _run(_transform(), msgs)
    from headroom.agent.arbiter import get_arbiter

    assert get_arbiter().status()["cache_mix"].get("write", 0) >= 1


def test_policy_unknown_forwards_original(fresh_store):
    t = _transform()
    msgs = _conversation()
    out = t.apply(
        msgs,
        Tok(),
        model=MODEL,
        model_limit=200_000,
        compression_policy=SimpleNamespace(),
        context="",
    )
    # SimpleNamespace has no limits -> gate rejects every candidate.
    assert _by_id(out.messages)["t1"] == pytest_output()


# ---------------------------------------------------------------------------
# End-to-end: the real `headroom wrap` proxy request path
# ---------------------------------------------------------------------------


def test_wrap_path_e2e_candidate_selection_and_health(tmp_path, monkeypatch):
    import httpx
    from fastapi.testclient import TestClient

    from headroom import paths
    from headroom.agent import registry
    from headroom.proxy.server import ProxyConfig, create_app
    from headroom.proxy.turn_hooks import clear_turn_hooks
    from headroom.rollout import apply_wrap_feature_defaults, resolve_rollout

    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "ws"))
    monkeypatch.setenv("HEADROOM_JEVK5", "off")
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    import headroom.agent.tool_admission.dispatcher as disp
    import headroom.cache.compression_store as cs

    monkeypatch.setattr(cs, "_compression_store", None)
    monkeypatch.setattr(disp, "_DISPATCHER", None)
    clear_turn_hooks()
    registry.clear_registry()
    config = ProxyConfig(
        port=18981,
        optimize=True,
        disable_kompress=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        memory_enabled=False,
        image_optimize=False,
        stateless=True,
        rollout=resolve_rollout(apply_wrap_feature_defaults({})),
    )
    sent: list[dict] = []
    upstream = {
        "id": "m",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 1},
    }
    try:
        with TestClient(
            create_app(config), base_url="http://127.0.0.1", client=("127.0.0.1", 50000)
        ) as client:

            async def fake(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
                sent.append(body if isinstance(body, dict) else json.loads(body))
                return httpx.Response(200, json=upstream)

            client.app.state.proxy._retry_request = fake
            for order in [(0, 1, 2, 3), (2, 3, 0, 1)]:
                resp = client.post(
                    "/v1/messages",
                    headers={"x-api-key": "k", "anthropic-version": "2023-06-01"},
                    json={"model": MODEL, "max_tokens": 64, "messages": _conversation(order)},
                )
                assert resp.status_code == 200, resp.text
            a, b = _by_id(sent[0]["messages"]), _by_id(sent[1]["messages"])
            assert a == b  # arbitrary ordering, same winners
            assert "FAILED tests/test_parser.py::test_unicode_escape" in a["t1"]
            health = client.get("/health").json()["tool_admission"]
            arb = health["arbiter"]
            assert sum(arb["decisions"].values()) >= 4
            assert arb["realized"]["tokens_saved"] > 0 and "avoided_turns" in arb["estimated"]
            assert arb["last_allocation"]["budget"]["reserved_generation"] == 64
            assert "speculation" in health and "feedback" in health
    finally:
        clear_turn_hooks()
        registry.clear_registry()
        paths.set_process_stateless(False)
