"""SafetyGate + enforced lossy policy + query conditioning (unified-21 #1, #2, #7, #13).

The gate is the Rust implementation behind ``headroom._core``; these tests pin
the Python facade to the shared parity fixtures and prove the gate runs in the
real ContentRouter hot path.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from headroom import _core
from headroom.cache.compression_store import get_compression_store, reset_compression_store
from headroom.proxy.auth_mode import AuthMode
from headroom.tokenizers.registry import get_tokenizer
from headroom.transforms.compression_policy import policy_for_mode
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
)
from headroom.transforms.safety_gate import (
    GateCandidate,
    Provenance,
    SafetyGate,
    policy_limits,
    task_payload,
)

FIXTURE = Path(__file__).parent / "fixtures" / "safety_gate_parity.json"


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", _fixture()["cases"], ids=lambda c: c["name"])
def test_gate_parity_with_rust_fixtures(case):
    d = SafetyGate.check(
        case["original"],
        case["candidate"],
        task=case.get("task"),
        policy=case["policy"],
        provenance=Provenance(**case["provenance"]),
        tier=case["tier"],
        model=_fixture()["model"],
    )
    assert d.allow is case["allow"], d.reasons
    for reason in case["reasons"]:
        assert reason in d.reasons
    assert d.verdict == ("ALLOW" if case["allow"] else "REJECT")


@pytest.mark.parametrize("case", _fixture()["ladder"], ids=lambda c: c["name"])
def test_ladder_parity_with_rust_fixtures(case):
    cands = [
        GateCandidate(c["tier"], c["text"], Provenance(**c["provenance"]), label=c["tier"])
        for c in case["candidates"]
    ]
    sel = SafetyGate.select(
        case["original"], cands, policy=case["policy"], model=_fixture()["model"]
    )
    assert sel.tier == case["tier"]


def test_unknown_or_invalid_limits_fail_to_original():
    assert policy_limits(None) is None
    bad = policy_limits({"max_lossy_ratio": float("nan"), "volatile_token_threshold": 5})
    original = "line of build output with words\n" * 40
    d = SafetyGate.check(original, original[:200], policy=bad)
    assert not d.allow and "policy_unknown" in d.reasons


def test_wire_drop_is_not_irreversible_loss_when_exactly_recoverable():
    reset_compression_store()
    store = get_compression_store()
    original = "".join(f"INFO step {i} ok\n" for i in range(300)) + "ERROR disk full on /var\n"
    key = store.store(original, "c")
    assert store.verify_exact(key, original)
    candidate = f"ERROR disk full on /var\n[301 lines compressed to 1. Retrieve more: hash={key}]"
    d = SafetyGate.check(
        original,
        candidate,
        policy=policy_for_mode(AuthMode.SUBSCRIPTION),
        provenance=Provenance(retrieval_available=True, ccr_payloads={key: original}),
    )
    assert d.allow, d.reasons
    assert d.wire_drop_ratio > 0.9
    assert d.irreversible_loss_ratio == 0.0
    assert d.exact_recoverable
    reset_compression_store()


def _log_with_user_entity() -> str:
    body = "".join(f"[INFO] worker {i} processed batch {i} ok\n" for i in range(400))
    return body + "[INFO] worker 377 rerouted tenant ACME-991 to shard 4\n"


def _router_returning(text: str, strategy=CompressionStrategy.LOG) -> ContentRouter:
    router = ContentRouter()

    def compress(self, content, **_kw):
        return SimpleNamespace(
            compressed=text,
            original=content,
            strategy_used=strategy,
            compression_ratio=len(text) / max(1, len(content)),
        )

    router.compress = compress.__get__(router)
    return router


def _apply(router: ContentRouter, content: str, **kw):
    tok = get_tokenizer("gpt-4o")
    # A tool message with no matching assistant call: plain shell output that
    # reaches compression (not a protected Read window).
    msgs = [{"role": "tool", "tool_call_id": "call_shell_1", "content": content}]
    return router.apply(
        msgs,
        tok,
        model_limit=200_000,
        protect_recent=0,
        protect_analysis_context=False,
        **kw,
    )


def test_router_hot_path_never_forwards_a_candidate_that_drops_a_user_named_entity():
    content = _log_with_user_entity()
    lossy = "".join(f"[INFO] worker {i} processed batch {i} ok\n" for i in range(300))
    result = _apply(
        _router_returning(lossy),
        content,
        context="what happened to tenant ACME-991?",
        compression_policy=_policy(0.6),
    )
    forwarded = result.messages[-1]["content"]
    assert "ACME-991" in forwarded
    assert any(t.startswith("safety_gate") for t in result.transforms_applied)


def _policy(max_lossy: float):
    from headroom.transforms.compression_policy import CompressionPolicy

    return CompressionPolicy(
        live_zone_only=False,
        cache_aligner_enabled=True,
        volatile_token_threshold=128,
        max_lossy_ratio=max_lossy,
        toin_read_only=False,
    )


def test_router_enforces_the_request_policy_budget():
    lines = [f"[INFO] worker {i} processed batch {i} ok\n" for i in range(200)]
    content = "".join(lines)
    candidate = "".join(lines[:100])  # ~50% irreversible drop, no marker
    loose = _apply(_router_returning(candidate), content, compression_policy=_policy(0.6))
    assert loose.messages[-1]["content"] == candidate
    strict = _apply(
        _router_returning(candidate),
        content,
        compression_policy=policy_for_mode(AuthMode.SUBSCRIPTION),
    )
    forwarded = strict.messages[-1]["content"]
    assert forwarded != candidate  # beyond Subscription 0.25: backed off
    tok = get_tokenizer("gpt-4o")
    kept = tok.count_text(forwarded) / tok.count_text(content)
    assert kept >= 0.75 or "<<ccr:" in forwarded


def test_task_payload_extracts_exact_entities_from_plain_context():
    task = task_payload(None, fallback_text="fix `parse_config` in src/app/config.py (HR-42)")
    assert "parse_config" in task["exact_entities"]
    assert "src/app/config.py" in task["exact_entities"]
    assert "HR-42" in task["exact_entities"]


def test_relevance_query_conditions_live_zone_binding():
    out_text = "".join(
        f"src/billing/invoice_{i}.py:{i}: total = compute_invoice_total(o)\n"
        f"src/auth/session_{i}.py:{i}: tok = refresh_session_token(u)\n"
        for i in range(150)
    )
    body = json.dumps(
        {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "look"}],
                },
                {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": out_text},
            ],
        }
    ).encode()

    def run(query: dict) -> str:
        out, modified, *_ = _core.compress_openai_responses_live_zone(
            body,
            "payg",
            "gpt-5.4",
            json.dumps(query),
            json.dumps({"max_lossy_ratio": 0.45, "volatile_token_threshold": 128}),
        )
        assert modified
        return json.loads(bytes(out))["input"][2]["output"]

    billing = run(
        {"v": 1, "text": "compute_invoice_total", "exact_entities": ["compute_invoice_total"]}
    )
    auth = run(
        {"v": 1, "text": "refresh_session_token", "exact_entities": ["refresh_session_token"]}
    )
    assert billing != auth
    assert billing.count("compute_invoice_total") > billing.count("refresh_session_token")
    assert auth.count("refresh_session_token") > auth.count("compute_invoice_total")


def test_compression_relevance_query_is_none_without_agent_runtime():
    from headroom.agent.hook import compression_relevance_query

    assert compression_relevance_query("anthropic", "m", [{"role": "user", "content": "x"}]) is None


def test_semantic_service_initializes_at_most_once_per_process():
    for _ in range(25):
        _core.hybrid_relevance_scores(["alpha beta", "gamma"], "alpha")
    _core.semantic_relevance_wait_ready(30.0)
    status = json.loads(_core.semantic_relevance_status())
    assert status["init_runs"] <= 1
    assert status["state"] in {"ready", "fallback_bm25", "disabled"}
