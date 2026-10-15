"""DecisionAdvisor + transport against a fake llama-server (no network)."""

from __future__ import annotations

import threading
import time

import pytest

from headroom.intelligence.advisor import DecisionAdvisor, RequestBudget
from headroom.intelligence.conformance import run_llama_conformance
from headroom.intelligence.settings import IntelligenceSettings, ServicePolicy
from headroom.intelligence.transport import (
    ContextOverflow,
    Deadline,
    DeadlineExceeded,
    is_loopback_url,
    make_deadline_adapter,
    probe_llama_server,
)

from .fake_llama import FakeLlamaServer

PARCEL = ["delivered", "misdelivered", "unknown"]


def _advisor(url, **overrides):
    settings = IntelligenceSettings(**overrides)
    return DecisionAdvisor(settings, lambda: ("llama-server", url))


def test_settings_defaults_and_bounds():
    s = IntelligenceSettings.from_env({})
    assert s.policy is ServicePolicy.AUTO and s.autostart
    assert (s.temperature, s.knockout_temperature) == (1.22, 0.93)
    assert s.ctx_size == 8192 and s.decision_timeout_ms == 1200
    assert s.calls_per_request == 4 and s.cache_ttl_s == 30.0
    assert s.min_confidence == 0.60 and s.effective_weight == 0.25
    assert s.breaker_failures == 3 and s.breaker_open_s == 60.0
    assert s.model_sha256 == "aea433883bc7ed399f2fbd539e53d2eac7caf71a946fe6650995a413979d4a30"
    capped = IntelligenceSettings.from_env({"HEADROOM_JEVK5_WEIGHT": "0.9",
                                            "HEADROOM_JEVK5_WEIGHT_CAP": "0.9"})
    assert capped.effective_weight == 0.40
    assert IntelligenceSettings.from_env({"HEADROOM_JEVK5": "off"}).policy is ServicePolicy.OFF
    assert IntelligenceSettings.from_env({"HEADROOM_JEVK5": "1"}).policy is ServicePolicy.ON
    # A custom model never inherits the default checksum.
    custom = IntelligenceSettings.from_env({"HEADROOM_JEVK5_MODEL": "/m/other.gguf"})
    assert custom.model_sha256 is None and not custom.model_operator_managed
    ctx = IntelligenceSettings.from_env({"HEADROOM_JEVK5_CTX_SIZE": "999999"})
    assert ctx.ctx_size == 16384


def test_loopback_detection():
    assert is_loopback_url("http://127.0.0.1:8091")
    assert is_loopback_url("http://localhost:1")
    assert not is_loopback_url("http://10.0.0.5:8091")


def test_probe_requires_all_endpoints():
    with FakeLlamaServer() as fake:
        probe = probe_llama_server(fake.url)
        assert probe.ok and probe.n_ctx == 8192
    with FakeLlamaServer(health_status=503) as fake:
        probe = probe_llama_server(fake.url)
        assert not probe.ok and "Loading model" in probe.detail


def test_choose_and_noul_through_jevk5_adapter():
    with FakeLlamaServer() as fake:
        advisor = _advisor(fake.url)
        budget = advisor.request_budget()
        parcel = advisor.choose("parcel", "Delivered to No. 17; lives at No. 71.",
                                "What happened?", PARCEL, budget=budget)
        assert parcel is not None and parcel.value == "misdelivered"
        assert parcel.confidence >= 0.6 and parcel.weight == 0.25
        refund = advisor.noul("refund", "No receipt.", "Is a refund permitted?", budget=budget)
        assert refund is not None and refund.value < 0.5
        hints = parcel.to_hints(("parcel",))
        assert hints.source == "jevk5" and hints.choice == "misdelivered"


def test_cache_budget_and_off_policy():
    with FakeLlamaServer() as fake:
        advisor = _advisor(fake.url)
        budget = RequestBudget(4)
        for _ in range(3):
            assert advisor.choose("f", "s", "q?", PARCEL, budget=budget) is not None
        assert fake.cfg.completions == 1  # cached for the TTL
        assert advisor.stats.cache_hits == 2
        tight = RequestBudget(1)
        assert advisor.choose("f", "s1", "q?", PARCEL, budget=tight) is not None
        assert advisor.choose("f", "s2", "q?", PARCEL, budget=tight) is None
        assert advisor.stats.budget_exhausted == 1
        off = _advisor(fake.url, policy=ServicePolicy.OFF)
        before = fake.cfg.completions
        assert off.choose("f", "s9", "q?", PARCEL, budget=RequestBudget(4)) is None
        assert fake.cfg.completions == before


def test_cache_expires_after_ttl():
    now = [100.0]
    with FakeLlamaServer() as fake:
        advisor = DecisionAdvisor(IntelligenceSettings(), lambda: ("llama-server", fake.url),
                                  clock=lambda: now[0])
        advisor.choose("f", "s", "q?", PARCEL, budget=RequestBudget(4))
        now[0] += 31
        advisor.choose("f", "s", "q?", PARCEL, budget=RequestBudget(4))
        assert fake.cfg.completions == 2


def test_low_confidence_is_ignored():
    with FakeLlamaServer() as fake:
        advisor = _advisor(fake.url, min_confidence=0.999)
        assert advisor.choose("f", "s", "q?", PARCEL, budget=RequestBudget(4)) is None
        assert advisor.stats.low_confidence == 1


def test_missing_letter_retry_then_success():
    with FakeLlamaServer(missing_below=100) as fake:
        advisor = _advisor(fake.url)
        advice = advisor.choose("f", "s", "q?", PARCEL, budget=RequestBudget(4))
        assert advice is not None and advice.value == "misdelivered"
        assert fake.cfg.n_probs_seen == [40, 128]
        assert advisor.stats.missing_letter_retries == 1


def test_missing_letter_still_absent_marks_unavailable():
    with FakeLlamaServer(missing_below=1000) as fake:
        advisor = _advisor(fake.url)
        assert advisor.choose("f", "s", "q?", PARCEL, budget=RequestBudget(4)) is None
        assert advisor.stats.missing_letter_failures == 1
        assert advisor.status()["missing_letter_unhealthy"] is True


def test_whole_decision_deadline_covers_every_pass():
    # 40 options -> knockout: 3 group passes + 1 final, 0.3 s each > 0.5 s budget.
    with FakeLlamaServer(delay_s=0.3) as fake:
        advisor = _advisor(fake.url, decision_timeout_ms=500)
        options = [f"option {i}" for i in range(40)]
        started = time.monotonic()
        assert advisor.choose("f", "s", "q?", options, budget=RequestBudget(4)) is None
        assert time.monotonic() - started < 1.0
        assert advisor.stats.timeouts == 1


def test_deadline_object_and_context_overflow():
    deadline = Deadline(0.0)
    with pytest.raises(DeadlineExceeded):
        deadline.budget()
    with FakeLlamaServer(tokens_per_char=1.0) as fake:
        adapter = make_deadline_adapter(
            fake.url, deadline=Deadline(5), temperature=1.22, knockout_temperature=0.93,
            top_k=40, max_prompt_tokens=100,
        )
        with pytest.raises(ContextOverflow):
            adapter.decide("x" * 500, {"type": "noul", "instructions": "q?"})


def test_circuit_breaker_opens_after_three_failures_and_recovers():
    now = [0.0]
    calls = []

    def endpoint():
        calls.append(1)
        return ("llama-server", "http://127.0.0.1:9")  # nothing listens: connection refused

    advisor = DecisionAdvisor(IntelligenceSettings(), endpoint, clock=lambda: now[0])
    for i in range(3):
        assert advisor.noul("f", f"s{i}", "q?", budget=RequestBudget(4)) is None
    assert advisor.status()["breaker"] == "open"
    failures = advisor.stats.failures
    assert advisor.noul("f", "s9", "q?", budget=RequestBudget(4)) is None
    assert advisor.stats.failures == failures and advisor.stats.breaker_rejections == 1
    with FakeLlamaServer() as fake:
        advisor._endpoint_provider = lambda: ("llama-server", fake.url)
        now[0] += 61  # half-open trial after the open window
        assert advisor.noul("f", "s10", "q?", budget=RequestBudget(4)) is not None
        assert advisor.status()["breaker"] == "closed"


def test_concurrent_decisions_do_not_bleed_adapter_state():
    """Per-decision adapters: results and metrics stay with their own request."""
    with FakeLlamaServer(delay_s=0.02) as fake:
        advisor = _advisor(fake.url, decision_timeout_ms=5000)
        made = []
        lock = threading.Lock()
        from headroom.intelligence import transport

        def factory(*args, **kwargs):
            adapter = transport.make_deadline_adapter(*args, **kwargs)
            with lock:
                made.append(adapter)
            return adapter

        advisor._adapter_factory = factory
        results: dict[int, object] = {}

        def run(i: int) -> None:
            if i % 2:
                question = ("parcel", f"Delivered to No. 17 ({i})", "What happened?", PARCEL)
                results[i] = advisor.choose(*question, budget=RequestBudget(4))
            else:
                results[i] = advisor.noul("refund", f"No receipt ({i})", "Refund?",
                                          budget=RequestBudget(4))

        threads = [threading.Thread(target=run, args=(i,)) for i in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        assert len({id(a) for a in made}) == 12  # one adapter per decision
        for i, advice in results.items():
            assert advice is not None
            assert advice.kind == ("choice" if i % 2 else "noul")
            if i % 2:
                assert advice.value == "misdelivered"
            else:
                assert advice.value < 0.5
        assert all(a.missing == 0 for a in made)
        assert all(a.last_pass_tokens > 0 for a in made)


@pytest.mark.asyncio
async def test_async_wrapper_never_blocks_past_deadline():
    with FakeLlamaServer(delay_s=1.0) as fake:
        advisor = _advisor(fake.url, decision_timeout_ms=200)
        started = time.monotonic()
        result = await advisor.adecide(
            "f", "s", {"type": "noul", "instructions": "q?"}, RequestBudget(4)
        )
        assert result is None and time.monotonic() - started < 0.6
        advisor.shutdown()


def test_conformance_against_fake():
    with FakeLlamaServer() as fake:
        report = run_llama_conformance(fake.url, IntelligenceSettings())
        assert report.passed, report.to_dict()
        assert report.details["parcel_choice"] == "misdelivered"
    with FakeLlamaServer(n_ctx=2048) as fake:
        report = run_llama_conformance(fake.url, IntelligenceSettings())
        assert not report.passed and report.checks["context"] is False
    with FakeLlamaServer(delay_s=0.6) as fake:
        report = run_llama_conformance(fake.url, IntelligenceSettings(decision_timeout_ms=500))
        assert report.checks["latency_within_deadline"] is False
