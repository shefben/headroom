"""Live JevK5 conformance against a real llama-server + the cached GGUF (opt-in).

Run with ``HEADROOM_JEVK5_LIVE_TEST=1``. Uses the llama-server on PATH (or
``HEADROOM_JEVK5_LLAMA_SERVER``) and the already-cached, checksum-verified
model (``HEADROOM_JEVK5_MODEL`` + ``HEADROOM_JEVK5_MODEL_SHA256`` may point at
a cache outside the workspace); never builds or downloads. Starts the managed
service, checks conformance (refund -> no, parcel -> misdelivered), doctor
fields, the whole-decision deadline, the shared per-request call cap and
concurrency isolation, then stops the service it started.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

# Captured at import: the suite's autouse fixture scrubs HEADROOM_* per test.
LIVE = os.environ.get("HEADROOM_JEVK5_LIVE_TEST") == "1"
LLAMA_SERVER = os.environ.get("HEADROOM_JEVK5_LLAMA_SERVER")
WORKSPACE = os.environ.get("HEADROOM_WORKSPACE_DIR")
MODEL = os.environ.get("HEADROOM_JEVK5_MODEL")
MODEL_SHA256 = os.environ.get("HEADROOM_JEVK5_MODEL_SHA256")

pytestmark = pytest.mark.skipif(not LIVE, reason="set HEADROOM_JEVK5_LIVE_TEST=1")


def test_live_setup_conformance_concurrency_and_stop(monkeypatch):
    from headroom.intelligence.advisor import DecisionAdvisor
    from headroom.intelligence.manager import IntelligenceManager
    from headroom.intelligence.process import is_running
    from headroom.intelligence.settings import IntelligenceSettings

    if LLAMA_SERVER:
        monkeypatch.setenv("HEADROOM_JEVK5_LLAMA_SERVER", LLAMA_SERVER)
    if WORKSPACE:
        monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", WORKSPACE)
    if MODEL:
        monkeypatch.setenv("HEADROOM_JEVK5_MODEL", MODEL)
    if MODEL_SHA256:
        monkeypatch.setenv("HEADROOM_JEVK5_MODEL_SHA256", MODEL_SHA256)
    settings = IntelligenceSettings.from_env()
    manager = IntelligenceManager(settings)
    report = manager.setup(log_fn=print, allow_build=False, allow_download=False)
    try:
        print(report.to_dict())
        assert report.ok, report.to_dict()
        checks = report.conformance.checks
        assert checks["refund_is_no"] and checks["parcel_is_misdelivered"]
        assert checks["answer_letters_present"] and checks["context"]
        assert manager.health()["advisor_state"] == "ready"

        # Hot-path advisor (real 1200 ms deadline) through the ready endpoint.
        budget = manager.advisor.request_budget()
        parcel = manager.advisor.choose(
            "parcel", "Delivered to No. 17; the customer lives at No. 71.",
            "What happened to the parcel?", ["delivered", "misdelivered", "unknown"],
            budget=budget,
        )
        assert parcel is not None and parcel.value == "misdelivered"
        print("hot-path parcel latency_ms", parcel.latency_ms)
        assert parcel.latency_ms <= settings.decision_timeout_ms

        # Doctor reports every field the plan names.
        doctor = manager.doctor()
        print("doctor:", {k: doctor[k] for k in (
            "jevk5_runtime", "endpoint_type", "effective_context", "calibration",
            "missing_letter_health", "process_ownership")})
        assert doctor["jevk5_runtime"] == "0.3.3"
        assert doctor["endpoint_type"] == "managed"
        assert doctor["effective_context"] == 8192
        assert doctor["calibration"]["temperature"] == 1.22
        assert doctor["calibration"]["knockout_temperature"] == 0.93
        assert doctor["missing_letter_health"] is True
        assert doctor["process_ownership"]["owned"] is True
        assert doctor["process_ownership"]["executable"]
        assert doctor["model"]["verified"] and doctor["model"]["sha256"]
        assert doctor["conformance"]["passed"]

        # Whole-decision deadline: a long state with >16 options (knockout,
        # several passes) under the minimum 100 ms budget is discarded on
        # time, counted as a timeout, never returned late.
        endpoint = manager.ready_endpoint()
        tight = DecisionAdvisor(
            settings.with_overrides(decision_timeout_ms=100, cache_ttl_s=0.0), lambda: endpoint
        )
        options = [f"option number {i} about the parcel" for i in range(20)]
        state = "The parcel log says: " + "scanned at depot, loaded, unloaded. " * 300
        started = time.monotonic()
        late = tight.choose("deadline", state, "Which option fits?", options,
                            budget=tight.request_budget())
        elapsed_ms = (time.monotonic() - started) * 1000.0
        print("deadline decision elapsed_ms", round(elapsed_ms, 1), tight.status()["stats"])
        assert late is None and tight.stats.timeouts == 1
        assert elapsed_ms < 100 + 250  # socket budget = remaining time, plus scheduling slack

        # 4 calls per request, shared by every consumer passing the request id.
        shared = DecisionAdvisor(settings.with_overrides(cache_ttl_s=0.0), lambda: endpoint)
        for i in range(6):
            shared.noul(f"cap{i}", f"Refunds need a receipt. Case {i}: no receipt.",
                        "Is a refund permitted?", budget=shared.request_budget("req-live"))
        assert shared.stats.calls == 4 and shared.stats.budget_exhausted == 2

        # Concurrency isolation on the shared server (longer budget: requests queue).
        concurrent = DecisionAdvisor(
            settings.with_overrides(decision_timeout_ms=20000, cache_ttl_s=0.0),
            lambda: endpoint,
        )
        results: dict[int, object] = {}

        def run(i: int) -> None:
            b = concurrent.request_budget()
            if i % 2:
                results[i] = concurrent.choose(
                    "parcel", f"Delivered to No. 17; the customer lives at No. 71. Ref {i}.",
                    "What happened to the parcel?", ["delivered", "misdelivered", "unknown"],
                    budget=b,
                )
            else:
                results[i] = concurrent.noul(
                    "refund",
                    "Refunds need a receipt and a purchase within 30 days. The customer bought "
                    f"12 days ago and has no receipt. Ref {i}.",
                    "Is a refund permitted under the policy?",
                    budget=b,
                )

        threads = [threading.Thread(target=run, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(120)
        assert len(results) == 8
        for i, advice in results.items():
            assert advice is not None, (i, concurrent.status())
            if i % 2:
                assert advice.kind == "choice" and advice.value == "misdelivered"
            else:
                assert advice.kind == "noul" and advice.value < 0.5
            # Metrics come from the decision's own adapter: no cross-thread bleed.
            assert advice.input_tokens > 0 and advice.pass_tokens == advice.input_tokens
    finally:
        info = manager.service.current(probe=False)
        stopped, detail = manager.stop()
        print("stop:", stopped, detail)
    assert stopped, detail
    assert info.pid is not None and not is_running(int(info.pid))
