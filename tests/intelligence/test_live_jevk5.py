"""Live JevK5 conformance against a real llama-server + the cached GGUF (opt-in).

Run with ``HEADROOM_JEVK5_LIVE_TEST=1``. Uses the llama-server on PATH (or
``HEADROOM_JEVK5_LLAMA_SERVER``) and the already-cached, checksum-verified
model; never builds or downloads. Starts the managed service, checks
conformance (refund -> no, parcel -> misdelivered), concurrency isolation,
then stops the service.
"""

from __future__ import annotations

import os
import threading

import pytest

# Captured at import: the suite's autouse fixture scrubs HEADROOM_* per test.
LIVE = os.environ.get("HEADROOM_JEVK5_LIVE_TEST") == "1"
LLAMA_SERVER = os.environ.get("HEADROOM_JEVK5_LLAMA_SERVER")
WORKSPACE = os.environ.get("HEADROOM_WORKSPACE_DIR")

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

        # Concurrency isolation on the shared server (longer budget: requests queue).
        endpoint = manager.ready_endpoint()
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
    finally:
        info = manager.service.current(probe=False)
        stopped, detail = manager.stop()
        print("stop:", stopped, detail)
    assert stopped, detail
    assert info.pid is not None and not is_running(int(info.pid))
