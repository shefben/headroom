"""Conformance checks a decision endpoint must pass before the advisor uses it.

For a llama-server endpoint:

1. ``/health``, ``/tokenize`` with ``parse_special=true`` (the chat-template
   marker is one token) and ``/completion`` returning finite
   ``completion_probabilities[0].top_logprobs``;
2. effective context ≥ the configured 8192 tokens;
3. a ``JevK5GGUF`` adapter with the configured calibration returns finite,
   normalized ``noul``/``choice``/``score`` answers, with the missing-letter
   retry applied;
4. sanity top choices: a no-receipt refund → *no*; a parcel delivered to the
   wrong house number → *misdelivered* (top choice, not exact confidence);
5. latency: the sanity decisions must finish inside the hot-path deadline,
   otherwise the advisor is marked too slow (e.g. a CPU fallback) and stays
   disabled for request traffic.

Setup and doctor use the longer load timeout; request traffic never runs this.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from headroom.intelligence.settings import IntelligenceSettings
from headroom.intelligence.transport import (
    Deadline,
    SystemOneClient,
    TransportError,
    make_deadline_adapter,
    probe_llama_server,
)

REFUND_STATE = (
    "Refunds need a receipt and a purchase within 30 days. The customer bought 12 days ago "
    "and has no receipt."
)
REFUND_QUESTION = {"type": "noul", "instructions": "Is a refund permitted under the policy?"}
PARCEL_STATE = "Delivered to No. 17; the customer lives at No. 71."
PARCEL_QUESTION = {
    "type": "choice",
    "instructions": "What happened to the parcel?",
    "criteria": ["delivered", "misdelivered", "unknown"],
}
SCORE_QUESTION = {
    "type": "score",
    "instructions": "How severe is the delivery problem?",
    "criteria": ["none", "minor", "serious"],
}


@dataclass
class ConformanceReport:
    endpoint_kind: str
    url: str
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(self.checks.values())

    @property
    def hot_path_ok(self) -> bool:
        return bool(self.checks.get("latency_within_deadline"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint_kind": self.endpoint_kind,
            "url": self.url,
            "passed": self.passed,
            "checks": dict(self.checks),
            "details": dict(self.details),
            "latency_ms": dict(self.latency_ms),
        }


def _finite_normalized(answer: dict, kind: str) -> bool:
    try:
        if kind == "noul":
            value = float(answer["noul"])
            return math.isfinite(value) and 0.0 <= value <= 1.0
        probs = [float(v) for v in answer["probabilities"].values()]
        return all(math.isfinite(p) and 0 <= p <= 1 for p in probs) and abs(sum(probs) - 1) < 1e-3
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def _decide_llama(
    url: str, settings: IntelligenceSettings, state: str, question: dict, timeout: float, ctx: int
) -> tuple[dict, bool, float]:
    """One decision with the missing-letter retry. Returns (answer, missing_after_retry, ms)."""
    started = time.monotonic()
    deadline = Deadline(timeout)
    adapter = make_deadline_adapter(
        url,
        deadline=deadline,
        temperature=settings.temperature,
        knockout_temperature=settings.knockout_temperature,
        top_k=settings.top_k,
        max_prompt_tokens=ctx,
    )
    answer = adapter.decide(state, dict(question))
    missing = bool(adapter.missing)
    if missing:
        retry = make_deadline_adapter(
            url,
            deadline=deadline,
            temperature=settings.temperature,
            knockout_temperature=settings.knockout_temperature,
            top_k=settings.missing_retry_n_probs,
            max_prompt_tokens=ctx,
        )
        answer = retry.decide(state, dict(question))
        missing = bool(retry.missing)
    return answer, missing, (time.monotonic() - started) * 1000.0


def run_llama_conformance(
    url: str, settings: IntelligenceSettings, *, timeout: float = 60.0
) -> ConformanceReport:
    report = ConformanceReport("llama-server", url)
    probe = probe_llama_server(url, timeout=min(timeout, 30.0))
    report.checks["endpoints"] = probe.ok
    report.details["probe"] = probe.detail
    report.details["n_ctx"] = probe.n_ctx
    if not probe.ok:
        return report
    ctx = probe.n_ctx or settings.ctx_size
    report.checks["context"] = ctx >= settings.ctx_size
    effective_ctx = min(ctx, settings.ctx_size)
    try:
        refund, refund_missing, refund_ms = _decide_llama(
            url, settings, REFUND_STATE, REFUND_QUESTION, timeout, effective_ctx
        )
        parcel, parcel_missing, parcel_ms = _decide_llama(
            url, settings, PARCEL_STATE, PARCEL_QUESTION, timeout, effective_ctx
        )
        score, score_missing, _score_ms = _decide_llama(
            url, settings, PARCEL_STATE, SCORE_QUESTION, timeout, effective_ctx
        )
        # Warm path: the first decision after a load includes cold kernels;
        # the hot-path budget is judged on a repeated decision.
        _warm, _m, warm_ms = _decide_llama(
            url, settings, REFUND_STATE, REFUND_QUESTION, timeout, effective_ctx
        )
    except (TransportError, ImportError, KeyError, ValueError) as exc:
        report.checks["decisions"] = False
        report.details["error"] = f"{type(exc).__name__}: {exc}"
        return report
    report.checks["finite_normalized"] = (
        _finite_normalized(refund, "noul")
        and _finite_normalized(parcel, "choice")
        and _finite_normalized(score, "score")
    )
    report.checks["answer_letters_present"] = not (refund_missing or parcel_missing or score_missing)
    report.checks["refund_is_no"] = float(refund.get("noul", 1.0)) < 0.5
    report.checks["parcel_is_misdelivered"] = parcel.get("choice") == "misdelivered"
    report.details["refund_noul"] = round(float(refund.get("noul", float("nan"))), 4)
    report.details["parcel_choice"] = parcel.get("choice")
    report.details["parcel_confidence"] = round(float(parcel.get("confidence", 0.0)), 4)
    report.latency_ms = {"refund": refund_ms, "parcel": parcel_ms, "warm": warm_ms}
    report.checks["latency_within_deadline"] = max(warm_ms, parcel_ms) <= settings.decision_timeout_ms
    return report


def run_systemone_conformance(
    url: str, settings: IntelligenceSettings, *, timeout: float = 30.0
) -> ConformanceReport:
    report = ConformanceReport("systemone", url)
    client = SystemOneClient(url)
    ok, detail = client.probe(timeout=min(timeout, 10.0))
    report.checks["endpoints"] = ok
    report.details["probe"] = detail
    if not ok:
        return report
    try:
        started = time.monotonic()
        refund = client.decide(REFUND_STATE, REFUND_QUESTION, deadline=Deadline(timeout))
        refund_ms = (time.monotonic() - started) * 1000.0
        started = time.monotonic()
        parcel = client.decide(PARCEL_STATE, PARCEL_QUESTION, deadline=Deadline(timeout))
        parcel_ms = (time.monotonic() - started) * 1000.0
    except TransportError as exc:
        report.checks["decisions"] = False
        report.details["error"] = str(exc)
        return report
    report.checks["finite_normalized"] = _finite_normalized(refund, "noul") and _finite_normalized(
        parcel, "choice"
    )
    report.checks["refund_is_no"] = float(refund.get("noul", 1.0)) < 0.5
    report.checks["parcel_is_misdelivered"] = parcel.get("choice") == "misdelivered"
    report.latency_ms = {"refund": refund_ms, "parcel": parcel_ms}
    report.checks["latency_within_deadline"] = max(refund_ms, parcel_ms) <= settings.decision_timeout_ms
    return report
