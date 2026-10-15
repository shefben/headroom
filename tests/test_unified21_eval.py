"""Tests for the unified-21 baseline evaluation harness (benchmarks/unified21_eval.py)."""

from __future__ import annotations

import json

import pytest

from benchmarks import unified21_eval as ev

REQUIRED_CATEGORIES = {
    "file_read",
    "search",
    "logs",
    "json",
    "diff",
    "reread",
    "tool_error",
    "conflict",
    "cache",
    "streaming",
}


def test_fixtures_are_deterministic_and_cover_required_categories():
    a = ev.build_fixtures()
    b = ev.build_fixtures()
    assert [json.dumps(ev.asdict(x), sort_keys=True) for x in a] == [
        json.dumps(ev.asdict(x), sort_keys=True) for x in b
    ]
    assert REQUIRED_CATEGORIES <= {f.category for f in a}
    assert len({f.name for f in a}) == len(a)


def test_every_must_keep_fact_is_present_in_raw_fixture():
    for fx in ev.build_fixtures():
        assert fx.must_keep, fx.name
        if fx.messages:
            raw = ev._texts_from_anthropic(fx.messages)
            for fact in fx.must_keep:
                assert fact in raw, (fx.name, fact)
        if fx.responses_body is not None:
            raw = ev._texts_from_responses(fx.responses_body)
            for fact in fx.must_keep:
                assert fact in raw, (fx.name, fact)


def test_score_facts_counts_missing_and_recall():
    res = ev.PathResult("unit")
    ev._score_facts(res, ["alpha", "beta"], "alpha only")
    assert res.must_keep_total == 2
    assert res.must_keep_forwarded == 1
    assert res.must_keep_recall == 0.5
    assert res.missing_facts == ["beta"]
    assert res.retrieval_tool_turns == 0


def test_structure_checks_detect_broken_responses_body():
    body = ev.build_fixtures()[-1].responses_body
    assert body is not None
    assert ev._responses_structure_ok(body, json.loads(json.dumps(body)))
    broken = json.loads(json.dumps(body))
    broken["stream"] = False
    assert not ev._responses_structure_ok(body, broken)


def test_run_eval_end_to_end_on_small_subset(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_OFFLINE", "1")
    monkeypatch.setenv("HEADROOM_STATELESS", "1")
    subset = [f for f in ev.build_fixtures() if f.name in {"json_records", "native_streamed_turn"}]
    report = ev.run_eval("unit", subset)
    assert report["schema_version"] == ev.SCHEMA_VERSION
    names = [f["name"] for f in report["fixtures"]]
    assert names == ["json_records", "native_streamed_turn"]
    if not report["summary"]:
        pytest.skip(f"no Headroom path initialised: {report['known_failures']}")
    for summary in report["summary"].values():
        assert summary["raw_tokens"] > 0
        assert summary["forwarded_tokens"] > 0
        assert 0.0 <= summary["must_keep_recall"] <= 1.0
    # The native streamed turn is only meaningful on the Rust live-zone path.
    streamed = report["fixtures"][1]["paths"]
    assert ev.PythonProxyPath.name not in streamed
    out = tmp_path / "r.json"
    assert ev.main(["--label", "unit", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["label"] == "unit"
