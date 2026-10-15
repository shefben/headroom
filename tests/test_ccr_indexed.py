"""Indexed / selective CCR (unified-21 #3): spans, FTS5 + lexical, cursors,
reachability, TTL, proxy CCR flows and selective proactive expansion."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from headroom.cache.compression_store import CompressionStore
from headroom.ccr.context_tracker import ContextTracker, ExpansionRecommendation
from headroom.ccr.indexed import (
    IndexedCCR,
    chunk_original,
    estimate_tokens,
    reset_indexed_ccr,
    retrieve_with_args,
)
from headroom.ccr.response_handler import CCRResponseHandler
from headroom.ccr.tool_injection import (
    CCR_TOOL_NAME,
    create_ccr_tool_definition,
    parse_retrieval_args,
    parse_tool_call,
)


def _big_log(target_tokens: int = 100_000) -> str:
    lines = []
    i = 0
    while sum(len(x) for x in lines) < target_tokens * 4:
        lines.append(
            f"2026-10-15T10:{i % 60:02d}:{i % 59:02d}Z INFO request_id=req-{i} "
            f"route=/api/v1/items/{i % 997} status=200 latency_ms={i % 300}\n"
        )
        i += 1
    # Five needle events, far apart, that share a vocabulary nothing else uses.
    total = len(lines)
    for n, frac in enumerate((0.1, 0.3, 0.5, 0.7, 0.9)):
        lines.insert(
            int(total * frac) + n,
            f"2026-10-15T11:00:0{n}Z ERROR payment_gateway timeout charging card "
            f"for invoice INV-77{n} (stripe_charge_failed)\n",
        )
    return "".join(lines)


@pytest.fixture(autouse=True)
def _fresh_index():
    reset_indexed_ccr()
    yield
    reset_indexed_ccr()


@pytest.fixture
def store(monkeypatch):
    s = CompressionStore(max_entries=100)
    import headroom.cache.compression_store as cs

    monkeypatch.setattr(cs, "_compression_store", s)
    return s


@pytest.mark.parametrize("use_fts", [True, False])
def test_100k_token_entry_returns_five_relevant_spans_and_full_is_exact(store, use_fts):
    original = _big_log()
    assert estimate_tokens(original) >= 100_000
    h = store.store(original, "compressed", tool_name="Bash")
    index = IndexedCCR(use_fts=use_fts)
    if use_fts and not index.fts:
        pytest.skip("SQLite without FTS5")

    found = index.retrieve(store, h, mode="search", query="stripe_charge_failed payment timeout")
    assert found["source_hash"] == h
    assert found["index"] == ("fts5" if use_fts else "lexical")
    results = found["results"]
    assert len(results) == 5
    assert all("stripe_charge_failed" in r["text"] for r in results)
    assert found["returned_tokens"] <= 1200
    assert found["complete"] is False
    raw = original.encode("utf-8")
    for r in results:
        # Spans are exact byte ranges of the stored original.
        assert raw[r["byte_start"] : r["byte_end"]].decode("utf-8") == r["text"]
        assert r["span_id"] == f"{h}:{r['ordinal']}"
        assert {"score", "type", "content_hash"} <= set(r)

    full = index.retrieve(store, h, mode="full")
    assert full["original_content"] == original  # byte-exact


def test_semantic_chunking_by_shape():
    rows = json.dumps([{"id": i, "path": f"src/f{i}.py", "ok": True} for i in range(5)])
    spans = chunk_original("h", rows)
    assert [s.type for s in spans] == ["json_row"] * 5
    assert spans[2].key == "id=2" and spans[2].path == "src/f2.py"

    diff = (
        "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n-a\n+b\n"
        "@@ -9,2 +9,2 @@\n-c\n+d\n"
    )
    dspans = chunk_original("h", diff)
    assert sum(1 for s in dspans if s.type == "diff_hunk") >= 2
    assert all(s.path in ("", "x.py") for s in dspans)

    code = "import os\n\ndef alpha():\n    return 1\n\n\nclass Beta:\n    pass\n"
    cspans = chunk_original("h", code)
    assert [s.symbol for s in cspans if s.symbol] == ["alpha", "Beta"]

    hits = "".join(f"src/a{i}.py:{i}: needle\n" for i in range(6))
    hspans = chunk_original("h", hits)
    assert all(s.type == "search_hit" for s in hspans) and hspans[3].path == "src/a3.py"

    trace = (
        'INFO start\nTraceback (most recent call last):\n  File "a.py", line 3, in f\n'
        "    g()\nKeyError: 'k'\nINFO after\n"
    )
    assert any(s.type == "trace" for s in chunk_original("h", trace))


def test_cursor_binds_workspace_entry_generation_and_query(store):
    original = _big_log(20_000)
    h = store.store(original, "c")
    other = store.store(original + "tail\n", "c")
    index = IndexedCCR()
    first = index.retrieve(
        store, h, mode="search", query="request_id status", top_k=2, workspace="ws_a"
    )
    assert first["has_more"] and first["cursor"]
    cur = first["cursor"]
    page2 = index.retrieve(
        store, h, mode="search", query="request_id status", top_k=2, cursor=cur, workspace="ws_a"
    )
    assert "error" not in page2
    assert {r["ordinal"] for r in page2["results"]}.isdisjoint(
        {r["ordinal"] for r in first["results"]}
    )
    # Another workspace, another entry, another query, a forged cursor: refused.
    assert (
        index.retrieve(
            store, h, mode="search", query="request_id status", cursor=cur, workspace="ws_b"
        )["error"]
        == "cursor_scope_mismatch"
    )
    assert (
        index.retrieve(
            store, other, mode="search", query="request_id status", cursor=cur, workspace="ws_a"
        )["error"]
        == "cursor_scope_mismatch"
    )
    assert (
        index.retrieve(store, h, mode="search", query="other", cursor=cur, workspace="ws_a")[
            "error"
        ]
        == "cursor_scope_mismatch"
    )
    assert (
        index.retrieve(store, h, mode="search", query="x", cursor="abc.def", workspace="ws_a")[
            "error"
        ]
        == "cursor_invalid"
    )
    # A re-index (new generation) makes old cursors stale, never silently wrong.
    index.purge(h)
    assert (
        index.retrieve(
            store, h, mode="search", query="request_id status", cursor=cur, workspace="ws_a"
        )["error"]
        == "cursor_stale"
    )


def test_reachability_and_ttl_apply_to_spans(store):
    h = store.store("alpha beta gamma\n" * 50, "c", ttl=1)
    index = IndexedCCR()
    assert index.retrieve(store, h, mode="metadata", reachable={"deadbeef" * 3})["error"] == (
        "not_reachable_in_this_conversation"
    )
    meta = index.retrieve(store, h, mode="metadata", reachable={h})
    assert meta["total_spans"] >= 1 and meta["complete"] is True
    assert h in index.indexed_hashes()
    time.sleep(1.2)
    gone = index.retrieve(store, h, mode="search", query="alpha", reachable={h})
    assert gone["error"] == "entry_not_found_or_expired"
    assert h not in index.indexed_hashes()  # spans died with their parent


def test_range_mode_pages_exact_spans(store):
    original = "".join(f"paragraph {i} " * 40 + "\n\n" for i in range(30))
    h = store.store(original, "c")
    out = retrieve_with_args(store, h, {"mode": "range", "start": 0, "end": 3})
    assert [r["ordinal"] for r in out["results"]] == [0, 1, 2]
    assert out["complete"] is False
    small = store.store("one\n\ntwo\n", "c")
    whole = retrieve_with_args(store, small, {"mode": "range"})
    assert whole["complete"] is True


def test_tool_definition_and_argument_parsing():
    tool = create_ccr_tool_definition("openai_responses")
    assert tool["parameters"]["required"] == ["hash"]
    assert set(tool["parameters"]["properties"]) >= {"hash", "query", "mode", "cursor", "top_k"}
    call = {
        "type": "function_call",
        "name": CCR_TOOL_NAME,
        "call_id": "c1",
        "arguments": json.dumps({"hash": "a" * 24, "query": "timeout", "top_k": 3}),
    }
    assert parse_tool_call(call, "openai_responses") == "a" * 24
    assert parse_retrieval_args(call, "openai_responses") == {"query": "timeout", "top_k": 3}
    plain = {"name": CCR_TOOL_NAME, "input": {"hash": "b" * 24}}
    assert parse_retrieval_args(plain, "anthropic") == {}


def _anthropic_messages(h: str) -> list[dict]:
    return [
        {"role": "user", "content": "why do payments fail?"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": f"[40000 lines compressed to 40. Retrieve more: hash={h}]",
                }
            ],
        },
    ]


def test_proxy_flow_search_then_full_with_mixed_client_tool(store):
    original = _big_log(30_000)
    h = store.store(original, "c")
    handler = CCRResponseHandler()
    seen: list[list[dict]] = []

    async def api_call(messages, tools):
        seen.append(list(messages))
        if len(seen) == 1:
            # Second round: the exact full retrieval.
            return {
                "content": [
                    {"type": "tool_use", "id": "r2", "name": CCR_TOOL_NAME, "input": {"hash": h}}
                ]
            }
        return {"content": [{"type": "text", "text": "done"}]}

    first = {
        "content": [
            {
                "type": "tool_use",
                "id": "r1",
                "name": CCR_TOOL_NAME,
                "input": {"hash": h, "query": "stripe_charge_failed", "top_k": 5},
            },
            # Mixed: a client tool in the same turn is dropped for re-issue.
            {"type": "tool_use", "id": "c1", "name": "Read", "input": {"path": "x"}},
        ]
    }
    final = asyncio.run(
        handler.handle_response(
            first, _anthropic_messages(h), None, api_call, provider="anthropic", workspace="ws_t"
        )
    )
    assert final["content"][0]["text"] == "done"
    search_result = json.loads(seen[0][-1]["content"][0]["content"])
    assert search_result["mode"] == "search" and len(search_result["results"]) == 5
    assert all("stripe_charge_failed" in r["text"] for r in search_result["results"])
    full_result = json.loads(seen[1][-1]["content"][0]["content"])
    assert full_result["original_content"] == original  # byte-exact full


def test_selective_retrieval_refuses_hash_not_in_conversation(store):
    h = store.store("alpha beta\n" * 30, "c")
    foreign = store.store("gamma delta\n" * 30, "c")
    handler = CCRResponseHandler()
    seen: list[list[dict]] = []

    async def api_call(messages, tools):
        seen.append(list(messages))
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "x1",
                            "type": "function",
                            "function": {
                                "name": CCR_TOOL_NAME,
                                "arguments": json.dumps({"hash": foreign, "query": "gamma"}),
                            },
                        }
                    ],
                }
            }
        ]
    }
    messages = [
        {"role": "user", "content": f"see [3 items compressed to 1. Retrieve more: hash={h}]"}
    ]
    asyncio.run(handler.handle_response(response, messages, None, api_call, provider="openai"))
    tool_msg = seen[0][-1]
    assert json.loads(tool_msg["content"])["error"] == "not_reachable_in_this_conversation"


def test_proactive_expansion_is_selective_and_bounded(store):
    original = _big_log(20_000)
    h = store.store(original, "c")
    tracker = ContextTracker()
    out = tracker.execute_expansions(
        [ExpansionRecommendation(hash_key=h, reason="test", relevance_score=0.9)],
        query="stripe_charge_failed invoice",
    )
    assert out and out[0]["type"] == "spans"
    assert estimate_tokens(out[0]["content"]) <= 1300
    assert "stripe_charge_failed" in out[0]["content"]
    # Not localizable and too big for the budget: nothing is dumped in full.
    none = tracker.execute_expansions(
        [ExpansionRecommendation(hash_key=h, reason="test", relevance_score=0.9)],
        query="zzzz_nothing_matches",
    )
    assert none == []
