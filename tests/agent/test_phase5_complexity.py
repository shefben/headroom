"""Unified-21 Phase 5C gate: complexity-aware effort/verbosity routing (#10)."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from headroom.agent import complexity
from headroom.agent.complexity import (
    TURN_BLOCK_TAG,
    lowered_effort,
    route_request,
)
from headroom.rollout import apply_wrap_feature_defaults, resolve_rollout

MODEL = "claude-sonnet-4-5"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("HEADROOM_OUTPUT_HOLDOUT", raising=False)
    monkeypatch.delenv("HEADROOM_OUTPUT_SHAPER", raising=False)
    monkeypatch.delenv("HEADROOM_VERBOSITY_LEVEL", raising=False)
    complexity.reset()
    yield
    complexity.reset()


def config(extra: dict | None = None):
    env = apply_wrap_feature_defaults(dict(extra or {}))
    return SimpleNamespace(rollout=resolve_rollout(env), optimize=True, mode="token")


def mechanical(
    result="All 12 checks passed.", is_error=False, effort="xhigh", first="Refactor the parser"
):
    return {
        "model": MODEL,
        "max_tokens": 1000,
        "output_config": {"effort": effort},
        "messages": [
            {"role": "user", "content": first},
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
                    },
                ],
            },
        ],
    }


def route(body, cfg=None, **kw):
    return route_request(
        body, shape=kw.pop("shape", "anthropic"), config=cfg or config(), model=MODEL, **kw
    )


def test_effort_ladder_never_raises_and_has_a_floor():
    assert lowered_effort("xhigh", 0) == "medium"
    assert lowered_effort("xhigh", 1) == "high"
    assert lowered_effort("xhigh", 2) == "xhigh"
    assert lowered_effort("medium", 0) == "medium"
    assert lowered_effort("low", 0) == "low"


def test_simple_mechanical_turn_lowers_client_effort():
    body = mechanical()
    d = route(body, session_key="a")
    assert d is not None and d.protected is None
    # No task state in this unit test -> "unknown requirements" counts as 1.
    assert d.score == 1 and d.effort_after == "high" and body["output_config"]["effort"] == "high"
    assert d.changed and any(label.startswith("agent:effort:xhigh->high") for label in d.labels)
    assert "thinking" not in body


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (
            lambda b: (
                b["messages"].append({"role": "assistant", "content": "ok"})
                or b["messages"].append({"role": "user", "content": "Now add caching"})
            ),
            "new_user_ask",
        ),
        (lambda b: b["messages"][-1]["content"][0].update(is_error=True), "error_continuation"),
        (
            lambda b: b["messages"][-1]["content"][0].update(
                content="Traceback (most recent call last):\nValueError: bad"
            ),
            "errors",
        ),
    ],
)
def test_protected_turns_keep_effort(mutate, reason):
    body = mechanical()
    mutate(body)
    before = copy.deepcopy(body["output_config"])
    d = route(body, session_key=reason)
    assert d.protected is not None and body["output_config"] == before


def test_explicit_deep_thinking_request_is_never_overridden():
    body = mechanical(first="ultrathink about the parser design")
    route(body, session_key="deep")
    assert body["output_config"]["effort"] == "xhigh"


def test_effort_is_never_injected_and_thinking_untouched():
    body = mechanical()
    del body["output_config"]
    body["thinking"] = {"type": "enabled", "budget_tokens": 4096}
    snapshot = copy.deepcopy(body)
    d = route(body, session_key="none")
    assert body == snapshot and d.effort_before is None


def test_responses_and_chat_effort_locations():
    responses = {
        "model": "gpt-5",
        "reasoning": {"effort": "high"},
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "refactor"}],
            },
            {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "done"},
        ],
    }
    d = route(responses, shape="responses", session_key="r")
    assert d.effort_after == "medium" and responses["reasoning"]["effort"] == "medium"
    chat = {
        "model": "gpt-5",
        "reasoning_effort": "high",
        "messages": [
            {"role": "user", "content": "refactor"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "shell", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "done"},
        ],
    }
    route(chat, shape="chat", session_key="c")
    assert chat["reasoning_effort"] == "medium"


def test_holdout_control_arm_is_untouched(monkeypatch):
    monkeypatch.setenv("HEADROOM_OUTPUT_HOLDOUT", "1.0")
    body = mechanical()
    d = route(body, session_key="ctl")
    assert d.arm == "control" and body["output_config"]["effort"] == "xhigh"


def test_feature_off_returns_none():
    body = mechanical()
    cfg = config({"HEADROOM_DISABLE_FEATURES": "agent_complexity_routing"})
    assert route(body, cfg) is None and body["output_config"]["effort"] == "xhigh"


def test_jevk5_scores_only_the_uncertain_middle(monkeypatch):
    calls = []

    class Adv:
        def request_budget(self):
            return SimpleNamespace()

        def score(self, family, state, instructions, levels, *, budget):
            calls.append(levels)
            return SimpleNamespace(value=0.2, confidence=0.9, weight=0.25)

    monkeypatch.setattr(
        complexity, "_snapshot_and_advisor", lambda hint: (None, Adv(), f"ws|{hint}")
    )
    # Score 1 (only unknown requirements): deterministic, no JevK5 call.
    route(mechanical(), session_key="j1")
    assert calls == []
    # Multi-file + unknown => score 2 => JevK5 0-4 scoring decides.
    body = mechanical(result="\n".join(f"src/pkg/mod_{i}.py: ok" for i in range(5)))
    d = route(body, session_key="j2")
    assert calls == [["0", "1", "2", "3", "4"]] and d.score_source == "jevk5" and d.score == 0
    assert body["output_config"]["effort"] == "medium"


def test_contradicted_evidence_and_open_gate_protect(monkeypatch):
    goal = SimpleNamespace(text="fix parser")
    acceptance = [SimpleNamespace(live=True, state="ACTIVE")]
    state = SimpleNamespace(goal=goal, acceptance=acceptance)
    snap = SimpleNamespace(state=state, evidence=(SimpleNamespace(status="CONTRADICTED"),))
    monkeypatch.setattr(
        complexity, "_snapshot_and_advisor", lambda hint: (snap, None, f"ws|{hint}")
    )
    body = mechanical()
    d = route(body, session_key="ev")
    assert d.protected == "contradictions" and body["output_config"]["effort"] == "xhigh"
    snap2 = SimpleNamespace(state=state, evidence=())
    monkeypatch.setattr(
        complexity, "_snapshot_and_advisor", lambda hint: (snap2, None, f"ws|{hint}")
    )
    body = mechanical()
    body["messages"][1]["content"][0]["text"] = "All done, the fix is complete."
    d = route(body, session_key="gate")
    assert d.protected == "unresolved_gate" and body["output_config"]["effort"] == "xhigh"


def test_verbosity_relaxed_for_complex_new_ask_without_touching_system(monkeypatch):
    monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "1")
    monkeypatch.setenv("HEADROOM_VERBOSITY_LEVEL", "3")
    cfg = config({"HEADROOM_FEATURES": "proxy_output_shaper"})
    body = {
        "model": "claude-opus-5",
        "system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
        "output_config": {"effort": "xhigh"},
        "messages": [
            {"role": "user", "content": "Explain why the cache misses and compare the trade-offs"}
        ],
    }
    system_before = copy.deepcopy(body["system"])
    d = route_request(body, shape="anthropic", config=cfg, model="claude-opus-5", session_key="v")
    assert d.verbosity_base == 3 and d.verbosity_turn == 1
    assert body["system"] == system_before
    last = body["messages"][-1]
    assert last["role"] == "system" and TURN_BLOCK_TAG in last["content"]
    assert body["output_config"]["effort"] == "xhigh"  # new ask: effort untouched (independent)
    # A model without mid-conversation system support gets no turn block.
    old_model = mechanical(first="Explain why the cache misses")
    old_model["messages"] = [{"role": "user", "content": "Explain why the cache misses"}]
    assert route(old_model, cfg, session_key="v1").verbosity_turn is None
    # A mechanical turn keeps the conversation level (no turn block).
    body2 = mechanical()
    d2 = route(body2, cfg, session_key="v2")
    assert d2.verbosity_turn is None and body2["messages"][-1]["role"] == "user"


def test_wrap_handler_routes_effort(tmp_path, monkeypatch):
    import json

    import httpx
    from fastapi.testclient import TestClient

    from headroom import paths
    from headroom.agent import registry
    from headroom.proxy.server import ProxyConfig, create_app
    from headroom.proxy.turn_hooks import clear_turn_hooks

    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "ws"))
    monkeypatch.setenv("HEADROOM_JEVK5", "off")
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    clear_turn_hooks()
    registry.clear_registry()
    cfg = ProxyConfig(
        port=18985,
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
            create_app(cfg), base_url="http://127.0.0.1", client=("127.0.0.1", 50000)
        ) as client:

            async def fake(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
                sent.append(body if isinstance(body, dict) else json.loads(body))
                return httpx.Response(200, json=upstream)

            client.app.state.proxy._retry_request = fake
            for body in (mechanical(), mechanical(first="Now add caching please")):
                r = client.post(
                    "/v1/messages",
                    headers={"x-api-key": "k", "anthropic-version": "2023-06-01"},
                    json=body,
                )
                assert r.status_code == 200, r.text
            assert sent[0]["output_config"]["effort"] in ("high", "medium")
            new_ask = mechanical(first="x")
            new_ask["messages"] = [{"role": "user", "content": "Now add caching please"}]
            r = client.post(
                "/v1/messages",
                headers={"x-api-key": "k", "anthropic-version": "2023-06-01"},
                json=new_ask,
            )
            assert r.status_code == 200 and sent[-1]["output_config"]["effort"] == "xhigh"
    finally:
        clear_turn_hooks()
        registry.clear_registry()
        paths.set_process_stateless(False)
