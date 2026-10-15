"""Unified-21 Phase 5B gate: progressive tool catalog (#9)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from headroom.agent.feedback import RetentionFeedback
from headroom.agent.tool_catalog import (
    CATALOG_TAG,
    REGISTRY,
    CatalogRequest,
    LocalTool,
    ProgressiveToolCatalog,
    signature,
    stub_tool,
)

CORE = ["Bash", "Read", "Edit", "Write", "Grep", "Glob"]
MCP = [
    (
        "mcp__linear__create_issue",
        "Create a Linear issue in a team. Use for tracking bugs.",
        {"title": {"type": "string"}, "team": {"type": "string"}, "priority": {"type": "integer"}},
        ["title", "team"],
    ),
    (
        "mcp__sentry__get_issue",
        "Fetch a Sentry error issue with its stack trace and events.",
        {"issue_id": {"type": "string"}, "full": {"type": "boolean"}},
        ["issue_id"],
    ),
    (
        "mcp__slack__post_message",
        "Post a message to a Slack channel.",
        {"channel": {"type": "string"}, "text": {"type": "string"}},
        ["channel", "text"],
    ),
    (
        "mcp__notion__search",
        "Search Notion pages and databases by keyword.",
        {"query": {"type": "string"}},
        ["query"],
    ),
    (
        "mcp__postgres__query",
        "Run a read-only SQL query against the Postgres database.",
        {"sql": {"type": "string"}, "limit": {"type": "integer"}},
        ["sql"],
    ),
    (
        "mcp__github__create_pr",
        "Open a GitHub pull request from a branch.",
        {"repo": {"type": "string"}, "head": {"type": "string"}, "base": {"type": "string"}},
        ["repo", "head"],
    ),
    (
        "mcp__github__list_issues",
        "List GitHub issues in a repository with filters.",
        {"repo": {"type": "string"}, "state": {"type": "string", "enum": ["open", "closed"]}},
        ["repo"],
    ),
    (
        "mcp__figma__get_file",
        "Download a Figma design file's node tree.",
        {"file_key": {"type": "string"}},
        ["file_key"],
    ),
    (
        "mcp__jira__transition",
        "Move a Jira ticket to another workflow state.",
        {"key": {"type": "string"}, "state": {"type": "string"}},
        ["key", "state"],
    ),
    (
        "mcp__datadog__query_metrics",
        "Query Datadog timeseries metrics for a time window.",
        {"query": {"type": "string"}, "from": {"type": "integer"}, "to": {"type": "integer"}},
        ["query"],
    ),
    (
        "mcp__stripe__refund",
        "Refund a Stripe charge fully or partially.",
        {"charge": {"type": "string"}, "amount": {"type": "integer"}},
        ["charge"],
    ),
    (
        "mcp__calendar__create_event",
        "Create a calendar event with attendees.",
        {"title": {"type": "string"}, "start": {"type": "string"}},
        ["title", "start"],
    ),
]


def anthropic_tools():
    tools = [
        {
            "name": n,
            "description": f"{n} tool. " + "Long usage notes. " * 40,
            "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}},
        }
        for n in CORE
    ]
    for name, desc, props, req in MCP:
        tools.append(
            {
                "name": name,
                "description": desc
                + " "
                + "Detailed guidance about every parameter and edge case. " * 30,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        k: {**v, "description": f"The {k} parameter " * 10}
                        for k, v in props.items()
                    },
                    "required": req,
                },
            }
        )
    tools[-1]["cache_control"] = {"type": "ephemeral"}
    return tools


def chat_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in anthropic_tools()
    ]


def req(tools, text="Why is Sentry issue PROJ-123 failing? Check its stack trace.", **kw):
    messages = kw.pop("messages", [{"role": "user", "content": text}])
    return CatalogRequest(
        tools=tools, messages=messages, session_key=kw.pop("session", "s1"), query=text, **kw
    )


def names(tools, shape="anthropic"):
    return [t["function"]["name"] if shape == "openai_chat" else t["name"] for t in tools]


def stubbed(tools):
    return {t["name"] for t in tools if CATALOG_TAG in (t.get("description") or "")}


def test_tool_free_request_is_never_given_tools():
    cat = ProgressiveToolCatalog()
    out = cat.apply(req([]))
    assert out.tools == [] and out.reason == "tool_free" and not out.changed


def test_native_tool_search_and_deferred_schemas_are_preserved():
    cat = ProgressiveToolCatalog()
    tools = anthropic_tools() + [
        {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}
    ]
    out = cat.apply(req(tools))
    assert out.tools == tools and out.reason == "native_search"
    deferred = anthropic_tools()
    deferred[8]["defer_loading"] = True
    assert cat.apply(req(deferred)).reason == "native_search"


def test_catalog_materializes_top5_keeps_core_and_signatures():
    cat = ProgressiveToolCatalog()
    tools = anthropic_tools()
    out = cat.apply(req(tools))
    assert out.changed and names(out.tools) == names(tools)  # every tool stays callable
    stubs = stubbed(out.tools)
    assert not stubs & set(CORE)  # resident coding tools keep their full schemas
    assert len([n for n, *_ in MCP if n not in stubs]) == 5
    assert "mcp__sentry__get_issue" not in stubs  # task-relevant tool materialized
    entry = next(t for t in out.tools if t["name"] == "mcp__linear__create_issue")
    if entry["name"] in stubs:
        assert entry["input_schema"]["required"] == ["title", "team"]
        assert (
            "mcp__linear__create_issue(title: string, team: string, priority?: integer)"
            in entry["description"]
        )
    last = out.tools[-1]
    assert last["cache_control"] == {"type": "ephemeral"}  # breakpoint kept on the stub
    mcp_names = {n for n, *_ in MCP}
    before = len(json.dumps([t for t in tools if t["name"] in mcp_names]))
    after = len(json.dumps([t for t in out.tools if t["name"] in mcp_names]))
    assert after < before * 0.75


def test_signature_and_stub_shapes():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        "required": ["a"],
    }
    assert signature("t", schema) == "t(a: string, b?: integer)"
    chat = stub_tool(
        {
            "type": "function",
            "function": {"name": "t", "description": "Does t. More.", "parameters": schema},
        },
        "openai_chat",
    )
    assert chat["type"] == "function" and chat["function"]["parameters"]["required"] == ["a"]
    assert chat["function"]["description"].startswith("Does t.")


def test_sticky_per_session_and_no_churn_on_hot_prefix():
    cat = ProgressiveToolCatalog()
    tools = anthropic_tools()
    first = cat.apply(req(tools))
    again = cat.apply(req(tools, prefix_hot=True))
    assert again.tools == first.tools and again.reused
    # Task drifts to Slack while the prefix is hot: no re-rank, same bytes.
    drift = cat.apply(
        req(
            tools,
            text="post the summary to slack channel #eng",
            prefix_hot=True,
            messages=[{"role": "user", "content": "post the summary to channel eng"}],
        )
    )
    assert json.dumps(drift.tools) == json.dumps(first.tools)


def test_tool_choice_and_user_named_tool_are_materialized():
    cat = ProgressiveToolCatalog()
    tools = anthropic_tools()
    out = cat.apply(req(tools, tool_choice={"type": "tool", "name": "mcp__stripe__refund"}))
    assert "mcp__stripe__refund" not in stubbed(out.tools)
    full = next(t for t in out.tools if t["name"] == "mcp__stripe__refund")
    assert full == next(t for t in tools if t["name"] == "mcp__stripe__refund")
    # Schema introspection: the user names a tool -> its full schema is present.
    cat2 = ProgressiveToolCatalog()
    out2 = cat2.apply(req(tools, text="What parameters does mcp__figma__get_file take?"))
    assert "mcp__figma__get_file" not in stubbed(out2.tools)
    # On a hot prefix, a forced tool is added (correctness) without dropping any.
    out3 = cat.apply(
        req(tools, prefix_hot=True, tool_choice={"type": "tool", "name": "mcp__jira__transition"})
    )
    assert (
        set(out.materialized) <= set(out3.materialized)
        and "mcp__jira__transition" in out3.materialized
    )


def test_calling_a_catalog_only_tool_materializes_it_and_feeds_back():
    cat = ProgressiveToolCatalog()
    fb = RetentionFeedback()
    tools = anthropic_tools()
    first = cat.apply(req(tools, feedback=fb))
    victim = sorted(stubbed(first.tools))[0]
    history = [
        {"role": "user", "content": "Why is Sentry issue PROJ-123 failing?"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "c1", "name": victim, "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "ok"}],
        },
    ]
    nxt = cat.apply(req(tools, messages=history, prefix_hot=True, feedback=fb))
    assert victim not in stubbed(nxt.tools)
    assert cat.status()["catalog_miss_calls"] == 1
    assert (
        fb.counters["registered"] == 0 and fb.status()["keys"] == 1
    )  # an observation, no raw name


def test_openai_chat_shape_and_strict_tools():
    cat = ProgressiveToolCatalog()
    tools = chat_tools()
    tools[-2]["function"]["strict"] = True
    out = cat.apply(req(tools))
    assert out.changed and names(out.tools, "openai_chat") == names(tools, "openai_chat")
    assert out.tools[-2] == tools[-2]  # strict schema never compacted


def test_registry_residents_are_never_stubbed():
    REGISTRY.register(LocalTool("mcp__notion__search", resident=True))
    try:
        cat = ProgressiveToolCatalog()
        out = cat.apply(req(anthropic_tools()))
        assert "mcp__notion__search" not in stubbed(out.tools)
    finally:
        REGISTRY.unregister("mcp__notion__search")
    assert REGISTRY.is_resident("headroom_retrieve")


def test_jevk5_breaks_ambiguous_ties_only():
    class Adv:
        def __init__(self):
            self.calls = 0

        def request_budget(self):
            return SimpleNamespace()

        def choose(self, family, state, instructions, options, *, budget):
            self.calls += 1
            assert len(options) <= 16
            pick = next(
                k for k, v in options.items() if v.startswith("mcp__calendar__create_event")
            )
            return SimpleNamespace(
                value=pick, probabilities={pick: 0.95}, confidence=0.95, weight=0.4
            )

    adv = Adv()
    cat = ProgressiveToolCatalog()
    out = cat.apply(req(anthropic_tools(), text="", advisor=adv))  # no task terms: ambiguous
    assert adv.calls == 1 and out.jev_weight == 0.4
    assert "mcp__calendar__create_event" not in stubbed(out.tools)


def test_too_few_tools_left_alone():
    cat = ProgressiveToolCatalog()
    small = anthropic_tools()[:8]
    out = cat.apply(req(small))
    assert out.tools == small and out.reason == "too_few_tools"


# ---------------------------------------------------------------------------
# Real wrap request path (Anthropic, tool search unavailable)
# ---------------------------------------------------------------------------


@pytest.fixture
def wrap_client(tmp_path, monkeypatch):
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
    monkeypatch.setenv("HEADROOM_TOOL_SEARCH", "0")  # no server-side deferral available
    import headroom.agent.tool_catalog as tc

    monkeypatch.setattr(tc, "_CATALOG", None)
    clear_turn_hooks()
    registry.clear_registry()

    def make(env):
        config = ProxyConfig(
            port=18983,
            optimize=True,
            disable_kompress=True,
            cache_enabled=False,
            rate_limit_enabled=False,
            memory_enabled=False,
            image_optimize=False,
            stateless=True,
            rollout=resolve_rollout(env),
        )
        return TestClient(
            create_app(config), base_url="http://127.0.0.1", client=("127.0.0.1", 50000)
        )

    yield make, apply_wrap_feature_defaults
    clear_turn_hooks()
    registry.clear_registry()
    paths.set_process_stateless(False)


UPSTREAM = {
    "id": "m",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 1},
}


def _post(client, body):
    sent: list[dict] = []

    async def fake(method, url, headers, payload, stream=False, **kwargs):  # noqa: ANN001
        sent.append(payload if isinstance(payload, dict) else json.loads(payload))
        return httpx.Response(200, json=UPSTREAM)

    client.app.state.proxy._retry_request = fake
    resp = client.post(
        "/v1/messages", headers={"x-api-key": "k", "anthropic-version": "2023-06-01"}, json=body
    )
    assert resp.status_code == 200, resp.text
    return sent[-1]


def test_wrap_path_catalog_applies_and_tool_free_untouched(wrap_client):
    make, wrap_env = wrap_client
    with make(wrap_env({})) as client:
        body = {
            "model": "claude-sonnet-4-5",
            "max_tokens": 32,
            "tools": anthropic_tools(),
            "messages": [{"role": "user", "content": "Why is Sentry issue PROJ-123 failing?"}],
        }
        fwd = _post(client, body)
        assert set(names(anthropic_tools())) <= set(names(fwd["tools"]))  # all still callable
        assert stubbed(fwd["tools"]) and "mcp__sentry__get_issue" not in stubbed(fwd["tools"])
        again = _post(client, body)
        assert again["tools"] == fwd["tools"]  # cache-stable bytes
        free = _post(
            client,
            {
                "model": "claude-sonnet-4-5",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        # The catalog adds nothing to a tool-free request (only Headroom's own
        # pre-existing sticky CCR retrieval tool may ride along).
        assert set(names(free.get("tools") or [])) <= {"headroom_retrieve"}
    with make(wrap_env({"HEADROOM_DISABLE_FEATURES": "agent_tool_catalog"})) as client:
        fwd = _post(
            client,
            {
                "model": "claude-sonnet-4-5",
                "max_tokens": 32,
                "tools": anthropic_tools(),
                "messages": [{"role": "user", "content": "Why is Sentry issue PROJ-123 failing?"}],
            },
        )
        assert not stubbed(fwd["tools"])
