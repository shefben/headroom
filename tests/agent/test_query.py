"""TaskContext / relevance query (2D): genuine user intent, exact entities first,
512-token cap, graph neighbors only for code tasks, Rust-boundary round-trip."""

from __future__ import annotations

from pathlib import Path

from headroom.agent.code_graph import CodeGraphAdapter
from headroom.agent.identity import WorkspaceIdentity
from headroom.agent.interfaces import (
    RELEVANCE_QUERY_MAX_TOKENS,
    RelevanceQuery,
    TaskContext,
    estimate_query_tokens,
)
from headroom.agent.normalize import normalize_messages
from headroom.agent.query import build_task_context, extract_entities
from headroom.agent.runtime import AgentRuntime
from headroom.agent.store import MemoryAgentStore
from headroom.proxy.turn_hooks import TurnContext


def _ctx(messages, provider="anthropic"):
    return TurnContext(provider=provider, model="claude-sonnet-5", messages=messages, tags={})


def test_extract_entities_prioritizes_exact_ids():
    text = (
        "decode_escape in `pkg/parser.py` raises UnicodeDecodeError (see #412, commit "
        "a1b2c3d4e5) — check ParserState.reset() and HTTP 502 from api/v2/items."
    )
    found = extract_entities(text)
    assert found["paths"][0] == "pkg/parser.py"
    assert "api/v2/items" in found["paths"]
    assert {"decode_escape", "ParserState.reset"} <= set(found["symbols"])
    assert "UnicodeDecodeError" in found["errors"] and "HTTP 502" in found["errors"]
    assert "#412" in found["identifiers"] and "a1b2c3d4e5" in found["identifiers"]


def test_harness_text_is_not_the_user_goal():
    identity = WorkspaceIdentity("wu_q", "su_q", False, "me")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "<system-reminder>Use TodoWrite. Fix auth.py!</system-reminder>",
                },
                {"type": "text", "text": "Make the retry loop in client.py exponential."},
            ],
        }
    ]
    turn = normalize_messages("anthropic", messages, identity)
    task = build_task_context(
        ctx=_ctx(messages), identity=identity, turn=turn, runtime=None, request_id="r1"
    )
    assert task.latest_user_goal == "Make the retry loop in client.py exponential."
    assert "auth.py" not in task.exact_paths and task.exact_paths == ("client.py",)
    q = task.relevance_query
    assert q is not None and "TodoWrite" not in q.text and "client.py" in q.text


def test_tool_intent_error_completeness_and_resource():
    identity = WorkspaceIdentity("wu_q", "su_q", False, "me")
    messages = [
        {"role": "user", "content": "why does test_parse fail?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "Bash",
                    "input": {"command": "pytest tests/test_p.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "is_error": True,
                    "content": "E   KeyError: 'token'\n1 failed\nExit code: 1",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t2", "name": "Read", "input": {"file_path": "src/p.py"}}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t2",
                    "content": "def parse():\n    pass\n[output truncated]",
                }
            ],
        },
    ]
    turn = normalize_messages("anthropic", messages, identity)
    task = build_task_context(
        ctx=_ctx(messages), identity=identity, turn=turn, runtime=None, request_id="r2"
    )
    assert task.tool_name == "Read" and task.tool_call_id == "t2"
    assert task.tool_intent == "read: src/p.py"
    assert task.resource_id == "path:src/p.py" and task.resource_hash
    assert task.result_truncated is True and task.result_complete is False
    assert task.relevance_query.active_error.startswith("KeyError")
    assert task.relevance_query.is_code
    # Interface round-trip for the Python/Rust boundary.
    assert TaskContext.from_dict(task.to_dict()) == task


def test_relevance_query_caps_at_512_tokens_entities_first():
    entities = tuple(f"module_{i}/file_{i}.py" for i in range(400))
    query = RelevanceQuery(
        explicit_user_goal="word " * 2000,
        tool_intent="read: x",
        exact_entities=entities,
        active_error="ValueError: bad",
        graph_neighbors=("n1", "n2"),
        is_code=True,
    )
    text = query.text
    assert estimate_query_tokens(text) <= RELEVANCE_QUERY_MAX_TOKENS
    assert entities[0] in text  # exact ids survive the cut; prose is trimmed instead
    small = RelevanceQuery(
        explicit_user_goal="fix it",
        exact_entities=("a.py",),
        graph_neighbors=("pkg/b.py",),
        is_code=False,
    )
    assert "pkg/b.py" not in small.text  # non-code tasks omit graph context
    assert RelevanceQuery.from_dict(query.to_dict()) == query


def test_graph_neighbors_attached_for_code_tasks(tmp_path):
    root = Path(tmp_path)
    (root / "pkg").mkdir()
    (root / "pkg" / "parser.py").write_text(
        "from pkg.lexer import tokenize\n\ndef decode_escape(t):\n    return tokenize(t)\n",
        encoding="utf-8",
    )
    (root / "pkg" / "lexer.py").write_text("def tokenize(t):\n    return t\n", encoding="utf-8")
    identity = WorkspaceIdentity("ws_q", "ss_q", True, "me", root=str(root))
    runtime = AgentRuntime(MemoryAgentStore())
    graph = CodeGraphAdapter(None, lister=lambda r: None)
    runtime.services["code_graph"] = graph
    graph.ensure_workspace(identity)
    assert graph.wait_idle(30)
    messages = [{"role": "user", "content": "decode_escape drops characters, fix it"}]
    turn = normalize_messages("anthropic", messages, identity)
    task = build_task_context(
        ctx=_ctx(messages), identity=identity, turn=turn, runtime=runtime, request_id="r3"
    )
    assert task.graph_neighbors and task.graph_neighbors[0].startswith("decode_escape")
    assert any("lexer" in n or "tokenize" in n for n in task.graph_neighbors)
    assert task.relevance_query.graph_neighbors == task.graph_neighbors
    prose = [{"role": "user", "content": "write a haiku about autumn"}]
    task2 = build_task_context(
        ctx=_ctx(prose),
        identity=identity,
        turn=normalize_messages("anthropic", prose, identity),
        runtime=runtime,
        request_id="r4",
    )
    assert task2.graph_neighbors == () and task2.relevance_query.is_code is False
    graph.close()
    runtime.stop()
