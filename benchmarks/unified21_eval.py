"""Unified-21 baseline evaluation harness.

Deterministic, offline evaluation of Headroom's *real* compression paths over a
fixed set of agent-shaped fixtures. Used as the Phase-0 baseline for the
unified-21 implementation plan and re-run after each phase to compare.

Paths exercised (no network, no LLM):

* ``python_proxy_anthropic`` -- the Python proxy's ``anthropic_pipeline.apply``
  (``HeadroomProxy`` built from ``ProxyConfig`` with the out-of-box ``coding``
  savings profile, the same kwargs the Anthropic handler passes: user query
  context, frozen prefix count, resolved ``CompressionPolicy`` and
  ``proxy_pipeline_kwargs``). This is the path ``headroom wrap claude`` uses.
* ``rust_live_zone_responses`` -- the Rust live-zone dispatcher exposed through
  ``headroom._core.compress_openai_responses_live_zone`` on an OpenAI Responses
  body (the Codex ``/v1/responses`` shape, including ``stream: true``).

Per fixture and path we record raw vs forwarded token counts (Headroom's own
tokenizer for the model), exact must-keep fact recall in the forwarded bytes,
whether missing facts are recoverable from the CCR store (and so would cost a
``headroom_retrieve`` tool turn), the fixture's tool turns, frozen-prefix byte
fidelity, structural validity of the forwarded body, wall time, and any error.

Run::

    python -m benchmarks.unified21_eval --label baseline --out benchmarks/results/baseline.json

``--jevk5 off|on`` additionally installs the agent runtime for the wrap path
exactly as ``create_app`` does (``install_agent_runtime``), with the JevK5
service policy ``HEADROOM_JEVK5`` set to the given value, so the transform
arbiter reaches the real ``DecisionAdvisor``. The advisor is activated
synchronously (``on`` reuses a running managed llama-server, or the one named by
``HEADROOM_JEVK5_LLAMA_URL``), and every advisor call's latency is recorded with
whether its advice cleared the confidence floor and was blended in.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import re
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MODEL = "claude-sonnet-4-5-20250929"
RESPONSES_MODEL = "gpt-5.4"
MODEL_LIMIT = 200_000

# Offline + stateless before any headroom import reads the environment.
_ENV_DEFAULTS = {
    "HEADROOM_OFFLINE": "1",
    "HEADROOM_STATELESS": "1",
    "HEADROOM_TELEMETRY": "off",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}

_CCR_HASH_RE = re.compile(r"(?:<<ccr:|hash=)([a-f0-9]{12,24})")


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@dataclass
class Fixture:
    """One evaluation case.

    ``messages`` is an Anthropic Messages conversation whose latest user turn is
    the live zone. ``must_keep`` lists exact substrings that a correct answer
    depends on. ``frozen_message_count`` marks a cache-hot prefix whose bytes
    must be forwarded unchanged.
    """

    name: str
    category: str
    messages: list[dict[str, Any]]
    must_keep: list[str]
    tool_turns: int
    frozen_message_count: int = 0
    responses_body: dict[str, Any] | None = None
    notes: str = ""


def _tool_use(tool_id: str, name: str, inp: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_use", "id": tool_id, "name": name, "input": inp}


def _tool_result(tool_id: str, text: str, is_error: bool = False) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_id, "content": text}
    if is_error:
        block["is_error"] = True
    return block


def _single_tool_conversation(
    question: str,
    tool_name: str,
    tool_input: dict[str, Any],
    output: str,
    *,
    is_error: bool = False,
) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll check that."},
                _tool_use("toolu_01", tool_name, tool_input),
            ],
        },
        {"role": "user", "content": [_tool_result("toolu_01", output, is_error=is_error)]},
    ]


def _responses_body_for(
    question: str, tool_name: str, tool_input: dict[str, Any], output: str
) -> dict[str, Any]:
    return {
        "model": RESPONSES_MODEL,
        "stream": True,
        "instructions": "You are a coding agent.",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": question}],
            },
            {
                "type": "function_call",
                "call_id": "call_01",
                "name": tool_name,
                "arguments": json.dumps(tool_input, sort_keys=True),
            },
            {"type": "function_call_output", "call_id": "call_01", "output": output},
        ],
        "tools": [
            {
                "type": "function",
                "name": tool_name,
                "description": f"{tool_name} tool",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    }


def _python_source(n_funcs: int = 60) -> str:
    lines: list[str] = ['"""Payload utilities."""', "import hashlib", "import json", ""]
    lines.append("MAX_RETRIES = 7")
    lines.append("")
    for i in range(n_funcs):
        if i == 41:
            lines += [
                "def compute_checksum(payload: bytes) -> str:",
                '    """Return the sha256 hex digest used by the uploader."""',
                "    return hashlib.sha256(payload).hexdigest()",
                "",
            ]
            continue
        lines += [
            f"def helper_{i:03d}(value: int) -> int:",
            f'    """Helper number {i}: scale and offset the input."""',
            f"    scaled = value * {i + 3}",
            f"    return scaled + {i % 11}",
            "",
        ]
    return "\n".join(f"{n:6d}\t{line}" for n, line in enumerate(lines, start=1))


def _search_output() -> str:
    rows = []
    for i in range(420):
        rows.append(
            f"src/module_{i % 37:02d}/file_{i:04d}.py:{(i * 7) % 900 + 1}:    budget = Budget(limit={i})"
        )
    rows.insert(211, "src/core/budget.py:212:    raise TokenBudgetExceeded(limit)")
    return "\n".join(rows)


def _pytest_output() -> str:
    lines = [
        "============================= test session starts =============================",
        "platform win32 -- Python 3.12.10, pytest-8.3.5",
        "collected 300 items",
        "",
    ]
    for i in range(298):
        lines.append(
            f"tests/test_module_{i % 23:02d}.py::test_case_{i:04d} PASSED [{(i + 1) * 100 // 300:3d}%]"
        )
    lines += [
        "tests/test_parser.py::test_unicode_escape FAILED",
        "tests/test_cache.py::test_eviction_order FAILED",
        "",
        "================================== FAILURES ===================================",
        "_____________________________ test_unicode_escape _____________________________",
        "    def test_unicode_escape():",
        ">       assert decode('\\\\xe9') == 'é'",
        "E       AssertionError: assert 'Ã©' == 'é'",
        "tests/test_parser.py:88: AssertionError",
        "_____________________________ test_eviction_order _____________________________",
        "E       assert ['b', 'a'] == ['a', 'b']",
        "tests/test_cache.py:142: AssertionError",
        "=========================== short test summary info ===========================",
        "FAILED tests/test_parser.py::test_unicode_escape - AssertionError: assert 'Ã©' == 'é'",
        "FAILED tests/test_cache.py::test_eviction_order - assert ['b', 'a'] == ['a', 'b']",
        "======================== 2 failed, 298 passed in 41.27s ========================",
    ]
    return "\n".join(lines)


def _service_log() -> str:
    lines = []
    for i in range(360):
        ts = f"2026-10-14T09:{i // 60:02d}:{i % 60:02d}.{i * 13 % 1000:03d}Z"
        lines.append(f"{ts} INFO  [http.server] GET /api/items/{i} 200 {i % 40 + 3}ms")
        if i == 233:
            lines.append(f"{ts} ERROR [db.pool] connection refused: 10.0.3.17:5432")
            lines.append(f"{ts} WARN  [db.pool] retrying in 2500ms (attempt 3/7)")
    return "\n".join(lines)


def _json_records() -> str:
    records = []
    for i in range(200):
        rec: dict[str, Any] = {
            "id": i,
            "status": "ok",
            "region": "us-east-1",
            "latency_ms": 20 + i % 9,
        }
        if i == 137:
            rec = {
                "id": 137,
                "status": "failed",
                "region": "eu-west-2",
                "latency_ms": 9001,
                "error_code": "E_QUOTA_4291",
            }
        records.append(rec)
    return json.dumps(records, indent=2)


def _git_diff() -> str:
    parts = []
    for f in range(12):
        parts += [
            f"diff --git a/src/pkg/mod_{f:02d}.py b/src/pkg/mod_{f:02d}.py",
            "index 1111111..2222222 100644",
            f"--- a/src/pkg/mod_{f:02d}.py",
            f"+++ b/src/pkg/mod_{f:02d}.py",
            f"@@ -{f * 10 + 1},7 +{f * 10 + 1},7 @@",
        ]
        for j in range(10):
            parts.append(f"     unchanged_line_{f}_{j} = {j}")
        parts += [f"-    label = 'old_{f}'", f"+    label = 'new_{f}'"]
    parts += [
        "diff --git a/src/net/client.py b/src/net/client.py",
        "index 3333333..4444444 100644",
        "--- a/src/net/client.py",
        "+++ b/src/net/client.py",
        "@@ -40,7 +40,7 @@ class Client:",
        "     def connect(self):",
        "-    timeout_ms = 1000",
        "+    timeout_ms = 2500",
        "         return self._open()",
    ]
    return "\n".join(parts)


def _tool_error() -> str:
    frames = []
    for i in range(30):
        frames.append(f'  File "/srv/app/lib/layer_{i:02d}.py", line {100 + i}, in call_{i}')
        frames.append(f"    return next_layer_{i + 1}(ctx)")
    return "\n".join(
        [
            "Traceback (most recent call last):",
            *frames,
            "PermissionError: [Errno 13] Permission denied: '/var/lib/app/state.db'",
        ]
    )


def build_fixtures() -> list[Fixture]:
    fx: list[Fixture] = []

    src = _python_source()
    q = "Where is compute_checksum defined and what is MAX_RETRIES?"
    fx.append(
        Fixture(
            "file_read",
            "file_read",
            _single_tool_conversation(q, "Read", {"file_path": "src/payload.py"}, src),
            ["def compute_checksum(payload: bytes) -> str:", "MAX_RETRIES = 7"],
            1,
            responses_body=_responses_body_for(q, "read_file", {"path": "src/payload.py"}, src),
        )
    )

    out = _search_output()
    q = "Find where TokenBudgetExceeded is raised."
    fx.append(
        Fixture(
            "large_search",
            "search",
            _single_tool_conversation(q, "Grep", {"pattern": "Budget"}, out),
            ["src/core/budget.py:212:    raise TokenBudgetExceeded(limit)"],
            1,
            responses_body=_responses_body_for(q, "shell", {"cmd": "rg Budget"}, out),
        )
    )

    out = _pytest_output()
    q = "Which tests failed and why?"
    fx.append(
        Fixture(
            "pytest_output",
            "logs",
            _single_tool_conversation(q, "Bash", {"command": "pytest -q"}, out),
            [
                "FAILED tests/test_parser.py::test_unicode_escape",
                "FAILED tests/test_cache.py::test_eviction_order",
                "2 failed, 298 passed",
            ],
            1,
            responses_body=_responses_body_for(q, "shell", {"cmd": "pytest -q"}, out),
        )
    )

    out = _service_log()
    q = "Why did the database calls fail?"
    fx.append(
        Fixture(
            "service_log",
            "logs",
            _single_tool_conversation(q, "Bash", {"command": "tail -n 400 service.log"}, out),
            ["ERROR [db.pool] connection refused: 10.0.3.17:5432"],
            1,
            responses_body=_responses_body_for(q, "shell", {"cmd": "tail -n 400 service.log"}, out),
        )
    )

    out = _json_records()
    q = "Which record failed and with what error code?"
    fx.append(
        Fixture(
            "json_records",
            "json",
            _single_tool_conversation(q, "Bash", {"command": "curl -s localhost/api/records"}, out),
            ["E_QUOTA_4291", "eu-west-2", "9001"],
            1,
            responses_body=_responses_body_for(
                q, "shell", {"cmd": "curl -s localhost/api/records"}, out
            ),
        )
    )

    out = _git_diff()
    q = "What changed in the network client timeout?"
    fx.append(
        Fixture(
            "git_diff",
            "diff",
            _single_tool_conversation(q, "Bash", {"command": "git diff"}, out),
            [
                "diff --git a/src/net/client.py b/src/net/client.py",
                "-    timeout_ms = 1000",
                "+    timeout_ms = 2500",
            ],
            1,
            responses_body=_responses_body_for(q, "shell", {"cmd": "git diff"}, out),
        )
    )

    # Same output read twice: the earlier read sits in history, the reread is live.
    src2 = _python_source()
    reread: list[dict[str, Any]] = [
        {"role": "user", "content": "Open src/payload.py."},
        {
            "role": "assistant",
            "content": [_tool_use("toolu_01", "Read", {"file_path": "src/payload.py"})],
        },
        {"role": "user", "content": [_tool_result("toolu_01", src2)]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Opened. It defines helpers."},
                _tool_use("toolu_02", "Read", {"file_path": "src/payload.py"}),
            ],
        },
        {
            "role": "user",
            "content": [
                _tool_result("toolu_02", src2),
                {"type": "text", "text": "Re-read it: what does compute_checksum return?"},
            ],
        },
    ]
    fx.append(
        Fixture(
            "same_output_reread",
            "reread",
            reread,
            [
                "def compute_checksum(payload: bytes) -> str:",
                "return hashlib.sha256(payload).hexdigest()",
            ],
            2,
            notes="identical Read output appears twice; dedup opportunity",
        )
    )

    out = _tool_error()
    q = "Why can't the service start?"
    fx.append(
        Fixture(
            "tool_error",
            "tool_error",
            _single_tool_conversation(q, "Bash", {"command": "python -m app"}, out, is_error=True),
            ["PermissionError: [Errno 13] Permission denied: '/var/lib/app/state.db'"],
            1,
            responses_body=_responses_body_for(q, "shell", {"cmd": "python -m app"}, out),
        )
    )

    pyproject = "\n".join(
        ["[project]", 'name = "widget"', 'version = "1.4.2"']
        + [f'dep_{i} = ">={i}.0"' for i in range(120)]
    )
    lock = "\n".join(
        [f'[[package]]\nname = "lib{i}"\nversion = "{i}.0.0"' for i in range(150)]
        + ['[[package]]\nname = "widget"\nversion = "1.5.0"']
    )
    conflict: list[dict[str, Any]] = [
        {"role": "user", "content": "What version of widget is actually installed?"},
        {
            "role": "assistant",
            "content": [
                _tool_use("toolu_01", "Read", {"file_path": "pyproject.toml"}),
                _tool_use("toolu_02", "Read", {"file_path": "uv.lock"}),
            ],
        },
        {
            "role": "user",
            "content": [_tool_result("toolu_01", pyproject), _tool_result("toolu_02", lock)],
        },
    ]
    fx.append(
        Fixture(
            "conflicting_evidence",
            "conflict",
            conflict,
            ['version = "1.4.2"', 'name = "widget"\nversion = "1.5.0"'],
            2,
            notes="declared vs locked version disagree; both must survive",
        )
    )

    # Cache-hot turn: an earlier, already-cached turn (cache_control) followed by a new delta.
    hist_log = _service_log()
    cache_hot: list[dict[str, Any]] = [
        {"role": "user", "content": "Check the service log."},
        {
            "role": "assistant",
            "content": [_tool_use("toolu_01", "Bash", {"command": "tail service.log"})],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01",
                    "content": hist_log,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "The DB pool refused connections."},
                _tool_use("toolu_02", "Bash", {"command": "pytest -q"}),
            ],
        },
        {"role": "user", "content": [_tool_result("toolu_02", _pytest_output())]},
    ]
    fx.append(
        Fixture(
            "cache_hot_turn",
            "cache",
            cache_hot,
            ["ERROR [db.pool] connection refused: 10.0.3.17:5432", "2 failed, 298 passed"],
            2,
            frozen_message_count=3,
            notes="first 3 messages are a cached prefix and must be byte-identical",
        )
    )

    # Native streamed turn (Codex Responses shape, stream=true) -- Rust live zone only.
    out = _search_output()
    streamed = _responses_body_for(
        "Find where TokenBudgetExceeded is raised.", "shell", {"cmd": "rg Budget"}, out
    )
    streamed["input"].insert(
        1,
        {
            "type": "reasoning",
            "id": "rs_01",
            "summary": [],
            "encrypted_content": "gAAAAB" + "x" * 64,
        },
    )
    fx.append(
        Fixture(
            "native_streamed_turn",
            "streaming",
            [],
            ["src/core/budget.py:212:    raise TokenBudgetExceeded(limit)"],
            1,
            responses_body=streamed,
            notes="stream=true Responses body with reasoning item; ids/stream flag must survive",
        )
    )
    return fx


# --------------------------------------------------------------------------
# Measurement helpers
# --------------------------------------------------------------------------


@dataclass
class PathResult:
    path: str
    raw_tokens: int = 0
    forwarded_tokens: int = 0
    reduction_pct: float = 0.0
    must_keep_total: int = 0
    must_keep_forwarded: int = 0
    must_keep_recall: float = 0.0
    missing_facts: list[str] = field(default_factory=list)
    missing_recoverable_via_ccr: int = 0
    retrieval_tool_turns: int = 0
    ccr_markers: int = 0
    frozen_prefix_identical: bool | None = None
    structure_valid: bool = True
    transforms: list[str] = field(default_factory=list)
    wall_ms: float = 0.0
    error: str | None = None


def _texts_from_anthropic(messages: list[dict[str, Any]]) -> str:
    out: list[str] = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list):
            for b in c:
                if not isinstance(b, dict):
                    continue
                if isinstance(b.get("text"), str):
                    out.append(b["text"])
                inner = b.get("content")
                if isinstance(inner, str):
                    out.append(inner)
                elif isinstance(inner, list):
                    out += [x.get("text", "") for x in inner if isinstance(x, dict)]
                if b.get("type") == "tool_use":
                    out.append(json.dumps(b.get("input", {}), sort_keys=True))
    return "\n".join(out)


def _texts_from_responses(body: dict[str, Any]) -> str:
    out: list[str] = [str(body.get("instructions", ""))]
    for item in body.get("input", []):
        t = item.get("type")
        if t == "message":
            out += [p.get("text", "") for p in item.get("content", []) if isinstance(p, dict)]
        elif t == "function_call":
            out.append(str(item.get("arguments", "")))
        elif t == "function_call_output":
            o = item.get("output")
            out.append(o if isinstance(o, str) else json.dumps(o))
    return "\n".join(out)


def _ccr_lookup(text: str) -> tuple[int, str]:
    """Return (marker count, concatenated originals retrievable from the CCR store)."""
    hashes = sorted(set(_CCR_HASH_RE.findall(text)))
    if not hashes:
        return 0, ""
    from headroom.cache.compression_store import get_compression_store

    store = get_compression_store()
    recovered: list[str] = []
    for h in hashes:
        entry = store.retrieve(h)
        if entry is not None:
            recovered.append(str(getattr(entry, "original_content", "") or ""))
    return len(hashes), "\n".join(recovered)


def _score_facts(res: PathResult, must_keep: list[str], forwarded_text: str) -> None:
    res.must_keep_total = len(must_keep)
    present = [f for f in must_keep if f in forwarded_text]
    res.must_keep_forwarded = len(present)
    res.must_keep_recall = round(len(present) / len(must_keep), 4) if must_keep else 1.0
    res.missing_facts = [f for f in must_keep if f not in forwarded_text]
    markers, recovered = _ccr_lookup(forwarded_text)
    res.ccr_markers = markers
    res.missing_recoverable_via_ccr = sum(1 for f in res.missing_facts if f in recovered)
    # One headroom_retrieve turn is needed if any missing fact is recoverable.
    res.retrieval_tool_turns = 1 if res.missing_recoverable_via_ccr else 0


def _pct(raw: int, fwd: int) -> float:
    return round(100.0 * (raw - fwd) / raw, 2) if raw else 0.0


# --------------------------------------------------------------------------
# Real Headroom paths
# --------------------------------------------------------------------------


class PythonProxyPath:
    """Drives ``HeadroomProxy.anthropic_pipeline.apply`` like the Anthropic handler."""

    name = "python_proxy_anthropic"

    def __init__(self) -> None:
        from headroom.agent_savings import (
            DEFAULT_PROFILE,
            apply_agent_savings_env_defaults,
            get_agent_savings_profile,
            proxy_pipeline_kwargs,
        )
        from headroom.proxy.models import ProxyConfig
        from headroom.proxy.server import HeadroomProxy
        from headroom.transforms.compression_policy import resolve_policy
        from headroom.utils import extract_user_query

        profile = get_agent_savings_profile(DEFAULT_PROFILE)
        apply_agent_savings_env_defaults(os.environ, profile)
        self.config = ProxyConfig(
            **self._extra_config(),
            savings_profile=profile.name,
            mode=profile.proxy_mode,
            code_aware_enabled=profile.code_aware,
            stateless=True,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
        )
        self.profile = profile.name
        self.proxy = HeadroomProxy(self.config)
        self._kwargs = proxy_pipeline_kwargs
        self._policy = resolve_policy
        self._query = extract_user_query
        self.tokenizer = self.proxy.anthropic_pipeline._get_tokenizer(MODEL)

    @staticmethod
    def _extra_config() -> dict[str, Any]:
        return {}

    def run(self, fx: Fixture) -> PathResult:
        res = PathResult(self.name)
        messages = copy.deepcopy(fx.messages)
        original = copy.deepcopy(fx.messages)
        res.raw_tokens = self.tokenizer.count_messages(original)
        t0 = time.perf_counter()
        try:
            out = self.proxy.anthropic_pipeline.apply(
                messages=messages,
                model=MODEL,
                model_limit=MODEL_LIMIT,
                context=self._query(messages),
                frozen_message_count=fx.frozen_message_count,
                compression_policy=self._policy(None),
                request_id=f"unified21-{fx.name}",
                **self._kwargs(self.config),
            )
            forwarded = out.messages
            res.transforms = list(out.transforms_applied)
        except Exception as exc:  # recorded, never hidden
            forwarded = original
            res.error = f"{type(exc).__name__}: {exc}"
        res.wall_ms = round((time.perf_counter() - t0) * 1000, 2)
        res.forwarded_tokens = self.tokenizer.count_messages(forwarded)
        res.reduction_pct = _pct(res.raw_tokens, res.forwarded_tokens)
        if fx.frozen_message_count:
            n = fx.frozen_message_count
            res.frozen_prefix_identical = json.dumps(forwarded[:n], sort_keys=True) == json.dumps(
                original[:n], sort_keys=True
            )
        res.structure_valid = _anthropic_structure_ok(original, forwarded)
        _score_facts(res, fx.must_keep, _texts_from_anthropic(forwarded))
        return res


class PythonProxyWrapPath(PythonProxyPath):
    """The same pipeline with the rollout ``headroom wrap`` launches (Phase 4+:
    tool admission on). Kept separate so the bare-proxy series stays comparable."""

    name = "python_proxy_anthropic_wrap"

    @staticmethod
    def _extra_config() -> dict[str, Any]:
        from headroom.rollout import apply_wrap_feature_defaults, resolve_rollout

        return {"rollout": resolve_rollout(apply_wrap_feature_defaults({}))}


def _anthropic_structure_ok(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> bool:
    if len(before) != len(after):
        return False
    for a, b in zip(before, after):
        if a.get("role") != b.get("role"):
            return False
        ids_a = (
            [
                x.get("id") or x.get("tool_use_id")
                for x in a.get("content", [])
                if isinstance(x, dict)
            ]
            if isinstance(a.get("content"), list)
            else []
        )
        ids_b = (
            [
                x.get("id") or x.get("tool_use_id")
                for x in b.get("content", [])
                if isinstance(x, dict)
            ]
            if isinstance(b.get("content"), list)
            else []
        )
        if ids_a != ids_b:
            return False
    return True


class RustLiveZonePath:
    """Drives ``headroom._core.compress_openai_responses_live_zone``."""

    name = "rust_live_zone_responses"

    def __init__(self) -> None:
        from headroom._core import compress_openai_responses_live_zone
        from headroom.tokenizers.registry import get_tokenizer

        self._compress: Callable[..., Any] = compress_openai_responses_live_zone
        self.tokenizer = get_tokenizer(RESPONSES_MODEL)

    def run(self, fx: Fixture) -> PathResult | None:
        if fx.responses_body is None:
            return None
        res = PathResult(self.name)
        raw = json.dumps(fx.responses_body).encode()
        res.raw_tokens = self.tokenizer.count_text(_texts_from_responses(fx.responses_body))
        t0 = time.perf_counter()
        body = fx.responses_body
        try:
            out, modified, _saved, transforms, reason = self._compress(raw, "payg", RESPONSES_MODEL)
            body = json.loads(bytes(out))
            res.transforms = list(transforms) + (
                [f"no_change:{reason}"] if reason and not modified else []
            )
        except Exception as exc:
            res.error = f"{type(exc).__name__}: {exc}"
        res.wall_ms = round((time.perf_counter() - t0) * 1000, 2)
        res.forwarded_tokens = self.tokenizer.count_text(_texts_from_responses(body))
        res.reduction_pct = _pct(res.raw_tokens, res.forwarded_tokens)
        res.structure_valid = _responses_structure_ok(fx.responses_body, body)
        _score_facts(res, fx.must_keep, _texts_from_responses(body))
        return res


def _responses_structure_ok(before: dict[str, Any], after: dict[str, Any]) -> bool:
    if before.get("stream") != after.get("stream") or before.get("model") != after.get("model"):
        return False
    a, b = before.get("input", []), after.get("input", [])
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x.get("type") != y.get("type") or x.get("call_id") != y.get("call_id"):
            return False
        if x.get("type") == "reasoning" and x != y:  # encrypted reasoning must be byte-stable
            return False
    return before.get("tools") == after.get("tools")


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def _environment(tokenizers: dict[str, str]) -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "offline": os.environ.get("HEADROOM_OFFLINE"),
        "stateless": os.environ.get("HEADROOM_STATELESS"),
        "tokenizers": tokenizers,
    }
    try:
        from headroom._version import __version__

        env["headroom_version"] = __version__
    except Exception:
        env["headroom_version"] = None
    return env


def _percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile (no interpolation: every value was measured)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * int(pct) // 100))
    return round(ordered[rank - 1], 2)


def _aggregate(rows: list[PathResult]) -> dict[str, Any]:
    raw = sum(r.raw_tokens for r in rows)
    fwd = sum(r.forwarded_tokens for r in rows)
    total = sum(r.must_keep_total for r in rows)
    kept = sum(r.must_keep_forwarded for r in rows)
    recoverable = sum(r.missing_recoverable_via_ccr for r in rows)
    return {
        "fixtures": len(rows),
        "raw_tokens": raw,
        "forwarded_tokens": fwd,
        "reduction_pct": _pct(raw, fwd),
        "must_keep_recall": round(kept / total, 4) if total else 1.0,
        "must_keep_recall_with_ccr": round((kept + recoverable) / total, 4) if total else 1.0,
        "retrieval_tool_turns": sum(r.retrieval_tool_turns for r in rows),
        "structure_failures": sum(1 for r in rows if not r.structure_valid),
        "frozen_prefix_violations": sum(1 for r in rows if r.frozen_prefix_identical is False),
        "errors": sum(1 for r in rows if r.error),
        "wall_ms": round(sum(r.wall_ms for r in rows), 2),
        "wall_ms_p50": _percentile([r.wall_ms for r in rows], 50),
        "wall_ms_p95": _percentile([r.wall_ms for r in rows], 95),
        # The first fixture of a path pays one-time process warm-up (parsers,
        # tokenizers, model probes); the warm figures exclude it.
        "wall_ms_warm_p50": _percentile([r.wall_ms for r in rows[1:]], 50),
        "wall_ms_warm_p95": _percentile([r.wall_ms for r in rows[1:]], 95),
    }


class AgentRuntimeSession:
    """The agent runtime + JevK5 advisor, installed as ``create_app`` installs it.

    Only the benchmark wraps ``advisor.choose`` to time each call; the advisor,
    its deadline and confidence floor, and the arbiter blend are untouched.
    """

    def __init__(self, config: Any, mode: str) -> None:
        from types import SimpleNamespace

        from headroom.agent.arbiter import reset_arbiter
        from headroom.agent.install import install_agent_runtime

        self.mode = mode
        self.fixture = ""
        self.calls: list[dict[str, Any]] = []
        reset_arbiter()
        self.app = SimpleNamespace(state=SimpleNamespace())
        runtime = install_agent_runtime(self.app, config)
        if runtime is None:
            raise RuntimeError("agent_runtime is not enabled by the wrap rollout")
        runtime.start()
        self.intelligence = self.app.state.agent_intelligence
        t0 = time.perf_counter()
        self.intelligence.activate()
        self.activation_ms = round((time.perf_counter() - t0) * 1000, 2)
        self.advisor = runtime.advisor
        inner = self.advisor.choose

        def timed_choose(family: str, *args: Any, **kwargs: Any) -> Any:
            t = time.perf_counter()
            advice = inner(family, *args, **kwargs)
            self.calls.append(
                {
                    "fixture": self.fixture,
                    "family": family,
                    "ms": round((time.perf_counter() - t) * 1000, 2),
                    "advice": advice is not None,
                    "confidence": None if advice is None else round(advice.confidence, 4),
                }
            )
            return advice

        self.advisor.choose = timed_choose

    def report(self) -> dict[str, Any]:
        from headroom.agent.arbiter import get_arbiter

        jev = get_arbiter().status().get("jev", {})
        called = len(self.calls)
        used = int(jev.get("applied", 0))
        return {
            "mode": self.mode,
            "advisor_health": self.intelligence.health(),
            "activation_ms": self.activation_ms,
            "advisor_stats": self.advisor.status()["stats"],
            "arbiter_jev": jev,
            "advice_requests": called,
            "advice_used": used,
            "acceptance_rate": round(used / called, 4) if called else None,
            "call_ms_p50": _percentile([c["ms"] for c in self.calls], 50),
            "call_ms_p95": _percentile([c["ms"] for c in self.calls], 95),
            "calls": self.calls,
        }

    def close(self) -> None:
        from headroom.agent.install import stop_agent_runtime

        stop_agent_runtime(self.app)


def run_eval(
    label: str, fixtures: list[Fixture] | None = None, *, jevk5: str | None = None
) -> dict[str, Any]:
    """Run every fixture through every available path.

    The offline/stateless defaults and the savings-profile env seeding are
    applied for the duration of the run only; the caller's environment is
    restored afterwards so in-process callers (tests) see no leaked settings.
    """
    saved_env = dict(os.environ)
    try:
        for k, v in _ENV_DEFAULTS.items():
            os.environ.setdefault(k, v)
        if jevk5 is not None:
            os.environ["HEADROOM_JEVK5"] = jevk5
        return _run_eval(label, fixtures if fixtures is not None else build_fixtures(), jevk5)
    finally:
        os.environ.clear()
        os.environ.update(saved_env)


def _run_eval(label: str, fixtures: list[Fixture], jevk5: str | None = None) -> dict[str, Any]:

    paths: list[Any] = []
    init_failures: dict[str, str] = {}
    for cls in (PythonProxyPath, PythonProxyWrapPath, RustLiveZonePath):
        try:
            paths.append(cls())
        except Exception as exc:
            init_failures[cls.name] = f"{type(exc).__name__}: {exc}"
    session: AgentRuntimeSession | None = None
    if jevk5 is not None:
        wrap = next((p for p in paths if isinstance(p, PythonProxyWrapPath)), None)
        if wrap is None:
            raise RuntimeError(f"--jevk5 needs the wrap path: {init_failures}")
        session = AgentRuntimeSession(wrap.config, jevk5)
    try:
        return _run_paths(label, fixtures, paths, init_failures, session)
    finally:
        if session is not None:
            session.close()


def _run_paths(
    label: str,
    fixtures: list[Fixture],
    paths: list[Any],
    init_failures: dict[str, str],
    session: AgentRuntimeSession | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    per_fixture: list[dict[str, Any]] = []
    by_path: dict[str, list[PathResult]] = {p.name: [] for p in paths}
    for fx in fixtures:
        entry: dict[str, Any] = {
            "name": fx.name,
            "category": fx.category,
            "tool_turns": fx.tool_turns,
            "must_keep": fx.must_keep,
            "notes": fx.notes,
            "paths": {},
        }
        for p in paths:
            if isinstance(p, PythonProxyPath) and not fx.messages:
                continue
            if session is not None:
                session.fixture = fx.name
            r = p.run(fx)
            if r is None:
                continue
            by_path[p.name].append(r)
            entry["paths"][p.name] = asdict(r)
        per_fixture.append(entry)

    known_failures = []
    for fx_entry in per_fixture:
        for pname, r in fx_entry["paths"].items():
            if r["error"] or not r["structure_valid"] or r["frozen_prefix_identical"] is False:
                known_failures.append(
                    {
                        "fixture": fx_entry["name"],
                        "path": pname,
                        "error": r["error"],
                        "structure_valid": r["structure_valid"],
                        "frozen_prefix_identical": r["frozen_prefix_identical"],
                    }
                )
            elif r["missing_facts"] and r["missing_recoverable_via_ccr"] < len(r["missing_facts"]):
                known_failures.append(
                    {
                        "fixture": fx_entry["name"],
                        "path": pname,
                        "unrecoverable_missing_facts": r["missing_facts"],
                    }
                )
    for pname, err in init_failures.items():
        known_failures.append({"path": pname, "init_error": err})

    tokenizers = {
        p.name: type(getattr(p.tokenizer, "_counter", p.tokenizer)).__name__ for p in paths
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "environment": _environment(tokenizers),
        "config": {
            "model": MODEL,
            "responses_model": RESPONSES_MODEL,
            "python_profile": getattr(
                next((p for p in paths if isinstance(p, PythonProxyPath)), None), "profile", None
            ),
        },
        "summary": {name: _aggregate(rows) for name, rows in by_path.items()},
        "fixtures": per_fixture,
        "known_failures": known_failures,
        "total_wall_ms": round((time.perf_counter() - started) * 1000, 2),
        **({"jevk5": session.report()} if session is not None else {}),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--jevk5",
        choices=("off", "on"),
        default=None,
        help="Install the agent runtime for the wrap path with HEADROOM_JEVK5=off|on.",
    )
    args = ap.parse_args(argv)
    report = run_eval(args.label, jevk5=args.jevk5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, s in report["summary"].items():
        print(
            f"{name}: reduction={s['reduction_pct']}% recall={s['must_keep_recall']} "
            f"recall_with_ccr={s['must_keep_recall_with_ccr']} errors={s['errors']}"
        )
    jev = report.get("jevk5")
    if jev:
        print(
            f"jevk5={jev['mode']} state={jev['advisor_health']['advisor_state']} "
            f"requests={jev['advice_requests']} used={jev['advice_used']} "
            f"call_p50={jev['call_ms_p50']}ms call_p95={jev['call_ms_p95']}ms"
        )
    print(f"known_failures={len(report['known_failures'])} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
