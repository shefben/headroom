"""Unified-21 Phase 4: envelope, storage sequence, family interceptors, deltas,
proxy live-zone fallback and the Claude Code host adapter (#4, #5, #14)."""

from __future__ import annotations

import json

import pytest

from headroom.agent.interfaces import ToolResultEnvelope, content_version_hash
from headroom.agent.tool_admission import host as host_adapter
from headroom.agent.tool_admission.delta import (
    ResourceVersionStore,
    compute_delta,
    version_for,
)
from headroom.agent.tool_admission.dispatcher import (
    AdmissionRequest,
    ToolAdmissionDispatcher,
)
from headroom.agent.tool_admission.families import (
    ToolFamily,
    build_envelope,
    classify,
    workspace_relative,
)
from headroom.agent.tool_admission.transform import ToolAdmissionTransform
from headroom.agent.tool_admission.views import ViewContext
from headroom.cache.compression_store import CompressionStore
from headroom.transforms.compression_policy import policy_default_payg

ROOT = "/work/repo"


# ---------------------------------------------------------------------------
# Fixtures: realistic tool outputs for the seven families
# ---------------------------------------------------------------------------


def python_source(n: int = 60) -> str:
    lines = ["import hashlib", "import os", "", "MAX_RETRIES = 7", ""]
    lines += [
        "def compute_checksum(payload: bytes) -> str:",
        '    """Hex digest of the payload."""',
        "    data = bytes(payload)",
        "    return hashlib.sha256(data).hexdigest()",
        "",
    ]
    for i in range(n):
        lines += [
            f"def helper_{i:03d}(value: int) -> int:",
            f"    total = value * {i}",
            "    for step in range(3):",
            "        total += step",
            "    return total",
            "",
        ]
    return "\n".join(lines)


def numbered(text: str) -> str:
    return "\n".join(f"{i + 1:6d}\t{line}" for i, line in enumerate(text.split("\n")))


def search_output() -> str:
    rows = [f"src/module_{i % 9}/file_{i:04d}.py:{i % 700 + 1}:    budget = Budget(limit={i})" for i in range(400)]
    rows.insert(211, "src/core/budget.py:212:    raise TokenBudgetExceeded(limit)")
    return "\n".join(rows)


def pytest_output(failed: tuple[str, ...] = ("tests/test_parser.py::test_unicode_escape",)) -> str:
    lines = ["============================= test session starts =============================", "collected 300 items", ""]
    for i in range(300):
        tid = f"tests/test_mod_{i % 20:02d}.py::test_case_{i:04d}"
        lines.append(f"{tid} PASSED [{i * 100 // 300:3d}%]")
    for tid in failed:
        lines.append(f"{tid} FAILED")
    lines += ["", "================================== FAILURES ==================================="]
    for tid in failed:
        lines += [
            f"_____ {tid.split('::')[1]} _____",
            ">       assert decode('\\xe9') == 'e'",
            "E       AssertionError: assert 'x' == 'e'",
            f"{tid.split('::')[0]}:42: AssertionError",
        ]
    lines.append("=========================== short test summary info ============================")
    for tid in failed:
        lines.append(f"FAILED {tid} - AssertionError: assert 'x' == 'e'")
    lines.append(f"======================== {len(failed)} failed, {300 - len(failed)} passed in 41.27s ========================")
    return "\n".join(lines)


def build_log() -> str:
    lines = [f"[{i:04d}/1200] Compiling crate_{i} v0.{i}.0" for i in range(400)]
    lines.insert(300, "error[E0308]: mismatched types")
    lines.insert(301, "  --> src/net/client.rs:88:17")
    lines.append("error: could not compile `netclient` (lib) due to 1 previous error")
    return "\n".join(lines)


def json_records(error_row: int = 137) -> str:
    rows = []
    for i in range(200):
        row = {"id": i, "latency_ms": 20 + i % 9, "region": "us-east-1", "status": "ok"}
        if i == error_row:
            row.update(status="failed", error_code="E_QUOTA_4291", region="eu-west-2", latency_ms=9001)
        rows.append(row)
    return json.dumps(rows, indent=2)


def git_diff(n_files: int = 30) -> str:
    out = []
    for f in range(n_files):
        out += [
            f"diff --git a/src/pkg/mod_{f:02d}.py b/src/pkg/mod_{f:02d}.py",
            "index 1111111..2222222 100644",
            f"--- a/src/pkg/mod_{f:02d}.py",
            f"+++ b/src/pkg/mod_{f:02d}.py",
            f"@@ -{f * 10 + 1},7 +{f * 10 + 1},7 @@",
            *(f"     unchanged_line_{f}_{k} = {k}" for k in range(3)),
            f"-    label = 'old_{f}'",
            f"+    label = 'new_{f}'",
            *(f"     unchanged_line_{f}_{k} = {k}" for k in range(3, 6)),
        ]
    return "\n".join(out)


def git_status(n: int = 120) -> str:
    lines = ["On branch main", "Changes not staged for commit:"]
    lines += [f" M src/pkg/file_{i:03d}.py" for i in range(n)]
    lines += [f"?? scratch/new_{i}.txt" for i in range(5)]
    return "\n".join(lines)


def file_list(n: int = 300) -> str:
    return "\n".join(f"src/pkg_{i % 12}/sub_{i % 3}/file_{i:04d}.py" for i in range(n))


@pytest.fixture
def store():
    return CompressionStore(max_entries=500)


@pytest.fixture
def dispatcher():
    return ToolAdmissionDispatcher(versions=ResourceVersionStore())


def request(store, query="", tool_input=None, scope="ws|s1", **kw):
    from headroom.transforms.safety_gate import task_payload

    task = task_payload(None, fallback_text=query)
    ctx = ViewContext(
        query=query,
        entities=tuple(task.get("exact_entities") or ()),
        working_set=kw.pop("working_set", frozenset()),
        root=ROOT,
        cwd=ROOT,
        tool_input=tool_input or {},
        applyable_patch=kw.pop("applyable_patch", False),
    )
    return AdmissionRequest(
        scope=scope, context=ctx, task=task, policy=policy_default_payg(), store=store, **kw
    )


def admit(dispatcher, store, tool, args, content, *, query="", call="c1", **kw):
    env = build_envelope(
        tool_name=tool, tool_input=args, content=content, call_id=call, root=ROOT, cwd=ROOT,
        partial_reason=kw.pop("partial_reason", None),
    )
    return env, dispatcher.admit(env, request(store, query, args, **kw))


def assert_exact_recovery(store, outcome, original):
    assert outcome.ccr_hash and f"hash={outcome.ccr_hash}" in outcome.text
    assert store.retrieve(outcome.ccr_hash).original_content == original


# ---------------------------------------------------------------------------
# Envelope and resource identity
# ---------------------------------------------------------------------------


def test_resource_id_is_stable_across_calls_and_not_the_call_id():
    a = build_envelope(tool_name="Read", tool_input={"file_path": f"{ROOT}/src/a.py"}, content="x", call_id="c1", root=ROOT)
    b = build_envelope(tool_name="Read", tool_input={"file_path": "src/a.py"}, content="y", call_id="c2", root=ROOT, cwd=ROOT)
    assert a.resource_id == b.resource_id and a.call_id != b.call_id
    assert a.version_hash != b.version_hash
    ranged = build_envelope(tool_name="Read", tool_input={"file_path": "src/a.py", "offset": 10, "limit": 5}, content="x", root=ROOT, cwd=ROOT)
    assert ranged.resource_id != a.resource_id and ranged.is_partial and ranged.truncation_reason == "range_read"


def test_windows_paths_fold_case_and_separators():
    root = "C:/Work/Repo"
    assert workspace_relative("c:\\work\\repo\\src\\A.py", root=root) == "src\\A.py".replace("\\", "/")
    assert workspace_relative("D:/elsewhere/x.py", root=root).startswith("ext:")


def test_command_grep_and_api_identities():
    one = build_envelope(tool_name="Bash", tool_input={"command": "pytest  -q tests"}, content="", root=ROOT, cwd=ROOT)
    two = build_envelope(tool_name="Bash", tool_input={"command": "pytest -q tests"}, content="", root=ROOT, cwd=ROOT)
    other_cwd = build_envelope(tool_name="Bash", tool_input={"command": "pytest -q tests"}, content="", root=ROOT, cwd=f"{ROOT}/sub")
    assert one.resource_id == two.resource_id != other_cwd.resource_id
    assert one.tool_family == "test"
    g1 = build_envelope(tool_name="Grep", tool_input={"pattern": "Budget", "path": "src"}, content="", root=ROOT, cwd=ROOT)
    g2 = build_envelope(tool_name="Grep", tool_input={"pattern": "Budget", "path": "lib"}, content="", root=ROOT, cwd=ROOT)
    assert g1.resource_id != g2.resource_id and g1.resource_id.startswith("search:")
    api = build_envelope(tool_name="mcp__db__query", tool_input={"sql": "select 1"}, content="[1]", root=ROOT)
    assert api.resource_id.startswith("api:")


@pytest.mark.parametrize(
    ("tool", "args", "content", "family"),
    [
        ("Read", {"file_path": "a.py"}, "x", ToolFamily.READ),
        ("Grep", {"pattern": "x"}, "", ToolFamily.SEARCH),
        ("Glob", {"pattern": "**/*.py"}, "", ToolFamily.LISTING),
        ("Bash", {"command": "cargo build"}, "Compiling", ToolFamily.SHELL),
        ("Bash", {"command": "cd x && python -m pytest -q"}, "", ToolFamily.TEST),
        ("Bash", {"command": "git --no-pager diff HEAD~1"}, "", ToolFamily.GIT),
        ("Bash", {"command": "rg -n Budget src"}, "", ToolFamily.SEARCH),
        ("PowerShell", {"command": "Get-ChildItem -Recurse"}, "", ToolFamily.LISTING),
        ("Bash", {"command": "curl -s localhost/api"}, '{"a": 1}', ToolFamily.JSON),
        ("mcp__headroom__headroom_retrieve", {"hash": "a"}, '{"a": 1}', ToolFamily.OTHER),
    ],
)
def test_family_classification(tool, args, content, family):
    assert classify(tool, args, content) is family


def test_envelope_partiality_and_v1_compat():
    piped = build_envelope(tool_name="Bash", tool_input={"command": "pytest -q | head -50"}, content="ok", root=ROOT)
    assert piped.is_partial and piped.partial_reason == "pipe_truncated" and not piped.is_complete
    limited = build_envelope(tool_name="Grep", tool_input={"pattern": "x", "head_limit": 20}, content="a", root=ROOT)
    assert limited.partial_reason == "result_limit:head_limit"
    meta = piped.to_metadata()
    assert "content" not in meta and "stderr" not in meta
    v1_fields = (
        "tool_call_id", "tool_name", "byte_length", "line_count", "is_error", "is_partial",
        "partial_reason", "source_truncated", "source_byte_length", "exit_code",
    )
    v1 = {k: piped.to_dict()[k] for k in v1_fields}
    v1["v"] = 1
    restored = ToolResultEnvelope.from_dict(v1)
    assert restored.v == 2 and restored.tool_family == "other" and restored.content == ""
    assert ToolResultEnvelope.from_dict(piped.to_dict()) == piped


# ---------------------------------------------------------------------------
# Seven families through the one dispatcher
# ---------------------------------------------------------------------------


def test_read_outline_keeps_own_line_numbers_and_named_symbol(dispatcher, store):
    src = numbered(python_source())
    _, out = admit(dispatcher, store, "Read", {"file_path": f"{ROOT}/src/payload.py"}, src,
                   query="What does compute_checksum return? Check MAX_RETRIES.")
    assert out.stage == "admitted" and out.family == "read", out
    assert "     6\tdef compute_checksum(payload: bytes) -> str:" in out.text
    assert "return hashlib.sha256(data).hexdigest()" in out.text  # named body whole
    assert "     4\tMAX_RETRIES = 7" in out.text
    assert "def helper_059(value: int) -> int:" in out.text
    assert "total += step" not in out.text  # unnamed bodies elided
    assert_exact_recovery(store, out, src)


def test_read_edit_target_range_read_and_small_stay_exact(dispatcher, store):
    src = numbered(python_source())
    args = {"file_path": f"{ROOT}/src/payload.py"}
    _, out = admit(dispatcher, store, "Read", args, src, working_set=frozenset({"src/payload.py"}))
    assert out.text is None and out.stage == "edit_target_exact"
    _, out = admit(dispatcher, store, "Read", {**args, "offset": 1, "limit": 2000}, src, call="c2")
    assert out.text is None
    _, out = admit(dispatcher, store, "Read", args, numbered(python_source(3)), call="c3")
    assert out.text is None and out.stage == "small"


def test_read_reread_after_lossy_view_is_exact(dispatcher, store):
    src = numbered(python_source())
    args = {"file_path": f"{ROOT}/src/payload.py"}
    _, first = admit(dispatcher, store, "Read", args, src, call="r1")
    assert first.admitted
    _, again = admit(dispatcher, store, "Read", args, src, call="r2")
    assert again.text is None and again.stage == "progressive_disclosure"


def test_search_grouped_with_queried_match_first(dispatcher, store):
    out_text = search_output()
    _, out = admit(dispatcher, store, "Grep", {"pattern": "Budget"}, out_text,
                   query="Find where TokenBudgetExceeded is raised.")
    assert out.stage == "admitted" and out.family == "search"
    assert "src/core/budget.py:212:    raise TokenBudgetExceeded(limit)" in out.text
    assert "grouped by file" in out.text
    assert len(out.text) < len(out_text) / 3
    assert_exact_recovery(store, out, out_text)


def test_listing_grouped_and_partial_never_claims_all(dispatcher, store):
    listing = file_list()
    _, out = admit(dispatcher, store, "Glob", {"pattern": "**/*.py"}, listing,
                   partial_reason="client_truncated")
    assert out.admitted and "grouped by directory" in out.text
    assert "PARTIAL" in out.text and "not all files" in out.text
    assert_exact_recovery(store, out, listing)


def test_shell_log_keeps_errors_and_boundaries(dispatcher, store):
    log = build_log()
    _, out = admit(dispatcher, store, "Bash", {"command": "cargo build"}, log + "\nexit code: 101")
    assert out.admitted and out.family == "shell"
    for needle in ("error[E0308]: mismatched types", "src/net/client.rs:88:17",
                   "could not compile `netclient`", "exit code: 101"):
        assert needle in out.text
    assert "[0001/1200] Compiling crate_1" not in out.text or len(out.text) < len(log) / 2


def test_test_family_keeps_failures_counts_and_assertions(dispatcher, store):
    out_text = pytest_output(("tests/test_parser.py::test_unicode_escape", "tests/test_cache.py::test_eviction_order"))
    _, out = admit(dispatcher, store, "Bash", {"command": "pytest -q"}, out_text)
    assert out.admitted and out.family == "test"
    for needle in ("FAILED tests/test_parser.py::test_unicode_escape",
                   "FAILED tests/test_cache.py::test_eviction_order",
                   "2 failed, 298 passed", "AssertionError"):
        assert needle in out.text
    assert_exact_recovery(store, out, out_text)


def test_git_diff_hunks_and_applyable_patch_stays_exact(dispatcher, store):
    diff = git_diff()
    _, out = admit(dispatcher, store, "Bash", {"command": "git diff"}, diff, query="what changed in mod_07?")
    assert out.admitted and out.family == "git"
    assert "diff --git a/src/pkg/mod_07.py b/src/pkg/mod_07.py" in out.text
    assert "+    label = 'new_7'" in out.text
    _, exact = admit(dispatcher, store, "Bash", {"command": "git diff"}, diff, call="c2",
                     applyable_patch=True, scope="other")
    assert exact.text is None


def test_git_status_keeps_every_path(dispatcher, store):
    status = git_status()
    _, out = admit(dispatcher, store, "Bash", {"command": "git status --porcelain"}, status)
    assert out.admitted
    for i in range(40):
        assert f"src/pkg/file_{i:03d}.py" in out.text
    assert "scratch/new_4.txt" in out.text


def test_json_keyed_rows_keep_error_fields(dispatcher, store):
    data = json_records()
    _, out = admit(dispatcher, store, "mcp__db__query", {"sql": "select * from records"}, data,
                   query="Which record failed and with what error code?")
    assert out.admitted and out.family == "json"
    for needle in ("E_QUOTA_4291", "eu-west-2", "9001"):
        assert needle in out.text
    assert_exact_recovery(store, out, data)


# ---------------------------------------------------------------------------
# Storage sequence failures forward the original
# ---------------------------------------------------------------------------


class _BrokenSelfTest(CompressionStore):
    def verify_exact(self, hash_key, original):  # noqa: ARG002
        return False


class _FailingStore(CompressionStore):
    def store(self, *a, **k):  # noqa: ARG002
        raise OSError("disk full")


def test_selftest_failure_store_failure_and_no_store_forward_original(dispatcher):
    data = pytest_output()
    for st, stage in ((_BrokenSelfTest(), "ccr_selftest_failed"), (_FailingStore(), "ccr_store_failed"), (None, "no_ccr_store")):
        _, out = admit(dispatcher, st, "Bash", {"command": "pytest"}, data, call=stage)
        assert out.text is None and out.stage == stage


def test_interceptor_crash_forwards_original(store):
    class Boom:
        family = ToolFamily.TEST

        def view(self, envelope, ctx):
            raise RuntimeError("boom")

    d = ToolAdmissionDispatcher(interceptors={ToolFamily.TEST: Boom()})
    _, out = admit(d, store, "Bash", {"command": "pytest"}, pytest_output())
    assert out.text is None and out.stage == "error"


def test_gate_rejection_forwards_original(dispatcher, store):
    # A partial (piped) test run: a view that dropped the truncation note would
    # misrepresent completeness; the family view keeps it, and an empty
    # policy makes the gate reject everything -> original.
    env = build_envelope(tool_name="Bash", tool_input={"command": "pytest"}, content=pytest_output(), call_id="g1", root=ROOT)
    req = request(store)
    req.policy = None  # unknown limits fail to the original
    out = dispatcher.admit(env, req)
    assert out.text is None and out.stage == "gate_rejected" and "policy_unknown" in out.reasons


# ---------------------------------------------------------------------------
# Delta (#5)
# ---------------------------------------------------------------------------


def _shell_text(n=200, changed=()):
    lines = [f"row {i}: value={i * 3}" for i in range(n)]
    for i in changed:
        lines[i] = f"row {i}: value=CHANGED"
    return "\n".join(lines)


def test_line_delta_counts_and_hashes(dispatcher, store):
    args = {"command": "./report.sh --all"}
    base_env, first = admit(dispatcher, store, "Bash", args, _shell_text(), call="d1")
    cur_text = _shell_text(changed=(10, 150)) + "\nrow 200: value=600"
    cur_env, out = admit(dispatcher, store, "Bash", args, cur_text, call="d2")
    assert out.admitted and out.delta is not None, out
    d = out.delta
    assert (d.changed, d.added, d.removed) == (2, 1, 0)
    assert d.unchanged == 198
    assert d.base_hash == base_env.version_hash and d.current_hash == cur_env.version_hash
    assert f"base version {base_env.version_hash[:12]}" in out.text
    assert "+ row 10: value=CHANGED" in out.text and "[- row 10: value=30]" in out.text
    assert_exact_recovery(store, out, cur_text)


def test_unchanged_reread_is_reference_only_after_exact_recovery(dispatcher, store):
    args = {"command": "./report.sh --all"}
    admit(dispatcher, store, "Bash", args, _shell_text(), call="u1")
    _, out = admit(dispatcher, store, "Bash", args, _shell_text(), call="u2")
    assert out.delta is not None and out.delta.reference_only
    assert "unchanged since an earlier" in out.text and len(out.text) < 400
    assert store.verify_exact(out.ccr_hash, _shell_text())


def test_json_row_delta(dispatcher, store):
    args = {"sql": "select * from records"}
    admit(dispatcher, store, "mcp__db__query", args, json_records(), call="j1")
    rows = json.loads(json_records())
    rows[5]["status"] = "degraded"
    rows.append({"id": 999, "latency_ms": 1, "region": "ap-south-1", "status": "ok"})
    rows = [r for r in rows if r["id"] != 7]
    _, out = admit(dispatcher, store, "mcp__db__query", args, json.dumps(rows, indent=2), call="j2")
    assert out.delta is not None, out
    assert (out.delta.changed, out.delta.added, out.delta.removed) == (1, 1, 1)
    assert '"status":"degraded"' in out.text and "ap-south-1" in out.text


def test_search_and_listing_set_deltas(dispatcher, store):
    args = {"pattern": "Budget"}
    admit(dispatcher, store, "Grep", args, search_output(), call="s1")
    new = search_output() + "\nsrc/new/thing.py:3:    budget = Budget(limit=-1)"
    _, out = admit(dispatcher, store, "Grep", args, new, call="s2")
    assert out.delta is not None and out.delta.added == 1 and out.delta.removed == 0
    largs = {"pattern": "**/*.py"}
    admit(dispatcher, store, "Glob", largs, file_list(), call="l1")
    files = file_list().split("\n")[1:] + ["src/added/new_file.py"]
    _, out = admit(dispatcher, store, "Glob", largs, "\n".join(files), call="l2")
    assert out.delta is not None and (out.delta.added, out.delta.removed) == (1, 1)
    assert "src/added/new_file.py" in out.text


def test_test_outcome_delta_keeps_summary(dispatcher, store):
    args = {"command": "pytest -q"}
    admit(dispatcher, store, "Bash", args, pytest_output(("tests/test_a.py::test_x",)), call="t1")
    now = pytest_output(("tests/test_b.py::test_y",))
    _, out = admit(dispatcher, store, "Bash", args, now, call="t2")
    assert out.admitted and "1 failed, 299 passed" in out.text
    assert "FAILED tests/test_b.py::test_y" in out.text


def test_partial_base_never_qualifies(store):
    env_partial = build_envelope(tool_name="Bash", tool_input={"command": "x | head"}, content=_shell_text(), root=ROOT)
    base = version_for(env_partial, ccr_hash="h", family=ToolFamily.SHELL)
    assert not base.complete
    cur = build_envelope(tool_name="Bash", tool_input={"command": "x"}, content=_shell_text(changed=(1,)), root=ROOT)
    assert compute_delta(ToolFamily.SHELL, cur, base, base_text=_shell_text(), ccr_hash="k", label="x") is None
    complete = version_for(cur, ccr_hash="h", family=ToolFamily.SHELL)
    assert compute_delta(ToolFamily.SHELL, env_partial, complete, base_text=None, ccr_hash="k", label="x") is None


def test_version_store_persists_through_ledger(tmp_path):
    from headroom.agent.ledger_db import AgentLedgerDB

    ledger = AgentLedgerDB(tmp_path / "agent_ledger.db")
    tasks = []
    vs = ResourceVersionStore()
    vs.bind(ledger, lambda fn: tasks.append(fn) or True)
    env = build_envelope(tool_name="Bash", tool_input={"command": "x"}, content=_shell_text(), call_id="p1", root=ROOT)
    vs.put("scope", version_for(env, ccr_hash="abc", family=ToolFamily.SHELL))
    for fn in tasks:
        fn()
    fresh = ResourceVersionStore()
    tasks.clear()
    fresh.bind(ledger, lambda fn: tasks.append(fn) or True)
    assert fresh.get("scope", env.resource_id) is None  # cold: load queued, not blocking
    for fn in tasks:
        fn()
    loaded = fresh.get("scope", env.resource_id)
    assert loaded is not None and loaded.version_hash == env.version_hash and loaded.complete
    ledger.close()


# ---------------------------------------------------------------------------
# Proxy live-zone fallback (no host hook)
# ---------------------------------------------------------------------------


class _Tok:
    def count_text(self, s):
        return max(1, len(s) // 4)

    def count_messages(self, ms):
        return len(json.dumps(ms)) // 4


def _conversation():
    return [
        {"role": "user", "content": "Which tests failed and why?"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_a", "name": "Bash", "input": {"command": "pytest -q"}},
            {"type": "tool_use", "id": "toolu_b", "name": "Grep", "input": {"pattern": "Budget"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_a", "content": pytest_output()},
            {"type": "tool_result", "tool_use_id": "toolu_b", "content": [{"type": "text", "text": search_output()}]},
            {"type": "tool_result", "tool_use_id": "toolu_orphan", "content": search_output()},
        ]},
    ]


@pytest.fixture
def global_store(monkeypatch):
    import headroom.cache.compression_store as cs

    s = CompressionStore(max_entries=500)
    monkeypatch.setattr(cs, "_compression_store", s)
    return s


def test_transform_admits_tail_and_is_byte_stable(global_store):
    msgs = _conversation()
    t = ToolAdmissionTransform(dispatcher=ToolAdmissionDispatcher(versions=ResourceVersionStore()))
    r1 = t.apply(json.loads(json.dumps(msgs)), _Tok(), context="Which tests failed?", frozen_message_count=0)
    blocks = r1.messages[2]["content"]
    assert "Retrieve more: hash=" in blocks[0]["content"]
    assert "Retrieve more: hash=" in blocks[1]["content"][0]["text"]
    assert blocks[2]["content"] == search_output()  # orphan: no tool identity, untouched
    assert sorted(r1.transforms_applied) == sorted(
        ["tool_admission:test:test_log", "tool_admission:search:search_grouped"]
    )
    assert r1.tokens_after < r1.tokens_before
    r2 = t.apply(json.loads(json.dumps(msgs)), _Tok(), context="Which tests failed?", frozen_message_count=0)
    assert json.dumps(r2.messages) == json.dumps(r1.messages)  # prefix-cache stable
    assert msgs == _conversation()  # input never mutated


def test_transform_never_touches_frozen_prefix(global_store):
    msgs = _conversation()
    t = ToolAdmissionTransform(dispatcher=ToolAdmissionDispatcher(versions=ResourceVersionStore()))
    r = t.apply(json.loads(json.dumps(msgs)), _Tok(), context="q", frozen_message_count=3)
    assert r.messages is not None and json.dumps(r.messages) == json.dumps(msgs)
    assert r.transforms_applied == []


def test_transform_openai_chat_shape(global_store):
    msgs = [
        {"role": "user", "content": "run the tests"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "shell", "arguments": json.dumps({"command": "pytest -q"})}}
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": pytest_output()},
    ]
    t = ToolAdmissionTransform(dispatcher=ToolAdmissionDispatcher(versions=ResourceVersionStore()))
    r = t.apply(msgs, _Tok(), context="run the tests", frozen_message_count=0)
    assert "Retrieve more: hash=" in r.messages[2]["content"]
    assert "FAILED tests/test_parser.py::test_unicode_escape" in r.messages[2]["content"]


def test_cross_conversation_base_is_not_used(global_store):
    d = ToolAdmissionDispatcher(versions=ResourceVersionStore())
    t = ToolAdmissionTransform(dispatcher=d)
    conv = lambda body: [  # noqa: E731
        {"role": "user", "content": "check"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "./report.sh"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": body}]},
    ]
    t.apply(conv(_shell_text()), _Tok(), context="check")
    # Same call id, same resource, different conversation: base bytes are not
    # visible here, so no "unchanged since earlier" claim and no delta.
    r = t.apply(conv(_shell_text(changed=(3,))), _Tok(), context="check")
    text = r.messages[2]["content"][0]["content"] if isinstance(r.messages[2]["content"][0]["content"], list) else r.messages[2]["content"][0]["content"]
    assert "headroom delta" not in str(text) and "unchanged since" not in str(text)


# ---------------------------------------------------------------------------
# Claude Code host adapter (shape-exact outputs)
# ---------------------------------------------------------------------------


class _Identity:
    workspace_id = "ws_test"
    session_id = "ss_test"
    root = ROOT
    trusted = True


def _post(tool, tool_input, response, **extra):
    return {
        "session_id": "conv-1",
        "transcript_path": "/tmp/t.jsonl",
        "cwd": ROOT,
        "hook_event_name": "PostToolUse",
        "tool_name": tool,
        "tool_input": tool_input,
        "tool_response": response,
        "tool_use_id": extra.pop("tool_use_id", "toolu_h1"),
        **extra,
    }


@pytest.fixture(autouse=True)
def _fresh_host(monkeypatch):
    import headroom.agent.tool_admission.dispatcher as disp

    monkeypatch.setattr(disp, "_DISPATCHER", ToolAdmissionDispatcher(versions=ResourceVersionStore()))
    host_adapter.reset()
    yield
    host_adapter.reset()


def test_host_bash_rewrite_keeps_output_shape(store):
    response = {"stdout": pytest_output(), "stderr": "", "interrupted": False, "isImage": False,
                "returnCodeInterpretation": "tests failed"}
    res = host_adapter.handle_hook(_post("Bash", {"command": "pytest -q"}, response), identity=_Identity(),
                                   query="which tests failed", store=store)
    out = res["output"]["hookSpecificOutput"]
    assert out["hookEventName"] == "PostToolUse"
    updated = out["updatedToolOutput"]
    assert set(updated) == set(response)
    assert {k: v for k, v in updated.items() if k != "stdout"} == {k: v for k, v in response.items() if k != "stdout"}
    assert len(updated["stdout"]) < len(response["stdout"]) / 3
    assert "FAILED tests/test_parser.py::test_unicode_escape" in updated["stdout"]


def test_host_grep_glob_and_mcp_shapes(store):
    files = file_list().split("\n")
    res = host_adapter.handle_hook(
        _post("Grep", {"pattern": "x"}, {"mode": "files_with_matches", "numFiles": len(files), "filenames": files}),
        identity=_Identity(), store=store)
    upd = res["output"]["hookSpecificOutput"]["updatedToolOutput"]
    assert upd["numFiles"] == len(files) and isinstance(upd["filenames"], list) and len(upd["filenames"]) < len(files)
    res = host_adapter.handle_hook(
        _post("Glob", {"pattern": "**/*.py"}, {"filenames": files, "durationMs": 3, "numFiles": len(files), "truncated": True}, tool_use_id="g2"),
        identity=_Identity(), store=store)
    upd = res["output"]["hookSpecificOutput"]["updatedToolOutput"]
    assert upd["truncated"] is True and any("PARTIAL" in f for f in upd["filenames"])
    res = host_adapter.handle_hook(
        _post("mcp__db__query", {"sql": "q"}, [{"type": "text", "text": json_records()}], tool_use_id="m1"),
        identity=_Identity(), query="error code", store=store)
    upd = res["output"]["hookSpecificOutput"]["updatedToolOutput"]
    assert isinstance(upd, list) and upd[0]["type"] == "text" and "E_QUOTA_4291" in upd[0]["text"]


def test_host_leaves_read_small_and_own_tools_alone(store):
    read = host_adapter.handle_hook(
        _post("Read", {"file_path": "a.py"}, {"type": "text", "file": {"content": python_source(), "numLines": 1, "startLine": 1, "totalLines": 1, "filePath": "a.py"}}),
        identity=_Identity(), store=store)
    assert read["output"] is None and read["stage"] == "left_to_proxy"
    small = host_adapter.handle_hook(_post("Bash", {"command": "ls"}, {"stdout": "a\nb", "stderr": "", "interrupted": False}), identity=_Identity(), store=store)
    assert small["output"] is None
    own = host_adapter.handle_hook(_post("mcp__headroom__headroom_retrieve", {"hash": "x"}, [{"type": "text", "text": pytest_output()}]), identity=_Identity(), store=store)
    assert own["output"] is None


def test_host_pre_tool_observes_edit_targets_without_output(store):
    pre = {"session_id": "conv-1", "hook_event_name": "PreToolUse", "tool_name": "Edit",
           "tool_input": {"file_path": f"{ROOT}/src/x.py", "old_string": "a", "new_string": "b"}, "cwd": ROOT}
    res = host_adapter.handle_hook(pre, identity=_Identity(), store=store)
    assert res["output"] is None and res["stage"] == "observed"
    assert "src/x.py" in host_adapter._working_set(f"{_Identity.workspace_id}|cc:conv-1")


def test_host_capabilities_are_honest():
    assert host_adapter.host_capability("claude") == host_adapter.HOST_PRE_CONTEXT
    assert host_adapter.host_capability("claude", hook_installed=False) == host_adapter.OBSERVATIONAL
    for client in ("codex", "opencode", "aider", "unknown-client"):
        assert host_adapter.host_capability(client) == "observational_or_proxy_only"


def test_content_version_hash_matches_envelope():
    env = build_envelope(tool_name="Bash", tool_input={"command": "x"}, content="abc", stderr="e", root=ROOT)
    assert env.version_hash == content_version_hash("abc", "e") != content_version_hash("abce")
