"""Phase 1B conformance: the requirements closed by the JevK5 conformance audit.

Each test names the plan requirement it proves. All run against the fake
llama-server (no GPU, no network) in an isolated workspace.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from headroom.intelligence import discovery
from headroom.intelligence import model as model_mod
from headroom.intelligence.advisor import DecisionAdvisor, RequestBudget, english_enough
from headroom.intelligence.discovery import LlamaBinary
from headroom.intelligence.manager import IntelligenceManager
from headroom.intelligence.process import ProcessInfo, matches_owned
from headroom.intelligence.service import BinaryLoadError, ServiceError, ServiceInfo
from headroom.intelligence.settings import IntelligenceSettings, ServicePolicy

from .fake_llama import FakeLlamaServer

PARCEL = ["delivered", "misdelivered", "unknown"]


@pytest.fixture(autouse=True)
def _workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "ws"))
    for name in (
        "HEADROOM_OFFLINE",
        "HEADROOM_STATELESS",
        "HEADROOM_JEVK5_LLAMA_URL",
        "HEADROOM_JEVK5_SYSTEMONE_URL",
        "HEADROOM_JEVK5_LLAMA_SERVER",
        "LLAMA_CPP_HOME",
    ):
        monkeypatch.delenv(name, raising=False)


def _advisor(url, **overrides):
    return DecisionAdvisor(IntelligenceSettings(**overrides), lambda: ("llama-server", url))


# -- confidence floor applies to cached answers (min confidence 0.60) ----------


def test_cached_low_confidence_answer_is_still_ignored():
    with FakeLlamaServer() as fake:
        advisor = _advisor(fake.url, min_confidence=0.999)
        for _ in range(3):
            assert advisor.choose("f", "s", "q?", PARCEL, budget=RequestBudget(4)) is None
        assert fake.cfg.completions == 1  # cached, not re-asked
        assert advisor.stats.cache_hits == 2
        assert advisor.stats.low_confidence == 3  # the floor held on every hit


# -- 4 calls per request, shared by every consumer of the request -------------


def test_request_budget_is_shared_across_consumers_of_one_request():
    with FakeLlamaServer() as fake:
        advisor = _advisor(fake.url)
        # Arbiter, catalog and effort routing each fetch the request's budget.
        arbiter = advisor.request_budget("req-1")
        catalog = advisor.request_budget("req-1")
        effort = advisor.request_budget("req-1")
        assert arbiter is catalog is effort
        for i in range(2):
            assert advisor.choose("transform_arbiter", f"a{i}", "q?", PARCEL, budget=arbiter)
        assert advisor.choose("tool_catalog", "c", "q?", PARCEL, budget=catalog)
        assert advisor.score("effort_complexity", "e", "how hard?", ["0", "1", "2"],
                             budget=effort) is not None
        assert advisor.choose("tool_catalog", "c2", "q?", PARCEL, budget=catalog) is None
        assert advisor.stats.budget_exhausted == 1 and fake.cfg.completions == 4
        # Another request has its own allowance; an unkeyed budget is private.
        assert advisor.request_budget("req-2") is not arbiter
        assert advisor.request_budget() is not advisor.request_budget()
        assert advisor.request_budget("req-2").remaining == 4


def test_request_budget_table_is_bounded_and_expires():
    now = [0.0]
    advisor = DecisionAdvisor(IntelligenceSettings(), lambda: None, clock=lambda: now[0])
    first = advisor.request_budget("r")
    first.take()
    now[0] += 301.0
    assert advisor.request_budget("r").remaining == 4  # expired -> fresh
    for i in range(5000):
        advisor.request_budget(f"k{i}")
    assert len(advisor._budgets) <= 4096


# -- English-only --------------------------------------------------------------


def test_non_english_decisions_are_not_sent():
    assert english_enough("Delivered to No. 17; the customer lives at No. 71.")
    assert english_enough("src/app.py:12 KeyError: 'x'", "Which file?")
    assert english_enough("Café résumé naïve façade déjà vu, the parcel arrived late.")
    assert not english_enough("包裹被送到了17号，但客户住在71号。请问发生了什么？")
    assert not english_enough("Посылка доставлена в дом 17, а клиент живёт в доме 71.")
    with FakeLlamaServer() as fake:
        advisor = _advisor(fake.url)
        result = advisor.choose(
            "parcel", "包裹被送到了17号，但客户住在71号。请问发生了什么事情？", "What happened?",
            PARCEL, budget=RequestBudget(4),
        )
        assert result is None and fake.cfg.completions == 0
        assert advisor.stats.non_english == 1 and advisor.stats.failures == 0


# -- 8192 effective context: an oversized state is refused, not a failure ------


def test_context_overflow_refuses_the_decision_without_tripping_the_breaker():
    with FakeLlamaServer(tokens_per_char=1.0) as fake:
        advisor = _advisor(fake.url, ctx_size=2048, max_state_tokens=6000)
        big = "word " * 900  # ~4500 tokens with this fake tokenizer > 2048
        for _ in range(4):
            assert advisor.noul("f", big, "q?", budget=RequestBudget(4)) is None
        assert advisor.stats.context_overflow == 4
        assert advisor.stats.failures == 0 and advisor.status()["breaker"] == "closed"
        assert fake.cfg.completions == 0  # refused at /tokenize, before any pass
        assert advisor.noul("f", "No receipt.", "Refund?", budget=RequestBudget(4)) is not None


# -- concurrency: results AND per-decision adapter metrics never bleed -------


def test_concurrent_decisions_keep_their_own_metrics_and_missing_letter_state():
    """Interleaved decisions of different sizes, some needing the n_probs retry.

    Each Advice carries the token counts read from its OWN adapter; a shared
    mutable adapter would mix ``last_pass_tokens``/``missing`` between threads.
    """
    with FakeLlamaServer(delay_s=0.02, missing_below=100, missing_marker="RETRYME",
                         tokens_per_char=0.25) as fake:
        advisor = _advisor(fake.url, decision_timeout_ms=10000)
        results: dict[int, object] = {}
        barrier = threading.Barrier(16)

        def run(i: int) -> None:
            state = ("RETRYME " if i % 3 == 0 else "") + ("detail " * (5 + 40 * i))
            barrier.wait()
            results[i] = advisor.choose(f"fam{i}", state, "What happened?", PARCEL,
                                        budget=RequestBudget(4))

        threads = [threading.Thread(target=run, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(20)
        assert len(results) == 16 and all(r is not None for r in results.values())
        tokens = {i: results[i].input_tokens for i in results}  # type: ignore[attr-defined]
        # Strictly increasing with the state size: no thread saw another's counts.
        ordered = [tokens[i] for i in range(16)]
        assert ordered == sorted(ordered) and len(set(ordered)) == 16
        for i, advice in results.items():
            assert advice.pass_tokens == advice.input_tokens  # one pass per decision
            assert advice.missing_retried is (i % 3 == 0)
            assert advice.value == "misdelivered"
        assert advisor.stats.missing_letter_retries == len([i for i in range(16) if i % 3 == 0])
        assert advisor.stats.missing_letter_failures == 0


# -- discovery: common build dirs after LLAMA_CPP_HOME ------------------------


def _exe_name() -> str:
    return "llama-server.exe" if sys.platform == "win32" else "llama-server"


def test_discovery_order_includes_common_install_dirs(tmp_path, monkeypatch):
    home = tmp_path / "home"
    common = home / "llama.cpp" / "build" / "bin"
    common.mkdir(parents=True)
    (common / _exe_name()).write_text("x")
    llama_home = tmp_path / "lch" / "build" / "bin"
    llama_home.mkdir(parents=True)
    (llama_home / _exe_name()).write_text("x")
    on_path = tmp_path / "pathbin"
    on_path.mkdir()
    (on_path / _exe_name()).write_text("x")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("LLAMA_CPP_HOME", str(tmp_path / "lch"))
    monkeypatch.setattr(discovery.shutil, "which",
                        lambda name: str(on_path / _exe_name()) if name == "llama-server" else None)
    found = discovery.candidate_binaries(None, None)
    sources = [source for _, source in found]
    assert sources[:3] == ["path", "LLAMA_CPP_HOME", "common-dir"]
    assert Path(found[2][0]).parent == common


def test_iter_llama_servers_is_lazy_and_stops_at_an_override(tmp_path, monkeypatch):
    probed: list[str] = []
    monkeypatch.setattr(discovery, "candidate_binaries",
                        lambda o, m: [("a", "override"), ("b", "path")])
    monkeypatch.setattr(discovery, "probe_binary",
                        lambda p, s: probed.append(p) or LlamaBinary(p, s, "v", True, True, True))
    assert [b.path for b in discovery.iter_llama_servers("a", None)] == ["a"]
    assert probed == ["a"]


# -- model load probe: skip a binary that cannot load the model ---------------


def _bin(path: str, source: str = "path") -> LlamaBinary:
    return LlamaBinary(path, source, "v", True, True, True)


class _StubService:
    """ServiceManager double: some binaries cannot load the model."""

    def __init__(self, cannot_load: set[str], port_error: bool = False) -> None:
        self.cannot_load = cannot_load
        self.port_error = port_error
        self.tried: list[str] = []

    def ensure_started(self, binary, model_path, sha, *, log=lambda _m: None):
        self.tried.append(binary.path)
        if self.port_error:
            raise ServiceError("another Headroom process is starting llama-server")
        if binary.path in self.cannot_load:
            raise BinaryLoadError("llama-server exited with 1 during load: unknown arch qwen35")
        return ServiceInfo("managed", "http://127.0.0.1:1", True, True, "ok", pid=7)

    def current(self, *, probe=True):
        return ServiceInfo("none", None, False, False, "none")

    def configured_endpoint(self):
        return None

    def stop(self):
        return False, "n/a"


def _verified(settings) -> model_mod.ModelStatus:
    return model_mod.ModelStatus(str(settings.model_path), True, True, "h", "h", False, None,
                                 "verified")


def _manager(monkeypatch, service, binaries, *, built: LlamaBinary | None = None):
    settings = IntelligenceSettings()
    manager = IntelligenceManager(settings, service=service)  # type: ignore[arg-type]
    monkeypatch.setattr(discovery, "jevk5_client_compatible", lambda: True)
    monkeypatch.setattr(discovery, "jevk5_client_version", lambda: "0.3.3")
    monkeypatch.setattr(discovery, "iter_llama_servers", lambda o, m: iter(binaries))
    monkeypatch.setattr(model_mod, "model_status", lambda s, **k: _verified(s))
    from headroom.intelligence import conformance
    from headroom.intelligence import manager as manager_mod

    report = conformance.ConformanceReport("llama-server", "u", checks={"endpoints": True})
    monkeypatch.setattr(manager_mod, "run_llama_conformance", lambda u, s, timeout: report)
    builds: list[int] = []
    if built is not None:
        monkeypatch.setattr(
            manager_mod.build_mod, "build_llama_server",
            lambda s, log: builds.append(1) or SimpleNamespace(binary=Path(built.path)),
        )
        monkeypatch.setattr(discovery, "probe_binary", lambda p, s: built)
    return manager, builds


def test_setup_skips_a_binary_that_cannot_load_the_model(monkeypatch):
    service = _StubService(cannot_load={"old"})
    # "incompat" fails the flag probe and is never started.
    binaries = [LlamaBinary("incompat", "path", None, False, False, False,
                            problems=("no --ctx-size flag",)), _bin("old"), _bin("good")]
    manager, builds = _manager(monkeypatch, service, binaries)
    report = manager.setup(allow_build=True, allow_download=False)
    assert report.ok, report.to_dict()
    assert service.tried == ["old", "good"] and builds == []
    setup = json.loads(manager.setup_manifest_path.read_text())
    assert setup["llama_server"] == "good"  # reused first next time


def test_setup_builds_a_managed_server_only_when_no_binary_loads(monkeypatch):
    service = _StubService(cannot_load={"old"})
    manager, builds = _manager(monkeypatch, service, [_bin("old")],
                               built=_bin("managed", "managed-build"))
    report = manager.setup(allow_build=True, allow_download=False)
    assert report.ok, report.to_dict()
    assert service.tried == ["old", "managed"] and builds == [1]


def test_setup_without_build_reports_every_load_problem(monkeypatch):
    service = _StubService(cannot_load={"a", "b"})
    manager, builds = _manager(monkeypatch, service, [_bin("a"), _bin("b")])
    report = manager.setup(allow_build=False, allow_download=False)
    assert not report.ok and builds == []
    failed = [s for s in report.steps if not s.ok][0]
    assert failed.name == "llama-server" and "a:" in failed.detail and "b:" in failed.detail


def test_a_port_or_lock_problem_is_not_blamed_on_the_binary(monkeypatch):
    service = _StubService(cannot_load=set(), port_error=True)
    manager, builds = _manager(monkeypatch, service, [_bin("a"), _bin("b")],
                               built=_bin("managed", "managed-build"))
    report = manager.setup(allow_build=True, allow_download=False)
    assert not report.ok
    assert service.tried == ["a"] and builds == []  # no fallback, no build


def test_background_activation_uses_the_same_fallback(monkeypatch):
    service = _StubService(cannot_load={"old"})
    manager, _ = _manager(monkeypatch, service, [_bin("old"), _bin("good")])
    manager.activate()
    assert manager.health()["advisor_state"] == "ready"
    assert service.tried == ["old", "good"]


# -- stateless: zero new files --------------------------------------------------


def _files(root: Path) -> set[Path]:
    return set(root.rglob("*")) if root.exists() else set()


def test_stateless_setup_and_activation_write_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_STATELESS", "1")
    service = _StubService(cannot_load=set())
    manager, _ = _manager(monkeypatch, service, [_bin("good")])
    ws = tmp_path / "ws"
    before = _files(ws)
    report = manager.setup(allow_build=False, allow_download=False)
    assert not report.ok and report.steps[-1].name == "stateless"
    manager.activate()  # nothing running: reuse-only, never starts one
    assert manager.health()["advisor_state"] == "unavailable"
    assert "HEADROOM_STATELESS" in manager.health()["advisor_detail"]
    assert service.tried == []
    assert _files(ws) == before


def test_stateless_model_verification_writes_no_sidecar(tmp_path, monkeypatch):
    from tests.intelligence.test_model_service_manager import _gguf, _sha

    monkeypatch.setenv("HEADROOM_STATELESS", "1")
    model = _gguf(tmp_path / "custom.gguf")
    settings = IntelligenceSettings(model_path_override=str(model), model_sha256=_sha(model))
    status = model_mod.model_status(settings)
    assert status.verified
    assert not list(tmp_path.glob("*.headroom-verified.json"))


def test_stateless_activation_reuses_a_running_service(monkeypatch):
    monkeypatch.setenv("HEADROOM_STATELESS", "1")
    with FakeLlamaServer() as fake:
        service = _StubService(cannot_load=set())
        service.configured_endpoint = lambda: ("llama-server", fake.url)  # type: ignore[method-assign]
        manager, _ = _manager(monkeypatch, service, [])
        manager.activate()
        assert manager.health()["advisor_state"] == "ready"
        assert service.tried == []  # reused, nothing started


# -- ownership: PID + executable + start time + command line ------------------


def test_ownership_requires_a_readable_matching_command_line():
    info = ProcessInfo(5, "/x/llama-server", ("/x/llama-server", "-m", "/m/a.gguf",
                                              "--port", "8091"), 100.0)
    ok = {"executable": "/x/llama-server", "create_time": 100.0, "model_path": "/m/a.gguf"}
    assert matches_owned(info, **ok, port=8091) == (True, "owned")
    assert not matches_owned(info, **ok, port=8092)[0]
    assert not matches_owned(info, **{**ok, "model_path": "/m/b.gguf"}, port=8091)[0]
    unreadable = ProcessInfo(5, "/x/llama-server", None, 100.0)
    owned, reason = matches_owned(unreadable, **ok, port=8091)
    assert not owned and "command line" in reason


@pytest.mark.skipif(sys.platform != "win32", reason="Windows command-line reader")
def test_windows_reads_a_process_command_line_without_psutil():
    import subprocess

    from headroom.intelligence import process

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)",
                              "--port", "4321"])
    try:
        time.sleep(0.3)
        info = process._windows_info(child.pid)
        assert info is not None and info.cmdline is not None
        assert "--port" in info.cmdline and "4321" in info.cmdline
    finally:
        child.kill()
        child.wait(5)


# -- policy: off makes no starts or calls --------------------------------------


def test_policy_off_never_activates_or_calls(monkeypatch):
    service = _StubService(cannot_load=set())
    settings = IntelligenceSettings(policy=ServicePolicy.OFF)
    manager = IntelligenceManager(settings, service=service)  # type: ignore[arg-type]
    manager.start_background()
    manager.activate()
    assert service.tried == [] and manager.health()["advisor_state"] == "disabled"
    assert manager.advisor.noul("f", "s", "q?", budget=RequestBudget(4)) is None


def test_stateless_setup_never_installs_the_client(monkeypatch):
    monkeypatch.setenv("HEADROOM_STATELESS", "1")
    installs: list[int] = []
    monkeypatch.setattr(discovery, "jevk5_client_compatible", lambda: False)
    monkeypatch.setattr(discovery, "install_jevk5_client", lambda: installs.append(1) or (True, ""))
    manager = IntelligenceManager(IntelligenceSettings(), service=_StubService(set()))  # type: ignore[arg-type]
    report = manager.setup(allow_build=False, allow_download=False)
    failed = [s for s in report.steps if not s.ok]
    assert failed and failed[0].name == "jevk5-client" and "STATELESS" in failed[0].detail
    assert installs == []


# -- Rust receives DecisionHints at the FFI boundary ---------------------------


def _responses_body(out_text: str) -> bytes:
    return json.dumps(
        {
            "model": "gpt-5.4",
            "input": [
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "look"}]},
                {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": out_text},
            ],
        }
    ).encode()


def test_relevance_query_hints_cross_the_rust_ffi_boundary():
    """``RelevanceQuery.to_dict()["hints"]`` is parsed by the Rust live zone.

    Exact entities past the query's token budget travel as relevance hints and
    steer the Rust ranking; the SafetyGate accepts the same payload (it never
    reads hints), and a malformed hint is a ``ValueError``, never ignored.
    """
    from headroom import _core
    from headroom.agent.interfaces import RelevanceQuery
    from headroom.transforms.safety_gate import SafetyGate

    entities = tuple(f"src/pkg_{i}/module_{i}.py" for i in range(40)) + ("refresh_session_token",)
    query = RelevanceQuery(explicit_user_goal="what is going on here", exact_entities=entities,
                           max_tokens=64)
    payload = query.to_dict()
    assert "refresh_session_token" not in query.text
    assert payload["hints"] is not None
    assert "refresh_session_token" in payload["hints"]["query_terms"]
    assert payload["hints"]["source"] == "deterministic" and payload["hints"]["weight"] == 0.0

    out_text = "".join(
        f"src/billing/invoice_{i}.py:{i}: total = compute_invoice_total(o)\n"
        f"src/auth/session_{i}.py:{i}: tok = refresh_session_token(u)\n"
        for i in range(150)
    )
    body = _responses_body(out_text)
    policy = json.dumps({"max_lossy_ratio": 0.45, "volatile_token_threshold": 128})

    def run(task: dict) -> str:
        out, modified, *_ = _core.compress_openai_responses_live_zone(
            body, "payg", "gpt-5.4", json.dumps(task), policy
        )
        return json.loads(bytes(out))["input"][2]["output"] if modified else out_text

    hinted = run(payload)
    plain = run({**payload, "hints": None})

    def count(text: str, needle: str) -> int:
        return sum(1 for line in text.splitlines() if needle in line)

    assert count(hinted, "refresh_session_token") > count(hinted, "compute_invoice_total") or (
        count(hinted, "refresh_session_token") > count(plain, "refresh_session_token")
    )
    decision = SafetyGate.check(out_text, out_text[: len(out_text) // 2], task=query)
    assert decision is not None
    with pytest.raises(ValueError):
        _core.compress_openai_responses_live_zone(
            body, "payg", "gpt-5.4", json.dumps({**payload, "hints": {"v": 99}}), policy
        )
