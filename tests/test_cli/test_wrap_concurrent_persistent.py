"""Concurrent ``headroom wrap`` against a task-managed persistent deployment.

Observed live on Windows, where ``install`` downgrades ``persistent-service`` to
``persistent-task`` (planner.py #2552), with a ``default`` deployment on 8787::

    headroom wrap claude --1m --dangerously-skip-permissions \
        --memory --learn --code-graph        # first session: works
    <same command again>                     # second session:
      Warning: task-based deployment 'default' cannot be auto-recovered via wrap
      Error: Persistent deployment 'default' on port 8787 is not healthy.

Two defects produced that dead end:

1. The healthy-deployment branch treated a *feature* shortfall as a reason to
   fall through to the ephemeral restart path, which kills whatever PID the
   running proxy reports and rebinds the port with a wrap-owned proxy. On a
   task deployment that silently destroyed task ownership: the PID file then
   named a dead process while a live Headroom proxy served the port. Nothing
   could restore the deployment's claim, because ``--learn``/``--code-graph``
   are not expressible in a manifest at all, so *every* wrap carrying those
   flags took that path.

2. With ownership so broken, ``runtime_ready`` is False while a Headroom proxy
   answers. ``_recover_persistent_proxy`` declines (task deployments are
   scheduler-owned) and the next branch raised unconditionally, without ever
   asking whether the live listener could serve this session.

These tests pin the ownership state machine: wrap classifies the port, never
kills or steals a deployment-owned runtime, and answers an unsatisfiable
request with a dedicated proxy on a free port instead of an exception.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any

import click
import pytest

import headroom.cli.wrap as wrap_cli
from headroom.install import runtime as install_runtime

_DEDICATED_PORT = 8788


@pytest.fixture(autouse=True)
def _clean_routing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep requested routing at its defaults so only features differ."""
    for name in (
        "ANTHROPIC_TARGET_API_URL",
        "OPENAI_TARGET_API_URL",
        "GEMINI_TARGET_API_URL",
        "CLOUDCODE_TARGET_API_URL",
        "VERTEX_TARGET_API_URL",
        "AUGMENT_TARGET_API_URL",
        "FACTORY_TARGET_API_URL",
        "HEADROOM_BACKEND",
        "HEADROOM_PROXY_EXTENSIONS",
    ):
        monkeypatch.delenv(name, raising=False)


class _TaskManifest:
    """A task-managed persistent deployment, as Windows ``install`` writes it."""

    profile = "default"
    preset = "persistent-task"
    runtime_kind = "python"
    supervisor_kind = "task"
    scope = "user"
    provider_mode = "auto"
    targets: list[str] = []
    port = 8787
    host = "127.0.0.1"
    backend = "anthropic"
    memory_enabled = True
    learn_enabled = False
    health_url = "http://127.0.0.1:8787/readyz"
    container_name = "headroom-default"
    service_name = "headroom-default"
    base_env: dict[str, str] = {}
    proxy_args: list[str] = []


class _ServiceManifest(_TaskManifest):
    """A supervised service deployment: wrap may drive its lifecycle."""

    preset = "persistent-service"
    supervisor_kind = "service"


def _health(**config: Any) -> dict[str, Any]:
    """A Headroom ``/health`` payload at the current version.

    Routing URL keys are deliberately absent: the requested routing resolves to
    ``None`` for each (see ``_clean_routing_env``), so an absent key compares
    equal and only the feature flags under test differ.
    """
    body = {
        "memory": False,
        "learn": False,
        "code_graph": False,
        "backend": "anthropic",
        "pid": 424242,
    }
    body.update(config)
    return {"version": wrap_cli._HEADROOM_VERSION, "config": body}


def _stub_proxy_io(
    monkeypatch: pytest.MonkeyPatch,
    *,
    manifest: Any,
    listening: bool,
    health: dict[str, Any] | None,
    attached: list[int] | None = None,
    runtime_ready: bool = False,
    runtime_claim: str = "released",
) -> dict[str, Any]:
    """Route ``_ensure_proxy_unlocked`` around real sockets and processes."""
    calls: dict[str, Any] = {"started": None, "killed": [], "restarted": 0, "recovered": 0}

    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda _port: manifest)
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda _port: listening)
    monkeypatch.setattr(wrap_cli, "_query_proxy_health", lambda _port: health)
    monkeypatch.setattr(wrap_cli, "_query_proxy_config", lambda _port: None)
    monkeypatch.setattr(wrap_cli, "_live_proxy_clients", lambda *_a, **_kw: list(attached or []))
    monkeypatch.setattr(install_runtime, "runtime_ready", lambda _m: runtime_ready)
    monkeypatch.setattr(install_runtime, "runtime_claim", lambda _m: runtime_claim)
    monkeypatch.setattr(wrap_cli, "_find_available_port", lambda _start, **_kw: _DEDICATED_PORT)

    def _kill(pid: int, port: int) -> bool:
        calls["killed"].append((pid, port))
        return True

    def _restart(_manifest: Any, _port: int) -> bool:
        calls["restarted"] += 1
        return True

    def _recover(_port: int) -> bool:
        calls["recovered"] += 1
        return False

    def _start(port: int, **_kw: Any) -> object:
        calls["started"] = port
        return object()

    monkeypatch.setattr(wrap_cli, "_kill_proxy_by_pid", _kill)
    monkeypatch.setattr(wrap_cli, "_restart_persistent_proxy", _restart)
    monkeypatch.setattr(wrap_cli, "_recover_persistent_proxy", _recover)
    monkeypatch.setattr(wrap_cli, "_start_proxy", _start)
    return calls


# --- Test A: a second concurrent wrap shares a compatible healthy deployment -


def test_second_wrap_reuses_healthy_compatible_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both wraps attach to the one healthy proxy; neither disturbs it."""
    calls = _stub_proxy_io(
        monkeypatch,
        manifest=_TaskManifest(),
        listening=True,
        health=_health(memory=True, learn=True, code_graph=True),
        attached=[999_001],  # the first wrap is still attached
        runtime_ready=True,
        runtime_claim="owned",
    )

    proc, port = wrap_cli._ensure_proxy(8787, False, memory=True, learn=True, code_graph=True)

    assert (proc, port) == (None, 8787), "a compatible shared proxy must be reused as-is"
    assert calls["killed"] == []
    assert calls["restarted"] == 0
    assert calls["started"] is None


# --- Test B: healthy task deployment missing a requested capability ---------


def test_capability_shortfall_never_replaces_task_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reserved port keeps its task-owned proxy; this wrap moves aside."""
    calls = _stub_proxy_io(
        monkeypatch,
        manifest=_TaskManifest(),
        listening=True,
        health=_health(memory=True),  # no learn, no code_graph
        attached=[],
        runtime_ready=True,
        runtime_claim="owned",
    )

    proc, port = wrap_cli._ensure_proxy(8787, False, memory=True, learn=True, code_graph=True)

    assert port == _DEDICATED_PORT, "must serve this session beside the deployment"
    assert proc is not None
    assert calls["started"] == _DEDICATED_PORT
    assert calls["killed"] == [], "a task-owned runtime is never killed by wrap"
    assert calls["restarted"] == 0, "restarting cannot add a capability the manifest lacks"


def test_capability_shortfall_reconciles_a_service_deployment_that_declares_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A drifted service runtime whose manifest does declare memory is restarted."""
    manifest = _ServiceManifest()
    calls = _stub_proxy_io(
        monkeypatch,
        manifest=manifest,
        listening=True,
        health=_health(memory=False),  # drifted: manifest says memory_enabled
        attached=[],
        runtime_ready=True,
        runtime_claim="owned",
    )

    proc, port = wrap_cli._ensure_proxy(8787, False, memory=True)

    assert (proc, port) == (None, 8787)
    assert calls["restarted"] == 1, "its own lifecycle mechanism can supply this"
    assert calls["killed"] == []
    assert calls["started"] is None


def test_capability_shortfall_dedicates_when_another_wrapper_is_attached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reconciliation is never performed under a live attached wrapper."""
    calls = _stub_proxy_io(
        monkeypatch,
        manifest=_ServiceManifest(),
        listening=True,
        health=_health(memory=False),
        attached=[999_002],
        runtime_ready=True,
        runtime_claim="owned",
    )

    _proc, port = wrap_cli._ensure_proxy(8787, False, memory=True)

    assert port == _DEDICATED_PORT
    assert calls["restarted"] == 0, "an attached wrapper outranks this session's upgrade"
    assert calls["killed"] == []


# --- Test C: the reported dead end -----------------------------------------


def test_unowned_headroom_listener_does_not_raise_not_healthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live Headroom proxy that the deployment no longer owns is not fatal."""
    calls = _stub_proxy_io(
        monkeypatch,
        manifest=_TaskManifest(),
        listening=True,
        health=_health(memory=True),  # serving, but missing learn/code_graph
        attached=[999_003],  # the first wrap is using it
        runtime_ready=False,  # ownership was lost
        runtime_claim="released",
    )

    proc, port = wrap_cli._ensure_proxy(8787, False, memory=True, learn=True, code_graph=True)

    assert port == _DEDICATED_PORT
    assert proc is not None
    assert calls["killed"] == [], "another wrapper is using that proxy"
    assert calls["restarted"] == 0


def test_unowned_compatible_headroom_listener_is_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the live listener already satisfies the request, share it."""
    calls = _stub_proxy_io(
        monkeypatch,
        manifest=_TaskManifest(),
        listening=True,
        health=_health(memory=True, learn=True, code_graph=True),
        attached=[999_004],
        runtime_ready=False,
        runtime_claim="released",
    )

    proc, port = wrap_cli._ensure_proxy(8787, False, memory=True, learn=True, code_graph=True)

    assert (proc, port) == (None, 8787)
    assert calls["started"] is None
    assert calls["killed"] == []


def test_unowned_listener_never_reported_as_a_healthy_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classification keeps 'a proxy is listening' apart from 'the deployment owns it'."""
    _stub_proxy_io(
        monkeypatch,
        manifest=_TaskManifest(),
        listening=True,
        health=_health(memory=True),
        runtime_ready=False,
        runtime_claim="released",
    )

    state = wrap_cli._persistent_port_state(_TaskManifest(), 8787)

    assert state == wrap_cli.PERSISTENT_PORT_UNOWNED_HEADROOM


# --- Test D: two simultaneous wrap starts ----------------------------------


def test_simultaneous_wraps_start_exactly_one_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serialized startup: the second caller re-checks under the lock and reuses.

    ``_proxy_start_lock`` is substituted with an in-process lock on purpose.
    The real one is an OS byte-range lock, and on Windows those are owned by the
    *process*, so two threads of one process cannot queue on it — the second
    gets ``PermissionError`` instead of waiting. The contract under test here is
    the one wrap depends on: the health/config checks and the start happen
    inside the critical section, so a caller that waits observes the finished
    proxy and reuses it rather than racing for the same port.
    """
    entered = 0
    concurrent = 0
    overlaps = 0
    guard = threading.Lock()
    serializer = threading.Lock()

    @contextmanager
    def _fake_start_lock(_port: int) -> Any:
        nonlocal entered, concurrent, overlaps
        with serializer:
            with guard:
                entered += 1
                concurrent += 1
                if concurrent > 1:
                    overlaps += 1
            try:
                yield
            finally:
                with guard:
                    concurrent -= 1

    monkeypatch.setattr(wrap_cli, "_proxy_start_lock", _fake_start_lock)
    state = {"listening": False, "starts": 0}
    lock = threading.Lock()

    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda _port: None)
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda _port: state["listening"])
    monkeypatch.setattr(
        wrap_cli,
        "_query_proxy_health",
        lambda _port: _health(memory=True) if state["listening"] else None,
    )
    monkeypatch.setattr(wrap_cli, "_query_proxy_config", lambda _port: None)
    monkeypatch.setattr(wrap_cli, "_live_proxy_clients", lambda *_a, **_kw: [])
    monkeypatch.setattr(wrap_cli, "_find_available_port", lambda start, **_kw: start)

    def _start(port: int, **_kw: Any) -> object:
        with lock:
            state["starts"] += 1
            state["listening"] = True
        return object()

    monkeypatch.setattr(wrap_cli, "_start_proxy", _start)

    results: list[tuple[Any, int]] = []
    errors: list[BaseException] = []

    def _run() -> None:
        try:
            results.append(wrap_cli._ensure_proxy(8787, False, memory=True))
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=_run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    assert len(results) == 2, "both wrappers must succeed"
    assert state["starts"] == 1, "the start lock must admit exactly one starter"
    assert {port for _proc, port in results} == {8787}
    assert entered == 2, "every caller must pass through the per-port start lock"
    assert overlaps == 0, "the health checks and the start must not interleave"


def test_ensure_proxy_runs_its_checks_inside_the_start_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lock must be held *around* the probes, not just around the spawn.

    A lock taken after the health check would let both callers decide "nothing
    is listening" before either started, which is the race the lock exists to
    prevent.
    """
    order: list[str] = []

    @contextmanager
    def _recording_lock(_port: int) -> Any:
        order.append("lock-enter")
        try:
            yield
        finally:
            order.append("lock-exit")

    monkeypatch.setattr(wrap_cli, "_proxy_start_lock", _recording_lock)
    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda _port: None)
    monkeypatch.setattr(
        wrap_cli, "_check_proxy", lambda _port: order.append("check-proxy") or False
    )
    monkeypatch.setattr(wrap_cli, "_query_proxy_health", lambda _port: None)
    monkeypatch.setattr(wrap_cli, "_query_proxy_config", lambda _port: None)
    monkeypatch.setattr(wrap_cli, "_live_proxy_clients", lambda *_a, **_kw: [])
    monkeypatch.setattr(wrap_cli, "_find_available_port", lambda start, **_kw: start)
    monkeypatch.setattr(
        wrap_cli, "_start_proxy", lambda port, **_kw: order.append("start") or object()
    )

    wrap_cli._ensure_proxy(8787, False, memory=True)

    assert order[0] == "lock-enter"
    assert order[-1] == "lock-exit"
    assert "check-proxy" in order[1:-1]
    assert order.index("check-proxy") < order.index("start")


def test_no_proxy_skips_the_start_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--no-proxy`` starts nothing, so it must not serialize behind starters."""
    taken: list[int] = []

    @contextmanager
    def _recording_lock(port: int) -> Any:
        taken.append(port)
        yield

    monkeypatch.setattr(wrap_cli, "_proxy_start_lock", _recording_lock)
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda _port: True)
    monkeypatch.setattr(wrap_cli, "_query_proxy_health", lambda _port: _health(memory=True))
    monkeypatch.setattr(wrap_cli, "_query_proxy_config", lambda _port: None)

    assert wrap_cli._ensure_proxy(8787, True) == (None, 8787)
    assert taken == []


# --- Test E: incompatible routing ------------------------------------------


def test_incompatible_routing_on_persistent_port_uses_dedicated_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A differently-routed deployment is never reused and never restarted."""
    calls = _stub_proxy_io(
        monkeypatch,
        manifest=_TaskManifest(),
        listening=True,
        health=_health(memory=True, learn=True, backend="openai"),
        attached=[999_005],
        runtime_ready=True,
        runtime_claim="owned",
    )

    proc, port = wrap_cli._ensure_proxy(8787, False, memory=True, learn=True, backend="anthropic")

    assert port == _DEDICATED_PORT
    assert proc is not None
    assert calls["killed"] == []
    assert calls["restarted"] == 0


def test_incompatible_routing_on_unowned_listener_uses_dedicated_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same strictness applies once deployment ownership is gone."""
    calls = _stub_proxy_io(
        monkeypatch,
        manifest=_TaskManifest(),
        listening=True,
        health=_health(memory=True, learn=True, backend="openai"),
        attached=[],
        runtime_ready=False,
        runtime_claim="released",
    )

    _proc, port = wrap_cli._ensure_proxy(8787, False, memory=True, learn=True, backend="anthropic")

    assert port == _DEDICATED_PORT, "routing is never silently inherited"
    assert calls["killed"] == []


# --- Test F: `install agent ensure` racing a wrap --------------------------


def test_wrap_does_not_claim_the_reserved_port_while_the_task_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deployment-owned runtime mid-start keeps its port; wrap moves aside."""
    calls = _stub_proxy_io(
        monkeypatch,
        manifest=_TaskManifest(),
        listening=False,  # bound soon, not yet
        health=None,
        attached=[],
        runtime_ready=False,
        runtime_claim="owned",  # the scheduler's process is alive
    )

    _proc, port = wrap_cli._ensure_proxy(8787, False, memory=True)

    assert port == _DEDICATED_PORT, "never bind a port a live deployment runtime owns"
    assert calls["killed"] == []
    assert calls["recovered"] == 0, "the task owner is already starting it"


def test_unverifiable_runtime_identity_is_treated_as_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows without psutil or argv markers must not let wrap steal the port."""
    calls = _stub_proxy_io(
        monkeypatch,
        manifest=_TaskManifest(),
        listening=True,
        health=_health(memory=True),
        attached=[],
        runtime_ready=False,
        runtime_claim="unverified",
    )

    _proc, port = wrap_cli._ensure_proxy(8787, False, memory=True, learn=True)

    assert port == _DEDICATED_PORT
    assert calls["killed"] == [], "unverified identity is never grounds for killing"


# --- Test G: stale PID metadata beside a live, valid proxy -----------------


def test_stale_pid_file_does_not_kill_a_live_compatible_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stale deployment metadata must not take down a working proxy."""
    calls = _stub_proxy_io(
        monkeypatch,
        manifest=_TaskManifest(),
        listening=True,
        health=_health(memory=True, learn=True, code_graph=True),
        attached=[],
        runtime_ready=False,  # PID file named a dead process
        runtime_claim="released",
    )

    proc, port = wrap_cli._ensure_proxy(8787, False, memory=True, learn=True, code_graph=True)

    assert (proc, port) == (None, 8787), "a live compatible proxy is still usable"
    assert calls["killed"] == []
    assert calls["started"] is None


def test_stale_pid_file_with_no_listener_leaves_a_recovery_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing serving and a scheduler-owned deployment: dedicate, never raise."""
    calls = _stub_proxy_io(
        monkeypatch,
        manifest=_TaskManifest(),
        listening=False,
        health=None,
        attached=[],
        runtime_ready=False,
        runtime_claim="released",
    )

    _proc, port = wrap_cli._ensure_proxy(8787, False, memory=True)

    assert port == _DEDICATED_PORT, "the scheduler's reserved port stays reserved"
    assert calls["killed"] == []


# --- Test H: the exact reported command, run twice -------------------------


_REAL_FLAGS = {"memory": True, "learn": True, "code_graph": True}


def test_reported_command_runs_twice_without_disrupting_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``wrap claude --memory --learn --code-graph`` twice over a task deployment.

    The persistent proxy offers only ``--memory``; ``--learn`` is not enabled in
    the manifest and ``--code-graph`` is session-scoped, so each wrap gets its
    own proxy. What matters is that the deployment keeps its port and its
    runtime, and that the second wrap neither raises nor disturbs the first.
    """
    ports: list[int] = []
    killed: list[tuple[int, int]] = []
    restarted: list[int] = []
    # Ports handed out in order: the first wrap takes 8788, the second 8789.
    free_ports = iter([8788, 8789])
    # Starts with only the task deployment attached; each wrap then registers.
    attached: list[int] = []

    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda _port: _TaskManifest())
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: port == 8787)
    monkeypatch.setattr(
        wrap_cli,
        "_query_proxy_health",
        lambda port: _health(memory=True) if port == 8787 else None,
    )
    monkeypatch.setattr(wrap_cli, "_query_proxy_config", lambda _port: None)
    monkeypatch.setattr(wrap_cli, "_live_proxy_clients", lambda *_a, **_kw: list(attached))
    monkeypatch.setattr(install_runtime, "runtime_ready", lambda _m: True)
    monkeypatch.setattr(install_runtime, "runtime_claim", lambda _m: "owned")
    monkeypatch.setattr(wrap_cli, "_find_available_port", lambda _start, **_kw: next(free_ports))
    monkeypatch.setattr(
        wrap_cli, "_kill_proxy_by_pid", lambda pid, port: killed.append((pid, port)) or True
    )
    monkeypatch.setattr(
        wrap_cli,
        "_restart_persistent_proxy",
        lambda _m, port: restarted.append(port) or True,
    )
    monkeypatch.setattr(
        wrap_cli, "_start_proxy", lambda port, **_kw: ports.append(port) or object()
    )

    first_proc, first_port = wrap_cli._ensure_proxy(8787, False, **_REAL_FLAGS)
    attached.append(999_101)  # the first wrap is now live

    second_proc, second_port = wrap_cli._ensure_proxy(8787, False, **_REAL_FLAGS)

    assert first_port == 8788
    assert second_port == 8789, "the second wrap must not fail or reuse the first's proxy"
    assert first_proc is not None and second_proc is not None
    assert ports == [8788, 8789]
    assert killed == [], "the task deployment's proxy is never killed"
    assert restarted == [], "nor restarted out from under the scheduler"


def test_reported_command_twice_raises_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard on the exact exception the user hit."""
    _stub_proxy_io(
        monkeypatch,
        manifest=_TaskManifest(),
        listening=True,
        health=_health(memory=True),
        attached=[999_102],
        runtime_ready=False,
        runtime_claim="released",
    )

    try:
        wrap_cli._ensure_proxy(8787, False, **_REAL_FLAGS)
    except click.ClickException as exc:  # pragma: no cover - the bug under test
        raise AssertionError(f"wrap must not fail here: {exc}") from exc
