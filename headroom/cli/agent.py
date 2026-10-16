"""``headroom agent`` — compact status and doctor for the wrap intelligence stack.

``status`` reads the running proxy's ``/health`` (loopback) and condenses the
agent blocks: effective rollout, which capabilities enforce at a host versus
only observe, trusted workspaces, the state DB, JevK5 readiness, enforcement
policies and per-feature counters. ``doctor`` adds local checks that do not
need a proxy (the rollout ``headroom wrap`` would apply, persistence, ripgrep,
JevK5 files). Neither command changes any state.
"""

from __future__ import annotations

import json
import os
from typing import Any

import click

from .main import main

AGENT_FEATURES = (
    "agent_runtime",
    "agent_task_state",
    "agent_evidence",
    "agent_code_graph",
    "agent_tool_admission",
    "agent_arbiter",
    "agent_tool_catalog",
    "agent_complexity_routing",
    "agent_feedback",
    "agent_speculation",
    "agent_change_scope",
    "agent_tool_contracts",
    "agent_test_impact",
    "agent_workflow_macros",
)


def _port(port: int | None) -> int:
    return port if port is not None else int(os.environ.get("HEADROOM_PORT", "8787"))


def _health(port: int) -> dict[str, Any] | None:
    from headroom.install.health import probe_json

    return probe_json(f"http://127.0.0.1:{port}/health", timeout=3.0)


def summarize(health: dict[str, Any]) -> dict[str, Any]:
    """The compact view of ``/health`` (pure; counts and modes only)."""
    runtime = health.get("agent_runtime") or {}
    if not isinstance(runtime, dict):
        runtime = {}
    inner = runtime.get("runtime") or {}
    admission = health.get("tool_admission") or {}
    enforcement = health.get("agent_enforcement") or {}
    verification = health.get("agent_verification") or {}
    workflows = verification.get("workflows") or {}
    scope = enforcement.get("change_scope") or {}
    intelligence = runtime.get("intelligence") or {}
    rollout = runtime.get("rollout")
    enabled: dict[str, bool] = {"agent_runtime": bool(runtime.get("feature_enabled"))}
    if isinstance(rollout, dict):
        enabled.update({str(k): bool(v) for k, v in rollout.items() if k in AGENT_FEATURES})
    else:  # an older proxy without the rollout block
        enabled.update({k: bool(v) for k, v in (runtime.get("features") or {}).items()})
        enabled["agent_tool_admission"] = bool(admission.get("enabled"))
        enabled["agent_change_scope"] = bool(enforcement.get("change_scope_enabled"))
        enabled["agent_tool_contracts"] = bool(enforcement.get("tool_contracts_enabled"))
        enabled["agent_test_impact"] = bool(verification.get("test_impact_enabled"))
        enabled["agent_workflow_macros"] = bool(workflows.get("enabled"))
    feedback = admission.get("feedback") or {}
    return {
        "features": enabled,
        "hook_installed": runtime.get("hook_installed"),
        "trusted_workspace_markers": runtime.get("trusted_workspace_markers"),
        "stateless": runtime.get("stateless"),
        "store": inner.get("store"),
        "jevk5": {
            "policy": intelligence.get("policy"),
            "state": intelligence.get("advisor_state"),
            "detail": intelligence.get("advisor_detail"),
            "breaker": intelligence.get("breaker"),
        },
        "capabilities": {
            "enforcement": enforcement.get("capabilities"),
            "proxy_path": enforcement.get("proxy_path"),
            "workflows": workflows.get("hosts"),
            "tool_admission_host": admission.get("host"),
        },
        "policies": scope.get("policy"),
        "holdout": {
            "fraction": feedback.get("holdout_fraction"),
            "reverted_families": sorted((feedback.get("reverted_families") or {}).keys()),
        },
        "counters": {
            "runtime": inner.get("counters"),
            "admission_stages": (admission.get("dispatcher") or {}).get("stages"),
            "arbiter_decisions": (admission.get("arbiter") or {}).get("decisions"),
            "enforcement": enforcement.get("counters"),
            "verification": (verification.get("test_impact") or {}).get("counters"),
        },
    }


def local_checks() -> list[tuple[str, bool, str]]:
    """Checks that need no proxy: (name, ok, detail)."""
    from headroom import paths
    from headroom.agent.workflow_hosts import ripgrep
    from headroom.rollout import (
        RolloutConfigurationError,
        apply_wrap_feature_defaults,
        resolve_rollout,
    )

    checks: list[tuple[str, bool, str]] = []
    try:
        snapshot = resolve_rollout(apply_wrap_feature_defaults(dict(os.environ)))
        on = [name for name in AGENT_FEATURES if snapshot.is_enabled(name)]
        off = [name for name in AGENT_FEATURES if name not in on]
        detail = f"{len(on)}/{len(AGENT_FEATURES)} on" + (f"; off: {', '.join(off)}" if off else "")
        checks.append(("wrap rollout", True, detail))
    except RolloutConfigurationError as exc:
        checks.append(("wrap rollout", False, str(exc)))
    if paths.process_is_stateless():
        checks.append(("state db", True, "stateless: in-memory only, nothing written"))
    else:
        db = paths.agent_state_db_path()
        parent = db.parent
        writable = os.access(parent if parent.exists() else paths.workspace_dir(), os.W_OK)
        checks.append(
            ("state db", writable, f"{db} ({'exists' if db.exists() else 'not yet created'})")
        )
    rg = ripgrep()
    checks.append(("ripgrep", True, rg or "not found: workflow Grep/Glob steps are ineligible"))
    from headroom.intelligence.manager import IntelligenceManager

    status = IntelligenceManager().status(probe=False)
    service = status.get("service") or {}
    checks.append(
        (
            "jevk5",
            True,
            f"policy {status.get('policy')}, setup {'done' if status.get('setup_completed') else 'not run'}, "
            f"service {service.get('kind')} (optional; deterministic behavior without it)",
        )
    )
    return checks


@main.group("agent")
def agent_group() -> None:
    """Wrap intelligence stack: compact status and doctor (read-only)."""


@agent_group.command("status")
@click.option("--port", type=int, default=None, help="Proxy port (default HEADROOM_PORT or 8787).")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def status_cmd(port: int | None, as_json: bool) -> None:
    """Summarize the running proxy's agent features, capabilities and counters."""
    health = _health(_port(port))
    if health is None:
        raise click.ClickException(f"no Headroom proxy answered on 127.0.0.1:{_port(port)}")
    view = summarize(health)
    if as_json:
        click.echo(json.dumps(view, indent=2, sort_keys=True, default=str))
        return
    on = [k for k, v in view["features"].items() if v]
    click.echo(f"features on: {', '.join(on) or 'none'}")
    click.echo(
        f"hook installed: {view['hook_installed']}; trusted workspaces: "
        f"{view['trusted_workspace_markers']}; stateless: {view['stateless']}"
    )
    click.echo(f"state store: {view['store']}")
    jev = view["jevk5"]
    click.echo(
        f"jevk5: {jev['policy']} / {jev['state']} ({jev['detail']}); breaker {jev['breaker']}"
    )
    caps = view["capabilities"]
    click.echo(f"enforcement: {caps['enforcement']}; proxy path: {caps['proxy_path']}")
    click.echo(f"policies: {view['policies']}")
    click.echo(f"holdout: {view['holdout']}")
    for name, counters in view["counters"].items():
        if counters:
            click.echo(f"{name}: {counters}")


@agent_group.command("doctor")
@click.option("--port", type=int, default=None, help="Proxy port (default HEADROOM_PORT or 8787).")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def doctor_cmd(port: int | None, as_json: bool) -> None:
    """Local checks plus, when a proxy is running, its agent status."""
    checks = local_checks()
    health = _health(_port(port))
    payload = {
        "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks],
        "proxy": summarize(health) if health is not None else None,
    }
    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        for name, ok, detail in checks:
            click.echo(f"[{'ok ' if ok else 'FAIL'}] {name}: {detail}")
        if health is None:
            click.echo(f"[info] no proxy on 127.0.0.1:{_port(port)}; run `headroom wrap <client>`")
        else:
            view = payload["proxy"]
            on = sum(1 for v in view["features"].values() if v)
            click.echo(f"[ok ] proxy: {on} agent features on, jevk5 {view['jevk5']['state']}")
    if not all(ok for _n, ok, _d in checks):
        raise SystemExit(1)
