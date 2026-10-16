"""Unified-21 Phase 8 operator interface: ``headroom agent status|doctor`` over
the real ``/health`` of a wrap-rollout proxy."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from headroom import paths
from headroom.agent import registry
from headroom.cli import agent as agent_cli
from headroom.cli.main import main
from headroom.proxy.turn_hooks import clear_turn_hooks
from tests.agent.test_phase2_integration import _config, _wrap_env


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "ws"))
    monkeypatch.setenv("HEADROOM_JEVK5", "off")
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    monkeypatch.delenv("HEADROOM_DISABLE_FEATURES", raising=False)
    monkeypatch.delenv("HEADROOM_FEATURES", raising=False)
    clear_turn_hooks()
    registry.clear_registry()
    yield
    clear_turn_hooks()
    registry.clear_registry()
    paths.set_process_stateless(False)


def _health(env: dict[str, str]) -> dict:
    from headroom.proxy.server import create_app

    app = create_app(_config(env))
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as client:
        return client.get("/health").json()


def test_status_summarizes_the_real_wrap_health_payload(monkeypatch):
    health = _health(_wrap_env(HEADROOM_DISABLE_FEATURES="agent_workflow_macros"))
    view = agent_cli.summarize(health)
    assert view["features"]["agent_runtime"] is True
    assert view["features"]["agent_tool_admission"] is True
    assert view["features"]["agent_workflow_macros"] is False  # the disable is honoured
    assert set(view["features"]) == set(agent_cli.AGENT_FEATURES)
    assert view["hook_installed"] is True and view["stateless"] is True
    assert view["store"]["backend"] == "memory"
    assert view["jevk5"]["policy"] == "off" and view["jevk5"]["state"] == "disabled"
    assert view["capabilities"]["enforcement"]["claude"] == "host_pretool"
    assert view["capabilities"]["enforcement"]["codex"] == "observe_only"
    assert view["capabilities"]["proxy_path"] == "observe_only"
    assert view["policies"] == {"change_scope": "warn", "tool_contracts": "warn"}
    assert view["holdout"] == {"fraction": 0.0, "reverted_families": []}

    monkeypatch.setattr(agent_cli, "_health", lambda port: health)
    result = CliRunner().invoke(main, ["agent", "status", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["features"] == view["features"]
    text = CliRunner().invoke(main, ["agent", "status"])
    assert text.exit_code == 0 and "jevk5: off / disabled" in text.output


def test_status_without_a_proxy_fails_and_doctor_still_runs_local_checks(monkeypatch):
    monkeypatch.setattr(agent_cli, "_health", lambda port: None)
    monkeypatch.setenv("HEADROOM_STATELESS", "1")
    status = CliRunner().invoke(main, ["agent", "status", "--port", "9"])
    assert status.exit_code != 0 and "no Headroom proxy" in status.output
    doctor = CliRunner().invoke(main, ["agent", "doctor", "--json"])
    assert doctor.exit_code == 0, doctor.output
    payload = json.loads(doctor.output)
    names = {c["name"]: c for c in payload["checks"]}
    assert names["wrap rollout"]["ok"] and names["wrap rollout"]["detail"].startswith("14/14 on")
    assert "stateless" in names["state db"]["detail"]
    assert {"ripgrep", "jevk5"} <= set(names)
    assert payload["proxy"] is None
    assert not (paths.workspace_dir() / "agent").exists()  # read-only: nothing created


def test_doctor_reports_a_disabled_wrap_feature(monkeypatch):
    monkeypatch.setattr(agent_cli, "_health", lambda port: None)
    monkeypatch.setenv("HEADROOM_DISABLE_FEATURES", "agent_feedback")
    payload = json.loads(CliRunner().invoke(main, ["agent", "doctor", "--json"]).output)
    rollout = next(c for c in payload["checks"] if c["name"] == "wrap rollout")
    assert rollout["detail"] == "13/14 on; off: agent_feedback"
