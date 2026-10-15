"""``headroom intelligence`` — set up and manage the local JevK5 decision advisor."""

from __future__ import annotations

import json
import sys

import click

from .main import main


def _manager():  # type: ignore[no-untyped-def]
    from headroom.intelligence.manager import IntelligenceManager

    return IntelligenceManager()


def _emit(payload: dict, as_json: bool, render) -> None:  # type: ignore[no-untyped-def]
    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        render(payload)


@main.group("intelligence")
def intelligence_group() -> None:
    """Local JevK5 decision advisor (llama.cpp): setup, status, doctor, stop.

    \b
    JevK5 is an optional advisory classifier. When it is unavailable, Headroom
    keeps working with its deterministic behavior.
    \b
    Policy: HEADROOM_JEVK5=auto|on|off (default auto).
    External endpoints: HEADROOM_JEVK5_LLAMA_URL (llama-server) or
    HEADROOM_JEVK5_SYSTEMONE_URL (/v1/systemone).
    """


@intelligence_group.command("setup")
@click.option("--no-build", is_flag=True, help="Do not build llama-server from source.")
@click.option("--no-download", is_flag=True, help="Use the verified model cache only.")
@click.option("--no-start", is_flag=True, help="Prepare files but do not start the service.")
@click.option("--json", "as_json", is_flag=True, help="Emit a JSON report.")
def setup_cmd(no_build: bool, no_download: bool, no_start: bool, as_json: bool) -> None:
    """Install the JevK5 client, find/build llama-server, fetch+verify the model, start it."""
    manager = _manager()
    report = manager.setup(
        log_fn=(lambda m: None) if as_json else (lambda m: click.echo(f"  {m}")),
        allow_build=not no_build,
        allow_download=not no_download,
        start_service=not no_start,
    )

    def render(payload: dict) -> None:
        for step in payload["steps"]:
            mark = "ok " if step["ok"] else "FAIL"
            click.echo(f"[{mark}] {step['name']}: {step['detail']}")
        conformance = payload.get("conformance")
        if conformance:
            click.echo(f"conformance: {'passed' if conformance['passed'] else 'failed'}")
            for name, ok in sorted(conformance["checks"].items()):
                click.echo(f"  {'ok ' if ok else 'FAIL'} {name}")

    _emit(report.to_dict(), as_json, render)
    if not report.ok:
        sys.exit(1)


@intelligence_group.command("status")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def status_cmd(as_json: bool) -> None:
    """Show policy, service ownership/health and advisor state."""
    payload = _manager().status(probe=True)

    def render(data: dict) -> None:
        service = data["service"]
        click.echo(f"Policy:   {data['policy']} (autostart={data['autostart']}, "
                   f"offline={data['offline']})")
        click.echo(f"Service:  {service['kind']} {service.get('url') or ''} "
                   f"owned={service['owned']} healthy={service['healthy']} ({service['detail']})")
        click.echo(f"Client:   jevk5 {data['jevk5_client'] or 'not installed'}")
        click.echo(f"Setup:    {'completed' if data['setup_completed'] else 'not run'}")

    _emit(payload, as_json, render)


@intelligence_group.command("doctor")
@click.option("--verify-hash", is_flag=True, help="Re-hash the model file (slow).")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def doctor_cmd(verify_hash: bool, as_json: bool) -> None:
    """Diagnose executable, runtime version, model hash, endpoint, context and ownership."""
    payload = _manager().doctor(verify_hash=verify_hash)

    def render(data: dict) -> None:
        binary = data.get("llama_server") or {}
        model = data["model"]
        own = data["process_ownership"]
        click.echo(f"Python:            {data['python']}")
        click.echo(f"JevK5 runtime:     {data['jevk5_runtime'] or 'missing'} "
                   f"(expected {data['jevk5_runtime_expected']})")
        click.echo(f"llama-server:      {binary.get('path') or 'not found'} "
                   f"{binary.get('version') or ''}")
        if binary.get("devices"):
            click.echo(f"Devices:           {', '.join(binary['devices'])}")
        click.echo(f"Model:             {model['path']} verified={model['verified']} "
                   f"({model['detail']})")
        click.echo(f"Model SHA-256:     {model['sha256'] or model['expected_sha256'] or '-'}")
        click.echo(f"Endpoint:          {data['endpoint_type']} {data['endpoint'] or ''}")
        click.echo(f"Effective context: {data['effective_context']} "
                   f"(configured {data['configured_context']})")
        cal = data["calibration"]
        click.echo(f"Calibration:       T={cal['temperature']} knockout_T="
                   f"{cal['knockout_temperature']} top_k={cal['top_k']} "
                   f"retry_n_probs={cal['missing_retry_n_probs']}")
        click.echo(f"Missing letters:   {data['missing_letter_health']}")
        click.echo(f"Ownership:         owned={own['owned']} pid={own['pid']} ({own['detail']})")
        conformance = data.get("conformance")
        if conformance:
            click.echo(f"Conformance:       {'passed' if conformance['passed'] else 'failed'}")
            for name, ok in sorted(conformance["checks"].items()):
                click.echo(f"  {'ok ' if ok else 'FAIL'} {name}")

    _emit(payload, as_json, render)


@intelligence_group.command("stop")
def stop_cmd() -> None:
    """Stop the Headroom-managed llama-server (never an operator or foreign process)."""
    stopped, detail = _manager().stop()
    click.echo(detail)
    if not stopped and ("refusing" in detail or "did not exit" in detail):
        sys.exit(1)
