"""First-party Phase-2 agent components, wired through the Phase-1 extension points.

Installed by :func:`headroom.agent.install.install_agent_runtime` (``create_app``)
when the ``agent_runtime`` rollout feature is on, each gated by its own
rollout feature in :data:`headroom.rollout.FEATURES`:

* ``agent_evidence`` — :class:`~headroom.agent.evidence.AgentEvidenceStore`
  plus its projector (raw events → facts, ``EVIDENCE_CHANGE`` replay);
* ``agent_task_state`` — :class:`~headroom.agent.task_state.TaskStateProjector`
  (CAS transitions, ``TASK_STATE_CHANGE`` replay), the
  :class:`~headroom.agent.task_state.TaskStateCompiler` snapshot publisher
  (commit listener) and the priority-80 request stage that folds the state
  block into the outbound request;
* ``agent_code_graph`` — :class:`~headroom.agent.code_graph.CodeGraphAdapter`,
  kept current by ``FILE_CHANGE`` events (commit listener) and a
  rate-limited stat rescan triggered from requests.

Services are exposed on ``runtime.services`` (``"evidence"``, ``"task_state"``,
``"code_graph"``, ``"ledger"``) for the request stages and later phases.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from headroom.agent import registry
from headroom.agent.events import AgentEvent, EventKind
from headroom.log_safety import describe_exception

log = logging.getLogger(__name__)

FEATURE_TASK_STATE = "agent_task_state"
FEATURE_EVIDENCE = "agent_evidence"
FEATURE_CODE_GRAPH = "agent_code_graph"
FEATURE_TOOL_ADMISSION = "agent_tool_admission"
FEATURE_FEEDBACK = "agent_feedback"
FEATURE_CHANGE_SCOPE = "agent_change_scope"
FEATURE_TOOL_CONTRACTS = "agent_tool_contracts"
FEATURE_TEST_IMPACT = "agent_test_impact"
FEATURE_WORKFLOW_MACROS = "agent_workflow_macros"

STAGE_TASK_STATE = "task_state"
LISTENER_SNAPSHOTS = "task_state_snapshots"
LISTENER_GRAPH = "code_graph_invalidate"

_REGISTERED_NAMES = (
    "evidence",
    "task_state",
    STAGE_TASK_STATE,
    LISTENER_SNAPSHOTS,
    LISTENER_GRAPH,
)


def _enabled(config: Any, feature: str) -> bool:
    rollout = getattr(config, "rollout", None)
    if rollout is None:
        return False
    try:
        return bool(rollout.is_enabled(feature))
    except KeyError:
        return False


def _evidence_lookup(evidence: Any) -> Callable[[AgentEvent], list[str]]:
    def lookup(event: AgentEvent) -> list[str]:
        facts = evidence.facts(event.workspace_id, event.session_id, source_event_id=event.id)
        ids = [f.id for f in facts]
        if ids:
            return ids
        rows = evidence.db.query(
            "SELECT id FROM evidence_facts WHERE workspace_id=? AND session_id=? "
            "AND status IN ('ACTIVE','CONTRADICTED') AND metadata LIKE ? LIMIT 8",
            (event.workspace_id, event.session_id, f'%"{event.id}"%'),
        )
        return [str(r[0]) for r in rows]

    return lookup


def _chooser_factory(runtime: Any) -> Callable[[AgentEvent], Any]:
    """JevK5 may pick WHICH existing atom a correction targets — nothing more."""

    def factory(event: AgentEvent) -> Any:
        advisor = runtime.advisor
        if advisor is None:
            return None
        budget = advisor.request_budget()

        def choose(correction: str, options: list[Any]) -> str | None:
            labels = {atom.id: atom.text[:200] for atom in options}
            advice = advisor.choose(
                "task_state_target",
                {"correction": correction[:600], "items": labels},
                "Which existing task item does the user's correction replace?",
                labels,
                budget=budget,
            )
            if advice is None:
                return None
            value = str(advice.value)
            return value if value in labels else None

        return choose

    return factory


def install_builtin_components(runtime: Any, config: Any) -> Callable[[], None]:
    """Register the enabled Phase-2 components; returns their teardown."""
    from headroom.agent.ledger_db import open_ledger

    task_on = _enabled(config, FEATURE_TASK_STATE)
    evidence_on = _enabled(config, FEATURE_EVIDENCE)
    graph_on = _enabled(config, FEATURE_CODE_GRAPH)
    admission_on = _enabled(config, FEATURE_TOOL_ADMISSION)
    feedback_on = _enabled(config, FEATURE_FEEDBACK)
    scope_on = _enabled(config, FEATURE_CHANGE_SCOPE)
    contracts_on = _enabled(config, FEATURE_TOOL_CONTRACTS)
    impact_on = _enabled(config, FEATURE_TEST_IMPACT)
    macros_on = _enabled(config, FEATURE_WORKFLOW_MACROS)
    if not (
        task_on
        or evidence_on
        or graph_on
        or admission_on
        or feedback_on
        or scope_on
        or contracts_on
        or impact_on
        or macros_on
    ):
        return lambda: None

    ledger = open_ledger(runtime.store)
    runtime.services["ledger"] = ledger
    from headroom.agent.retention import LISTENER as LISTENER_RETENTION
    from headroom.agent.retention import RetentionScheduler

    # Phase 8 bounded retention: pruned on a background thread, never on a request.
    retention = RetentionScheduler(runtime, ledger)
    runtime.services["retention"] = retention
    registry.register_commit_listener(LISTENER_RETENTION, retention.on_commit)
    evidence = None
    compiler = None
    graph = None

    if evidence_on:
        from headroom.agent.evidence import AgentEvidenceStore, EvidenceProjector

        evidence = AgentEvidenceStore(ledger)
        runtime.services["evidence"] = evidence

    if task_on:
        from headroom.agent.task_state import (
            TaskStateCompiler,
            TaskStateProjector,
            task_state_stage,
        )

        compiler = TaskStateCompiler(runtime, evidence=evidence)
        runtime.services["task_state"] = compiler
        projector = TaskStateProjector(
            ledger,
            emit=runtime.submit,
            evidence_lookup=_evidence_lookup(evidence) if evidence is not None else None,
            chooser_factory=_chooser_factory(runtime),
        )
        compiler.projector = projector
        registry.register_projector(projector)
        registry.register_commit_listener(LISTENER_SNAPSHOTS, compiler.on_commit)
        registry.register_request_stage(STAGE_TASK_STATE, task_state_stage, order=50)

    if evidence is not None:
        registry.register_projector(
            EvidenceProjector(
                evidence,
                emit=runtime.submit,
                active_task=(compiler.active_task_id if compiler is not None else None),
            )
        )

    if graph_on:
        from headroom.agent.code_graph import CodeGraphAdapter

        graph = CodeGraphAdapter(ledger)
        runtime.services["code_graph"] = graph

        def invalidate(events: list[AgentEvent], _runtime: Any) -> None:
            for event in events:
                if event.kind == EventKind.FILE_CHANGE.value and (
                    event.resource_id or ""
                ).startswith("path:"):
                    graph.invalidate_file(event.workspace_id, str(event.resource_id))

        registry.register_commit_listener(LISTENER_GRAPH, invalidate)

    if admission_on:
        # Tool-admission resource versions persist through the same ledger,
        # written and cold-loaded on the runtime worker only.
        from headroom.agent.tool_admission.service import bind_persistence

        bind_persistence(runtime)

    if feedback_on:
        # Retention-feedback priors (#12): generalized keys only, persisted
        # through the same ledger on the runtime worker.
        from headroom.agent.feedback import get_feedback

        feedback = get_feedback()
        feedback.enabled = True
        try:
            feedback.bind(ledger, runtime.submit_task)
        except Exception as exc:  # noqa: BLE001 - memory-only priors still work
            log.warning("feedback persistence unavailable: %s", describe_exception(exc))
        runtime.services["feedback"] = feedback

    enforcement_names = _install_enforcement(runtime, scope_on=scope_on, contracts_on=contracts_on)
    phase7_names = _install_verification(runtime, ledger, impact_on=impact_on, macros_on=macros_on)

    def teardown() -> None:
        for name in phase7_names:
            registry.unregister(name)
        for key in ("test_impact", "workflow_macros"):
            service = runtime.services.pop(key, None)
            unbind = getattr(service, "unbind", None) or getattr(
                getattr(service, "history", None), "unbind", None
            )
            if callable(unbind):
                unbind()
        for name in enforcement_names:
            registry.unregister(name)
        for key in ("change_scope", "tool_contracts"):
            runtime.services.pop(key, None)
        if feedback_on:
            from headroom.agent.feedback import get_feedback

            get_feedback().unbind()
            runtime.services.pop("feedback", None)
        if admission_on:
            from headroom.agent.tool_admission.service import unbind_persistence

            unbind_persistence(runtime)
        registry.unregister(LISTENER_RETENTION)
        runtime.services.pop("retention", None)
        retention.close()
        for name in _REGISTERED_NAMES:
            registry.unregister(name)
        for key in ("task_state", "evidence", "code_graph", "ledger"):
            runtime.services.pop(key, None)
        if graph is not None:
            graph.close()
        try:
            ledger.close()
        except Exception as exc:  # noqa: BLE001
            log.debug("agent ledger close failed: %s", describe_exception(exc))

    return teardown


def _install_enforcement(runtime: Any, *, scope_on: bool, contracts_on: bool) -> list[str]:
    """Phase 6: change-scope firewall (#23) first, then tool contracts (#19).

    The firewall owns change contracts and revision-pinned write
    authorization; executed writes (``FILE_CHANGE``) bump its resource
    revisions. The observe-only request stage classifies calls the client
    already executed (every client); enforcement happens only at the host
    pre-tool hook (:mod:`headroom.agent.enforcement`).
    """
    if not (scope_on or contracts_on):
        return []
    from headroom.agent.change_scope import ChangeScopeFirewall, EnforcementPolicy
    from headroom.agent.enforcement import LISTENER_WRITES, STAGE_OBSERVE, observe_stage

    policy = EnforcementPolicy.from_env()
    names: list[str] = []
    firewall = None
    if scope_on:
        firewall = ChangeScopeFirewall(runtime, policy=policy)
        runtime.services["change_scope"] = firewall
        registry.register_commit_listener(LISTENER_WRITES, firewall.on_commit)
        names.append(LISTENER_WRITES)
    if contracts_on:
        from headroom.agent.tool_contracts import ToolContractValidator

        runtime.services["tool_contracts"] = ToolContractValidator(
            runtime, firewall=firewall, policy=policy
        )
    registry.register_request_stage(STAGE_OBSERVE, observe_stage, order=60)
    names.append(STAGE_OBSERVE)
    return names


def _install_verification(
    runtime: Any, ledger: Any, *, impact_on: bool, macros_on: bool
) -> list[str]:
    """Phase 7: the verification planner (#24) first, then workflow macros (#20).

    The planner parses every ``TEST_RESULT`` (projector), records runs and
    refreshes plans after commits (listener) and folds one advisory block per
    request (stage order 65, which also carries the eligible workflows). The
    macro registry observes closed tool segments (stage order 62, recorded on
    the worker) and registers ``headroom_workflow`` in the shared tool registry.
    """
    names: list[str] = []
    if impact_on:
        from headroom.agent.test_impact import LISTENER_PLANNER, TestImpactPlanner

        planner = TestImpactPlanner(runtime)
        try:
            planner.history.bind(ledger, runtime.submit_task)
        except Exception as exc:  # noqa: BLE001 - memory-only history still works
            log.warning("verification history persistence unavailable: %s", describe_exception(exc))
        runtime.services["test_impact"] = planner
        registry.register_projector(planner)
        registry.register_commit_listener(LISTENER_PLANNER, planner.on_commit)
        names += [planner.name, LISTENER_PLANNER]
    if macros_on:
        from headroom.agent.workflow_macros import (
            STAGE_OBSERVE,
            WorkflowMacroRegistry,
            observe_stage,
            register_local_tool,
        )

        macros = WorkflowMacroRegistry(runtime)
        try:
            macros.bind(ledger, runtime.submit_task)
        except Exception as exc:  # noqa: BLE001 - memory-only macros still work
            log.warning("workflow macro persistence unavailable: %s", describe_exception(exc))
        runtime.services["workflow_macros"] = macros
        register_local_tool()
        registry.register_request_stage(STAGE_OBSERVE, observe_stage, order=62)
        names.append(STAGE_OBSERVE)
    if impact_on or macros_on:
        from headroom.agent.test_impact import STAGE_VERIFICATION, verification_stage

        registry.register_request_stage(STAGE_VERIFICATION, verification_stage, order=65)
        names.append(STAGE_VERIFICATION)
    return names
