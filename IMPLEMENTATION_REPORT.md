# Unified-21 Implementation Report

Spec: `HEADROOM_WRAP_UNIFIED_21_FEATURE_IMPLEMENTATION_PLAN_OPUS_5_5.md`. Branch `feature/unified-21`.

## Refs and state

| Item | Value |
|---|---|
| Upstream merged | tag `v0.41.0`, `76133cff` (matches the plan). The old checkout `d13e1966` (0.39.1) had no local commits, so the merge was a fast-forward with no conflicts. |
| Start / end of work | `76133cff` → HEAD of `feature/unified-21` (the commit carrying this report) |
| Safety refs | branch `backup/main-pre-unified21` (`d13e1966`); `stash@{0}` kept; backups of `.claude/.headroom_wrap_settings.lock` and `.serena/project.yml` in `../_safety_backup_20261014_main/` |
| User files never committed | `AGENTS.md`, the plan, `.clinerules-*`, `.claude/*`, `.serena/*` |
| Paritok/JevK5 code before | none; this checkout was plain upstream. The shell's ~40 Paritok-era `HEADROOM_*` exports name features v0.41 lacks, so every test and eval runs with them unset. |
| Environment | Windows 10 19045, Python 3.12.10, rustc 1.95.0, RTX A4000 16 GB, WinGet llama-server b44da65e. TMP/TEMP on F: (C: is nearly full). |

**Completion status.** Phases 0–8 are implemented and committed. Under the plan's definition the job is done only once the full suite passes. **The full suite was not run, at the user's direction** ("do not run the unified tester script"), so final completion is pending that run. Command, from the repo root with the Paritok-era variables unset:

```
env $(cat target/tmp/unset_args.txt) TMP=$PWD/target/tmp/t TEMP=$PWD/target/tmp/t .venv/Scripts/python.exe -m pytest -p no:cacheprovider -n 8 --timeout=300 -o addopts="" tests
CARGO_TARGET_DIR=$PWD/target cargo test --workspace
```

## Feature status (all 21 reach a production runtime path)

Wrap defaults are enabled by `headroom wrap` through `rollout.apply_wrap_feature_defaults`. They are opt-in for a bare proxy, and `HEADROOM_DISABLE_FEATURES` disables any of them.

| ID | Feature | Owner (path) | Runtime path | Status |
|---|---|---|---|---|
| 1 | Task-conditioned live-zone compression | `headroom_core` live zone, `agent/hook.compression_relevance_query` | Python handlers → Rust `compress_*_live_zone_with_options` | Done |
| 2 | Rust hybrid semantic relevance | `relevance::semantic`, `HybridScorer` | live zone, catalog, CCR span ranking; cache-only model, BM25 fallback | Done |
| 3 | Indexed/selective CCR, exact full retrieval | `ccr/indexed.py` | response handler, MCP, `/v1/retrieve*`, proactive expansion | Done |
| 4 | Pre-context admission/externalization | `agent/tool_admission/`, `host_hook` | Claude PostToolUse hook; proxy fallback transform | Done (host pre-context: Claude only) |
| 5 | Cross-turn semantic/structural deltas | `tool_admission/delta.py` | dispatcher | Done |
| 6 | Transform Arbiter | `agent/arbiter.py` | dispatcher `prepare → allocate → decide` | Done |
| 7 | Semantic invariants + automatic backoff | `headroom_core::safety_gate`, `transforms/safety_gate.py` | every Rust block, every router/admission candidate | Done |
| 8 | Global token allocator | `agent/budget.py` | proxy transform | Done |
| 9 | Progressive tool catalog | `agent/tool_catalog.py` | Anthropic `/v1/messages`, OpenAI chat | Done |
| 10 | Effort/verbosity routing | `agent/complexity.py` | Anthropic, chat, Responses (holdout treatment arm) | Done |
| 11 | Graph/symbol-scoped coding context | `agent/code_graph.py` | relevance query, Read outline, scope dependencies | Done |
| 12 | Retrieval feedback / retention adaptation | `agent/feedback.py` | arbiter prior, CCR retrieval hook; Phase 8 holdout + family revert | Done |
| 13 | Enforced lossy-risk / volatile policy | `CompressionPolicy` via SafetyGate | all compression paths | Done |
| 14 | Read/Grep/Glob/Bash/test/diff/JSON interceptors | `tool_admission/views.py` | dispatcher | Done |
| 15 | Bounded speculation | `agent/speculation.py` | one lane per request | Done |
| 16 | Task State Compiler | `agent/task_state.py` | TurnHook stage 50 | Done |
| 17 | Evidence Ledger | `agent/evidence.py` | event projector | Done |
| 19 | Tool Contract validator | `agent/tool_contracts.py` | Claude PreToolUse (Bash/PowerShell); proxy observe | Done (enforced on Claude only) |
| 20 | Workflow Macro Compiler | `agent/workflow_macros.py`, `workflow_hosts.py` | `headroom_workflow` MCP tool + PreToolUse ticket | Done (executes on Claude only) |
| 23 | Change-Scope firewall | `agent/change_scope.py` | Claude PreToolUse; proxy observe | Done (enforced on Claude only) |
| 24 | Test Impact planner | `agent/test_impact.py` | stage 65 + `TEST_RESULT` projector | Done |
| — | JevK5 setup/management/advisor | `headroom/intelligence/` | `IntelligenceManager` in `create_app`; arbiter, catalog, complexity, task-state, repair ranking | Done |

## Per-phase summary (condensed)

- **0** Eval harness `benchmarks/unified21_eval.py` (11 offline fixtures, real Python pipeline and Rust live zone) and baseline.
- **1** `headroom/agent/` runtime: digest-only `AgentEvent`s, trusted identity (wrap token + live marker), WAL SQLite store or a stateless memory store, one stream-safe TurnHook at priority 80. JevK5: typed settings, GGUF+SHA-256 verification, llama-server discovery or a managed CMake build, a 127.0.0.1 service that never kills foreign processes (PID+exe+start-time check), a shared monotonic deadline, an n_probs=128 retry, a circuit breaker, distinct llama-server and `/v1/systemone` clients, and `headroom intelligence setup|status|doctor|stop`.
- **2** Task state (#16), evidence (#17) and code graph (#11) on the ledger. Atoms are CAS-projected and replayable. The state block is appended after the live turn and never touches the cached prefix.
- **3** One SafetyGate in Rust with Python parity fixtures, a lossy budget separate from the wire drop, a candidate ladder, the semantic scorer, query-conditioned live zone and indexed CCR with HMAC cursors.
- **4** Envelope v2, admission dispatcher (exact original into CCR, read-back self-test, gate, replace), per-family interceptors, structured deltas, Claude PostToolUse `updatedToolOutput`.
- **5** Budget + arbiter, progressive catalog, effort/verbosity routing, retention feedback, speculation.
- **6** Change scope (#23), then tool contracts (#19), with honest enforce-vs-observe per host.
- **7** Verification planner (pytest/Cargo/CTest/Jest/Vitest), then safe learned workflows (ticketed, argv `shell=False`, process-group kill, side-effect digests). Follow-up: client-independent engine with host adapters, ripgrep Grep/Glob parity with Claude Code 2.1, coverage-data robustness, Linux qualification.
- **8** (this phase) The plan item 8 holdout, plus automatic revert of a regressing transform family. `headroom agent status|doctor`. A JevK5 OFF/ON harness mode. Live baseline on v0.41.0. Four upstream tests aligned. This report.

**Phase 8 code.** `HEADROOM_TRANSFORM_HOLDOUT` is a fraction, default 0, max 0.5, read live. A scope-stable share of sessions gets exact originals. The arbiter's would-be lossy choice registers a *control* observation on the same reread/retrieval/repair/correction signals, and its savings count as forgone, never realized. Per tool family: once the treatment arm has ≥8 resolved observations and its negative rate exceeds control by 0.25 (or reaches 0.5 without control data), the arbiter admits only reversible forms for that family. The feature stays on and safe; `reinstate()` or `reset()` clears it. Control observations never move strategy priors. `/health.tool_admission.feedback` reports arms and reverted families. `headroom agent status` summarizes `/health`: effective rollout of all 14 agent features (new `/health.agent_runtime.rollout`), host-enforcing vs observe-only capabilities, trusted workspaces, the store, JevK5 state, policies, holdout and counters. `doctor` adds read-only local checks. Neither command writes state.

## Files changed

171 files, +57,001/−321 against `76133cff`. Main components:

- `headroom/agent/` (31 modules + `tool_admission/` 8) and `headroom/intelligence/` (11).
- `headroom/ccr/indexed.py`, `ccr/mcp_server.py`, `transforms/safety_gate.py`, `transforms/content_router.py`.
- `proxy/server.py`, `proxy/handlers/{anthropic,openai}.py`, `rollout.py`, `paths.py`, `host_hook.py`.
- `cli/{agent,intelligence,wrap,main}.py`, `providers/claude` hook wiring.
- Rust: `crates/headroom-core/src/{safety_gate.rs,relevance/semantic.rs,transforms/live_zone*}`, `crates/headroom-py/src`, `crates/headroom-proxy/src/compression`.
- Tests: `tests/agent/` (Phase 1–8 gates and live host-hook tests), `tests/intelligence/`, `tests/test_safety_gate.py`, `tests/test_ccr_indexed.py`, Rust `live_zone_query_safety`, `safety_gate_parity`, `relevance_semantic`.
- Bench: `benchmarks/unified21_eval.py`, `benchmarks/results/*.json`.

## Tests (Phase 8, Windows; targeted, not the full suite)

| Set | Result |
|---|---|
| Rust `headroom-core --lib` | 994 passed, 0 failed, 1 ignored |
| Rust `headroom-proxy --lib` | 340 passed, 0 failed (an earlier run with the shell's `HEADROOM_FEATURES` exported failed 2 rollout tests; the clean env passes) |
| A: `tests/agent`, `intelligence`, `cli`, `test_cli`, `test_mcp_registry`, `test_install` | 1976 passed, 4 failed, 18 skipped |
| B: `test_proxy`, `test_cache`, `test_transforms`, `test_compression`, `test_providers`, `test_tokenizers`, `test_learn`, `test_storage`, `gateway`, `parity` | 2278 passed, 9 failed, 17 skipped |
| C: 323 top-level files (wrap, rollout, health, CCR, SafetyGate, byte-faithful, prefix/cache, Responses/Codex/OpenCode, streaming, tool search, output shaping, savings, offline/stateless/egress, pipeline, interceptors) | 5743 passed, 25 failed, 50 skipped |
| New in Phase 8 | `test_phase8_holdout_revert` (7), `test_phase8_agent_cli` (3), eval harness (+2): all pass |
| Live host-hook gates (real hook subprocess + uvicorn proxy) | Phase 4/6/7 gates: 12 passed, including "one workflow call replaces 3 model tool turns" |
| After the CLI commit: agent, intelligence, CLI, health, eval (`-n 8`) | 1542 passed, 4 failed (2 known; `test_phase7_test_impact` real-pytest run and the eval subset failed under load and pass serially) |

**Failure triage.** All 38 distinct failures were rerun serially, then on clean worktrees of `1ca86bef` (pre-Phase 8) and `76133cff` (upstream v0.41.0, its own `_core` built), both since removed.

- **7 failed only on this branch, fixed in tests** (e4bd3c8c). The upstream tests had not been updated for earlier phases' intended interface changes. Two client-decision tests (×2 params) now also assert that the resolved `CompressionPolicy` is passed. Two plumbing tests with fabricated summaries use the existing `permissive_lossy_budget` fixture. The buffered-timeout CCR double gains the real handler's `workspace` kwarg. No assertion was removed.
- **Also failing on upstream v0.41.0 in this environment (25):** `test_context_tool_cleanup` ×7, `test_wrap_code_memory` serena ×3, `test_unwrap_claude…`, `test_wrap_stale_marker` signal handler, kompress `thread_caps`, `test_install` planner/runtime, `test_learn` ×2 (Windows paths), `test_storage` sqlite init, orphan watchdog, fsync, TCP keepalive proxy mounts, ast-grep slice, cache breakpoint diagnostics log, ML stage deadline, Codex init config. These are Windows/host-environment failures, all also failing on clean v0.41.0.
- **Order- or flake-dependent, same on both trees:** `gateway/test_parity_proxy_vs_gateway` (an unmocked litellm price fetch; fails standalone on both trees), `test_cc_switch_reconciler` (1 of 3 fails on both), `test_savings_ledger_offload` concurrent append (fails 3 of 3 on HEAD, 2 of 3 on `1ca86bef`; Phase 8 does not touch the savings ledger).

**Platform.** Linux (WSL2 Ubuntu 22.04, Python 3.10, ext4) was qualified in the Phase 7 follow-up. Results: Rust 993 + 340 passed. Python targeted set 2894 passed; after serial rerun, 7 remained: 3 `tomllib` tests needing Python 3.11, 1 MCP collection error needing 3.11, `canonical_pipeline` (fixed now), and an environment PATH case. Two real bugs were fixed: drvfs case folding and zombie reaping. JevK5 on CPU is correct but takes 2.6–2.9 s, so the advisor correctly disables itself. Phase 8 changes no platform-sensitive code (holdout arithmetic, a `/health` field, a read-only CLI), so Linux was not rerun. Not covered on Linux: GPU llama-server, managed CMake build, real Claude/Codex/OpenCode binaries, JS runners.

## Benchmarks (`benchmarks.unified21_eval`; offline, stateless, 10 Anthropic + 8 Responses fixtures)

Same harness for every row. The baseline row is a live run on a temporary v0.41.0 worktree (`baseline_v0410_live.json`); it matches the recorded Phase 0 `baseline.json` to the token. Upstream has no wrap rollout, so the wrap path did not exist there. Latency is per-request pipeline time, nearest-rank p50/p95 over the fixtures. The bare-Python p95 includes the first fixture's one-time warm-up, roughly 2.7 s in every configuration.

| Config | Path | Tokens before → after | Reduction | Must-keep recall | Errors / structure / prefix | p50 / p95 ms |
|---|---|---|---|---|---|---|
| Baseline v0.41.0 | Python bare | 75,997 → 53,659 | 29.39% | 1.00 | 0 / 0 / 0 | 7.3 / 2737 (warm p95 80) |
| Baseline v0.41.0 | Rust live zone | 48,857 → 14,323 | 70.68% | 1.00 | 0 / 0 / 0 | 1.2 / 6.9 |
| JevK5 OFF | Python bare | 75,997 → 53,659 | 29.39% | 1.00 | 0 / 0 / 0 | 8.9 / 2772 (warm p95 52) |
| JevK5 OFF | **Python wrap** | 75,997 → **33,477** | **55.95%** | 1.00 | 0 / 0 / 0 | 22.1 / 55.2 |
| JevK5 OFF | Rust live zone | 48,857 → 31,664 | 35.19% | 1.00 | 0 / 0 / 0 | 4.7 / 15.5 |
| JevK5 ON | **Python wrap** | 75,997 → **31,817** | **58.13%** | 1.00 | 0 / 0 / 0 | 256.5 / 301.3 |
| JevK5 ON | Python bare / Rust | unchanged from OFF | 29.39% / 35.19% | 1.00 | 0 / 0 / 0 | 7.4 / 2786; 4.2 / 14.5 |

**JevK5 ON** (`phase8_jevk5_on.json`). Real managed llama-server started by `headroom intelligence setup --no-build --no-download`: 4B v0.3 Q8_0, SHA verified, RTX A4000, conformance passed; activation took 1.6 s. Per-request advisor budget: 10 arbiter advice requests, 10 successes, 0 timeouts or failures. **5 of 10 cleared the 0.60 confidence floor and were blended in, an acceptance rate of 50%.** Advisor call latency: p50 236 ms, p95 263 ms. A repeat run gave identical tokens and the same 5/10. The service was stopped afterwards (`stopped managed llama-server`), and none was left running.

**Is JevK5 ON better?** Tokens: modestly, −1,660 forwarded tokens (+2.18 pp), with recall unchanged at 1.0. The gain is not uniform:

- On `same_output_reread`, the advice picked a smaller safe form: 6,369 → 3,750.
- On `file_read`, it shifted the blend to the exact original: 2,747 → 3,706, a loss.
- The other 8 fixtures were byte-identical.

The cost is about +235 ms of local latency per advised unit. A 10-fixture offline eval cannot show a quality or task-success gain from JevK5, so none is claimed.

**Upgraded vs baseline.**

- The wrap path saves 55.95% against 29.39% for the same pipeline without the wrap rollout. That is 20,182 fewer forwarded tokens, at recall 1.0.
- The Rust live zone falls from 70.68% to 35.19%. This is the enforced irreversible-loss budget, because the binding has no retrieval path: search and log output now cut about 42% instead of 98%, and a context-trimming diff is rejected.
- Phase 5's arbiter keeps a full traceback (`tool_error` 71% → 0%) because the folded view lost visible facts.

**Workflow turns** (Phase 7 live gate, rerun this phase): one `headroom_workflow` call replaced 3 model tool turns, an estimated 2 turns avoided per run. A stale scope cancelled the run with 0 steps executed.

**Not measured:**

- Provider-billed savings, cache-hit/write costs and task success. The eval is offline and makes no provider calls.
- Live re-reads and retrievals. The harness records 0 `headroom_retrieve` turns in every configuration.
- The holdout itself: its fraction defaults to 0, and the revert logic is verified by tests, not by traffic.

## Provider/platform coverage (plan 8)

| Surface | Verified here |
|---|---|
| Claude Code `wrap` | Proxy live zone and caching, tool catalog, PreToolUse (scope, contracts, workflow ticket) and PostToolUse (admission) via `--settings`, state and evidence, tests and macros. Live hook-subprocess gates pass. Real 2.1.301 Grep/Glob parity is recorded. |
| Codex `wrap` | Responses HTTP/streaming/WS contract tests, exact call ids, state and evidence. Pre-tool effects are `observe_only` (no host API). |
| OpenCode 2.x `wrap` | Current plugin path (`headroom_retrieve`, transport, `shell.env`) and routing tests. Observe-only. |
| Python proxy | First-party hooks, exact CCR, typed advisor, savings attribution, tool-free requests stay tool-free. |
| Native Rust proxy | v0.41 `/stats` and dashboard unchanged (proxy lib 340 pass). It shares the SafetyGate and live zone through the core. It does not run Python hooks, and the report does not claim it does. |
| Windows / Linux | Windows: everything above. Linux: see Platform. No Docker or root required, and no console windows (`CREATE_NO_WINDOW`). |

## Configuration, disable and rollback (documented once, here)

- **Rollout names:** `agent_runtime`, `agent_task_state`, `agent_evidence`, `agent_code_graph`, `agent_tool_admission`, `agent_arbiter`, `agent_tool_catalog`, `agent_complexity_routing`, `agent_feedback`, `agent_speculation`, `agent_change_scope`, `agent_tool_contracts`, `agent_test_impact`, `agent_workflow_macros`. Disable with `HEADROOM_DISABLE_FEATURES=a,b`. `agent_runtime` off removes the hook entirely.
- **Enforcement policies** are separate from rollout: `HEADROOM_CHANGE_SCOPE_MODE` and `HEADROOM_TOOL_CONTRACTS_MODE`, each `observe|warn|protect`, default `warn`. Blocking happens only at a trusted host pre-tool hook.
- **JevK5:** `HEADROOM_JEVK5=auto|on|off` is the service policy. `HEADROOM_JEVK5_LLAMA_URL` points at an operator-managed llama-server (`/health`, `/tokenize`, `/completion`). `HEADROOM_JEVK5_SYSTEMONE_URL` points at a `/v1/systemone` service. One API is never inferred from the other. Model override: `HEADROOM_JEVK5_MODEL` plus `HEADROOM_JEVK5_MODEL_SHA256`, or `…_OPERATOR_MANAGED=1`. Limits: `HEADROOM_JEVK5_DECISION_TIMEOUT_MS` (1200), `…_CALLS_PER_REQUEST` (4), `…_MIN_CONFIDENCE` (0.60), `…_WEIGHT` (0.25, cap 0.40). Stop the service with `headroom intelligence stop`.
- **Holdout:** `HEADROOM_TRANSFORM_HOLDOUT` (0–0.5) for transforms and `HEADROOM_OUTPUT_HOLDOUT` for effort/verbosity.
- **Offline and stateless:** `HEADROOM_OFFLINE=1` means no Headroom-initiated egress; JevK5 then needs an installed binary and a verified model. `HEADROOM_STATELESS=1` means memory stores only and no new files. Data lives under `HEADROOM_WORKSPACE_DIR` (default `~/.headroom`): `agent/agent_state.db`, `agent/agent_ledger.db`, `intelligence/`.
- **Retention** (`agent/retention.py`): events 30 days (`HEADROOM_AGENT_EVENT_RETENTION_DAYS`); per-session projections, evidence facts and resource versions 30 days after the session's last update (`…_SESSION_RETENTION_DAYS`); workspace aggregates (verification runs, workflow observations and macros, feedback priors, graph index) 90 days (`…_WORKSPACE_RETENTION_DAYS`). A commit listener starts at most one prune every 6 h on its own daemon thread; it never runs on the runtime worker, `flush` or a request. Stateless memory stores keep only their count bounds. The in-memory learned-recovery table (7-day TTL) is not persisted, so there is nothing to prune there. CCR keeps its own TTL; at most 32 active macros per workspace. `headroom agent purge --workspace <dir>` deletes one workspace's rows from the two agent databases only.
- **Host hooks:** `wrap claude` passes them through `--settings` and edits no file. `--project-settings` writes them under the lock and they are removed on the last session or by `unwrap`. User `--settings` win.
- **Restore / rollback:** disable features as above, or check out `backup/main-pre-unified21`. Inspect with `headroom agent status|doctor` and `headroom intelligence status|doctor`.

## Follow-up: concurrent wrap vs task-managed persistent deployments

Reported live on Windows (where `install` downgrades `persistent-service` to
`persistent-task`, planner.py #2552) with a `default` deployment on 8787: a
second `headroom wrap claude --1m --dangerously-skip-permissions --memory
--learn --code-graph` failed with `task-based deployment 'default' cannot be
auto-recovered via wrap` followed by `Persistent deployment 'default' on port
8787 is not healthy.`

**Root cause.** `wrap` asked only "is the deployment ready?" and had no state
for "a Headroom proxy is serving this port but the deployment does not own it".
Two defects followed from that:

1. The healthy-deployment branch treated a *capability* shortfall as a reason to
   fall through to the ephemeral restart path, which kills the PID the running
   proxy reports and rebinds the port with a wrap-owned proxy. On a task
   deployment this destroyed task ownership: the PID file then named a dead
   process while a live proxy held 8787. `--learn` and `--code-graph` were not
   expressible in a manifest at all, so *every* wrap carrying them took this
   path. (`tests/test_cli/test_wrap_persistent.py` previously asserted this
   kill as correct behaviour.)
2. Once ownership was broken, `runtime_ready` was False while a proxy answered.
   `_recover_persistent_proxy` declined (task deployments are the scheduler's to
   start) and the next branch raised unconditionally — without ever asking
   whether the live listener could serve the session.

**Fix.** `runtime_claim` (install/runtime.py) reports whether a deployment still
holds a live runtime process, separately from whether it is *ready*;
`unverified` (Windows with neither psutil nor argv markers) counts as owned, so
an unprovable claim is never grounds for taking the port. `_persistent_port_state`
(cli/wrap.py) classifies the port as deployment-healthy / -starting / -unhealthy,
unowned-headroom, foreign, or free, and each state has a defined action. wrap
never kills or rebinds a port a deployment still owns, never rewrites its
ownership state, and never fails the session: it shares a listener whose routing
and capabilities match, and otherwise starts a dedicated proxy one port up.
A capability shortfall is reconciled through the deployment's own lifecycle only
when the manifest declares the capability (so the restart can actually supply
it) and nothing else is attached; otherwise the session gets its own proxy.

**Capability classification.** `learn` became a manifest capability
(`learn_enabled`, `install apply --learn`, implying `--memory` as the proxy CLI
does) because the learner is user/agent-scoped and writes through the memory
backend. `code_graph` did **not**: `CodeGraphWatcher(project_dir=Path.cwd())`
binds to the proxy process's own working directory, so a shared runtime would
index the supervisor's directory while reporting success. It is classified
session-scoped and always answered with a dedicated proxy.

Files: `headroom/cli/wrap.py`, `headroom/install/{runtime,planner,models}.py`,
`headroom/cli/install.py`, `wiki/troubleshooting.md`. Tests:
`tests/test_cli/test_wrap_concurrent_persistent.py` (new, 18 cases covering the
reported command run twice, concurrent reuse, stale PID beside a live proxy,
unverifiable identity, an `ensure` race, routing strictness and start-lock
serialization) plus additions to `tests/test_install/test_{runtime,planner}.py`.

## Open issues

### Intentional design choices
- Phase 7 (test selection, macro promotion, authorization) is deterministic and never consults JevK5.
- Coverage uses existing coverage.py data only; stale or foreign data is reported as unavailable.
- A reverted family falls back to reversible forms, not to "off". Reverts are process-local, so a restart re-measures.
- Workflow Grep content/count output is path-ordered, because the host leaves that order unspecified. Read/search tools of hosts without a verified dialect are never generalized.

### Host capability limitations
- Codex, OpenCode, Copilot, Cursor, Aider and Gemini expose no Headroom pre-tool boundary, so scope, contracts and workflows only observe for them. Pre-context admission is Claude-only; the others use the proxy fallback.
- Grep/Glob workflow steps are pinned to Claude Code 2.1. Prefix-dependent slash globs and context options are ineligible. The bundled `rg` may differ from the operator's.
- The native Rust proxy has no Python hooks or admission (separate runtime).

### Verified platform limitations
- WSL drvfs/9p: no POSIX modes, no AF_UNIX sockets or FIFOs, 100 ns timestamps. Keep the workspace on a native filesystem.
- JevK5 on CPU exceeds 1200 ms, so the advisor disables itself. Python 3.10 cannot run 3 upstream `tomllib` tests.
- 25 targeted Windows tests also fail on clean upstream v0.41.0 in this environment (listed above).

### Actual unfinished implementation
- **The full test suite has not been run** (user direction); final completion per the plan awaits it (command above).
- Benchmarks are offline. Provider-billed savings, cache costs, task success and live holdout outcomes still need real traffic.
- Not covered on Linux: GPU llama-server, managed CMake build, real client binaries, JS runners.
