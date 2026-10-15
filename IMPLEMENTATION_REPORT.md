# Unified-21 Implementation Report

Spec: `HEADROOM_WRAP_UNIFIED_21_FEATURE_IMPLEMENTATION_PLAN_OPUS_5_5.md`. One short section per phase.

## Phase 0: safety, sync, inventory, baseline

**Checkout and sync.** `feature/unified-21` starts at upstream tag `v0.41.0` (`76133cff`), which matches the plan's SHA. The remote `origin` is the upstream `headroomlabs-ai/headroom` repo only, with tags fetched. The old HEAD `d13e1966` (the 0.39.1 release) had no local commits ahead of `origin/main`. Basing the branch on v0.41.0 therefore counts as the merge: a fast-forward with no conflicts in tracked code.

**Safety refs.**
- Branch `backup/main-pre-unified21` at `d13e1966`.
- Stash `stash@{0}` "pre-unified21 user changes", kept.
- Copies of `.claude/.headroom_wrap_settings.lock` and `.serena/project.yml` are in `../_safety_backup_20261014_main/`.
- The lock file was deleted upstream in 0.41. The user's copy is restored on disk as an untracked, ignored file.
- These user files are untouched: `AGENTS.md`, the plan, `.clinerules-*`, `.claude/*` and the modified `.serena/project.yml`.

**No Paritok/Jev code here.** This checkout is plain upstream. It has no Paritok, JevK5 or llama-server code. The user's shell does export about 40 Paritok-era `HEADROOM_*` variables, for example `HEADROOM_FEATURES=…,paritok_content_compress,paritok_history_summarize`, `HEADROOM_ROLLOUT_CHANNEL=beta`, `HEADROOM_SYSTEM_ONE_MODEL=alibiserikbay/JevK5` and `HEADROOM_SYSTEM_ONE_ENDPOINT=http://127.0.0.1:8090`. Most name features that v0.41 does not have. Tests and evals must run with them unset.

**Environment** (Windows 10 19045):

| Item | Version or detail |
|---|---|
| Python (repo venv) | 3.12.10 at `.venv/Scripts/python.exe`, created with uv 0.7.9 |
| Headroom | 0.41.1-dev, editable install |
| Install command | `uv pip install --python .venv/Scripts/python.exe -e ".[dev,proxy,code,relevance,mcp,reports]" pytest-xdist pytest-timeout` |
| Rust extension | Maturin builds `headroom/_core.pyd` (gitignored) during the editable install |
| Rust toolchain | rustc/cargo 1.95.0, matching `rust-toolchain.toml` |
| C compiler | MSVC 19.29 (`cl`) and 19.44 (link) |
| llama-server | WinGet build `version: 3 (44da65e)`, MSVC 19.44 |
| GPU | NVIDIA RTX A4000 16 GB, driver 582.16 |
| Optional libraries | tree-sitter (language pack) and fastembed both import |

**Disk space blocker.** The C: drive is full, with 65 MB free. That caused MSVC `LNK1108` link failures and SQLite "database or disk is full" errors under the default `%TEMP%`. Point TMP and TEMP at `target/tmp/t` on the F: drive (gitignored) before building or testing.

**Commands that work.**
- Python: `env -u <every HEADROOM_*/ANTHROPIC_*/OPENAI_API_KEY> TMP=… TEMP=… .venv/Scripts/python.exe -m pytest -p no:cacheprovider -n 8 --timeout=300 -o addopts="" <paths>`
- Rust: `CARGO_TARGET_DIR=$PWD/target cargo test -p headroom-core --lib` and `cargo test -p headroom-proxy --lib`

**Inventory** (status relative to the plan feature; paths are real entry points):

| ID | Status | Evidence |
|---|---|---|
| 1 | absent | `crates/headroom-core/src/transforms/live_zone.rs:145` sets `EMPTY_QUERY = ""`. It is used at `:1660` (SmartCrusher), `:1684` (Search) and `:1696` (Diff). `dispatch_compressor` at `:1637` takes no query. |
| 2 | partial | `relevance/hybrid.rs:69` defines `HybridScorer`. `embedding.rs:45/92` has `try_new` but nothing outside tests calls it, so the default is BM25 only (`bm25.rs:40`). In `relevance/mod.rs:44`, "embedding" returns an error. |
| 3 | partial | Exact retrieval by hash: `cache/compression_store.py:458`, `core/src/ccr/mod.rs:40`. Tool injection: `ccr/tool_injection.py:189`. MCP server: `ccr/mcp_server.py:378`. There is no search index. |
| 4 | partial | `proxy/interceptors/base.py:192` and `transforms/read_maturation.py:134`. Compression happens after admission, with no pre-context externalization. |
| 5 | partial | `transforms/cross_turn_dedup.py:86` does line-level dedup only. Prefix tracking: `cache/prefix_tracker.py:1137`. |
| 6 | absent | Dispatch is single-path: `live_zone.rs:1637` and `transforms/content_router.py`. |
| 7 | partial | `compress.py` `_verify_and_revert` checks lossless only. Rust has a per-block token gate. |
| 8 | absent | `transforms/adaptive_sizer.py` sizes each item separately; nothing allocates across items. |
| 9 | absent | Only native tool-search env and hints: `cli/wrap.py:418`, `proxy/helpers.py:3570`. |
| 10 | partial | `proxy/model_router.py:147` (opt-in), `verbosity_controller.py:59`, `output_shaper.py` |
| 11 | absent | `graph/watcher.py:96` only reindexes the external codebase-memory MCP server. tree-sitter is used only for compression (`transforms/code_compressor.py`, Rust `code_compressor.rs`). |
| 12 | partial | `cache/compression_feedback.py:161` and `ccr/context_tracker.py:126` |
| 13 | partial | `transforms/compression_policy.py:91` and `core/src/compression_policy.rs:162` cover auth mode and cache only. There is no lossy-risk policy. |
| 14 | partial | `proxy/interceptors/astgrep.py:311` handles Read outline only. Other output types are handled after the fact by the compressors. |
| 15, 16, 17, 19, 20, 23, 24 | absent | No code. `memory/traffic_learner.py:463` learns patterns but has no macros. `SavingsLedger` tracks savings, not provenance. |
| JevK5 | absent | No code in this repository. |

**Anchors to reuse.**
- `proxy/turn_hooks.py`: TurnContext `:44`, `record_savings` `:69`, TurnHook `:94`, `register_turn_hook` `:141`
- `proxy/extensions.py:117,163`
- `rollout.py`: `FEATURES` `:98` (4 features), `RolloutSnapshot` `:219`, `resolve_rollout` `:380`
- `proxy/server.py`: `HeadroomProxy` `:945`, `anthropic_pipeline` `:1148`
- Python handler: `handlers/anthropic.py:1971+` calls `anthropic_pipeline.apply`
- Rust proxy: `crates/headroom-proxy/src/proxy.rs:908-946` calls `compression::compress_*_request`
- headroom-py: `#[pymodule]` at `crates/headroom-py/src/lib.rs:1803`, which exports the compressors and `compress_openai_responses_live_zone` (`:1597`)

**Smoke tests** (targeted; clean env; TMP on F:):
- Rust: `headroom-core --lib` 976 passed / 0 failed / 1 ignored. `headroom-proxy --lib` 340 passed / 0 failed.
- Python (wrap claude/codex/opencode, live zone and PyO3, compression policy, CCR, byte fidelity and prefix cache, tool deferral, memory, output shaping, eval harness): 30 files, `-n 8`: **831 passed, 3 failed, 12 skipped** (skips = live-API tests without keys).
- Pre-existing failures are listed in `../_safety_backup_20261014_main/phase0_preexisting_failures.txt`. The 3 failures are xdist worker crashes in `test_memory_system.py` under 8-way load; that file alone passes 104/104 both serially and with `-n 8`. In the user's raw shell, 2 live tests return 401 (invalid exported `OPENAI_API_KEY`). With C:-based TEMP, the disk-full errors were SQLite and link failures.

**Baseline** (`benchmarks/unified21_eval.py` produced `benchmarks/results/baseline.json`; offline, stateless, 11 fixtures):

| Path | Raw tokens | Forwarded tokens | Reduction | Must-keep recall | Errors / structure / prefix issues |
|---|---|---|---|---|---|
| Python proxy `anthropic_pipeline` (coding profile) | 75,997 | 53,659 | 29.39% | 1.00 | 0 / 0 / 0 |
| Rust live zone (Responses, via headroom-py) | 48,857 | 14,323 | 70.68% | 1.00 | 0 / 0 / 0 |

- The Python path saves nothing on file reads, pytest output, tool errors, conflicting reads or the cache-hot delta. Large search output is excluded, by tool type, from compression. The service log drops 97%, JSON 70% and the same-output reread 49%.
- Rust leaves pytest output and file reads unchanged. It cuts search and log output by about 98% and the diff by 50%.
- No `headroom_retrieve` turns were needed.

## Phase 1: runtime foundation and JevK5 (commits f8db3eed, 6d2e75a2, this entry)

**1A runtime (`headroom/agent/`).** The plan's v1 kinds are reserved as `AgentEvent`s. Events store only digests, and the idempotency key is (session, source, kind, index, digest). Normalization is read-only and covers Anthropic, OpenAI chat, OpenAI Responses and Google. A message counts as authentic USER input only after harness wrappers are stripped. Identity: a session is trusted only when its wrap token matches a live marker and the registered cwd still exists. `workspace_id` hashes the principal plus the worktree root, so linked worktrees are distinct scopes. `session_id` is a hash of the token. The SQLite store runs in WAL mode under `paths.agent_state_db_path()`. It has a schema version, inserts events and their projection in one transaction, uses CAS on revisions, rebuilds by replay and salvages corrupt files read-only. In stateless mode an equivalent bounded in-memory store is used and nothing is written to disk. A background worker with a bounded queue keeps SQLite off the event loop. One stream-safe TurnHook at priority 80 is installed once from `create_app`, gated by the new opt-in rollout feature `agent_runtime`. With the feature off, no hook is registered. `/health` reports the effective rollout, whether the hook is installed, the trusted-marker count, runtime counters and advisor state. When wrap reuses a proxy, it reports mismatches in `agent_runtime` and `HEADROOM_JEVK5`.

**Phase 2+ extension APIs.** In `headroom.agent.registry`: `register_request_stage`, `register_projector`, `register_commit_listener`, `register_component`. In `headroom.agent.events`: `register_event_kind`, `make_event`. In `headroom.agent.interfaces`: `TaskContext`, `DecisionHints`, `ToolResultEnvelope`, `ActionAuthorization`. On the runtime: `AgentRuntime.authorize_action`, which returns an authorization only for trusted identities. From a TurnContext: `headroom.agent.hook.task_context_for(ctx)` and `identity_for(ctx)`. For JevK5: `IntelligenceManager.advisor`, i.e. `DecisionAdvisor.choose|noul|score|adecide` plus `request_budget()`.

**1B JevK5 (`headroom/intelligence/`).**
- Typed settings: policy `HEADROOM_JEVK5=auto|on|off`; calibration T=1.22, knockout T=0.93; ctx 8192; deadline 1200 ms; 4 calls per request; cache 30 s; minimum confidence 0.60; weight 0.25 capped at 0.40; circuit breaker opens after 3 failures for 60 s.
- Setup checks GGUF metadata and SHA-256 and resumes downloads from `.part` files. A custom model must supply its own hash or be marked operator-managed.
- llama-server is discovered in this order: override, manifest, PATH, `LLAMA_CPP_HOME`. If none is found, setup builds a managed one with CMake.
- The managed service starts or is reused on 127.0.0.1 (port 8091 preferred). Before any stop, Headroom checks PID, executable and start time, and it never kills a foreign process.
- Every request in a decision shares one monotonic deadline. A fresh adapter is created per decision. If answer letters are missing, it retries once with n_probs=128. A separate typed client handles `/v1/systemone`.
- CLI: `headroom intelligence setup|status|doctor|stop`. The JevK5 transport has an operator-endpoint entry in the egress allowlist.

**Tests.**
- New: 62 passed in `tests/agent` and `tests/intelligence`, using a fake llama-server.
- Live run (`HEADROOM_JEVK5_LIVE_TEST=1`, RTX A4000, cached verified 4B Q8_0 model, WinGet llama-server b44da65e): setup OK and conformance passed. Refund gave noul 0.06, i.e. no. Parcel gave misdelivered at 0.95. Decisions took about 140–156 ms. An 8-way concurrency isolation run passed, and the service stopped cleanly.
- Existing targeted suites: wrap/CLI/rollout 749 passed; paths/stateless/workspace 265 passed; hooks/health/CCR slice 416 passed.
- Pre-existing failures, which also fail on the clean Phase 0 worktree: `test_turn_hook_usage` (4), `test_anthropic_buffered_ccr_usage` (6) and 2 `TestHeadroomCloudCompressionOffline` cases, all from an unmocked litellm price fetch.

**Eval** (`benchmarks/results/phase1.json`): identical to baseline. Python is 29.39%, Rust 70.68%, recall 1.0, 0 errors. Warm wall time is 2900 ms against 2888 ms in the baseline.

## Phase 2: task state, evidence, code graph, query model (commits 63a1ca94, 70933fb8, this entry)

**Wiring.** `headroom/agent/builtin.py` is installed from `install_agent_runtime` (so from `create_app`). Each component has its own rollout feature: `agent_task_state`, `agent_evidence` and `agent_code_graph`. They are opt-in for a bare proxy. `headroom wrap` adds them, plus `agent_runtime`, through `rollout.apply_wrap_feature_defaults`. `HEADROOM_DISABLE_FEATURES` still turns any of them off. The services are registered on `runtime.services` as `task_state`, `evidence`, `code_graph` and `ledger`, and `/health` reports their counters. Relational state lives in `agent_ledger.db`, which sits beside the event store and is in memory when stateless. Request stages never touch SQLite: when a snapshot is missing, they queue a warm-up read with `runtime.submit_task`.

**2A Task state (#16).** Atoms are typed and carry their origin, authority, source events, evidence ids and supersession link. Corrections supersede only the atom they target. JevK5 `choose` may pick only among existing atoms, and an unknown id is ignored. An agent's "done" stays PROPOSED. Completion needs user acceptance, or acceptance criteria satisfied by test evidence. A truncated result never counts. Transitions apply through projection CAS, with one retry on a stale revision, then rejection and a counter. They are written as digest-only `TASK_STATE_CHANGE` events; atom text sits content-addressed in the ledger, so replay rebuilds state exactly. Only trusted (`ws_`) scopes are compiled, and state persists across trimmed history. A priority-80 stage (order 50) appends one stable, hash-tagged block after the live turn. On Anthropic it is a mid-conversation `system` message, sent only to models that support one (the shared `anthropic_supports_mid_conversation_system`). On Responses it is a trailing `developer` item. The system prompt, cached prefix and latest user message are never modified. The budget is min(900 tokens, 2% of the window, the tag `agent_state_budget_tokens`); user constraints and blockers fill it first. The block is omitted when it is unchanged in context, when the request is untrusted, or when the current turn is a correction or a new task.

**2B Evidence (#17).** Facts are extracted from test, exit, error, file, hash, git, tool-availability, port and user-statement events. Negative facts are first-class. Values are scrubbed of secrets and bounded at 160 characters, and no whole files are stored. Claim keys are normalized; a newer observation supersedes an older one, and corroboration raises confidence by source type. A conflict between a user statement and an observation is labelled CONTRADICTED until resolved. Volatile facts have a TTL, after which they go STALE. File edits stale facts tied to the old content, and git mutations stale git state. Retraction is supported. `EVIDENCE_CHANGE` events allow replay. Indexes cover (ws, claim_key, status), (task, status), source_event_id and time, and every read is scoped by workspace and session.

**2C Code graph (#11).** It reuses the compression tree-sitter parser cache, the ast-grep runner (only through `binaries.which`, so it never downloads) and the watcher's extension and ignore tables. When a language is unsupported, it falls back to lexical extraction. Nodes and edges are file/module/class/function and imports/includes/calls/references/inherits/contains/defines. Each worktree keeps incremental hashes, persisted in `graph_files`. Updates come from `FILE_CHANGE` events and a stat rescan limited to once every 60 s. Vendor, generated and gitignored paths are skipped. Indexing runs on one background thread.

**2D Query model.** `TaskContext` was extended with optional fields; it is not replaced. `RelevanceQuery` holds goal, intent, exact entities, active error and graph neighbors (graph only for code tasks). It is capped at 512 tokens with entities budgeted first. It is built in the hook from authentic user text only.

**Tests.** `tests/agent`: 75 passed: 43 new, 32 pre-existing; the new tests are in `test_task_state`, `test_evidence`, `test_code_graph`, `test_query` and `test_phase2_integration`. 780 passed and 1 skipped across nearby suites: agent, intelligence, turn hooks, handler helpers, rollout, agent savings, proxy env, byte-faithful forwarding and the wrap CLI.

**Eval** (`benchmarks/results/phase2.json`): identical to Phase 1. Python is 29.39%, Rust 70.68%, recall 1.0, 0 errors. The eval exercises compression only, and Phase 2 has no lossy transforms. Warm wall time was 4456 ms against 2900 ms, measured right after a large test run; the eval path contains no agent code.

## Phase 3: SafetyGate, policy, semantic relevance, query-conditioned live zone, indexed CCR (commits 2eafd618, 93eb47a8, b0f4dc8d, 8dd4bb61, fbaf87e1, this entry)

**3A SafetyGate (#7, #13).** One implementation, `headroom_core::safety_gate` (`check`, `select`), exposed as `headroom._core.safety_gate_*` and wrapped by `headroom/transforms/safety_gate.py`. Hard invariants: user-named entities, failure lines and traceback `file:line` (search hits and JSON excepted, but a non-hit failure line in "search" output is protected), exit codes, test counts (changed counts reject), definitions of named symbols, diff headers, existing CCR ids, truncation boundaries, JSON syntax/bool/key facts. New markers must be verified (stored payload read back) and a retrieval tool must exist. `max_lossy_ratio` bounds *irreversible* loss; `wire_drop_ratio` is separate, so exact CCR externalization may drop more. `volatile_token_threshold` bounds volatile tokens a candidate introduces. Missing/invalid policy, no token savings, or a frozen-prefix block all reject. Ladder: aggressive → conservative (bias re-run, then policy-fit line selection) → lossless (JSON minify / router fold, verified reversible compaction) → CCR exact stub → original; a lossy rung no smaller than an admissible lossless rung is skipped. Enforced on every Rust live-zone block and every Python router candidate (string, block, cache hit, Responses unit and batch). Rust proxy passes the resolved policy. Shared fixtures: `tests/fixtures/safety_gate_parity.json`.

**3B Semantics (#2).** `relevance::semantic`: lazy once-per-process background init, cache-only (Xenova/bge-small-en-v1.5 from `HEADROOM_FASTEMBED_CACHE_DIR`/HF caches), intra-op threads ≤4, 4096-entry embedding LRU, batched. `HybridScorer::default()` fuses it with an exact-identifier floor (0.75) and stays on boosted BM25 until ready; no `ml`, offline-without-cache, no ORT, no AVX2 or a miss → BM25, never retried per request. `semantic_relevance_prefetch` is the only download path and refuses without an operator directory. Verified both ways: fallback, and a real model in `target/fastembed` (F:) loads once (~0.4 s) with paraphrase ranking.

**3C Query-conditioned live zone (#1).** `EMPTY_QUERY` is gone. `LiveZoneOptions{query, policy, ccr_store}` and `compress_*_live_zone_with_options`. The query is the agent `RelevanceQuery` or is derived from the request's latest user text. It reaches SmartCrusher, search, diff and code; newest user prose is excluded; CCR originals are stored and verified before a marker. Python handlers build the RelevanceQuery before compression (`agent.hook.compression_relevance_query`, reused by the hook via `TurnContext.request_id`). The Responses unit cache key excludes the query so history stays prefix-stable (cached forms are re-gated).

**3D Indexed CCR (#3).** `headroom/ccr/indexed.py` adds spans for JSON rows, logs, traces, search hits, code symbols, diff hunks and paragraphs. Each span keeps exact byte ranges. Lookup uses FTS5 BM25, or lexical BM25 when FTS5 is unavailable; semantic and graph scores are hints. Cursors are HMAC-bound to (workspace, entry, generation, mode, query). Spans follow the parent entry's TTL and deletion. Selective modes serve only hashes reachable in the conversation. The tool schema adds optional `mode/query/top_k/cursor/start/end`; `hash` alone stays byte-exact. These modes are wired into the response handler (mixed calls), MCP, `/v1/retrieve*` and proactive expansion, which now returns selected spans ≤1200 tokens.

**Tests.** Rust: core lib 994 passed / 1 ignored; 16 live-zone/CCR/gate integration binaries pass, including the new `live_zone_query_safety`, `safety_gate_parity` and `relevance_semantic`; proxy lib 340 and proxy compression/Responses integration 41 pass. Python: new `test_safety_gate` (25) and `test_ccr_indexed` (10); targeted suites 2167 passed. Plumbing tests with fabricated summaries opt into `conftest.permissive_lossy_budget`. Pre-existing failures, reproduced on the pre-change tree: xdist-flaky `test_ccr_inline_resolve_handlers`/`test_ccr_row_drop_store_bridge`, `test_kompress…thread_caps`, `test_wrap_code_memory` serena ×3.

**Eval** (`benchmarks/results/phase3.json`): Python 29.39% (unchanged), Rust 35.19% (phase 2: 70.68%), recall 1.0, 0 errors/structure/prefix violations. The Rust drop is the policy working: the binding has no retrieval path, so PAYG caps irreversible loss at 45% (search/log 98% → 42%; the context-trimming diff is rejected; lossless JSON compaction still 67%). One eval run stalled at 0% CPU and was not reproduced in four reruns.

## Phase 4: host admission, structured delta, tool-family interceptors (commits 421dc838, 06751097, c1e9f097, this entry)

**Envelope (#4).** `ToolResultEnvelope` v2 extends the Phase-1 type (v1 still deserializes): `tool_family`, `session`, `workspace`, stable `resource_id`, `version_hash`, `content`, `stderr`, `source_metadata`, plus `call_id`/`truncation_reason`/`is_complete`. `resource_id` is a digest of the canonical key: workspace-relative path + range, normalized command + cwd, query + scope, or API args. The tool-call id is never part of it. Explicit partiality comes from range reads, pipes into `head`/`tail`, result limits and host flags, and is never inferred as complete. `to_metadata()` stays content-free.

**Dispatcher and storage sequence (`headroom/agent/tool_admission/`).** One `ToolAdmissionDispatcher` runs: completeness → version hash → exact original into CCR → read-back self-test → delta and family-view candidates → `SafetyGate.select` → replace only when a candidate is admitted. Any failure forwards the original (store error, failed self-test, no store, interceptor crash, gate reject, unknown policy). Outcomes are memoized per (scope, call, content), so later requests render the same bytes (prefix-stable).

**Interceptors (#14).** One per family, all reusing existing code:
- Read: code-graph/ast-grep outline that keeps the client's own line numbers and keeps user-named definitions whole. Edit targets, range reads and small reads stay exact. A reread after a lossy view is exact (progressive disclosure).
- Search: Rust `SearchCompressor`, grouped by file.
- Listing: grouped by directory; PARTIAL is stated.
- Shell and test: Rust `LogCompressor`, plus pinning of failures, exit codes, summaries and failing ids.
- Git: `DiffCompressor` hunks; status grouping keeps every path. A request for an applyable patch stays exact.
- JSON: SmartCrusher keyed rows.

**Delta (#5).** `ResourceVersionStore` keeps the last version per (scope, resource): hash, CCR hash, unit identities, complete flag and presented-exact flag. It is bounded and persisted to `tool_resource_versions` in the agent ledger on the runtime worker. `compute_delta` covers lines, keyed JSON rows, search rows, listing entries and test outcomes, and states base/current hashes and exact +/−/~/unchanged counts. A partial base or current is never compared. An unchanged reread becomes a reference only after the new version's exact bytes verify. A base must be visible in the same request (same call id and bytes), so no delta crosses conversations.

**Host adapter.** `wrap claude` installs `PreToolUse` (observe edit targets; prints nothing, so no permission effect) and `PostToolUse` (`hookSpecificOutput.updatedToolOutput` in the tool's own shape: Bash/PowerShell stdout, Grep/Glob filenames or content, MCP text blocks). A normal launch passes the hooks through `--settings` and touches no file; a user `--settings` wins and the session reports observational. `--project-settings` writes them under the settings lock with holder tracking; the last session and `unwrap` remove them. Read is left to the proxy, because Claude Code renumbers Read content. The hook command `python -m headroom.host_hook` is stdlib-only, loopback-only and fail-open (prints nothing, exit 0) and posts to `/v1/agent/host-hook`. Every other client reports `observational_or_proxy_only`. The proxy fallback `ToolAdmissionTransform` is the first pipeline transform and never touches the frozen prefix; `/health.tool_admission` reports both. Rollout feature `agent_tool_admission`: wrap default, opt-in for a bare proxy.

**Tests.**
- New: `tests/agent/test_tool_admission.py` (44), `test_tool_admission_wrap_proxy.py` (5: settings preservation, real `/v1/messages` fallback with exact `/v1/retrieve`) and `test_host_hook_live.py` (3). The live tests run the real hook subprocess against a uvicorn proxy with real hook JSON; host-visible Bash bytes are cut more than 3× and the original is recovered exactly.
- Targeted: 346 passed in agent/interceptors/rollout/gate/CCR/wrap-settings; 512 passed in wrap CLI; 204 passed in prefix/CCR-marker/pipeline.
- Failures that predate Phase 4 (both reproduce on a clean `ff3f18c9` worktree): `test_unwrap_claude_removes_mcp_purges_retired_hook_and_stops_proxy` and `test_wrap_stale_marker::test_signal_handler_unwinds…`. `test_ccr_marker_resolution` fails only under xdist.

**Eval** (`benchmarks/results/phase4.json`): Python bare proxy 29.39% (unchanged; admission is off there), Rust 35.19% (unchanged), recall 1.0, 0 errors/structure/prefix violations. A new path, `python_proxy_anthropic_wrap` (wrap rollout), reaches **56.91%** with recall 1.0. Read, search, pytest, tool_error and git_diff go from 0–7% to 26–94%. `same_output_reread` drops from 49% to 14%: the reread is now exact by design. Not covered: Windows was tested, Linux was not run in this phase, and the native Rust proxy has no admission (it is a separate runtime).

## Phase 5: budget + arbiter, progressive catalog, complexity routing, feedback, speculation (commits 8b5ccb95, 58506ed8, 03f9631b, this entry)

**5A Budget and arbiter (#8, #6).** `headroom/agent/budget.py` computes the live-zone budget from the handler's context-guard effective limit (learned cap honoured; `context-1m` honoured only for models with a real 1M window; `HEADROOM_CONTEXT_GUARD=0` keeps the registry window), minus frozen prefix, system/tools, `max_tokens` (else 10%, ≤32k) and a margin (max(1024, 2%)). `x-headroom-keep-last-turns` is recorded; task state is not in trimmed history. The allocator is order-independent: each unit's smallest invariant-safe candidate first, then optional tokens by (relevance + novelty + anomaly + 1−recoverability)/token after a 15% per-source diversity share; an unfittable floor marks `insufficient`. `headroom/agent/arbiter.py` (`TransformArbiter`) takes over from the ladder in the dispatcher (`prepare → allocate → decide`). Candidates are original, lossless, type_specific, query_aware (`SafetyGate.fit`), structural, delta and indexed_externalization (CCR stub). Hard filters run first: SafetyGate check, then billed cost after cache multiplier (1.25×/2× on write) plus expected retrieval overhead. Survivors are scored with the spec weights (.28/.20/.16/.14/.10/.07/.05). Latency uses a deterministic estimate, so the winner never depends on scheduling. Ties go to less loss, then lower cost, then the original. JevK5 sees at most 16 finalists, only when the task is ambiguous or the margin is under .15, with weight min(w, .40); with no advice, w = 0. `RequestLimits` caps candidates and views per request. Memoized units stay byte-identical across requests.

**5B Progressive catalog (#9).** `headroom/agent/tool_catalog.py` runs after server-side tool search, only when the tools array does not defer (client `ENABLE_TOOL_SEARCH`/`defer_loading` or Headroom deferral untouched), never on tool-free requests, and on ≥12 tools. Resident tools (core set, `ToolRegistry` residents such as `headroom_retrieve`, strict schemas, typed server tools) and the top 5 ranked tools keep their full compacted schema. The rest become callable stubs: purpose + `name(req: type, opt?: type)` with required/types/enums and `cache_control` kept. Ranking: lexical, Rust hybrid scorer, graph neighbours, recent successful use, feedback prior; JevK5 breaks ambiguous top-16 ties. The set is sticky per (session, tool digest) and never shrinks on a hot prefix. It grows only for `tool_choice`, a user-named tool (introspection) or a call to a stubbed tool (a miss, fed back). Wired for Anthropic `/v1/messages` and OpenAI chat; savings are booked to `tool_catalog`.

**5C Complexity routing (#10).** `headroom/agent/complexity.py` decides effort and verbosity independently, treatment arm of `HEADROOM_OUTPUT_HOLDOUT` only. Effort: only a client-sent `output_config.effort` / `reasoning.effort` / `reasoning_effort` is ever lowered (floor `medium`); nothing is injected and `thinking` is untouched. New asks, errors, novel errors, failed retries, contradicted evidence, open acceptance gates and explicit deep-thinking requests are protected. Otherwise score 0–4 from unknown requirements, new symbols, multi-file scope and uncertainty; JevK5 scores only the middle (2). Verbosity: the shaper's conversation-stable level stays in the system tail. A complex new ask can relax it for one reply via a trailing turn block, only on models that accept mid-conversation system. Wired in the Anthropic, chat and Responses paths.

**5D Feedback and speculation (#12, #15).** `RetentionFeedback` keeps pending observations per lossy decision, keyed by (family, content type, task class, strategy, model family, language, workspace hash), never raw text. They resolve once: negative on CCR retrieval (`IndexedCCR.retrieve` and `CompressionStore.retrieve`), reread/re-search, failed edit repair or user correction; 0.25 on a test failure; positive after 3 quiet requests. The EWMA applies only after 5 observations, clamped to .1–.9; there is revert/reset, a neutral prior when off, and ledger persistence. `RequestSpeculation` gives one lane per request under a process cap (≤4). View previews run while the current unit is gated, and CCR span indexes build after admission while the request awaits the model. `take` never blocks, late results are discarded, `close` never waits, and store writes and selection stay authoritative. `/health.tool_admission` reports arbiter (decisions, rejections, confidence bins, cache mix, p50/p95 overhead, realized tokens/cost apart from estimated avoided turns), feedback and speculation.

**Tests.** New: `tests/agent/test_phase5_{arbiter,catalog,complexity,feedback_speculation}.py` (56), including e2e `/v1/messages` through the wrap rollout, arbitrary tool ordering, cache/policy decisions, tool-free and introspection cases, bounded and reversible feedback, and speculation on/off giving identical bytes. `tests/agent` 183 passed. Targeted CCR/gate/prefix/pipeline/wrap: 997 passed (1 failure that predates Phase 5, `test_canonical_pipeline::…compresses_tool_outputs`, also on clean HEAD). Proxy/handler/output/tool suites: 2485 passed. Their 7 failures reproduce on clean HEAD (loopback gating ×2, client-decision ×4, fsync). `test_proxy_anthropic_cache_stability` failed once under xdist and passes on rerun. Live JevK5 (llama-server, 4B Q8_0): arbiter blend applied (w .25, conf .81); catalog and effort advice were low-confidence and correctly ignored.

**Eval** (`benchmarks/results/phase5.json`): Python bare 29.39% and Rust 35.19%, both unchanged. Wrap path 55.95% (Phase 4: 56.91%), recall 1.0, 0 errors/structure/prefix violations. The only move is `tool_error`, 71% → 0%: the arbiter keeps the full traceback because the folded view shows only 19% of the traceback facts (they were recoverable, not visible).
