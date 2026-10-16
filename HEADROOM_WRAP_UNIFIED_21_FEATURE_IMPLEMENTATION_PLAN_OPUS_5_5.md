# Headroom Wrap: Unified Intelligence + Token-Efficiency Implementation Plan

**Target agent:** Claude Opus 5.5  
**Status:** Neither previous plan has begun. This document **replaces** both plans, not supplements them.  
**Scope:** Original improvements **1–15**, additions **16, 17, 19, 20, 23, 24**, and fully automated **JevK5 / llama.cpp** integration. Exactly **21 feature IDs**.  
**Upstream reference:** `headroomlabs-ai/headroom` **v0.41.0**, `main` SHA `76133cff59b71284399bf760c5578dc373c97231` (checked 2026-10-14 UTC).  
**JevK5 reference:** runtime **v0.3.3**; latest inspected `allebee/jevk5` main SHA `f26426d16f59e8bbe1470e5b162cc89329e29b29` (2026-09-28). Default model remains **JevK5 4B v0.3 Q8_0**; v0.3.3 adds the newer 9B line but does not supersede the tested 4B GGUF behavior.  
**Development branch:** the user's existing custom Headroom/Paritok checkout, **not** a fresh upstream clone.

> **Outcome:** One working, production-wired, cross-platform Headroom `wrap` intelligence stack. Reduce billed tokens and unnecessary tool turns without sacrificing answer correctness, exact retrieval, existing provider compatibility, or cache benefits. Do not claim higher quality where benchmarks do not demonstrate it.

## Nonnegotiable agent contract

1. **Perform the phases in order, finishing and testing each before beginning the next.** Do not build an interface now that requires its previous phase to be rewritten later. A later phase may register a consumer through an intentional extension interface created earlier; it must not replace that interface.
2. **Implement all 21 feature IDs.** No stubs, uncalled classes, test-only implementations, TODO substitutes, or “future work” for required functionality. Do not invent features outside the listed scope.
3. **Inspect actual local source before editing.** Local custom work is authoritative. Reuse existing Headroom/Paritok code. Never blindly overwrite `settings.json`, `settings.local.json`, authentication, provider routing, or the user's unrelated configuration.
4. **Synchronize with upstream safely, never destructively.** Never `reset --hard`, discard dirty work, force-push, or overwrite modified files. Resolve merge conflicts semantically. Keep rollback refs and backups. If a merge cannot be completed without risking user data, retain the intact checkout and document the blocker rather than damaging it.
5. **Reuse current upstream APIs.** No parallel proxy, memory engine, embedding system, telemetry platform, feature-toggle parser, code graph, or failure learner merely to make the plan easier to implement.
6. **Preserve request wire correctness, cache-hot prefix bytes, current/latest user message, code read protection, context-guard behavior, CCR exact retrieval, structured output, streaming and native tool calls.** A local optimization failure passes through the original operation; never substitute a fabricated answer or false success.
7. **JevK5 is an optional advisory classifier.** It cannot independently establish facts, execute commands, authorize file changes, overrule invariant checks, or demand a large model call. When it fails, deterministic behavior continues.
8. **Run on Windows and Linux without mandatory Docker or root/admin.** Don't assume Bash on Windows, systemd on Linux, or an NVIDIA device. Honor `HEADROOM_OFFLINE` and `HEADROOM_STATELESS`.
9. **All personal/project state stays local by default.** Secret-bearing payloads are not logged. Path-sensitive actions require a trusted wrap-registered workspace, not model-supplied paths or `X-Headroom-Project` attribution.
10. **Complete each phase as a functional slice.** At its gate, run relevant unit + integration tests, fix regressions, commit only the intended files, and add a short completion entry to `IMPLEMENTATION_REPORT.md`. No architectural re-planning or sprawling phase reports.

## Features and dependency placement

| ID | Implemented capability | Phase |
|---|---|---|
| 1 | Task-conditioned live-zone compression | 3 |
| 2 | Active Rust hybrid semantic relevance | 3 |
| 3 | Indexed/selective CCR, exact full retrieval retained | 3 |
| 4 | Pre-context tool-result admission/externalization | 4 |
| 5 | Cross-turn semantic/structural deltas | 4 |
| 6 | Multi-candidate Transform Arbiter | 5 |
| 7 | Deterministic semantic invariant and automatic backoff | 3 |
| 8 | Global marginal-information token allocator | 5 |
| 9 | Semantic progressive tool catalog | 5 |
| 10 | Complexity-aware effort/verbosity routing | 5 |
| 11 | Graph/symbol-scoped coding context | 2 |
| 12 | Retrieval-feedback/retention adaptation | 5 |
| 13 | Enforced lossy-risk and volatile-content policy | 3 |
| 14 | Rich Read/Grep/Glob/Bash/test/diff/JSON interceptors | 4 |
| 15 | Bounded speculative candidate preparation | 5 |
| 16 | Task State Compiler | 2 |
| 17 | Provenance-aware Evidence Ledger | 2 |
| 19 | Tool Contract and Argument Validator | 6 |
| 20 | Safe learned Workflow Macro Compiler | 7 |
| 23 | Change-Scope / Drift Firewall | 6 |
| 24 | Test Impact and Verification Planner | 7 |
| — | JevK5 setup, management, typed decision advisor | 1 |

**Execution sequence:** Phase 0 safety/sync → 1 runtime and JevK5 → 2 state/evidence/graph → 3 query, safety and CCR → 4 admission/delta/interceptors → 5 arbiter/adaptive tooling → 6 scope and validation → 7 testing and macros → 8 final cross-platform qualification.

## Existing upstream integration anchors: prefer these over new subsystems

- `headroom/cli/wrap.py` + provider-specific `headroom/providers/*` launch/registration paths. `wrap` is a launcher; compression belongs in proxy/Rust core or host hooks, not an unrelated wrapper process.
- `headroom/proxy/turn_hooks.py` (`TurnHook`, ordered `priority`, `stream_safe`, `TurnContext.record_savings`) and `headroom/proxy/extensions.py`. A first-party central installer must make features reachable in wrapped sessions without an extra manual third-party extension opt-in. **TurnHooks cannot block a tool already executing inside the client**: use a genuine pre-tool host hook for that capability, otherwise observe/warn only.
- `headroom/proxy/workspace_registry.py` and `project_context.get_registered_cwd()`; wrap-issued session-token/owner-only marker validates the workspace. Header-only project names are **attribution**, not filesystem authority. Preserve shared/reused/dedicated proxy identity and extension inheritance.
- `headroom/rollout.py` (`FEATURES`, immutable `RolloutSnapshot`, `HEADROOM_FEATURES` / `HEADROOM_DISABLE_FEATURES`); `headroom/proxy/runtime_env.py` only for settings actually read live. Do **not** create another `auto|on|off` parser for rollout features.
- `headroom/paths.py` including `persistence_allowed()` and stateless mode; `headroom/offline.py` egress guard; `headroom/log_safety.py` redacted warnings/exceptions; `headroom/proxy/savings_attribution.py` and current `/stats` rather than counting phantom savings. Honor `HEADROOM_WORKSPACE_DIR`.
- `headroom/memory/traffic_learner.py` and `headroom/learn/_shared.py` for existing error→recovery learning; `headroom/cache/prefix_tracker.py`, `headroom/proxy/context_guard.py`, `headroom/proxy/output_shaper.py`, `headroom/proxy/output_turn_policy.py`, `headroom/ccr/*` and the live-zone Rust modules.
- Current v0.41.0 adds/changes: Rust-native `/stats`/dashboard, improved CCR mixed tool calls, Claude/Haiku 5.5 message handling, context compaction protection, per-request ML compression caps, changed OpenCode 2.x integration, and substantial code-deprecation cleanup. **Never call removed Python log/search selector mirrors or deprecated memory budget/ranker/tool adapter classes.**

### Shared engineering rules

**Single owning services:** `AgentRuntime` (session/events), `DecisionAdvisor` (JevK5), `TaskContext` (request facts), `TaskStateCompiler`, `AgentEvidenceStore`, `CodeGraphAdapter`, `RelevanceService`, `SafetyGate`, `IndexedCCR`, `ToolResultEnvelope`, `TransformArbiter`, `ToolContractValidator`, `ChangeScopeFirewall`, `TestImpactPlanner`, `WorkflowMacroRegistry`. Extend matching existing classes; these are ownership names, **not mandatory duplicate files**.

**Provider-neutral core, provider-specific transport:** normalize provider data to internal typed events, never reserialize immutable unrelated API bytes. For Rust live-zone functions pass a compact query/decision-hint structure from Python where possible. Native Rust proxy is a **separate runtime** from the Python proxy: test its actual supported integration surface; do not claim Python TurnHooks run inside it. Do not silently reroute users to a different proxy implementation.

**Security and safety boundaries:** explicit latest user instructions supersede inferred state; tool text is untrusted evidence, not instructions; a model/JevK5 confidence score is never filesystem authorization. No auto-shell execution of strings built from untrusted tool results. Preserve exact recovered originals if promising recoverability, atomic writes, owner-only private files (POSIX 0600/0700; Windows user-scoped ACLs), no cross-user/project bleed, bounded caches and timeouts.

**Feature flags:** Register `agent_task_state`, `agent_evidence`, `agent_tool_contracts`, `agent_change_scope`, `agent_test_impact`, `agent_workflow_macros` plus the needed original Phase-1 feature names in **existing** `headroom.rollout.FEATURES`; use channel/snapshot semantics. Make per-feature enforcement behavior `observe|warn|protect` a separate typed policy, **not** competing rollout states. `HEADROOM_DISABLE_FEATURES` alone disables a feature. After acceptance tests, default-enable safe non-destructive features for `wrap` in this fork; protection remains opt-in or explicit host-authorized, with `warn` as the default for changes. Preserve upstream rollout-channel safety rules and defaults for unrelated functionality. Intentionally preserve upstream defaults for unrelated settings.

**Configuration defaults (single typed object; overridden via existing config mechanism):** Jev model `jevk5-4b-v0.3-Q8_0.gguf`, Q8, llama context **8192 effective tokens**, local host `127.0.0.1`, GPU layers `auto`, decision hard timeout 1200ms, call cap 4/request, cache TTL 30s, min confidence 0.60, advisor utility weight 0.25 capped at 0.40, open circuit after 3 failures for 60s; generic prompt-state injection max 900 tokens (request allocator may lower it); local decision state <=6000 input tokens; 5 top tool schemas where native deferral absent; graph BFS radius 2; CCR retrieval default top_k=5 max 1200 tokens; all tunable but bounds tested. Use distinct external endpoints: `HEADROOM_JEVK5_LLAMA_URL` means an operator-managed **llama-server** exposing `/health`, `/tokenize`, and `/completion`; `HEADROOM_JEVK5_SYSTEMONE_URL` means an operator-managed JevK5-compatible **`/v1/systemone`** service. Never infer one API from the other merely because a URL responds.

---

# Phase 0 — Protect checkout, update to upstream v0.41.0+, inventory and baseline

**Deliverable:** a safely merged working branch and a truthful map of existing features. Finish this before creating new runtime code.

1. In the **local custom checkout**, capture branch, HEAD, status (including untracked files), remotes, Python/Rust/compiler versions, current Headroom and `llama-server`, and working native Python/Rust test commands. Store in `IMPLEMENTATION_REPORT.md`. Verify user changes to Paritok, model routing, settings, authentication, CLI and proxy behavior.
2. Make a unique local safety branch/ref at current HEAD and secure copies of **both staged and unstaged binary diffs and untracked work** outside repo. If tree is dirty, stash only after preservation, retaining the stash until validation. Never change or delete original config files to make tests pass.
3. Find/add an upstream remote pointing **only** to `headroomlabs-ai/headroom`, fetch `main` and tags, and record fetched commit. Merge `upstream/main` (whatever remote name is appropriate) into a new feature branch. Reconcile rather than overwrite modified custom files. If conflicts affect protected user files, resolve with targeted semantic edits and test.
4. Inventory real implementations: Rust live-zone dispatch/query handling, embedding initialization, CCR indexing, code graph, tool search, AST tools, Python `TurnHook` hooks, native Rust proxy, Jev/Paritok custom code, rollout flags, memory TrafficLearner, provider handlers, test/metrics utilities. Mark each feature `complete|partial|absent` with actual path/entry point. **Never assume the previous 15 features already exist.**
5. Run and record original smoke tests for wrap Claude/Codex/OpenCode, core live-zone, CCR, prompt-cache byte fidelity, tool deferral, memory, output shaping, Rust proxy, Windows or Linux platform. Existing failures can be isolated with evidence; do not label them feature regressions.
6. Establish baseline evaluation fixtures: file reads, large search, logs/pytest, JSON, diff, same-output reread, tool errors, conflicting evidence, cache-hot turn, native streamed turn. Save raw/forwarded token counts, provider cache mix where known, tool turns, wall time, correctness, and known failures. Preserve exact baseline before modifications.

**Gate 0:** safely merged or explicitly documented non-destructive merge blocker; full feature inventory; baseline measurements; no user data/config lost; commit only intended integration fixes. Proceed only on a usable local checkout.

# Phase 1 — Runtime foundation and zero-manual JevK5 setup

**Deliverable:** session-safe shared runtime, events and storage, and `headroom intelligence setup/status/doctor/stop` with a functioning local classifier and deterministic fallback. No other phase should need to invent these primitives.

### 1A. Shared runtime, provenance, and hook lifecycle

- Centralize first-party installation in existing proxy initialization/hook registration; register once, safely order request message-fold hooks at priority **80** (prior to tool deferral 200+). Only `stream_safe=True` for strictly synchronous request folds that never re-drive. Avoid unbounded future request mutation or duplicate per-provider handlers.
- Define `AgentEvent(v,id,workspace_id,session_id,task_id,kind,origin,source_id,resource_id,resource_hash,revision,ts,payload_digest,redacted_metadata)` with stable idempotency key from `(session, source call/id, event kind, index, content digest)`. Normalize user/tool/model/file/git/test events. Process each event **at most once** even on streaming replay/retry. `origin=USER` only for authentic, explicitly authored user turns, not harness-injected user-role messages.
- Shared workspace identity = authenticated wrap session token plus server-registered CWD; derive stable **hashed** `workspace_id` from the canonical validated root and principal, per-session `session_id` from registered token digest; support linked Git worktrees as distinct mutation scopes. If missing or stale, allow in-memory observation, **never path-sensitive protection/execution**. Other sessions/users are isolated.
- Shared `AgentRuntime` keeps append-only bounded `agent_events` plus SQLite materialized tables under `headroom.paths` (new `agent_state_db_path` helper if no equivalent). SQLite WAL (fallback if unavailable), schema version, transactional event+projection update, replay-to-rebuild, revision compare-and-swap, read-only corruption recovery. Limit large payloads to digests/structured redacted summaries; store exact originals through existing CCR, **not** duplicate huge tool blobs.
- Stateless mode: in-memory bounded state and **zero new filesystem writes**; use `persistence_allowed`. Offline: zero new network connections/downloads. Owner-only directories/DB, symlink-escape protection, atomic config writes, bounded locks and queues; no blocking SQLite or Jev work on request event loop. Terminate owned background tasks cleanly.
- Reserve stable event states now: `USER_INTENT`, `TOOL_CALL`, `TOOL_RESULT`, `FILE_CHANGE`, `GIT_CHANGE`, `TEST_RESULT`, `TASK_STATE_CHANGE`, `EVIDENCE_CHANGE`, `SCOPE_DECISION`, `CONTRACT_DECISION`, `VERIFICATION_DECISION`, `WORKFLOW_OBSERVED/EXECUTED`; unused future producers register through versioned event types, not schema rewrites.
- Define small versioned interfaces `TaskContext`, `DecisionHints`, `ToolResultEnvelope` (with `is_partial`, `partial_reason`, source truncation metadata, `exit_code`) and `ActionAuthorization(workspace_id,session_id,task_revision,scope_revision,resource_revision,expires_at,decision)`. Later phases **consume** these, not redefine them.
- Proxy health reflects effective rollout, trusted workspace mapping, runtime hook installed, and service status. Shared/reused/dedicated proxy must report mismatched config rather than silently ignoring features. Scope user-specific state by request, never process-wide globals.

### 1B. JevK5 setup and service lifecycle

- Use **JevK5 runtime v0.3.3** (`allebee/jevk5`, release tag `v0.3.3`) with `alibiserikbay/JevK5-GGUF` file **`jevk5-4b-v0.3-Q8_0.gguf`** (~4.48 GB); calibration `temperature=1.22`, `knockout_temperature=0.93`. Keep this 4B model as the tested default: v0.3.3 adds improved 9B weights but does not establish the 9B as a better Headroom decision model. JevK5 provides typed `noul`, `choice`, and `score`, up to 16 options per pass plus recursive multi-pass handling beyond 16; it is English-only and is not a prose LLM or embedding model. Its model ceiling is 16,384 tokens, but **Headroom's configured llama context of 8192 is the effective ceiling for this integration** and every decision must fit the actual running server context. **Do not substitute JevK5-Lite or 9B weights as an automatic default.**
- Require **Python >=3.10** before setup. `headroom intelligence setup` must discover a user-installed `llama-server` (`HEADROOM_JEVK5_LLAMA_SERVER` executable override; previous managed manifest; PATH; `LLAMA_CPP_HOME` common Linux/Windows build dirs); probe `--help`, `--list-devices`, Qwen3.5 model loading and required endpoints; reuse a compatible existing binary. If unavailable, build a **separate managed** llama.cpp `llama-server` from official source with CMake (`GGML_CUDA=ON` when available; CPU otherwise); never modify the user's installation. Install only the lightweight GGUF client path via `sys.executable -m pip install --no-deps 'jevk5 @ git+https://github.com/allebee/jevk5@v0.3.3'` when missing/incompatible; do not pull JevK5's torch runtime merely to use GGUF. Respect the active venv; no root/admin.
- Download/cache the GGUF once via supported `--hf-repo ... --hf-file ...` or a resumable Hugging Face fallback. Use `.part` + atomic rename. Setup is complete only after validating GGUF metadata **and** the publisher's SHA-256 for the default file: `aea433883bc7ed399f2fbd539e53d2eac7caf71a946fe6650995a413979d4a30` for `jevk5-4b-v0.3-Q8_0.gguf`. Reuse existing files only after verification. A custom model override must resolve its own trusted checksum or be explicitly marked operator-managed; never apply the default model hash to another file. **In offline mode, use verified cache only; never attempt git/pip/HF download.** If no viable model is available, mark the advisor unavailable and fail open.
- Managed `llama-server` runs on `127.0.0.1` free port (prefer 8091, dynamically allocate/retry), `--ctx-size 8192 --n-gpu-layers auto` when supported. `HEADROOM_JEVK5_LLAMA_URL` is an operator-managed llama-server and is never started/stopped by Headroom; capability-probe `/health`, `/tokenize`, and `/completion` before use. `HEADROOM_JEVK5_SYSTEMONE_URL` is a separate optional operator-managed TypeSafe-style JevK5 endpoint and must be probed as `/v1/systemone`; **never pass it to `JevK5GGUF`**. Save PID, executable, ownership, timestamp, port, and model hash only for Headroom-owned processes; validate PID+command before stopping. On Windows suppress unwanted console windows/use owned process handling; on Linux use a separate owned process group. Never kill another app occupying the port.
- Readiness requires `/health`, `/tokenize` with `parse_special=true`, and `/completion` with `n_predict:1,n_probs:40,temperature:0,cache_prompt:false`; parse `completion_probabilities[0].top_logprobs`. Then verify `JevK5GGUF(temperature=1.22, knockout_temperature=0.93, timeout_s=<bounded>)` returns finite normalized `noul/choice/score` values. Required answer letters missing from `top_logprobs` are **not silently accepted as reliable advice**: detect the adapter's missing-letter condition, retry once with a larger bounded `n_probs` (default retry 128), then mark the advisory unavailable if required letters are still absent. Test a no-receipt refund (no) and misdelivered parcel (misdelivered) as top-choice sanity, not exact confidence. `doctor` reports executable, JevK5 runtime version, model/hash, endpoint type, effective context, calibration, missing-letter health and process ownership.
- One **internal Python `DecisionAdvisor`** owns all JevK5 use. For local/external llama-server it calls `JevK5GGUF`; for an explicitly configured `HEADROOM_JEVK5_SYSTEMONE_URL` it uses a separate typed client for that API. `choose/noul/score` use cached, bounded typed prompts and decision-family tags. **Do not create another `/v1/systemone` gateway by default**: Python proxy consumes the adapter directly, Rust receives compact `DecisionHints` at a stable FFI boundary. Only if an actual native consumer needs HTTP should a Headroom-owned authenticated loopback gateway be added and integration-tested.
- **Do not share one mutable `JevK5GGUF` instance across concurrent decisions without synchronization.** The current adapter updates `last_pass_tokens`, `last_seconds`, and `missing`; use a short-lived adapter per active decision or serialize access with a bounded lock while the `llama-server` process remains shared. Add a concurrency regression test proving results and adapter metrics cannot bleed between requests.
- `HEADROOM_JEVK5=auto|on|off` may exist as **typed service policy**, not as a rollout flag. `auto` opportunistically uses the service but never blocks ordinary `wrap` startup on model download; `on` gives an explicit setup diagnostic on failure while **request traffic still fails open**; `off` makes no starts/calls. `HEADROOM_JEVK5_AUTOSTART=1` default; service/GPU are shared and bounded, not restarted for each request. In `auto`, first-run background setup is allowed only when online and without delaying user traffic.
- Per decision: **1200ms hard wall-clock deadline across the entire decision**, including `/tokenize`, every `/completion` pass, a missing-letter retry, and >16-option knockout passes; 4 calls/request max, 30-second cache, circuit breaker after 3 consecutive failures for 60 seconds. **Do not rely on upstream `JevK5GGUF(timeout_s=600)`**: instantiate it with a bounded timeout and enforce a monotonic overall deadline so each HTTP operation receives no more than the remaining budget. A timeout or late result is discarded and deterministic behavior continues. Top-choice confidence <0.60 means ignore advisory; advisor weight <=0.40 and default 0.25. No synchronous JevK5 per line/tool item, no raw state logs, no model hallucination into the evidence store. CPU fallback is allowed only if it meets the hot-path deadline; otherwise disable the adviser. Setup/doctor may use a separate, longer bounded load timeout.

**Gate 1:** CLI works on Windows/Linux; service starts/reuses/stops correctly, passes conformance checks, unavailable JevK5 cannot block `wrap`; events are isolated, replayable, idempotent and stateless-safe; proxy+native route unaffected without feature flags. Pin the JevK5 runtime to **v0.3.3** for reproducibility; inspect moving `main` only for compatibility evidence, never install production behavior from an unpinned branch.

# Phase 2 — State, evidence, code graph and task query foundation (11, 16, 17)

**Deliverable:** functional task/evidence intelligence before any lossy changes, plus a graph-backed relevance query used by later phases. No generic “conversation summary” substitute.

### 2A. Task State Compiler (#16)

- `TaskState(id,workspace_id,session_id,revision,status,goal,subgoals[],hard_constraints[],acceptance[],decisions[],completed[],blockers[],unknowns[],assumptions[])`. Each `StateAtom(id,kind,text,origin,confidence,state,source_event_ids[],evidence_ids[],supersedes_id,updated_at)`; atom states `PROPOSED|ACTIVE|SATISFIED|VIOLATED|SUPERSEDED|REJECTED`; task statuses `ACTIVE|BLOCKED|COMPLETE|ABANDONED`.
- Authority: explicit user instruction > tool-confirmed observation for empirical claims > agent declaration > inference. An agent saying “done” is a **claim**, not proof; completion requires user acceptance criteria or test/observable evidence. User corrections supersede only the targeted atom; preserve historical provenance. Never treat a truncated result as proof of completeness. JevK5 may classify which existing atom a result updates; it cannot invent goals or mark untested requirements satisfied.
- Deterministic extraction from true user turns, structured tool/test events, touched resources, accepted decisions, explicit errors and corrections; write atom transitions as events with CAS revision. Resolve duplicate/superseded state. Ensure two concurrent turns cannot overwrite newer state; stale revisions must retry projection once or reject and record, never silently commit. State works with missing/trimmed historical turns.
- Inject **one compact stable block** via safe request-fold hook (priority 80) near the supported live message/context area; never rewrite a cached system prefix or last user message, never inject a tool into tool-free requests, preserve Claude Haiku 5.5 mid-system messages. Use original Phase-1 allocator once available, until then a 900-token hard ceiling and no more than 2% of effective available input window; absolute user requirements/active errors first. If injection increases cache-adjusted cost or contradicts current request, omit it. Stable contents/hash to avoid per-turn churn.

### 2B. Evidence Ledger (#17)

- `EvidenceFact(id,claim_key,subject,predicate,value_json,polarity,source_kind,source_event_id,source_hash,workspace_id,session_id,task_id,confidence,status,valid_from,valid_until,supersedes_id,metadata)`. Status `ACTIVE|SUPERSEDED|CONTRADICTED|STALE|RETRACTED`. Raw tool text is untrusted **evidence** only. This is not CCR raw storage, semantic memory, or the savings ledger.
- Extract concrete facts from exit codes, test statuses/counts, error signatures, files/hash/existence, git state/changed paths, configured tool availability and authoritative user statements. Treat negative observations, permission errors, and missing-file results as first-class facts. Do not persist credentials/token fragments or entire files. Normalize `claim_key` to avoid duplicates; include source hash/digest and time.
- Freshness: validate volatile environment/port/process facts against current observation; a changed file hash supersedes assertions tied to old content; contradictory observations coexist with conflict label until authoritative resolution. No averaging incompatible booleans. Confidence reflects **source type and corroboration**, not whether JevK5 sounds certain. Evidence supporting a scope expansion/test requirement must be traceable to exact event/source ID.
- SQLite indexed by `(workspace,claim_key,status)`, `(task,status)`, `(source_event_id)`, timestamps. Maintain materialized heads from canonical events. Return compact facts only when germane to active TaskState. Never expose other projects or users. Test contradiction, supersession, invalidation, and provenance replay.

### 2C. Graph/symbol-scoped code context (#11)

- **Reuse** existing Headroom code graph/Serena/tree-sitter/code-aware infrastructure; do not add another downloader/parser. Nodes `file/module/class/function/symbol`, edges `imports/includes/calls/references/inherits/contains/defines`; per-worktree incremental hashes; respect ignored/generated/vendor paths. Unsupported languages fall back to exact path and lexical search.
- API `neighbors(workspace, symbol|path, radius=2)`, `symbols_for_file`, `affected_resources`, `invalidate_file`; scope to trusted workspace; prefer exact active symbol, same file, then adjacent callers/imports, then breadth 2. Graph ranking is one feature of retrieval, **never a hard justification to omit user-named evidence or required diff context**. Update changed-file indexes automatically; no unsafe repo-wide scan on each turn.

### 2D. TaskContext/query model shared by every downstream consumer

- Build per-request `TaskContext` containing explicit latest user goal, tool intent/name/call ID, exact paths/symbols/errors/identifiers, workspace, content type, result completeness/truncation, resource identity/version hash, cache policy, graph neighbors and current TaskState revision. Extract genuine latest user intent using current provider walkers (not harness-added user-role boilerplate). Prioritize exact user-mentioned IDs/paths before prose; cap relevance query 512 tokens.
- For code tasks create `relevance_query = explicit_user_goal + tool_intent + exact_entities + active_error + top_graph_neighbors`; for other tasks omit graph. No old conversation dump, no generated “thoughts” as facts. Pass typed compact query through future Python/Rust boundary.

**Gate 2:** multi-turn task tracks requirements and corrections accurately; evidence provenance/contradiction tests pass; graph updates incrementally; trusted multiple sessions isolated; 900-token injection/cache boundary tested; actual normal `wrap` request uses the state fold.

# Phase 3 — Query-conditioned semantics, safety policy and selective CCR (1, 2, 3, 7, 13)

**Deliverable:** a complete *safe compression/retrieval substrate*. No later transform may bypass its admission gates.

### 3A. Deterministic semantic invariants and policy (#7, #13)

- `SafetyGate.check(original,candidate,task,policy,provenance)` returns `ALLOW|REJECT` plus reason codes and invariant recall. Preserve hard: user-named IDs/paths/symbols/URLs/hashes, failure/exception lines, exit/status codes, test pass/fail/skip counts, `file:line`, meaningful numeric values/units/dates, JSON booleans/negation/key invariants, diff headers/hunks, active signatures, CCR identifiers, truncation/partial boundaries, and applicable syntax validity. Never claim completeness for already truncated input.
- Reject candidate if required facts are changed, irrecoverably omitted, partial→complete misrepresented, retrieval marker invalid, structured syntax broken, tokenizer says no savings, or cache-effective cost gets worse. Candidate may omit a protected exact item **only** if a reliable exact retrieval path and explicit recoverability contract cover that item; otherwise original wins.
- Enforce `CompressionPolicy.max_lossy_ratio` and `volatile_token_threshold` in the real hot path, not just metrics; distinguish `wire_drop_ratio` from **irreversible loss**. Preserve existing per-auth policy defaults (subscription lower tolerance than PAYG) and upstream revised values if different on live checkout. Exact CCR externalization can be highly compressed on wire because original is recoverable; lossy deletion may not exceed the actual policy budget. Fail to original on unknown limits.
- Backoff candidates: aggressive→conservative→structural/lossless→CCR exact externalized→original. **One central implementation** in Rust/Python with parity fixtures; JevK5 never vetoes the safety gate.

### 3B. Activate existing Rust hybrid semantics (#2)

- Existing `crates/headroom-core/src/relevance/embedding.rs` has actual `fastembed-rs` BGE-small `try_new()` behind `ml`; `HybridScorer::default()` may be unloaded. Add lazy process-owned initialized scorer, bounded LRU item embedding cache, batched chunk embeddings, cached one query embedding/request, and **BM25/exact identifier boost**. No second embedding engine or mandatory second `llama-server`.
- Respect offline/no `ml`, ONNX unavailable, AVX mismatch, unavailable cached model and inference failures: deterministic BM25 fallback; zero repeated attempts to download on each request. Use bounded ONNX workers and avoid idle-core spin; do not block full agent operation on model init. Prove paraphrase relevance and exact identifier protection via fixtures.

### 3C. Query-conditioned live-zone (#1)

- Inspect the **current** Rust live-zone dispatcher and provider walkers; eliminate production **empty-query fallbacks where query exists**. Pass `TaskContext` query into SmartCrusher, search/log/diff, SourceCode/PlainText arms where supported. Retain source byte-range surgery: frozen prefix and untouched blocks must remain **byte identical**. Use strict task/user explicit-identifier retention bias; don't compress the newest explicit user ask or active source read beyond present read protection.
- Test identical tool content under two different user goals produces appropriately different retained rows while preserving every hard invariant; provider shapes Anthropic, OpenAI Chat, Responses, streaming or native Rust as supported by actual route.

### 3D. Indexed CCR (#3)

- Extend **existing CCR `CompressionEntry`/store and headroom_retrieve behavior** (including mixed client/Headroom tool calls); do not replace hashes/retrieval markers. Preserve `headroom_retrieve(hash)` exact original byte-for-byte. Add optional `query,top_k,cursor,mode` with modes `full|search|range|metadata` and item-span index: `(entry_hash,span_id,ordinal,byte_start,byte_end,type,content_hash,key/path/symbol metadata)`; semantic chunking by JSON row, log/trace, search hit, code symbol, diff hunk, paragraph. Store exact originals first; never index rehydrated markers as originals.
- SQLite FTS5/BM25 where available; deterministic lexical fallback without FTS5; embedding and graph proximity are ranking hints. Search result includes exact source hash, span ID, score, ordinal, has_more/cursor and completeness flag. Cursor must bind workspace, entry hash and index generation, preventing cross-entry leakage or stale pagination.
- Proactive CCR expands **selected** spans under 1200 tokens by default; full only when explicitly requested, localization impossible, or safety logic requires it. No cross-conversation expansion: only entries reachable in requesting session. Apply TTL/deletion to spans with parent entry, verify exact recovery before emitting marker. Test 100k-token stored data → 5 relevant spans; `full` remains exact.

**Gate 3:** tokenizer/cache/invariant parity tests pass; query-conditioned output varies appropriately by task; semantic model loads once or falls back offline; exact indexed CCR search and full retrieval both work in actual proxy/mixed streaming flows; no cached prefix mutation.

# Phase 4 — Host admission, structured delta and interceptors (4, 5, 14)

**Deliverable:** pre-context control where host supports it and safe live-zone fallback everywhere else. Every interceptor must be reachable from the real tool path.

1. Normalize all observed tool results to `ToolResultEnvelope(tool_family,call_id,session,workspace,resource_id,version_hash,content,stderr,exit_code,is_partial,truncation_reason,source_metadata)`. `resource_id` is stable across calls: canonical workspace-relative read path+range, normalized command + cwd, grep query+scope, API signature, or JSON resource key. Tool-call ID is **not** resource identity.
2. Add one host-side **pre-tool/post-tool adapter** for actual wrapped clients supporting hooks (Claude Code and equivalent supported host). Verify hook invocation with a live fixture. For other clients the proxy compresses tool results only when they appear in live-zone, without falsely claiming they never entered client history. No injection/execution of new tools into tool-free model requests.
3. Storage sequence: capture tool completeness, determine resource hash, index exact original in CCR, retrieval self-test, build candidate structural view, run SafetyGate, then replace result only if authorized. If any step fails, forward original. Never assume already truncated input is complete or infer total items.
4. Delta (#5): persist prior complete resource hash/span identities and content version; compute added/removed/changed content by line/diff, keyed JSON rows, search-row sets, listings or test outcomes; include base/current hashes and precise omitted/changed counts. A partial base never qualifies as a complete comparison. Re-read unchanged resource can emit small reference **only after verifying exact recovery for the new version**. Semantic redundancy classification may remove prose repetition only behind same invariants and recovery.
5. Tool-specific interceptors (#14), all through one dispatcher, not separate provider rewrites:

| Tool | Required view and protected data |
|---|---|
| Read | Small/current edit target exact; large complete read AST/symbol outline plus retrieval; partial read marked partial; active working set preserved. |
| Grep/search | Group by file/symbol, exact queried matches + failures first; task/graph rank; full result indexed. |
| Glob/LS | Group by directory/package, show additions/removals on rerun; no nonexistent “all files” claim. |
| Bash/build/log | Preserve exit code, stderr, errors, stack traces, first/last meaningful boundaries and changed failures; fold repetitive progress only. |
| pytest/other test | Preserve failing IDs, assertions, counts, skip/xpass/xfail as applicable, summary, traceback, exit status. |
| git diff/status | Preserve names, status semantics, unmodified patch if an applyable diff is needed; selectively return hunks otherwise. |
| JSON/API/DB | SmartCrusher keyed rows with exact user fields/errors/numeric identifiers preserved and full indexed origin. |

6. For host clients without pre-context rewrite support, report capability `observational_or_proxy_only`, not protection. Never break host tool cancellation, permission prompts, hook error propagation, or expected output MIME/tool-result formats. Use existing AST interceptors and first-plan completed components if already present.

**Gate 4:** end-to-end host hook demonstrates bytes eliminated *before host context* where supported; proxy fallback works without host hook; exact CCR recovery, delta correctness, truncation provenance and seven tool families tested on Windows/Linux; failures pass through originals.

# Phase 5 — Budget/arbiter, adaptive tools, output, feedback and speculation (6, 8, 9, 10, 12, 15)

**Deliverable:** final working compression-selection and adaptive context stack. **Implement the following subsections in their order, finishing each within this phase**; shared interfaces were already finalized in Phases 1–4.

### 5A. Global budget then multi-candidate arbiter (#8, #6)

- Compute available live-zone tokens using provider model **effective context limit** (including `context-1m` beta and context-guard learned cap), frozen prefix, mandatory tools/system, reserved generation, a safety margin; never use “1M” simply because a client claims it. Respect `HEADROOM_CONTEXT_GUARD` client-compaction behavior and `x-headroom-keep-last-turns` without losing TaskState.
- Allocate **mandatory** tokens first (user goal, active errors, user-named paths/symbols, tool statuses, truncation metadata), then optional chunks by marginal relevance+novelty+anomaly+recoverability per token; allow a modest cross-source diversity share. If budget insufficient for hard invariants, prefer exact CCR externalization or original with context-guard compaction rather than lie.
- Generate applicable `Candidate(original,lossless,type_specific,query_aware,structural,delta,indexed_externalization)`; record token count, invariant scores, policy, recoverability, cache write/read mix, retrieval overhead and preparation latency. Run hard filters **before** candidate scoring. Original always included.
- For safe survivors use deterministic utility with weights **relevance .28, soft recall .20, recovery .16, net token savings .14, cache economics .10, learned prior .07, latency .05** (sum 1). Stable tie-break by less irreversible loss, then lower net cost, then original. JevK5 compares at most 16 finalists only when task is ambiguous or utility scores within 0.15; blend `(1-w)*utility + w*probability`, default `w=.25`, max `.40`. Jev unavailable ⇒ `w=0`. A compressed winner must save actual billed-token cost after cache effects, or original wins. Do not claim theoretical raw-token reduction is realized provider savings.
- Preserve existing provider byte-range rewrites and response format. Limit candidate expansion/ML work per **request**, not just per block.

### 5B. Progressive tool catalog (#9)

- Preserve native Claude Code `ENABLE_TOOL_SEARCH` and existing deferred schemas. Where native search is unavailable, expose a compact discoverable catalog with name/purpose/required parameter signatures; materialize default top 5 full schemas through existing tool-schema compaction. Task relevance, lexical+embedding, graph, recent successful use can rank the shortlist; JevK5 may break ambiguous top-16 ties.
- Materialized schema set cannot churn a cache-hot prefix merely to shave nominal schema tokens. Provider-specific deferral formats and `tool_choice` semantics remain valid. Dynamic local tools added in later phases (validator/macro) use the **same registry interface** and must not require changing this catalog architecture. Test tool-free requests and schema introspection.

### 5C. Complexity-aware output/reasoning (#10)

- Independently resolve **reasoning effort** and **visible answer verbosity**. Use turn structure plus novel errors, failed retries, conflicting EvidenceFacts, unknown task requirements, new symbols, multi-file scope, uncertainty. Never lower effort for a new user ask, errors, contradictions, repeated failures or unresolved completion gates; JevK5 scores 0–4 when uncertain. Do not add unsupported provider thinking parameters or override explicit high-effort requests.
- Reuse current output shaper/verbosity learning, cache-stable steering placement and known model support. High effort + short answer is valid. Use existing `HEADROOM_OUTPUT_HOLDOUT` sampling instead of designing a parallel experiment.

### 5D. Retention feedback and speculative prep (#12, #15)

- Feedback signals: CCR span re-retrieval, immediate reread/re-search, repair after omitted evidence, false candidate after test, correction. Bind only to generalized `(tool family,content type,task class,strategy,model,language)` and optional workspace hash; no raw private text in global priors. EWMA update `prior ← clamp(.95*prior + .05*outcome,.1,.9)` after **at least 5 independent observations**. Can tweak relevance/arbiter/catalog preference within hard policy, never override invariants or user explicitly required detail.
- Speculatively prepare hashes, bounded chunks, embeddings, invariants, candidate previews and CCR indexes **while** buffered tool content arrives or request awaits model; use 1 bounded worker per request, shared process cap, cancellation/timeouts. Final guard, task context, selection and writes wait for complete authoritative input. Late speculative work is discarded; request completion does not wait for stale futures; deterministic winner independent of scheduling.
- Expose realized token/cost accounting separately from **estimated avoided turns/time**. Record decision family, confidence bins, cache mix, selected strategy, re-retrieval rate, candidate rejection, p50/p95 local overhead; avoid payload logs.

**Gate 5:** true end-to-end candidate selection via normal `wrap`, correct policy/cache decisions and arbitrary tool ordering; no new output quality regression; tool discovery works for supported providers; feedback bounded and reversible; speculation never delays baseline path or leaks work.

# Phase 6 — Change scope first, then tool-call contracts (23, 19)

**Deliverable:** state/evidence-driven authorization and validated calls, close to actual client execution. Change-scope must be working **before** Tool Contract references it.

### 6A. Change-Scope / Drift Firewall (#23)

- Build `ChangeContract(workspace_id,task_id,revision,goal,expected_paths[],dependency_paths[],generated_paths[],forbidden_paths[],permitted_effects[],mode,created_at)` from **explicit user instructions, TaskState, validated EvidenceFacts and recent authorized writes**. `LEARNING` (new goal, observe), `ACTIVE` (expected scope established), `FROZEN` (user/accepted plan explicitly locks). No inference from unrelated reads alone. Default to **warn** on unexplained paths, **protect** only on explicit forbidden locations/operations and when host truly supports pre-tool blocking.
- Classify edits `IN_SCOPE|DEPENDENCY_SCOPE|GENERATED_EFFECT|UNRELATED|FORBIDDEN`. Dependency expansion requires a **traceable** reason: direct import/build link, failing compiler/test evidence, necessary interface, required platform/config, or related tests. Never use a bare JevK5 vote as authorization. Generated artifacts only if legitimately part of task; never default-delete untracked files.
- Before/after edit: compare planned resource and filesystem/git diff against current ChangeContract; `FORBIDDEN` blocks when enforceable; unrelated changes warn/observe and produce an event; unknown workspace never grants write authority. Hard-protect credentials, private keys, `.git` internals, unrelated provider config, and user-excluded paths. Restrict to registered worktree; explicit user approval may expand boundaries through TaskState revision.
- Every potentially mutating action checks `ActionAuthorization` **immediately before** execution; validate current task/scope/resource revision and path canonicalization/symlink resolution. Mismatch ⇒ re-evaluate once, otherwise deny protected action with stable reason code. Preview/dry-run remains available. Preserve Git worktree boundaries, Windows UNC/drive rules and case-sensitivity.

### 6B. Tool Contract + Argument Validator (#19)

- Canonical `ToolContract(tool_family,source,operation_class,side_effect,schema_version,hard_rules[],advisory_rules[],learned_recovery[])`, with `Decision(ALLOW|WARN|BLOCK|REPAIR_SUGGEST,reason_code,validated_arguments,task_revision,scope_revision)`. Map Claude/Codex/OpenCode tool names to supported families **without rewriting provider schemas**. Required families: Read, Edit/Write, Bash/exec, Grep/Glob, Git, test/build, HTTP/API; unknown tools are observed, never falsely certified.
- Validate: JSON tool-schema syntax/types and required args, mutually exclusive args, cwd existence, input file existence where required, allowed workspace and symlink/UNC paths, command availability including shell built-ins, platform-specific switches, Git ref syntax and non-destructive state, no opportunistic privilege escalation, malformed test invocation. **Do not prohibit legitimate user-requested external paths by assumption**: explicit user intent and host permission remain authoritative; ensure scoped paths are actually trusted before enforceable protections.
- Reuse current `TrafficLearner`/`classify_error` for learned failure→recovery patterns. Promote an advisory repair only after **3 corroborating successful recoveries**, with environment fingerprint, TTL and contradicting-evidence invalidation. Never transform a command with a guessed argument into auto-execution. JevK5 can rank already-safe suggested repairs but cannot override hard invalid syntax or scope.
- Interception matrix: native host pre-tool hook validates/blocks/asks user per host API; `TurnHook`/proxy receives tool-call metadata for **advisory-only** where the host owns execution. No fabricated “blocked tool” status in proxy-only paths. Fail-open for uncertain/tool-free unknown operations; fail-closed **only** for explicitly enforceable hard-protected actions at a trusted host boundary. Do not alter permission prompts.
- Surface reasoned compact diagnostics in normal tool context, not large injected boilerplate; record prevention vs warning vs observed failure separately.

**Gate 6:** explicit forbidden-path write blocked in a genuine host-pretool fixture; legitimate edits and dependency expansion allowed; repaired typo proposed but not automatically executed; concurrent revisions cannot authorize stale actions; proxy-only clients accurately show observe mode; existing permissions unaffected.

# Phase 7 — Test-impact verification, then safe learned macros (24, 20)

**Deliverable:** completed verification planner before macro compiler consumes it; learned macros safely collapse host/model round trips.

### 7A. Test Impact and Verification Planner (#24)

- `VerificationPlan(workspace_id,task_id,task_revision,scope_revision,changeset_hash,plan_revision,frameworks[],tier1[],tier2[],tier3[],required_checks[],risk_score,reason_codes[])`; test selection must be tied to current Git diff, task acceptance, EvidenceFacts, affected graph nodes and prior tests. Invalidate on changed files/task/scope/runner config; never run a stale plan as current.
- Framework adapters: pytest (`--collect-only`, targeted node IDs, existing coverage when available), Rust Cargo (`cargo metadata`, crate/target tests), C/C++ CTest (`ctest --show-only=json-v1` when supported), JS Jest/Vitest (respect detected npm/pnpm/yarn/bun). Preserve project-selected runner; never invent a build directory or demand a fresh configure as casual discovery. Unsupported frameworks fall back to canonical project test command or an explicit unknown result.
- Impact edges by coverage where available, import/build dependency graph, test naming/selection, historical failure association. Sort deterministically with exact failure regressions prioritized, then direct scope, dependency neighbors and project smoke. Tier 1 = directly related tests, Tier 2 = dependency/interface neighbors, Tier 3 = full suite; for high-risk public API, security, build config, provider wire, platform changes, and final completion, **full suite is mandatory** even if Tier 1 passes.
- Risk score from interface/config/security, historical failures, platform breadth, evidence uncertainty, change size and scope drift; choose staged escalation. Failing Tier 1 triggers investigation/targeted rerun; flaky tests must be tracked separately and cannot be hidden as success. Record executed tests, exit codes, identities, platform, version and evidence links in Evidence Ledger. Do not claim green without real run.
- Running tests is actual tool execution; only a genuine host permission boundary can automatically start them under host policy. Report test-selection rationale and avoided-work **estimate**, not guaranteed bill savings. A test result changes revision/evidence before next action.

### 7B. Workflow Macro Compiler (#20)

- Observe repeated **successful** sequences of **2–8 supported tools** without intervening user goal change, failures, corrections or scope violations. Normalize commands to typed families plus parameter slots; no generalizing credentials/URLs/untrusted script strings. A candidate needs **>=3 occurrences**, >=90% success, at least 2 sessions or 3 clean repeats in one, no contract/scope violation, and estimated reduction >=2 model-mediated tool turns before promotion.
- Auto-promote **read-only or verification-only** macros; mutating, deletion, network, package install, deployment, credential, elevation operations **never auto-promote or auto-execute**. Use deterministic macro IDs based on command family/template and workspace; cap to 32 active macros/workspace, max 8 steps, max 120 seconds/macro (per-step smaller deadlines, total strict). Macros are workspace-owned/versioned and invalidated when tool versions, permissions, TaskState, scope or relevant environment change.
- `headroom_workflow(macro_id,typed_inputs,dry_run)` is a local tool via the existing tool registry; advertise only eligible macros, stable schema, no duplicate tool catalog. For actual execution use **direct validated argv with `shell=False`** for compiled safe supported operations, no arbitrary model-supplied command strings, no shell interpolation, no hidden tool invocation. Apply current ToolContract + ChangeScope authorization **before every step**, with fresh revision compare, host permission, cancellation, process-group cleanup, bounded output, exit/side-effect verification, and stop on first failure. Do not “undo” irreversible external actions automatically.
- One structured result includes step IDs, status, exit codes, useful errors, test summary, timing, CCR index for large exact originals. Evidence/TaskState/feedback update through existing event ingestion, once per step; no double-count. Unknown/missing tool/hook ⇒ disable macro and transparently fall back to ordinary agent tool flow.

**Gate 7:** tests selected correctly on pytest/Cargo/CTest/Jest or Vitest fixtures, suite escalates for risky changes, flakes cannot masquerade as green; three clean macro observations promote one safe workflow; macro reduces model/tool turns in live fixture; a stale scope/step violation cancels it before execution.

# Phase 8 — Full-path integration, evaluation, deployment and completion

**Deliverable:** one coherent implementation with evidence that the 21 features are reachable, safe, and beneficial where applicable. Phase 8 should not require rewriting any earlier design.

### Required end-to-end execution path

```text
headroom wrap
  -> trusted session/worktree + rollout snapshot + JevK5 service health (Phase 1)
  -> AgentEvent ingest -> TaskState + Evidence + TaskContext + graph (Phase 2)
  -> task relevance + hybrid scorer -> SafetyGate/risk -> indexed CCR (Phase 3)
  -> optional real host pre-context hook -> structured delta/interceptors (Phase 4)
  -> global budget -> candidate generation -> safety check -> TransformArbiter (Phase 5)
  -> cache-safe provider request + tool catalog + independent effort/verbosity
  -> host tool proposal -> ChangeScope -> ToolContract -> actual host tool (Phase 6)
  -> evidence updates -> impact test selection -> optional safe macro (Phase 7)
  -> completion evidence / remaining TaskState acceptance checks
```

**Strict order within each request:** trusted identity → user task/evidence snapshot → preserve exact originals → independent safe candidates → deterministic invariant/policy/recovery check → optional Jev advice → score/commit → attribution. A stale authorization must be rechecked before any mutation/verification/macro action. Original stays available. Only the host may claim to have prevented host-owned tool execution.

### Provider/platform coverage

| Surface | Minimum verified behavior |
|---|---|
| Claude Code `wrap` | Proxy caching/live-zone, tool search, host pre/post tool hooks where available, state/evidence, scope/validator, tests/macros. |
| Codex `wrap` | OpenAI Responses and streaming/WS contract, exact tool-call IDs, state/evidence; enforced pre-tool effects **only** if supported host API, else observe. |
| OpenCode 2.x `wrap` | Use current upstream plugin API, avoid stale 1.x hooks; verify provider & plugin routing; host protections only where actually supported. |
| Python proxy | Native Python first-party hooks, exact CCR, typed Jev adviser, savings attribution, no invalid tool injection. |
| Native Rust proxy | Keep v0.41.0 `/stats`/dashboard and response formats working; wire equivalent capabilities only through existing native extension/FFI path when actually used. No assumption Python hooks run here. |
| Windows and Linux | Paths, file locks, SQLite, process ownership, shell differences, no console windows unexpectedly, llama.cpp discovery, no Docker/root requirement. |

### Required tests (automated; add fixtures where none exist)

1. **Baseline parity:** all existing relevant Python/Rust/CLI/streaming/provider tests pass. Preserve unknown model/tool behavior and cache-hot unmodified bytes; newest user instruction exact; Haiku 5.5 system blocks preserved; tool-free requests remain tool-free.
2. **Data safety:** 100k JSON/log result → exact CCR full retrieval plus tiny correct partial retrieval; client-truncated Read never reported complete; missing/expired CCR entry causes pass-through; cross-workspace retrieval and forged project headers rejected.
3. **Jev:** v0.3.3 runtime + 4B v0.3 Q8_0 SHA-256 verification; Python >=3.10; installed llama-server reuse; managed CUDA/CPU fallback; cold offline/absent/corrupt model; distinct external llama-server vs `/v1/systemone` endpoints; malformed response, missing option letters + bounded `n_probs` retry, timeout/OOM, circuit breaker, concurrent decision/metric isolation, external services never killed, both platforms, English-only gating, 8192 effective-context enforcement and >16-option deadline handling. No request waits on downloading a model.
4. **Quality:** two goals on identical tool output retain different correct facts; lossless/irreversible policy gates; numerical, error, ID, negation, exit code and test counts preserved; graph relevance does not hide exact active symbol.
5. **Concurrency:** overlapping task/scope changes, duplicate events, interrupted SQLite update, deterministic replay, stale test/macro authorization; `HEADROOM_STATELESS=1` creates **no** new state files; `HEADROOM_OFFLINE=1` causes **no** Headroom-initiated egress.
6. **Execution safety:** forged cwd header cannot authorize reads/writes; symlink traversal blocked; Windows drive/UNC handling; unexpected generated edits warned; host-gated forbidden mutation blocked; unknown host remains observe; macros cannot elevate/network/deploy/modify credentials.
7. **Lifecycle:** shared-to-dedicated proxy retains feature configuration and extensions; repeated wrap starts share compatible Jev instance without port conflicts; stopping managed child does not kill any foreign process; restart does not lose valid TaskState, learned contract or indexed CCR when persistence enabled.
8. **Performance/quality comparison:** compare original upstream/custom baseline, upgraded with Jev OFF, upgraded with Jev ON. Measure input/output tokens, cache-hit/write costs, *actual* provider-billed savings, p50/p95 additional latency, tool turns and failures, tests run, re-reads/retrievals, task success and acceptance fulfillment. Do not fabricate percentage improvements. Include a holdout; automatically revert a regressing transform family to observe/off, keeping the completed feature available but safe.

### Final operator interface and documentation

- Keep `headroom wrap <client>` as primary entrypoint and current CLI command behavior stable. Expose `headroom intelligence setup/status/doctor/stop` and a compact `headroom agent status/doctor` (or matching current CLI namespace, do not duplicate old commands); report effective rollout mode, capabilities actually enforcing vs observing, trusted root, DB status, JevK5 readiness, selected policies, and per-feature counters.
- Document configuration **only once**: valid rollout names and how to disable each, Jev auto/on/off *service* semantics, offline/stateless, optional local model override/checksum, distinct `HEADROOM_JEVK5_LLAMA_URL` vs `HEADROOM_JEVK5_SYSTEMONE_URL` contracts, protective host-hook requirements, data paths, retention, how to restore/safely disable behavior, and what each feature does. Do not overwrite `CLAUDE.md`, `settings.json`, or user-authored project instructions.
- Keep retention bounded: events 30 days by default; derived per-session state 30 days after last activity; workspace error/recovery aggregate up to 90 days unless user config says otherwise; exact CCR retains its existing TTL; macro max 32/workspace. Prune in background/session end only, never blocking a model request. Operator may purge scoped data without deleting unrelated Headroom memory.
- Final `IMPLEMENTATION_REPORT.md` must be **concise**: starting/ending SHAs, upstream merged SHA, paths of modified components, implemented feature IDs, JevK5 platform/test state, test totals + failures, real benchmark table (Jev off/on), known residual constraints and rollback/disable instructions. No more than ~250 lines unless unavoidable.

### Acceptance and stop condition

The job is **done** only when all 21 feature IDs are invoked through actual supported runtime paths, all nine phase gates pass, no stale code paths/TODO placeholders remain, JevK5 can be automatically set up without the user manually installing it, Windows/Linux flows are tested or explicitly marked unavailable rather than assumed, old custom work/config remains intact, and the final report is committed with the code. **If a host integration cannot truly intercept an action, deliver correct observation instead of falsely claiming prevention.**
