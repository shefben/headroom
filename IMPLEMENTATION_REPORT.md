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
