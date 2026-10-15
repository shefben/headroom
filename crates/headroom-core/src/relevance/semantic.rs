//! Process-owned semantic relevance service (unified-21 #2).
//!
//! Activates the existing fastembed `EmbeddingScorer` for the hot path
//! without blocking it:
//!
//! * **Lazy, once per process.** The first scoring call spawns one
//!   background init thread and answers `None` (callers use BM25). The
//!   outcome — loaded, or a deterministic failure — is final for the life
//!   of the process: there is never a per-request retry.
//! * **Cache-only.** The hot path loads BAAI/bge-small-en-v1.5
//!   (`Xenova/bge-small-en-v1.5`, the fastembed `BGESmallENV15` source)
//!   from a local HuggingFace-layout cache and never touches the network.
//!   [`prefetch_model`] is the explicit, offline-guarded download step,
//!   and it only writes into an operator-chosen directory.
//! * **Fallback.** No `ml` feature, `HEADROOM_RELEVANCE_EMBEDDINGS=off`,
//!   `HEADROOM_OFFLINE` without a cached model, missing ONNX Runtime, a
//!   CPU without AVX2, an uncached model or an inference error all leave
//!   the scorer on deterministic BM25.
//! * **Bounded.** ONNX intra-op threads are capped; item embeddings are
//!   kept in a bounded LRU keyed by text hash, so the per-request query
//!   embedding and repeated tool rows are embedded once.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Kill switch for the semantic half of hybrid relevance.
pub const EMBEDDINGS_ENV: &str = "HEADROOM_RELEVANCE_EMBEDDINGS";
/// Operator-chosen model cache directory (HF hub layout). Consulted
/// first by the cache-only loader and the only place `prefetch_model`
/// writes to unless a directory is passed explicitly.
pub const CACHE_DIR_ENV: &str = "HEADROOM_FASTEMBED_CACHE_DIR";
/// HuggingFace repository fastembed resolves `BGESmallENV15` from.
pub const MODEL_REPO: &str = "Xenova/bge-small-en-v1.5";
/// Embedding LRU capacity (entries; 384 f32 each ≈ 1.5 KB).
pub const EMBED_CACHE_CAPACITY: usize = 4096;
/// Max items embedded per ONNX batch.
pub const EMBED_BATCH: usize = 64;

mod state {
    pub const UNINIT: u8 = 0;
    pub const INITIALIZING: u8 = 1;
    pub const READY: u8 = 2;
    pub const FAILED: u8 = 3;
    pub const DISABLED: u8 = 4;
}

static STATE: AtomicU8 = AtomicU8::new(state::UNINIT);
static INIT_RUNS: AtomicUsize = AtomicUsize::new(0);
static LAST_ERROR: Mutex<Option<String>> = Mutex::new(None);

/// Observable service status (exposed to Python `/health`-style callers).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticStatus {
    pub state: &'static str,
    pub init_runs: usize,
    pub model: &'static str,
    pub error: Option<String>,
    pub cached_embeddings: usize,
}

fn disabled_by_env() -> bool {
    std::env::var(EMBEDDINGS_ENV)
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "off" | "0" | "false" | "no" | "disabled"
            )
        })
        .unwrap_or(false)
}

/// Candidate cache roots, in precedence order: the operator directory,
/// fastembed's own `FASTEMBED_CACHE_DIR`, then the standard HF hub roots.
pub fn cache_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for var in [CACHE_DIR_ENV, "FASTEMBED_CACHE_DIR", "HF_HUB_CACHE"] {
        if let Ok(v) = std::env::var(var) {
            if !v.trim().is_empty() {
                roots.push(PathBuf::from(v));
            }
        }
    }
    if let Ok(v) = std::env::var("HF_HOME") {
        if !v.trim().is_empty() {
            roots.push(PathBuf::from(v).join("hub"));
        }
    }
    for var in ["HOME", "USERPROFILE"] {
        if let Ok(v) = std::env::var(var) {
            if !v.trim().is_empty() {
                roots.push(
                    PathBuf::from(v)
                        .join(".cache")
                        .join("huggingface")
                        .join("hub"),
                );
            }
        }
    }
    roots
}

/// Locate a snapshot directory of [`MODEL_REPO`] that holds every file
/// the loader needs. Never touches the network.
pub fn cached_model_dir() -> Option<PathBuf> {
    const NEEDED: [&str; 5] = [
        "onnx/model.onnx",
        "tokenizer.json",
        "config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
    ];
    let repo_dir = format!("models--{}", MODEL_REPO.replace('/', "--"));
    for root in cache_roots() {
        let snapshots = root.join(&repo_dir).join("snapshots");
        let Ok(entries) = std::fs::read_dir(&snapshots) else {
            continue;
        };
        for snap in entries.flatten() {
            let dir = snap.path();
            if NEEDED.iter().all(|rel| dir.join(rel).is_file()) {
                return Some(dir);
            }
        }
    }
    None
}

// ─── Embedding LRU ────────────────────────────────────────────────────

struct EmbedCache {
    map: HashMap<u64, (Arc<Vec<f32>>, u64)>,
    clock: u64,
}

static CACHE: OnceLock<Mutex<EmbedCache>> = OnceLock::new();

fn cache() -> &'static Mutex<EmbedCache> {
    CACHE.get_or_init(|| {
        Mutex::new(EmbedCache {
            map: HashMap::new(),
            clock: 0,
        })
    })
}

fn text_key(text: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut h);
    h.finish()
}

impl EmbedCache {
    fn get(&mut self, key: u64) -> Option<Arc<Vec<f32>>> {
        self.clock += 1;
        let clock = self.clock;
        self.map.get_mut(&key).map(|(v, stamp)| {
            *stamp = clock;
            Arc::clone(v)
        })
    }

    fn put(&mut self, key: u64, value: Arc<Vec<f32>>) {
        self.clock += 1;
        if self.map.len() >= EMBED_CACHE_CAPACITY && !self.map.contains_key(&key) {
            // Evict the oldest eighth in one pass to amortize the scan.
            let mut stamps: Vec<(u64, u64)> = self.map.iter().map(|(k, (_, s))| (*s, *k)).collect();
            stamps.sort_unstable();
            for (_, k) in stamps.into_iter().take(EMBED_CACHE_CAPACITY / 8) {
                self.map.remove(&k);
            }
        }
        self.map.insert(key, (value, self.clock));
    }
}

// ─── Model slot (ml) ──────────────────────────────────────────────────

#[cfg(feature = "ml")]
static MODEL: OnceLock<Mutex<fastembed::TextEmbedding>> = OnceLock::new();

#[cfg(feature = "ml")]
fn intra_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| (n.get() / 2).clamp(1, 4))
        .unwrap_or(1)
}

#[cfg(feature = "ml")]
fn load_from_cache() -> Result<fastembed::TextEmbedding, String> {
    use fastembed::{
        InitOptionsUserDefined, Pooling, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel,
    };
    let dir = cached_model_dir().ok_or_else(|| {
        format!("{MODEL_REPO} is not in any local model cache; semantic relevance stays on BM25")
    })?;
    if !crate::onnx_cpu::onnx_runtime_supported_by_cpu() {
        return Err("ONNX Runtime requires AVX2 on this CPU".to_string());
    }
    crate::transforms::magika_detector::dynamic_ort_loader_ready()
        .map_err(|e| format!("ONNX Runtime unavailable: {e}"))?;
    let read = |rel: &str| std::fs::read(dir.join(rel)).map_err(|e| format!("read {rel}: {e}"));
    let model = UserDefinedEmbeddingModel::new(
        read("onnx/model.onnx")?,
        TokenizerFiles {
            tokenizer_file: read("tokenizer.json")?,
            config_file: read("config.json")?,
            special_tokens_map_file: read("special_tokens_map.json")?,
            tokenizer_config_file: read("tokenizer_config.json")?,
        },
    )
    .with_pooling(Pooling::Cls);
    TextEmbedding::try_new_from_user_defined(
        model,
        InitOptionsUserDefined::new().with_intra_threads(intra_threads()),
    )
    .map_err(|e| format!("model init failed: {e}"))
}

#[cfg(feature = "ml")]
fn run_init() {
    INIT_RUNS.fetch_add(1, Ordering::Relaxed);
    let ready = match load_from_cache() {
        Ok(m) => MODEL.set(Mutex::new(m)).is_ok() || MODEL.get().is_some(),
        Err(e) => {
            tracing::info!(event = "semantic_relevance_unavailable", reason = %e);
            if let Ok(mut g) = LAST_ERROR.lock() {
                *g = Some(e);
            }
            false
        }
    };
    STATE.store(
        if ready { state::READY } else { state::FAILED },
        Ordering::Release,
    );
    tracing::info!(event = "semantic_relevance_init_complete", ready);
}

/// Non-blocking readiness probe. The first call starts the one-time
/// background init; until it completes every call answers `false`.
pub fn ensure_started() -> bool {
    #[cfg(not(feature = "ml"))]
    {
        STATE.store(state::DISABLED, Ordering::Release);
        false
    }
    #[cfg(feature = "ml")]
    {
        match STATE.load(Ordering::Acquire) {
            state::READY => true,
            state::UNINIT => {
                if disabled_by_env() {
                    STATE.store(state::DISABLED, Ordering::Release);
                    return false;
                }
                if STATE
                    .compare_exchange(
                        state::UNINIT,
                        state::INITIALIZING,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    if let Err(e) = std::thread::Builder::new()
                        .name("semantic-relevance-init".into())
                        .spawn(run_init)
                    {
                        STATE.store(state::FAILED, Ordering::Release);
                        tracing::warn!(event = "semantic_relevance_spawn_failed", error = %e);
                    }
                }
                false
            }
            _ => false,
        }
    }
}

/// Block until the one-time init has settled (startup warm-up and tests;
/// never called from a request path). Returns readiness.
pub fn wait_ready(timeout: std::time::Duration) -> bool {
    ensure_started();
    let start = std::time::Instant::now();
    loop {
        match STATE.load(Ordering::Acquire) {
            state::READY => return true,
            state::FAILED | state::DISABLED => return false,
            _ if start.elapsed() >= timeout => return false,
            _ => std::thread::sleep(std::time::Duration::from_millis(20)),
        }
    }
}

/// Number of init attempts this process made (one, plus one after an
/// explicit successful [`prefetch_model`] that followed a cache miss).
pub fn init_runs() -> usize {
    INIT_RUNS.load(Ordering::Relaxed)
}

pub fn status() -> SemanticStatus {
    let state = match STATE.load(Ordering::Acquire) {
        state::UNINIT => "uninitialized",
        state::INITIALIZING => "initializing",
        state::READY => "ready",
        state::FAILED => "fallback_bm25",
        _ => "disabled",
    };
    SemanticStatus {
        state,
        init_runs: init_runs(),
        model: MODEL_REPO,
        error: LAST_ERROR.lock().ok().and_then(|g| g.clone()),
        cached_embeddings: cache().lock().map(|c| c.map.len()).unwrap_or(0),
    }
}

/// Embed `texts` through the LRU (uncached texts batched). `None` when
/// the model is not ready or inference failed — the caller falls back.
pub fn embed(texts: &[&str]) -> Option<Vec<Arc<Vec<f32>>>> {
    if !ensure_started() {
        return None;
    }
    #[cfg(not(feature = "ml"))]
    {
        let _ = texts;
        None
    }
    #[cfg(feature = "ml")]
    {
        let model = MODEL.get()?;
        let keys: Vec<u64> = texts.iter().map(|t| text_key(t)).collect();
        let mut out: Vec<Option<Arc<Vec<f32>>>> = {
            let mut c = cache().lock().ok()?;
            keys.iter().map(|k| c.get(*k)).collect()
        };
        let missing: Vec<usize> = (0..texts.len()).filter(|&i| out[i].is_none()).collect();
        if !missing.is_empty() {
            let batch: Vec<&str> = missing.iter().map(|&i| texts[i]).collect();
            let vectors = {
                let mut guard = model.lock().ok()?;
                match guard.embed(batch, Some(EMBED_BATCH)) {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::warn!(event = "semantic_relevance_inference_failed", error = %e);
                        return None;
                    }
                }
            };
            if vectors.len() != missing.len() {
                return None;
            }
            let mut c = cache().lock().ok()?;
            for (slot, vec) in missing.into_iter().zip(vectors) {
                let arc = Arc::new(vec);
                c.put(keys[slot], Arc::clone(&arc));
                out[slot] = Some(arc);
            }
        }
        out.into_iter().collect()
    }
}

/// Cosine similarity of each item to `query`, in `[0, 1]`, or `None` to
/// fall back to BM25.
pub fn similarity_batch(items: &[&str], query: &str) -> Option<Vec<f64>> {
    if items.is_empty() || query.trim().is_empty() {
        return None;
    }
    let mut all: Vec<&str> = Vec::with_capacity(items.len() + 1);
    all.push(query);
    all.extend_from_slice(items);
    let vecs = embed(&all)?;
    let q = &vecs[0];
    Some(vecs[1..].iter().map(|v| cosine(q, v)).collect())
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        (dot / (na.sqrt() * nb.sqrt())).clamp(0.0, 1.0)
    }
}

/// Explicit, offline-guarded model download into `cache_dir` (or
/// [`CACHE_DIR_ENV`]). Never called from a request path. Refuses when no
/// directory was chosen, so nothing is ever written to a default
/// home-directory cache implicitly.
pub fn prefetch_model(cache_dir: Option<&str>) -> Result<PathBuf, String> {
    let dir = cache_dir
        .map(str::to_string)
        .filter(|d| !d.trim().is_empty())
        .or_else(|| {
            std::env::var(CACHE_DIR_ENV)
                .ok()
                .filter(|d| !d.trim().is_empty())
        })
        .ok_or_else(|| format!("no model cache directory: set {CACHE_DIR_ENV} or pass one"))?;
    crate::offline::guard_egress("semantic relevance model download", MODEL_REPO)
        .map_err(|e| format!("download refused: {e}"))?;
    #[cfg(not(feature = "ml"))]
    {
        let _ = dir;
        Err("built without the `ml` feature".to_string())
    }
    #[cfg(feature = "ml")]
    {
        use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
        let path = PathBuf::from(&dir);
        std::fs::create_dir_all(&path).map_err(|e| format!("create {dir}: {e}"))?;
        crate::transforms::magika_detector::dynamic_ort_loader_ready()
            .map_err(|e| format!("ONNX Runtime unavailable: {e}"))?;
        TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::BGESmallENV15)
                .with_cache_dir(path.clone())
                .with_show_download_progress(false),
        )
        .map_err(|e| format!("download failed: {e}"))?;
        // A completed download is the one event that may re-arm a cache
        // miss: the next scoring call loads once more from the new cache.
        let _ = STATE.compare_exchange(
            state::FAILED,
            state::UNINIT,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lru_evicts_oldest_and_keeps_bound() {
        let mut c = EmbedCache {
            map: HashMap::new(),
            clock: 0,
        };
        for k in 0..(EMBED_CACHE_CAPACITY as u64 + 10) {
            c.put(k, Arc::new(vec![k as f32]));
        }
        assert!(c.map.len() <= EMBED_CACHE_CAPACITY);
        assert!(c.get(0).is_none(), "oldest entry evicted");
        assert!(c.get(EMBED_CACHE_CAPACITY as u64 + 9).is_some());
    }

    #[test]
    fn cosine_bounds() {
        assert_eq!(cosine(&[1.0, 0.0], &[1.0, 0.0]), 1.0);
        assert_eq!(cosine(&[1.0, 0.0], &[-1.0, 0.0]), 0.0);
        assert_eq!(cosine(&[], &[]), 0.0);
    }

    #[test]
    fn prefetch_without_directory_refuses() {
        let _g = crate::test_support::env_lock();
        let prev = std::env::var(CACHE_DIR_ENV).ok();
        std::env::remove_var(CACHE_DIR_ENV);
        assert!(prefetch_model(None).unwrap_err().contains(CACHE_DIR_ENV));
        if let Some(p) = prev {
            std::env::set_var(CACHE_DIR_ENV, p);
        }
    }
}
