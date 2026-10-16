//! Relevance scoring — Rust port of `headroom/relevance/`.
//!
//! Used by SmartCrusher's planning layer to decide which items in a tool
//! output match the user's query (the user's recent prompts plus the
//! assistant's tool-call argument JSON, joined). Items above a relevance
//! threshold are pinned into `keep_indices`.
//!
//! # Scorer ladder
//!
//! 1. **BM25** (`bm25`): keyword overlap with TF-IDF + length
//!    normalization. No ML deps. Excellent for exact-match cases (UUIDs,
//!    field=value filters). Tool-call arguments are usually literal
//!    keywords that appear verbatim in the response, so BM25 catches
//!    most cases.
//! 2. **Embedding** (`embedding`): fastembed BGE-small ONNX model for
//!    semantic matching when query and items use different vocabularies.
//! 3. **Hybrid** (`hybrid`): fuses BM25 and embedding signals. The
//!    default scorer draws on the process-owned `semantic` service (lazy,
//!    cache-only, bounded) and stays on boosted BM25 until it is ready.
//!
//! `query` holds the task relevance query shared with the live zone and
//! the safety gate.
//!
//! Each scorer implements the `RelevanceScorer` trait — same surface
//! as Python's abstract base class.

mod base;
mod bm25;
mod embedding;
pub mod hints;
mod hybrid;
pub mod query;
pub mod semantic;

pub use base::{default_batch_score, RelevanceScore, RelevanceScorer};
pub use bm25::BM25Scorer;
pub use embedding::EmbeddingScorer;
pub use hybrid::{exact_identifier_terms, HybridScorer, EXACT_IDENTIFIER_FLOOR};
pub use hints::{DecisionHints, HintSource};
pub use query::TaskQuery;

/// Factory mirroring Python's `relevance.create_scorer` (`__init__.py:72`).
///
/// Returns a boxed trait object so callers don't have to know which
/// concrete scorer they got. `tier`:
///
/// - `"hybrid"` (default) — `HybridScorer` (BM25 + embedding fusion;
///   gracefully falls back to BM25 + boost when embeddings stubbed).
/// - `"bm25"` — `BM25Scorer` (pure keyword).
/// - `"embedding"` — the shared semantic service when its model is
///   loaded; `Err` (mirroring Python's `RuntimeError`) while it is not.
pub fn create_scorer(tier: &str) -> Result<Box<dyn RelevanceScorer + Send + Sync>, String> {
    match tier.to_lowercase().as_str() {
        "bm25" => Ok(Box::new(BM25Scorer::default())),
        "hybrid" => Ok(Box::new(HybridScorer::default())),
        "embedding" => {
            if semantic::ensure_started() {
                Ok(Box::new(SharedEmbeddingScorer))
            } else {
                Err(format!(
                    "EmbeddingScorer unavailable ({}); semantic relevance stays on BM25",
                    semantic::status().state
                ))
            }
        }
        other => Err(format!(
            "Unknown scorer tier: {}. Valid tiers: bm25, embedding, hybrid",
            other
        )),
    }
}

/// Pure-embedding scorer over the process-owned semantic service.
struct SharedEmbeddingScorer;

impl RelevanceScorer for SharedEmbeddingScorer {
    fn score(&self, item: &str, context: &str) -> RelevanceScore {
        self.score_batch(&[item], context).pop().unwrap_or_default()
    }

    fn score_batch(&self, items: &[&str], context: &str) -> Vec<RelevanceScore> {
        match semantic::similarity_batch(items, context) {
            Some(sims) => sims
                .into_iter()
                .map(|s| RelevanceScore::new(s, format!("Embedding: {s:.2}"), Vec::new()))
                .collect(),
            None => items
                .iter()
                .map(|_| RelevanceScore::empty("Embedding: model not available"))
                .collect(),
        }
    }

    fn is_available(&self) -> bool {
        semantic::ensure_started()
    }
}
