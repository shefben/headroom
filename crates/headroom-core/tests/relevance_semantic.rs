//! Semantic relevance service (unified-21 #2): one init per process, a
//! deterministic BM25 fallback when the model is not cached, and — when a
//! cached model is available (`HEADROOM_FASTEMBED_CACHE_DIR`) — paraphrase
//! relevance with exact-identifier protection.
//!
//! One `#[test]` per binary: the service state is process-global.

use headroom_core::relevance::{semantic, HybridScorer, RelevanceScorer};

fn items() -> Vec<&'static str> {
    vec![
        "the car would not start this morning",
        "database connection pool exhausted after deploy",
        "user login failed with a bad password",
        "compile error in parse_config at src/app/config.py",
    ]
}

#[test]
fn semantic_service_initializes_once_and_falls_back_or_ranks_paraphrases() {
    let model_available = semantic::cached_model_dir().is_some();
    let scorer = HybridScorer::default();
    // Many scoring calls before readiness: never blocks, never re-inits.
    let first: Vec<f64> = scorer
        .score_batch(&items(), "vehicle engine failure")
        .into_iter()
        .map(|s| s.score)
        .collect();
    for _ in 0..50 {
        let _ = scorer.score_batch(&items(), "vehicle engine failure");
    }
    let ready = semantic::wait_ready(std::time::Duration::from_secs(120));
    assert_eq!(semantic::init_runs(), 1, "exactly one init attempt per process");
    for _ in 0..20 {
        let _ = scorer.score_batch(&items(), "auth problem");
    }
    assert_eq!(semantic::init_runs(), 1, "no per-request retry");

    if !model_available {
        // Deterministic BM25 fallback: no lexical overlap -> all zero, and
        // identical to the pre-init answer.
        assert!(!ready);
        assert_eq!(semantic::status().state, "fallback_bm25");
        let again: Vec<f64> = scorer
            .score_batch(&items(), "vehicle engine failure")
            .into_iter()
            .map(|s| s.score)
            .collect();
        assert_eq!(first, again);
        assert!(again.iter().all(|s| *s == 0.0));
        return;
    }
    assert!(ready, "cached model must load: {:?}", semantic::status());
    let para: Vec<f64> = scorer
        .score_batch(&items(), "vehicle engine failure")
        .into_iter()
        .map(|s| s.score)
        .collect();
    let best = para
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(best, 0, "paraphrase match ranks first: {para:?}");
    let exact: Vec<f64> = scorer
        .score_batch(&items(), "parse_config")
        .into_iter()
        .map(|s| s.score)
        .collect();
    assert!(exact[3] >= headroom_core::relevance::EXACT_IDENTIFIER_FLOOR);
    assert!(exact[3] > exact[0] && exact[3] > exact[1] && exact[3] > exact[2]);
    assert!(semantic::status().cached_embeddings > 0);
}
