//! Unified-21 Phase 3 bindings: the central SafetyGate / backoff ladder
//! (#7, #13), the process-owned semantic relevance service (#2) and the
//! task-query plumbing for the live zone (#1).
//!
//! Structured arguments cross the boundary as JSON strings so the Python
//! wrapper (`headroom/transforms/safety_gate.py`) owns the typed objects
//! and the Rust gate stays the single implementation.

use headroom_core::compression_policy::CompressionPolicy;
use headroom_core::relevance::{semantic, TaskQuery};
use headroom_core::safety_gate::{self, Candidate, Decision, GatePolicy, Provenance, Tier};
use headroom_core::tokenizer::get_tokenizer;
use headroom_core::transforms::live_zone::DEFAULT_MODEL;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use serde_json::{json, Value};

fn parse_json(raw: Option<&str>, what: &str) -> PyResult<Option<Value>> {
    match raw {
        None => Ok(None),
        Some(s) if s.trim().is_empty() || s.trim() == "null" => Ok(None),
        Some(s) => serde_json::from_str(s)
            .map(Some)
            .map_err(|e| PyValueError::new_err(format!("invalid {what} JSON: {e}"))),
    }
}

pub(crate) fn task_from_json(raw: Option<&str>) -> PyResult<TaskQuery> {
    match parse_json(raw, "task")? {
        None => Ok(TaskQuery::default()),
        Some(v) => TaskQuery::from_json(&v).map_err(PyValueError::new_err),
    }
}

/// `None` (absent) stays `None` so the gate fails to original; present but
/// malformed limits become NaN, which the gate also rejects.
fn gate_policy_from_json(raw: Option<&str>) -> PyResult<Option<GatePolicy>> {
    Ok(parse_json(raw, "policy")?.map(|v| GatePolicy {
        max_lossy_ratio: v
            .get("max_lossy_ratio")
            .and_then(Value::as_f64)
            .unwrap_or(f64::NAN),
        volatile_token_threshold: v
            .get("volatile_token_threshold")
            .and_then(Value::as_u64)
            .map(|n| n.min(u64::from(u32::MAX)) as u32)
            .unwrap_or(0),
    }))
}

pub(crate) fn compression_policy_from_json(
    raw: Option<&str>,
) -> PyResult<Option<CompressionPolicy>> {
    Ok(gate_policy_from_json(raw)?.map(|g| {
        let mut p = CompressionPolicy::for_mode(headroom_core::auth_mode::AuthMode::Payg);
        p.max_lossy_ratio = g.max_lossy_ratio as f32;
        p.volatile_token_threshold = g.volatile_token_threshold;
        p
    }))
}

fn provenance_from_value(v: Option<&Value>) -> PyResult<Provenance> {
    match v {
        None | Some(Value::Null) => Ok(Provenance::default()),
        Some(v) => serde_json::from_value(v.clone())
            .map_err(|e| PyValueError::new_err(format!("invalid provenance: {e}"))),
    }
}

fn tier_from(s: &str) -> PyResult<Tier> {
    Tier::parse(s).ok_or_else(|| PyValueError::new_err(format!("unknown ladder tier {s:?}")))
}

fn decision_json(d: &Decision) -> Value {
    let mut v = serde_json::to_value(d).unwrap_or(Value::Null);
    if let Value::Object(map) = &mut v {
        map.insert("verdict".into(), Value::String(d.verdict().into()));
    }
    v
}

fn model_or_default(model: &str) -> &str {
    if model.is_empty() {
        DEFAULT_MODEL
    } else {
        model
    }
}

/// Evaluate one candidate. Returns the decision as a JSON string.
#[pyfunction]
#[pyo3(signature = (original, candidate, task = None, policy = None, provenance = None, tier = "aggressive", model = ""))]
pub(crate) fn safety_gate_check(
    py: Python<'_>,
    original: &str,
    candidate: &str,
    task: Option<&str>,
    policy: Option<&str>,
    provenance: Option<&str>,
    tier: &str,
    model: &str,
) -> PyResult<String> {
    let task = task_from_json(task)?;
    let policy = gate_policy_from_json(policy)?;
    let prov = provenance_from_value(parse_json(provenance, "provenance")?.as_ref())?;
    let tier = tier_from(tier)?;
    let model = model_or_default(model).to_string();
    let d = py.detach(|| {
        let tok = get_tokenizer(&model);
        safety_gate::check(
            original,
            candidate,
            &task,
            policy.as_ref(),
            &prov,
            tier,
            tok.as_ref(),
        )
    });
    Ok(decision_json(&d).to_string())
}

/// Run the central backoff ladder over `candidates_json`
/// (`[{"tier": ..., "text": ..., "provenance": {...}}]`). Returns
/// `{"chosen": index|null, "tier": ..., "decisions": [{"index", ...}]}`.
#[pyfunction]
#[pyo3(signature = (original, candidates_json, task = None, policy = None, model = ""))]
pub(crate) fn safety_gate_select(
    py: Python<'_>,
    original: &str,
    candidates_json: &str,
    task: Option<&str>,
    policy: Option<&str>,
    model: &str,
) -> PyResult<String> {
    let task = task_from_json(task)?;
    let policy = gate_policy_from_json(policy)?;
    let raw: Vec<Value> = serde_json::from_str(candidates_json)
        .map_err(|e| PyValueError::new_err(format!("invalid candidates JSON: {e}")))?;
    let mut cands = Vec::with_capacity(raw.len());
    for c in &raw {
        cands.push(Candidate {
            tier: tier_from(
                c.get("tier")
                    .and_then(Value::as_str)
                    .unwrap_or("aggressive"),
            )?,
            text: c
                .get("text")
                .and_then(Value::as_str)
                .ok_or_else(|| PyValueError::new_err("candidate without text"))?
                .to_string(),
            provenance: provenance_from_value(c.get("provenance"))?,
        });
    }
    let model = model_or_default(model).to_string();
    let sel = py.detach(|| {
        let tok = get_tokenizer(&model);
        safety_gate::select(original, &cands, &task, policy.as_ref(), tok.as_ref())
    });
    let decisions: Vec<Value> = sel
        .decisions
        .iter()
        .map(|(i, d)| {
            let mut v = decision_json(d);
            if let Value::Object(m) = &mut v {
                m.insert("index".into(), json!(i));
                m.insert("tier".into(), json!(cands[*i].tier.as_str()));
            }
            v
        })
        .collect();
    Ok(json!({"chosen": sel.chosen, "tier": sel.tier, "decisions": decisions}).to_string())
}

/// Policy-fit conservative candidate (see `safety_gate::fit_lossy_budget`).
#[pyfunction]
#[pyo3(signature = (original, aggressive = None, task = None, policy = None, content_type = "text", model = ""))]
pub(crate) fn safety_gate_fit(
    py: Python<'_>,
    original: &str,
    aggressive: Option<&str>,
    task: Option<&str>,
    policy: Option<&str>,
    content_type: &str,
    model: &str,
) -> PyResult<Option<String>> {
    let task = task_from_json(task)?;
    let Some(policy) = gate_policy_from_json(policy)? else {
        return Ok(None);
    };
    if !safety_gate::line_fittable(content_type) {
        return Ok(None);
    }
    let model = model_or_default(model).to_string();
    Ok(py.detach(|| {
        let tok = get_tokenizer(&model);
        safety_gate::fit_lossy_budget(
            original,
            aggressive,
            &task,
            &policy,
            content_type,
            tok.as_ref(),
        )
    }))
}

/// Structural/lossless JSON candidate, or `None`.
#[pyfunction]
pub(crate) fn safety_gate_minify_json(original: &str) -> Option<String> {
    safety_gate::minify_json(original)
}

/// CCR exact-externalization stub for an original already stored under `hash`.
#[pyfunction]
pub(crate) fn safety_gate_ccr_stub(original: &str, hash: &str) -> String {
    safety_gate::ccr_exact_stub(original, hash)
}

/// Semantic relevance service status (non-blocking).
#[pyfunction]
pub(crate) fn semantic_relevance_status() -> String {
    let s = semantic::status();
    json!({
        "state": s.state,
        "init_runs": s.init_runs,
        "model": s.model,
        "error": s.error,
        "cached_embeddings": s.cached_embeddings,
    })
    .to_string()
}

/// Start (if needed) and wait for the one-time init. Startup/test use only.
#[pyfunction]
#[pyo3(signature = (timeout_s = 30.0))]
pub(crate) fn semantic_relevance_wait_ready(py: Python<'_>, timeout_s: f64) -> bool {
    let timeout = std::time::Duration::from_secs_f64(timeout_s.max(0.0));
    py.detach(|| semantic::wait_ready(timeout))
}

/// Explicit offline-guarded model download into an operator directory.
#[pyfunction]
#[pyo3(signature = (cache_dir = None))]
pub(crate) fn semantic_relevance_prefetch(
    py: Python<'_>,
    cache_dir: Option<String>,
) -> PyResult<String> {
    py.detach(|| semantic::prefetch_model(cache_dir.as_deref()))
        .map(|p| p.display().to_string())
        .map_err(PyRuntimeError::new_err)
}

/// Hybrid relevance scores (BM25 + shared semantic when loaded).
#[pyfunction]
pub(crate) fn hybrid_relevance_scores(py: Python<'_>, items: Vec<String>, query: &str) -> Vec<f64> {
    use headroom_core::relevance::{HybridScorer, RelevanceScorer};
    py.detach(|| {
        let refs: Vec<&str> = items.iter().map(String::as_str).collect();
        HybridScorer::default()
            .score_batch(&refs, query)
            .into_iter()
            .map(|s| s.score)
            .collect()
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(safety_gate_check, m)?)?;
    m.add_function(wrap_pyfunction!(safety_gate_select, m)?)?;
    m.add_function(wrap_pyfunction!(safety_gate_fit, m)?)?;
    m.add_function(wrap_pyfunction!(safety_gate_minify_json, m)?)?;
    m.add_function(wrap_pyfunction!(safety_gate_ccr_stub, m)?)?;
    m.add_function(wrap_pyfunction!(semantic_relevance_status, m)?)?;
    m.add_function(wrap_pyfunction!(semantic_relevance_wait_ready, m)?)?;
    m.add_function(wrap_pyfunction!(semantic_relevance_prefetch, m)?)?;
    m.add_function(wrap_pyfunction!(hybrid_relevance_scores, m)?)?;
    Ok(())
}
