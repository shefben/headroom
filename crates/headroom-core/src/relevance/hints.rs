//! `DecisionHints`: the compact advisory structure Python hands to the Rust
//! live zone (unified-21 Phase 1B).
//!
//! The Python proxy owns every JevK5 call (`headroom/intelligence/advisor.py`).
//! Rust never talks to llama-server or a `/v1/systemone` gateway; it receives
//! the outcome of Python-side decisions as this typed, versioned hint at the
//! FFI boundary, inside the task query JSON (`"hints"`), mirroring
//! `headroom.agent.interfaces.DecisionHints.to_dict()`.
//!
//! Hints are **relevance-only**: their `query_terms` widen the relevance
//! query the live-zone compressors rank with. They never add protected exact
//! entities, never reach the safety gate's invariant check and never
//! authorize anything. An advisory (JevK5) hint's weight is capped at
//! [`MAX_ADVISOR_WEIGHT`]; a deterministic hint always carries weight 0.

use serde_json::Value;

/// Wire version of the hint structure (`DECISION_HINTS_VERSION` in Python).
pub const DECISION_HINTS_VERSION: i64 = 1;
/// Maximum relevance terms retained from one hint.
pub const MAX_HINT_TERMS: usize = 64;
/// Maximum advisory options retained (JevK5 reads at most 256 per decision).
pub const MAX_ADVISORY_OPTIONS: usize = 256;
/// The advisor utility weight never exceeds this, whatever the sender says.
pub const MAX_ADVISOR_WEIGHT: f64 = 0.40;
/// Bytes per relevance term.
const MAX_TERM_BYTES: usize = 120;

/// Where a hint came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HintSource {
    /// Derived by deterministic Python code (query builder).
    #[default]
    Deterministic,
    /// The JevK5 decision advisor.
    Jevk5,
}

impl HintSource {
    pub fn as_str(self) -> &'static str {
        match self {
            HintSource::Deterministic => "deterministic",
            HintSource::Jevk5 => "jevk5",
        }
    }
}

/// Validated, bounded advisory hints for one request.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DecisionHints {
    pub family: String,
    pub source: HintSource,
    pub query_terms: Vec<String>,
    /// Option label -> probability, in sender order.
    pub advisory: Vec<(String, f64)>,
    pub choice: Option<String>,
    pub confidence: f64,
    weight: f64,
}

fn unit_interval(v: &Value, what: &str) -> Result<f64, String> {
    let x = v
        .as_f64()
        .ok_or_else(|| format!("DecisionHints {what} must be a number"))?;
    if !x.is_finite() || !(0.0..=1.0).contains(&x) {
        return Err(format!("DecisionHints {what} must be in [0, 1]"));
    }
    Ok(x)
}

impl DecisionHints {
    /// Parse `DecisionHints.to_dict()`. Unknown versions, unknown sources,
    /// non-finite or out-of-range probabilities and a choice that is not an
    /// advisory option are rejected so a schema change or a corrupted hint
    /// can never be silently misread.
    pub fn from_json(value: &Value) -> Result<Self, String> {
        let obj = value.as_object().ok_or("DecisionHints must be an object")?;
        match obj.get("v").and_then(Value::as_i64) {
            Some(DECISION_HINTS_VERSION) => {}
            other => return Err(format!("unsupported DecisionHints version {other:?}")),
        }
        let source = match obj
            .get("source")
            .and_then(Value::as_str)
            .unwrap_or("deterministic")
        {
            "deterministic" => HintSource::Deterministic,
            "jevk5" => HintSource::Jevk5,
            other => return Err(format!("unknown DecisionHints source {other:?}")),
        };
        let family = obj
            .get("family")
            .and_then(Value::as_str)
            .unwrap_or("")
            .chars()
            .take(64)
            .collect::<String>();
        let mut query_terms: Vec<String> = Vec::new();
        if let Some(terms) = obj.get("query_terms").and_then(Value::as_array) {
            for term in terms.iter().filter_map(Value::as_str) {
                let term = term.trim();
                if term.len() < 2 || term.len() > MAX_TERM_BYTES {
                    continue;
                }
                if !query_terms.iter().any(|t| t == term) {
                    query_terms.push(term.to_string());
                }
                if query_terms.len() >= MAX_HINT_TERMS {
                    break;
                }
            }
        }
        let mut advisory: Vec<(String, f64)> = Vec::new();
        if let Some(map) = obj.get("advisory").and_then(Value::as_object) {
            if map.len() > MAX_ADVISORY_OPTIONS {
                return Err("DecisionHints advisory has too many options".into());
            }
            for (label, p) in map {
                advisory.push((label.clone(), unit_interval(p, "advisory probability")?));
            }
        }
        let choice = obj
            .get("choice")
            .and_then(Value::as_str)
            .map(str::to_string);
        if let Some(c) = &choice {
            if !advisory.is_empty() && !advisory.iter().any(|(label, _)| label == c) {
                return Err("DecisionHints choice is not an advisory option".into());
            }
        }
        let confidence = match obj.get("confidence") {
            None | Some(Value::Null) => 0.0,
            Some(v) => unit_interval(v, "confidence")?,
        };
        let raw_weight = match obj.get("weight") {
            None | Some(Value::Null) => 0.0,
            Some(v) => {
                let w = v.as_f64().ok_or("DecisionHints weight must be a number")?;
                if !w.is_finite() || w < 0.0 {
                    return Err("DecisionHints weight must be finite and non-negative".into());
                }
                w
            }
        };
        let weight = match source {
            HintSource::Deterministic => 0.0,
            HintSource::Jevk5 => raw_weight.min(MAX_ADVISOR_WEIGHT),
        };
        Ok(DecisionHints {
            family,
            source,
            query_terms,
            advisory,
            choice,
            confidence,
            weight,
        })
    }

    /// The utility weight a consumer may give the advisory (0 for
    /// deterministic hints, never above [`MAX_ADVISOR_WEIGHT`]).
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// Terms that may widen the relevance query (never protection).
    pub fn relevance_terms(&self) -> &[String] {
        &self.query_terms
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_python_to_dict_shape() {
        let h = DecisionHints::from_json(&json!({
            "v": 1, "family": "transform_arbiter", "source": "jevk5",
            "query_terms": ["parse_config", "x", "parse_config", "KeyError"],
            "advisory": {"c0": 0.7, "c1": 0.3}, "choice": "c0",
            "confidence": 0.7, "weight": 0.9
        }))
        .unwrap();
        assert_eq!(h.source, HintSource::Jevk5);
        assert_eq!(h.relevance_terms(), ["parse_config", "KeyError"]);
        assert_eq!(h.choice.as_deref(), Some("c0"));
        assert!(
            (h.weight() - MAX_ADVISOR_WEIGHT).abs() < 1e-12,
            "weight capped"
        );
    }

    #[test]
    fn deterministic_hints_carry_no_weight() {
        let h = DecisionHints::from_json(&json!({
            "v": 1, "family": "relevance_query", "source": "deterministic",
            "query_terms": ["foo.py"], "advisory": {}, "choice": null,
            "confidence": 0.0, "weight": 0.3
        }))
        .unwrap();
        assert_eq!(h.weight(), 0.0);
    }

    #[test]
    fn rejects_malformed_hints() {
        for bad in [
            json!({"v": 2}),
            json!({"family": "x"}),
            json!({"v": 1, "source": "oracle"}),
            json!({"v": 1, "advisory": {"a": 1.5}}),
            json!({"v": 1, "advisory": {"a": "high"}}),
            json!({"v": 1, "advisory": {"a": 0.5}, "choice": "b"}),
            json!({"v": 1, "confidence": -0.1}),
            json!({"v": 1, "source": "jevk5", "weight": -1.0}),
            json!([1, 2]),
        ] {
            assert!(DecisionHints::from_json(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn bounds_terms() {
        let terms: Vec<String> = (0..200).map(|i| format!("term{i}")).collect();
        let h = DecisionHints::from_json(&json!({"v": 1, "query_terms": terms})).unwrap();
        assert_eq!(h.relevance_terms().len(), MAX_HINT_TERMS);
    }
}
