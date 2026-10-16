//! Task relevance query shared by the live-zone dispatcher and the
//! safety gate (unified-21 #1 / #7).
//!
//! The Python agent runtime builds a `RelevanceQuery`
//! (`headroom/agent/interfaces.py`) from the authentic latest user goal,
//! the tool intent, exact user-named entities, the active error and graph
//! neighbors, and serializes it with `to_dict()`. [`TaskQuery::from_json`]
//! is the Rust side of that boundary.
//!
//! When no explicit query is handed in (the native Rust proxy has no
//! agent runtime), [`TaskQuery::from_user_text`] derives one from the
//! request's own latest user text so the compressors never run with an
//! empty query while a real user ask exists.

use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;

use super::hints::DecisionHints;

/// Hard cap on the query text handed to compressors (bytes). The Python
/// builder caps at 512 tokens; ~3 bytes/token keeps the two in step.
pub const QUERY_TEXT_MAX_BYTES: usize = 1536;
/// Maximum exact entities retained.
pub const MAX_ENTITIES: usize = 24;

/// Compact, typed relevance query.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TaskQuery {
    /// Query text in priority order (entities, error, goal, intent).
    pub text: String,
    /// Exact identifiers, paths, symbols and error ids the user named.
    pub exact_entities: Vec<String>,
    /// Signature of the active failure, if any.
    pub active_error: String,
    /// Whether the task is a code task (graph neighbors were consulted).
    pub is_code: bool,
    /// Advisory relevance hints from Python (`"hints"` in the query JSON).
    /// Relevance-only: they widen [`TaskQuery::live_zone_query`] and are
    /// never consulted by the safety gate's invariants.
    pub hints: Option<DecisionHints>,
}

static BACKTICK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"`([^`\n]{2,120})`").expect("backtick regex"));
static PATH: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?:[A-Za-z]:[\\/])?(?:\.{1,2}[\\/])?(?:[\w.\-]+[\\/])*[\w\-]+\.(?:py|pyi|js|jsx|ts|tsx|go|rs|java|kt|rb|c|h|cc|cpp|hpp|cs|swift|php|lua|zig|sql|md|rst|toml|ya?ml|json|ini|cfg|sh|ps1|txt|html|css|scss|vue|svelte|ex|exs|scala|r|jl)\b",
    )
    .expect("path regex")
});
static CAMEL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(?:[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+|[a-z]+(?:[A-Z][a-z0-9]+)+)\b")
        .expect("camel regex")
});
static SNAKE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(?:[a-z][a-z0-9]*(?:_[a-z0-9]+)+|[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\b")
        .expect("snake regex")
});
static CALL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b([A-Za-z_][\w.]*)\(\)").expect("call regex"));
static ERROR_ID: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(?:\w*(?:Error|Exception)|E\d{3,5}|TS\d{4}|HTTP\s?[1-5]\d\d)\b")
        .expect("error id regex")
});
static IDENT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"\b(?:[0-9a-f]{7,40}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}|[A-Z][A-Z0-9]{1,9}-\d{1,6})\b",
    )
    .expect("identifier regex")
});
static HARNESS_WRAPPER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<(system-reminder|command-[a-z-]+|local-command-[a-z-]+|environment_context|user_instructions)>.*?</[a-z_-]+>")
        .expect("harness wrapper regex")
});

/// Distinct exact entities named in `text`, in priority order
/// (backticked, paths, calls, CamelCase, snake_case, error ids, ids).
pub fn extract_entities(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut push = |v: &str| {
        let cleaned = v
            .trim()
            .trim_matches(|c: char| ".,;:'\"()[]{}".contains(c))
            .trim_end_matches("()");
        if cleaned.len() >= 2 && !out.iter().any(|e| e == cleaned) && out.len() < MAX_ENTITIES {
            out.push(cleaned.to_string());
        }
    };
    for c in BACKTICK.captures_iter(text) {
        push(&c[1]);
    }
    for m in PATH.find_iter(text) {
        push(m.as_str());
    }
    for c in CALL.captures_iter(text) {
        push(&c[1]);
    }
    for m in CAMEL.find_iter(text) {
        push(m.as_str());
    }
    for m in SNAKE.find_iter(text) {
        push(m.as_str());
    }
    for m in ERROR_ID.find_iter(text) {
        push(m.as_str());
    }
    for m in IDENT.find_iter(text) {
        push(m.as_str());
    }
    out
}

fn clip_bytes(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

impl TaskQuery {
    /// True when there is nothing to condition on.
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty() && self.exact_entities.is_empty()
    }

    /// Build from an explicit query text + entities, enforcing the caps.
    pub fn new(text: &str, exact_entities: Vec<String>, active_error: &str, is_code: bool) -> Self {
        let mut entities = exact_entities;
        entities.retain(|e| e.trim().len() >= 2);
        entities.truncate(MAX_ENTITIES);
        TaskQuery {
            text: clip_bytes(text.trim(), QUERY_TEXT_MAX_BYTES).to_string(),
            exact_entities: entities,
            active_error: clip_bytes(active_error.trim(), 400).to_string(),
            is_code,
            hints: None,
        }
    }

    /// Derive a query from raw latest-user text (harness wrappers
    /// stripped). Entities are budgeted before prose, mirroring the
    /// Python `RelevanceQuery.text` priority.
    pub fn from_user_text(raw: &str) -> Self {
        let stripped = HARNESS_WRAPPER.replace_all(raw, " ");
        let goal = stripped.trim();
        if goal.is_empty() {
            return TaskQuery::default();
        }
        let entities = extract_entities(goal);
        let mut text = String::new();
        for e in &entities {
            if text.len() + e.len() + 1 > QUERY_TEXT_MAX_BYTES / 2 {
                break;
            }
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(e);
        }
        let remaining = QUERY_TEXT_MAX_BYTES.saturating_sub(text.len() + 1);
        if remaining > 0 {
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(clip_bytes(goal, remaining));
        }
        TaskQuery::new(&text, entities, "", false)
    }

    /// Parse the Python `RelevanceQuery.to_dict()` shape (or a bare
    /// `{"text": ..., "exact_entities": [...]}`). Unknown versions are
    /// rejected so a schema change cannot be silently misread.
    pub fn from_json(value: &Value) -> Result<Self, String> {
        let obj = value
            .as_object()
            .ok_or("relevance query must be an object")?;
        if let Some(v) = obj.get("v") {
            if v.as_i64() != Some(1) {
                return Err(format!("unsupported RelevanceQuery version {v}"));
            }
        }
        let entities: Vec<String> = obj
            .get("exact_entities")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        let active_error = obj
            .get("active_error")
            .and_then(Value::as_str)
            .unwrap_or("");
        let is_code = obj.get("is_code").and_then(Value::as_bool).unwrap_or(false);
        let text = match obj.get("text").and_then(Value::as_str) {
            Some(t) if !t.trim().is_empty() => t.to_string(),
            _ => {
                // Rebuild in the Python priority order when only fields came.
                let goal = obj
                    .get("explicit_user_goal")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let intent = obj.get("tool_intent").and_then(Value::as_str).unwrap_or("");
                [
                    entities.join(" "),
                    active_error.to_string(),
                    goal.to_string(),
                    intent.to_string(),
                ]
                .into_iter()
                .filter(|s| !s.trim().is_empty())
                .collect::<Vec<_>>()
                .join("\n")
            }
        };
        let hints = match obj.get("hints") {
            None | Some(Value::Null) => None,
            Some(h) => Some(DecisionHints::from_json(h)?),
        };
        let mut query = TaskQuery::new(&text, entities, active_error, is_code);
        query.hints = hints;
        Ok(query)
    }

    /// Query string handed to compressors: the text, with the active
    /// error appended when the text does not already carry it.
    pub fn compressor_query(&self) -> String {
        if self.active_error.is_empty() || self.text.contains(&self.active_error) {
            self.text.clone()
        } else if self.text.is_empty() {
            self.active_error.clone()
        } else {
            format!("{}\n{}", self.text, self.active_error)
        }
    }

    /// Query string for the live-zone compressors: [`Self::compressor_query`]
    /// widened by the hint relevance terms it does not already contain
    /// (bounded to [`QUERY_TEXT_MAX_BYTES`] extra bytes). Ranking only: the
    /// safety gate keeps using `compressor_query` and the exact entities.
    pub fn live_zone_query(&self) -> String {
        let mut query = self.compressor_query();
        let Some(hints) = &self.hints else {
            return query;
        };
        let lower = query.to_ascii_lowercase();
        let mut extra = String::new();
        for term in hints.relevance_terms() {
            if lower.contains(&term.to_ascii_lowercase()) {
                continue;
            }
            if extra.len() + term.len() + 1 > QUERY_TEXT_MAX_BYTES {
                break;
            }
            if !extra.is_empty() {
                extra.push(' ');
            }
            extra.push_str(term);
        }
        if !extra.is_empty() {
            if !query.is_empty() {
                query.push('\n');
            }
            query.push_str(&extra);
        }
        query
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entities_prioritize_exact_names() {
        let e =
            extract_entities("fix `parse_config` in src/app/config.py raising KeyError at HR-123");
        assert_eq!(e[0], "parse_config");
        assert!(e.contains(&"src/app/config.py".to_string()));
        assert!(e.contains(&"KeyError".to_string()));
        assert!(e.contains(&"HR-123".to_string()));
    }

    #[test]
    fn from_user_text_strips_harness_wrappers_and_caps() {
        let raw = format!(
            "<system-reminder>ignore me SECRETWORD</system-reminder>why does test_eviction_order fail? {}",
            "x ".repeat(4000)
        );
        let q = TaskQuery::from_user_text(&raw);
        assert!(!q.text.contains("SECRETWORD"));
        assert!(q.text.starts_with("test_eviction_order"));
        assert!(q.text.len() <= QUERY_TEXT_MAX_BYTES);
    }

    #[test]
    fn from_json_matches_python_dict_and_rejects_unknown_version() {
        let v = serde_json::json!({
            "v": 1, "explicit_user_goal": "why", "tool_intent": "test: pytest",
            "exact_entities": ["foo.py"], "active_error": "KeyError: x",
            "graph_neighbors": [], "is_code": true, "max_tokens": 512,
            "text": "why\nfoo.py"
        });
        let q = TaskQuery::from_json(&v).unwrap();
        assert_eq!(q.exact_entities, vec!["foo.py"]);
        assert!(q.is_code);
        assert!(q.compressor_query().contains("KeyError: x"));
        assert!(TaskQuery::from_json(&serde_json::json!({"v": 9})).is_err());
    }

    #[test]
    fn hints_cross_the_boundary_and_widen_only_the_live_zone_query() {
        let v = serde_json::json!({
            "v": 1, "text": "fix the parser", "exact_entities": [],
            "hints": {
                "v": 1, "family": "relevance_query", "source": "deterministic",
                "query_terms": ["parser", "src/deep/config_loader.py"],
                "advisory": {}, "choice": null, "confidence": 0.0, "weight": 0.0
            }
        });
        let q = TaskQuery::from_json(&v).unwrap();
        let hints = q.hints.as_ref().expect("hints parsed");
        assert_eq!(hints.relevance_terms().len(), 2);
        assert_eq!(q.compressor_query(), "fix the parser");
        assert_eq!(
            q.live_zone_query(),
            "fix the parser\nsrc/deep/config_loader.py",
            "terms already in the text are not repeated"
        );
        // A malformed hint is an error, never silently dropped.
        let bad = serde_json::json!({"v": 1, "text": "x", "hints": {"v": 7}});
        assert!(TaskQuery::from_json(&bad).is_err());
        // Absent / null hints keep the old behavior.
        let plain = TaskQuery::from_json(&serde_json::json!({"v": 1, "text": "x", "hints": null}));
        assert!(plain.unwrap().hints.is_none());
    }

    #[test]
    fn empty_text_is_empty_query() {
        assert!(TaskQuery::from_user_text("  <system-reminder>x</system-reminder> ").is_empty());
    }
}
