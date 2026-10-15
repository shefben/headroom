//! SafetyGate: deterministic semantic invariants, the enforced lossy-risk
//! policy and the central backoff ladder (unified-21 #7 / #13).
//!
//! Every compression candidate — from the Rust live-zone dispatcher, or
//! from the Python `ContentRouter` through the `headroom._core` bindings —
//! passes [`check`] before it may replace original bytes. The gate is the
//! single implementation; Python wraps it rather than re-implementing it,
//! and `tests/fixtures/safety_gate_parity.json` pins the verdicts both
//! sides must agree on.
//!
//! # What is protected
//!
//! Facts are extracted from the *original* and must survive verbatim in
//! the candidate, or be covered by a verified exact retrieval payload:
//!
//! * user-named entities (ids, paths, symbols, URLs, hashes) present in the
//!   original;
//! * failure / exception lines and traceback frames (`file:line`), except in
//!   search hits and JSON, where only user-named entities are hard-protected
//!   (a grep for `Error` makes every hit a "failure line");
//! * exit / status codes and test pass/fail/skip counts;
//! * definition lines of user-named symbols (active signatures);
//! * diff file headers and hunk headers;
//! * CCR identifiers already present in the original;
//! * truncation / partial boundaries (never claim completeness for input
//!   that was already truncated);
//! * for JSON: syntax validity, boolean values and the key set.
//!
//! "Meaningful numeric values / units / dates" are protected through the
//! lines that carry them (failure, summary, status lines) and through
//! user-named entities; numbers elsewhere are ordinary content subject to
//! the lossy budget.
//!
//! # Policy
//!
//! `irreversible_loss_ratio` (tokens dropped with no exact recovery path)
//! may not exceed `CompressionPolicy::max_lossy_ratio`; `wire_drop_ratio`
//! (tokens removed from the wire) is reported separately and is not
//! limited when the dropped bytes are exactly recoverable. Volatile tokens
//! (timestamps, durations, uuids, addresses) a candidate introduces that the
//! original did not carry may not exceed `volatile_token_threshold`. An
//! absent or out-of-range policy rejects (fail to original).
//!
//! # Backoff ladder
//!
//! [`select`] orders candidates aggressive → conservative →
//! structural/lossless → CCR-exact-externalized and returns the first one
//! the gate allows, or the original. JevK5 never participates.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::compression_policy::CompressionPolicy;
use crate::relevance::{BM25Scorer, RelevanceScorer, TaskQuery};
use crate::tokenizer::Tokenizer;

/// Ladder tiers, in backoff order. `Original` is the implicit last rung.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    Aggressive,
    Conservative,
    Lossless,
    CcrExact,
}

impl Tier {
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Aggressive => "aggressive",
            Tier::Conservative => "conservative",
            Tier::Lossless => "lossless",
            Tier::CcrExact => "ccr_exact",
        }
    }

    pub fn parse(s: &str) -> Option<Tier> {
        match s {
            "aggressive" => Some(Tier::Aggressive),
            "conservative" => Some(Tier::Conservative),
            "lossless" | "structural" => Some(Tier::Lossless),
            "ccr_exact" | "ccr" => Some(Tier::CcrExact),
            _ => None,
        }
    }
}

/// The two policy limits the gate enforces.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GatePolicy {
    pub max_lossy_ratio: f64,
    pub volatile_token_threshold: u32,
}

impl From<CompressionPolicy> for GatePolicy {
    fn from(p: CompressionPolicy) -> Self {
        GatePolicy {
            max_lossy_ratio: f64::from(p.max_lossy_ratio),
            volatile_token_threshold: p.volatile_token_threshold,
        }
    }
}

impl GatePolicy {
    fn is_valid(&self) -> bool {
        self.max_lossy_ratio.is_finite() && (0.0..=1.0).contains(&self.max_lossy_ratio)
    }
}

/// How the candidate was produced and what makes it recoverable.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Provenance {
    /// Detected content type (Rust `ContentType::as_str` or the Python
    /// `ContentType` value).
    pub content_type: Option<String>,
    /// Compressor / strategy label, for observability.
    pub strategy: String,
    /// The producer verified the transform is byte-reversible.
    pub reversible: bool,
    /// A retrieval tool (`headroom_retrieve`) is available to the model
    /// for this request: the explicit recoverability contract.
    pub retrieval_available: bool,
    /// Verified CCR entries: hash → exact stored payload.
    pub ccr_payloads: HashMap<String, String>,
    /// The source tool already truncated this output.
    pub source_truncated: bool,
    /// The block sits in the provider-cached prefix.
    pub cache_frozen: bool,
    /// Cache-effective net gain from the caller's net-cost model, if any.
    pub net_cost_gain: Option<f64>,
    /// Caller-side token counts (tokenizer parity with the Python proxy).
    pub original_tokens: Option<usize>,
    pub candidate_tokens: Option<usize>,
}

/// Gate verdict for one candidate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Decision {
    pub allow: bool,
    pub reasons: Vec<String>,
    pub invariant_recall: f64,
    pub facts_total: usize,
    pub facts_preserved: usize,
    pub facts_recoverable: usize,
    pub original_tokens: usize,
    pub candidate_tokens: usize,
    pub wire_drop_ratio: f64,
    pub irreversible_loss_ratio: f64,
    pub volatile_introduced: usize,
    pub exact_recoverable: bool,
}

impl Decision {
    pub fn verdict(&self) -> &'static str {
        if self.allow {
            "ALLOW"
        } else {
            "REJECT"
        }
    }
}

/// One rung offered to [`select`].
#[derive(Debug, Clone)]
pub struct Candidate {
    pub tier: Tier,
    pub text: String,
    pub provenance: Provenance,
}

/// Ladder outcome. `chosen == None` means the original wins.
#[derive(Debug, Clone)]
pub struct Selection {
    pub chosen: Option<usize>,
    pub tier: &'static str,
    /// `(candidate index, decision)` in evaluation order.
    pub decisions: Vec<(usize, Decision)>,
}

// ─── Patterns ─────────────────────────────────────────────────────────

static FAILURE_LINE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(\b(error|errors|exception|traceback|failed|failure|fatal|panic|panicked|segfault|aborted|assertionerror)\b|^E\s{2,}\S)",
    )
    .expect("failure regex")
});
static TEST_COUNT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"\b\d+ (?:passed|failed|skipped|errors?|xfailed|xpassed|deselected|warnings?|tests? (?:passed|failed|ran))\b|\btest result: \w+\. \d+ passed; \d+ failed|\bTests?:\s+\d+ (?:passed|failed|total)[^\n]{0,60}",
    )
    .expect("test count regex")
});
static EXIT_STATUS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)\b(?:exit(?:ed)?(?: with)? (?:code|status)[:= ]\s*-?\d+|returned non-zero exit status -?\d+|exit_code["']?\s*[:=]\s*-?\d+|process finished with exit code -?\d+|status code:? \d{3})"#,
    )
    .expect("exit status regex")
});
static FILE_LINE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?:[\w./\\-]+\.[A-Za-z0-9]{1,6}:\d+(?::\d+)?|File "[^"\n]+", line \d+)"#)
        .expect("file:line regex")
});
static TRUNCATION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(\btruncated\b|<truncated|\[\.\.\.\s*\d+ (?:more )?(?:lines|bytes|chars|characters|items|matches)|\(\d+ (?:more )?(?:lines|bytes|chars) (?:omitted|truncated)\)|\boutput (?:was )?(?:cut|clipped)\b|\bpartial (?:output|result)s?\b)",
    )
    .expect("truncation regex")
});
static CCR_ID: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:<<ccr:|hash=)([a-f0-9]{12,24})\b").expect("ccr id regex"));
static DIFF_HEADER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:diff --git |--- (?:a/|/dev/null)|\+\+\+ (?:b/|/dev/null)|@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@)")
        .expect("diff header regex")
});
static TIMESTAMP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?|\b\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b")
        .expect("timestamp regex")
});
static VOLATILE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?|\b\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b|\b\d+(?:\.\d+)?\s?(?:ms|µs|us|ns|secs?|seconds)\b|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b|\b0x[0-9a-f]{6,}\b|\bpid[=: ]\d+\b",
    )
    .expect("volatile regex")
});
/// A grep/search hit line: `path:line:` with a path-like (dotted or
/// slashed) prefix. Timestamps (`2026-10-14T09:03:53`) do not match.
static SEARCH_HIT_LINE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:[A-Za-z]:)?[^\s:]*[./\\][^\s:]*:\d+[:\-]").expect("search hit regex")
});
/// SmartCrusher compaction block header: `[200]{id:int,name:string}`.
static COMPACTION_HEADER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\[\d+\]\{[^}\n]*\}").expect("compaction header regex"));
static WS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").expect("ws regex"));

const SIGNATURE_KEYWORDS: &str = r"(?:pub(?:\([a-z]+\))?\s+)?(?:async\s+)?(?:def|fn|func|function|class|struct|enum|trait|interface|impl|type)\s+";

// ─── Facts ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum FactKind {
    UserEntity,
    FailureLine,
    TestCount,
    ExitStatus,
    FileLine,
    Signature,
    DiffHeader,
    CcrId,
    Truncation,
}

impl FactKind {
    fn as_str(self) -> &'static str {
        match self {
            FactKind::UserEntity => "user_entity",
            FactKind::FailureLine => "failure_line",
            FactKind::TestCount => "test_count",
            FactKind::ExitStatus => "exit_status",
            FactKind::FileLine => "file_line",
            FactKind::Signature => "signature",
            FactKind::DiffHeader => "diff_header",
            FactKind::CcrId => "ccr_id",
            FactKind::Truncation => "truncation",
        }
    }
}

#[derive(Debug, Clone)]
struct Fact {
    kind: FactKind,
    /// Exact text (token facts) or the trimmed line (line facts).
    text: String,
    /// Line facts compare on a timestamp-stripped, whitespace-collapsed
    /// form so a compressor may dedupe repeats and drop leading clocks.
    line: bool,
    /// Index of the source line in the original (line facts and token
    /// facts both record it so the budget fitter can pin it).
    line_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    Search,
    Json,
    Diff,
    Other,
}

fn shape_of(content_type: Option<&str>, original: &str) -> Shape {
    match content_type.unwrap_or("") {
        "search" | "search_results" => Shape::Search,
        "json_array" | "json" | "json_object" => Shape::Json,
        "diff" | "git_diff" => Shape::Diff,
        _ => {
            let t = original.trim_start();
            if (t.starts_with('{') || t.starts_with('['))
                && serde_json::from_str::<Value>(original.trim()).is_ok()
            {
                Shape::Json
            } else if original.contains("\ndiff --git ") || original.starts_with("diff --git ") {
                Shape::Diff
            } else {
                Shape::Other
            }
        }
    }
}

fn normalize_line(s: &str) -> String {
    let no_ts = TIMESTAMP.replace_all(s, " ");
    WS.replace_all(no_ts.trim(), " ").into_owned()
}

fn is_definition_of(line: &str, name: &str) -> bool {
    // `name` comes from user text; escape before building the pattern.
    let pat = format!(r"^\s*{}{}\b", SIGNATURE_KEYWORDS, regex::escape(name));
    Regex::new(&pat).map(|r| r.is_match(line)).unwrap_or(false)
}

fn extract_facts(
    original: &str,
    task: &TaskQuery,
    shape: Shape,
    source_truncated: bool,
) -> Vec<Fact> {
    let mut facts: Vec<Fact> = Vec::new();
    let mut seen: HashSet<(FactKind, String)> = HashSet::new();
    let mut add = |facts: &mut Vec<Fact>, kind: FactKind, text: String, line: bool, idx: usize| {
        let key_text = if line {
            normalize_line(&text)
        } else {
            text.clone()
        };
        if key_text.is_empty() {
            return;
        }
        if seen.insert((kind, key_text)) {
            facts.push(Fact {
                kind,
                text,
                line,
                line_index: idx,
            });
        }
    };

    let entities: Vec<&str> = task
        .exact_entities
        .iter()
        .map(String::as_str)
        .filter(|e| e.len() >= 2)
        .collect();
    let symbol_entities: Vec<&str> = entities
        .iter()
        .copied()
        .filter(|e| e.chars().all(|c| c.is_alphanumeric() || c == '_'))
        .collect();

    let mut in_traceback = false;
    for (idx, raw_line) in original.lines().enumerate() {
        let line = raw_line.trim_end();
        let trimmed = line.trim();
        if trimmed.is_empty() {
            in_traceback = false;
            continue;
        }
        for e in &entities {
            if line.contains(e) {
                add(
                    &mut facts,
                    FactKind::UserEntity,
                    (*e).to_string(),
                    false,
                    idx,
                );
            }
        }
        for c in CCR_ID.captures_iter(line) {
            add(&mut facts, FactKind::CcrId, c[1].to_string(), false, idx);
        }
        if TRUNCATION.is_match(line) {
            add(
                &mut facts,
                FactKind::Truncation,
                trimmed.to_string(),
                true,
                idx,
            );
        }
        for m in TEST_COUNT.find_iter(line) {
            add(
                &mut facts,
                FactKind::TestCount,
                m.as_str().to_string(),
                false,
                idx,
            );
        }
        for m in EXIT_STATUS.find_iter(line) {
            add(
                &mut facts,
                FactKind::ExitStatus,
                m.as_str().to_string(),
                false,
                idx,
            );
        }
        for s in &symbol_entities {
            if line.contains(s) && is_definition_of(line, s) {
                add(
                    &mut facts,
                    FactKind::Signature,
                    trimmed.to_string(),
                    true,
                    idx,
                );
            }
        }
        match shape {
            Shape::Diff => {
                if DIFF_HEADER.is_match(line) {
                    add(
                        &mut facts,
                        FactKind::DiffHeader,
                        trimmed.to_string(),
                        true,
                        idx,
                    );
                }
            }
            Shape::Search => {
                // A grep for `Error` makes every hit a "failure line", so only
                // lines that are NOT search hits (a mis-detected log, a tool's
                // own error footer) are protected as failures.
                if FAILURE_LINE.is_match(line) && !SEARCH_HIT_LINE.is_match(trimmed) {
                    add(
                        &mut facts,
                        FactKind::FailureLine,
                        trimmed.to_string(),
                        true,
                        idx,
                    );
                }
            }
            Shape::Json => {}
            Shape::Other => {
                if trimmed.starts_with("Traceback") {
                    in_traceback = true;
                }
                let failure = FAILURE_LINE.is_match(line);
                if failure {
                    add(
                        &mut facts,
                        FactKind::FailureLine,
                        trimmed.to_string(),
                        true,
                        idx,
                    );
                }
                if failure || in_traceback {
                    for m in FILE_LINE.find_iter(line) {
                        add(
                            &mut facts,
                            FactKind::FileLine,
                            m.as_str().to_string(),
                            false,
                            idx,
                        );
                    }
                }
                if in_traceback && !raw_line.starts_with(' ') && !trimmed.starts_with("Traceback") {
                    // The exception line closes a Python traceback block.
                    in_traceback = false;
                }
            }
        }
    }
    if source_truncated && !facts.iter().any(|f| f.kind == FactKind::Truncation) {
        add(
            &mut facts,
            FactKind::Truncation,
            "[source truncated]".to_string(),
            true,
            0,
        );
    }
    facts
}

fn json_bool_pairs(v: &Value, out: &mut BTreeSet<(String, bool)>, keys: &mut BTreeSet<String>) {
    match v {
        Value::Object(map) => {
            for (k, val) in map {
                keys.insert(k.clone());
                if let Value::Bool(b) = val {
                    out.insert((k.clone(), *b));
                }
                json_bool_pairs(val, out, keys);
            }
        }
        Value::Array(items) => {
            for item in items {
                json_bool_pairs(item, out, keys);
            }
        }
        _ => {}
    }
}

fn collect_objects<'a>(v: &'a Value, out: &mut Vec<&'a serde_json::Map<String, Value>>) {
    match v {
        Value::Object(map) => {
            out.push(map);
            for val in map.values() {
                collect_objects(val, out);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_objects(item, out);
            }
        }
        _ => {}
    }
}

/// A candidate object whose non-boolean scalars identify one or more
/// original objects must carry the booleans of one of them (no flips).
fn json_bool_flipped(orig: &Value, cand: &Value) -> bool {
    let (mut o, mut c) = (Vec::new(), Vec::new());
    collect_objects(orig, &mut o);
    collect_objects(cand, &mut c);
    if o.len().saturating_mul(c.len()) > 2_000_000 {
        return false;
    }
    for co in &c {
        let bools: Vec<(&String, &Value)> = co.iter().filter(|(_, v)| v.is_boolean()).collect();
        if bools.is_empty() {
            continue;
        }
        let scalars: Vec<(&String, &Value)> = co
            .iter()
            .filter(|(_, v)| !v.is_boolean() && !v.is_object() && !v.is_array())
            .collect();
        if scalars.is_empty() {
            continue;
        }
        let mut identified = false;
        let mut consistent = false;
        for oo in &o {
            if scalars.iter().all(|(k, v)| oo.get(*k) == Some(*v)) {
                identified = true;
                if bools.iter().all(|(k, v)| oo.get(*k) == Some(*v)) {
                    consistent = true;
                    break;
                }
            }
        }
        if identified && !consistent {
            return true;
        }
    }
    false
}

/// Candidate text with retrieval-marker-only lines removed (markers are
/// appended on their own line by every producer).
fn strip_marker_lines(s: &str) -> String {
    s.lines()
        .filter(|l| {
            let t = l.trim();
            !(t.starts_with("<<ccr:") && t.ends_with(">>"))
                && !(t.starts_with('[') && t.contains("hash=") && t.ends_with(']'))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn non_ws_equal(a: &str, b: &str) -> bool {
    a.chars()
        .filter(|c| !c.is_whitespace())
        .eq(b.chars().filter(|c| !c.is_whitespace()))
}

fn volatile_tokens_introduced(original: &str, candidate: &str) -> usize {
    VOLATILE
        .find_iter(candidate)
        .filter(|m| !original.contains(m.as_str()))
        .map(|m| m.as_str().len().div_ceil(4).max(1))
        .sum()
}

// ─── check ────────────────────────────────────────────────────────────

/// Evaluate one candidate. Pure and deterministic for fixed inputs.
pub fn check(
    original: &str,
    candidate: &str,
    task: &TaskQuery,
    policy: Option<&GatePolicy>,
    prov: &Provenance,
    tier: Tier,
    tokenizer: &dyn Tokenizer,
) -> Decision {
    let mut reasons: Vec<String> = Vec::new();
    let mut push = |r: String| {
        if !reasons.contains(&r) {
            reasons.push(r);
        }
    };

    let policy = match policy {
        Some(p) if p.is_valid() => Some(*p),
        _ => {
            push("policy_unknown".to_string());
            None
        }
    };
    if prov.cache_frozen {
        push("cache_prefix_mutation".to_string());
    }
    if let Some(g) = prov.net_cost_gain {
        if !g.is_finite() || g <= 0.0 {
            push("cache_cost_worse".to_string());
        }
    }

    let original_tokens = prov
        .original_tokens
        .unwrap_or_else(|| tokenizer.count_text(original));
    let candidate_tokens = prov
        .candidate_tokens
        .unwrap_or_else(|| tokenizer.count_text(candidate));
    if candidate_tokens >= original_tokens {
        push("no_token_savings".to_string());
    }

    // Retrieval markers the candidate introduces must be verified.
    let original_ids: HashSet<&str> = CCR_ID
        .captures_iter(original)
        .filter_map(|c| c.get(1).map(|m| m.as_str()))
        .collect();
    let mut new_ids: Vec<String> = Vec::new();
    for c in CCR_ID.captures_iter(candidate) {
        let h = &c[1];
        if !original_ids.contains(h) && !new_ids.iter().any(|x| x == h) {
            new_ids.push(h.to_string());
        }
    }
    let mut payloads: Vec<&str> = Vec::new();
    for h in &new_ids {
        match prov.ccr_payloads.get(h) {
            Some(p) => payloads.push(p.as_str()),
            None => push("ccr_marker_invalid".to_string()),
        }
    }
    if !new_ids.is_empty() && !prov.retrieval_available {
        push("ccr_retrieval_unavailable".to_string());
    }
    let recovery_ok = prov.retrieval_available && !payloads.is_empty();
    let exact_recoverable = recovery_ok && payloads.iter().any(|p| *p == original);
    if tier == Tier::CcrExact && !exact_recoverable {
        push("ccr_unavailable".to_string());
    }

    // Facts.
    let shape = shape_of(prov.content_type.as_deref(), original);
    let facts = extract_facts(original, task, shape, prov.source_truncated);
    let cand_norm = normalize_line(candidate);
    let mut preserved = 0usize;
    let mut recoverable = 0usize;
    for f in &facts {
        let present = if f.kind == FactKind::Truncation {
            TRUNCATION.is_match(candidate) || candidate.to_lowercase().contains("partial")
        } else if f.line {
            cand_norm.contains(&normalize_line(&f.text))
        } else {
            candidate.contains(&f.text)
        };
        if present {
            preserved += 1;
            continue;
        }
        let covered = f.kind != FactKind::Truncation
            && recovery_ok
            && payloads.iter().any(|p| p.contains(f.text.as_str()));
        if covered {
            recoverable += 1;
        } else if f.kind == FactKind::Truncation {
            push("partial_misrepresented".to_string());
        } else {
            push(format!("fact_omitted:{}", f.kind.as_str()));
        }
    }

    // Changed facts: summary-shaped facts in the candidate must exist in
    // the original (a compressor may drop, never rewrite, them).
    for re_kind in [
        (&*TEST_COUNT, FactKind::TestCount),
        (&*EXIT_STATUS, FactKind::ExitStatus),
        (&*FILE_LINE, FactKind::FileLine),
    ] {
        // Compressor annotations (`[149 lines omitted: 80 ERROR ...]`) are
        // the producer's own bookkeeping, not restated facts.
        let changed = candidate
            .lines()
            .filter(|l| {
                let t = l.trim();
                !(t.starts_with('[') && t.ends_with(']'))
            })
            .flat_map(|l| re_kind.0.find_iter(l))
            .any(|m| !original.contains(m.as_str()));
        if changed {
            push(format!("fact_changed:{}", re_kind.1.as_str()));
        }
    }

    // JSON invariants.
    let mut json_equal = false;
    // A producer-verified reversible re-encoding (e.g. SmartCrusher's
    // `[N]{schema}` CSV compaction, verified by round-trip) is a different
    // documented format, not broken JSON.
    if shape == Shape::Json && !prov.reversible {
        if let Ok(orig_v) = serde_json::from_str::<Value>(original.trim()) {
            let body = strip_marker_lines(candidate);
            let t = body.trim();
            if (t.starts_with('{') || t.starts_with('[')) && !COMPACTION_HEADER.is_match(t) {
                match serde_json::from_str::<Value>(t) {
                    Ok(cand_v) => {
                        json_equal = cand_v == orig_v;
                        let (mut ob, mut ok) = (BTreeSet::new(), BTreeSet::new());
                        let (mut cb, mut ck) = (BTreeSet::new(), BTreeSet::new());
                        json_bool_pairs(&orig_v, &mut ob, &mut ok);
                        json_bool_pairs(&cand_v, &mut cb, &mut ck);
                        if !cb.is_subset(&ob) || json_bool_flipped(&orig_v, &cand_v) {
                            push("fact_changed:json_bool".to_string());
                        }
                        if ck.iter().any(|k| {
                            !ok.contains(k) && !k.starts_with("_headroom") && !k.starts_with("_ccr")
                        }) {
                            push("fact_changed:json_key".to_string());
                        }
                    }
                    Err(_) => push("syntax_broken:json".to_string()),
                }
            }
        }
    }

    // Diff syntax: a candidate that keeps a hunk body must keep its header.
    if shape == Shape::Diff
        && candidate
            .lines()
            .any(|l| l.starts_with('+') || l.starts_with('-'))
        && original.contains("@@ ")
        && !candidate.contains("@@ ")
        && !exact_recoverable
    {
        push("syntax_broken:diff".to_string());
    }

    // Lossy budget.
    let wire_drop_ratio = if original_tokens == 0 {
        0.0
    } else {
        (original_tokens.saturating_sub(candidate_tokens)) as f64 / original_tokens as f64
    };
    let lossless = prov.reversible
        || json_equal
        || (tier == Tier::Lossless && non_ws_equal(original, candidate));
    let irreversible_loss_ratio = if exact_recoverable || lossless || original_tokens == 0 {
        0.0
    } else {
        let recoverable_tokens: usize = if recovery_ok {
            payloads
                .iter()
                .map(|p| tokenizer.count_text(p))
                .sum::<usize>()
                .min(original_tokens)
        } else {
            0
        };
        let dropped = original_tokens
            .saturating_sub(candidate_tokens)
            .saturating_sub(recoverable_tokens);
        dropped as f64 / original_tokens as f64
    };
    if let Some(p) = policy {
        if irreversible_loss_ratio > p.max_lossy_ratio + 1e-9 {
            push("lossy_budget_exceeded".to_string());
        }
    }

    // Volatile content.
    let volatile_introduced = volatile_tokens_introduced(original, candidate);
    if let Some(p) = policy {
        if volatile_introduced > p.volatile_token_threshold as usize {
            push("volatile_introduced".to_string());
        }
    }

    let total = facts.len();
    let invariant_recall = if total == 0 {
        1.0
    } else {
        (preserved + recoverable) as f64 / total as f64
    };
    Decision {
        allow: reasons.is_empty(),
        reasons,
        invariant_recall,
        facts_total: total,
        facts_preserved: preserved,
        facts_recoverable: recoverable,
        original_tokens,
        candidate_tokens,
        wire_drop_ratio,
        irreversible_loss_ratio,
        volatile_introduced,
        exact_recoverable,
    }
}

/// The central backoff ladder: candidates are evaluated in tier order
/// (stable within a tier) and the first ALLOW wins; otherwise the original.
pub fn select(
    original: &str,
    candidates: &[Candidate],
    task: &TaskQuery,
    policy: Option<&GatePolicy>,
    tokenizer: &dyn Tokenizer,
) -> Selection {
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_by_key(|&i| candidates[i].tier);
    let mut decisions = Vec::with_capacity(order.len());
    // Dominance: a lossy rung that is no smaller than an admissible
    // structural/lossless rung can only add loss, so it is skipped. Lossless
    // rungs are evaluated up front (there are at most a couple).
    let mut lossless_floor: Option<usize> = None;
    let mut lossless_decisions: Vec<(usize, Decision)> = Vec::new();
    for &i in &order {
        let c = &candidates[i];
        if c.tier == Tier::Lossless {
            let d = check(
                original,
                &c.text,
                task,
                policy,
                &c.provenance,
                c.tier,
                tokenizer,
            );
            if d.allow {
                lossless_floor =
                    Some(lossless_floor.map_or(d.candidate_tokens, |f| f.min(d.candidate_tokens)));
            }
            lossless_decisions.push((i, d));
        }
    }
    for i in order {
        let c = &candidates[i];
        let d = if c.tier == Tier::Lossless {
            match lossless_decisions.iter().position(|(j, _)| *j == i) {
                Some(pos) => lossless_decisions.swap_remove(pos).1,
                None => continue,
            }
        } else {
            let mut d = check(
                original,
                &c.text,
                task,
                policy,
                &c.provenance,
                c.tier,
                tokenizer,
            );
            if matches!(c.tier, Tier::Aggressive | Tier::Conservative)
                && d.allow
                && lossless_floor.is_some_and(|floor| d.candidate_tokens >= floor)
            {
                d.allow = false;
                d.reasons.push("dominated_by_lossless".to_string());
            }
            d
        };
        let allow = d.allow;
        decisions.push((i, d));
        if allow {
            return Selection {
                chosen: Some(i),
                tier: c.tier.as_str(),
                decisions,
            };
        }
    }
    Selection {
        chosen: None,
        tier: "original",
        decisions,
    }
}

// ─── Candidate builders shared by both runtimes ───────────────────────

/// Whether line-level budget fitting applies to this content type.
pub fn line_fittable(content_type: &str) -> bool {
    matches!(
        content_type,
        "build" | "build_output" | "search" | "search_results" | "text" | "plain_text" | "log"
    )
}

/// Conservative candidate that honors the lossy budget by construction:
/// keep every protected line and every line the aggressive output kept,
/// then add the most query-relevant remaining lines (BM25 against the task
/// query; head/tail lines break ties) until at least
/// `(1 - max_lossy_ratio)` of the original tokens remain. Gaps become
/// `[... N lines omitted ...]`. `None` when nothing would be dropped.
pub fn fit_lossy_budget(
    original: &str,
    aggressive: Option<&str>,
    task: &TaskQuery,
    policy: &GatePolicy,
    content_type: &str,
    tokenizer: &dyn Tokenizer,
) -> Option<String> {
    if !policy.is_valid() || policy.max_lossy_ratio <= 0.0 {
        return None;
    }
    let lines: Vec<&str> = original.split_inclusive('\n').collect();
    if lines.len() < 8 {
        return None;
    }
    let shape = shape_of(Some(content_type), original);
    let mut keep = vec![false; lines.len()];
    for f in extract_facts(original, task, shape, false) {
        if f.line_index < keep.len() {
            keep[f.line_index] = true;
        }
    }
    if let Some(agg) = aggressive {
        let kept: HashSet<&str> = agg.lines().map(str::trim).filter(|l| l.len() > 3).collect();
        for (i, l) in lines.iter().enumerate() {
            if kept.contains(l.trim()) {
                keep[i] = true;
            }
        }
    }
    let line_tokens: Vec<usize> = lines.iter().map(|l| tokenizer.count_text(l)).collect();
    let total: usize = line_tokens.iter().sum();
    if total == 0 {
        return None;
    }
    let target = ((1.0 - policy.max_lossy_ratio) * total as f64).ceil() as usize + total / 50;
    let mut kept_tokens: usize = keep
        .iter()
        .zip(&line_tokens)
        .filter(|(k, _)| **k)
        .map(|(_, t)| *t)
        .sum();
    if kept_tokens < target {
        let query = task.compressor_query();
        let scores: Vec<f64> = if query.trim().is_empty() {
            vec![0.0; lines.len()]
        } else {
            let items: Vec<&str> = lines.iter().map(|l| l.trim_end()).collect();
            BM25Scorer::default()
                .score_batch(&items, &query)
                .into_iter()
                .map(|s| s.score)
                .collect()
        };
        let n = lines.len();
        let mut rest: Vec<usize> = (0..n).filter(|&i| !keep[i]).collect();
        rest.sort_by(|&a, &b| {
            let edge = |i: usize| if i < 5 || i + 5 >= n { 0.05 } else { 0.0 };
            let sa = scores[a] + edge(a);
            let sb = scores[b] + edge(b);
            sb.partial_cmp(&sa)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        for i in rest {
            if kept_tokens >= target {
                break;
            }
            keep[i] = true;
            kept_tokens += line_tokens[i];
        }
    }
    if keep.iter().all(|k| *k) {
        return None;
    }
    let mut out = String::with_capacity(original.len());
    let mut gap = 0usize;
    for (i, l) in lines.iter().enumerate() {
        if keep[i] {
            if gap > 0 {
                out.push_str(&format!("[... {gap} lines omitted ...]\n"));
                gap = 0;
            }
            out.push_str(l);
        } else {
            gap += 1;
        }
    }
    if gap > 0 {
        out.push_str(&format!("[... {gap} lines omitted ...]\n"));
    }
    Some(out)
}

/// Structural/lossless candidate for JSON: strip insignificant whitespace
/// outside strings. Returned only when it parses to the same value.
pub fn minify_json(original: &str) -> Option<String> {
    let t = original.trim();
    if !(t.starts_with('{') || t.starts_with('[')) {
        return None;
    }
    let mut out = String::with_capacity(t.len());
    let mut in_str = false;
    let mut escaped = false;
    for ch in t.chars() {
        if in_str {
            out.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_str = false;
            }
        } else if ch == '"' {
            in_str = true;
            out.push(ch);
        } else if !ch.is_whitespace() {
            out.push(ch);
        }
    }
    if out.len() >= original.len() {
        return None;
    }
    let a: Value = serde_json::from_str(t).ok()?;
    let b: Value = serde_json::from_str(&out).ok()?;
    (a == b).then_some(out)
}

/// CCR-exact externalization stub: a short head preview plus the marker.
/// The original must already be stored (and verified) under `hash`.
pub fn ccr_exact_stub(original: &str, hash: &str) -> String {
    let total_lines = original.lines().count();
    let mut preview = String::new();
    for l in original.lines().take(6) {
        if preview.len() + l.len() > 480 {
            break;
        }
        preview.push_str(l);
        preview.push('\n');
    }
    format!(
        "{preview}[... original externalized: {total_lines} lines, {} bytes; exact bytes via headroom_retrieve(hash={hash}) ...]\n{}",
        original.len(),
        crate::ccr::marker_for(hash)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::get_tokenizer;

    fn payg() -> GatePolicy {
        GatePolicy {
            max_lossy_ratio: 0.45,
            volatile_token_threshold: 128,
        }
    }

    fn tok() -> Box<dyn Tokenizer> {
        get_tokenizer("gpt-4o")
    }

    fn log_fixture() -> String {
        let mut s = String::new();
        for i in 0..200 {
            s.push_str(&format!("INFO worker {i} processed batch ok\n"));
        }
        s.push_str("ERROR db connection refused at src/db/pool.py:88\n");
        s.push_str("2 failed, 298 passed in 12.3s\n");
        s
    }

    #[test]
    fn omitted_failure_line_rejects_unless_recoverable() {
        let orig = log_fixture();
        let cand = "INFO worker 0 processed batch ok\n2 failed, 298 passed in 12.3s\n";
        let d = check(
            &orig,
            cand,
            &TaskQuery::default(),
            Some(&payg()),
            &Provenance::default(),
            Tier::Aggressive,
            tok().as_ref(),
        );
        assert!(!d.allow);
        assert!(d.reasons.contains(&"fact_omitted:failure_line".to_string()));
        assert!(d.reasons.contains(&"lossy_budget_exceeded".to_string()));

        let hash = crate::ccr::compute_key(orig.as_bytes());
        let mut prov = Provenance {
            retrieval_available: true,
            ..Default::default()
        };
        prov.ccr_payloads.insert(hash.clone(), orig.clone());
        let cand2 = format!("{cand}{}", crate::ccr::marker_for(&hash));
        let d2 = check(
            &orig,
            &cand2,
            &TaskQuery::default(),
            Some(&payg()),
            &prov,
            Tier::Aggressive,
            tok().as_ref(),
        );
        assert!(d2.allow, "{:?}", d2.reasons);
        assert!(d2.exact_recoverable);
        assert_eq!(d2.irreversible_loss_ratio, 0.0);
        assert!(d2.wire_drop_ratio > 0.9);
    }

    #[test]
    fn changed_test_count_and_unverified_marker_reject() {
        let orig = log_fixture();
        let cand = orig.replace("298 passed", "299 passed");
        let d = check(
            &orig,
            &cand,
            &TaskQuery::default(),
            Some(&payg()),
            &Provenance::default(),
            Tier::Aggressive,
            tok().as_ref(),
        );
        assert!(d
            .reasons
            .iter()
            .any(|r| r.starts_with("fact_changed:test_count")
                || r.starts_with("fact_omitted:test_count")));
        let cand2 = format!("{}\n<<ccr:0123456789abcdef01234567>>", &orig[..200]);
        let d2 = check(
            &orig,
            &cand2,
            &TaskQuery::default(),
            Some(&payg()),
            &Provenance {
                retrieval_available: true,
                ..Default::default()
            },
            Tier::Aggressive,
            tok().as_ref(),
        );
        assert!(d2.reasons.contains(&"ccr_marker_invalid".to_string()));
    }

    #[test]
    fn user_entity_and_truncation_are_hard_invariants() {
        let mut orig = String::new();
        for i in 0..100 {
            orig.push_str(&format!("src/m{i}.py:{i}: value = {i}\n"));
        }
        orig.push_str("src/core/budget.py:212:    raise TokenBudgetExceeded(limit)\n[... 400 more lines truncated]\n");
        let q = TaskQuery::new(
            "TokenBudgetExceeded",
            vec!["TokenBudgetExceeded".into()],
            "",
            false,
        );
        let prov = Provenance {
            content_type: Some("search".into()),
            ..Default::default()
        };
        let cand: String = orig.lines().take(50).map(|l| format!("{l}\n")).collect();
        let d = check(
            &orig,
            &cand,
            &q,
            Some(&GatePolicy {
                max_lossy_ratio: 1.0,
                volatile_token_threshold: 128,
            }),
            &prov,
            Tier::Aggressive,
            tok().as_ref(),
        );
        assert!(d.reasons.contains(&"fact_omitted:user_entity".to_string()));
        assert!(d.reasons.contains(&"partial_misrepresented".to_string()));
        assert!(d.invariant_recall < 1.0);
    }

    #[test]
    fn json_bool_flip_and_syntax_break_reject_minify_is_lossless() {
        let orig = "[\n  {\"id\": 1, \"ok\": true},\n  {\"id\": 2, \"ok\": false}\n]";
        let flipped = "[{\"id\":1,\"ok\":false}]";
        let d = check(
            orig,
            flipped,
            &TaskQuery::default(),
            Some(&payg()),
            &Provenance::default(),
            Tier::Aggressive,
            tok().as_ref(),
        );
        assert!(d.reasons.contains(&"fact_changed:json_bool".to_string()));
        let broken = "[{\"id\":1,\"ok\":true}";
        let d = check(
            orig,
            broken,
            &TaskQuery::default(),
            Some(&payg()),
            &Provenance::default(),
            Tier::Aggressive,
            tok().as_ref(),
        );
        assert!(d.reasons.contains(&"syntax_broken:json".to_string()));
        let min = minify_json(orig).unwrap();
        let d = check(
            orig,
            &min,
            &TaskQuery::default(),
            Some(&GatePolicy {
                max_lossy_ratio: 0.0,
                volatile_token_threshold: 0,
            }),
            &Provenance::default(),
            Tier::Lossless,
            tok().as_ref(),
        );
        assert!(d.allow, "{:?}", d.reasons);
        assert_eq!(d.irreversible_loss_ratio, 0.0);
    }

    #[test]
    fn unknown_policy_and_volatile_and_frozen_reject() {
        let orig = log_fixture();
        let cand = format!(
            "{}\nprocessed in 2024-01-01T10:00:00Z 12ms 13ms",
            &orig[orig.len() - 120..]
        );
        let d = check(
            &orig,
            &cand,
            &TaskQuery::default(),
            None,
            &Provenance::default(),
            Tier::Aggressive,
            tok().as_ref(),
        );
        assert!(d.reasons.contains(&"policy_unknown".to_string()));
        let bad = GatePolicy {
            max_lossy_ratio: f64::NAN,
            volatile_token_threshold: 1,
        };
        assert!(
            !check(
                &orig,
                &cand,
                &TaskQuery::default(),
                Some(&bad),
                &Provenance::default(),
                Tier::Aggressive,
                tok().as_ref()
            )
            .allow
        );
        let strict = GatePolicy {
            max_lossy_ratio: 1.0,
            volatile_token_threshold: 1,
        };
        let d = check(
            &orig,
            &cand,
            &TaskQuery::default(),
            Some(&strict),
            &Provenance::default(),
            Tier::Aggressive,
            tok().as_ref(),
        );
        assert!(d.reasons.contains(&"volatile_introduced".to_string()));
        let frozen = Provenance {
            cache_frozen: true,
            ..Default::default()
        };
        assert!(check(
            &orig,
            &cand,
            &TaskQuery::default(),
            Some(&strict),
            &frozen,
            Tier::Aggressive,
            tok().as_ref()
        )
        .reasons
        .contains(&"cache_prefix_mutation".to_string()));
    }

    #[test]
    fn ladder_backs_off_to_policy_fit_then_original() {
        let orig = log_fixture();
        let task = TaskQuery::from_user_text("why is the db pool refusing connections?");
        let aggressive =
            "ERROR db connection refused at src/db/pool.py:88\n2 failed, 298 passed in 12.3s\n"
                .to_string();
        let fit = fit_lossy_budget(
            &orig,
            Some(&aggressive),
            &task,
            &payg(),
            "build",
            tok().as_ref(),
        )
        .unwrap();
        let cands = vec![
            Candidate {
                tier: Tier::Conservative,
                text: fit.clone(),
                provenance: Provenance::default(),
            },
            Candidate {
                tier: Tier::Aggressive,
                text: aggressive,
                provenance: Provenance::default(),
            },
        ];
        let sel = select(&orig, &cands, &task, Some(&payg()), tok().as_ref());
        assert_eq!(sel.tier, "conservative");
        assert_eq!(sel.chosen, Some(0));
        assert_eq!(sel.decisions.len(), 2);
        let d = &sel.decisions[1].1;
        assert!(d.irreversible_loss_ratio <= 0.45 + 1e-9);
        assert!(fit.contains("ERROR db connection refused"));
        // Nothing allowed → original.
        let none = select(&orig, &cands[1..], &task, Some(&payg()), tok().as_ref());
        assert_eq!(none.tier, "original");
        assert!(none.chosen.is_none());
    }

    #[test]
    fn lossy_rung_dominated_by_smaller_lossless_rung_is_skipped() {
        let pad = " ".repeat(40);
        let item = |id: u32, ok: bool| {
            format!(
                "{pad}{{
{pad}{pad}\"id\": {id},
{pad}{pad}\"ok\": {ok}
{pad}}}"
            )
        };
        let orig = format!(
            "[
{},
{},
{}
]",
            item(1, true),
            item(2, false),
            item(3, true)
        );
        let lossy = format!(
            "[
{},
{}
]",
            item(1, true),
            item(2, false)
        );
        let min = minify_json(&orig).unwrap();
        let cands = vec![
            Candidate {
                tier: Tier::Conservative,
                text: lossy,
                provenance: Provenance::default(),
            },
            Candidate {
                tier: Tier::Lossless,
                text: min,
                provenance: Provenance::default(),
            },
        ];
        let loose = GatePolicy {
            max_lossy_ratio: 1.0,
            volatile_token_threshold: 128,
        };
        let sel = select(
            &orig,
            &cands,
            &TaskQuery::default(),
            Some(&loose),
            tok().as_ref(),
        );
        assert_eq!(sel.tier, "lossless");
        let lossy_decision = sel.decisions.iter().find(|(i, _)| *i == 0).unwrap();
        assert!(lossy_decision
            .1
            .reasons
            .contains(&"dominated_by_lossless".to_string()));
    }

    #[test]
    fn search_shape_protects_non_hit_failures_but_not_grep_hits() {
        let loose = GatePolicy {
            max_lossy_ratio: 1.0,
            volatile_token_threshold: 128,
        };
        let prov = Provenance {
            content_type: Some("search".into()),
            ..Default::default()
        };
        let mut hits = String::new();
        for i in 0..60 {
            hits.push_str(&format!(
                "src/m{i}.py:{i}:    raise ValueError(\"bad {i}\")\n"
            ));
        }
        let cand: String = hits.lines().take(10).map(|l| format!("{l}\n")).collect();
        let d = check(
            &hits,
            &cand,
            &TaskQuery::default(),
            Some(&loose),
            &prov,
            Tier::Aggressive,
            tok().as_ref(),
        );
        assert!(d.allow, "grep hits are not failure facts: {:?}", d.reasons);

        // A log mis-detected as search output keeps its error lines.
        let mut log = String::new();
        for i in 0..60 {
            log.push_str(&format!(
                "2026-10-14T09:00:{:02}Z INFO ok 10.0.3.{i}:5432\n",
                i % 60
            ));
        }
        log.push_str("2026-10-14T09:03:53Z ERROR [db.pool] connection refused: 10.0.3.17:5432\n");
        let cand: String = log.lines().take(10).map(|l| format!("{l}\n")).collect();
        let d = check(
            &log,
            &cand,
            &TaskQuery::default(),
            Some(&loose),
            &prov,
            Tier::Aggressive,
            tok().as_ref(),
        );
        assert!(d.reasons.contains(&"fact_omitted:failure_line".to_string()));
    }

    #[test]
    fn compaction_block_is_a_format_not_broken_json() {
        let orig = "[\n  {\"id\": 1, \"ok\": true},\n  {\"id\": 2, \"ok\": false}\n]";
        let cand = "[2]{id:int,ok:bool}\n1,true\n2,false\n";
        let loose = GatePolicy {
            max_lossy_ratio: 1.0,
            volatile_token_threshold: 128,
        };
        let d = check(
            orig,
            cand,
            &TaskQuery::default(),
            Some(&loose),
            &Provenance::default(),
            Tier::Aggressive,
            tok().as_ref(),
        );
        assert!(
            !d.reasons.contains(&"syntax_broken:json".to_string()),
            "{:?}",
            d.reasons
        );
        let rev = Provenance {
            reversible: true,
            ..Default::default()
        };
        let strict = GatePolicy {
            max_lossy_ratio: 0.0,
            volatile_token_threshold: 128,
        };
        let d = check(
            orig,
            cand,
            &TaskQuery::default(),
            Some(&strict),
            &rev,
            Tier::Aggressive,
            tok().as_ref(),
        );
        assert_eq!(d.irreversible_loss_ratio, 0.0);
    }

    #[test]
    fn ccr_exact_stub_requires_exact_payload() {
        let orig = log_fixture();
        let hash = crate::ccr::compute_key(orig.as_bytes());
        let stub = ccr_exact_stub(&orig, &hash);
        let mut prov = Provenance {
            retrieval_available: true,
            ..Default::default()
        };
        let d = check(
            &orig,
            &stub,
            &TaskQuery::default(),
            Some(&payg()),
            &prov,
            Tier::CcrExact,
            tok().as_ref(),
        );
        assert!(!d.allow);
        prov.ccr_payloads.insert(hash, orig.clone());
        let d = check(
            &orig,
            &stub,
            &TaskQuery::default(),
            Some(&payg()),
            &prov,
            Tier::CcrExact,
            tok().as_ref(),
        );
        assert!(d.allow, "{:?}", d.reasons);
    }
}
