//! Unified-21 Gate 3 (Rust): query-conditioned live zone (#1), the
//! SafetyGate + lossy policy in the real dispatcher (#7/#13), and cached
//! prefix byte-stability.

mod common;

use common::{body_of, dispatch_without_store, sha256};
use headroom_core::ccr::{CcrStore, InMemoryCcrStore};
use headroom_core::compression_policy::CompressionPolicy;
use headroom_core::relevance::TaskQuery;
use headroom_core::tokenizer::get_tokenizer;
use headroom_core::transforms::live_zone::DEFAULT_MODEL;
use headroom_core::transforms::{
    compress_anthropic_live_zone_with_options, compress_openai_chat_live_zone_with_options,
    compress_openai_responses_live_zone_with_options, AuthMode, BlockAction, ExclusionReason,
    LiveZoneOptions, LiveZoneOutcome,
};
use serde_json::{json, Value};

/// grep-style search output: 300 hits across two subsystems.
fn search_output() -> String {
    let mut s = String::new();
    for i in 0..150 {
        s.push_str(&format!(
            "src/billing/invoice_{i}.py:{}:    total = compute_invoice_total(order_{i})\n",
            10 + i
        ));
        s.push_str(&format!(
            "src/auth/session_{i}.py:{}:    token = refresh_session_token(user_{i})\n",
            20 + i
        ));
    }
    s
}

fn anthropic_body(goal: &str, tool_output: &str) -> Vec<u8> {
    body_of(&json!({
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "system": "you are a coding agent",
        "messages": [
            {"role": "user", "content": goal},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Grep", "input": {"pattern": "def"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": tool_output}
            ]}
        ]
    }))
}

fn compressed_text(out: &LiveZoneOutcome) -> Option<String> {
    match out {
        LiveZoneOutcome::Modified { new_body, .. } => {
            let v: Value = serde_json::from_str(new_body.get()).unwrap();
            Some(
                v["messages"][2]["content"][0]["content"]
                    .as_str()
                    .unwrap()
                    .to_string(),
            )
        }
        LiveZoneOutcome::NoChange { .. } => None,
    }
}

fn run_anthropic(
    body: &[u8],
    store: Option<&dyn CcrStore>,
    query: Option<TaskQuery>,
) -> LiveZoneOutcome {
    compress_anthropic_live_zone_with_options(
        body,
        0,
        AuthMode::Payg,
        DEFAULT_MODEL,
        &LiveZoneOptions {
            query,
            policy: None,
            ccr_store: store,
        },
    )
    .unwrap()
}

#[test]
fn same_tool_output_two_goals_retains_different_rows() {
    let out_text = search_output();
    let store = InMemoryCcrStore::new();
    let billing = compressed_text(&run_anthropic(
        &anthropic_body("why is compute_invoice_total wrong for refunds?", &out_text),
        Some(&store),
        None,
    ))
    .expect("billing goal compresses");
    let auth = compressed_text(&run_anthropic(
        &anthropic_body("why does refresh_session_token expire early?", &out_text),
        Some(&store),
        None,
    ))
    .expect("auth goal compresses");
    assert_ne!(billing, auth, "retained rows must depend on the task");
    let count = |s: &str, needle: &str| s.lines().filter(|l| l.contains(needle)).count();
    assert!(count(&billing, "compute_invoice_total") > count(&billing, "refresh_session_token"));
    assert!(count(&auth, "refresh_session_token") > count(&auth, "compute_invoice_total"));
    // Hard invariant: every user-named entity survives (or is recoverable
    // through the verified marker, which both carry).
    assert!(billing.contains("compute_invoice_total") && billing.contains("<<ccr:"));
    assert!(auth.contains("refresh_session_token") && auth.contains("<<ccr:"));
}

#[test]
fn explicit_query_overrides_body_derived_query() {
    let out_text = search_output();
    let body = anthropic_body("why is compute_invoice_total wrong?", &out_text);
    let store = InMemoryCcrStore::new();
    let q = TaskQuery::new(
        "refresh_session_token",
        vec!["refresh_session_token".into()],
        "",
        true,
    );
    let text = compressed_text(&run_anthropic(&body, Some(&store), Some(q))).unwrap();
    let count = |needle: &str| text.lines().filter(|l| l.contains(needle)).count();
    assert!(count("refresh_session_token") > count("compute_invoice_total"));
}

#[test]
fn without_retrieval_path_irreversible_loss_stays_within_policy() {
    let out_text = search_output();
    let body = anthropic_body("why is compute_invoice_total wrong for refunds?", &out_text);
    let out = dispatch_without_store(&body);
    let tok = get_tokenizer(DEFAULT_MODEL);
    let orig_tokens = tok.count_text(&out_text);
    match &out {
        LiveZoneOutcome::Modified { manifest, .. } => {
            let text = compressed_text(&out).unwrap();
            assert!(
                !text.contains("<<ccr:") && !text.contains("hash="),
                "no unredeemable markers"
            );
            let kept = tok.count_text(&text) as f64;
            let max_lossy = f64::from(
                CompressionPolicy::for_mode(headroom_core::auth_mode::AuthMode::Payg)
                    .max_lossy_ratio,
            );
            assert!(
                kept >= (1.0 - max_lossy) * orig_tokens as f64 - 1.0,
                "loss within PAYG budget"
            );
            assert!(matches!(
                manifest.block_outcomes[0].action,
                BlockAction::Compressed { .. }
            ));
        }
        LiveZoneOutcome::NoChange { manifest } => {
            assert!(matches!(
                manifest.block_outcomes[0].action,
                BlockAction::RejectedBySafetyGate { .. }
            ));
        }
    }
    // Subscription is stricter than PAYG: never keeps fewer tokens.
    let sub = compress_anthropic_live_zone_with_options(
        &body,
        0,
        AuthMode::Subscription,
        DEFAULT_MODEL,
        &LiveZoneOptions::default(),
    )
    .unwrap();
    let kept = |o: &LiveZoneOutcome| {
        compressed_text(o)
            .map(|t| tok.count_text(&t))
            .unwrap_or(orig_tokens)
    };
    assert!(kept(&sub) >= kept(&out));
}

#[test]
fn failure_lines_survive_without_store() {
    let mut log = String::new();
    for i in 0..300 {
        log.push_str(&format!("[INFO] step {i} ok pool=default\n"));
    }
    log.push_str("[ERROR] KeyError: 'tenant_id' in src/app/handlers.py:142\n");
    log.push_str("FAILED tests/test_handlers.py::test_tenant - KeyError\n");
    log.push_str("1 failed, 211 passed in 3.2s\n");
    let body = anthropic_body("fix the failing tenant handler test", &log);
    let out = dispatch_without_store(&body);
    if let Some(text) = compressed_text(&out) {
        assert!(text.contains("KeyError: 'tenant_id' in src/app/handlers.py:142"));
        assert!(text.contains("FAILED tests/test_handlers.py::test_tenant"));
        assert!(text.contains("1 failed, 211 passed"));
    }
}

#[test]
fn frozen_prefix_bytes_identical_and_user_ask_untouched() {
    let out_text = search_output();
    let long_ask = format!(
        "Please investigate the invoice totals carefully. {}",
        "Explain each step of the reasoning in plain prose. ".repeat(200)
    );
    let body = body_of(&json!({
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "earlier turn"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t0", "name": "Grep", "input": {}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t0", "content": out_text},
            ]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Grep", "input": {}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": out_text},
                {"type": "text", "text": long_ask}
            ]}
        ]
    }));
    let store = InMemoryCcrStore::new();
    let out = compress_anthropic_live_zone_with_options(
        &body,
        3,
        AuthMode::Payg,
        DEFAULT_MODEL,
        &LiveZoneOptions {
            ccr_store: Some(&store),
            ..LiveZoneOptions::default()
        },
    )
    .unwrap();
    let LiveZoneOutcome::Modified { new_body, manifest } = out else {
        panic!("live tool_result should compress");
    };
    let new_bytes = new_body.get().as_bytes();
    // Everything up to the live message's first block is byte-identical.
    let live_start = std::str::from_utf8(&body)
        .unwrap()
        .rfind("\"tool_use_id\":\"t1\"")
        .unwrap();
    assert_eq!(
        sha256(&body[..live_start]),
        sha256(&new_bytes[..live_start])
    );
    // The user's own prose ask is excluded, not compressed.
    let ask = manifest
        .block_outcomes
        .iter()
        .find(|b| b.block_type == "text")
        .unwrap();
    assert!(matches!(
        ask.action,
        BlockAction::Excluded {
            reason: ExclusionReason::LatestUserAsk
        }
    ));
    let v: Value = serde_json::from_str(new_body.get()).unwrap();
    assert_eq!(
        v["messages"][4]["content"][1]["text"].as_str().unwrap(),
        long_ask
    );
    // Exact original recoverable from the verified store.
    let marker_hash = headroom_core::ccr::compute_key(out_text.as_bytes());
    assert_eq!(store.get(&marker_hash).as_deref(), Some(out_text.as_str()));
}

#[test]
fn openai_chat_and_responses_are_query_conditioned() {
    let out_text = search_output();
    let chat = |goal: &str| {
        body_of(&json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": goal},
                {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
                    "function": {"name": "grep", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": out_text}
            ]
        }))
    };
    let responses = |goal: &str| {
        body_of(&json!({
            "model": "gpt-5.4",
            "stream": true,
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": goal}]},
                {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": out_text}
            ]
        }))
    };
    let store = InMemoryCcrStore::new();
    let opts = LiveZoneOptions {
        ccr_store: Some(&store),
        ..LiveZoneOptions::default()
    };
    let get_chat = |b: &[u8]| -> String {
        match compress_openai_chat_live_zone_with_options(b, AuthMode::Payg, "gpt-4o", &opts)
            .unwrap()
        {
            LiveZoneOutcome::Modified { new_body, .. } => {
                let v: Value = serde_json::from_str(new_body.get()).unwrap();
                v["messages"][2]["content"].as_str().unwrap().to_string()
            }
            other => panic!("chat should compress: {other:?}"),
        }
    };
    let get_resp = |b: &[u8]| -> String {
        match compress_openai_responses_live_zone_with_options(b, AuthMode::Payg, "gpt-5.4", &opts)
            .unwrap()
        {
            LiveZoneOutcome::Modified { new_body, .. } => {
                let v: Value = serde_json::from_str(new_body.get()).unwrap();
                v["input"][2]["output"].as_str().unwrap().to_string()
            }
            other => panic!("responses should compress: {other:?}"),
        }
    };
    let a = get_chat(&chat("compute_invoice_total is off by one"));
    let b = get_chat(&chat("refresh_session_token never refreshes"));
    assert_ne!(a, b);
    let c = get_resp(&responses("compute_invoice_total is off by one"));
    let d = get_resp(&responses("refresh_session_token never refreshes"));
    assert_ne!(c, d);
    assert!(c.contains("compute_invoice_total") && d.contains("refresh_session_token"));
}

#[test]
fn decision_hints_at_the_ffi_boundary_steer_live_zone_relevance() {
    // A vague goal plus Python DecisionHints naming the auth subsystem: the
    // hint terms reach the dispatcher's ranking query and pull auth rows in,
    // while an otherwise identical query without hints does not.
    let out_text = search_output();
    let body = anthropic_body("what is going on in these results?", &out_text);
    let store = InMemoryCcrStore::new();
    let base = json!({"v": 1, "text": "what is going on in these results?", "exact_entities": []});
    let mut hinted = base.clone();
    hinted["hints"] = json!({
        "v": 1, "family": "relevance_query", "source": "deterministic",
        "query_terms": ["refresh_session_token", "session"],
        "advisory": {}, "choice": null, "confidence": 0.0, "weight": 0.0
    });
    let plain_q = TaskQuery::from_json(&base).unwrap();
    let hinted_q = TaskQuery::from_json(&hinted).unwrap();
    assert!(hinted_q.hints.is_some() && plain_q.hints.is_none());
    let count = |s: &str, needle: &str| s.lines().filter(|l| l.contains(needle)).count();
    let plain = compressed_text(&run_anthropic(&body, Some(&store), Some(plain_q)))
        .unwrap_or_else(|| out_text.clone());
    let steered = compressed_text(&run_anthropic(&body, Some(&store), Some(hinted_q)))
        .expect("hinted query compresses");
    assert!(
        count(&steered, "refresh_session_token") > count(&plain, "refresh_session_token")
            || count(&steered, "refresh_session_token") > count(&steered, "compute_invoice_total"),
        "hint terms must raise the hinted rows"
    );
    // A JevK5-sourced hint is capped and still relevance-only.
    let jev = TaskQuery::from_json(&json!({
        "v": 1, "text": "x", "hints": {"v": 1, "source": "jevk5", "weight": 3.0,
                                       "advisory": {"c0": 0.9, "c1": 0.1}, "choice": "c0",
                                       "confidence": 0.9}
    }))
    .unwrap();
    assert!(jev.hints.unwrap().weight() <= headroom_core::relevance::hints::MAX_ADVISOR_WEIGHT);
}
