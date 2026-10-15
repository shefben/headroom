//! SafetyGate parity fixtures (unified-21 #7/#13). The same JSON file is
//! asserted by `tests/test_safety_gate.py` through the `headroom._core`
//! bindings, so the Rust gate and the Python facade can never disagree.

use headroom_core::relevance::TaskQuery;
use headroom_core::safety_gate::{check, select, Candidate, GatePolicy, Provenance, Tier};
use headroom_core::tokenizer::get_tokenizer;
use serde_json::Value;

const FIXTURE: &str = include_str!("../../../tests/fixtures/safety_gate_parity.json");

fn policy(v: &Value) -> Option<GatePolicy> {
    if v.is_null() {
        return None;
    }
    Some(GatePolicy {
        max_lossy_ratio: v["max_lossy_ratio"].as_f64().unwrap_or(f64::NAN),
        volatile_token_threshold: v["volatile_token_threshold"].as_u64().unwrap_or(0) as u32,
    })
}

fn task(v: &Value) -> TaskQuery {
    if v.is_null() {
        TaskQuery::default()
    } else {
        TaskQuery::from_json(v).unwrap()
    }
}

fn prov(v: &Value) -> Provenance {
    if v.is_null() {
        Provenance::default()
    } else {
        serde_json::from_value(v.clone()).unwrap()
    }
}

#[test]
fn gate_cases_match_fixture() {
    let root: Value = serde_json::from_str(FIXTURE).unwrap();
    let tok = get_tokenizer(root["model"].as_str().unwrap());
    for case in root["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let p = policy(&case["policy"]);
        let d = check(
            case["original"].as_str().unwrap(),
            case["candidate"].as_str().unwrap(),
            &task(&case["task"]),
            p.as_ref(),
            &prov(&case["provenance"]),
            Tier::parse(case["tier"].as_str().unwrap()).unwrap(),
            tok.as_ref(),
        );
        assert_eq!(
            d.allow,
            case["allow"].as_bool().unwrap(),
            "{name}: {:?}",
            d.reasons
        );
        for r in case["reasons"].as_array().unwrap() {
            let r = r.as_str().unwrap();
            assert!(
                d.reasons.iter().any(|x| x == r),
                "{name}: missing {r} in {:?}",
                d.reasons
            );
        }
        if d.allow {
            assert!(
                (d.invariant_recall - 1.0).abs() < 1e-9,
                "{name}: recall {}",
                d.invariant_recall
            );
        }
    }
}

#[test]
fn ladder_cases_match_fixture() {
    let root: Value = serde_json::from_str(FIXTURE).unwrap();
    let tok = get_tokenizer(root["model"].as_str().unwrap());
    for case in root["ladder"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let cands: Vec<Candidate> = case["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| Candidate {
                tier: Tier::parse(c["tier"].as_str().unwrap()).unwrap(),
                text: c["text"].as_str().unwrap().to_string(),
                provenance: prov(&c["provenance"]),
            })
            .collect();
        let p = policy(&case["policy"]);
        let sel = select(
            case["original"].as_str().unwrap(),
            &cands,
            &TaskQuery::default(),
            p.as_ref(),
            tok.as_ref(),
        );
        assert_eq!(sel.tier, case["tier"].as_str().unwrap(), "{name}");
    }
}
