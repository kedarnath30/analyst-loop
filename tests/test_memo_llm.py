"""Tests for memo.py and llm.py: provenance and the validation gate."""

import json

from analyst_loop.llm import MockTransport, ValidatedMemoGenerator, validate_draft
from analyst_loop.memo import build_memo, render_markdown
from analyst_loop.pipeline import run_scenario
from analyst_loop.scenarios import get_scenario


def test_memo_claims_carry_provenance():
    result = run_scenario(get_scenario("01_real_lift"))
    memo = result["memo"]
    assert memo["decision"] == "SHIP"
    for claim in memo["claims"]:
        assert claim["metric"] and claim["date_range"] and claim["check_id"]


def test_memo_markdown_contains_decision_and_checks():
    result = run_scenario(get_scenario("01_real_lift"))
    md = render_markdown(result["memo"])
    assert "SHIP" in md and "SHP-01" in md
    assert "2026-08-03" in md  # cited date range


def test_build_memo_round_trip():
    memo = build_memo(
        "99_test", "Test scenario",
        type("D", (), {"label": "HOLD", "reasons": ["r"], "check_ids": ["EFF-02"]})(),
        [{"text": "t", "metric": "m", "date_range": "d", "check_id": "C"}],
        {"n": "1"},
    )
    assert memo["decision"] == "HOLD" and memo["check_ids"] == ["EFF-02"]


def test_llm_valid_draft_passes():
    computed = {"effect": "+5.00", "decision": "SHIP"}
    draft = json.dumps({
        "decision": "SHIP",
        "claims": [{"text": "effect is +5.00", "quote": "+5.00"}],
    })
    out = ValidatedMemoGenerator(MockTransport(draft)).generate(computed)
    assert not out["escalated"] and out["decision"] == "SHIP"


def test_llm_unknown_label_escalates():
    computed = {"effect": "+5.00"}
    draft = json.dumps({"decision": "LAUNCH", "claims": []})
    out = validate_draft(draft, json.dumps(computed, sort_keys=True))
    assert out["escalated"] and out["decision"] == "NEEDS-HUMAN"


def test_llm_nonverbatim_quote_escalates():
    computed = {"effect": "+5.00"}
    draft = json.dumps({
        "decision": "SHIP",
        "claims": [{"text": "effect is +5.01", "quote": "+5.01"}],
    })
    out = validate_draft(draft, json.dumps(computed, sort_keys=True))
    assert out["escalated"] and "verbatim" in out["reason"]


def test_llm_invalid_json_escalates():
    out = validate_draft("not json at all", "{}")
    assert out["escalated"]


def test_llm_no_transport_escalates():
    out = ValidatedMemoGenerator(None).generate({"effect": "+5.00"})
    assert out["escalated"] and "offline" in out["reason"]
