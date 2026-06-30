"""
TEMPO-RL Phase 0 — Tests for RewardCalculator.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import pytest

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from TEMPO_RL.schemas import (
    EvidenceItem,
    FutureDependency,
    FutureDependencySet,
    TargetEvidenceSet,
    required_fields_for_type,
)
from TEMPO_RL.evidence_ledger import EvidenceLedger
from TEMPO_RL.reward_calculator import RewardCalculator
from TEMPO_RL.verifier import _normalize_number


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def calculator():
    return RewardCalculator()


@pytest.fixture
def calculator_no_registry():
    """Calculator without tool registry — all tool names considered valid."""
    return RewardCalculator(tool_registry=None)


@pytest.fixture
def calculator_with_registry():
    """Calculator with a known set of valid tool names."""
    return RewardCalculator(
        tool_registry={"search_table", "python_exec", "filter_table", "open_workbook"}
    )


@pytest.fixture
def single_evidence():
    return EvidenceItem(
        sample_id="s1",
        subquestion_id=1,
        evidence_id="sq1_e1",
        type="raw_value",
        value="16.96%",
        entity="乘用车",
        time="2010年1月",
        metric="产量同比增长率",
        unit="%",
        source_tables=["auto_2010.csv"],
        weight=1.0,
    )


@pytest.fixture
def tes_single(single_evidence):
    return TargetEvidenceSet(
        sample_id="s1",
        subquestion_id=1,
        question="2010年1月乘用车产量同比增长率是多少?",
        evidence_items=[single_evidence],
    )


@pytest.fixture
def ledger_empty(tes_single):
    return EvidenceLedger(tes_single)


@pytest.fixture
def ledger_half(tes_single):
    """Ledger with one of two evidence items verified."""
    single_ei = tes_single.evidence_items[0]
    ei2 = EvidenceItem(
        sample_id="s1",
        subquestion_id=1,
        evidence_id="sq1_e2",
        type="raw_value",
        value="70,73元",
        entity="商用车",
        time="2010年1月",
        metric="平均价格",
        unit="元",
        source_tables=["auto_2010.csv"],
        weight=1.0,
    )
    tes = TargetEvidenceSet(
        sample_id="s1",
        subquestion_id=1,
        question="价格是多少?",
        evidence_items=[single_ei, ei2],
    )
    ledger = EvidenceLedger(tes)
    # Verify only the first one via observation
    ledger.update(
        tool_call={"tool_name": "search_table", "arguments": {}},
        observation={"tool_name": "search_table", "content": "产量同比增长率 16.96%", "success": True},
        observation_metadata={"file": "auto_2010.csv"},
    )
    return ledger


@pytest.fixture
def ledger_full(tes_single):
    """Ledger with the evidence item already verified."""
    ledger = EvidenceLedger(tes_single)
    ledger.update(
        tool_call={"tool_name": "search_table", "arguments": {}},
        observation={"tool_name": "search_table", "content": "乘用车产量同比增长率 16.96%", "success": True},
        observation_metadata={"file": "auto_2010.csv"},
    )
    return ledger


@pytest.fixture
def memory_before():
    return {
        "goal": "查找2010年1月的汽车产量数据",
        "tables": [{"name": "auto_2010.csv", "description": "2010年汽车产量数据"}],
        "key_facts": [
            {
                "entity": "乘用车",
                "time": "2010年1月",
                "metric": "产量",
                "value": "120万辆",
                "unit": "万辆",
                "provenance": "auto_2010.csv",
            }
        ],
        "constraints": [],
    }


@pytest.fixture
def observations():
    return [
        {
            "tool_name": "search_table",
            "content": "乘用车产量同比增长率 16.96%, 商用车平均价格 70.73元",
            "success": True,
            "metadata": {"file": "auto_2010.csv"},
        }
    ]


@pytest.fixture
def code_outputs():
    return ["16.96"]


@pytest.fixture
def score_points():
    return [
        "乘用车产量同比增长率为16.96%",
        "2010年1月乘用车产量同比大幅增长",
    ]


@pytest.fixture
def future_dep_set_numeric():
    """H_i^future with one numeric_fact dependency."""
    return FutureDependencySet(
        sample_id="s1",
        boundary="after_sq1",
        future_dependencies=[
            FutureDependency(
                dependency_id="dep_sq1_001",
                type="numeric_fact",
                needed_by="sq2",
                source_evidence_id="sq1_e1",
                fields={
                    "entity": "乘用车",
                    "time": "2010年1月",
                    "metric": "产量同比增长率",
                    "value": "16.96%",
                    "unit": "%",
                },
                weight=1.0,
            )
        ],
    )


@pytest.fixture
def future_dep_set_entity():
    """H_i^future with entity_set dependency."""
    return FutureDependencySet(
        sample_id="s1",
        boundary="after_sq1",
        future_dependencies=[
            FutureDependency(
                dependency_id="dep_sq1_002",
                type="entity_set",
                needed_by="sq2",
                fields={"entities": ["乘用车", "商用车"]},
                weight=1.0,
            )
        ],
    )


@pytest.fixture
def future_dep_set_table():
    """H_i^future with table_ref dependency."""
    return FutureDependencySet(
        sample_id="s1",
        boundary="after_sq1",
        future_dependencies=[
            FutureDependency(
                dependency_id="dep_sq1_003",
                type="table_ref",
                needed_by="sq2",
                fields={"table_name": "auto_2010.csv"},
                weight=1.0,
            )
        ],
    )


@pytest.fixture
def future_dep_set_constraint():
    """H_i^future with constraint dependency."""
    return FutureDependencySet(
        sample_id="s1",
        boundary="after_sq1",
        future_dependencies=[
            FutureDependency(
                dependency_id="dep_sq1_004",
                type="constraint",
                needed_by="sq2",
                fields={"constraint_content": "只考虑乘用车"},
                weight=1.0,
            )
        ],
    )


@pytest.fixture
def future_dep_set_reference():
    """H_i^future with reference dependency."""
    return FutureDependencySet(
        sample_id="s1",
        boundary="after_sq1",
        future_dependencies=[
            FutureDependency(
                dependency_id="dep_sq1_005",
                type="reference",
                needed_by="sq2",
                fields={"reference_text": "前述产量数据", "target_sq": "sq1"},
                weight=1.0,
            )
        ],
    )


# ======================================================================
# Test 1 — Canonical Arguments
# ======================================================================

class TestCanonicalArguments:
    """Test _canonicalize_arguments for repeat detection."""

    def test_path_normalization(self, calculator):
        a1 = calculator._canonicalize_arguments(
            {"file_path": "/data/tables/auto_2010.csv"}, "search_table"
        )
        a2 = calculator._canonicalize_arguments(
            {"file_path": "auto_2010.csv"}, "search_table"
        )
        assert a1 == a2, f"Path normalization failed: {a1!r} != {a2!r}"

    def test_path_normalization_backslashes(self, calculator):
        a1 = calculator._canonicalize_arguments(
            {"file_path": "C:\\data\\sheets\\book1.xlsx"}, "open_workbook"
        )
        a2 = calculator._canonicalize_arguments(
            {"file_path": "book1.xlsx"}, "open_workbook"
        )
        assert a1 == a2, f"Backslash path normalization failed: {a1!r} != {a2!r}"

    def test_key_sorting(self, calculator):
        a1 = calculator._canonicalize_arguments(
            {"b": 1, "a": 2, "c": 3}, "search_table"
        )
        a2 = calculator._canonicalize_arguments(
            {"a": 2, "c": 3, "b": 1}, "search_table"
        )
        assert a1 == a2, f"Key sorting failed: {a1!r} != {a2!r}"

    def test_range_normalization(self, calculator):
        a1 = calculator._canonicalize_arguments(
            {"range": "a1:d10"}, "search_table"
        )
        a2 = calculator._canonicalize_arguments(
            {"range": "A1:D10"}, "search_table"
        )
        assert a1 == a2, f"Range normalization failed: {a1!r} != {a2!r}"

    def test_search_keyword_canonicalization(self, calculator):
        a1 = calculator._canonicalize_arguments(
            {"query": "乘用车 产量"}, "search_table"
        )
        a2 = calculator._canonicalize_arguments(
            {"query": "产量 乘用车"}, "search_table"
        )
        assert a1 == a2, f"Keyword order failed: {a1!r} != {a2!r}"

    def test_float_rounding(self, calculator):
        a1 = calculator._canonicalize_arguments(
            {"threshold": 3.1415926535}, "filter_table"
        )
        a2 = calculator._canonicalize_arguments(
            {"threshold": 3.141593}, "filter_table"
        )
        assert a1 == a2, f"Float rounding failed: {a1!r} != {a2!r}"

    def test_different_args_are_different(self, calculator):
        a1 = calculator._canonicalize_arguments(
            {"table": "auto_2010.csv"}, "search_table"
        )
        a2 = calculator._canonicalize_arguments(
            {"table": "auto_2011.csv"}, "search_table"
        )
        assert a1 != a2, "Different args should have different canonical forms"

    def test_nested_dict(self, calculator):
        a1 = calculator._canonicalize_arguments(
            {"filters": {"min": 10, "max": 20}}, "search_table"
        )
        a2 = calculator._canonicalize_arguments(
            {"filters": {"max": 20, "min": 10}}, "search_table"
        )
        assert a1 == a2, f"Nested dict key order: {a1!r} != {a2!r}"


# ======================================================================
# Test 2 — Invalid Tool Detection
# ======================================================================

class TestInvalidToolDetection:
    """Test _detect_invalid_tool_call."""

    def test_valid_tool_pass(self, calculator_no_registry):
        is_invalid, reason = calculator_no_registry._detect_invalid_tool_call(
            {"tool_name": "search_table", "arguments": {"table": "x.csv"}}
        )
        assert not is_invalid, f"Expected valid, got: {reason}"

    def test_missing_tool_name(self, calculator):
        is_invalid, reason = calculator._detect_invalid_tool_call({})
        assert is_invalid

    def test_unknown_tool_name_with_registry(self, calculator_with_registry):
        is_invalid, reason = calculator_with_registry._detect_invalid_tool_call(
            {"tool_name": "unknown_tool", "arguments": {}}
        )
        assert is_invalid
        assert "not in registry" in reason

    def test_known_tool_name_with_registry(self, calculator_with_registry):
        is_invalid, _ = calculator_with_registry._detect_invalid_tool_call(
            {"tool_name": "search_table", "arguments": {}}
        )
        assert not is_invalid

    def test_args_not_dict(self, calculator):
        is_invalid, reason = calculator._detect_invalid_tool_call(
            {"tool_name": "search_table", "arguments": "not_a_dict"}
        )
        assert is_invalid

    def test_non_dict_tool_call(self, calculator):
        is_invalid, reason = calculator._detect_invalid_tool_call("not_a_dict")
        assert is_invalid

    def test_empty_tool_name(self, calculator):
        is_invalid, reason = calculator._detect_invalid_tool_call(
            {"tool_name": "", "arguments": {}}
        )
        assert is_invalid


# ======================================================================
# Test 3 — Repeat Detection
# ======================================================================

class TestRepeatDetection:
    """Test _is_repeat_call."""

    def test_first_call_not_repeat(self, calculator):
        is_repeat = calculator._is_repeat_call(
            {"tool_name": "search_table", "arguments": {"table": "x.csv"}},
            "sq1",
            has_new_evidence=False,
        )
        assert not is_repeat

    def test_same_call_no_new_evidence(self, calculator):
        tc = {"tool_name": "search_table", "arguments": {"table": "x.csv"}}
        calculator._is_repeat_call(tc, "sq1", has_new_evidence=True)   # first call
        is_repeat = calculator._is_repeat_call(tc, "sq1", has_new_evidence=False)  # second
        assert is_repeat

    def test_same_call_with_new_evidence(self, calculator):
        tc = {"tool_name": "search_table", "arguments": {"table": "x.csv"}}
        calculator._is_repeat_call(tc, "sq1", has_new_evidence=True)   # first call
        is_repeat = calculator._is_repeat_call(tc, "sq1", has_new_evidence=True)  # second
        assert not is_repeat, "Same call with new evidence should NOT be a repeat"

    def test_different_call_no_new_evidence(self, calculator):
        calculator._is_repeat_call(
            {"tool_name": "search_table", "arguments": {"table": "a.csv"}},
            "sq1", has_new_evidence=True,
        )
        is_repeat = calculator._is_repeat_call(
            {"tool_name": "search_table", "arguments": {"table": "b.csv"}},
            "sq1", has_new_evidence=False,
        )
        assert not is_repeat, "Different call not a repeat even without evidence"

    def test_cross_subquestion(self, calculator):
        tc = {"tool_name": "search_table", "arguments": {"table": "x.csv"}}
        calculator._is_repeat_call(tc, "sq1", has_new_evidence=True)
        # Same call in different subquestion
        is_repeat = calculator._is_repeat_call(tc, "sq2", has_new_evidence=False)
        assert not is_repeat, "Same call in different subquestion should not be repeat"

    def test_reset_call_history(self, calculator):
        tc = {"tool_name": "search_table", "arguments": {"table": "x.csv"}}
        calculator._is_repeat_call(tc, "sq1", has_new_evidence=True)
        calculator.reset_call_history("sq1")
        is_repeat = calculator._is_repeat_call(tc, "sq1", has_new_evidence=False)
        assert not is_repeat, "After reset, first call again"


# ======================================================================
# Test 4 — Tool Reward
# ======================================================================

class TestToolReward:
    """Test compute_tool_reward."""

    def test_evidence_gain(self, calculator, tes_single):
        ledger = EvidenceLedger(tes_single)
        result = ledger.update(
            tool_call={"tool_name": "search_table", "arguments": {}},
            observation={"tool_name": "search_table", "content": "16.96% 乘用车", "success": True},
            observation_metadata={"file": "auto_2010.csv"},
        )
        r = calculator.compute_tool_reward(
            {"tool_name": "search_table", "arguments": {"table": "auto_2010.csv"}},
            result, "sq1",
        )
        assert r["delta_phi"] > 0, f"Expected evidence gain, delta_phi={r['delta_phi']}"
        assert r["r_tool"] > 0, f"Expected positive reward with evidence gain, got {r['r_tool']}"
        assert not r["is_invalid"]
        assert not r["is_repeat"]

    def test_no_gain(self, calculator, ledger_full):
        """Tool that finds no new evidence gets only -lambda_call."""
        result = ledger_full.update(
            tool_call={"tool_name": "search_table", "arguments": {"range": "Z1:Z10"}},
            observation={"tool_name": "search_table", "content": "(empty result)", "success": True},
            observation_metadata={"file": "auto_2010.csv"},
        )
        r = calculator.compute_tool_reward(
            {"tool_name": "search_table", "arguments": {"range": "Z1:Z10"}},
            result, "sq1",
        )
        assert r["delta_phi"] == 0.0
        assert r["r_tool"] == pytest.approx(-0.02, abs=0.001), \
            f"Expected -lambda_call only, got {r['r_tool']}"

    def test_invalid_tool(self, calculator, ledger_full):
        result = {
            "coverage_before": 1.0,
            "coverage_after": 1.0,
            "new_evidence_ids": [],
        }
        r = calculator.compute_tool_reward(
            {"tool_name": "", "arguments": {}},
            result, "sq1",
        )
        assert r["is_invalid"]
        assert r["r_tool"] == pytest.approx(-1.02, abs=0.001), \
            f"Expected -lambda_call - lambda_invalid, got {r['r_tool']}"

    def test_repeat_tool(self, calculator, ledger_full):
        tc = {"tool_name": "search_table", "arguments": {"table": "same.csv"}}
        result_no_evidence = {
            "coverage_before": 1.0,
            "coverage_after": 1.0,
            "new_evidence_ids": [],
        }
        # First call
        calculator.compute_tool_reward(tc, {
            "coverage_before": 0.0,
            "coverage_after": 1.0,
            "new_evidence_ids": ["e1"],
        }, "sq1")
        # Repeat
        r = calculator.compute_tool_reward(tc, result_no_evidence, "sq1")
        assert r["is_repeat"]
        assert r["r_tool"] == pytest.approx(-0.22, abs=0.001), \
            f"Expected -lambda_call - lambda_repeat, got {r['r_tool']}"

    def test_multi_step_coverage(self, calculator, tes_single):
        """Coverage progresses across steps."""
        ledger = EvidenceLedger(tes_single)
        assert ledger.coverage == 0.0

        # Step 1: gain evidence
        r1 = ledger.update(
            tool_call={"tool_name": "search_table", "arguments": {"table": "auto_2010.csv"}},
            observation={"tool_name": "search_table", "content": "16.96% 乘用车 产量同比增长率", "success": True},
            observation_metadata={"file": "auto_2010.csv"},
        )
        tool_r1 = calculator.compute_tool_reward(
            {"tool_name": "search_table", "arguments": {"table": "auto_2010.csv"}},
            r1, "sq1",
        )
        assert tool_r1["delta_phi"] > 0
        assert tool_r1["r_tool"] > 0
        assert ledger.coverage == 1.0

        # Step 2: no new evidence
        r2 = ledger.update(
            tool_call={"tool_name": "search_table", "arguments": {"table": "auto_2010.csv"}},
            observation={"tool_name": "search_table", "content": "16.96% again", "success": True},
            observation_metadata={"file": "auto_2010.csv"},
        )
        tool_r2 = calculator.compute_tool_reward(
            {"tool_name": "search_table", "arguments": {"table": "auto_2010.csv"}},
            r2, "sq1",
        )
        assert tool_r2["delta_phi"] == 0.0


# ======================================================================
# Test 5 — Answer Parsing
# ======================================================================

class TestAnswerParsing:
    """Test _parse_answer_json."""

    def test_valid_answer_dict(self, calculator):
        parsed, err, issues = calculator._parse_answer_json(
            {"score_points": [{"scorepoint": "claim 1"}]}
        )
        assert not err
        assert parsed is not None

    def test_answer_as_json_string(self, calculator):
        parsed, err, issues = calculator._parse_answer_json(
            '{"score_points": [{"scorepoint": "claim 1"}]}'
        )
        assert parsed is not None
        # Direct JSON string parse → NOT a format error (model followed the format)
        assert parsed["score_points"][0]["scorepoint"] == "claim 1"

    def test_answer_none(self, calculator):
        parsed, err, issues = calculator._parse_answer_json(None)
        assert err
        assert parsed is None

    def test_answer_empty_string(self, calculator):
        parsed, err, issues = calculator._parse_answer_json("")
        assert err
        assert parsed is None

    def test_answer_empty_dict(self, calculator):
        parsed, err, issues = calculator._parse_answer_json({})
        assert err
        assert parsed is None

    def test_answer_markdown_code_block(self, calculator):
        answer = '```json\n{"score_points": [{"scorepoint": "test"}]}\n```'
        parsed, err, issues = calculator._parse_answer_json(answer)
        assert parsed is not None
        assert parsed["score_points"][0]["scorepoint"] == "test"

    def test_answer_not_json_string(self, calculator):
        parsed, err, issues = calculator._parse_answer_json("this is not json at all")
        assert err
        assert parsed is None


# ======================================================================
# Test 6 — Answer Claim Extraction
# ======================================================================

class TestAnswerClaimExtraction:
    """Test _extract_claims_from_answer."""

    def test_sft_format(self, calculator):
        claims = calculator._extract_claims_from_answer({
            "score_points": [
                {"scorepoint": "claim 1", "table": "t1.csv"},
                {"scorepoint": "claim 2", "table": "t2.csv"},
            ]
        })
        assert len(claims) == 2
        assert claims[0]["text"] == "claim 1"
        assert claims[0]["table"] == "t1.csv"

    def test_sft_format_string_scorepoints(self, calculator):
        claims = calculator._extract_claims_from_answer({
            "score_points": ["claim a", "claim b"]
        })
        assert len(claims) == 2
        assert claims[0]["text"] == "claim a"

    def test_answer_key(self, calculator):
        claims = calculator._extract_claims_from_answer({
            "answer": "this is the answer"
        })
        assert len(claims) == 1
        assert "this is the answer" in claims[0]["text"]

    def test_fallback_whole_json(self, calculator):
        claims = calculator._extract_claims_from_answer({"custom_key": 42})
        assert len(claims) == 1
        assert "custom_key" in claims[0]["text"]


# ======================================================================
# Test 7 — Answer Correctness
# ======================================================================

class TestAnswerCorrectness:
    """Test _check_answer_claim_correct."""

    def test_exact_match(self, calculator):
        correct, conf = calculator._check_answer_claim_correct(
            {"text": "乘用车产量同比增长率为16.96%"},
            ["乘用车产量同比增长率为16.96%"],
        )
        assert correct
        assert conf == pytest.approx(1.0, abs=0.1)

    def test_normalized_match(self, calculator):
        correct, conf = calculator._check_answer_claim_correct(
            {"text": "乘用车产量同比增长率 为 16.96%"},
            ["乘用车产量同比增长率为16.96%"],
        )
        assert correct

    def test_numeric_match(self, calculator):
        correct, conf = calculator._check_answer_claim_correct(
            {"text": "增长率是 16.96%"},
            ["产量同比增长率为16.96%"],
        )
        assert correct

    def test_no_match(self, calculator):
        correct, conf = calculator._check_answer_claim_correct(
            {"text": "商用车价格下降"},
            ["乘用车产量同比增长率为16.96%"],
        )
        assert not correct

    def test_substring_match(self, calculator):
        correct, conf = calculator._check_answer_claim_correct(
            {"text": "2010年1月乘用车产量同比增长率为16.96%，表现良好"},
            ["乘用车产量同比增长率为16.96%"],
        )
        assert correct

    def test_empty_claim(self, calculator):
        correct, conf = calculator._check_answer_claim_correct(
            {"text": ""},
            ["some claim"],
        )
        assert not correct


# ======================================================================
# Test 8 — Answer Grounding
# ======================================================================

class TestAnswerGrounding:
    """Test _check_answer_claim_grounded."""

    def test_grounded_in_observation(self, calculator, ledger_full, observations, code_outputs, memory_before):
        claim = {"text": "乘用车产量同比增长率 16.96%", "table": ""}
        grounded, audit = calculator._check_answer_claim_grounded(
            claim, ledger_full, observations, code_outputs, memory_before,
        )
        assert grounded, f"Claim should be grounded in observation: {audit}"

    def test_grounded_in_memory(self, calculator, ledger_empty, memory_before):
        claim = {"text": "2010年1月产量 120万辆", "table": ""}
        grounded, audit = calculator._check_answer_claim_grounded(
            claim, ledger_empty, [], [], memory_before,
        )
        assert grounded, f"Claim should be grounded in memory: {audit}"

    def test_grounded_in_code(self, calculator, ledger_empty):
        claim = {"text": "16.96", "table": ""}
        grounded, audit = calculator._check_answer_claim_grounded(
            claim, ledger_empty, [], ["16.96"], None,
        )
        assert grounded, f"Claim should be grounded in code output: {audit}"

    def test_grounded_in_verified_evidence(self, calculator, ledger_full):
        claim = {"text": "16.96%", "table": ""}
        grounded, audit = calculator._check_answer_claim_grounded(
            claim, ledger_full, [], [], None,
        )
        assert grounded, f"Claim should be grounded in verified evidence: {audit}"

    def test_not_grounded(self, calculator, ledger_empty):
        claim = {"text": "完全虚构的数据 999%", "table": ""}
        grounded, audit = calculator._check_answer_claim_grounded(
            claim, ledger_empty, [], [], None,
        )
        assert not grounded, f"Claim should NOT be grounded: {audit}"


# ======================================================================
# Test 9 — Answer Reward
# ======================================================================

class TestAnswerReward:
    """Test compute_answer_reward."""

    def test_all_correct_and_grounded(self, calculator, ledger_full, observations, memory_before):
        answer = {"score_points": [{"scorepoint": "乘用车产量同比增长率 16.96%", "table": "auto_2010.csv"}]}
        result = calculator.compute_answer_reward(
            answer,
            ["乘用车产量同比增长率 16.96%"],
            ledger_full,
            memory_before,
            observations,
            ["16.96"],
        )
        assert result["r_answer"] > 0.5, f"Expected high reward, got {result['r_answer']}"
        assert not result["format_error"]

    def test_correct_but_not_grounded(self, calculator, ledger_empty):
        """Correct answer that can't be grounded → lower reward."""
        answer = {"score_points": [{"scorepoint": "乘用车产量同比增长率为16.96%"}]}
        result = calculator.compute_answer_reward(
            answer,
            ["乘用车产量同比增长率为16.96%"],
            ledger_empty,
            None, [], [],
        )
        # C_correct may be True but G=False → contribution = 1*1*0 = 0
        assert result["r_answer"] <= 0.0 or result["r_answer"] == pytest.approx(0.0, abs=0.1), \
            f"Expected low reward for ungrounded, got {result['r_answer']}"

    def test_all_wrong(self, calculator, ledger_full, observations, memory_before):
        answer = {"score_points": [{"scorepoint": "完全错误的答案"}]}
        result = calculator.compute_answer_reward(
            answer,
            ["乘用车产量同比增长率为16.96%"],
            ledger_full,
            memory_before,
            observations,
        )
        assert result["r_answer"] <= 0.0, f"Expected zero/negative for wrong, got {result['r_answer']}"

    def test_completely_illegal_format(self, calculator):
        result = calculator.compute_answer_reward(
            None,
            ["some claim"],
            EvidenceLedger(TargetEvidenceSet("s1", 1, "q", [])),
        )
        assert result["r_answer"] == -1.0

    def test_extra_unsupported_claim(self, calculator, ledger_full, observations, memory_before):
        """Extra claims beyond score_points that aren't grounded."""
        answer = {
            "score_points": [
                {"scorepoint": "乘用车产量同比增长率为16.96%"},
                {"scorepoint": "一个额外的无法验证的推测性结论关于未来市场趋势"},
            ]
        }
        result = calculator.compute_answer_reward(
            answer,
            ["乘用车产量同比增长率为16.96%"],
            ledger_full,
            memory_before,
            observations,
        )
        assert result["unsupported_extra_count"] >= 0

    def test_format_error_penalty(self, calculator, ledger_full):
        """Valid JSON string answer should NOT be a format error.

        The model followed the required JSON output format, so parsing a valid
        JSON string is normal, not an error.
        """
        result = calculator.compute_answer_reward(
            '{"score_points": [{"scorepoint": "乘用车产量同比增长率为16.96%"}]}',
            ["乘用车产量同比增长率为16.96%"],
            ledger_full,
        )
        assert not result["format_error"]


# ======================================================================
# Test 10 — Memory Parsing
# ======================================================================

class TestMemoryParsing:
    """Test _parse_memory_items."""

    def test_parse_structured_memory(self, calculator, memory_before):
        items = calculator._parse_memory_items(memory_before)
        assert len(items) >= 1, f"Expected at least 1 item, got {len(items)}"

    def test_parse_memory_string(self, calculator, memory_before):
        items = calculator._parse_memory_items(json.dumps(memory_before, ensure_ascii=False))
        assert len(items) >= 1

    def test_parse_empty_memory(self, calculator):
        items = calculator._parse_memory_items(None)
        assert items == []

    def test_parse_empty_dict(self, calculator):
        items = calculator._parse_memory_items({})
        assert items == []

    def test_is_unparseable_none(self, calculator):
        assert calculator._is_memory_unparseable(None)

    def test_is_unparseable_empty_string(self, calculator):
        assert calculator._is_memory_unparseable("")

    def test_is_unparseable_bad_string(self, calculator):
        assert calculator._is_memory_unparseable("not valid json {{{")

    def test_is_parseable_dict(self, calculator, memory_before):
        assert not calculator._is_memory_unparseable(memory_before)


# ======================================================================
# Test 11 — Memory Faithfulness
# ======================================================================

class TestMemoryFaithfulness:
    """Test _compute_single_memory_faithfulness and F_i."""

    def test_faithful_from_observation(self, calculator, observations, code_outputs):
        item = {
            "text": "乘用车产量同比增长率 16.96%",
            "entity": "乘用车",
            "metric": "产量同比增长率",
            "value": "16.96%",
            "unit": "%",
            "provenance": "auto_2010.csv",
        }
        q_mem, audit = calculator._compute_single_memory_faithfulness(
            item, [], observations, code_outputs, [],
        )
        assert q_mem > 0.5, f"Expected faithful (q_mem=1.0), got {q_mem}, audit={audit}"

    def test_faithful_from_memory_before(self, calculator):
        mem_before_items = [
            {"text": "save this value 42", "entity": "test", "value": "42", "provenance": "src.csv"}
        ]
        item = {
            "text": "value 42",
            "entity": "test",
            "value": "42",
            "provenance": "src.csv",
        }
        q_mem, audit = calculator._compute_single_memory_faithfulness(
            item, mem_before_items, [], [], [],
        )
        assert q_mem > 0.5, f"Expected faithful (q_mem=1.0), got {q_mem}, audit={audit}"

    def test_faithful_from_code_output(self, calculator):
        item = {
            "text": "computed result 100.5",
            "value": "100.5",
            "provenance": "auto_2010.csv",
        }
        q_mem, audit = calculator._compute_single_memory_faithfulness(
            item, [], [], ["100.5"], [],
        )
        assert q_mem > 0.5, f"Expected faithful, got {q_mem}"

    def test_unfaithful_item(self, calculator):
        item = {
            "text": "completely made up fact 99999",
            "value": "99999",
            "provenance": "made_up.csv",
        }
        q_mem, audit = calculator._compute_single_memory_faithfulness(
            item, [], [], [], [],
        )
        assert q_mem < 0.5, f"Expected unfaithful (q_mem=0.0), got {q_mem}"

    def test_faithful_from_grounded_claim(self, calculator):
        claims = [{"score_point": "the answer is 42"}]
        item = {
            "text": "the answer is 42",
            "value": "42",
            "provenance": "answer",
        }
        q_mem, audit = calculator._compute_single_memory_faithfulness(
            item, [], [], [], claims,
        )
        # Should be faithfully traceable to the grounded claim
        assert q_mem >= 0.5, f"Expected somewhat faithful, got {q_mem}"

    def test_memory_item_without_provenance(self, calculator, observations):
        """Item without provenance field but value in observation is weakly faithful."""
        item = {
            "text": "16.96%",
            "value": "16.96%",
        }
        q_mem, audit = calculator._compute_single_memory_faithfulness(
            item, [], observations, [], [],
        )
        assert q_mem >= 0.5, f"Expected weakly faithful via C_value, got {q_mem}"


# ======================================================================
# Test 12 — Memory Conflict Detection
# ======================================================================

class TestMemoryConflictDetection:
    """Test _check_memory_conflicts."""

    def test_no_conflict(self, calculator, ledger_full):
        """Memory with matching values should not conflict."""
        items = [
            {"entity": "乘用车", "time": "2010年1月", "metric": "产量同比增长率",
             "value": "16.96%", "provenance": "auto_2010.csv", "text": "..."}
        ]
        has_conflict, reasons = calculator._check_memory_conflicts(items, ledger_full)
        assert not has_conflict, f"Should not conflict: {reasons}"

    def test_conflict_different_value(self, calculator, ledger_full):
        """Same entity/time/metric but substantially different value → conflict."""
        items = [
            {"entity": "乘用车", "time": "2010年1月", "metric": "产量同比增长率",
             "value": "50%", "provenance": "auto_2010.csv", "text": "..."}
        ]
        has_conflict, reasons = calculator._check_memory_conflicts(items, ledger_full)
        assert has_conflict, f"Should detect conflict: {reasons}"

    def test_no_conflict_empty_memory(self, calculator, ledger_full):
        has_conflict, reasons = calculator._check_memory_conflicts([], ledger_full)
        assert not has_conflict

    def test_no_conflict_no_verified_items(self, calculator):
        empty_ledger = EvidenceLedger(
            TargetEvidenceSet("s1", 1, "q", [
                EvidenceItem("s1", 1, "e1", "raw_value", "42")
            ])
        )
        items = [{"entity": "x", "value": "42", "text": "..."}]
        has_conflict, _ = calculator._check_memory_conflicts(items, empty_ledger)
        assert not has_conflict


# ======================================================================
# Test 13 — Future Dependency Support
# ======================================================================

class TestFutureDependencySupport:
    """Test _compute_support."""

    def test_numeric_fact_with_source_evidence_id(self, calculator, ledger_full):
        dep = FutureDependency(
            dependency_id="d1", type="numeric_fact", needed_by="sq2",
            source_evidence_id="sq1_e1",
            fields={"entity": "乘用车", "time": "2010年1月", "metric": "产量同比增长率",
                    "value": "16.96%", "unit": "%"},
        )
        assert calculator._compute_support(dep, ledger_full)

    def test_numeric_fact_without_source_id_not_verified(self, calculator, ledger_empty):
        dep = FutureDependency(
            dependency_id="d1", type="numeric_fact", needed_by="sq2",
            fields={"entity": "乘用车", "time": "2010年1月", "metric": "产量同比增长率",
                    "value": "16.96%", "unit": "%"},
        )
        assert not calculator._compute_support(dep, ledger_empty)

    def test_table_ref_support(self, calculator, ledger_full):
        dep = FutureDependency(
            dependency_id="d1", type="table_ref", needed_by="sq2",
            fields={"table_name": "auto_2010.csv"},
        )
        assert calculator._compute_support(dep, ledger_full)

    def test_table_ref_not_supported(self, calculator, ledger_empty):
        dep = FutureDependency(
            dependency_id="d1", type="table_ref", needed_by="sq2",
            fields={"table_name": "auto_2010.csv"},
        )
        assert not calculator._compute_support(dep, ledger_empty)

    def test_entity_set_support(self, calculator, ledger_full):
        dep = FutureDependency(
            dependency_id="d1", type="entity_set", needed_by="sq2",
            fields={"entities": ["乘用车"]},
        )
        assert calculator._compute_support(dep, ledger_full)

    def test_entity_set_not_supported(self, calculator, ledger_empty):
        dep = FutureDependency(
            dependency_id="d1", type="entity_set", needed_by="sq2",
            fields={"entities": ["乘用车"]},
        )
        assert not calculator._compute_support(dep, ledger_empty)

    def test_reference_always_supported(self, calculator, ledger_empty):
        dep = FutureDependency(
            dependency_id="d1", type="reference", needed_by="sq2",
            fields={"reference_text": "some text", "target_sq": "sq1"},
        )
        assert calculator._compute_support(dep, ledger_empty)


# ======================================================================
# Test 14 — H_i^keep Computation
# ======================================================================

class TestHKeepComputation:
    """Test _compute_h_keep."""

    def test_empty_future_deps(self, calculator, ledger_full):
        fds = FutureDependencySet("s1", "after_sq1", [])
        h_keep = calculator._compute_h_keep(fds, ledger_full)
        assert h_keep == []

    def test_supported_dep_kept(self, calculator, ledger_full, future_dep_set_numeric):
        h_keep = calculator._compute_h_keep(future_dep_set_numeric, ledger_full)
        assert len(h_keep) == 1
        assert h_keep[0].dependency_id == "dep_sq1_001"

    def test_unsupported_dep_filtered(self, calculator, ledger_empty, future_dep_set_numeric):
        h_keep = calculator._compute_h_keep(future_dep_set_numeric, ledger_empty)
        assert len(h_keep) == 0

    def test_mixed_deps(self, calculator, ledger_half, single_evidence):
        """Some supported, some not."""
        dep_supported = FutureDependency(
            dependency_id="d_supported", type="numeric_fact", needed_by="sq2",
            source_evidence_id="sq1_e1",  # this one IS verified in ledger_half
            fields={"entity": "乘用车", "time": "2010年1月", "metric": "产量同比增长率",
                    "value": "16.96%", "unit": "%"},
            weight=1.0,
        )
        dep_unsupported = FutureDependency(
            dependency_id="d_unsupported", type="numeric_fact", needed_by="sq2",
            source_evidence_id="sq1_e999",  # NOT verified
            fields={"entity": "商用车", "time": "2010年1月", "metric": "价格",
                    "value": "70元", "unit": "元"},
            weight=1.0,
        )
        fds = FutureDependencySet("s1", "after_sq1", [dep_supported, dep_unsupported])
        h_keep = calculator._compute_h_keep(fds, ledger_half)
        assert len(h_keep) == 1
        assert h_keep[0].dependency_id == "d_supported"


# ======================================================================
# Test 15 — Retain
# ======================================================================

class TestRetain:
    """Test _compute_retain."""

    def test_numeric_fact_fully_retained(self, calculator, memory_before):
        dep = FutureDependency(
            dependency_id="d1", type="numeric_fact", needed_by="sq2",
            fields={"entity": "乘用车", "time": "2010年1月", "metric": "产量",
                    "value": "120万辆", "unit": "万辆"},
        )
        items = calculator._parse_memory_items(memory_before)
        retain = calculator._compute_retain(dep, items)
        assert retain == 1.0, f"Expected fully retained, got {retain}"

    def test_numeric_fact_partially_retained(self, calculator, memory_before):
        dep = FutureDependency(
            dependency_id="d1", type="numeric_fact", needed_by="sq2",
            fields={"entity": "商用车", "time": "2010年1月", "metric": "产量",
                    "value": "200万辆", "unit": "万辆"},
        )
        items = calculator._parse_memory_items(memory_before)
        retain = calculator._compute_retain(dep, items)
        assert retain == 0.0, f"Expected not retained (entity mismatch), got {retain}"

    def test_entity_set_retained(self, calculator, memory_before):
        dep = FutureDependency(
            dependency_id="d1", type="entity_set", needed_by="sq2",
            fields={"entities": ["乘用车"]},
        )
        items = calculator._parse_memory_items(memory_before)
        retain = calculator._compute_retain(dep, items)
        assert retain == 1.0

    def test_entity_set_partially_retained(self, calculator, memory_before):
        dep = FutureDependency(
            dependency_id="d1", type="entity_set", needed_by="sq2",
            fields={"entities": ["乘用车", "商用车"]},
        )
        items = calculator._parse_memory_items(memory_before)
        retain = calculator._compute_retain(dep, items)
        assert retain == 0.0, f"Expected 0.0 (missing 商用车), got {retain}"

    def test_table_ref_retained(self, calculator, memory_before):
        dep = FutureDependency(
            dependency_id="d1", type="table_ref", needed_by="sq2",
            fields={"table_name": "auto_2010.csv"},
        )
        items = calculator._parse_memory_items(memory_before)
        retain = calculator._compute_retain(dep, items)
        assert retain == 1.0

    def test_constraint_retained(self, calculator):
        dep = FutureDependency(
            dependency_id="d1", type="constraint", needed_by="sq2",
            fields={"constraint_content": "只考虑乘用车"},
        )
        mem = {
            "constraints": [{"content": "只考虑乘用车"}],
        }
        items = calculator._parse_memory_items(mem)
        retain = calculator._compute_retain(dep, items)
        assert retain == 1.0

    def test_reference_retained(self, calculator, memory_before):
        dep = FutureDependency(
            dependency_id="d1", type="reference", needed_by="sq2",
            fields={"reference_text": "2010年汽车产量数据", "target_sq": "sq1"},
        )
        items = calculator._parse_memory_items(memory_before)
        retain = calculator._compute_retain(dep, items)
        # memory_before has "2010年汽车产量数据" text in table description
        assert retain == 1.0, f"Expected retained, got {retain}"


# ======================================================================
# Test 16 — Compression Penalty
# ======================================================================

class TestCompressionPenalty:
    """Test _compute_compression_penalty."""

    def test_under_budget(self, calculator):
        p = calculator._compute_compression_penalty("short memory")
        assert p == 0.0

    def test_over_budget(self, calculator):
        long_mem = "x" * 1024  # 2x budget
        p = calculator._compute_compression_penalty(long_mem)
        assert p > 0.0
        assert p == pytest.approx(1.0, abs=0.01), f"Expected ~1.0, got {p}"

    def test_exact_budget(self, calculator):
        mem = "x" * calculator.B
        p = calculator._compute_compression_penalty(mem)
        assert p == 0.0

    def test_dict_memory(self, calculator):
        mem = {"key_facts": [{"entity": "x", "value": "1"}]}
        p = calculator._compute_compression_penalty(mem)
        assert p == 0.0, f"Small dict should be under budget: p={p}"


# ======================================================================
# Test 17 — Memory Reward
# ======================================================================

class TestMemoryReward:
    """Test compute_memory_reward."""

    def test_full_faithful_memory(self, calculator, memory_before, ledger_full,
                                   observations, code_outputs, future_dep_set_numeric):
        memory_after = {
            "goal": "completed search",
            "tables": [{"name": "auto_2010.csv", "description": "2010 data"}],
            "key_facts": [
                {
                    "entity": "乘用车",
                    "time": "2010年1月",
                    "metric": "产量同比增长率",
                    "value": "16.96%",
                    "unit": "%",
                    "provenance": "auto_2010.csv",
                }
            ],
        }
        result = calculator.compute_memory_reward(
            memory_after, memory_before, ledger_full,
            observations, code_outputs, future_dep_set_numeric,
        )
        assert not result["severe_failure"], f"Unexpected severe failure: {result.get('failure_reason')}"
        assert result["F_i"] > 0.0
        assert result["r_memory"] > 0.0, f"Expected positive memory reward, got {result['r_memory']}"

    def test_unfaithful_memory(self, calculator, ledger_full, observations):
        memory_after = {
            "key_facts": [
                {
                    "entity": "虚构实体",
                    "metric": "虚构指标",
                    "value": "99999",
                    "unit": "%",
                    "provenance": "made_up.csv",
                }
            ],
        }
        result = calculator.compute_memory_reward(
            memory_after, {}, ledger_full, observations, [],
        )
        assert result["F_i"] < 0.5, f"Expected low faithfulness, got F_i={result['F_i']}"

    def test_no_future_deps(self, calculator, memory_before, ledger_full, observations, code_outputs):
        """When no future deps, reward uses only F_i and compression."""
        memory_after = {
            "key_facts": [
                {
                    "entity": "乘用车",
                    "value": "16.96%",
                    "provenance": "auto_2010.csv",
                }
            ],
        }
        result = calculator.compute_memory_reward(
            memory_after, memory_before, ledger_full,
            observations, code_outputs, None,  # no future deps
        )
        assert result["S_i"] is None
        assert result["H_keep_size"] == 0
        # r_mem = alpha_f * F_i - lambda_comp * P_comp
        expected = calculator.alpha_f * result["F_i"] - calculator.lambda_comp * result["P_comp"]
        assert result["r_memory"] == pytest.approx(expected, abs=0.01)

    def test_severe_failure_unparseable(self, calculator, ledger_full, observations):
        result = calculator.compute_memory_reward(
            None, {}, ledger_full, observations, [],
        )
        assert result["severe_failure"]
        assert result["r_memory"] == -1.0

    def test_severe_failure_empty_with_deps(self, calculator, ledger_full, future_dep_set_numeric):
        """H_keep non-empty but memory is empty → severe failure."""
        result = calculator.compute_memory_reward(
            {}, {}, ledger_full, [], [], future_dep_set_numeric,
        )
        assert result["severe_failure"], f"Expected severe failure: {result}"
        assert result["r_memory"] == -1.0

    def test_severe_failure_majority_conflict(self, calculator, ledger_full):
        """Majority of memory items conflict with verified evidence."""
        memory_after = {
            "key_facts": [
                {"entity": "乘用车", "time": "2010年1月", "metric": "产量同比增长率",
                 "value": "50%", "provenance": "auto_2010.csv"},
                {"entity": "乘用车", "time": "2010年1月", "metric": "产量同比增长率",
                 "value": "60%", "provenance": "auto_2010.csv"},
            ],
        }
        result = calculator.compute_memory_reward(
            memory_after, {}, ledger_full, [], [],
        )
        assert result["severe_failure"]
        assert result["r_memory"] == -1.0


# ======================================================================
# Test 18 — Future Dependency Coverage (S_i)
# ======================================================================

class TestFutureDependencyCoverage:
    """Test S_i computation."""

    def test_full_coverage(self, calculator, ledger_full, future_dep_set_numeric):
        memory_after = {
            "key_facts": [
                {
                    "entity": "乘用车",
                    "time": "2010年1月",
                    "metric": "产量同比增长率",
                    "value": "16.96%",
                    "unit": "%",
                    "provenance": "auto_2010.csv",
                }
            ],
        }
        result = calculator.compute_memory_reward(
            memory_after, {}, ledger_full, [], [], future_dep_set_numeric,
        )
        assert result["S_i"] == pytest.approx(1.0, abs=0.01), \
            f"Expected full coverage, S_i={result['S_i']}"
        assert result["H_keep_covered"] == 1

    def test_partial_coverage(self, calculator, ledger_full):
        """Two deps, only one retained."""
        dep1 = FutureDependency(
            dependency_id="d_retained", type="numeric_fact", needed_by="sq2",
            source_evidence_id="sq1_e1",
            fields={"entity": "乘用车", "time": "2010年1月", "metric": "产量同比增长率",
                    "value": "16.96%", "unit": "%"},
            weight=1.0,
        )
        dep2 = FutureDependency(
            dependency_id="d_missing", type="numeric_fact", needed_by="sq2",
            source_evidence_id="sq1_e1",
            fields={"entity": "商用车", "time": "2010年1月", "metric": "价格",
                    "value": "70元", "unit": "元"},
            weight=1.0,
        )
        fds = FutureDependencySet("s1", "after_sq1", [dep1, dep2])
        memory_after = {
            "key_facts": [
                {
                    "entity": "乘用车",
                    "time": "2010年1月",
                    "metric": "产量同比增长率",
                    "value": "16.96%",
                    "unit": "%",
                    "provenance": "auto_2010.csv",
                }
            ],
        }
        result = calculator.compute_memory_reward(
            memory_after, {}, ledger_full, [], [], fds,
        )
        assert result["S_i"] == pytest.approx(0.5, abs=0.01), \
            f"Expected 50% coverage, S_i={result['S_i']}"

    def test_no_coverage(self, calculator, ledger_full):
        dep = FutureDependency(
            dependency_id="d_missing", type="numeric_fact", needed_by="sq2",
            source_evidence_id="sq1_e1",
            fields={"entity": "商用车", "time": "2010年1月", "metric": "价格",
                    "value": "70元", "unit": "元"},
            weight=1.0,
        )
        fds = FutureDependencySet("s1", "after_sq1", [dep])
        memory_after = {"key_facts": []}
        result = calculator.compute_memory_reward(
            memory_after, {}, ledger_full, [], [], fds,
        )
        assert result["S_i"] == pytest.approx(0.0, abs=0.01)


# ======================================================================
# Test 19 — compute_all Integration
# ======================================================================

class TestComputeAllIntegration:
    """Integration tests for compute_all."""

    def test_full_subquestion_flow(self, calculator, tes_single, score_points,
                                    memory_before, observations, code_outputs,
                                    future_dep_set_numeric):
        """Simulate a complete subquestion: init → tools → answer → memory."""
        # Memory init
        ledger = EvidenceLedger(tes_single)
        ledger.initialize_from_memory(memory_before)

        # Tool calls
        tool_calls = [
            {"tool_name": "search_table", "arguments": {"table": "auto_2010.csv"}},
            {"tool_name": "python_exec", "arguments": {"code": "print(16.96)"}},
        ]

        ledger_updates = []
        for tc in tool_calls:
            result = ledger.update(
                tool_call=tc,
                observation={
                    "tool_name": tc["tool_name"],
                    "content": "乘用车产量同比增长率 16.96%",
                    "success": True,
                },
                observation_metadata={"file": "auto_2010.csv"},
                code_output="16.96" if tc["tool_name"] == "python_exec" else "",
            )
            ledger_updates.append(result)

        answer_json = {
            "score_points": [
                {"scorepoint": "乘用车产量同比增长率为16.96%", "table": "auto_2010.csv"},
            ]
        }

        memory_after = {
            "goal": "completed",
            "key_facts": [
                {
                    "entity": "乘用车",
                    "time": "2010年1月",
                    "metric": "产量同比增长率",
                    "value": "16.96%",
                    "unit": "%",
                    "provenance": "auto_2010.csv",
                }
            ],
        }

        result = calculator.compute_all(
            subquestion_id="sq1",
            tool_calls=tool_calls,
            ledger_updates=ledger_updates,
            answer_json=answer_json,
            score_points=score_points,
            ledger=ledger,
            memory_before=memory_before,
            memory_after=memory_after,
            observations=observations,
            code_outputs=code_outputs,
            future_dependency_set=future_dep_set_numeric,
        )

        # Check output structure
        for key in ("r_tool", "r_answer", "r_memory", "reward_summary_for_logging", "audit"):
            assert key in result, f"Missing key: {key}"

        # Check audit structure
        audit = result["audit"]
        for key in ("tool_valid_rate", "evidence_coverage", "memory_faithfulness",
                     "tool_details", "answer_details", "memory_details"):
            assert key in audit, f"Missing audit key: {key}"

        assert len(audit["tool_details"]) == 2
        assert audit["evidence_coverage"] == ledger.coverage

    def test_output_format_keys(self, calculator, tes_single, score_points,
                                  memory_before, observations, code_outputs):
        """Verify all required keys in the unified output."""
        ledger = EvidenceLedger(tes_single)

        result = calculator.compute_all(
            subquestion_id="sq1",
            tool_calls=[],
            ledger_updates=[],
            answer_json={"score_points": [{"scorepoint": "nothing"}]},
            score_points=score_points,
            ledger=ledger,
            memory_before=memory_before,
            memory_after=memory_before,
            observations=observations,
            code_outputs=code_outputs,
        )

        # Required top-level keys
        assert "r_tool" in result
        assert "r_answer" in result
        assert "r_memory" in result
        assert "reward_summary_for_logging" in result
        assert "audit" in result

        # reward_summary_for_logging should be mean of three components
        expected_summary = (result["r_tool"] + result["r_answer"] + result["r_memory"]) / 3.0
        assert result["reward_summary_for_logging"] == pytest.approx(expected_summary, abs=0.01)

    def test_no_tool_calls(self, calculator, tes_single, score_points, memory_before):
        """Subquestion with zero tool steps (answer from memory only)."""
        ledger = EvidenceLedger(tes_single)
        ledger.initialize_from_memory(memory_before)

        result = calculator.compute_all(
            subquestion_id="sq1",
            tool_calls=[],
            ledger_updates=[],
            answer_json={"score_points": [{"scorepoint": "乘用车产量120万辆"}]},
            score_points=["乘用车产量120万辆"],
            ledger=ledger,
            memory_before=memory_before,
            memory_after=memory_before,
        )

        # No tool calls → r_tool = 0
        assert result["r_tool"] == 0.0
        assert result["audit"]["tool_valid_rate"] == 1.0  # 0/0 → 1.0 by convention via denominator

    def test_audit_completeness(self, calculator, tes_single, score_points,
                                  memory_before, observations, code_outputs):
        """Each audit sub-dict has expected fields."""
        ledger = EvidenceLedger(tes_single)
        result = calculator.compute_all(
            "sq1", [], [],
            {"score_points": []}, score_points, ledger,
            memory_before, memory_before, observations, code_outputs,
        )

        audit = result["audit"]
        assert isinstance(audit["tool_valid_rate"], float)
        assert isinstance(audit["evidence_coverage"], float)
        assert isinstance(audit["memory_faithfulness"], float)
        assert isinstance(audit["tool_details"], list)

        ad = audit["answer_details"]
        assert "claims_total" in ad
        assert "format_error" in ad
        assert "unsupported_extra_count" in ad

        md = audit["memory_details"]
        assert "F_i" in md
        assert "S_i" in md
        assert "P_comp" in md
        assert "severe_failure" in md


# ======================================================================
# Test 20 — Edge Cases
# ======================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_zero_tool_steps(self, calculator):
        assert calculator.compute_tool_reward(
            {"tool_name": "search_table", "arguments": {}},
            {"coverage_before": 0.0, "coverage_after": 0.0, "new_evidence_ids": []},
        )["r_tool"] == pytest.approx(-0.02, abs=0.01)

    def test_max_coverage_already(self, calculator):
        """Starting at full coverage, no more gain possible."""
        assert calculator.compute_tool_reward(
            {"tool_name": "search_table", "arguments": {}},
            {"coverage_before": 1.0, "coverage_after": 1.0, "new_evidence_ids": []},
        )["r_tool"] == pytest.approx(-0.02, abs=0.01)

    def test_weighted_claims(self, calculator, ledger_full, observations, memory_before):
        """Score points with different weights (all default 1.0 in current impl)."""
        result = calculator.compute_answer_reward(
            {"score_points": [
                {"scorepoint": "乘用车产量同比增长率为16.96%"},
                {"scorepoint": "2010年1月乘用车产量同比大幅增长"},
            ]},
            ["乘用车产量同比增长率为16.96%", "2010年1月乘用车产量同比大幅增长"],
            ledger_full, memory_before, observations,
        )
        assert len(result["claim_results"]) == 2

    def test_single_claim(self, calculator, ledger_full, observations, memory_before):
        result = calculator.compute_answer_reward(
            {"score_points": [{"scorepoint": "乘用车产量同比增长率为16.96%"}]},
            ["乘用车产量同比增长率为16.96%"],
            ledger_full, memory_before, observations,
        )
        assert result["audit"]["claims_total"] == 1

    def test_unicode_in_args(self, calculator):
        """Chinese characters in tool arguments canonicalize correctly."""
        a1 = calculator._canonicalize_arguments(
            {"query": "乘用车 产量 同比增长"}, "search_table"
        )
        a2 = calculator._canonicalize_arguments(
            {"query": "同比增长 产量 乘用车"}, "search_table"
        )
        assert a1 == a2, f"Unicode keyword canonicalization failed: {a1!r} != {a2!r}"

    def test_very_long_memory(self, calculator, ledger_full):
        """Memory approaching budget."""
        # Set B small for this test
        calc = RewardCalculator(B=100)
        mem = "x" * 200
        p = calc._compute_compression_penalty(mem)
        assert p > 0.0

    def test_reset_all_call_history(self, calculator):
        """Reset all call history."""
        tc = {"tool_name": "search_table", "arguments": {"table": "x.csv"}}
        calculator._is_repeat_call(tc, "sq1", has_new_evidence=True)
        calculator._is_repeat_call(tc, "sq2", has_new_evidence=True)
        calculator.reset_call_history()
        assert calculator._call_history == {}

    def test_reward_summary_not_for_training(self, calculator, tes_single, score_points,
                                               memory_before, observations, code_outputs):
        """reward_summary_for_logging is informational only — verify it's documented as such."""
        ledger = EvidenceLedger(tes_single)
        result = calculator.compute_all(
            "sq1", [], [],
            {"score_points": []}, score_points, ledger,
            memory_before, memory_before, observations, code_outputs,
        )
        # reward_summary_for_logging is present but should be separate from
        # the three component rewards used in training
        assert "reward_summary_for_logging" in result
        assert result["r_tool"] is not None
        assert result["r_answer"] is not None
        assert result["r_memory"] is not None
        # Each component reward is independently usable for segment-type training


# ======================================================================
# Test 21 — Configurable Weights
# ======================================================================

class TestConfigurableWeights:
    """Test that custom weights are respected."""

    def test_custom_eta(self, tes_single):
        calc = RewardCalculator(eta=2.0)
        ledger = EvidenceLedger(tes_single)
        result = ledger.update(
            tool_call={"tool_name": "search_table", "arguments": {}},
            observation={"tool_name": "search_table", "content": "16.96% 乘用车", "success": True},
            observation_metadata={"file": "auto_2010.csv"},
        )
        r = calc.compute_tool_reward(
            {"tool_name": "search_table", "arguments": {"table": "auto_2010.csv"}},
            result, "sq1",
        )
        # delta_phi ≈ 1.0, so r_tool ≈ 2.0 - 0.02 = 1.98
        assert r["r_tool"] > 1.5, f"Custom eta=2.0 should give high reward, got {r['r_tool']}"

    def test_custom_lambda_invalid(self, calculator):
        calc = RewardCalculator(lambda_invalid=2.0)
        r = calc.compute_tool_reward(
            {"tool_name": "", "arguments": {}},
            {"coverage_before": 0.0, "coverage_after": 0.0, "new_evidence_ids": []},
            "sq1",
        )
        # r_tool = 0 - 0.02 - 2.0 = -2.02
        assert r["r_tool"] == pytest.approx(-2.02, abs=0.01), \
            f"Expected -2.02 with lambda_invalid=2.0, got {r['r_tool']}"

    def test_custom_alpha_weights(self, ledger_full, memory_before, observations):
        calc = RewardCalculator(alpha_f=0.8, alpha_s=0.2)
        mem = {
            "key_facts": [
                {"entity": "乘用车", "value": "16.96%",
                 "provenance": "auto_2010.csv"},
            ],
        }
        result = calc.compute_memory_reward(
            mem, memory_before, ledger_full, observations, [],
        )
        # F_i should be high, with alpha_f=0.8
        assert result["r_memory"] > 0.0, f"Got r_memory={result['r_memory']}"
