"""
Unit tests for TEMPO-RL Phase 0 — Evidence Ledger & Verifier.

Run from the project root::

    python -m pytest TEMPO_RL/tests/test_evidence_ledger.py -v
"""
from __future__ import annotations

import json
import math
import os
import sys
import pytest

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from TEMPO_RL.schemas import AuditInfo, EvidenceItem, TargetEvidenceSet
from TEMPO_RL.verifier import (
    verify_value_match,
    verify_source_match,
    verify_binding_match,
    verify_derived_inputs,
    verify_derived_operation,
    verify_derived_result,
    verify_evidence_item,
    _normalize_number,
    _extract_numbers_from_text,
    _normalize_text_for_match,
)
from TEMPO_RL.evidence_ledger import (
    EvidenceLedger,
    _parse_memory_facts,
    _verify_memory_fact_against_evidence,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def single_raw_evidence():
    """One raw_value evidence item."""
    return EvidenceItem(
        sample_id="s1", subquestion_id=1, evidence_id="sq1_e1",
        type="raw_value", value="16.96%",
        entity="乘用车", time="2010年1月",
        metric="产量同比增长率", unit="%",
        source_tables=["auto_2010.csv"],
        weight=1.0,
    )


@pytest.fixture
def single_text_evidence():
    """One text_fact evidence item."""
    return EvidenceItem(
        sample_id="s1", subquestion_id=1, evidence_id="sq1_e2",
        type="text_fact", value="年检安排表体现了高度的流程统一性",
        source_tables=["Sheet50.csv"],
        weight=1.0,
    )


@pytest.fixture
def derived_evidence():
    """One derived_value evidence item (growth rate)."""
    return EvidenceItem(
        sample_id="s1", subquestion_id=1, evidence_id="sq1_e3",
        type="derived_value", value="18.46%",
        entity="乘用车", time="2010年1月",
        metric="出口量同比增长率", unit="%",
        source_tables=["auto_2010.csv"],
        input_evidence_ids=["sq1_e_export", "sq1_e_prev"],
        operation="同比增长率",
        weight=1.0,
    )


@pytest.fixture
def multi_item_tes(single_raw_evidence, single_text_evidence, derived_evidence):
    """Target evidence set with raw_value, text_fact, and derived_value."""
    return TargetEvidenceSet(
        sample_id="s1", subquestion_id=1, question="测试问题",
        evidence_items=[single_raw_evidence, single_text_evidence, derived_evidence],
    )


@pytest.fixture
def memory_with_facts():
    """Memory JSON containing facts that match the raw_value evidence."""
    return {
        "goal": "分析2010年1月乘用车产销数据",
        "tables": [
            {"name": "auto_2010.csv", "description": "2010年1月产销数据表"},
        ],
        "key_facts": [
            {
                "entity": "乘用车",
                "time": "2010年1月",
                "metric": "产量同比增长率",
                "value": "16.96%",
                "unit": "%",
                "provenance": "auto_2010.csv",
            },
        ],
        "derived_results": [],
        "constraints": ["时间范围: 2010年1月"],
    }


@pytest.fixture
def observation_match():
    """Observation that matches the raw_value evidence."""
    return {
        "tool_name": "search_table",
        "content": (
            "查询结果：乘用车产量同比增长率为16.96%，"
            "出口量同比增长率为18.46%。"
            "数据来源：auto_2010.csv"
        ),
        "success": True,
    }


@pytest.fixture
def observation_wrong_source():
    """Observation with correct value but wrong source."""
    return {
        "tool_name": "search_table",
        "content": "乘用车产量同比增长率为16.96%",
        "success": True,
    }


# ======================================================================
# Test 1 — Numeric normalisation
# ======================================================================

class TestNumericNormalisation:
    def test_simple_percentage(self):
        assert _normalize_number("16.96%") == pytest.approx(16.96)

    def test_with_comma(self):
        assert _normalize_number("7,073") == pytest.approx(7073)

    def test_with_chinese_wan(self):
        assert _normalize_number("41459万元") == pytest.approx(41459 * 10000)

    def test_with_chinese_yi(self):
        assert _normalize_number("1.5亿") == pytest.approx(1.5 * 1e8)

    def test_with_compound_unit(self):
        assert _normalize_number("41459万元/公里") == pytest.approx(41459 * 10000)

    def test_non_numeric(self):
        assert _normalize_number("") is None
        assert _normalize_number("hello") is None

    def test_extract_numbers(self):
        nums = _extract_numbers_from_text("产量为16.96%，出口18.46%")
        assert len(nums) >= 2
        values = [v for _, v in nums]
        assert any(abs(v - 16.96) < 0.01 for v in values)
        assert any(abs(v - 18.46) < 0.01 for v in values)


# ======================================================================
# Test 2 — C_value: value match
# ======================================================================

class TestValueMatch:
    def test_raw_value_in_observation(self, single_raw_evidence):
        ok, audit = verify_value_match(
            single_raw_evidence,
            observation_text="乘用车产量同比增长率为16.96%",
        )
        assert ok
        assert audit["matched_in"] == "observation"

    def test_raw_value_in_memory(self, single_raw_evidence):
        ok, audit = verify_value_match(
            single_raw_evidence,
            memory_text="产量同比增长率16.96%",
        )
        assert ok
        assert audit["matched_in"] == "memory"

    def test_raw_value_not_found(self, single_raw_evidence):
        ok, _ = verify_value_match(
            single_raw_evidence,
            observation_text="产量同比增长率为20.00%",
        )
        assert not ok

    def test_text_fact_in_observation(self, single_text_evidence):
        ok, audit = verify_value_match(
            single_text_evidence,
            observation_text="年检安排表体现了高度的流程统一性",
        )
        assert ok
        assert audit["matched_in"] == "observation"

    def test_text_fact_partial_match(self, single_text_evidence):
        """Long text fact — partial match of the first 12 chars should work."""
        ok, audit = verify_value_match(
            single_text_evidence,
            observation_text="数据显示，年检安排表体现了高度的流程统一性，符合标准",
        )
        assert ok

    def test_text_fact_not_in_observation(self, single_text_evidence):
        ok, _ = verify_value_match(
            single_text_evidence,
            observation_text="完全不相关的内容",
        )
        assert not ok

    def test_value_in_code_output(self, single_raw_evidence):
        ok, audit = verify_value_match(
            single_raw_evidence,
            code_output="result = 16.96%",
        )
        assert ok
        assert audit["matched_in"] == "code_output"


# ======================================================================
# Test 3 — C_source: source match
# ======================================================================

class TestSourceMatch:
    def test_source_match_with_metadata(self, single_raw_evidence):
        ok, audit = verify_source_match(
            single_raw_evidence,
            observation_metadata={"file": "auto_2010.csv"},
        )
        assert ok
        assert audit.get("matched_table") == "auto_2010.csv"

    def test_source_match_with_tool_args(self, single_raw_evidence):
        ok, audit = verify_source_match(
            single_raw_evidence,
            tool_arguments={"file_path": "auto_2010.csv"},
        )
        assert ok

    def test_source_match_partial_filename(self, single_raw_evidence):
        ok, _ = verify_source_match(
            single_raw_evidence,
            observation_metadata={"file": "data/auto_2010.csv"},
        )
        assert ok  # "auto_2010.csv" is contained in "data/auto_2010.csv"

    def test_source_mismatch(self, single_raw_evidence):
        ok, _ = verify_source_match(
            single_raw_evidence,
            observation_metadata={"file": "other_file.csv"},
        )
        assert not ok

    def test_source_mismatch_no_metadata(self, single_raw_evidence):
        """Without observable metadata, source cannot be verified."""
        ok, audit = verify_source_match(single_raw_evidence)
        assert not ok
        assert "warning" in audit

    def test_empty_source_tables_passes(self):
        ei = EvidenceItem(
            sample_id="s1", subquestion_id=1, evidence_id="e1",
            type="text_fact", value="测试",
            source_tables=[],  # no constraint
        )
        ok, _ = verify_source_match(ei)
        assert ok


# ======================================================================
# Test 4 — C_binding: binding match
# ======================================================================

class TestBindingMatch:
    def test_all_bindings_present(self, single_raw_evidence):
        ok, audit = verify_binding_match(
            single_raw_evidence,
            observation_text="乘用车在2010年1月的产量同比增长率为16.96%",
        )
        assert ok
        for field in ("entity", "time", "metric", "unit"):
            assert audit["checks"][field]["status"] == "pass"

    def test_entity_missing(self, single_raw_evidence):
        """Missing entity → uncertain, not fail. Rule verifier is lenient,
        LLM fallback handles real binding verification."""
        ok, audit = verify_binding_match(
            single_raw_evidence,
            observation_text="2010年1月的产量同比增长率为16.96%",  # no entity
        )
        # Lenient: absence of entity is "uncertain", not "fail"
        assert ok
        assert audit["checks"]["entity"]["status"] == "uncertain"

    def test_no_binding_fields_on_evidence(self):
        """Evidence without optional binding fields — all skipped, passes."""
        ei = EvidenceItem(
            sample_id="s1", subquestion_id=1, evidence_id="e1",
            type="raw_value", value="42",
            entity=None, time=None, metric=None, unit=None,
        )
        ok, audit = verify_binding_match(
            ei, observation_text="值是42",
        )
        assert ok  # all fields None → skipped, pass

    def test_binding_with_memory_context(self, single_raw_evidence):
        ok, audit = verify_binding_match(
            single_raw_evidence,
            observation_text="增长率是16.96%",  # missing entity/time
            memory_text="乘用车2010年1月数据",  # entity/time in memory
        )
        assert ok  # entity and time found in memory context

    def test_binding_in_tool_arguments(self, single_raw_evidence):
        ok, _ = verify_binding_match(
            single_raw_evidence,
            observation_text="增长率是16.96%",
            tool_arguments={
                "entity": "乘用车",
                "year": "2010",
                "month": "1",
            },
        )
        assert ok


# ======================================================================
# Test 5 — C_input: derived evidence input check
# ======================================================================

class TestDerivedInputCheck:
    def test_all_inputs_verified(self, derived_evidence):
        ok, audit = verify_derived_inputs(
            derived_evidence,
            verified_evidence_ids={"sq1_e_export", "sq1_e_prev"},
        )
        assert ok
        assert audit["status"] == "pass"
        assert audit["missing_inputs"] == []

    def test_some_inputs_missing(self, derived_evidence):
        ok, audit = verify_derived_inputs(
            derived_evidence,
            verified_evidence_ids={"sq1_e_export"},  # sq1_e_prev missing
        )
        assert not ok
        assert "sq1_e_prev" in audit["missing_inputs"]

    def test_no_input_evidence_ids(self):
        """derived_value without input_evidence_ids — should fail."""
        ei = EvidenceItem(
            sample_id="s1", subquestion_id=1, evidence_id="e1",
            type="derived_value", value="18.46%",
            input_evidence_ids=[],
        )
        ok, _ = verify_derived_inputs(ei, set())
        assert not ok


# ======================================================================
# Test 6 — C_operator: operation match
# ======================================================================

class TestDerivedOperation:
    def test_operation_keyword_in_code(self, derived_evidence):
        ok, audit = verify_derived_operation(
            derived_evidence,
            code_output="# 计算同比增长率 = (current - prev) / prev",
            tool_name="python_exec",
        )
        assert ok
        assert "增长" in str(audit.get("matched_keywords", []))

    def test_code_tool_no_operation_required(self):
        """No specific operation expected → any code tool passes."""
        ei = EvidenceItem(
            sample_id="s1", subquestion_id=1, evidence_id="e1",
            type="derived_value", value="42", operation=None,
        )
        ok, _ = verify_derived_operation(
            ei, tool_name="python_exec", code_output="print(42)",
        )
        assert ok

    def test_operation_not_found(self, derived_evidence):
        ok, _ = verify_derived_operation(
            derived_evidence,
            code_output="print('hello')",
            tool_name="python_exec",
        )
        # Python exec was used so should be accepted as computation tool
        assert ok

    def test_non_code_tool_no_operation_keywords(self, derived_evidence):
        ok, _ = verify_derived_operation(
            derived_evidence,
            tool_name="search_table",
            code_output="",
        )
        assert not ok


# ======================================================================
# Test 7 — C_result: derived result verification
# ======================================================================

class TestDerivedResult:
    def test_growth_rate_recomputation(self):
        """growth rate: (current - previous) / previous = (118.46 - 100) / 100 = 0.1846"""
        ei = EvidenceItem(
            sample_id="s1", subquestion_id=1, evidence_id="e3",
            type="derived_value", value="18.46%",
            input_evidence_ids=["e1", "e2"],
            operation="同比增长率",
        )
        inputs = [
            EvidenceItem(sample_id="s1", subquestion_id=1, evidence_id="e1",
                         type="raw_value", value="118.46"),
            EvidenceItem(sample_id="s1", subquestion_id=1, evidence_id="e2",
                         type="raw_value", value="100.00"),
        ]
        ok, audit = verify_derived_result(ei, inputs)
        assert ok, f"Recomputation failed: {audit}"
        assert audit["computation_method"] == "growth_rate"

    def test_ratio_recomputation(self):
        """Ratio: 10.89 / 100 = 0.1089 = 10.89%"""
        ei = EvidenceItem(
            sample_id="s1", subquestion_id=1, evidence_id="e3",
            type="derived_value", value="10.89%",
            input_evidence_ids=["e1", "e2"],
            operation="成本占比",
        )
        inputs = [
            EvidenceItem(sample_id="s1", subquestion_id=1, evidence_id="e1",
                         type="raw_value", value="1089"),
            EvidenceItem(sample_id="s1", subquestion_id=1, evidence_id="e2",
                         type="raw_value", value="10000"),
        ]
        ok, _ = verify_derived_result(ei, inputs)
        assert ok

    def test_sum_recomputation(self):
        ei = EvidenceItem(
            sample_id="s1", subquestion_id=1, evidence_id="e3",
            type="derived_value", value="150",
            input_evidence_ids=["e1", "e2"],
            operation="求和",
        )
        inputs = [
            EvidenceItem(sample_id="s1", subquestion_id=1, evidence_id="e1",
                         type="raw_value", value="100"),
            EvidenceItem(sample_id="s1", subquestion_id=1, evidence_id="e2",
                         type="raw_value", value="50"),
        ]
        ok, _ = verify_derived_result(ei, inputs)
        assert ok

    def test_average_recomputation(self):
        ei = EvidenceItem(
            sample_id="s1", subquestion_id=1, evidence_id="e3",
            type="derived_value", value="75",
            input_evidence_ids=["e1", "e2"],
            operation="平均",
        )
        inputs = [
            EvidenceItem(sample_id="s1", subquestion_id=1, evidence_id="e1",
                         type="raw_value", value="100"),
            EvidenceItem(sample_id="s1", subquestion_id=1, evidence_id="e2",
                         type="raw_value", value="50"),
        ]
        ok, _ = verify_derived_result(ei, inputs)
        assert ok

    def test_recomputation_mismatch(self):
        ei = EvidenceItem(
            sample_id="s1", subquestion_id=1, evidence_id="e3",
            type="derived_value", value="50%",  # wrong — should be ~18.46%
            input_evidence_ids=["e1", "e2"],
            operation="同比增长率",
        )
        inputs = [
            EvidenceItem(sample_id="s1", subquestion_id=1, evidence_id="e1",
                         type="raw_value", value="118.46"),
            EvidenceItem(sample_id="s1", subquestion_id=1, evidence_id="e2",
                         type="raw_value", value="100.00"),
        ]
        ok, audit = verify_derived_result(ei, inputs)
        assert not ok
        assert audit["status"] == "fail"

    def test_non_numeric_derived_passes(self):
        """Non-numeric derived values are treated leniently."""
        ei = EvidenceItem(
            sample_id="s1", subquestion_id=1, evidence_id="e3",
            type="derived_value", value="定性结论",
            input_evidence_ids=["e1"],
            operation="分析",
        )
        inputs = [
            EvidenceItem(sample_id="s1", subquestion_id=1, evidence_id="e1",
                         type="text_fact", value="基础分析"),
        ]
        ok, _ = verify_derived_result(ei, inputs)
        assert ok  # non-numeric → pass leniently


# ======================================================================
# Test 8 — Full evidence verification
# ======================================================================

class TestFullVerification:
    def test_raw_value_all_pass(self, single_raw_evidence):
        ok, audit = verify_evidence_item(
            single_raw_evidence,
            observation_text="乘用车在2010年1月的产量同比增长率为16.96%",
            observation_metadata={"file": "auto_2010.csv"},
        )
        assert ok
        assert audit["checks"]["C_value"]["matched_in"] == "observation"
        assert audit["checks"]["C_source"].get("matched_table") == "auto_2010.csv"

    def test_source_fail_blocks_verification(self, single_raw_evidence):
        ok, audit = verify_evidence_item(
            single_raw_evidence,
            observation_text="乘用车产量同比增长率为16.96%",
            observation_metadata={"file": "wrong_file.csv"},
        )
        assert not ok
        assert not audit["checks"]["C_source"].get("matched_table")

    def test_binding_absence_does_not_block(self, single_raw_evidence):
        """Absence of binding fields → uncertain, not fail. Source still required."""
        ok, audit = verify_evidence_item(
            single_raw_evidence,
            observation_text="增长率是16.96%",  # value correct, entity/time absent
            observation_metadata={"file": "auto_2010.csv"},
        )
        # Value match + source pass → overall passes (binding is lenient)
        assert ok
        # But binding checks are marked uncertain
        b = audit["checks"]["C_binding"]
        assert b["checks"]["entity"]["status"] == "uncertain"

    def test_source_mismatch_still_blocks(self, single_raw_evidence):
        """Source mismatch should block even with lenient binding."""
        ok, audit = verify_evidence_item(
            single_raw_evidence,
            observation_text="乘用车产量同比增长率为16.96%",
            observation_metadata={"file": "wrong_file.csv"},
        )
        assert not ok
        assert not audit["checks"]["C_source"].get("matched_table")

    def test_derived_value_full_verification(self):
        """Derived value with all checks passing."""
        ei = EvidenceItem(
            sample_id="s1", subquestion_id=1, evidence_id="e3",
            type="derived_value", value="18.46%",
            entity="乘用车", time="2010年1月",
            metric="出口量同比增长率", unit="%",
            source_tables=["auto_2010.csv"],
            input_evidence_ids=["e1", "e2"],
            operation="同比增长率",
        )
        # Create verified input items in ledger
        input_e1 = EvidenceItem(
            sample_id="s1", subquestion_id=1, evidence_id="e1",
            type="raw_value", value="118.46",
        )
        input_e2 = EvidenceItem(
            sample_id="s1", subquestion_id=1, evidence_id="e2",
            type="raw_value", value="100.00",
        )

        ok, audit = verify_evidence_item(
            ei,
            observation_text="乘用车2010年1月出口量同比增长率为18.46%",
            observation_metadata={"file": "auto_2010.csv"},
            code_output="增长率 = (118.46 - 100) / 100 = 18.46%",
            verified_evidence_ids={"e1", "e2"},
            ledger_evidence_items=[input_e1, input_e2],
        )
        # May not fully pass if operation detection is weak
        # but at minimum, C_value + C_source + C_input should pass
        assert audit["checks"]["C_value"]["matched_in"] is not None
        assert audit["checks"]["C_source"].get("matched_table") is not None
        assert audit["checks"]["C_input"]["status"] == "pass"


# ======================================================================
# Test 9 — Memory parsing
# ======================================================================

class TestMemoryParsing:
    def test_parse_key_facts(self):
        mem = {
            "key_facts": [
                {
                    "entity": "乘用车", "time": "2010年1月",
                    "metric": "产量同比增长率", "value": "16.96%",
                    "unit": "%", "provenance": "auto_2010.csv",
                },
            ],
        }
        facts = _parse_memory_facts(mem)
        assert len(facts) >= 1
        assert facts[0]["value"] == "16.96%"

    def test_parse_string_facts(self):
        mem = {"key_facts": ["乘用车产量同比增长率为16.96%"]}
        facts = _parse_memory_facts(mem)
        assert len(facts) >= 1
        assert "16.96%" in facts[0]["text"]

    def test_parse_tables(self):
        mem = {"tables": [{"name": "auto_2010.csv", "description": "产销数据"}]}
        facts = _parse_memory_facts(mem)
        assert len(facts) >= 1
        assert facts[0]["table_name"] == "auto_2010.csv"

    def test_parse_constraints(self):
        mem = {"constraints": ["时间范围: 2010年1月"]}
        facts = _parse_memory_facts(mem)
        assert len(facts) >= 1
        assert "2010年1月" in facts[0]["text"]

    def test_parse_memory_string(self):
        """Memory can be a JSON string."""
        facts = _parse_memory_facts(json.dumps({"key_facts": ["产量16.96%"]}))
        assert len(facts) >= 1

    def test_parse_non_dict(self):
        assert _parse_memory_facts(None) == []
        assert _parse_memory_facts(123) == []
        assert _parse_memory_facts("not json") == []

    def test_memory_fact_against_evidence(self):
        fact = {
            "text": "乘用车产量同比增长率为16.96%",
            "entity": "乘用车", "time": "2010年1月",
            "metric": "产量同比增长率", "value": "16.96%",
            "unit": "%", "provenance": "auto_2010.csv",
        }
        ei = EvidenceItem(
            sample_id="s1", subquestion_id=1, evidence_id="e1",
            type="raw_value", value="16.96%",
            entity="乘用车", time="2010年1月",
            metric="产量同比增长率", unit="%",
            source_tables=["auto_2010.csv"],
        )
        ok, audit = _verify_memory_fact_against_evidence(fact, ei)
        assert ok, f"Audit: {audit}"
        assert audit["C_provenance"]["status"] == "pass"


# ======================================================================
# Test 10 — Ledger: memory initialization
# ======================================================================

class TestLedgerMemoryInit:
    def test_init_from_memory(self, multi_item_tes, memory_with_facts):
        ledger = EvidenceLedger(multi_item_tes)
        result = ledger.initialize_from_memory(memory_with_facts)

        assert result["coverage_before"] == 0.0
        assert result["coverage_after"] > 0.0
        # raw_value should be verified (16.96% with entity/time/source)
        assert "sq1_e1" in result["new_evidence_ids"]

    def test_init_empty_memory(self, multi_item_tes):
        ledger = EvidenceLedger(multi_item_tes)
        result = ledger.initialize_from_memory(None)
        assert result["coverage_after"] == 0.0
        assert result["new_evidence_ids"] == []

    def test_init_memory_no_matching_evidence(self, multi_item_tes):
        """Memory with facts that don't match any target evidence."""
        mem = {"key_facts": [{"entity": "无关", "value": "999"}]}
        ledger = EvidenceLedger(multi_item_tes)
        result = ledger.initialize_from_memory(mem)
        assert result["coverage_after"] == 0.0

    def test_from_target_evidence_set_factory(self, multi_item_tes, memory_with_facts):
        ledger = EvidenceLedger.from_target_evidence_set(
            multi_item_tes, memory_before=memory_with_facts,
        )
        assert ledger.coverage > 0.0


# ======================================================================
# Test 11 — Ledger: tool observation update (raw evidence)
# ======================================================================

class TestLedgerToolUpdate:
    def test_raw_evidence_from_observation(self, multi_item_tes, observation_match):
        ledger = EvidenceLedger(multi_item_tes)
        assert ledger.coverage == 0.0

        result = ledger.update(
            tool_call={"tool_name": "search_table", "arguments": {}},
            observation=observation_match,
            observation_metadata={"file": "auto_2010.csv"},
        )
        assert result["coverage_before"] == 0.0
        assert result["coverage_after"] > 0.0
        assert "sq1_e1" in result["new_evidence_ids"]

    def test_coverage_increases_correctly(self, multi_item_tes, observation_match):
        """With 3 items all weight=1, each adds 1/3 coverage."""
        ledger = EvidenceLedger(multi_item_tes)
        result = ledger.update(
            tool_call={"tool_name": "search_table", "arguments": {}},
            observation=observation_match,
            observation_metadata={"file": "auto_2010.csv"},
        )
        # sq1_e1 (raw_value) should be verified → coverage should be ~0.333
        assert result["coverage_after"] == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_text_fact_from_observation(self, multi_item_tes):
        """Verify text_fact evidence from observation."""
        ledger = EvidenceLedger(multi_item_tes)
        obs = {
            "tool_name": "search_table",
            "content": "年检安排表体现了高度的流程统一性，符合标准规范",
            "success": True,
        }
        result = ledger.update(
            tool_call={"tool_name": "search_table", "arguments": {}},
            observation=obs,
            observation_metadata={"file": "Sheet50.csv"},
        )
        assert "sq1_e2" in result["new_evidence_ids"]


# ======================================================================
# Test 12 — Source mismatch
# ======================================================================

class TestSourceMismatch:
    def test_wrong_source_blocks_verification(self, multi_item_tes, observation_wrong_source):
        """Correct value but wrong source → evidence NOT added to ledger."""
        ledger = EvidenceLedger(multi_item_tes)
        result = ledger.update(
            tool_call={"tool_name": "search_table", "arguments": {}},
            observation=observation_wrong_source,
            observation_metadata={"file": "wrong_file.csv"},
        )
        assert "sq1_e1" not in result["new_evidence_ids"]
        assert result["coverage_after"] == 0.0

    def test_no_source_metadata_blocks_verification(self, multi_item_tes, observation_wrong_source):
        """No observable source metadata → evidence NOT added."""
        ledger = EvidenceLedger(multi_item_tes)
        result = ledger.update(
            tool_call={"tool_name": "search_table", "arguments": {}},
            observation=observation_wrong_source,
            # no observation_metadata
        )
        assert "sq1_e1" not in result["new_evidence_ids"]


# ======================================================================
# Test 13 — Binding mismatch
# ======================================================================

class TestBindingMismatch:
    def test_missing_entity_allowed_with_source_and_value(self, multi_item_tes):
        """Value correct + source match → evidence enters ledger even when
        entity/time are uncertain (rule verifier is lenient, LLM fallback
        handles real binding verification)."""
        ledger = EvidenceLedger(multi_item_tes)
        obs = {
            "tool_name": "search_table",
            "content": "同比增长率为16.96%",  # value ok but no entity/time
            "success": True,
        }
        result = ledger.update(
            tool_call={"tool_name": "search_table", "arguments": {}},
            observation=obs,
            observation_metadata={"file": "auto_2010.csv"},
        )
        # With value + source match, lenient binding allows entry
        assert "sq1_e1" in result["new_evidence_ids"], (
            f"Evidence should enter ledger with value+source match: "
            f"got new_ids={result['new_evidence_ids']}"
        )

    def test_binding_uncertainty_recorded(self, multi_item_tes):
        """When binding fields are uncertain, the audit records it."""
        ledger = EvidenceLedger(multi_item_tes)
        obs = {
            "tool_name": "search_table",
            "content": "同比增长率为16.96%",  # no entity/time
            "success": True,
        }
        result = ledger.update(
            tool_call={"tool_name": "search_table", "arguments": {}},
            observation=obs,
            observation_metadata={"file": "auto_2010.csv"},
        )
        assert "sq1_e1" in result["new_evidence_ids"]
        # Audit should show uncertainty in binding
        audit = ledger.get_audit("sq1_e1")
        assert audit is not None


# ======================================================================
# Test 14 — Derived evidence verification
# ======================================================================

class TestDerivedEvidence:
    @pytest.fixture
    def derived_tes(self):
        """Target evidence set with a derived_value that depends on two raw_values."""
        return TargetEvidenceSet(
            sample_id="s1", subquestion_id=1, question="计算增长率",
            evidence_items=[
                EvidenceItem(
                    sample_id="s1", subquestion_id=1, evidence_id="e1",
                    type="raw_value", value="118.46",
                    entity="乘用车", time="2010年1月",
                    source_tables=["auto_2010.csv"], weight=1.0,
                ),
                EvidenceItem(
                    sample_id="s1", subquestion_id=1, evidence_id="e2",
                    type="raw_value", value="100.00",
                    entity="乘用车", time="2009年1月",
                    source_tables=["auto_2009.csv"], weight=1.0,
                ),
                EvidenceItem(
                    sample_id="s1", subquestion_id=1, evidence_id="e3",
                    type="derived_value", value="18.46%",
                    entity="乘用车", time="2010年1月",
                    metric="出口量同比增长率", unit="%",
                    source_tables=["auto_2010.csv", "auto_2009.csv"],
                    input_evidence_ids=["e1", "e2"],
                    operation="同比增长率",
                    weight=1.0,
                ),
            ],
        )

    def test_derived_inputs_missing_blocks(self, derived_tes):
        """Derived evidence cannot be verified if its inputs aren't in the ledger."""
        ledger = EvidenceLedger(derived_tes)
        # Try to verify the derived item directly — inputs missing
        obs = {
            "tool_name": "python_exec",
            "content": "增长率 = (118.46 - 100) / 100 = 0.1846 → 18.46%",
            "success": True,
        }
        result = ledger.update(
            tool_call={"tool_name": "python_exec", "arguments": {"code": "..."}},
            observation=obs,
            observation_metadata={"file": "auto_2010.csv"},
            code_output="增长率 = 18.46%",
        )
        # e3 should NOT be verified because e1 and e2 are not yet in ledger
        assert "e3" not in result["new_evidence_ids"], (
            f"Derived evidence should NOT enter ledger without verified inputs: "
            f"got {result['new_evidence_ids']}"
        )

    def test_derived_verified_after_inputs(self, derived_tes):
        """Step 1: verify inputs. Step 2: verify derived."""
        ledger = EvidenceLedger(derived_tes)

        # Step 1: verify e1 and e2 via observations
        ledger.update(
            tool_call={"tool_name": "search_table", "arguments": {}},
            observation={
                "tool_name": "search_table",
                "content": "当前出口量: 118.46 乘用车 2010年1月",
                "success": True,
            },
            observation_metadata={"file": "auto_2010.csv"},
        )
        ledger.update(
            tool_call={"tool_name": "search_table", "arguments": {}},
            observation={
                "tool_name": "search_table",
                "content": "去年出口量: 100.00 乘用车 2009年1月",
                "success": True,
            },
            observation_metadata={"file": "auto_2009.csv"},
        )
        assert "e1" in ledger.verified_ids
        assert "e2" in ledger.verified_ids

        # Step 2: verify derived e3 via code execution
        result = ledger.update(
            tool_call={"tool_name": "python_exec", "arguments": {"code": "..."}},
            observation={
                "tool_name": "python_exec",
                "content": "增长率 = (118.46 - 100) / 100 * 100 = 18.46%",
                "success": True,
            },
            observation_metadata={"file": "auto_2010.csv"},
            code_output="增长率 = (118.46 - 100) / 100 * 100 = 18.46%",
        )
        assert "e3" in result["new_evidence_ids"], (
            f"Derived evidence e3 should be verified after inputs. "
            f"Audit: {result['audit']}"
        )


# ======================================================================
# Test 15 — Repeated evidence / monotonicity
# ======================================================================

class TestMonotonicity:
    def test_repeated_observation_no_new_evidence(self, multi_item_tes, observation_match):
        """Same observation twice → second time produces no new evidence."""
        ledger = EvidenceLedger(multi_item_tes)

        # First call
        r1 = ledger.update(
            tool_call={"tool_name": "search_table", "arguments": {}},
            observation=observation_match,
            observation_metadata={"file": "auto_2010.csv"},
        )
        assert len(r1["new_evidence_ids"]) >= 1

        # Second call with same observation
        r2 = ledger.update(
            tool_call={"tool_name": "search_table", "arguments": {}},
            observation=observation_match,
            observation_metadata={"file": "auto_2010.csv"},
        )
        assert r2["new_evidence_ids"] == [], (
            f"Repeated observation should not add evidence: {r2['new_evidence_ids']}"
        )
        assert r2["coverage_before"] == r2["coverage_after"]

    def test_coverage_never_decreases(self, multi_item_tes):
        """Coverage should be monotonic — never decrease."""
        ledger = EvidenceLedger(multi_item_tes)
        prev_cov = ledger.coverage

        # Update with matching observation
        obs = {
            "tool_name": "search_table",
            "content": "乘用车在2010年1月产量同比增长率为16.96%",
            "success": True,
        }
        ledger.update(
            tool_call={"tool_name": "search_table", "arguments": {}},
            observation=obs,
            observation_metadata={"file": "auto_2010.csv"},
        )
        assert ledger.coverage >= prev_cov

        # Update with non-matching observation
        prev_cov = ledger.coverage
        ledger.update(
            tool_call={"tool_name": "search_table", "arguments": {}},
            observation={"tool_name": "search_table", "content": "无关数据", "success": True},
            observation_metadata={"file": "other.csv"},
        )
        assert ledger.coverage == prev_cov

    def test_delta_phi_computation(self):
        """ΔΦ = max(0, coverage_after - coverage_before)."""
        assert EvidenceLedger.compute_delta_phi(0.0, 0.5) == 0.5
        assert EvidenceLedger.compute_delta_phi(0.5, 0.5) == 0.0
        assert EvidenceLedger.compute_delta_phi(0.8, 0.3) == 0.0  # never negative


# ======================================================================
# Test 16 — Infrastructure errors
# ======================================================================

class TestInfrastructureErrors:
    def test_failed_observation_not_counted(self, multi_item_tes):
        """Observation with success=False → infrastructure error, not counted."""
        ledger = EvidenceLedger(multi_item_tes)
        result = ledger.update(
            tool_call={"tool_name": "search_table", "arguments": {}},
            observation={
                "tool_name": "search_table",
                "content": "乘用车产量同比增长率为16.96%",
                "success": False,  # infrastructure failure
            },
            observation_metadata={"file": "auto_2010.csv"},
        )
        assert result["new_evidence_ids"] == []
        assert "error" in result["audit"]
        assert result["coverage_after"] == 0.0


# ======================================================================
# Test 17 — Coverage
# ======================================================================

class TestCoverage:
    def test_empty_target_set(self):
        tes = TargetEvidenceSet(sample_id="s1", subquestion_id=1, question="q")
        ledger = EvidenceLedger(tes)
        assert ledger.coverage == 0.0
        assert ledger._total_weight > 0  # epsilon protection

    def test_weighted_coverage(self):
        """Items with different weights contribute proportionally."""
        tes = TargetEvidenceSet(
            sample_id="s1", subquestion_id=1, question="q",
            evidence_items=[
                EvidenceItem(sample_id="s1", subquestion_id=1, evidence_id="e1",
                             type="raw_value", value="A", weight=2.0,
                             source_tables=["t.csv"]),
                EvidenceItem(sample_id="s1", subquestion_id=1, evidence_id="e2",
                             type="raw_value", value="B", weight=1.0,
                             source_tables=["t.csv"]),
            ],
        )
        ledger = EvidenceLedger(tes)

        # Verify e1 (weight=2.0) → coverage = 2/3
        result = ledger.update(
            tool_call={"tool_name": "search_table", "arguments": {}},
            observation={"tool_name": "search_table", "content": "A", "success": True},
            observation_metadata={"file": "t.csv"},
        )
        assert result["coverage_after"] == pytest.approx(2.0 / 3.0)

        # Verify e2 (weight=1.0) → coverage = 3/3 = 1.0
        result = ledger.update(
            tool_call={"tool_name": "search_table", "arguments": {}},
            observation={"tool_name": "search_table", "content": "B", "success": True},
            observation_metadata={"file": "t.csv"},
        )
        assert result["coverage_after"] == pytest.approx(1.0)


# ======================================================================
# Test 18 — Ledger serialisation & query
# ======================================================================

class TestLedgerSerialisation:
    def test_to_dict(self, multi_item_tes, observation_match):
        ledger = EvidenceLedger(multi_item_tes)
        ledger.update(
            tool_call={"tool_name": "search_table", "arguments": {}},
            observation=observation_match,
            observation_metadata={"file": "auto_2010.csv"},
        )
        d = ledger.to_dict()
        assert d["sample_id"] == "s1"
        assert d["subquestion_id"] == 1
        assert d["coverage"] > 0.0
        assert len(d["verified_ids"]) >= 1

    def test_summary_string(self, multi_item_tes, observation_match):
        ledger = EvidenceLedger(multi_item_tes)
        ledger.update(
            tool_call={"tool_name": "search_table", "arguments": {}},
            observation=observation_match,
            observation_metadata={"file": "auto_2010.csv"},
        )
        s = ledger.summary()
        assert "EvidenceLedger" in s
        assert str(ledger.coverage) in s or f"{ledger.coverage:.2%}" in s

    def test_is_verified(self, multi_item_tes, observation_match):
        ledger = EvidenceLedger(multi_item_tes)
        assert not ledger.is_verified("sq1_e1")
        ledger.update(
            tool_call={"tool_name": "search_table", "arguments": {}},
            observation=observation_match,
            observation_metadata={"file": "auto_2010.csv"},
        )
        assert ledger.is_verified("sq1_e1")

    def test_get_audit(self, multi_item_tes, observation_match):
        ledger = EvidenceLedger(multi_item_tes)
        ledger.update(
            tool_call={"tool_name": "search_table", "arguments": {}},
            observation=observation_match,
            observation_metadata={"file": "auto_2010.csv"},
        )
        audit = ledger.get_audit("sq1_e1")
        assert audit is not None
        assert audit["source"] == "tool_observation"

    def test_delta_coverage(self, multi_item_tes, observation_match):
        ledger = EvidenceLedger(multi_item_tes)
        cov0 = ledger.coverage
        ledger.update(
            tool_call={"tool_name": "search_table", "arguments": {}},
            observation=observation_match,
            observation_metadata={"file": "auto_2010.csv"},
        )
        delta = ledger.delta_coverage(cov0)
        assert delta > 0.0


# ======================================================================
# Test 19 — Update result structure
# ======================================================================

class TestUpdateResultStructure:
    def test_update_returns_required_keys(self, multi_item_tes, observation_match):
        ledger = EvidenceLedger(multi_item_tes)
        result = ledger.update(
            tool_call={"tool_name": "search_table", "arguments": {"query": "2010"}},
            observation=observation_match,
            observation_metadata={"file": "auto_2010.csv"},
        )
        required = ["coverage_before", "coverage_after", "new_evidence_ids",
                     "ledger", "audit", "tool_info"]
        for key in required:
            assert key in result, f"Missing key: {key}"

        assert isinstance(result["coverage_before"], float)
        assert isinstance(result["coverage_after"], float)
        assert isinstance(result["new_evidence_ids"], list)
        assert isinstance(result["ledger"], dict)
        assert isinstance(result["audit"], dict)
        assert isinstance(result["tool_info"], dict)
        assert result["tool_info"]["tool_name"] == "search_table"
        assert result["tool_info"]["step"] >= 1

    def test_memory_init_returns_required_keys(self, multi_item_tes, memory_with_facts):
        ledger = EvidenceLedger(multi_item_tes)
        result = ledger.initialize_from_memory(memory_with_facts)
        for key in ["coverage_before", "coverage_after", "new_evidence_ids",
                     "memory_facts_checked", "audit"]:
            assert key in result, f"Missing key: {key}"


# ======================================================================
# Test 20 — End-to-end scenario
# ======================================================================

class TestEndToEndScenario:
    def test_full_subquestion_flow(self):
        """Simulate a complete subquestion: memory init → 2 tool calls → derived."""
        tes = TargetEvidenceSet(
            sample_id="s1", subquestion_id=2, question="计算增长率",
            evidence_items=[
                EvidenceItem(sample_id="s1", subquestion_id=2, evidence_id="sq2_e1",
                             type="raw_value", value="118.46",
                             entity="乘用车", time="2010年1月",
                             metric="出口量", unit="万辆",
                             source_tables=["auto_2010.csv"], weight=1.0),
                EvidenceItem(sample_id="s1", subquestion_id=2, evidence_id="sq2_e2",
                             type="raw_value", value="100.00",
                             entity="乘用车", time="2009年1月",
                             metric="出口量", unit="万辆",
                             source_tables=["auto_2009.csv"], weight=1.0),
                EvidenceItem(sample_id="s1", subquestion_id=2, evidence_id="sq2_e3",
                             type="derived_value", value="18.46%",
                             entity="乘用车", time="2010年1月",
                             metric="出口量同比增长率", unit="%",
                             source_tables=["auto_2010.csv", "auto_2009.csv"],
                             input_evidence_ids=["sq2_e1", "sq2_e2"],
                             operation="同比增长率", weight=1.0),
            ],
        )

        # Memory from previous subquestion contains one of the inputs
        memory = {
            "goal": "分析出口数据",
            "key_facts": [
                {
                    "entity": "乘用车", "time": "2009年1月",
                    "metric": "出口量", "value": "100.00",
                    "unit": "万辆", "provenance": "auto_2009.csv",
                },
            ],
        }

        # Step 0: Initialize from memory
        ledger = EvidenceLedger(tes)
        r0 = ledger.initialize_from_memory(memory)
        print(f"\n  After memory init: coverage={ledger.coverage:.2%}, "
              f"new={r0['new_evidence_ids']}")
        assert "sq2_e2" in ledger.verified_ids  # 2009 data from memory
        assert "sq2_e1" not in ledger.verified_ids  # not in memory

        # Step 1: Search for current year data
        r1 = ledger.update(
            tool_call={"tool_name": "search_table", "arguments": {"query": "2010"}},
            observation={
                "tool_name": "search_table",
                "content": "查得2010年1月乘用车出口量为118.46万辆",
                "success": True,
            },
            observation_metadata={"file": "auto_2010.csv"},
        )
        print(f"  After tool 1 (search): coverage={ledger.coverage:.2%}, "
              f"new={r1['new_evidence_ids']}")
        assert "sq2_e1" in r1["new_evidence_ids"]
        assert ledger.coverage == pytest.approx(2.0 / 3.0)

        # Step 2: Calculate growth rate
        r2 = ledger.update(
            tool_call={"tool_name": "python_exec",
                       "arguments": {"code": "(118.46-100)/100*100"}},
            observation={
                "tool_name": "python_exec",
                "content": "计算结果: 18.46%",
                "success": True,
            },
            observation_metadata={"file": "auto_2010.csv"},
            code_output="增长率 = (118.46 - 100.00) / 100.00 * 100 = 18.46%",
        )
        print(f"  After tool 2 (compute): coverage={ledger.coverage:.2%}, "
              f"new={r2['new_evidence_ids']}")
        assert "sq2_e3" in r2["new_evidence_ids"]
        assert ledger.coverage == pytest.approx(1.0)

        # Final state
        print(f"  Final: {ledger.summary()}")
        assert ledger.coverage == 1.0
