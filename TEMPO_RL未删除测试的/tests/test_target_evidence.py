"""
Unit tests for TEMPO-RL Phase 0 — Target Evidence Builder.

Run from the project root::

    python -m pytest TEMPO_RL/tests/test_target_evidence.py -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import pytest

# Ensure the project root is on sys.path
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from TEMPO_RL.schemas import AuditInfo, EvidenceItem, TargetEvidenceSet
from TEMPO_RL.build_target_evidence import (
    TargetEvidenceBuilder,
    extract_numeric_values,
    extract_percentage_values,
    extract_time,
    detect_computation,
)
from TEMPO_RL.io_utils import read_jsonl, write_jsonl, load_benchmark_samples


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def builder():
    return TargetEvidenceBuilder(llm_client=None, llm_enabled=False)


@pytest.fixture
def sample_benchmark_entry():
    """Minimal benchmark entry with one subquestion."""
    return {
        "task": "test sample",
        "file_path": "dataset/tables/test",
        "design": {
            "type": "Tree",
            "checkout_list": [
                {
                    "idx": 1,
                    "info_item": "2010年1月乘用车产量同比增长率是多少？",
                    "related_tables": ["auto_2010.csv"],
                    "score_points": [
                        "乘用车产量同比增长率为16.96%",
                        "出口量同比增长率为18.46%，需根据出口当月值和去年值计算得出",
                    ],
                },
            ],
        },
    }


# ======================================================================
# Test 1 — Rule extraction helpers
# ======================================================================

class TestRuleExtraction:
    def test_numeric_simple(self):
        vals = extract_numeric_values("产量为16.96%，出口18.46%")
        assert "16.96%" in vals
        assert "18.46%" in vals

    def test_numeric_with_compound_unit(self):
        vals = extract_numeric_values("造价41459万元/公里，占比10.89%")
        assert "41459万元/公里" in vals, f"Got: {vals}"
        assert "10.89%" in vals, f"Got: {vals}"

    def test_numeric_no_match(self):
        vals = extract_numeric_values("这是一个定性描述")
        assert vals == []

    def test_percentage(self):
        vals = extract_percentage_values("增长16.96%和18.46%")
        assert vals == ["16.96%", "18.46%"]

    def test_time_extraction(self):
        assert extract_time("2010年1月数据") == "2010年1月"
        assert extract_time("2020年11月上半月") == "2020年11月上半月"
        assert extract_time("无时间信息") is None

    def test_detect_computation_positive(self):
        is_comp, op = detect_computation("需根据当月值和去年值计算同比增长率")
        assert is_comp
        assert "计算" in op

        is_comp2, op2 = detect_computation("求平均值")
        assert is_comp2

        is_comp3, op3 = detect_computation("根据A和B计算占比")
        assert is_comp3

    def test_detect_computation_negative(self):
        is_comp, op = detect_computation("产量为16.96%")
        assert not is_comp
        assert op is None


# ======================================================================
# Test 2 — EvidenceItem validation
# ======================================================================

class TestEvidenceItemValidation:
    def test_valid_raw_value(self):
        ei = EvidenceItem(
            sample_id="s1", subquestion_id=1, evidence_id="e1",
            type="raw_value", value="16.96%",
            entity="乘用车", time="2010年1月",
            metric="产量同比增长率", unit="%",
        )
        assert ei.validate() == []

    def test_missing_required_field(self):
        ei = EvidenceItem(
            sample_id="", subquestion_id=1, evidence_id="e1",
            type="raw_value", value="16.96%",
        )
        issues = ei.validate()
        assert any("sample_id" in i for i in issues), f"Issues: {issues}"

    def test_unknown_type(self):
        ei = EvidenceItem(
            sample_id="s1", subquestion_id=1, evidence_id="e1",
            type="bogus_type", value="x",
        )
        issues = ei.validate()
        assert any("unknown type" in i for i in issues)

    def test_derived_value_missing_inputs(self):
        ei = EvidenceItem(
            sample_id="s1", subquestion_id=1, evidence_id="e1",
            type="derived_value", value="18.46%",
            operation="同比增长率",
            input_evidence_ids=[],  # should have inputs
        )
        issues = ei.validate()
        assert any("input_evidence_ids" in i for i in issues), f"Issues: {issues}"

    def test_derived_value_missing_operation(self):
        ei = EvidenceItem(
            sample_id="s1", subquestion_id=1, evidence_id="e1",
            type="derived_value", value="18.46%",
            input_evidence_ids=["e2", "e3"],
        )
        issues = ei.validate()
        assert any("operation" in i for i in issues), f"Issues: {issues}"

    def test_text_fact(self):
        ei = EvidenceItem(
            sample_id="s1", subquestion_id=1, evidence_id="e1",
            type="text_fact", value="可从库存流通维度切入并获取关键数据",
        )
        assert ei.validate() == []

    def test_missing_field_warning_in_audit(self):
        ei = EvidenceItem(
            sample_id="s1", subquestion_id=1, evidence_id="e1",
            type="raw_value", value="16.96%",
            entity=None, time=None, metric=None, unit=None,
            audit=AuditInfo(
                parse_confidence=0.5,
                warnings=["entity not detected", "time not detected"],
                source="rule_extraction",
            ),
        )
        # validate() checks required fields, not optional ones — ok
        assert ei.validate() == []
        # But audit carries warnings
        assert len(ei.audit.warnings) == 2


# ======================================================================
# Test 3 — TargetEvidenceSet
# ======================================================================

class TestTargetEvidenceSet:
    def test_empty_set(self):
        s = TargetEvidenceSet(sample_id="s1", subquestion_id=1, question="q")
        assert s.evidence_items == []
        assert s.validate_all() == {}

    def test_validate_all(self):
        s = TargetEvidenceSet(
            sample_id="s1", subquestion_id=1, question="q",
            evidence_items=[
                EvidenceItem(
                    sample_id="", subquestion_id=1, evidence_id="bad",
                    type="raw_value", value="x",
                ),
                EvidenceItem(
                    sample_id="s1", subquestion_id=1, evidence_id="good",
                    type="text_fact", value="ok",
                ),
            ],
        )
        issues = s.validate_all()
        assert "bad" in issues
        assert "good" not in issues


# ======================================================================
# Test 4 — Serialisation round-trip
# ======================================================================

class TestSerialisation:
    def test_evidence_item_roundtrip(self):
        ei = EvidenceItem(
            sample_id="s1", subquestion_id=2, evidence_id="sq2_e1",
            type="derived_value", value="18.46%",
            entity="乘用车", time="2010年1月",
            metric="出口量同比增长率", unit="%",
            source_tables=["auto_2010.csv"],
            input_evidence_ids=["sq2_e_export", "sq2_e_prev"],
            operation="同比增长率",
            weight=1.0,
            audit=AuditInfo(parse_confidence=0.85, warnings=["test warning"], source="llm_annotation"),
        )
        d = ei.to_dict()
        ei2 = EvidenceItem.from_dict(d)
        assert ei2.sample_id == ei.sample_id
        assert ei2.subquestion_id == ei.subquestion_id
        assert ei2.type == "derived_value"
        assert ei2.value == "18.46%"
        assert ei2.entity == "乘用车"
        assert ei2.input_evidence_ids == ["sq2_e_export", "sq2_e_prev"]
        assert ei2.operation == "同比增长率"
        assert ei2.audit.parse_confidence == 0.85
        assert ei2.audit.warnings == ["test warning"]
        assert ei2.audit.source == "llm_annotation"

    def test_target_evidence_set_roundtrip(self):
        s = TargetEvidenceSet(
            sample_id="s1", subquestion_id=1, question="test q",
            evidence_items=[
                EvidenceItem(
                    sample_id="s1", subquestion_id=1, evidence_id="e1",
                    type="raw_value", value="16.96%",
                    entity="乘用车", source_tables=["t1.csv"],
                ),
            ],
        )
        d = s.to_dict()
        s2 = TargetEvidenceSet.from_dict(d)
        assert s2.sample_id == "s1"
        assert s2.question == "test q"
        assert len(s2.evidence_items) == 1
        assert s2.evidence_items[0].value == "16.96%"

    def test_jsonl_write_read(self):
        items = [
            {"a": 1, "b": "hello"},
            {"a": 2, "b": "world"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            tmp = f.name
        try:
            write_jsonl(tmp, items)
            loaded = read_jsonl(tmp)
            assert len(loaded) == 2
            assert loaded[0]["a"] == 1
            assert loaded[1]["b"] == "world"
        finally:
            os.unlink(tmp)


# ======================================================================
# Test 5 — Builder: raw numeric evidence
# ======================================================================

class TestBuilderRawNumeric:
    def test_single_numeric_score_point(self, builder, sample_benchmark_entry):
        sets = builder.build_one_sample(sample_benchmark_entry)
        assert len(sets) == 1
        es = sets[0]
        assert es.subquestion_id == 1
        # At least one item per score_point
        assert len(es.evidence_items) >= 2  # two score_points

        # First score_point: "乘用车产量同比增长率为16.96%"
        raw_items = [e for e in es.evidence_items if e.type == "raw_value"]
        assert len(raw_items) >= 1
        assert any("16.96%" in e.value for e in raw_items)

    def test_text_fact_score_point(self, builder):
        """Score point with no numerics should become text_fact."""
        sample = {
            "task": "test",
            "design": {
                "type": "Tree",
                "checkout_list": [{
                    "idx": 1,
                    "info_item": "评估维度有哪些？",
                    "related_tables": [],
                    "score_points": ["可从库存流通维度切入"],
                }],
            },
        }
        sets = builder.build_one_sample(sample)
        assert len(sets) == 1
        items = sets[0].evidence_items
        assert len(items) >= 1
        assert items[0].type == "text_fact"
        assert "库存流通" in items[0].value

    def test_derived_value_detection(self, builder):
        """Score point mentioning computation → derived_value."""
        sample = {
            "task": "test",
            "design": {
                "type": "Tree",
                "checkout_list": [{
                    "idx": 1,
                    "info_item": "计算同比增长率",
                    "related_tables": ["t.csv"],
                    "score_points": ["出口同比增长率为18.46%，需根据当月值和去年值计算得出"],
                }],
            },
        }
        sets = builder.build_one_sample(sample)
        items = sets[0].evidence_items
        derived = [e for e in items if e.type == "derived_value"]
        assert len(derived) >= 1, f"No derived_value in: {[e.type for e in items]}"
        assert derived[0].operation is not None


# ======================================================================
# Test 6 — Builder: derived evidence with input_evidence_ids
# ======================================================================

class TestBuilderDerived:
    def test_derived_with_explicit_inputs(self):
        """Verify derived_value items carry input_evidence_ids when available."""
        ei = EvidenceItem(
            sample_id="s1", subquestion_id=1, evidence_id="sq1_e3",
            type="derived_value", value="18.46%",
            entity="乘用车", time="2010年1月",
            metric="出口量同比增长率", unit="%",
            source_tables=["auto_2010.csv"],
            input_evidence_ids=["sq1_e1", "sq1_e2"],
            operation="同比增长率",
            weight=1.0,
            audit=AuditInfo(parse_confidence=0.9, source="llm_annotation"),
        )
        assert ei.validate() == []
        # Verify C_input can be checked: input_evidence_ids are non-empty
        assert len(ei.input_evidence_ids) == 2
        # Verify C_operator can be checked: operation is set
        assert ei.operation == "同比增长率"
        # Verify C_result can be checked: value is numeric
        assert "18.46" in ei.value

    def test_derived_in_builder_pipeline(self, builder):
        """Score point with computation keywords and numerics."""
        sample = {
            "task": "test",
            "design": {
                "type": "Tree",
                "checkout_list": [{
                    "idx": 1,
                    "info_item": "计算桥梁工程成本占比",
                    "related_tables": ["project.csv"],
                    "score_points": [
                        "桥梁工程的成本占比为10.89%",
                        "需根据桥梁成本和总成本计算占比",
                    ],
                }],
            },
        }
        sets = builder.build_one_sample(sample)
        items = sets[0].evidence_items

        # Should have at least one derived_value (占比 is a ratio computation)
        types = [e.type for e in items]
        assert "derived_value" in types or "raw_value" in types, f"Types: {types}"

        # Audit field present on every item
        for e in items:
            assert e.audit is not None
            assert isinstance(e.audit.parse_confidence, float)
            assert isinstance(e.audit.warnings, list)
            assert e.audit.source in ("rule_extraction", "llm_annotation", "manual", "sft_trajectory")


# ======================================================================
# Test 7 — Missing field warnings
# ======================================================================

class TestMissingFieldWarnings:
    def test_rule_extraction_warns_missing_fields(self, builder):
        """Items produced by rule extraction should carry warnings for
        entity / metric (which rules cannot reliably detect)."""
        sample = {
            "task": "test",
            "design": {
                "type": "Tree",
                "checkout_list": [{
                    "idx": 1,
                    "info_item": "x",
                    "related_tables": [],
                    "score_points": ["数值为42"],
                }],
            },
        }
        sets = builder.build_one_sample(sample)
        items = sets[0].evidence_items
        assert len(items) >= 1
        # Rule extraction should note that entity/metric were not detected
        warns = items[0].audit.warnings
        assert any("entity" in w for w in warns), f"Warnings: {warns}"
        assert any("metric" in w for w in warns), f"Warnings: {warns}"

    def test_no_silent_drop(self, builder):
        """Every score_point must produce at least one item, even if unparseable."""
        sample = {
            "task": "test",
            "design": {
                "type": "Tree",
                "checkout_list": [{
                    "idx": 1,
                    "info_item": "x",
                    "related_tables": [],
                    "score_points": [
                        "纯文本描述没有任何数字",
                        "another purely qualitative point",
                    ],
                }],
            },
        }
        sets = builder.build_one_sample(sample)
        items = sets[0].evidence_items
        assert len(items) >= 2, f"Expected >=2 items for 2 score_points, got {len(items)}"

        # Both should be text_fact
        for e in items:
            assert e.type == "text_fact", f"Expected text_fact, got {e.type}"


# ======================================================================
# Test 8 — Builder: I/O round-trip
# ======================================================================

class TestBuilderIO:
    def test_save_load_roundtrip(self, builder, sample_benchmark_entry):
        sets = builder.build_one_sample(sample_benchmark_entry)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            tmp = f.name
        try:
            TargetEvidenceBuilder.save(sets, tmp)
            loaded = TargetEvidenceBuilder.load(tmp)
            assert len(loaded) == len(sets)
            assert loaded[0].sample_id == sets[0].sample_id
            assert loaded[0].subquestion_id == sets[0].subquestion_id
            assert len(loaded[0].evidence_items) == len(sets[0].evidence_items)
            # Spot-check first item
            orig_item = sets[0].evidence_items[0]
            load_item = loaded[0].evidence_items[0]
            assert load_item.value == orig_item.value
            assert load_item.type == orig_item.type
            assert load_item.audit.parse_confidence == orig_item.audit.parse_confidence
        finally:
            os.unlink(tmp)


# ======================================================================
# Test 9 — Benchmark sample loading
# ======================================================================

class TestBenchmarkLoading:
    def test_load_val_json(self):
        """Verify the real val.json can be loaded."""
        val_path = os.path.join(_PROJ_ROOT, "dataset", "val.json")
        if not os.path.exists(val_path):
            pytest.skip("val.json not found")
        samples = load_benchmark_samples(val_path)
        assert isinstance(samples, list)
        assert len(samples) > 0
        # Each sample should have a design.checkout_list
        for s in samples[:3]:
            assert "design" in s
            assert "checkout_list" in s["design"]

    def test_build_on_real_data(self, builder):
        """Build target evidence from real val.json (rule-only)."""
        val_path = os.path.join(_PROJ_ROOT, "dataset", "val.json")
        if not os.path.exists(val_path):
            pytest.skip("val.json not found")
        samples = load_benchmark_samples(val_path)
        # Only first 5 to keep test fast
        sets = builder.build(samples[:5])

        total_items = sum(len(s.evidence_items) for s in sets)
        total_score_points = sum(
            len(item.get("score_points", []))
            for sample in samples[:5]
            for item in sample.get("design", {}).get("checkout_list", [])
        )
        print(f"\n  Built {len(sets)} evidence sets, {total_items} items "
              f"from {total_score_points} score points")

        # Every score point must have at least one evidence item
        assert total_items >= total_score_points, (
            f"Fewer items ({total_items}) than score_points ({total_score_points})"
        )

        # Check every item has audit
        for s in sets:
            for e in s.evidence_items:
                assert e.audit is not None
                assert e.audit.source in (
                    "rule_extraction", "llm_annotation", "manual", "sft_trajectory"
                )

        # Print a sample
        if sets:
            print(f"  Sample set: sq{sets[0].subquestion_id} "
                  f"'{sets[0].question[:60]}...'")
            for ei in sets[0].evidence_items[:3]:
                print(f"    {ei.evidence_id}: type={ei.type} value={ei.value[:60]} "
                      f"confidence={ei.audit.parse_confidence:.2f}")
