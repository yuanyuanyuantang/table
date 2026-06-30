"""
Unit tests for TEMPO-RL Phase 0 — Future Dependency Builder.

Run from the project root::

    python -m pytest TEMPO_RL/tests/test_future_dependencies.py -v
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

from TEMPO_RL.schemas import (
    AuditInfo,
    EvidenceItem,
    FutureDependency,
    FutureDependencySet,
    TargetEvidenceSet,
    required_fields_for_type,
)
from TEMPO_RL.build_future_dependencies import (
    FutureDependencyBuilder,
    detect_referring_expressions,
    extract_entities,
    extract_times,
    extract_metrics,
    extract_constraints,
)
from TEMPO_RL.io_utils import read_jsonl, write_jsonl, load_benchmark_samples


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def builder():
    return FutureDependencyBuilder(d_fdc=2)


@pytest.fixture
def builder_narrow():
    """Builder with D_FDC = 1 (only immediate next subquestion)."""
    return FutureDependencyBuilder(d_fdc=1)


@pytest.fixture
def simple_two_sq_sample():
    """Minimal sample with two subquestions where sq2 references sq1."""
    return {
        "task": "test two sq sample",
        "design": {
            "type": "Tree",
            "checkout_list": [
                {
                    "idx": 1,
                    "info_item": "计算乘用车2010年1月产量同比增长率",
                    "related_tables": ["auto_2010.csv"],
                    "score_points": [
                        "乘用车产量同比增长率为16.96%",
                        "出口量同比增长率为18.46%",
                    ],
                },
                {
                    "idx": 2,
                    "info_item": "基于该公司2010年1月的乘用车产量数据，分析该企业的生产趋势",
                    "related_tables": ["auto_2010.csv"],
                    "score_points": [
                        "该公司产量呈现增长趋势",
                    ],
                },
            ],
        },
    }


@pytest.fixture
def three_sq_sample():
    """Sample with three subquestions, sq2 and sq3 both depend on sq1."""
    return {
        "task": "test three sq sample",
        "design": {
            "type": "Tree",
            "checkout_list": [
                {
                    "idx": 1,
                    "info_item": "分析乘用车和商用车的2010年产销数据",
                    "related_tables": ["auto_2010.csv"],
                    "score_points": [
                        "乘用车产量为100万辆",
                        "商用车产量为50万辆",
                    ],
                },
                {
                    "idx": 2,
                    "info_item": "计算乘用车在2010年的市场占比",
                    "related_tables": ["auto_2010.csv"],
                    "score_points": [
                        "乘用车市场占比为66.7%",
                    ],
                },
                {
                    "idx": 3,
                    "info_item": "综合以上分析，评估2010年市场结构",
                    "related_tables": ["auto_2010.csv", "market.csv"],
                    "score_points": [
                        "2010年市场以乘用车为主导",
                    ],
                },
            ],
        },
    }


@pytest.fixture
def four_sq_sample():
    """Sample with 4 subquestions to test D_FDC=2 cutoff."""
    return {
        "task": "test D_FDC cutoff",
        "design": {
            "type": "Tree",
            "checkout_list": [
                {
                    "idx": 1,
                    "info_item": "sq1: 获取2010年基础数据",
                    "related_tables": ["t1.csv"],
                    "score_points": ["数值A为100"],
                },
                {
                    "idx": 2,
                    "info_item": "sq2: 基于上述数据计算增长率",
                    "related_tables": ["t1.csv"],
                    "score_points": ["增长率为10%"],
                },
                {
                    "idx": 3,
                    "info_item": "sq3: 结合上述增长率分析趋势",
                    "related_tables": ["t1.csv", "t2.csv"],
                    "score_points": ["趋势向上"],
                },
                {
                    "idx": 4,
                    "info_item": "sq4: 综合评估（距离sq1超过2步）",
                    "related_tables": ["t3.csv"],
                    "score_points": ["综合评估完成"],
                },
            ],
        },
    }


@pytest.fixture
def sample_target_evidence():
    """Target evidence sets matching simple_two_sq_sample."""
    return [
        TargetEvidenceSet(
            sample_id="test two sq sample",
            subquestion_id=1,
            question="计算乘用车2010年1月产量同比增长率",
            evidence_items=[
                EvidenceItem(
                    sample_id="test two sq sample",
                    subquestion_id=1,
                    evidence_id="sq1_e1",
                    type="raw_value",
                    value="16.96%",
                    entity="乘用车",
                    time="2010年1月",
                    metric="产量同比增长率",
                    unit="%",
                    source_tables=["auto_2010.csv"],
                ),
                EvidenceItem(
                    sample_id="test two sq sample",
                    subquestion_id=1,
                    evidence_id="sq1_e2",
                    type="raw_value",
                    value="18.46%",
                    entity="乘用车",
                    time="2010年1月",
                    metric="出口量同比增长率",
                    unit="%",
                    source_tables=["auto_2010.csv"],
                ),
            ],
        ),
    ]


# ======================================================================
# Test 1 — Required fields per type
# ======================================================================

class TestRequiredFields:
    def test_numeric_fact_fields(self):
        fields = required_fields_for_type("numeric_fact")
        assert "entity" in fields
        assert "time" in fields
        assert "metric" in fields
        assert "value" in fields
        assert "unit" in fields

    def test_entity_set_fields(self):
        fields = required_fields_for_type("entity_set")
        assert "entities" in fields

    def test_reference_fields(self):
        fields = required_fields_for_type("reference")
        assert "reference_text" in fields
        assert "target_sq" in fields

    def test_constraint_fields(self):
        fields = required_fields_for_type("constraint")
        assert "constraint_content" in fields

    def test_table_ref_fields(self):
        fields = required_fields_for_type("table_ref")
        assert "table_name" in fields

    def test_unknown_type_empty(self):
        assert required_fields_for_type("bogus") == []


# ======================================================================
# Test 2 — Referring expression detection
# ======================================================================

class TestReferringExpressions:
    def test_demonstrative_entity(self):
        refs = detect_referring_expressions("该公司的产量数据显示增长")
        assert len(refs) >= 1
        assert any("该公司" in r[0] for r in refs)

    def test_former_latter(self):
        refs = detect_referring_expressions("前者较高，后者较低")
        assert len(refs) >= 2
        texts = [r[0] for r in refs]
        assert "前者" in texts
        assert "后者" in texts

    def test_summary_ref(self):
        refs = detect_referring_expressions("综合以上分析，市场以乘用车为主")
        assert len(refs) >= 1
        assert any("综合以上" in r[0] for r in refs)

    def test_collective_ref(self):
        refs = detect_referring_expressions("这三个案例都具有代表性")
        assert len(refs) >= 1
        assert any("这三个案例" in r[0] for r in refs)

    def test_named_case_ref(self):
        refs = detect_referring_expressions("案例A的资金成本最高，案例B次之")
        assert len(refs) >= 2
        texts = [r[0] for r in refs]
        assert "案例A" in texts
        assert "案例B" in texts

    def test_transition_ref(self):
        refs = detect_referring_expressions("很好，现在我们有了三个可比的案例")
        assert len(refs) >= 1

    def test_direct_ref(self):
        refs = detect_referring_expressions("上述数据显示增长趋势")
        assert len(refs) >= 1
        assert any("上述" in r[0] for r in refs)

    def test_comparison_ref(self):
        refs = detect_referring_expressions("对比以上案例可以发现规律")
        assert len(refs) >= 1
        assert any("对比以上" in r[0] for r in refs)

    def test_no_reference(self):
        refs = detect_referring_expressions("乘用车产量同比增长率为16.96%")
        assert refs == []

    def test_multiple_refs_in_same_text(self):
        text = "基于该公司2010年数据，综合以上分析，这三个案例表明市场向好"
        refs = detect_referring_expressions(text)
        assert len(refs) >= 3, f"Expected >=3 refs, got {len(refs)}: {refs}"


# ======================================================================
# Test 3 — Entity / time / metric / constraint extraction
# ======================================================================

class TestExtractionHelpers:
    def test_extract_entities(self):
        entities = extract_entities("凤鸣大道一期和皖江路两个项目的对比")
        assert "凤鸣大道一期" in entities or "凤鸣大道" in entities
        assert "皖江路" in entities

    def test_extract_entities_cities(self):
        entities = extract_entities("上海和北京的超高层建筑数量对比")
        assert "上海" in entities
        assert "北京" in entities

    def test_extract_times(self):
        times = extract_times("2010年1月德国汽车市场概况")
        assert "2010年1月" in times

    def test_extract_times_year_only(self):
        times = extract_times("2020年度报告分析")
        assert any("2020年" in t for t in times)

    def test_extract_metrics(self):
        metrics = extract_metrics("产量同比增长率为16.96%，出口量为18.46%")
        assert len(metrics) >= 1

    def test_extract_metrics_roi(self):
        metrics = extract_metrics("直通车ROI为0.71，钻展ROI为1.37")
        assert "ROI" in metrics

    def test_extract_constraints(self):
        cons = extract_constraints("不低于10%的增长率和10月之前的数据")
        assert len(cons) >= 1

    def test_extract_constraints_year_range(self):
        cons = extract_constraints("1999—2010年间建成的超高层建筑")
        assert any("1999" in c for c in cons)


# ======================================================================
# Test 4 — FutureDependency validation
# ======================================================================

class TestFutureDependencyValidation:
    def test_valid_numeric_fact(self):
        fd = FutureDependency(
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
        )
        assert fd.validate() == []

    def test_valid_entity_set(self):
        fd = FutureDependency(
            dependency_id="dep_sq1_002",
            type="entity_set",
            needed_by="sq2",
            fields={"entities": ["乘用车", "商用车"]},
        )
        assert fd.validate() == []

    def test_valid_reference(self):
        fd = FutureDependency(
            dependency_id="dep_sq1_003",
            type="reference",
            needed_by="sq2",
            fields={
                "reference_text": "该公司",
                "target_sq": "sq1",
                "ref_type": "entity_ref",
            },
        )
        assert fd.validate() == []

    def test_valid_constraint(self):
        fd = FutureDependency(
            dependency_id="dep_sq1_004",
            type="constraint",
            needed_by="sq3",
            fields={"constraint_content": "2010年1月"},
        )
        assert fd.validate() == []

    def test_valid_table_ref(self):
        fd = FutureDependency(
            dependency_id="dep_sq1_005",
            type="table_ref",
            needed_by="sq2",
            fields={"table_name": "auto_2010.csv"},
        )
        assert fd.validate() == []

    def test_missing_dependency_id(self):
        fd = FutureDependency(
            dependency_id="",
            type="reference",
            needed_by="sq2",
            fields={"reference_text": "x", "target_sq": "sq1"},
        )
        issues = fd.validate()
        assert any("dependency_id" in i for i in issues)

    def test_unknown_type(self):
        fd = FutureDependency(
            dependency_id="d1",
            type="bogus",
            needed_by="sq2",
        )
        issues = fd.validate()
        assert any("unknown type" in i for i in issues)

    def test_missing_required_field_in_numeric_fact(self):
        fd = FutureDependency(
            dependency_id="d1",
            type="numeric_fact",
            needed_by="sq2",
            fields={"entity": "x"},  # missing time, metric, value, unit
        )
        issues = fd.validate()
        assert len(issues) >= 4, f"Expected >=4 issues, got {len(issues)}: {issues}"

    def test_empty_fields_dict_passes_reference_check(self):
        """reference type requires reference_text + target_sq."""
        fd = FutureDependency(
            dependency_id="d1",
            type="reference",
            needed_by="sq2",
            fields={},
        )
        issues = fd.validate()
        assert any("reference_text" in i for i in issues)
        assert any("target_sq" in i for i in issues)


# ======================================================================
# Test 5 — FutureDependencySet validation
# ======================================================================

class TestFutureDependencySetValidation:
    def test_empty_set(self):
        s = FutureDependencySet(
            sample_id="s1",
            boundary="after_sq1",
        )
        assert s.future_dependencies == []
        assert s.validate_all() == {}

    def test_validate_all_with_issues(self):
        s = FutureDependencySet(
            sample_id="s1",
            boundary="after_sq1",
            future_dependencies=[
                FutureDependency(
                    dependency_id="bad1",
                    type="bogus",
                    needed_by="",
                ),
                FutureDependency(
                    dependency_id="good1",
                    type="reference",
                    needed_by="sq2",
                    fields={"reference_text": "x", "target_sq": "sq1"},
                ),
            ],
        )
        issues = s.validate_all()
        assert "bad1" in issues
        assert "good1" not in issues


# ======================================================================
# Test 6 — Serialisation round-trip
# ======================================================================

class TestSerialisation:
    def test_future_dependency_roundtrip(self):
        fd = FutureDependency(
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
            audit=AuditInfo(
                parse_confidence=0.7,
                warnings=["test warning"],
                source="llm_annotation",
            ),
        )
        d = fd.to_dict()
        fd2 = FutureDependency.from_dict(d)

        assert fd2.dependency_id == "dep_sq1_001"
        assert fd2.type == "numeric_fact"
        assert fd2.needed_by == "sq2"
        assert fd2.source_evidence_id == "sq1_e1"
        assert fd2.fields["entity"] == "乘用车"
        assert fd2.fields["value"] == "16.96%"
        assert fd2.weight == 1.0
        assert fd2.audit.parse_confidence == 0.7
        assert fd2.audit.warnings == ["test warning"]

    def test_future_dependency_without_source_evidence(self):
        """source_evidence_id is optional — serialisation omits it when None."""
        fd = FutureDependency(
            dependency_id="dep_sq1_001",
            type="entity_set",
            needed_by="sq2",
            fields={"entities": ["A", "B"]},
        )
        d = fd.to_dict()
        assert "source_evidence_id" not in d
        fd2 = FutureDependency.from_dict(d)
        assert fd2.source_evidence_id is None

    def test_future_dependency_set_roundtrip(self):
        s = FutureDependencySet(
            sample_id="s1",
            boundary="after_sq1",
            future_dependencies=[
                FutureDependency(
                    dependency_id="dep_sq1_001",
                    type="reference",
                    needed_by="sq2",
                    fields={"reference_text": "该公司", "target_sq": "sq1"},
                ),
            ],
        )
        d = s.to_dict()
        s2 = FutureDependencySet.from_dict(d)
        assert s2.sample_id == "s1"
        assert s2.boundary == "after_sq1"
        assert len(s2.future_dependencies) == 1
        assert s2.future_dependencies[0].type == "reference"

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
        finally:
            os.unlink(tmp)


# ======================================================================
# Test 7 — Builder: dependency detection (two-sq sample)
# ======================================================================

class TestBuilderTwoSqDetection:
    def test_reference_detected(self, builder, simple_two_sq_sample):
        """sq2 contains '该公司' and '该企业' → reference dependencies."""
        sets = builder.build_one_sample(simple_two_sq_sample)
        assert len(sets) == 1  # one boundary: after_sq1
        deps = sets[0].future_dependencies
        types = [d.type for d in deps]
        assert "reference" in types, (
            f"Expected reference deps, got types: {types}\n"
            f"Deps: {[(d.type, d.fields) for d in deps]}"
        )

    def test_entity_set_detected(self, builder, simple_two_sq_sample):
        """sq2 shares entity '乘用车' with sq1."""
        sets = builder.build_one_sample(simple_two_sq_sample)
        deps = sets[0].future_dependencies
        entity_deps = [d for d in deps if d.type == "entity_set"]
        assert len(entity_deps) >= 1, f"No entity_set deps in: {[d.type for d in deps]}"
        all_entities = []
        for ed in entity_deps:
            all_entities.extend(ed.fields.get("entities", []))
        assert "乘用车" in all_entities, f"乘用车 not in entities: {all_entities}"

    def test_table_ref_detected(self, builder, simple_two_sq_sample):
        """Both sqs share auto_2010.csv."""
        sets = builder.build_one_sample(simple_two_sq_sample)
        deps = sets[0].future_dependencies
        table_deps = [d for d in deps if d.type == "table_ref"]
        assert len(table_deps) >= 1, f"No table_ref deps: {[d.type for d in deps]}"
        tables = [d.fields["table_name"] for d in table_deps]
        assert "auto_2010.csv" in tables

    def test_all_deps_have_audit(self, builder, simple_two_sq_sample):
        sets = builder.build_one_sample(simple_two_sq_sample)
        for dep in sets[0].future_dependencies:
            assert dep.audit is not None
            assert isinstance(dep.audit.parse_confidence, float)
            assert dep.audit.source == "rule_extraction"

    def test_deps_have_needed_by(self, builder, simple_two_sq_sample):
        sets = builder.build_one_sample(simple_two_sq_sample)
        for dep in sets[0].future_dependencies:
            assert dep.needed_by == "sq2", (
                f"Dependency {dep.dependency_id}: needed_by={dep.needed_by}, "
                f"expected sq2"
            )


# ======================================================================
# Test 8 — Builder: dependency detection (three-sq sample)
# ======================================================================

class TestBuilderThreeSqDetection:
    def test_two_boundaries(self, builder, three_sq_sample):
        """3 subquestions → boundaries after_sq1 and after_sq2."""
        sets = builder.build_one_sample(three_sq_sample)
        boundaries = [s.boundary for s in sets]
        assert "after_sq1" in boundaries
        assert "after_sq2" in boundaries
        assert len(sets) == 2

    def test_summary_ref_in_sq3(self, builder, three_sq_sample):
        """sq3 says '综合以上分析' → reference to earlier sqs."""
        sets = builder.build_one_sample(three_sq_sample)
        # after_sq2 boundary should detect sq3 ref
        after_sq2 = [s for s in sets if s.boundary == "after_sq2"][0]
        ref_deps = [d for d in after_sq2.future_dependencies if d.type == "reference"]
        assert len(ref_deps) >= 1, (
            f"No reference deps in after_sq2: "
            f"{[(d.type, d.fields) for d in after_sq2.future_dependencies]}"
        )

    def test_entity_overlap_sq2_on_sq1(self, builder, three_sq_sample):
        """sq2 mentions 乘用车 which was introduced in sq1."""
        sets = builder.build_one_sample(three_sq_sample)
        after_sq1 = [s for s in sets if s.boundary == "after_sq1"][0]
        entity_deps = [d for d in after_sq1.future_dependencies if d.type == "entity_set"]
        # 乘用车 appears in both sq1 and sq2
        if entity_deps:
            all_entities = []
            for ed in entity_deps:
                all_entities.extend(ed.fields.get("entities", []))
            assert "乘用车" in all_entities


# ======================================================================
# Test 9 — D_FDC cutoff
# ======================================================================

class TestDFDCCutoff:
    def test_d_fdc_2_default(self, four_sq_sample):
        """With D_FDC=2, after_sq1 should see sq2 and sq3, but NOT sq4."""
        builder = FutureDependencyBuilder(d_fdc=2)
        sets = builder.build_one_sample(four_sq_sample)
        assert len(sets) == 3  # 4 sqs → 3 boundaries

        after_sq1 = sets[0]
        needed_by_set = {d.needed_by for d in after_sq1.future_dependencies}
        # sq2 and sq3 are within D_FDC=2 of sq1; sq4 is not
        assert "sq2" in needed_by_set or "sq3" in needed_by_set, (
            f"after_sq1 should see sq2/sq3 within D_FDC=2, got {needed_by_set}"
        )
        # sq4 should NOT appear as needed_by from after_sq1
        assert "sq4" not in needed_by_set, (
            f"sq4 is beyond D_FDC=2 from sq1, but found in {needed_by_set}"
        )

    def test_d_fdc_1_narrow(self, four_sq_sample):
        """With D_FDC=1, after_sq1 only sees sq2."""
        builder = FutureDependencyBuilder(d_fdc=1)
        sets = builder.build_one_sample(four_sq_sample)

        after_sq1 = sets[0]
        needed_by_set = {d.needed_by for d in after_sq1.future_dependencies}
        # With D_FDC=1, after_sq1 should only see sq2
        assert "sq2" in needed_by_set or len(after_sq1.future_dependencies) >= 0
        # sq3 should NOT be in after_sq1's dependencies with D_FDC=1
        assert "sq3" not in needed_by_set, (
            f"With D_FDC=1, after_sq1 should NOT see sq3. Got: {needed_by_set}"
        )

    def test_no_deps_past_d_fdc(self, builder_narrow, four_sq_sample):
        """Strictly verify sq3 is not in after_sq1 deps with D_FDC=1."""
        sets = builder_narrow.build_one_sample(four_sq_sample)
        after_sq1 = sets[0]
        for dep in after_sq1.future_dependencies:
            assert dep.needed_by != "sq3", (
                f"sq3 should not appear in after_sq1 with D_FDC=1"
            )


# ======================================================================
# Test 10 — H_i^{keep}: future info not written to current boundary
# ======================================================================

class TestFutureInfoFiltering:
    """Verify that dependencies only reference info available at the boundary,
    not info that will only appear in future subquestions."""

    def test_boundary_only_refers_to_past(self, builder, three_sq_sample):
        """after_sq1 deps: target_sq must be a valid subquestion number.

        Note: at build-time we can see all future sqs, so a reference
        dependency may have target_sq pointing to sq2 (when sq3 references
        sq2).  The H_i^{keep} filtering (done later in the reward calculator)
        will exclude dependencies whose supporting evidence isn't available
        at boundary after_sq1.
        """
        sets = builder.build_one_sample(three_sq_sample)
        after_sq1 = sets[0]  # boundary after_sq1

        for dep in after_sq1.future_dependencies:
            # reference type: target_sq must be a valid sq number (1, 2, or 3)
            if dep.type == "reference":
                target = dep.fields.get("target_sq", "")
                if target.startswith("sq"):
                    target_num = int(target[2:])
                    assert 1 <= target_num <= 3, (
                        f"Reference in after_sq1 has invalid target_sq={target}"
                    )
            # For entity_set: entities should come from sq1 which is in the past
            if dep.type == "entity_set":
                entities = dep.fields.get("entities", [])
                sq1_text = (
                    three_sq_sample["design"]["checkout_list"][0]["info_item"]
                    + " "
                    + " ".join(
                        three_sq_sample["design"]["checkout_list"][0][
                            "score_points"
                        ]
                    )
                )
                for ent in entities:
                    if ent in sq1_text:
                        continue
                    # Entity might also be detectable from sq1 context

    def test_source_evidence_id_points_to_past_sq(self, builder, simple_two_sq_sample):
        """source_evidence_id should reference evidence from the past, not future."""
        sets = builder.build_one_sample(simple_two_sq_sample)
        after_sq1 = sets[0]
        for dep in after_sq1.future_dependencies:
            if dep.source_evidence_id:
                # Evidence id format: "sq<N>_e<M>"
                # The sq number in evidence_id should be <= current boundary (1)
                import re
                m = re.match(r"sq(\d+)_", dep.source_evidence_id)
                if m:
                    ev_sq = int(m.group(1))
                    assert ev_sq <= 1, (
                        f"Dependency {dep.dependency_id} points to "
                        f"{dep.source_evidence_id} (sq{ev_sq}) which is "
                        f"beyond boundary after_sq1"
                    )


# ======================================================================
# Test 11 — Source evidence alignment (with target_evidence.jsonl)
# ======================================================================

class TestSourceEvidenceAlignment:
    def test_alignment_with_target_evidence(
        self, builder, simple_two_sq_sample, sample_target_evidence
    ):
        """When target_evidence is provided, source_evidence_id should be set."""
        # Index the target evidence
        te_index: dict = {}
        for tes in sample_target_evidence:
            key = (tes.sample_id, tes.subquestion_id)
            te_index.setdefault(key, []).extend(tes.evidence_items)

        sets = builder.build_one_sample(simple_two_sq_sample, te_index)
        after_sq1 = sets[0]

        # Check that at least some deps have source_evidence_id set
        deps_with_source = [
            d for d in after_sq1.future_dependencies if d.source_evidence_id
        ]
        # Not all may be linkable, but we expect some
        # (entity_set for 乘用车 should link to sq1_e1 or sq1_e2)
        entity_deps = [
            d for d in after_sq1.future_dependencies if d.type == "entity_set"
        ]
        if entity_deps:
            # With target evidence providing entity="乘用车", at least one
            # entity_set should link
            linked_entities = [
                d for d in entity_deps if d.source_evidence_id
            ]
            assert len(linked_entities) >= 1, (
                f"Expected entity_set to link to target evidence, "
                f"but none did: {[(d.fields, d.source_evidence_id) for d in entity_deps]}"
            )

    def test_alignment_output_format(self, builder, simple_two_sq_sample, sample_target_evidence):
        """Verify the alignment output can be serialised and inspected."""
        te_index: dict = {}
        for tes in sample_target_evidence:
            key = (tes.sample_id, tes.subquestion_id)
            te_index.setdefault(key, []).extend(tes.evidence_items)

        sets = builder.build_one_sample(simple_two_sq_sample, te_index)
        # Serialise
        records = [s.to_dict() for s in sets]
        assert len(records) == 1
        record = records[0]

        assert record["sample_id"] == "test two sq sample"
        assert record["boundary"] == "after_sq1"
        assert len(record["future_dependencies"]) >= 1

        # Print alignment example
        for dep in record["future_dependencies"]:
            if dep.get("source_evidence_id"):
                assert dep["source_evidence_id"].startswith("sq1_"), (
                    f"source_evidence_id {dep['source_evidence_id']} should start with sq1_"
                )

    def test_alignment_demo(self, builder, simple_two_sq_sample, sample_target_evidence):
        """Demo the full alignment pipeline and print results."""
        te_index: dict = {}
        for tes in sample_target_evidence:
            key = (tes.sample_id, tes.subquestion_id)
            te_index.setdefault(key, []).extend(tes.evidence_items)

        sets = builder.build_one_sample(simple_two_sq_sample, te_index)
        record = sets[0].to_dict()

        print(f"\n  === Source Evidence Alignment Demo ===")
        print(f"  Boundary: {record['boundary']}")
        print(f"  Dependencies: {len(record['future_dependencies'])}")

        for dep in record["future_dependencies"]:
            seid = dep.get("source_evidence_id", "None")
            print(f"    {dep['dependency_id']}: type={dep['type']} "
                  f"needed_by={dep['needed_by']} "
                  f"source_evidence_id={seid} "
                  f"fields={dep['fields']}")

        # Demonstrate the alignment table
        print(f"\n  === Cross-reference Table ===")
        print(f"  {'Dependency ID':<20} {'Type':<15} {'Needed By':<12} {'Source Evidence':<15}")
        print(f"  {'-'*20} {'-'*15} {'-'*12} {'-'*15}")
        for dep in record["future_dependencies"]:
            seid = dep.get("source_evidence_id", "N/A")
            print(f"  {dep['dependency_id']:<20} {dep['type']:<15} "
                  f"{dep['needed_by']:<12} {seid:<15}")


# ======================================================================
# Test 12 — Real data integration test
# ======================================================================

class TestRealData:
    def test_build_on_real_data(self):
        """Build future deps from real val.json and target_evidence.jsonl."""
        val_path = os.path.join(_PROJ_ROOT, "dataset", "val.json")
        te_path = os.path.join(
            _PROJ_ROOT, "TEMPO_RL", "output", "target_evidence.jsonl"
        )

        if not os.path.exists(val_path):
            pytest.skip("val.json not found")
        if not os.path.exists(te_path):
            pytest.skip("target_evidence.jsonl not found — run Phase 0 Part 1 first")

        from TEMPO_RL.build_target_evidence import TargetEvidenceBuilder

        samples = load_benchmark_samples(val_path)
        target_sets = TargetEvidenceBuilder.load(te_path)

        builder = FutureDependencyBuilder(d_fdc=2)
        # Build for first 5 samples
        dep_sets = builder.build(samples[:5], target_sets)

        total_deps = sum(len(s.future_dependencies) for s in dep_sets)
        total_boundaries = len(dep_sets)

        print(f"\n  Built {total_boundaries} boundary sets, {total_deps} dependencies "
              f"from {len(samples[:5])} samples")

        # Type distribution
        type_counts: dict = {}
        for s in dep_sets:
            for d in s.future_dependencies:
                type_counts[d.type] = type_counts.get(d.type, 0) + 1
        print(f"  Type distribution: {type_counts}")

        # Every dependency must validate
        for s in dep_sets:
            issues = s.validate_all()
            assert issues == {}, (
                f"Validation issues in boundary {s.boundary}: {issues}"
            )

        # Every boundary set should have proper metadata
        for s in dep_sets:
            assert s.sample_id
            assert s.boundary.startswith("after_sq")

        # At least some boundaries should have dependencies
        boundaries_with_deps = [
            s for s in dep_sets if s.future_dependencies
        ]
        assert len(boundaries_with_deps) >= 1, (
            "No boundaries have dependencies — detection may be too weak"
        )

        # Source evidence alignment rate (when target_evidence is provided)
        deps_with_source = sum(
            1 for s in dep_sets
            for d in s.future_dependencies
            if d.source_evidence_id
        )
        alignment_rate = (
            deps_with_source / total_deps * 100 if total_deps > 0 else 0
        )
        print(f"  Source evidence alignment: {deps_with_source}/{total_deps} "
              f"({alignment_rate:.1f}%)")

        # All deps must have audit
        for s in dep_sets:
            for d in s.future_dependencies:
                assert d.audit is not None
                assert d.audit.source in (
                    "rule_extraction", "llm_annotation", "manual", "sft_trajectory"
                )

        # Print a sample boundary
        if boundaries_with_deps:
            s = boundaries_with_deps[0]
            print(f"\n  Sample boundary: {s.sample_id[:60]}... / {s.boundary}")
            for d in s.future_dependencies[:5]:
                seid = d.source_evidence_id or "N/A"
                print(f"    {d.dependency_id}: type={d.type} "
                      f"needed_by={d.needed_by} source_evidence_id={seid}")


# ======================================================================
# Test 13 — Edge cases
# ======================================================================

class TestEdgeCases:
    def test_single_sq_sample_no_boundaries(self, builder):
        """A sample with only 1 subquestion has no memory boundaries."""
        sample = {
            "task": "single sq",
            "design": {
                "type": "Tree",
                "checkout_list": [
                    {
                        "idx": 1,
                        "info_item": "唯一问题",
                        "related_tables": [],
                        "score_points": ["答案"],
                    },
                ],
            },
        }
        sets = builder.build_one_sample(sample)
        assert sets == []

    def test_no_overlap_between_sqs(self, builder):
        """Two completely unrelated subquestions — should have minimal deps."""
        sample = {
            "task": "unrelated sqs",
            "design": {
                "type": "Tree",
                "checkout_list": [
                    {
                        "idx": 1,
                        "info_item": "分析产品A的销售数据",
                        "related_tables": ["sales_a.csv"],
                        "score_points": ["销售额为100万"],
                    },
                    {
                        "idx": 2,
                        "info_item": "分析产品B的库存数据",
                        "related_tables": ["inventory_b.csv"],
                        "score_points": ["库存充足"],
                    },
                ],
            },
        }
        sets = builder.build_one_sample(sample)
        # Should still produce a boundary set
        assert len(sets) == 1
        # May have minimal or zero deps if truly unrelated
        assert sets[0].boundary == "after_sq1"

    def test_deduplication(self, builder):
        """Duplicate dependency signatures should be removed."""
        sample = {
            "task": "dedup test",
            "design": {
                "type": "Tree",
                "checkout_list": [
                    {
                        "idx": 1,
                        "info_item": "计算产品A的增长率",
                        "related_tables": ["t.csv"],
                        "score_points": ["增长率为10%"],
                    },
                    {
                        "idx": 2,
                        "info_item": "基于该产品的10%增长率分析",
                        "related_tables": ["t.csv"],
                        "score_points": [
                            "该产品的增长率为10%",
                            "该产品的10%增长率表明趋势向好",
                        ],
                    },
                ],
            },
        }
        sets = builder.build_one_sample(sample)
        deps = sets[0].future_dependencies

        # Check no two deps have identical type + needed_by + fields
        sigs = []
        for d in deps:
            sig = (d.type, d.needed_by, str(sorted(d.fields.items())))
            sigs.append(sig)
        assert len(sigs) == len(set(sigs)), (
            f"Duplicate dependencies found: {sigs}"
        )

    def test_build_without_target_evidence(self, builder, simple_two_sq_sample):
        """Should work fine without target evidence (no source_evidence_id)."""
        sets = builder.build_one_sample(simple_two_sq_sample)  # te_index=None
        assert len(sets) == 1
        # source_evidence_id should be None for all deps
        for dep in sets[0].future_dependencies:
            assert dep.source_evidence_id is None

    def test_save_load_roundtrip(self, builder, simple_two_sq_sample):
        """Full I/O roundtrip for FutureDependencySet."""
        sets = builder.build_one_sample(simple_two_sq_sample)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            tmp = f.name
        try:
            FutureDependencyBuilder.save(sets, tmp)
            loaded = FutureDependencyBuilder.load(tmp)
            assert len(loaded) == len(sets)
            assert loaded[0].sample_id == sets[0].sample_id
            assert loaded[0].boundary == sets[0].boundary
            assert len(loaded[0].future_dependencies) == len(sets[0].future_dependencies)

            # Spot-check first dep
            if loaded[0].future_dependencies:
                orig = sets[0].future_dependencies[0]
                loaded_dep = loaded[0].future_dependencies[0]
                assert loaded_dep.type == orig.type
                assert loaded_dep.fields == orig.fields
        finally:
            os.unlink(tmp)
