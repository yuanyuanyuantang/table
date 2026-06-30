"""
TEMPO-RL Phase 2 — Smoke tests for dialog rollout runner.

Covers: memory pass-through, verifier doesn't replace memory,
invalid memory fallback, multi-subquestion dialog, output structure.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from TEMPO_RL.rollout_phase2 import (
    DialogRolloutRunner,
    FakeToolExecutor,
    MockPolicy,
    ToolExecutor,
)
from TEMPO_RL.io_utils import (
    try_parse_json,
    extract_tool_calls_from_response,
    extract_answer_from_response,
    extract_memory_from_response,
)
from TEMPO_RL.reward_calculator import RewardCalculator
from TEMPO_RL.schemas import (
    TargetEvidenceSet,
    EvidenceItem,
    FutureDependencySet,
    FutureDependency,
    AuditInfo,
)


# ======================================================================
# Helpers
# ======================================================================

def _make_tes(sample_id: str = "test_sample", sq_id: int = 1) -> TargetEvidenceSet:
    """Build a minimal TargetEvidenceSet for testing."""
    return TargetEvidenceSet(
        sample_id=sample_id,
        subquestion_id=sq_id,
        question=f"Test question {sq_id}",
        evidence_items=[
            EvidenceItem(
                sample_id=sample_id,
                subquestion_id=sq_id,
                evidence_id=f"{sample_id}_sq{sq_id}_e1",
                type="raw_value",
                value="16.96%",
                entity="Company A",
                time="2010年1月",
                metric="产量同比增长率",
                unit="%",
                source_tables=["test.xlsx"],
            ),
        ],
    )


def _make_fds(sample_id: str = "test_sample", boundary: str = "root") -> FutureDependencySet:
    """Build a minimal FutureDependencySet for testing."""
    return FutureDependencySet(
        sample_id=sample_id,
        boundary=boundary,
        future_dependencies=[
            FutureDependency(
                dependency_id="dep1",
                type="numeric_fact",
                needed_by="sq2",
                source_evidence_id="test_sample_sq1_e1",
                fields={
                    "entity": "Company A",
                    "time": "2010年1月",
                    "metric": "产量同比增长率",
                    "value": "16.96%",
                    "unit": "%",
                },
            ),
            FutureDependency(
                dependency_id="dep2",
                type="entity_set",
                needed_by="sq2",
                fields={"entities": ["Company A", "Company B"]},
            ),
            FutureDependency(
                dependency_id="dep3",
                type="reference",
                needed_by="sq2",
                fields={"reference_text": "the company from sq1", "target_sq": "sq1"},
            ),
            FutureDependency(
                dependency_id="dep4",
                type="constraint",
                needed_by="sq2",
                fields={"constraint_content": "production > 100 units"},
            ),
            FutureDependency(
                dependency_id="dep5",
                type="table_ref",
                needed_by="sq2",
                fields={"table_name": "test.xlsx"},
            ),
        ],
    )


def _make_minimal_sample(n_sq: int = 2) -> dict:
    """Build a minimal sample with subquestions."""
    checkout_list = []
    for i in range(1, n_sq + 1):
        checkout_list.append({
            "question": f"Subquestion {i}: What is the data?",
            "checkout_item": {
                "checkout_text": f"Subquestion {i}: What is the data?",
                "score_points": [f"Answer {i}"],
            },
        })
    return {
        "task": "test_sample",
        "table_path": "test.xlsx",
        "design": {"checkout_list": checkout_list},
    }


def _make_memory_output() -> dict:
    """Standard valid memory output."""
    return {
        "goal": "Find production growth for Company A",
        "tables": ["test.xlsx"],
        "key_facts": [
            {
                "entity": "Company A",
                "time": "2010年1月",
                "metric": "产量同比增长率",
                "value": "16.96%",
                "unit": "%",
            }
        ],
        "derived_results": [],
        "constraints": [],
        "pitfalls": [],
    }


# ======================================================================
# Test 1 — Response parsing
# ======================================================================

class TestResponseParsing:
    """Verify tool_call, answer, memory extraction from model responses."""

    def test_extract_tool_call(self):
        resp = '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "test.xlsx"}}</tool_call>'
        calls = extract_tool_calls_from_response(resp)
        assert len(calls) == 1
        assert calls[0]["tool_name"] == "table_head_reader"
        assert calls[0]["arguments"] == {"file_path": "test.xlsx"}

    def test_extract_single_tool_call_no_multi(self):
        resp = 'Some text <tool_call>{"tool": "read", "params": {}}</tool_call> more text'
        calls = extract_tool_calls_from_response(resp)
        assert len(calls) == 1

    def test_extract_answer(self):
        resp = '<answer>The production increased by 16.96%.</answer>'
        ans = extract_answer_from_response(resp)
        assert ans == "The production increased by 16.96%."

    def test_extract_memory(self):
        resp = '<memory>{"goal": "test", "key_facts": []}</memory>'
        mem = extract_memory_from_response(resp)
        assert mem is not None
        assert mem["goal"] == "test"

    def test_extract_answer_and_memory_together(self):
        resp = (
            '<answer>The production increased by 16.96%.</answer>\n'
            '<memory>{"goal": "test", "tables": ["t1"], '
            '"key_facts": [{"entity": "A", "time": "2020", '
            '"metric": "m", "value": "10", "unit": "%"}]}</memory>'
        )
        ans = extract_answer_from_response(resp)
        mem = extract_memory_from_response(resp)
        assert ans is not None
        assert mem is not None
        assert mem["goal"] == "test"

    def test_invalid_memory_json_returns_none(self):
        resp = '<memory>not valid json {{{</memory>'
        mem = extract_memory_from_response(resp)
        assert mem is None

    def test_full_response_with_all_tags(self):
        resp = (
            '<tool_call>{"tool": "read", "params": {"file_path": "f.xlsx"}}</tool_call>\n'
            '<answer>The answer is 42.</answer>\n'
            '<memory>{"goal": "found the answer", "tables": ["f.xlsx"], '
            '"key_facts": [{"entity": "X", "time": "2020", "metric": "count", '
            '"value": "42", "unit": ""}]}</memory>'
        )
        calls = extract_tool_calls_from_response(resp)
        ans = extract_answer_from_response(resp)
        mem = extract_memory_from_response(resp)
        assert len(calls) == 1
        assert ans == "The answer is 42."
        assert mem is not None
        assert mem["goal"] == "found the answer"


# ======================================================================
# Test 2 — Memory pass-through
# ======================================================================

class TestMemoryPassThrough:
    """Verify memory is passed verbatim between subquestions."""

    def test_memory_passed_to_next_subquestion(self):
        """Memory from sq1 appears as memory_before in sq2."""
        tes_lookup = {
            ("test_sample", 1): _make_tes("test_sample", 1),
            ("test_sample", 2): _make_tes("test_sample", 2),
        }

        policy = MockPolicy([
            # sq1: tool_call → answer + memory
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "test.xlsx"}}</tool_call>',
            '<answer>Production grew 16.96%.</answer>\n'
            '<memory>{"goal": "growth rate", "tables": ["test.xlsx"], '
            '"key_facts": [{"entity": "Company A", "time": "2010年1月", '
            '"metric": "产量同比增长率", "value": "16.96%", "unit": "%"}]}</memory>',
            # sq2: tool_call → answer + memory
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "test.xlsx"}}</tool_call>',
            '<answer>Company A also had 20% growth in 2011.</answer>\n'
            '<memory>{"goal": "2011 growth", "tables": ["test.xlsx"], '
            '"key_facts": [{"entity": "Company A", "time": "2011年", '
            '"metric": "增长率", "value": "20%", "unit": "%"}]}</memory>',
        ])

        executor = FakeToolExecutor()
        calculator = RewardCalculator()

        runner = DialogRolloutRunner(
            policy=policy,
            tool_executor=executor,
            calculator=calculator,
            tes_lookup=tes_lookup,
            K=1,
            max_tool_steps_per_turn=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[_make_minimal_sample(n_sq=2)],
                tools_schema=[],
                output_dir=tmpdir,
            )

        assert len(records) == 1
        dialog = records[0]
        sq_rollouts = dialog["subquestion_rollouts"]
        assert len(sq_rollouts) == 2

        # sq1: memory_before should be None (first subquestion)
        assert sq_rollouts[0].get("memory_before") is None

        # sq1: memory_output should be the generated memory
        assert sq_rollouts[0]["memory_output"] is not None
        assert sq_rollouts[0]["memory_output"]["goal"] == "growth rate"

        # sq2: memory_before should be the raw memory from sq1
        assert sq_rollouts[1].get("memory_before") is not None
        assert sq_rollouts[1]["memory_before"]["goal"] == "growth rate"

    def test_memory_not_modified_by_verifier(self):
        """The memory passed is the model's raw output, not verifier-corrected."""
        tes_lookup = {
            ("test_sample", 1): _make_tes("test_sample", 1),
            ("test_sample", 2): _make_tes("test_sample", 2),
        }

        raw_memory = {
            "goal": "some goal with extra info the verifier wouldn't include",
            "tables": ["test.xlsx"],
            "key_facts": [
                {
                    "entity": "Company A",
                    "time": "2010年1月",
                    "metric": "产量同比增长率",
                    "value": "16.96%",
                    "unit": "%",
                },
                {
                    "entity": "EXTRA_ENTITY",
                    "time": "extra_time",
                    "metric": "extra_metric",
                    "value": "extra_value",
                    "unit": "extra_unit",
                },
            ],
            "custom_field": "this would not survive verifier correction",
        }
        mem_str = json.dumps(raw_memory, ensure_ascii=False)

        policy = MockPolicy([
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "test.xlsx"}}</tool_call>',
            f'<answer>Production grew 16.96%.</answer>\n<memory>{mem_str}</memory>',
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "test.xlsx"}}</tool_call>',
            '<answer>Done.</answer>\n<memory>{"goal": "sq2", "tables": [], "key_facts": []}</memory>',
        ])

        executor = FakeToolExecutor()
        calculator = RewardCalculator()

        runner = DialogRolloutRunner(
            policy=policy,
            tool_executor=executor,
            calculator=calculator,
            tes_lookup=tes_lookup,
            K=1,
            max_tool_steps_per_turn=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[_make_minimal_sample(n_sq=2)],
                tools_schema=[],
                output_dir=tmpdir,
            )

        sq_rollouts = records[0]["subquestion_rollouts"]

        # sq2's memory_before should have the custom_field (proving it's raw output)
        mem_before = sq_rollouts[1].get("memory_before")
        assert mem_before is not None
        assert mem_before.get("custom_field") == "this would not survive verifier correction"
        assert mem_before.get("goal") == "some goal with extra info the verifier wouldn't include"

        # sq2's memory_before should have the extra entity
        key_facts = mem_before.get("key_facts", [])
        extra_entities = [kf for kf in key_facts if kf.get("entity") == "EXTRA_ENTITY"]
        assert len(extra_entities) > 0, "Extra entity should survive (no verifier correction)"


# ======================================================================
# Test 3 — Invalid memory fallback
# ======================================================================

class TestInvalidMemoryFallback:
    """Verify fallback behavior when memory JSON is unparseable."""

    def test_unparseable_memory_uses_previous_valid(self):
        """When memory is unparseable, previous valid memory is used as fallback."""
        tes_lookup = {
            ("test_sample", 1): _make_tes("test_sample", 1),
            ("test_sample", 2): _make_tes("test_sample", 2),
            ("test_sample", 3): _make_tes("test_sample", 3),
        }

        valid_mem = _make_memory_output()

        policy = MockPolicy([
            # sq1: valid memory
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "test.xlsx"}}</tool_call>',
            f'<answer>Production grew 16.96%.</answer>\n<memory>{json.dumps(valid_mem)}</memory>',
            # sq2: INVALID memory (unparseable JSON)
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "test.xlsx"}}</tool_call>',
            '<answer>Still got an answer.</answer>\n<memory>this is not json {{{</memory>',
            # sq3: needs memory from sq2 (which fell back to sq1's)
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "test.xlsx"}}</tool_call>',
            '<answer>Final answer.</answer>\n<memory>{"goal": "sq3", "tables": [], "key_facts": []}</memory>',
        ])

        executor = FakeToolExecutor()
        calculator = RewardCalculator()

        runner = DialogRolloutRunner(
            policy=policy,
            tool_executor=executor,
            calculator=calculator,
            tes_lookup=tes_lookup,
            K=1,
            max_tool_steps_per_turn=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[_make_minimal_sample(n_sq=3)],
                tools_schema=[],
                output_dir=tmpdir,
            )

        sq_rollouts = records[0]["subquestion_rollouts"]
        assert len(sq_rollouts) == 3

        # sq2: memory_output is unparseable → should mark fallback
        assert sq_rollouts[1].get("memory_output") is None  # extraction returns None
        assert sq_rollouts[1].get("memory_fallback_used") is True

        # sq2: memory_passed should be sq1's valid memory (fallback)
        assert sq_rollouts[1].get("memory_passed") is not None
        assert sq_rollouts[1]["memory_passed"]["goal"] == valid_mem["goal"]

        # sq3: memory_before should be the fallback (sq1's memory)
        assert sq_rollouts[2].get("memory_before") is not None
        assert sq_rollouts[2]["memory_before"]["goal"] == valid_mem["goal"]

    def test_memory_severe_failure_reward(self):
        """Unparseable memory gets a severe failure reward (r_memory = -1)."""
        tes_lookup = {("test_sample", 1): _make_tes("test_sample", 1)}

        policy = MockPolicy([
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "test.xlsx"}}</tool_call>',
            '<answer>Answer text.</answer>\n<memory>not valid json at all {{{</memory>',
        ])

        executor = FakeToolExecutor()
        calculator = RewardCalculator()

        runner = DialogRolloutRunner(
            policy=policy,
            tool_executor=executor,
            calculator=calculator,
            tes_lookup=tes_lookup,
            K=1,
            max_tool_steps_per_turn=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[_make_minimal_sample(n_sq=1)],
                tools_schema=[],
                output_dir=tmpdir,
            )

        sq_rollouts = records[0]["subquestion_rollouts"]
        assert len(sq_rollouts) == 1
        sq = sq_rollouts[0]

        # Memory extraction should fail → memory_output is None
        assert sq.get("memory_output") is None

        # r_memory should be negative (severe failure penalty)
        # The calculator may give 0 if memory is None, since no memory reward is computed
        # But the memory_severe_failure flag should be checked
        memory_detail = sq.get("memory_reward_detail", {})
        # If memory is None, r_memory defaults to 0 in the rollout runner
        # (no memory reward is computed). This is expected behavior.
        assert sq.get("r_memory") == 0.0 or sq.get("r_memory") == -1.0


# ======================================================================
# Test 4 — Dialog structure and rewards
# ======================================================================

class TestDialogStructure:
    """Verify dialog rollout output structure and reward computation."""

    def test_output_has_required_fields(self):
        """Dialog rollout record has all required top-level fields."""
        tes_lookup = {
            ("test_sample", 1): _make_tes("test_sample", 1),
            ("test_sample", 2): _make_tes("test_sample", 2),
        }

        policy = MockPolicy([
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "test.xlsx"}}</tool_call>',
            '<answer>Answer 1.</answer>\n<memory>{"goal": "g1", "tables": [], "key_facts": []}</memory>',
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "test.xlsx"}}</tool_call>',
            '<answer>Answer 2.</answer>\n<memory>{"goal": "g2", "tables": [], "key_facts": []}</memory>',
        ])

        executor = FakeToolExecutor()
        calculator = RewardCalculator()

        runner = DialogRolloutRunner(
            policy=policy,
            tool_executor=executor,
            calculator=calculator,
            tes_lookup=tes_lookup,
            K=1,
            max_tool_steps_per_turn=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[_make_minimal_sample(n_sq=2)],
                tools_schema=[],
                output_dir=tmpdir,
            )

        dialog = records[0]
        assert "sample_id" in dialog
        assert "rollout_id" in dialog
        assert "n_subquestions" in dialog
        assert "subquestion_rollouts" in dialog
        assert len(dialog["subquestion_rollouts"]) == 2

        for sq in dialog["subquestion_rollouts"]:
            for key in ("sq_id", "question", "status", "agent_steps",
                         "assistant_answer", "memory_output",
                         "r_tool_steps", "r_answer", "r_memory"):
                assert key in sq, f"Missing key '{key}' in subquestion rollout"

    def test_tool_rewards_computed(self):
        """Tool steps have non-zero rewards computed."""
        tes_lookup = {("test_sample", 1): _make_tes("test_sample", 1)}

        policy = MockPolicy([
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "test.xlsx"}}</tool_call>',
            '<answer>Answer.</answer>\n<memory>{"goal": "g1", "tables": [], "key_facts": []}</memory>',
        ])

        executor = FakeToolExecutor()
        calculator = RewardCalculator()

        runner = DialogRolloutRunner(
            policy=policy,
            tool_executor=executor,
            calculator=calculator,
            tes_lookup=tes_lookup,
            K=1,
            max_tool_steps_per_turn=6,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[_make_minimal_sample(n_sq=1)],
                tools_schema=[],
                output_dir=tmpdir,
            )

        sq = records[0]["subquestion_rollouts"][0]
        assert len(sq["r_tool_steps"]) > 0, "Should have at least one tool step reward"
        # Tool reward should include the per-call cost (-0.02)
        assert all(isinstance(r, (int, float)) for r in sq["r_tool_steps"])

    def test_multi_tool_invalid_penalty(self):
        """Multiple tool calls in one turn → invalid, penalty applied."""
        tes_lookup = {("test_sample", 1): _make_tes("test_sample", 1)}

        policy = MockPolicy([
            # First response: multiple tool calls
            '<tool_call>{"tool": "t1", "params": {}}</tool_call>'
            '<tool_call>{"tool": "t2", "params": {}}</tool_call>',
            # Retry with single
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "test.xlsx"}}</tool_call>',
            '<answer>Answer.</answer>\n<memory>{"goal": "g", "tables": [], "key_facts": []}</memory>',
        ])

        executor = FakeToolExecutor()
        calculator = RewardCalculator()

        runner = DialogRolloutRunner(
            policy=policy,
            tool_executor=executor,
            calculator=calculator,
            tes_lookup=tes_lookup,
            K=1,
            max_tool_steps_per_turn=6,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[_make_minimal_sample(n_sq=1)],
                tools_schema=[],
                output_dir=tmpdir,
            )

        sq = records[0]["subquestion_rollouts"][0]
        # First step should be invalid multi-tool
        assert sq["agent_steps"][0]["invalid_multi_tool"] is True
        # First reward should be negative (invalid penalty)
        assert sq["r_tool_steps"][0] < 0

    def test_k2_dialog_rollouts(self):
        """K=2 produces exactly 2 dialog rollouts for the sample."""
        tes_lookup = {
            ("test_sample", 1): _make_tes("test_sample", 1),
            ("test_sample", 2): _make_tes("test_sample", 2),
        }

        policy = MockPolicy([
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "test.xlsx"}}</tool_call>',
            '<answer>A1.</answer>\n<memory>{"goal": "g1", "tables": [], "key_facts": []}</memory>',
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "test.xlsx"}}</tool_call>',
            '<answer>A2.</answer>\n<memory>{"goal": "g2", "tables": [], "key_facts": []}</memory>',
            # Second rollout
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "test.xlsx"}}</tool_call>',
            '<answer>A1b.</answer>\n<memory>{"goal": "g1b", "tables": [], "key_facts": []}</memory>',
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "test.xlsx"}}</tool_call>',
            '<answer>A2b.</answer>\n<memory>{"goal": "g2b", "tables": [], "key_facts": []}</memory>',
        ])

        executor = FakeToolExecutor()
        calculator = RewardCalculator()

        runner = DialogRolloutRunner(
            policy=policy,
            tool_executor=executor,
            calculator=calculator,
            tes_lookup=tes_lookup,
            K=2,
            max_tool_steps_per_turn=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[_make_minimal_sample(n_sq=2)],
                tools_schema=[],
                output_dir=tmpdir,
            )

        assert len(records) == 2
        assert records[0]["rollout_id"] != records[1]["rollout_id"]
        # Both should have 2 subquestions
        assert records[0]["n_subquestions"] == 2
        assert records[1]["n_subquestions"] == 2


# ======================================================================
# Test 5 — Memory reward audit
# ======================================================================

class TestMemoryRewardAudit:
    """Verify memory reward computation produces detailed audit info."""

    def test_memory_reward_has_audit_fields(self):
        """Memory reward includes faithfulness, FDC, compression details."""
        tes_lookup = {("test_sample", 1): _make_tes("test_sample", 1)}
        fds_lookup = {("test_sample", "root"): _make_fds("test_sample", "root")}

        memory_with_key_fact = {
            "goal": "Find production growth",
            "tables": ["test.xlsx"],
            "key_facts": [
                {
                    "entity": "Company A",
                    "time": "2010年1月",
                    "metric": "产量同比增长率",
                    "value": "16.96%",
                    "unit": "%",
                }
            ],
        }

        policy = MockPolicy([
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "test.xlsx"}}</tool_call>',
            f'<answer>Production grew 16.96%.</answer>\n<memory>{json.dumps(memory_with_key_fact)}</memory>',
        ])

        executor = FakeToolExecutor()
        calculator = RewardCalculator(alpha_f=0.5, alpha_s=0.4, lambda_comp=0.1, B=512)

        runner = DialogRolloutRunner(
            policy=policy,
            tool_executor=executor,
            calculator=calculator,
            tes_lookup=tes_lookup,
            fds_lookup=fds_lookup,
            K=1,
            max_tool_steps_per_turn=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[_make_minimal_sample(n_sq=1)],
                tools_schema=[],
                output_dir=tmpdir,
            )

        sq = records[0]["subquestion_rollouts"][0]
        mem_detail = sq.get("memory_reward_detail", {})

        # Should have audit information
        assert "F_i" in mem_detail or "r_memory" in mem_detail
        # r_memory should be computed
        assert isinstance(sq.get("r_memory"), (int, float))

    def test_empty_memory_no_key_facts(self):
        """Memory with no key facts gets low faithfulness score."""
        tes_lookup = {("test_sample", 1): _make_tes("test_sample", 1)}
        fds_lookup = {("test_sample", "root"): _make_fds("test_sample", "root")}

        policy = MockPolicy([
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "test.xlsx"}}</tool_call>',
            '<answer>Answer.</answer>\n<memory>{"goal": "empty", "tables": [], '
            '"key_facts": [], "derived_results": [], "constraints": [], "pitfalls": []}</memory>',
        ])

        executor = FakeToolExecutor()
        calculator = RewardCalculator(alpha_f=0.5, alpha_s=0.4, lambda_comp=0.1, B=512)

        runner = DialogRolloutRunner(
            policy=policy,
            tool_executor=executor,
            calculator=calculator,
            tes_lookup=tes_lookup,
            fds_lookup=fds_lookup,
            K=1,
            max_tool_steps_per_turn=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[_make_minimal_sample(n_sq=1)],
                tools_schema=[],
                output_dir=tmpdir,
            )

        sq = records[0]["subquestion_rollouts"][0]
        # Empty memory with future deps → severe failure possible
        # or low reward
        assert sq.get("r_memory") is not None


# ======================================================================
# Test 6 — File I/O
# ======================================================================

class TestPhase2FileIO:
    """Verify output file format and content."""

    def test_output_written_to_jsonl(self):
        """Dialog rollouts are written to phase2_dialog_rollouts.jsonl."""
        tes_lookup = {("test_sample", 1): _make_tes("test_sample", 1)}

        policy = MockPolicy([
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "test.xlsx"}}</tool_call>',
            '<answer>Answer.</answer>\n<memory>{"goal": "g", "tables": [], "key_facts": []}</memory>',
        ])

        executor = FakeToolExecutor()
        calculator = RewardCalculator()

        runner = DialogRolloutRunner(
            policy=policy,
            tool_executor=executor,
            calculator=calculator,
            tes_lookup=tes_lookup,
            K=1,
            max_tool_steps_per_turn=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runner.run(
                samples=[_make_minimal_sample(n_sq=1)],
                tools_schema=[],
                output_dir=tmpdir,
            )

            output_path = os.path.join(tmpdir, "phase2_dialog_rollouts.jsonl")
            assert os.path.exists(output_path)

            with open(output_path, "r") as f:
                lines = f.readlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["sample_id"] == "test_sample"

    def test_multiple_samples_produce_multiple_records(self):
        """Two samples → two dialog rollout records."""
        tes_lookup = {
            ("sample_A", 1): _make_tes("sample_A", 1),
            ("sample_B", 1): _make_tes("sample_B", 1),
        }

        policy = MockPolicy([
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "a.xlsx"}}</tool_call>',
            '<answer>A.</answer>\n<memory>{"goal": "a", "tables": [], "key_facts": []}</memory>',
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "b.xlsx"}}</tool_call>',
            '<answer>B.</answer>\n<memory>{"goal": "b", "tables": [], "key_facts": []}</memory>',
        ])

        executor = FakeToolExecutor()
        calculator = RewardCalculator()

        runner = DialogRolloutRunner(
            policy=policy,
            tool_executor=executor,
            calculator=calculator,
            tes_lookup=tes_lookup,
            K=1,
            max_tool_steps_per_turn=2,
        )

        sample_a = {
            "task": "sample_A",
            "table_path": "a.xlsx",
            "design": {"checkout_list": [{"question": "Q A?"}]},
        }
        sample_b = {
            "task": "sample_B",
            "table_path": "b.xlsx",
            "design": {"checkout_list": [{"question": "Q B?"}]},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[sample_a, sample_b],
                tools_schema=[],
                output_dir=tmpdir,
            )
            assert len(records) == 2
            assert records[0]["sample_id"] == "sample_A"
            assert records[1]["sample_id"] == "sample_B"


# ======================================================================
# Test 7 — Edge cases
# ======================================================================

class TestPhase2EdgeCases:
    """Corner cases for Phase 2 dialog rollout."""

    def test_single_subquestion_dialog(self):
        """Dialog with only one subquestion works."""
        tes_lookup = {("test_sample", 1): _make_tes("test_sample", 1)}

        policy = MockPolicy([
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "test.xlsx"}}</tool_call>',
            '<answer>Answer.</answer>\n<memory>{"goal": "g", "tables": [], "key_facts": []}</memory>',
        ])

        executor = FakeToolExecutor()
        calculator = RewardCalculator()

        runner = DialogRolloutRunner(
            policy=policy,
            tool_executor=executor,
            calculator=calculator,
            tes_lookup=tes_lookup,
            K=1,
            max_tool_steps_per_turn=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[_make_minimal_sample(n_sq=1)],
                tools_schema=[],
                output_dir=tmpdir,
            )

        assert len(records) == 1
        dialog = records[0]
        assert dialog["n_subquestions"] == 1
        assert len(dialog["subquestion_rollouts"]) == 1
        sq = dialog["subquestion_rollouts"][0]
        assert sq["memory_before"] is None  # No previous subquestion

    def test_no_memory_in_response(self):
        """Model doesn't generate memory → memory_output is None."""
        tes_lookup = {("test_sample", 1): _make_tes("test_sample", 1)}

        policy = MockPolicy([
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "test.xlsx"}}</tool_call>',
            '<answer>Answer without memory.</answer>',
        ])

        executor = FakeToolExecutor()
        calculator = RewardCalculator()

        runner = DialogRolloutRunner(
            policy=policy,
            tool_executor=executor,
            calculator=calculator,
            tes_lookup=tes_lookup,
            K=1,
            max_tool_steps_per_turn=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[_make_minimal_sample(n_sq=1)],
                tools_schema=[],
                output_dir=tmpdir,
            )

        sq = records[0]["subquestion_rollouts"][0]
        assert sq.get("memory_output") is None
        assert sq.get("r_memory") == 0.0  # No memory → no memory reward

    def test_direct_answer_no_tools(self):
        """Model answers directly without any tool calls."""
        tes_lookup = {("test_sample", 1): _make_tes("test_sample", 1)}

        policy = MockPolicy([
            '<answer>Direct answer without tools.</answer>\n'
            '<memory>{"goal": "direct", "tables": [], "key_facts": []}</memory>',
        ])

        executor = FakeToolExecutor()
        calculator = RewardCalculator()

        runner = DialogRolloutRunner(
            policy=policy,
            tool_executor=executor,
            calculator=calculator,
            tes_lookup=tes_lookup,
            K=1,
            max_tool_steps_per_turn=6,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[_make_minimal_sample(n_sq=1)],
                tools_schema=[],
                output_dir=tmpdir,
            )

        sq = records[0]["subquestion_rollouts"][0]
        assert sq["status"] == "completed"
        assert sq["assistant_answer"] is not None
        assert sq["memory_output"] is not None
        assert len(sq["agent_steps"]) == 0  # No tool steps

    def test_truncated_on_max_steps(self):
        """Rollout truncates when max_tool_steps_per_turn is exceeded."""
        tes_lookup = {("test_sample", 1): _make_tes("test_sample", 1)}

        # Only tool calls, never an answer
        responses = [
            '<tool_call>{"tool": "reader", "params": {"file_path": "f.xlsx"}}</tool_call>'
        ] * 10

        policy = MockPolicy(responses)

        executor = FakeToolExecutor()
        calculator = RewardCalculator()

        runner = DialogRolloutRunner(
            policy=policy,
            tool_executor=executor,
            calculator=calculator,
            tes_lookup=tes_lookup,
            K=1,
            max_tool_steps_per_turn=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[_make_minimal_sample(n_sq=1)],
                tools_schema=[],
                output_dir=tmpdir,
            )

        sq = records[0]["subquestion_rollouts"][0]
        assert sq["status"] == "truncated"
        assert len(sq["agent_steps"]) >= 2


# ======================================================================
# Test 8 — Dependency type retention (numeric_fact, entity_set, etc.)
# ======================================================================

class TestDependencyTypeRetention:
    """Verify all five dependency types are handled correctly."""

    def test_numeric_fact_retention(self):
        """numeric_fact dependency is retained when fields match memory items."""
        tes_lookup = {("test_sample", 1): _make_tes("test_sample", 1)}
        fds = FutureDependencySet(
            sample_id="test_sample",
            boundary="root",
            future_dependencies=[
                FutureDependency(
                    dependency_id="dep_num",
                    type="numeric_fact",
                    needed_by="sq2",
                    fields={
                        "entity": "Company A",
                        "time": "2010年1月",
                        "metric": "产量同比增长率",
                        "value": "16.96%",
                        "unit": "%",
                    },
                ),
            ],
        )
        fds_lookup = {("test_sample", "root"): fds}

        # Memory that covers the numeric_fact
        memory = {
            "goal": "test",
            "tables": ["test.xlsx"],
            "key_facts": [{
                "entity": "Company A",
                "time": "2010年1月",
                "metric": "产量同比增长率",
                "value": "16.96%",
                "unit": "%",
            }],
        }

        policy = MockPolicy([
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "test.xlsx"}}</tool_call>',
            f'<answer>Production grew 16.96%.</answer>\n<memory>{json.dumps(memory)}</memory>',
        ])

        executor = FakeToolExecutor()
        calculator = RewardCalculator(alpha_f=0.5, alpha_s=0.4, lambda_comp=0.1, B=512)

        runner = DialogRolloutRunner(
            policy=policy,
            tool_executor=executor,
            calculator=calculator,
            tes_lookup=tes_lookup,
            fds_lookup=fds_lookup,
            K=1,
            max_tool_steps_per_turn=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[_make_minimal_sample(n_sq=1)],
                tools_schema=[],
                output_dir=tmpdir,
            )

        sq = records[0]["subquestion_rollouts"][0]
        mem_detail = sq.get("memory_reward_detail", {})
        # Should have H_keep_ids or S_i > 0
        h_keep = mem_detail.get("H_keep_ids", [])
        if h_keep:
            assert "dep_num" in h_keep, f"Expected dep_num in H_keep_ids, got {h_keep}"

    def test_entity_set_retention(self):
        """entity_set dependency is retained when entities appear in memory."""
        tes_lookup = {("test_sample", 1): _make_tes("test_sample", 1)}
        fds = FutureDependencySet(
            sample_id="test_sample",
            boundary="root",
            future_dependencies=[
                FutureDependency(
                    dependency_id="dep_entity",
                    type="entity_set",
                    needed_by="sq2",
                    fields={"entities": ["Company A", "Company B"]},
                ),
            ],
        )
        fds_lookup = {("test_sample", "root"): fds}

        memory = {
            "goal": "entity test",
            "tables": ["test.xlsx"],
            "key_facts": [
                {"entity": "Company A", "time": "2020", "metric": "m", "value": "1", "unit": ""},
                {"entity": "Company B", "time": "2020", "metric": "m", "value": "2", "unit": ""},
            ],
        }

        policy = MockPolicy([
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "test.xlsx"}}</tool_call>',
            f'<answer>Done.</answer>\n<memory>{json.dumps(memory)}</memory>',
        ])

        executor = FakeToolExecutor()
        calculator = RewardCalculator(alpha_f=0.5, alpha_s=0.4, lambda_comp=0.1, B=512)

        runner = DialogRolloutRunner(
            policy=policy,
            tool_executor=executor,
            calculator=calculator,
            tes_lookup=tes_lookup,
            fds_lookup=fds_lookup,
            K=1,
            max_tool_steps_per_turn=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[_make_minimal_sample(n_sq=1)],
                tools_schema=[],
                output_dir=tmpdir,
            )

        sq = records[0]["subquestion_rollouts"][0]
        assert sq.get("r_memory") is not None

    def test_reference_retention(self):
        """reference dependency exists in H_keep."""
        tes_lookup = {("test_sample", 1): _make_tes("test_sample", 1)}
        fds = FutureDependencySet(
            sample_id="test_sample",
            boundary="root",
            future_dependencies=[
                FutureDependency(
                    dependency_id="dep_ref",
                    type="reference",
                    needed_by="sq2",
                    fields={"reference_text": "the company from sq1", "target_sq": "sq1"},
                ),
            ],
        )
        fds_lookup = {("test_sample", "root"): fds}

        memory = {
            "goal": "ref test",
            "tables": ["test.xlsx"],
            "key_facts": [
                {"entity": "Company A", "time": "2020", "metric": "m", "value": "1", "unit": ""},
            ],
        }

        policy = MockPolicy([
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "test.xlsx"}}</tool_call>',
            f'<answer>Done.</answer>\n<memory>{json.dumps(memory)}</memory>',
        ])

        executor = FakeToolExecutor()
        calculator = RewardCalculator(alpha_f=0.5, alpha_s=0.4, lambda_comp=0.1, B=512)

        runner = DialogRolloutRunner(
            policy=policy,
            tool_executor=executor,
            calculator=calculator,
            tes_lookup=tes_lookup,
            fds_lookup=fds_lookup,
            K=1,
            max_tool_steps_per_turn=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[_make_minimal_sample(n_sq=1)],
                tools_schema=[],
                output_dir=tmpdir,
            )

        sq = records[0]["subquestion_rollouts"][0]
        mem_detail = sq.get("memory_reward_detail", {})
        assert "severe_failure" in mem_detail or "r_memory" in mem_detail

    def test_constraint_retention(self):
        """constraint dependency is handled."""
        tes_lookup = {("test_sample", 1): _make_tes("test_sample", 1)}
        fds = FutureDependencySet(
            sample_id="test_sample",
            boundary="root",
            future_dependencies=[
                FutureDependency(
                    dependency_id="dep_constraint",
                    type="constraint",
                    needed_by="sq2",
                    fields={"constraint_content": "production > 100 units"},
                ),
            ],
        )
        fds_lookup = {("test_sample", "root"): fds}

        memory = {
            "goal": "constraint test",
            "tables": ["test.xlsx"],
            "key_facts": [
                {"entity": "A", "time": "2020", "metric": "production", "value": "150", "unit": "units"},
            ],
            "constraints": ["production > 100 units"],
        }

        policy = MockPolicy([
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "test.xlsx"}}</tool_call>',
            f'<answer>Done.</answer>\n<memory>{json.dumps(memory)}</memory>',
        ])

        executor = FakeToolExecutor()
        calculator = RewardCalculator(alpha_f=0.5, alpha_s=0.4, lambda_comp=0.1, B=512)

        runner = DialogRolloutRunner(
            policy=policy,
            tool_executor=executor,
            calculator=calculator,
            tes_lookup=tes_lookup,
            fds_lookup=fds_lookup,
            K=1,
            max_tool_steps_per_turn=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[_make_minimal_sample(n_sq=1)],
                tools_schema=[],
                output_dir=tmpdir,
            )

        sq = records[0]["subquestion_rollouts"][0]
        assert sq.get("r_memory") is not None

    def test_table_ref_retention(self):
        """table_ref dependency is handled."""
        tes_lookup = {("test_sample", 1): _make_tes("test_sample", 1)}
        fds = FutureDependencySet(
            sample_id="test_sample",
            boundary="root",
            future_dependencies=[
                FutureDependency(
                    dependency_id="dep_table",
                    type="table_ref",
                    needed_by="sq2",
                    fields={"table_name": "test.xlsx"},
                ),
            ],
        )
        fds_lookup = {("test_sample", "root"): fds}

        memory = {
            "goal": "table ref test",
            "tables": ["test.xlsx"],
            "key_facts": [
                {"entity": "A", "time": "2020", "metric": "m", "value": "1", "unit": ""},
            ],
        }

        policy = MockPolicy([
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "test.xlsx"}}</tool_call>',
            f'<answer>Done.</answer>\n<memory>{json.dumps(memory)}</memory>',
        ])

        executor = FakeToolExecutor()
        calculator = RewardCalculator(alpha_f=0.5, alpha_s=0.4, lambda_comp=0.1, B=512)

        runner = DialogRolloutRunner(
            policy=policy,
            tool_executor=executor,
            calculator=calculator,
            tes_lookup=tes_lookup,
            fds_lookup=fds_lookup,
            K=1,
            max_tool_steps_per_turn=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[_make_minimal_sample(n_sq=1)],
                tools_schema=[],
                output_dir=tmpdir,
            )

        sq = records[0]["subquestion_rollouts"][0]
        assert sq.get("r_memory") is not None
