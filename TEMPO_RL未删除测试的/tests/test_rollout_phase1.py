"""
Tempo-RL Phase 1 — Smoke tests for rollout runner.

Uses mock policy + mock tool executor to verify the rollout pipeline
end-to-end without needing a real LLM or real table files.
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

from TEMPO_RL.rollout_phase1 import (
    RolloutRunner,
    ToolExecutor,
    MockPolicy,
    extract_tool_calls_from_response,
    extract_answer_from_response,
    count_tool_calls,
)
from TEMPO_RL.schemas import TargetEvidenceSet, EvidenceItem
from TEMPO_RL.reward_calculator import RewardCalculator


# ======================================================================
# Helper factories
# ======================================================================

def _make_minimal_tes(sample_id: str = "test_sample", sq_id: int = 1) -> TargetEvidenceSet:
    """Build a minimal TargetEvidenceSet for testing."""
    items = [
        EvidenceItem(
            sample_id=sample_id,
            subquestion_id=sq_id,
            evidence_id=f"sq{sq_id}_e1",
            type="text_fact",
            value="总产量为120.5万辆",
            entity="总产量",
            metric="产量",
            unit="万辆",
            weight=1.0,
        ),
        EvidenceItem(
            sample_id=sample_id,
            subquestion_id=sq_id,
            evidence_id=f"sq{sq_id}_e2",
            type="raw_value",
            value="同比增长率5.3%",
            entity="同比增长率",
            metric="增长率",
            unit="%",
            weight=1.0,
        ),
    ]
    return TargetEvidenceSet(
        sample_id=sample_id,
        subquestion_id=sq_id,
        question="What is the total production and YoY growth rate?",
        evidence_items=items,
    )


class FakeToolExecutor(ToolExecutor):
    """Tool executor that returns controlled fake observations."""

    def __init__(self, observations: list = None):
        super().__init__(table_root="")
        self._obs = observations or []
        self._obs_idx = 0
        # Real execute_tool needs tables to exist, so we override

    def execute(self, tool_name: str, arguments: dict) -> dict:
        self._step_counter += 1
        if self._obs_idx < len(self._obs):
            obs = dict(self._obs[self._obs_idx])
            self._obs_idx += 1
            obs.setdefault("tool_call_id", f"tc_{self._step_counter}")
            return obs
        # Default fake observation with value hints for evidence detection
        return {
            "tool_call_id": f"tc_{self._step_counter}",
            "tool_name": tool_name,
            "content": f"[SUCCESS] Found data: total production 120.5万辆, YoY 5.3% growth",
            "success": True,
        }


# ======================================================================
# Test 1 — Response parsing
# ======================================================================

class TestResponseParsing:
    """Verify tool_call and answer extraction from model responses."""

    def test_extract_single_tool_call(self):
        resp = '<think>Need to look up data</think>\n\n<tool_call>{"tool": "table_head_reader", "params": {"file_path": "/tmp/test.csv", "start": 0, "n": 20}}</tool_call>'
        tcs = extract_tool_calls_from_response(resp)
        assert len(tcs) == 1
        assert tcs[0]["tool_name"] == "table_head_reader"
        assert tcs[0]["arguments"] == {"file_path": "/tmp/test.csv", "start": 0, "n": 20}

    def test_extract_zero_tool_calls(self):
        resp = "<answer>Production was 120.5万辆 with 5.3% growth.</answer>"
        tcs = extract_tool_calls_from_response(resp)
        assert len(tcs) == 0

    def test_extract_multiple_tool_calls(self):
        resp = """
<tool_call>{"tool": "table_head_reader", "params": {"file_path": "a.csv"}}</tool_call>
<tool_call>{"tool": "grep_search", "params": {"path": ".", "pattern": "test"}}</tool_call>
"""
        tcs = extract_tool_calls_from_response(resp)
        assert len(tcs) == 2
        assert tcs[0]["tool_name"] == "table_head_reader"
        assert tcs[1]["tool_name"] == "grep_search"

    def test_count_tool_calls(self):
        resp = '<tool_call>{"tool":"a"}</tool_call>\n<tool_call>{"tool":"b"}</tool_call>'
        assert count_tool_calls(resp) == 2
        resp2 = '<answer>done</answer>'
        assert count_tool_calls(resp2) == 0

    def test_extract_answer(self):
        resp = '<think>thinking...</think>\n\n<answer>Production: 120.5万辆</answer>'
        ans = extract_answer_from_response(resp)
        assert ans == "Production: 120.5万辆"

    def test_extract_answer_none(self):
        resp = '<tool_call>{"tool": "cmd_executor", "params": {"command": "ls"}}</tool_call>'
        ans = extract_answer_from_response(resp)
        assert ans is None

    def test_tool_call_with_name_not_tool(self):
        """Some formats use 'name' instead of 'tool'."""
        resp = '<tool_call>{"name": "table_head_reader", "arguments": {"file_path": "x.csv"}}</tool_call>'
        tcs = extract_tool_calls_from_response(resp)
        assert len(tcs) == 1
        assert tcs[0]["tool_name"] == "table_head_reader"

    def test_tool_call_with_string_arguments(self):
        """Arguments may be a JSON string."""
        resp = '<tool_call>{"tool": "grep_search", "params": "{\\"pattern\\": \\"产量\\"}"}</tool_call>'
        tcs = extract_tool_calls_from_response(resp)
        assert len(tcs) == 1


# ======================================================================
# Test 2 — Mock Policy
# ======================================================================

class TestMockPolicy:
    """Verify mock policy returns pre-configured responses in order."""

    def test_returns_in_order(self):
        mock = MockPolicy([
            '<tool_call>{"tool": "cmd_executor", "params": {"command": "ls"}}</tool_call>',
            '<answer>All done.</answer>',
        ])
        r1 = mock.call([], [])
        assert "cmd_executor" in r1["content"]
        r2 = mock.call([], [])
        assert "All done" in r2["content"]

    def test_fallback_when_exhausted(self):
        mock = MockPolicy(['<answer>done.</answer>'])
        mock.call([], [])  # consume
        r = mock.call([], [])  # exhausted → fallback
        assert "answer" in r["content"].lower()

    def test_call_count(self):
        mock = MockPolicy(['<answer>x</answer>', '<answer>y</answer>'])
        assert mock.call_count == 0
        mock.call([], [])
        mock.call([], [])
        assert mock.call_count == 2

    def test_set_responses(self):
        mock = MockPolicy(['<answer>a</answer>'])
        mock.set_responses(['<answer>b</answer>'])
        r = mock.call([], [])
        assert "b" in r["content"]


# ======================================================================
# Test 3 — Rollout with valid single tool call → answer
# ======================================================================

class TestRolloutValidSingleTool:
    """Rollout where the model emits exactly one tool call per turn,
    then answers — should complete successfully."""

    def test_single_tool_then_answer(self):
        tes = _make_minimal_tes()
        # Tool call that would find the target data
        resp1 = '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "data.csv", "start": 0, "n": 10}}</tool_call>'
        resp2 = '<answer>Total production: 120.5万辆, YoY growth: 5.3%</answer>'

        mock = MockPolicy([resp1, resp2])
        tool_exec = FakeToolExecutor([
            {
                "tool_call_id": "tc_1",
                "tool_name": "table_head_reader",
                "content": "[SUCCESS] col1: total production 120.5万辆, col2: YoY growth 5.3%",
                "success": True,
            }
        ])
        calc = RewardCalculator()

        runner = RolloutRunner(
            policy=mock,
            tool_executor=tool_exec,
            calculator=calc,
            tes_lookup={("test_sample", 1): tes},
            K=1,
            max_tool_steps=4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[{
                    "task": "test_sample",
                    "file_path": "",
                    "design": {
                        "checkout_list": [{
                            "idx": 1,
                            "info_item": "What is the total production?",
                            "score_points": ["总产量为120.5万辆", "同比增长率5.3%"],
                        }]
                    }
                }],
                tools_schema=[],
                output_dir=tmpdir,
            )

            assert len(records) == 1
            r = records[0]
            assert r["status"] == "completed"
            assert len(r["agent_steps"]) == 1
            assert r["assistant_answer"] is not None
            assert "120.5" in str(r["assistant_answer"])
            assert len(r["r_tool_steps"]) == 1
            assert isinstance(r["r_tool"], float)
            assert isinstance(r["r_answer"], float)

    def test_no_tool_steps_direct_answer(self):
        """Model answers immediately without any tool calls."""
        tes = _make_minimal_tes()
        mock = MockPolicy(['<answer>Direct answer without tools.</answer>'])
        tool_exec = FakeToolExecutor()
        calc = RewardCalculator()

        runner = RolloutRunner(
            policy=mock,
            tool_executor=tool_exec,
            calculator=calc,
            tes_lookup={("test_sample", 1): tes},
            K=1,
            max_tool_steps=4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[{
                    "task": "test_sample",
                    "file_path": "",
                    "design": {
                        "checkout_list": [{
                            "idx": 1,
                            "info_item": "Question?",
                            "score_points": ["答案1"],
                        }]
                    }
                }],
                tools_schema=[],
                output_dir=tmpdir,
            )

            assert len(records) == 1
            r = records[0]
            assert r["status"] == "completed"
            assert len(r["agent_steps"]) == 0
            assert r["assistant_answer"] is not None


# ======================================================================
# Test 4 — Rollout with multiple tool calls → invalid
# ======================================================================

class TestRolloutMultiToolInvalid:
    """When the model emits multiple tool calls in one turn,
    it should be marked invalid, no tools executed, penalty applied."""

    def test_multi_tool_call_marked_invalid(self):
        tes = _make_minimal_tes()
        resp1 = """
<tool_call>{"tool": "table_head_reader", "params": {"file_path": "a.csv"}}</tool_call>
<tool_call>{"tool": "grep_search", "params": {"path": ".", "pattern": "x"}}</tool_call>
"""
        resp2 = '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "a.csv"}}</tool_call>'
        resp3 = '<answer>Result: 120.5万辆</answer>'

        mock = MockPolicy([resp1, resp2, resp3])
        tool_exec = FakeToolExecutor([
            {
                "tool_name": "table_head_reader",
                "content": "[SUCCESS] found: 120.5万辆",
                "success": True,
            }
        ])
        calc = RewardCalculator()

        runner = RolloutRunner(
            policy=mock,
            tool_executor=tool_exec,
            calculator=calc,
            tes_lookup={("test_sample", 1): tes},
            K=1,
            max_tool_steps=4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[{
                    "task": "test_sample",
                    "file_path": "",
                    "design": {
                        "checkout_list": [{
                            "idx": 1,
                            "info_item": "Question?",
                            "score_points": ["总产量为120.5万辆"],
                        }]
                    }
                }],
                tools_schema=[],
                output_dir=tmpdir,
            )

            assert len(records) == 1
            r = records[0]
            # First step should be invalid_multi_tool
            first_step = r["agent_steps"][0]
            assert first_step["invalid_multi_tool"] is True
            assert len(first_step["observations"]) == 0  # no tools executed
            # r_tool for invalid step should be negative (penalty)
            assert r["r_tool_steps"][0] < 0, f"Expected negative reward for invalid step, got {r['r_tool_steps'][0]}"

    def test_multi_tool_counted_as_step(self):
        """A multi-tool turn still counts toward max_tool_steps."""
        tes = _make_minimal_tes()
        # Return multi-tool every time → should hit max_tool_steps
        multi = """
<tool_call>{"tool": "a", "params": {}}</tool_call>
<tool_call>{"tool": "b", "params": {}}</tool_call>
"""
        mock = MockPolicy([multi] * 6)  # 6 responses, each multi-tool
        tool_exec = FakeToolExecutor()
        calc = RewardCalculator()

        runner = RolloutRunner(
            policy=mock,
            tool_executor=tool_exec,
            calculator=calc,
            tes_lookup={("test_sample", 1): tes},
            K=1,
            max_tool_steps=3,  # low limit
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[{
                    "task": "test_sample",
                    "file_path": "",
                    "design": {
                        "checkout_list": [{
                            "idx": 1,
                            "info_item": "Q?",
                            "score_points": ["A"],
                        }]
                    }
                }],
                tools_schema=[],
                output_dir=tmpdir,
            )

            r = records[0]
            assert r["status"] == "truncated"
            assert r["num_tool_steps"] == 3
            assert all(
                step["invalid_multi_tool"] for step in r["agent_steps"]
            )


# ======================================================================
# Test 5 — Rollout truncated at max_tool_steps
# ======================================================================

class TestRolloutTruncated:
    """When the model never emits an answer, it should be truncated
    after max_tool_steps."""

    def test_truncated_after_max_steps(self):
        tes = _make_minimal_tes()
        tool_call = '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "x.csv"}}</tool_call>'
        # Return tool_call forever, never answer
        mock = MockPolicy([tool_call] * 10)
        tool_exec = FakeToolExecutor()
        calc = RewardCalculator()

        runner = RolloutRunner(
            policy=mock,
            tool_executor=tool_exec,
            calculator=calc,
            tes_lookup={("test_sample", 1): tes},
            K=1,
            max_tool_steps=3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[{
                    "task": "test_sample",
                    "file_path": "",
                    "design": {
                        "checkout_list": [{
                            "idx": 1,
                            "info_item": "Q?",
                            "score_points": ["A"],
                        }]
                    }
                }],
                tools_schema=[],
                output_dir=tmpdir,
            )

            r = records[0]
            assert r["status"] == "truncated"
            assert r["assistant_answer"] is None
            assert r["num_tool_steps"] == 3
            assert len(r["agent_steps"]) == 3

    def test_truncated_has_no_answer_reward(self):
        """When truncated, r_answer should be 0 (no answer parsed)."""
        tes = _make_minimal_tes()
        tool_call = '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "x.csv"}}</tool_call>'
        mock = MockPolicy([tool_call] * 5)
        tool_exec = FakeToolExecutor()
        calc = RewardCalculator()

        runner = RolloutRunner(
            policy=mock,
            tool_executor=tool_exec,
            calculator=calc,
            tes_lookup={("test_sample", 1): tes},
            K=1,
            max_tool_steps=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[{
                    "task": "test_sample",
                    "file_path": "",
                    "design": {
                        "checkout_list": [{
                            "idx": 1,
                            "info_item": "Q?",
                            "score_points": ["A"],
                        }]
                    }
                }],
                tools_schema=[],
                output_dir=tmpdir,
            )

            r = records[0]
            assert r["r_answer"] == 0.0
            assert r["assistant_answer"] is None


# ======================================================================
# Test 6 — Output structure validation
# ======================================================================

class TestOutputStructure:
    """Verify the output JSONL records have all required fields."""

    REQUIRED_FIELDS = [
        "sample_id",
        "subquestion_id",
        "rollout_id",
        "agent_steps",
        "assistant_answer",
        "ledger_trace",
        "r_tool_steps",
        "r_tool",
        "r_answer",
        "status",
    ]

    def test_all_required_fields_present(self):
        tes = _make_minimal_tes()
        mock = MockPolicy([
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "x.csv"}}</tool_call>',
            '<answer>Result: 120.5万辆</answer>',
        ])
        tool_exec = FakeToolExecutor()
        calc = RewardCalculator()

        runner = RolloutRunner(
            policy=mock,
            tool_executor=tool_exec,
            calculator=calc,
            tes_lookup={("test_sample", 1): tes},
            K=1,
            max_tool_steps=4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[{
                    "task": "test_sample",
                    "file_path": "",
                    "design": {
                        "checkout_list": [{
                            "idx": 1,
                            "info_item": "Q?",
                            "score_points": ["A"],
                        }]
                    }
                }],
                tools_schema=[],
                output_dir=tmpdir,
            )

            for r in records:
                for field in self.REQUIRED_FIELDS:
                    assert field in r, f"Missing field: {field}"

    def test_ledger_trace_monotonic(self):
        """Ledger trace should grow or stay flat (never shrink)."""
        tes = _make_minimal_tes()
        mock = MockPolicy([
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "x.csv"}}</tool_call>',
            '<answer>Done.</answer>',
        ])
        tool_exec = FakeToolExecutor()
        calc = RewardCalculator()

        runner = RolloutRunner(
            policy=mock,
            tool_executor=tool_exec,
            calculator=calc,
            tes_lookup={("test_sample", 1): tes},
            K=1,
            max_tool_steps=4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[{
                    "task": "test_sample",
                    "file_path": "",
                    "design": {
                        "checkout_list": [{
                            "idx": 1,
                            "info_item": "Q?",
                            "score_points": ["A"],
                        }]
                    }
                }],
                tools_schema=[],
                output_dir=tmpdir,
            )

            trace = records[0]["ledger_trace"]
            prev = 0
            for snapshot in trace:
                cov = snapshot.get("coverage", 0)
                assert cov >= prev, f"Coverage went backwards: {prev} → {cov}"
                prev = cov

    def test_jsonl_file_written(self):
        """phase1_rollouts.jsonl is created and parseable."""
        tes = _make_minimal_tes()
        mock = MockPolicy(['<answer>Done.</answer>'])
        tool_exec = FakeToolExecutor()
        calc = RewardCalculator()

        runner = RolloutRunner(
            policy=mock,
            tool_executor=tool_exec,
            calculator=calc,
            tes_lookup={("test_sample", 1): tes},
            K=1,
            max_tool_steps=4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runner.run(
                samples=[{
                    "task": "test_sample",
                    "file_path": "",
                    "design": {
                        "checkout_list": [{
                            "idx": 1,
                            "info_item": "Q?",
                            "score_points": ["A"],
                        }]
                    }
                }],
                tools_schema=[],
                output_dir=tmpdir,
            )

            fpath = os.path.join(tmpdir, "phase1_rollouts.jsonl")
            assert os.path.exists(fpath)
            with open(fpath, "r") as f:
                lines = [json.loads(l) for l in f if l.strip()]
            assert len(lines) == 1

    def test_status_values(self):
        """Verify status field uses expected values."""
        tes = _make_minimal_tes()
        # Direct answer → completed
        mock = MockPolicy(['<answer>Done.</answer>'])
        tool_exec = FakeToolExecutor()
        calc = RewardCalculator()

        runner = RolloutRunner(
            policy=mock,
            tool_executor=tool_exec,
            calculator=calc,
            tes_lookup={("test_sample", 1): tes},
            K=1,
            max_tool_steps=4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[{
                    "task": "test_sample",
                    "file_path": "",
                    "design": {
                        "checkout_list": [{
                            "idx": 1,
                            "info_item": "Q?",
                            "score_points": ["A"],
                        }]
                    }
                }],
                tools_schema=[],
                output_dir=tmpdir,
            )
            assert records[0]["status"] == "completed"


# ======================================================================
# Test 7 — Answer + tool_call in same response
# ======================================================================

class TestMixedToolCallAndAnswer:
    """Responses containing both tool_call and answer tags."""

    def test_both_present_prioritizes_tool(self):
        """When both tags are present, treat as tool_call (not answer)."""
        resp = '<tool_call>{"tool": "cmd_executor", "params": {"command": "ls"}}</tool_call>\n<answer>Some premature answer</answer>'
        tcs = extract_tool_calls_from_response(resp)
        assert len(tcs) == 1
        ans = extract_answer_from_response(resp)
        assert ans is not None  # present but should be ignored during rollout

    def test_both_tags_in_rollout_treated_as_tool(self):
        """In a rollout, a response with both tags continues the loop."""
        tes = _make_minimal_tes()
        # First response: tool_call + answer (should be treated as tool call)
        resp1 = '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "x.csv"}}</tool_call>\n<answer>premature answer</answer>'
        # Second response: proper answer
        resp2 = '<answer>Final: 120.5万辆</answer>'

        mock = MockPolicy([resp1, resp2])
        tool_exec = FakeToolExecutor()
        calc = RewardCalculator()

        runner = RolloutRunner(
            policy=mock,
            tool_executor=tool_exec,
            calculator=calc,
            tes_lookup={("test_sample", 1): tes},
            K=1,
            max_tool_steps=4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[{
                    "task": "test_sample",
                    "file_path": "",
                    "design": {
                        "checkout_list": [{
                            "idx": 1,
                            "info_item": "Q?",
                            "score_points": ["A"],
                        }]
                    }
                }],
                tools_schema=[],
                output_dir=tmpdir,
            )

            r = records[0]
            assert r["status"] == "completed"
            assert len(r["agent_steps"]) == 1  # one tool step
            assert r["assistant_answer"] is not None


# ======================================================================
# Test 8 — Multiple rollouts (K > 1)
# ======================================================================

class TestMultipleRollouts:
    """Verify K > 1 generates multiple rollouts per subquestion."""

    def test_k2_generates_two_records(self):
        tes = _make_minimal_tes()
        mock = MockPolicy(['<answer>Answer.</answer>', '<answer>Another answer.</answer>'])
        tool_exec = FakeToolExecutor()
        calc = RewardCalculator()

        runner = RolloutRunner(
            policy=mock,
            tool_executor=tool_exec,
            calculator=calc,
            tes_lookup={("test_sample", 1): tes},
            K=2,
            max_tool_steps=4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[{
                    "task": "test_sample",
                    "file_path": "",
                    "design": {
                        "checkout_list": [{
                            "idx": 1,
                            "info_item": "Q?",
                            "score_points": ["A"],
                        }]
                    }
                }],
                tools_schema=[],
                output_dir=tmpdir,
            )

            assert len(records) == 2
            assert records[0]["rollout_id"].endswith("_k0")
            assert records[1]["rollout_id"].endswith("_k1")
            # Each rollout should get different responses
            assert records[0]["assistant_answer"]["content"] != records[1]["assistant_answer"]["content"]

    def test_k3_all_completed(self):
        tes = _make_minimal_tes()
        answers = ['<answer>A1</answer>', '<answer>A2</answer>', '<answer>A3</answer>']
        mock = MockPolicy(answers)
        tool_exec = FakeToolExecutor()
        calc = RewardCalculator()

        runner = RolloutRunner(
            policy=mock,
            tool_executor=tool_exec,
            calculator=calc,
            tes_lookup={("test_sample", 1): tes},
            K=3,
            max_tool_steps=4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[{
                    "task": "test_sample",
                    "file_path": "",
                    "design": {
                        "checkout_list": [{
                            "idx": 1,
                            "info_item": "Q?",
                            "score_points": ["A"],
                        }]
                    }
                }],
                tools_schema=[],
                output_dir=tmpdir,
            )

            assert len(records) == 3
            for r in records:
                assert r["status"] == "completed"


# ======================================================================
# Test 9 — Error handling
# ======================================================================

class TestErrorHandling:
    """Edge cases and error handling."""

    def test_sample_with_no_checkout_list(self):
        """Sample without checkout_list should be skipped gracefully."""
        tes = _make_minimal_tes()
        mock = MockPolicy(['<answer>x</answer>'])
        tool_exec = FakeToolExecutor()
        calc = RewardCalculator()

        runner = RolloutRunner(
            policy=mock,
            tool_executor=tool_exec,
            calculator=calc,
            tes_lookup={("test_sample", 1): tes},
            K=1,
            max_tool_steps=4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[{"task": "test_sample", "design": {}, "file_path": ""}],
                tools_schema=[],
                output_dir=tmpdir,
            )
            assert len(records) == 0  # No checkout_list → skipped

    def test_sample_with_no_task(self):
        """Sample without task field should be skipped."""
        tes = _make_minimal_tes()
        mock = MockPolicy(['<answer>x</answer>'])
        tool_exec = FakeToolExecutor()
        calc = RewardCalculator()

        runner = RolloutRunner(
            policy=mock,
            tool_executor=tool_exec,
            calculator=calc,
            tes_lookup={("test_sample", 1): tes},
            K=1,
            max_tool_steps=4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[{"design": {"checkout_list": [{"idx": 1, "info_item": "Q"}]}}],
                tools_schema=[],
                output_dir=tmpdir,
            )
            assert len(records) == 0

    def test_no_target_evidence_for_subquestion(self):
        """Subquestion with no target evidence is skipped."""
        mock = MockPolicy(['<answer>x</answer>'])
        tool_exec = FakeToolExecutor()
        calc = RewardCalculator()

        runner = RolloutRunner(
            policy=mock,
            tool_executor=tool_exec,
            calculator=calc,
            tes_lookup={},  # empty → nothing matches
            K=1,
            max_tool_steps=4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[{
                    "task": "test_sample",
                    "file_path": "",
                    "design": {
                        "checkout_list": [{
                            "idx": 1,
                            "info_item": "Q?",
                            "score_points": ["A"],
                        }]
                    }
                }],
                tools_schema=[],
                output_dir=tmpdir,
            )
            assert len(records) == 0

    def test_mock_with_empty_response(self):
        """Empty model response should be handled (no tool_call, no answer → error)."""
        tes = _make_minimal_tes()
        mock = MockPolicy([""])
        tool_exec = FakeToolExecutor()
        calc = RewardCalculator()

        runner = RolloutRunner(
            policy=mock,
            tool_executor=tool_exec,
            calculator=calc,
            tes_lookup={("test_sample", 1): tes},
            K=1,
            max_tool_steps=4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[{
                    "task": "test_sample",
                    "file_path": "",
                    "design": {
                        "checkout_list": [{
                            "idx": 1,
                            "info_item": "Q?",
                            "score_points": ["A"],
                        }]
                    }
                }],
                tools_schema=[],
                output_dir=tmpdir,
            )

            assert len(records) == 1
            assert records[0]["status"] in ("error", "truncated")

    def test_malformed_tool_call_json(self):
        """Bad JSON in tool_call should be ignored."""
        resp = '<tool_call>not valid json{{{</tool_call>'
        tcs = extract_tool_calls_from_response(resp)
        assert len(tcs) == 0

    def test_unicode_in_response(self):
        """Responses with Chinese characters should be handled."""
        resp = '<tool_call>{"tool": "grep_search", "params": {"path": ".", "pattern": "产量"}}</tool_call>'
        tcs = extract_tool_calls_from_response(resp)
        assert len(tcs) == 1
        assert tcs[0]["arguments"]["pattern"] == "产量"


# ======================================================================
# Test 10 — Step-level detail
# ======================================================================

class TestStepDetail:
    """Verify per-step records are correctly structured."""

    def test_step_structure(self):
        tes = _make_minimal_tes()
        resp1 = '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "a.csv"}}</tool_call>'
        resp2 = '<answer>Answer text.</answer>'

        mock = MockPolicy([resp1, resp2])
        tool_exec = FakeToolExecutor()
        calc = RewardCalculator()

        runner = RolloutRunner(
            policy=mock,
            tool_executor=tool_exec,
            calculator=calc,
            tes_lookup={("test_sample", 1): tes},
            K=1,
            max_tool_steps=4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[{
                    "task": "test_sample",
                    "file_path": "",
                    "design": {
                        "checkout_list": [{
                            "idx": 1,
                            "info_item": "Q?",
                            "score_points": ["A"],
                        }]
                    }
                }],
                tools_schema=[],
                output_dir=tmpdir,
            )

            r = records[0]
            step = r["agent_steps"][0]
            assert step["type"] == "tool_call"
            assert step["step_index"] == 0
            assert not step["invalid_multi_tool"]
            assert len(step["tool_calls"]) == 1
            assert len(step["observations"]) == 1
            assert step["tool_calls"][0]["tool_name"] == "table_head_reader"

    def test_ledger_trace_entries_match_steps(self):
        """Ledger trace has one entry per step + initial."""
        tes = _make_minimal_tes()
        mock = MockPolicy([
            '<tool_call>{"tool": "table_head_reader", "params": {"file_path": "a.csv"}}</tool_call>',
            '<tool_call>{"tool": "grep_search", "params": {"path": ".", "pattern": "产量"}}</tool_call>',
            '<answer>Done.</answer>',
        ])
        tool_exec = FakeToolExecutor()
        calc = RewardCalculator()

        runner = RolloutRunner(
            policy=mock,
            tool_executor=tool_exec,
            calculator=calc,
            tes_lookup={("test_sample", 1): tes},
            K=1,
            max_tool_steps=4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            records = runner.run(
                samples=[{
                    "task": "test_sample",
                    "file_path": "",
                    "design": {
                        "checkout_list": [{
                            "idx": 1,
                            "info_item": "Q?",
                            "score_points": ["A"],
                        }]
                    }
                }],
                tools_schema=[],
                output_dir=tmpdir,
            )

            r = records[0]
            # trace = initial + 2 tool steps
            assert len(r["ledger_trace"]) == 3
            assert len(r["r_tool_steps"]) == 2
            assert len(r["agent_steps"]) == 2
