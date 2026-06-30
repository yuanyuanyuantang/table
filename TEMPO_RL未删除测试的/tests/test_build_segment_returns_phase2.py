"""
TEMPO-RL Phase 2 — Smoke tests for segment return builder.

Covers: tool/answer/memory segments, memory advantage writing,
normalisation across three segment types, conversation masks with memory.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile

import pytest

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from TEMPO_RL.build_segment_returns_phase2 import (
    SegmentReturnBuilderPhase2,
    compute_tool_return_to_go,
    compute_answer_return,
    compute_memory_return,
    _z_normalize,
    _reconstruct_conversation_messages,
)
from TEMPO_RL.io_utils import read_jsonl


# ======================================================================
# Helpers
# ======================================================================

def _make_dialog_rollout(
    rollout_id: str = "test_sample_k0",
    n_sq: int = 2,
    n_tool_steps_per_sq: int = 1,
) -> dict:
    """Build a synthetic dialog rollout record."""
    sq_rollouts = []
    for sq_id in range(1, n_sq + 1):
        r_tool_steps = [-0.02] * n_tool_steps_per_sq
        agent_steps = []
        for t in range(n_tool_steps_per_sq):
            agent_steps.append({
                "step_index": t,
                "type": "tool_call",
                "tool_calls": [{"tool_name": "table_head_reader", "arguments": {}}],
                "observations": [{"tool_name": "table_head_reader", "content": "[SUCCESS] data", "success": True}],
                "invalid_multi_tool": False,
            })

        sq_rollouts.append({
            "sq_id": sq_id,
            "question": f"Test question {sq_id}",
            "status": "completed",
            "agent_steps": agent_steps,
            "assistant_answer": {"content": f"Answer for sq{sq_id}."},
            "memory_before": {"goal": "previous"} if sq_id > 1 else None,
            "memory_output": {"goal": f"memory_{sq_id}", "tables": ["test.xlsx"], "key_facts": [
                {"entity": f"E{sq_id}", "time": "2020", "metric": "m", "value": str(sq_id), "unit": ""}
            ]},
            "memory_severe_failure": False,
            "r_tool_steps": r_tool_steps,
            "r_answer": 0.5 if sq_id == 1 else 0.3,
            "r_memory": 0.7 if sq_id == 1 else 0.4,
        })
    return {
        "sample_id": "test_sample",
        "rollout_id": rollout_id,
        "n_subquestions": n_sq,
        "subquestion_rollouts": sq_rollouts,
    }


def _make_simple_rollout_with_memory(r_memory: float = 0.5):
    """Single subquestion dialog with memory."""
    return {
        "sample_id": "test",
        "rollout_id": "test_k0",
        "n_subquestions": 1,
        "subquestion_rollouts": [{
            "sq_id": 1,
            "question": "Q1",
            "status": "completed",
            "agent_steps": [{
                "step_index": 0,
                "type": "tool_call",
                "tool_calls": [{"tool_name": "read", "arguments": {}}],
                "observations": [{"tool_name": "read", "content": "[SUCCESS]", "success": True}],
                "invalid_multi_tool": False,
            }],
            "assistant_answer": {"content": "Answer."},
            "memory_before": None,
            "memory_output": {"goal": "g", "tables": [], "key_facts": []},
            "memory_severe_failure": False,
            "r_tool_steps": [-0.02],
            "r_answer": 0.5,
            "r_memory": r_memory,
        }],
    }


# ======================================================================
# Test 1 — Memory return computation
# ======================================================================

class TestMemoryReturn:
    """Verify memory return computation."""

    def test_positive(self):
        assert compute_memory_return(0.8) == pytest.approx(0.8)

    def test_negative(self):
        assert compute_memory_return(-1.0) == pytest.approx(-1.0)

    def test_zero(self):
        assert compute_memory_return(0.0) == pytest.approx(0.0)

    def test_severe_failure(self):
        assert compute_memory_return(-1.0) < 0.0


# ======================================================================
# Test 2 — Conversation masks with memory
# ======================================================================

class TestConversationMasksPhase2:
    """Verify conversation masks include memory messages."""

    def test_memory_appears_in_messages(self):
        """Memory message is in the reconstructed conversation."""
        sq = {
            "question": "Q1",
            "agent_steps": [{
                "step_index": 0,
                "type": "tool_call",
                "tool_calls": [{"tool_name": "read", "arguments": {}}],
                "observations": [{"tool_name": "read", "content": "[SUCCESS]", "success": True}],
                "invalid_multi_tool": False,
            }],
            "assistant_answer": {"content": "Answer."},
            "memory_output": {"goal": "g", "key_facts": []},
        }
        messages = _reconstruct_conversation_messages(sq)
        content_types = [m["content_type"] for m in messages]
        assert "memory" in content_types, f"Memory missing from messages: {content_types}"

    def test_memory_message_is_trainable(self):
        """Memory message is trainable (assistant-generated)."""
        sq = {
            "question": "Q1",
            "agent_steps": [],
            "assistant_answer": {"content": "A."},
            "memory_output": {"goal": "g"},
        }
        messages = _reconstruct_conversation_messages(sq)
        mem_msgs = [m for m in messages if m["content_type"] == "memory"]
        assert len(mem_msgs) == 1
        assert mem_msgs[0]["trainable"] is True
        assert mem_msgs[0]["role"] == "assistant"

    def test_no_memory_when_none(self):
        """When memory_output is None, no memory message appears."""
        sq = {
            "question": "Q1",
            "agent_steps": [],
            "assistant_answer": {"content": "A."},
            "memory_output": None,
        }
        messages = _reconstruct_conversation_messages(sq)
        content_types = [m["content_type"] for m in messages]
        assert "memory" not in content_types

    def test_system_and_user_masked(self):
        """System and user messages are not trainable."""
        sq = {
            "question": "Q1",
            "agent_steps": [{
                "step_index": 0,
                "type": "tool_call",
                "tool_calls": [{"tool_name": "r", "arguments": {}}],
                "observations": [{"tool_name": "r", "content": "[SUCCESS]", "success": True}],
                "invalid_multi_tool": False,
            }],
            "assistant_answer": {"content": "A."},
            "memory_output": {"goal": "g"},
        }
        messages = _reconstruct_conversation_messages(sq)
        for m in messages:
            if m["role"] in ("system", "user", "tool"):
                assert m["trainable"] is False, \
                    f"Role {m['role']} should be masked, got trainable=True"
            elif m["role"] == "assistant":
                assert m["trainable"] is True, \
                    f"Role assistant ({m['content_type']}) should be trainable"

    def test_message_sequence_order(self):
        """Messages are in correct order: system, user, tools..., answer, memory."""
        sq = {
            "question": "Q1",
            "agent_steps": [{
                "step_index": 0,
                "type": "tool_call",
                "tool_calls": [{"tool_name": "r", "arguments": {}}],
                "observations": [{"tool_name": "r", "content": "[SUCCESS]", "success": True}],
                "invalid_multi_tool": False,
            }],
            "assistant_answer": {"content": "A."},
            "memory_output": {"goal": "g"},
        }
        messages = _reconstruct_conversation_messages(sq)
        types = [m["content_type"] for m in messages]
        # system first, user second, then tool_call, tool_observation, answer, memory
        assert types[0] == "system_prompt"
        assert types[1] == "user_question"
        assert types[-2] == "answer"
        assert types[-1] == "memory"

    def test_invalid_multi_tool_message_sequence(self):
        """Invalid multi-tool steps produce assistant + error_feedback messages."""
        sq = {
            "question": "Q1",
            "agent_steps": [{
                "step_index": 0,
                "type": "tool_call",
                "tool_calls": [
                    {"tool_name": "t1", "arguments": {}},
                    {"tool_name": "t2", "arguments": {}},
                ],
                "observations": [],
                "invalid_multi_tool": True,
            }],
            "assistant_answer": {"content": "A."},
            "memory_output": {"goal": "g"},
        }
        messages = _reconstruct_conversation_messages(sq)
        types = [m["content_type"] for m in messages]
        assert "error_feedback" in types, f"Expected error_feedback after invalid multi-tool: {types}"


# ======================================================================
# Test 3 — Segment builder with memory
# ======================================================================

class TestSegmentBuilderPhase2:
    """Verify segment building includes memory segments."""

    def test_all_three_segment_types_present(self):
        """Dialog produces tool, answer, and memory segments."""
        rollout = _make_simple_rollout_with_memory()
        builder = SegmentReturnBuilderPhase2()

        result = builder.build([rollout])
        segs = result["segments"]

        types = set(s["segment_type"] for s in segs)
        assert types == {"tool", "answer", "memory"}, f"Expected all 3 types, got {types}"

    def test_memory_segment_advantage(self):
        """Memory segment advantage equals r_memory (before normalisation)."""
        rollout = _make_simple_rollout_with_memory(r_memory=0.7)
        builder = SegmentReturnBuilderPhase2()

        result = builder.build([rollout])
        mem_segs = [s for s in result["segments"] if s["segment_type"] == "memory"]

        assert len(mem_segs) == 1
        assert mem_segs[0]["raw_return"] == 0.7
        # After normalisation with n=1, advantage becomes 0
        assert mem_segs[0]["advantage"] == pytest.approx(0.0)

    def test_memory_segment_advantage_multi_sample(self):
        """With multiple memory segments in same group, advantages vary."""
        rollout1 = _make_simple_rollout_with_memory(r_memory=0.8)
        rollout1["rollout_id"] = "test_sq1_k0"
        rollout2 = _make_simple_rollout_with_memory(r_memory=-0.4)
        rollout2["rollout_id"] = "test_sq1_k1"

        builder = SegmentReturnBuilderPhase2()
        result = builder.build([rollout1, rollout2])
        mem_segs = [s for s in result["segments"] if s["segment_type"] == "memory"]

        assert len(mem_segs) == 2
        # With 2 values, normalisation should produce non-zero advantages
        advantages = [s["advantage"] for s in mem_segs]
        raw = [s["raw_return"] for s in mem_segs]
        # Higher raw → higher normalized
        if raw[0] > raw[1]:
            assert advantages[0] > advantages[1]
        else:
            assert advantages[0] < advantages[1]

    def test_segment_ids_unique(self):
        """All segment IDs are unique."""
        rollout = _make_dialog_rollout(n_sq=2, n_tool_steps_per_sq=2)
        builder = SegmentReturnBuilderPhase2()
        result = builder.build([rollout])
        seg_ids = [s["segment_id"] for s in result["segments"]]
        assert len(seg_ids) == len(set(seg_ids))

    def test_memory_segment_has_content_type(self):
        """Memory segment has content_type='memory'."""
        rollout = _make_simple_rollout_with_memory()
        builder = SegmentReturnBuilderPhase2()
        result = builder.build([rollout])
        mem_segs = [s for s in result["segments"] if s["segment_type"] == "memory"]
        assert mem_segs[0]["content_type"] == "memory"
        assert mem_segs[0]["message_role"] == "assistant"
        assert mem_segs[0]["trainable"] is True

    def test_memory_segment_has_severe_failure_flag(self):
        """Memory segment carries severe_failure flag."""
        rollout = _make_simple_rollout_with_memory()
        # Set severe failure on the subquestion
        rollout["subquestion_rollouts"][0]["memory_severe_failure"] = True
        rollout["subquestion_rollouts"][0]["r_memory"] = -1.0

        builder = SegmentReturnBuilderPhase2()
        result = builder.build([rollout])
        mem_segs = [s for s in result["segments"] if s["segment_type"] == "memory"]

        assert mem_segs[0]["severe_failure"] is True
        assert mem_segs[0]["raw_return"] == -1.0


# ======================================================================
# Test 4 — Normalisation with 3 types
# ======================================================================

class TestNormalisationPhase2:
    """Verify segment-type normalisation with tool/answer/memory."""

    def test_three_types_normalised_separately(self):
        """Each segment type is normalised within its group (GRPO-style)."""
        # Use rollout_ids with _k{N} so they share the same group
        rollout1 = _make_dialog_rollout("test_sq1_k0", n_sq=1, n_tool_steps_per_sq=2)
        rollout2 = _make_dialog_rollout("test_sq1_k1", n_sq=1, n_tool_steps_per_sq=2)

        # Override rewards
        rollout1["subquestion_rollouts"][0]["r_tool_steps"] = [0.1, 0.3]
        rollout1["subquestion_rollouts"][0]["r_answer"] = 0.8
        rollout1["subquestion_rollouts"][0]["r_memory"] = 0.6

        rollout2["subquestion_rollouts"][0]["r_tool_steps"] = [-0.1, -0.3]
        rollout2["subquestion_rollouts"][0]["r_answer"] = -0.2
        rollout2["subquestion_rollouts"][0]["r_memory"] = -0.4

        builder = SegmentReturnBuilderPhase2()
        result = builder.build([rollout1, rollout2])

        stats = result["normalisation"]
        # Keys are "{group}/{type}" after group-relative normalisation
        group_key = "test_sq1_sq1"  # strips _k0/_k1 from rollout_id_sq1
        assert f"{group_key}/tool" in stats
        assert f"{group_key}/answer" in stats
        assert f"{group_key}/memory" in stats

        # Tool: 4 values (2 per rollout) in same group
        assert stats[f"{group_key}/tool"]["count"] == 4
        # Answer: 2 values in same group
        assert stats[f"{group_key}/answer"]["count"] == 2
        # Memory: 2 values in same group
        assert stats[f"{group_key}/memory"]["count"] == 2

    def test_memory_segment_normalised(self):
        """Memory segments are z-score normalised within group."""
        memory_returns = [0.5, 0.1, -0.3]  # three different values
        rollouts = []
        for i, r_mem in enumerate(memory_returns):
            r = _make_simple_rollout_with_memory(r_memory=r_mem)
            r["rollout_id"] = f"test_sq1_k{i}"
            rollouts.append(r)

        builder = SegmentReturnBuilderPhase2()
        result = builder.build(rollouts)
        mem_segs = [s for s in result["segments"] if s["segment_type"] == "memory"]

        # After normalisation, mean should be ~0
        mean_adv = sum(s["advantage"] for s in mem_segs) / len(mem_segs)
        assert abs(mean_adv) < 1e-6, f"Normalised memory advantages should have mean ≈ 0, got {mean_adv}"

        # Raw returns should be preserved
        raw_returns = [s["raw_return"] for s in mem_segs]
        assert sorted(raw_returns) == pytest.approx(sorted(memory_returns))


# ======================================================================
# Test 5 — Z-score normalisation edge cases
# ======================================================================

class TestZNormalizeEdgeCases:
    """Edge cases for normalisation function."""

    def test_empty(self):
        normed, mean, std = _z_normalize([])
        assert normed == []
        assert mean == 0.0
        assert std == 0.0

    def test_single_value(self):
        normed, mean, std = _z_normalize([0.5])
        assert normed == [0.0]
        assert mean == 0.5
        assert std == 0.0

    def test_all_same(self):
        normed, mean, std = _z_normalize([0.3, 0.3, 0.3])
        assert all(v == 0.0 for v in normed)
        assert mean == 0.3
        assert std == 0.0

    def test_different_values(self):
        normed, mean, std = _z_normalize([1.0, 3.0, 5.0])
        assert mean == 3.0
        assert normed[0] < 0
        assert normed[2] > 0


# ======================================================================
# Test 6 — File I/O
# ======================================================================

class TestBuilderFileIO:
    """Verify Phase 2 builder writes files correctly."""

    def test_writes_segment_returns_jsonl(self):
        rollout = _make_dialog_rollout(n_sq=1, n_tool_steps_per_sq=1)
        builder = SegmentReturnBuilderPhase2()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "phase2_segment_returns.jsonl")
            result = builder.build_and_save(
                dialog_rollouts=[rollout],
                output_path=out_path,
            )

            assert os.path.exists(out_path)
            loaded = read_jsonl(out_path)
            assert len(loaded) == 3  # tool + answer + memory

    def test_writes_norm_stats(self):
        rollout = _make_dialog_rollout(n_sq=1, n_tool_steps_per_sq=1)
        builder = SegmentReturnBuilderPhase2()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "phase2_segment_returns.jsonl")
            builder.build_and_save(
                dialog_rollouts=[rollout],
                output_path=out_path,
            )

            norm_path = out_path.replace(".jsonl", "_norm_stats.json")
            assert os.path.exists(norm_path)
            with open(norm_path) as f:
                norm_stats = json.load(f)
            # Keys are "{group}/{type}" after group-relative normalisation
            assert any("/tool" in k for k in norm_stats), f"Missing /tool in {list(norm_stats.keys())}"
            assert any("/answer" in k for k in norm_stats)
            assert any("/memory" in k for k in norm_stats)

    def test_writes_conversation_masks(self):
        rollout = _make_dialog_rollout(n_sq=1, n_tool_steps_per_sq=1)
        builder = SegmentReturnBuilderPhase2()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "phase2_segment_returns.jsonl")
            result = builder.build_and_save(
                dialog_rollouts=[rollout],
                output_path=out_path,
            )

            mask_path = out_path.replace(".jsonl", "_conversation_masks.jsonl")
            assert os.path.exists(mask_path)
            masks = read_jsonl(mask_path)
            assert len(masks) == 1  # One subquestion
            # Check memory message is in the mask
            msg_types = [m["content_type"] for m in masks[0]["messages"]]
            assert "memory" in msg_types


# ======================================================================
# Test 7 — Multi-subquestion dialog
# ======================================================================

class TestMultiSQDialog:
    """Verify multi-subquestion dialog segment building."""

    def test_two_subquestions_produce_segments(self):
        """2 subquestions → 2*(tool + answer + memory) segments."""
        rollout = _make_dialog_rollout(n_sq=2, n_tool_steps_per_sq=1)
        builder = SegmentReturnBuilderPhase2()
        result = builder.build([rollout])

        segs = result["segments"]
        n_tool = sum(1 for s in segs if s["segment_type"] == "tool")
        n_answer = sum(1 for s in segs if s["segment_type"] == "answer")
        n_memory = sum(1 for s in segs if s["segment_type"] == "memory")

        assert n_tool == 2  # 1 per subquestion
        assert n_answer == 2
        assert n_memory == 2
        assert len(segs) == 6

    def test_two_subquestions_produce_two_masks(self):
        """Each subquestion produces its own conversation mask."""
        rollout = _make_dialog_rollout(n_sq=2, n_tool_steps_per_sq=1)
        builder = SegmentReturnBuilderPhase2()
        result = builder.build([rollout])

        masks = result["conversation_masks"]
        assert len(masks) == 2
        assert masks[0]["sq_id"] == 1
        assert masks[1]["sq_id"] == 2

    def test_rollout_ids_differ_by_sq(self):
        """Different subquestions get different rollout_ids."""
        rollout = _make_dialog_rollout(n_sq=2, n_tool_steps_per_sq=1)
        builder = SegmentReturnBuilderPhase2()
        result = builder.build([rollout])

        sq1_rollout_ids = set(s["rollout_id"] for s in result["segments"]
                              if "sq1" in s["rollout_id"])
        sq2_rollout_ids = set(s["rollout_id"] for s in result["segments"]
                              if "sq2" in s["rollout_id"])
        assert len(sq1_rollout_ids) > 0
        assert len(sq2_rollout_ids) > 0
        assert sq1_rollout_ids != sq2_rollout_ids


# ======================================================================
# Test 8 — Tool return-to-go formula verification
# ======================================================================

class TestToolReturnToGoVerification:
    """Verify the tool return-to-go formula for Phase 2 data."""

    def test_formula_with_memory_reward(self):
        """Tool return-to-go does NOT include memory reward (only answer reward)."""
        r_tool = [0.1, 0.2]
        r_answer = 0.5
        r_memory = 0.7

        G = compute_tool_return_to_go(r_tool, r_answer, gamma_tool=0.95, kappa_ans=1.0)

        # G[1] = r_tool[1] + kappa * gamma^1 * r_answer
        #       = 0.2 + 1.0 * 0.95 * 0.5 = 0.2 + 0.475 = 0.675
        expected_G1 = 0.2 + 1.0 * (0.95 ** 1) * 0.5
        assert G[1] == pytest.approx(expected_G1)

        # Memory reward is NOT included in tool return-to-go
        assert len(G) == 2  # Only tool steps, no memory step

    def test_gamma_decay(self):
        """Earlier steps discount future rewards more."""
        r_tool = [0.0, 0.0, 0.0]
        r_answer = 1.0

        G = compute_tool_return_to_go(r_tool, r_answer, gamma_tool=0.5, kappa_ans=1.0)

        # G[0] = r_answer * gamma^3 = 1.0 * 0.125 = 0.125
        # G[1] = r_answer * gamma^2 = 1.0 * 0.25 = 0.25
        # G[2] = r_answer * gamma^1 = 1.0 * 0.5 = 0.5
        assert G[0] < G[1] < G[2]
        assert G[0] == pytest.approx(0.125)
        assert G[1] == pytest.approx(0.25)
        assert G[2] == pytest.approx(0.5)


# ======================================================================
# Test 9 — Edge cases
# ======================================================================

class TestPhase2BuilderEdgeCases:
    """Edge cases for Phase 2 segment builder."""

    def test_empty_dialog_rollouts(self):
        """Empty input produces empty segments."""
        builder = SegmentReturnBuilderPhase2()
        result = builder.build([])
        assert len(result["segments"]) == 0
        assert len(result["conversation_masks"]) == 0

    def test_no_tool_steps(self):
        """Subquestion with no tool steps still produces answer + memory segments."""
        rollout = {
            "sample_id": "test",
            "rollout_id": "test_k0",
            "n_subquestions": 1,
            "subquestion_rollouts": [{
                "sq_id": 1,
                "question": "Q1",
                "status": "completed",
                "agent_steps": [],
                "assistant_answer": {"content": "Direct answer."},
                "memory_before": None,
                "memory_output": {"goal": "g", "key_facts": []},
                "memory_severe_failure": False,
                "r_tool_steps": [],
                "r_answer": 0.0,
                "r_memory": 0.3,
            }],
        }

        builder = SegmentReturnBuilderPhase2()
        result = builder.build([rollout])
        segs = result["segments"]

        types = set(s["segment_type"] for s in segs)
        assert types == {"answer", "memory"}
        assert len(segs) == 2

    def test_no_answer_truncated(self):
        """Truncated subquestion with no answer produces only tool segments."""
        rollout = {
            "sample_id": "test",
            "rollout_id": "test_k0",
            "n_subquestions": 1,
            "subquestion_rollouts": [{
                "sq_id": 1,
                "question": "Q1",
                "status": "truncated",
                "agent_steps": [{
                    "step_index": 0,
                    "type": "tool_call",
                    "tool_calls": [{"tool_name": "r", "arguments": {}}],
                    "observations": [{"tool_name": "r", "content": "[SUCCESS]", "success": True}],
                    "invalid_multi_tool": False,
                }],
                "assistant_answer": None,
                "memory_before": None,
                "memory_output": None,
                "memory_severe_failure": False,
                "r_tool_steps": [-0.02],
                "r_answer": 0.0,
                "r_memory": 0.0,
            }],
        }

        builder = SegmentReturnBuilderPhase2()
        result = builder.build([rollout])
        segs = result["segments"]

        types = set(s["segment_type"] for s in segs)
        assert types == {"tool"}
        assert len(segs) == 1

    def test_severe_failure_memory_segment(self):
        """Severe memory failure produces r_memory = -1 segment."""
        rollout = _make_simple_rollout_with_memory(r_memory=-1.0)
        rollout["subquestion_rollouts"][0]["memory_severe_failure"] = True

        builder = SegmentReturnBuilderPhase2()
        result = builder.build([rollout])
        mem_segs = [s for s in result["segments"] if s["segment_type"] == "memory"]

        assert len(mem_segs) == 1
        assert mem_segs[0]["severe_failure"] is True
        assert mem_segs[0]["raw_return"] == -1.0

    def test_invalid_multi_tool_segments(self):
        """Invalid multi-tool steps produce tool segments with invalid_multi_tool=True."""
        rollout = {
            "sample_id": "test",
            "rollout_id": "test_k0",
            "n_subquestions": 1,
            "subquestion_rollouts": [{
                "sq_id": 1,
                "question": "Q1",
                "status": "invalid_retry",
                "agent_steps": [{
                    "step_index": 0,
                    "type": "tool_call",
                    "tool_calls": [
                        {"tool_name": "t1", "arguments": {}},
                        {"tool_name": "t2", "arguments": {}},
                    ],
                    "observations": [],
                    "invalid_multi_tool": True,
                }],
                "assistant_answer": None,
                "memory_before": None,
                "memory_output": None,
                "r_tool_steps": [-1.0],
                "r_answer": 0.0,
                "r_memory": 0.0,
            }],
        }

        builder = SegmentReturnBuilderPhase2()
        result = builder.build([rollout])
        tool_segs = [s for s in result["segments"] if s["segment_type"] == "tool"]

        assert len(tool_segs) == 1
        assert tool_segs[0]["invalid_multi_tool"] is True
        assert tool_segs[0]["raw_return"] == -1.0
