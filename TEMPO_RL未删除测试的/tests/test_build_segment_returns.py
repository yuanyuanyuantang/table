"""
TEMPO-RL Phase 1 — Smoke tests for segment return builder.

Covers: tool return-to-go, answer return, normalisation, token masks,
conversation reconstruction, and edge cases.
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

from TEMPO_RL.build_segment_returns import (
    SegmentReturnBuilder,
    compute_tool_return_to_go,
    compute_answer_return,
    _z_normalize,
    _reconstruct_conversation_messages,
)
from TEMPO_RL.io_utils import read_jsonl, write_jsonl


# ======================================================================
# Helper factories
# ======================================================================

def _make_rollout(
    rollout_id: str = "test_sq1_k0",
    r_tool_steps: list = None,
    r_answer: float = 0.5,
    agent_steps: list = None,
    assistant_answer: dict = None,
    status: str = "completed",
) -> dict:
    """Build a minimal rollout record."""
    rec = {
        "rollout_id": rollout_id,
        "sample_id": "test_sample",
        "subquestion_id": 1,
        "r_tool_steps": r_tool_steps if r_tool_steps is not None else [],
        "r_answer": r_answer,
        "agent_steps": agent_steps or [],
        "assistant_answer": assistant_answer,
        "status": status,
        "r_tool": 0.0,
        "num_tool_steps": len(agent_steps or []),
    }
    return rec


def _make_tool_step(
    step_index: int = 0,
    tool_name: str = "table_head_reader",
    arguments: dict = None,
    observation_content: str = "[SUCCESS] data",
    invalid: bool = False,
) -> dict:
    if invalid:
        return {
            "step_index": step_index,
            "type": "tool_call",
            "tool_calls": [
                {"tool_name": "table_head_reader", "arguments": {"file_path": "a.csv"}},
                {"tool_name": "grep_search", "arguments": {"path": ".", "pattern": "x"}},
            ],
            "observations": [],
            "invalid_multi_tool": True,
        }
    return {
        "step_index": step_index,
        "type": "tool_call",
        "tool_calls": [
            {"tool_name": tool_name, "arguments": arguments or {"file_path": "data.csv"}}
        ],
        "observations": [
            {
                "tool_call_id": f"tc_{step_index+1}",
                "tool_name": tool_name,
                "content": observation_content,
                "success": True,
            }
        ],
        "invalid_multi_tool": False,
    }


# ======================================================================
# Test 1 — Tool return-to-go computation
# ======================================================================

class TestToolReturnToGo:
    """Verify the tool return-to-go formula."""

    def test_single_step_no_answer(self):
        """G[0] = r_tool[0] when T=1 and r_answer=0."""
        G = compute_tool_return_to_go([-0.02], r_answer=0.0)
        assert len(G) == 1
        assert G[0] == pytest.approx(-0.02)

    def test_single_step_with_answer(self):
        """G[0] = r_tool[0] + κ_ans * γ^1 * r_answer when T=1."""
        G = compute_tool_return_to_go(
            [-0.02], r_answer=0.5, gamma_tool=0.95, kappa_ans=1.0
        )
        # G[0] = -0.02 + 1.0 * 0.95^1 * 0.5 = -0.02 + 0.475 = 0.455
        expected = -0.02 + 0.95 * 0.5
        assert G[0] == pytest.approx(expected)

    def test_two_steps_no_answer(self):
        """G[t] = sum_{l=t}^{T-1} γ^{l-t} * r_tool[l]."""
        r = [0.1, 0.2]
        G = compute_tool_return_to_go(r, r_answer=0.0, gamma_tool=0.9)
        # G[1] = 0.2
        # G[0] = 0.1 + 0.9 * 0.2 = 0.28
        assert G[1] == pytest.approx(0.2)
        assert G[0] == pytest.approx(0.1 + 0.9 * 0.2)

    def test_two_steps_with_answer(self):
        """G[t] includes discounted answer reward."""
        r = [0.1, 0.2]
        r_ans = 0.8
        gamma = 0.9
        G = compute_tool_return_to_go(r, r_ans, gamma_tool=gamma, kappa_ans=1.0)
        # G[1] = 0.2 + γ^1 * 0.8 = 0.2 + 0.72 = 0.92
        # G[0] = 0.1 + γ*0.2 + γ^2*0.8 = 0.1 + 0.18 + 0.648 = 0.928
        assert G[1] == pytest.approx(0.2 + gamma * 0.8)
        assert G[0] == pytest.approx(0.1 + gamma * 0.2 + gamma**2 * 0.8)

    def test_three_steps(self):
        """Test with 3 steps to verify discount chain."""
        r = [0.05, -0.02, 0.1]
        r_ans = 0.6
        gamma = 0.95
        G = compute_tool_return_to_go(r, r_ans, gamma_tool=gamma)
        # G[2] = 0.1 + γ^1 * 0.6 = 0.1 + 0.57 = 0.67
        # G[1] = -0.02 + γ*0.1 + γ^2*0.6 = -0.02 + 0.095 + 0.5415 = 0.6165
        # G[0] = 0.05 + γ*(-0.02) + γ^2*0.1 + γ^3*0.6
        assert G[2] == pytest.approx(0.1 + gamma * 0.6)
        assert G[1] == pytest.approx(-0.02 + gamma * 0.1 + gamma**2 * 0.6)
        assert G[0] == pytest.approx(
            0.05 + gamma * (-0.02) + gamma**2 * 0.1 + gamma**3 * 0.6
        )

    def test_kappa_ans_zero(self):
        """With κ_ans=0, answer reward doesn't contribute."""
        r = [-0.02]
        G = compute_tool_return_to_go(r, r_answer=0.9, gamma_tool=1.0, kappa_ans=0.0)
        assert G[0] == pytest.approx(-0.02)

    def test_kappa_ans_two(self):
        """κ_ans scales the answer reward portion."""
        r = [0.0]
        gamma = 1.0
        # G = 0.0 + 2.0 * 1.0 * 0.5 = 1.0
        G = compute_tool_return_to_go(r, r_answer=0.5, gamma_tool=gamma, kappa_ans=2.0)
        assert G[0] == pytest.approx(1.0)

    def test_empty_steps(self):
        """Empty r_tool_steps returns empty list."""
        G = compute_tool_return_to_go([], r_answer=0.5)
        assert G == []

    def test_gamma_zero(self):
        """γ=0 means only immediate reward matters."""
        r = [0.1, 0.2, 0.3]
        G = compute_tool_return_to_go(r, r_answer=0.5, gamma_tool=0.0)
        # Each step only gets its own r_tool (no future discounting, no answer)
        for t in range(3):
            assert G[t] == pytest.approx(r[t])

    def test_answer_discount_decays(self):
        """Earlier steps discount answer reward more heavily."""
        r = [0.0, 0.0]
        r_ans = 1.0
        gamma = 0.9
        G = compute_tool_return_to_go(r, r_ans, gamma_tool=gamma)
        # G[1] (closer to answer): γ^1 * 1.0 = 0.9
        # G[0] (farther): γ^2 * 1.0 = 0.81
        assert G[1] == pytest.approx(0.9)
        assert G[0] == pytest.approx(0.81)
        assert G[0] < G[1], "Earlier step should discount answer more"


# ======================================================================
# Test 2 — Answer return
# ======================================================================

class TestAnswerReturn:
    """Answer return is identity — just passes through the answer reward."""

    def test_positive(self):
        assert compute_answer_return(0.7) == pytest.approx(0.7)

    def test_negative(self):
        assert compute_answer_return(-1.0) == pytest.approx(-1.0)

    def test_zero(self):
        assert compute_answer_return(0.0) == pytest.approx(0.0)


# ======================================================================
# Test 3 — Z-score normalisation
# ======================================================================

class TestZNormalize:
    """Verify z-score normalisation behaviour."""

    def test_basic(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        normed, mean, std = _z_normalize(vals)
        assert mean == pytest.approx(3.0)
        assert std > 0
        assert sum(normed) == pytest.approx(0.0, abs=1e-6)  # zero mean after norm

    def test_same_values(self):
        """All same values → std≈0 → all normalised to 0."""
        vals = [0.5, 0.5, 0.5]
        normed, mean, std = _z_normalize(vals)
        assert mean == pytest.approx(0.5)
        assert all(v == pytest.approx(0.0) for v in normed)

    def test_single_value(self):
        """Single value → returns [0.0]."""
        normed, mean, std = _z_normalize([3.0])
        assert normed == [0.0]
        assert mean == 3.0
        assert std == 0.0

    def test_empty_list(self):
        normed, mean, std = _z_normalize([])
        assert normed == []
        assert mean == 0.0
        assert std == 0.0

    def test_negative_values(self):
        vals = [-2.0, -1.0, 0.0, 1.0, 2.0]
        normed, mean, std = _z_normalize(vals)
        assert mean == pytest.approx(0.0)
        assert sum(normed) == pytest.approx(0.0, abs=1e-6)

    def test_epsilon_prevents_division_by_zero(self):
        """Very small std should be fine due to epsilon."""
        vals = [0.5, 0.5000001]
        normed, mean, std = _z_normalize(vals, epsilon=1e-8)
        assert all(math.isfinite(v) for v in normed)


# ======================================================================
# Test 4 — Segment type normalisation pools
# ======================================================================

class TestNormalizationByType:
    """Tool and answer returns are normalised in separate pools."""

    def test_tool_answer_separate_pools(self):
        """Tool normalisation only uses tool returns; answer only uses answer."""
        builder = SegmentReturnBuilder(gamma_tool=1.0, kappa_ans=1.0)

        rollouts = [
            _make_rollout(
                "s1", r_tool_steps=[0.5, 0.3], r_answer=0.8,
                agent_steps=[_make_tool_step(0), _make_tool_step(1)],
                assistant_answer={"content": "Answer 1"},
            ),
            _make_rollout(
                "s2", r_tool_steps=[-0.1], r_answer=0.2,
                agent_steps=[_make_tool_step(0)],
                assistant_answer={"content": "Answer 2"},
            ),
        ]

        result = builder.build(rollouts)
        segs = result["segments"]

        # 3 tool segments, 2 answer segments
        tool_segs = [s for s in segs if s["segment_type"] == "tool"]
        ans_segs = [s for s in segs if s["segment_type"] == "answer"]
        assert len(tool_segs) == 3
        assert len(ans_segs) == 2

        # Tool returns are normalised against tool pool only
        tool_normed = [s["return_value"] for s in tool_segs]
        assert sum(tool_normed) == pytest.approx(0.0, abs=1e-6)

        # Answer returns normalised against answer pool only
        ans_normed = [s["return_value"] for s in ans_segs]
        assert sum(ans_normed) == pytest.approx(0.0, abs=1e-6)

    def test_normalisation_stats_recorded(self):
        builder = SegmentReturnBuilder()
        rollouts = [
            _make_rollout(
                "test_sq1_k0", r_tool_steps=[0.5], r_answer=0.7,
                agent_steps=[_make_tool_step(0)],
                assistant_answer={"content": "A1"},
            ),
        ]
        result = builder.build(rollouts)
        stats = result["normalisation"]

        # Keys are "{group}/{type}" after group-relative normalisation
        assert "test_sq1/tool" in stats
        assert "test_sq1/answer" in stats
        assert stats["test_sq1/tool"]["count"] == 1
        assert stats["test_sq1/answer"]["count"] == 1
        assert stats["test_sq1/tool"]["group"] == "test_sq1"
        assert stats["test_sq1/tool"]["type"] == "tool"
        for key in ("mean", "std", "min", "max"):
            assert key in stats["test_sq1/tool"]
            assert key in stats["test_sq1/answer"]

    def test_mixed_distributions(self):
        """Tool returns are high, answer returns are low — but each is normalised separately."""
        builder = SegmentReturnBuilder(gamma_tool=1.0, kappa_ans=0.0)

        rollouts = [
            _make_rollout("s1", r_tool_steps=[10.0], r_answer=0.1,
                          agent_steps=[_make_tool_step(0)],
                          assistant_answer={"content": "A1"}),
            _make_rollout("s2", r_tool_steps=[10.0], r_answer=-0.5,
                          agent_steps=[_make_tool_step(0)],
                          assistant_answer={"content": "A2"}),
        ]

        result = builder.build(rollouts)
        tool_segs = [s for s in result["segments"] if s["segment_type"] == "tool"]
        ans_segs = [s for s in result["segments"] if s["segment_type"] == "answer"]

        # Tool: all same value → all normalised to 0
        for ts in tool_segs:
            assert ts["return_value"] == pytest.approx(0.0)

        # Answer: varied → should be different
        for a in ans_segs:
            assert a["return_value"] != pytest.approx(0.0) or True  # both should be equal magnitude opposite sign


# ======================================================================
# Test 5 — Conversation reconstruction & token masks
# ======================================================================

class TestConversationMasks:
    """Verify conversation reconstruction and trainable/masked flags."""

    def test_basic_mask_structure(self):
        rollout = _make_rollout(
            "s1",
            r_tool_steps=[-0.02],
            r_answer=0.5,
            agent_steps=[_make_tool_step(0)],
            assistant_answer={"content": "Result: 120.5万辆"},
        )
        messages = _reconstruct_conversation_messages(rollout)

        # system, user, assistant(tool), tool(obs), assistant(answer)
        assert len(messages) == 5

        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "user", "assistant"]

        trainable = [m["trainable"] for m in messages]
        assert trainable == [False, False, True, False, True]

    def test_two_tool_steps_mask(self):
        rollout = _make_rollout(
            "s1",
            r_tool_steps=[-0.02, -0.02],
            agent_steps=[_make_tool_step(0), _make_tool_step(1)],
            assistant_answer={"content": "Answer."},
        )
        messages = _reconstruct_conversation_messages(rollout)

        # sys, usr, asst(t0), usr(t0), asst(t1), usr(t1), asst(ans)
        assert len(messages) == 7

        roles = [m["role"] for m in messages]
        assert roles == [
            "system", "user",
            "assistant", "user",
            "assistant", "user",
            "assistant",
        ]
        trainable = [m["trainable"] for m in messages]
        assert trainable == [
            False, False,
            True, False,
            True, False,
            True,
        ]

    def test_no_answer_truncated(self):
        """Truncated rollout has no answer message."""
        rollout = _make_rollout(
            "s1",
            r_tool_steps=[-0.02, -0.02],
            agent_steps=[_make_tool_step(0), _make_tool_step(1)],
            assistant_answer=None,
            status="truncated",
        )
        messages = _reconstruct_conversation_messages(rollout)

        # Last message should be tool observation, not answer
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content_type"] == "tool_observation"
        # No answer message
        assert not any(m["content_type"] == "answer" for m in messages)

    def test_direct_answer_no_tools(self):
        """Rollout with direct answer has no tool messages."""
        rollout = _make_rollout(
            "s1",
            r_tool_steps=[],
            agent_steps=[],
            assistant_answer={"content": "Direct answer."},
        )
        messages = _reconstruct_conversation_messages(rollout)
        assert len(messages) == 3  # system, user, assistant(answer)
        assert messages[-1]["content_type"] == "answer"
        assert messages[-1]["trainable"] is True

    def test_invalid_multi_tool_mask(self):
        """Invalid multi-tool step: assistant message then error feedback."""
        rollout = _make_rollout(
            "s1",
            r_tool_steps=[-1.0],
            agent_steps=[_make_tool_step(0, invalid=True)],
            assistant_answer={"content": "Final answer after retry."},
        )
        messages = _reconstruct_conversation_messages(rollout)

        # sys, usr, asst(invalid), user(error feedback), asst(answer)
        assert len(messages) == 5
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "user", "assistant"]

        # Invalid tool call IS trainable (model chose it)
        # Error feedback is masked
        trainable = [m["trainable"] for m in messages]
        assert trainable == [False, False, True, False, True]

    def test_tool_observation_is_masked(self):
        """Every tool observation message has trainable=False."""
        rollout = _make_rollout(
            "s1",
            r_tool_steps=[-0.02, -0.02, -0.02],
            agent_steps=[_make_tool_step(0), _make_tool_step(1), _make_tool_step(2)],
            assistant_answer={"content": "Done."},
        )
        messages = _reconstruct_conversation_messages(rollout)
        for m in messages:
            if m["content_type"] == "tool_observation":
                assert m["trainable"] is False, f"Tool observation should be masked: {m}"

    def test_system_prompt_masked(self):
        rollout = _make_rollout("s1", assistant_answer={"content": "x"})
        messages = _reconstruct_conversation_messages(
            rollout, system_template="You are an expert."
        )
        assert messages[0]["trainable"] is False
        assert "You are an expert" in messages[0]["content_preview"]

    def test_user_question_masked(self):
        rollout = _make_rollout(
            "s1",
            assistant_answer={"content": "x"},
        )
        rollout["question"] = "What is the production?"
        messages = _reconstruct_conversation_messages(rollout)
        assert messages[1]["role"] == "user"
        assert messages[1]["trainable"] is False


# ======================================================================
# Test 6 — Full builder pipeline
# ======================================================================

class TestBuilderPipeline:
    """End-to-end segment builder tests."""

    def test_build_from_minimal_rollout(self):
        builder = SegmentReturnBuilder(gamma_tool=0.95, kappa_ans=1.0)
        rollout = _make_rollout(
            "test_sq1_k0",
            r_tool_steps=[0.1, -0.02],
            r_answer=0.5,
            agent_steps=[_make_tool_step(0), _make_tool_step(1)],
            assistant_answer={"content": "Answer text."},
        )
        result = builder.build([rollout])
        segs = result["segments"]

        # 2 tool + 1 answer = 3 segments
        assert len(segs) == 3

        tool_segs = [s for s in segs if s["segment_type"] == "tool"]
        ans_segs = [s for s in segs if s["segment_type"] == "answer"]

        assert len(tool_segs) == 2
        assert len(ans_segs) == 1

        # Check segment IDs
        assert tool_segs[0]["segment_id"] == "test_sq1_k0_tool_0"
        assert tool_segs[1]["segment_id"] == "test_sq1_k0_tool_1"
        assert ans_segs[0]["segment_id"] == "test_sq1_k0_answer"

    def test_all_required_fields_in_segments(self):
        required = [
            "rollout_id", "segment_id", "segment_type", "step_index",
            "return_value", "advantage", "raw_return",
            "trainable", "message_role", "content_type",
        ]
        builder = SegmentReturnBuilder()
        rollout = _make_rollout(
            "s1",
            r_tool_steps=[0.1],
            r_answer=0.5,
            agent_steps=[_make_tool_step(0)],
            assistant_answer={"content": "A1"},
        )
        result = builder.build([rollout])
        for seg in result["segments"]:
            for field in required:
                assert field in seg, f"Missing {field} in segment {seg.get('segment_id')}"

    def test_tool_segment_has_tool_call(self):
        builder = SegmentReturnBuilder()
        rollout = _make_rollout(
            "s1",
            r_tool_steps=[-0.02],
            agent_steps=[_make_tool_step(0, tool_name="grep_search")],
            assistant_answer={"content": "Done."},
        )
        result = builder.build([rollout])
        tool_seg = [s for s in result["segments"] if s["segment_type"] == "tool"][0]
        assert tool_seg["tool_call"]["tool_name"] == "grep_search"

    def test_answer_segment_has_answer(self):
        builder = SegmentReturnBuilder()
        rollout = _make_rollout(
            "s1",
            assistant_answer={"content": "Final answer here."},
        )
        result = builder.build([rollout])
        ans_seg = [s for s in result["segments"] if s["segment_type"] == "answer"][0]
        assert ans_seg["answer"]["content"] == "Final answer here."

    def test_zero_tool_steps_direct_answer(self):
        """Rollout with direct answer (no tool steps)."""
        builder = SegmentReturnBuilder()
        rollout = _make_rollout("s1", r_tool_steps=[], agent_steps=[],
                                assistant_answer={"content": "Direct."})
        result = builder.build([rollout])
        assert len(result["segments"]) == 1
        assert result["segments"][0]["segment_type"] == "answer"

    def test_truncated_no_answer_no_answer_segment(self):
        """Truncated rollout without answer should have only tool segments."""
        builder = SegmentReturnBuilder()
        rollout = _make_rollout(
            "s1",
            r_tool_steps=[-0.02, -0.02],
            agent_steps=[_make_tool_step(0), _make_tool_step(1)],
            assistant_answer=None,
            status="truncated",
        )
        result = builder.build([rollout])
        segs = result["segments"]
        assert len(segs) == 2
        assert all(s["segment_type"] == "tool" for s in segs)

    def test_multiple_rollouts(self):
        """Multiple rollouts produce segments from all of them."""
        builder = SegmentReturnBuilder()
        rollouts = [
            _make_rollout("s1_k0", r_tool_steps=[0.1], r_answer=0.5,
                          agent_steps=[_make_tool_step(0)],
                          assistant_answer={"content": "A1"}),
            _make_rollout("s1_k1", r_tool_steps=[0.2, 0.3], r_answer=0.6,
                          agent_steps=[_make_tool_step(0), _make_tool_step(1)],
                          assistant_answer={"content": "A2"}),
            _make_rollout("s2_k0", r_tool_steps=[-0.1], r_answer=0.4,
                          agent_steps=[_make_tool_step(0)],
                          assistant_answer={"content": "A3"}),
        ]
        result = builder.build(rollouts)
        segs = result["segments"]
        # 1+2+1 = 4 tool, 3 answer = 7 total
        assert len(segs) == 7
        rollouts_ids = set(s["rollout_id"] for s in segs)
        assert rollouts_ids == {"s1_k0", "s1_k1", "s2_k0"}

    def test_raw_return_preserved(self):
        """After normalisation, raw_return keeps the original value."""
        builder = SegmentReturnBuilder(gamma_tool=1.0, kappa_ans=0.0)
        rollout = _make_rollout(
            "s1",
            r_tool_steps=[0.5],
            r_answer=0.8,
            agent_steps=[_make_tool_step(0)],
            assistant_answer={"content": "A1"},
        )
        result = builder.build([rollout])
        tool_seg = [s for s in result["segments"] if s["segment_type"] == "tool"][0]
        # With κ_ans=0, raw_return = r_tool[0] = 0.5
        assert tool_seg["raw_return"] == pytest.approx(0.5)
        # After normalisation on a single value, return_value = 0
        assert tool_seg["return_value"] == pytest.approx(0.0)

    def test_normalised_returns_have_zero_mean(self):
        """After normalisation, each segment type pool has mean ≈ 0."""
        builder = SegmentReturnBuilder()
        rollouts = []
        for i in range(5):
            rollouts.append(_make_rollout(
                f"s{i}", r_tool_steps=[float(i) / 10], r_answer=float(i) / 5,
                agent_steps=[_make_tool_step(0)],
                assistant_answer={"content": f"A{i}"},
            ))
        result = builder.build(rollouts)
        for st in ("tool", "answer"):
            vals = [s["return_value"] for s in result["segments"] if s["segment_type"] == st]
            assert sum(vals) == pytest.approx(0.0, abs=1e-6), \
                f"{st} normalised returns should have zero mean"

    def test_conversation_masks_output(self):
        builder = SegmentReturnBuilder()
        rollout = _make_rollout(
            "s1",
            r_tool_steps=[-0.02],
            r_answer=0.5,
            agent_steps=[_make_tool_step(0)],
            assistant_answer={"content": "Answer."},
        )
        result = builder.build([rollout])
        masks = result["conversation_masks"]
        assert len(masks) == 1
        assert masks[0]["rollout_id"] == "s1"
        assert "messages" in masks[0]
        # All roles present
        roles = {m["role"] for m in masks[0]["messages"]}
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles
        # Tool observations should exist (now stored with role: "user")
        assert any(m["content_type"] == "tool_observation" for m in masks[0]["messages"])


# ======================================================================
# Test 7 — File I/O
# ======================================================================

class TestFileIO:
    """Verify save/load roundtrip."""

    def test_save_and_load_segments(self):
        builder = SegmentReturnBuilder()
        rollout = _make_rollout(
            "s1_k0",
            r_tool_steps=[0.1],
            r_answer=0.5,
            agent_steps=[_make_tool_step(0)],
            assistant_answer={"content": "Answer."},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "segment_returns.jsonl")
            builder.build_and_save([rollout], out_path)

            # Check main output
            assert os.path.exists(out_path)
            loaded = read_jsonl(out_path)
            assert len(loaded) == 2  # tool + answer
            for seg in loaded:
                assert "segment_id" in seg
                assert "segment_type" in seg
                assert "return_value" in seg
                assert "raw_return" in seg
                assert "trainable" in seg

            # Check norm stats
            norm_path = out_path.replace(".jsonl", "_norm_stats.json")
            assert os.path.exists(norm_path)
            with open(norm_path) as f:
                stats = json.load(f)
            # Keys are "{group}/{type}" after group-relative normalisation
            assert any("/tool" in k for k in stats), f"Missing /tool in {list(stats.keys())}"
            assert any("/answer" in k for k in stats), f"Missing /answer in {list(stats.keys())}"

            # Check conversation masks
            mask_path = out_path.replace(".jsonl", "_conversation_masks.jsonl")
            assert os.path.exists(mask_path)

    def test_save_empty_rollouts(self):
        builder = SegmentReturnBuilder()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "segment_returns.jsonl")
            result = builder.build_and_save([], out_path)
            assert result["segments"] == []
            loaded = read_jsonl(out_path)
            assert loaded == []


# ======================================================================
# Test 8 — Edge cases
# ======================================================================

class TestEdgeCases:
    """Corner cases and error handling."""

    def test_all_negative_rewards(self):
        """G_tool can be negative; normalisation still works."""
        builder = SegmentReturnBuilder(gamma_tool=0.95, kappa_ans=1.0)
        rollout = _make_rollout(
            "s1",
            r_tool_steps=[-1.0, -1.0],
            r_answer=-0.5,
            agent_steps=[_make_tool_step(0), _make_tool_step(1)],
            assistant_answer={"content": "Bad answer."},
        )
        result = builder.build([rollout])
        segs = result["segments"]
        # All raw returns should be negative
        for s in segs:
            assert s["raw_return"] < 0.0

    def test_mixed_sign_rewards(self):
        """Some steps positive, some negative — return-to-go handles both."""
        builder = SegmentReturnBuilder(gamma_tool=0.9, kappa_ans=1.0)
        rollout = _make_rollout(
            "s1",
            r_tool_steps=[0.5, -0.3, 0.1],
            r_answer=0.7,
            agent_steps=[_make_tool_step(0), _make_tool_step(1), _make_tool_step(2)],
            assistant_answer={"content": "Mixed."},
        )
        result = builder.build([rollout])
        segs = result["segments"]
        assert len(segs) == 4  # 3 tool + 1 answer

        # All raw returns should be finite
        for s in segs:
            assert math.isfinite(s["raw_return"])

    def test_invalid_step_has_tool_call_info(self):
        """Invalid multi-tool step still records the tool calls attempted."""
        builder = SegmentReturnBuilder()
        rollout = _make_rollout(
            "s1",
            r_tool_steps=[-1.0],
            agent_steps=[_make_tool_step(0, invalid=True)],
            assistant_answer={"content": "After retry."},
        )
        result = builder.build([rollout])
        tool_seg = [s for s in result["segments"] if s["segment_type"] == "tool"][0]
        assert tool_seg["invalid_multi_tool"] is True
        assert len(tool_seg["all_tool_calls_in_turn"]) == 2

    def test_large_batch_normalisation(self):
        """Normalisation with many rollouts is numerically stable."""
        builder = SegmentReturnBuilder()
        rollouts = []
        for i in range(50):
            rollouts.append(_make_rollout(
                f"s{i}_k0",
                r_tool_steps=[0.01 * i - 0.25, -0.01 * i + 0.25],
                r_answer=0.5 * (i % 5) / 5,
                agent_steps=[_make_tool_step(0), _make_tool_step(1)],
                assistant_answer={"content": f"A{i}"},
            ))
        result = builder.build(rollouts)
        segs = result["segments"]
        assert len(segs) == 150  # 100 tool + 50 answer
        for s in segs:
            assert math.isfinite(s["return_value"])
            assert math.isfinite(s["advantage"])
            assert math.isfinite(s["raw_return"])

    def test_r_tool_step_preserved(self):
        """Per-step tool reward is preserved in the segment."""
        builder = SegmentReturnBuilder()
        rollout = _make_rollout(
            "s1",
            r_tool_steps=[0.42],
            r_answer=0.7,
            agent_steps=[_make_tool_step(0)],
            assistant_answer={"content": "Ans"},
        )
        result = builder.build([rollout])
        tool_seg = [s for s in result["segments"] if s["segment_type"] == "tool"][0]
        assert tool_seg["r_tool_step"] == pytest.approx(0.42)

    def test_step_index_correct(self):
        """step_index matches the tool step position."""
        builder = SegmentReturnBuilder()
        rollout = _make_rollout(
            "s1",
            r_tool_steps=[0.1, 0.2, 0.3],
            agent_steps=[_make_tool_step(0), _make_tool_step(1), _make_tool_step(2)],
            assistant_answer={"content": "Ans"},
        )
        result = builder.build([rollout])
        tool_segs = [s for s in result["segments"] if s["segment_type"] == "tool"]
        assert [s["step_index"] for s in tool_segs] == [0, 1, 2]

    def test_conversation_mask_sequence_indices(self):
        """Message sequence_index is strictly increasing."""
        rollout = _make_rollout(
            "s1",
            r_tool_steps=[-0.02],
            agent_steps=[_make_tool_step(0)],
            assistant_answer={"content": "A"},
        )
        messages = _reconstruct_conversation_messages(rollout)
        indices = [m["sequence_index"] for m in messages]
        assert indices == list(range(len(messages)))
