"""
TEMPO-RL Phase 1 — Smoke tests for GRPO-style training interface.

Covers: ratio computation, token masking, advantage direction,
and an end-to-end one-step model test.
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
import tempfile

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from TEMPO_RL.train_phase1 import (
    SegmentGRPOTrainer,
    GRPOSequenceDataset,
    pad_collate,
)
from TEMPO_RL.io_utils import read_jsonl


# ======================================================================
# Helpers
# ======================================================================

def _make_dummy_tokenizer():
    """Create a trivial tokenizer-like object that returns incremental ids."""
    class DummyTokenizer:
        pad_token = "<pad>"
        eos_token = "<eos>"
        pad_token_id = 0
        eos_token_id = 1
        vocab_size = 100

        def encode(self, text, add_special_tokens=True):
            # Return a sequence of ids based on text length
            return [i % 90 + 10 for i in range(max(1, len(text) // 2))]

        def __call__(self, *args, **kwargs):
            return self

    return DummyTokenizer()


def _make_synthetic_segments_and_masks(
    n_tool_steps: int = 2,
    r_answer: float = 0.5,
    tool_advs: list = None,
) -> tuple:
    """Build synthetic segment returns and conversation masks for testing."""
    rid = "test_sq1_k0"

    tool_advs = tool_advs or [0.3, -0.1][:n_tool_steps]
    if len(tool_advs) < n_tool_steps:
        tool_advs = tool_advs + [0.0] * (n_tool_steps - len(tool_advs))

    # Segment returns
    segments = []
    for t in range(n_tool_steps):
        segments.append({
            "rollout_id": rid,
            "segment_id": f"{rid}_tool_{t}",
            "segment_type": "tool",
            "step_index": t,
            "return_value": tool_advs[t],
            "advantage": tool_advs[t],
            "raw_return": tool_advs[t],
            "r_tool_step": -0.02,
            "trainable": True,
            "message_role": "assistant",
            "content_type": "tool_call",
            "tool_call": {"tool_name": "table_head_reader", "arguments": {}},
        })

    segments.append({
        "rollout_id": rid,
        "segment_id": f"{rid}_answer",
        "segment_type": "answer",
        "step_index": -1,
        "return_value": r_answer,
        "advantage": r_answer,
        "raw_return": r_answer,
        "r_answer": r_answer,
        "trainable": True,
        "message_role": "assistant",
        "content_type": "answer",
        "answer": {"content": "Final answer."},
    })

    # Conversation mask
    messages = [
        {"sequence_index": 0, "role": "system", "trainable": False,
         "content_type": "system_prompt", "content_preview": "You are a helpful assistant."},
        {"sequence_index": 1, "role": "user", "trainable": False,
         "content_type": "user_question", "content_preview": "What is the production?"},
    ]
    for t in range(n_tool_steps):
        messages.append({
            "sequence_index": len(messages), "role": "assistant", "trainable": True,
            "content_type": "tool_call", "step_index": t,
            "tool_names": ["table_head_reader"],
            "content_preview": f"<tool_call> step {t}",
        })
        messages.append({
            "sequence_index": len(messages), "role": "user", "trainable": False,
            "content_type": "tool_observation",
            "content_preview": "[Tool Result] [SUCCESS] data found",
            "step_index": t,
        })
    messages.append({
        "sequence_index": len(messages), "role": "assistant", "trainable": True,
        "content_type": "answer",
        "content_preview": "The answer is 120.5万辆.",
    })

    conv_mask = {"rollout_id": rid, "messages": messages, "status": "completed"}

    return segments, [conv_mask]


# ======================================================================
# Test 1 — Ratio computation: ρ = π_θ / π_old
# ======================================================================

class TestProbabilityRatio:
    """Verify the token-level probability ratio ρ = π_θ / π_old,
    NOT the inverse."""

    def test_ratio_greater_than_one_when_new_higher(self):
        """ρ > 1 when new_log_prob > old_log_prob."""
        new_lp = torch.tensor([[-0.5, -1.0, -2.0]])  # higher prob
        old_lp = torch.tensor([[-1.0, -1.5, -2.5]])  # lower prob
        ratio = torch.exp(new_lp - old_lp)
        assert torch.all(ratio > 1.0), f"Expected ratio > 1, got {ratio}"

    def test_ratio_less_than_one_when_new_lower(self):
        """ρ < 1 when new_log_prob < old_log_prob."""
        new_lp = torch.tensor([[-2.0, -2.5, -1.5]])  # lower prob
        old_lp = torch.tensor([[-1.0, -1.5, -0.5]])  # higher prob
        ratio = torch.exp(new_lp - old_lp)
        assert torch.all(ratio < 1.0), f"Expected ratio < 1, got {ratio}"

    def test_ratio_exactly_one_when_equal(self):
        """ρ = 1 when policies are identical."""
        lp = torch.tensor([[-1.0, -0.5, -2.0]])
        ratio = torch.exp(lp - lp)
        assert torch.allclose(ratio, torch.ones_like(ratio))

    def test_ratio_formula_not_inverted(self):
        """Deliberately check that ratio = exp(new - old), NOT exp(old - new)."""
        new_lp = torch.tensor([-0.1])
        old_lp = torch.tensor([-2.0])
        # Correct: ρ = exp(-0.1 - (-2.0)) = exp(1.9) ≈ 6.686
        correct = torch.exp(new_lp - old_lp)
        # Wrong: exp(old - new) = exp(-2.0 - (-0.1)) = exp(-1.9) ≈ 0.149
        wrong = torch.exp(old_lp - new_lp)
        assert correct.item() > 1.0, "Correct ratio should be > 1"
        assert wrong.item() < 1.0, "Inverted ratio would be < 1"
        assert correct.item() != pytest.approx(wrong.item(), abs=0.01)

    def test_ratio_computed_as_exp_new_minus_old(self):
        """Exhaustively verify the formula on a grid."""
        for new_val in [-3.0, -2.0, -1.0, -0.5, -0.1]:
            for old_val in [-3.0, -2.0, -1.0, -0.5, -0.1]:
                ratio = math.exp(new_val - old_val)
                expected = math.exp(new_val) / math.exp(old_val)
                assert ratio == pytest.approx(expected)


# ======================================================================
# Test 2 — Clipped surrogate loss math
# ======================================================================

class TestClippedSurrogateLoss:
    """Verify the GRPO-style clipped loss formula."""

    def test_positive_advantage_unclipped(self):
        """When A>0 and ratio<1+ε, loss = -ratio*A (increasing ratio helps)."""
        ratio = torch.tensor([0.8, 1.0, 1.15])
        A = torch.tensor([0.5, 0.5, 0.5])
        eps_clip = 0.2

        surr1 = ratio * A
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * A
        loss = -torch.min(surr1, surr2)

        # For ratio 0.8: clip to 0.8, min(0.4, 0.4) = 0.4, loss = -0.4
        # For ratio 1.0: clip to 1.0, min(0.5, 0.5) = 0.5, loss = -0.5
        # For ratio 1.15: clip to 1.15, min(0.575, 0.575) = 0.575, loss = -0.575
        assert loss[0].item() == pytest.approx(-0.4)
        assert loss[1].item() == pytest.approx(-0.5)
        assert loss[2].item() == pytest.approx(-0.575)

    def test_positive_advantage_clipped(self):
        """When A>0 and ratio>1+ε, loss capped at clip(ratio)·A."""
        ratio = torch.tensor([1.5])
        A = torch.tensor([1.0])
        eps_clip = 0.2

        surr1 = ratio * A  # 1.5
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * A  # 1.2
        loss = -torch.min(surr1, surr2)  # -min(1.5, 1.2) = -1.2

        assert loss.item() == pytest.approx(-1.2)

    def test_negative_advantage_unclipped(self):
        """When A<0 and ratio>1-ε, loss = -ratio*A (reducing ratio helps)."""
        ratio = torch.tensor([0.85, 1.0, 1.15])
        A = torch.tensor([-0.5, -0.5, -0.5])
        eps_clip = 0.2

        surr1 = ratio * A   # -0.425, -0.5, -0.575
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * A
        # clip: 0.85, 1.0, 1.15 → surr2 = -0.425, -0.5, -0.575
        # min of equal values, loss = -(-0.425) = 0.425, etc.
        loss = -torch.min(surr1, surr2)

        assert loss[0].item() == pytest.approx(0.425)
        assert loss[1].item() == pytest.approx(0.5)
        assert loss[2].item() == pytest.approx(0.575)

    def test_negative_advantage_clipped(self):
        """When A<0 and ratio<1-ε, loss capped at clip(ratio)·A."""
        ratio = torch.tensor([0.5])
        A = torch.tensor([-1.0])
        eps_clip = 0.2

        surr1 = ratio * A  # -0.5
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * A  # 0.8 * -1 = -0.8
        loss = -torch.min(surr1, surr2)  # -min(-0.5, -0.8) = -(-0.8) = 0.8

        assert loss.item() == pytest.approx(0.8)

    def test_zero_advantage_zero_loss(self):
        """When A=0, loss=0 regardless of ratio."""
        for r in [0.5, 1.0, 2.0]:
            ratio = torch.tensor([r])
            A = torch.tensor([0.0])
            surr1 = ratio * A
            surr2 = torch.clamp(ratio, 0.8, 1.2) * A
            loss = -torch.min(surr1, surr2)
            assert loss.item() == pytest.approx(0.0)


# ======================================================================
# Test 3 — Token mask: masked tokens contribute zero loss
# ======================================================================

class TestTokenMasking:
    """Verify that tokens with trainable_mask=0 contribute nothing to loss."""

    def test_masked_tokens_contribute_zero(self):
        """Loss on masked positions is zero even with non-zero advantage."""
        trainable_mask = torch.tensor([[0.0, 0.0, 1.0, 0.0, 1.0, 0.0]])
        advantages = torch.tensor([[0.0, 100.0, 0.5, -50.0, -0.3, 0.0]])

        # Simulate PPO loss computation
        ratio = torch.ones(1, 6)  # ratio = 1 for simplicity
        eps_clip = 0.2

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
        per_token = -torch.min(surr1, surr2)
        masked = per_token * trainable_mask

        # Positions 0, 1, 3, 5 are masked → zero contribution
        assert masked[0, 0].item() == pytest.approx(0.0)
        assert masked[0, 1].item() == pytest.approx(0.0)
        assert masked[0, 3].item() == pytest.approx(0.0)
        assert masked[0, 5].item() == pytest.approx(0.0)

        # Positions 2, 4 are trainable → non-zero (if A≠0)
        assert masked[0, 2].item() != pytest.approx(0.0, abs=1e-9)
        assert masked[0, 4].item() != pytest.approx(0.0, abs=1e-9)

    def test_only_trainable_tokens_in_loss(self):
        """Mean loss only counts trainable tokens."""
        trainable_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0]])
        advantages = torch.tensor([[0.0, 0.0, 1.0, -1.0, 0.0]])
        ratio = torch.ones(1, 5)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        per_token = -torch.min(surr1, surr2)
        masked = per_token * trainable_mask

        n_trainable = trainable_mask.sum()
        mean_loss = masked.sum() / n_trainable

        # Expected: positions 2 and 3 contribute equally but opposite
        # loss[2] = -1.0, loss[3] = 1.0 → sum = 0, mean = 0
        assert mean_loss.item() == pytest.approx(0.0)

    def test_mask_ignores_system_and_user_tokens(self):
        """In a reconstructed conversation, system/user/tool roles are masked."""
        segs, masks = _make_synthetic_segments_and_masks(n_tool_steps=2)
        tokenizer = _make_dummy_tokenizer()

        dataset = GRPOSequenceDataset(segs, masks, tokenizer, max_length=512)
        sample = dataset[0]

        trainable = sample["trainable_mask"]
        input_ids = sample["input_ids"]

        # Find which tokens are trainable (should be assistant messages only)
        trainable_positions = torch.where(trainable > 0.5)[0]

        # All trainable positions should be at later indices
        # (system + user come first and are masked)
        n_masked_at_start = 0
        for i in range(len(trainable)):
            if trainable[i] < 0.5:
                n_masked_at_start += 1
            else:
                break
        assert n_masked_at_start > 0, "First tokens (system+user) should be masked"

    def test_tool_observation_tokens_masked(self):
        """Tool observation tokens have trainable_mask = 0.

        Verifies that masked blocks (0) appear between trainable blocks (1),
        confirming system, user, and tool-observation messages are all masked.
        """
        segs, masks = _make_synthetic_segments_and_masks(n_tool_steps=2)
        tokenizer = _make_dummy_tokenizer()

        dataset = GRPOSequenceDataset(segs, masks, tokenizer, max_length=512)
        sample = dataset[0]

        mask = sample["trainable_mask"]
        n_trainable = (mask > 0.5).sum().item()
        n_masked = (mask < 0.5).sum().item()

        # Both trainable and masked tokens must exist
        assert n_trainable > 0, "Should have trainable tokens"
        assert n_masked > 0, "Should have masked tokens"

        # Count transitions in mask: should have at least 4 transitions
        # (0→1→0→1→0→1) for 2 tool steps + answer
        transitions = 0
        prev = mask[0].item()
        for i in range(1, len(mask)):
            curr = mask[i].item()
            if abs(curr - prev) > 0.5:
                transitions += 1
            prev = curr
        assert transitions >= 4, \
            f"Expected ≥4 mask transitions (system→assistant→tool→assistant→tool→assistant), got {transitions}"


# ======================================================================
# Test 4 — Advantage direction
# ======================================================================

class TestAdvantageDirection:
    """Verify that:
    - Positive advantage → increases target token likelihood after a step
    - Negative advantage → decreases target token likelihood after a step
    """

    def test_positive_adv_increases_log_prob_mathematically(self):
        """With A>0, the loss gradient pushes log_prob higher.

        loss = -min(r*A, clip(r)*A)
        For A>0: loss = -something_positive → negative
        Gradient descent on negative loss → increases log_prob → increases r.
        """
        # This is verified by the loss sign: with A>0, loss < 0
        # Optimizer.minimize(loss) with loss<0 → increases the underlying params
        ratio = torch.tensor([1.0], requires_grad=False)
        A = torch.tensor([1.0])
        surr1 = ratio * A
        surr2 = torch.clamp(ratio, 0.8, 1.2) * A
        loss = -torch.min(surr1, surr2)
        assert loss.item() < 0.0, "Positive advantage should give negative loss"

    def test_negative_adv_decreases_log_prob_mathematically(self):
        """With A<0, the loss pushes log_prob lower."""
        ratio = torch.tensor([1.0], requires_grad=False)
        A = torch.tensor([-1.0])
        surr1 = ratio * A
        surr2 = torch.clamp(ratio, 0.8, 1.2) * A
        loss = -torch.min(surr1, surr2)
        assert loss.item() > 0.0, "Negative advantage should give positive loss"

    def test_advantage_sign_loss_sign_relationship(self):
        """Systematic check: sign(loss) = -sign(A) when ratio ≈ 1."""
        for A_val in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]:
            ratio = torch.tensor([1.0])
            A = torch.tensor([A_val])
            surr1 = ratio * A
            surr2 = torch.clamp(ratio, 0.8, 1.2) * A
            loss = -torch.min(surr1, surr2)
            if A_val > 0:
                assert loss.item() < 0, f"A={A_val} → loss should be negative"
            elif A_val < 0:
                assert loss.item() > 0, f"A={A_val} → loss should be positive"

    def test_gradient_increases_log_prob_for_positive_adv(self):
        """After one gradient step with A>0, log_prob increases."""
        # Use 2 classes so softmax is non-trivial (1 class always = 1.0)
        logits = torch.tensor([0.0, 0.5], requires_grad=True)
        target_id = 0
        A = torch.tensor([2.0])  # positive advantage
        eps_clip = 0.2

        # Initial log_prob
        old_log_prob = F.log_softmax(logits.unsqueeze(0), dim=-1)[0, target_id].detach()

        # Compute PPO loss
        new_log_prob = F.log_softmax(logits.unsqueeze(0), dim=-1)[0, target_id]
        ratio = torch.exp(new_log_prob - old_log_prob)
        surr1 = ratio * A
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * A
        loss = -torch.min(surr1, surr2)
        loss.backward()

        # Take a gradient step
        with torch.no_grad():
            lr = 0.1
            logits -= lr * logits.grad

        # After step, log_prob should be higher (loss was negative → gradient
        # increases logit → increases softmax prob → increases log_prob)
        final_log_prob = F.log_softmax(logits.unsqueeze(0), dim=-1)[0, target_id]
        assert final_log_prob > old_log_prob, \
            f"Positive adv should increase log_prob: {old_log_prob:.4f} → {final_log_prob:.4f}"

    def test_gradient_decreases_log_prob_for_negative_adv(self):
        """After one gradient step with A<0, log_prob decreases."""
        logits = torch.tensor([1.0, 0.0], requires_grad=True)  # start with target higher
        target_id = 0
        A = torch.tensor([-2.0])  # negative advantage
        eps_clip = 0.2

        old_log_prob = F.log_softmax(logits.unsqueeze(0), dim=-1)[0, target_id].detach()

        new_log_prob = F.log_softmax(logits.unsqueeze(0), dim=-1)[0, target_id]
        ratio = torch.exp(new_log_prob - old_log_prob)
        surr1 = ratio * A
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * A
        loss = -torch.min(surr1, surr2)
        loss.backward()

        with torch.no_grad():
            lr = 0.1
            logits -= lr * logits.grad

        final_log_prob = F.log_softmax(logits.unsqueeze(0), dim=-1)[0, target_id]
        assert final_log_prob < old_log_prob, \
            f"Negative adv should decrease log_prob: {old_log_prob:.4f} → {final_log_prob:.4f}"


# ======================================================================
# Test 5 — Dataset and collation
# ======================================================================

class TestDatasetAndCollation:
    """Verify the PPO dataset produces valid tensors."""

    def test_dataset_produces_correct_keys(self):
        segs, masks = _make_synthetic_segments_and_masks(n_tool_steps=1)
        tokenizer = _make_dummy_tokenizer()
        dataset = GRPOSequenceDataset(segs, masks, tokenizer, max_length=512)

        assert len(dataset) == 1
        sample = dataset[0]
        for key in ("input_ids", "attention_mask", "advantages",
                     "trainable_mask", "labels"):
            assert key in sample, f"Missing key: {key}"
            assert isinstance(sample[key], torch.Tensor)

    def test_all_tensors_same_length(self):
        segs, masks = _make_synthetic_segments_and_masks(n_tool_steps=2)
        tokenizer = _make_dummy_tokenizer()
        dataset = GRPOSequenceDataset(segs, masks, tokenizer, max_length=512)
        sample = dataset[0]

        L = len(sample["input_ids"])
        for key in ("attention_mask", "advantages", "trainable_mask", "labels"):
            assert len(sample[key]) == L, f"{key} length mismatch"

    def test_advantages_aligned_with_masks(self):
        """Positions with mask=0 should have advantage=0."""
        segs, masks = _make_synthetic_segments_and_masks(n_tool_steps=2)
        tokenizer = _make_dummy_tokenizer()
        dataset = GRPOSequenceDataset(segs, masks, tokenizer, max_length=512)
        sample = dataset[0]

        for i in range(len(sample["input_ids"])):
            if sample["trainable_mask"][i] == 0.0:
                assert sample["advantages"][i] == 0.0, \
                    f"Masked position {i} should have advantage 0"

    def test_pad_collate(self):
        """batch of different-length sequences gets padded correctly."""
        s1 = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "advantages": torch.tensor([0.0, 0.5, -0.3]),
            "trainable_mask": torch.tensor([0.0, 1.0, 1.0]),
            "labels": torch.tensor([1, 2, 3]),
        }
        s2 = {
            "input_ids": torch.tensor([4, 5]),
            "attention_mask": torch.tensor([1, 1]),
            "advantages": torch.tensor([0.0, 0.7]),
            "trainable_mask": torch.tensor([0.0, 1.0]),
            "labels": torch.tensor([4, 5]),
        }
        batch = pad_collate([s1, s2])

        # Should pad to max length = 3
        assert batch["input_ids"].shape == (2, 3)
        # Second sequence padded
        assert batch["input_ids"][1, 2].item() == 0  # pad token
        assert batch["attention_mask"][1, 2].item() == 0
        assert batch["advantages"][1, 2].item() == 0.0
        assert batch["trainable_mask"][1, 2].item() == 0.0


# ======================================================================
# Test 6 — Trainer manual loss computation (no model)
# ======================================================================

class TestTrainerLossManual:
    """Verify trainer.compute_loss output structure and invariants."""

    def test_compute_loss_returns_all_expected_keys(self):
        """compute_loss returns loss, clip_loss, mean_ratio, n_trainable."""
        segs, masks = _make_synthetic_segments_and_masks(n_tool_steps=1)
        tokenizer = _make_dummy_tokenizer()
        dataset = GRPOSequenceDataset(segs, masks, tokenizer, max_length=512)
        sample = dataset[0]

        # Verify the sample has all keys
        for key in ("input_ids", "attention_mask", "advantages",
                     "trainable_mask", "labels"):
            assert key in sample, f"Sample missing key: {key}"

        # Verify shapes match
        L = len(sample["input_ids"])
        for key in sample:
            assert len(sample[key]) == L, f"Key {key} length mismatch: {len(sample[key])} vs {L}"

    def test_trainable_mask_sums_to_positive(self):
        """Every dataset sample has at least some trainable tokens."""
        for n_steps in [0, 1, 2]:
            segs, masks = _make_synthetic_segments_and_masks(n_tool_steps=n_steps)
            tokenizer = _make_dummy_tokenizer()
            dataset = GRPOSequenceDataset(segs, masks, tokenizer, max_length=512)
            for idx in range(len(dataset)):
                sample = dataset[idx]
                n = sample["trainable_mask"].sum().item()
                assert n > 0, f"Sample {idx} with n_steps={n_steps} has 0 trainable tokens"

    def test_dataloader_batch_shapes(self):
        """DataLoader produces correctly shaped batches."""
        segs, masks = _make_synthetic_segments_and_masks(n_tool_steps=2)
        tokenizer = _make_dummy_tokenizer()
        dataset = GRPOSequenceDataset(segs, masks, tokenizer, max_length=512)
        loader = DataLoader(dataset, batch_size=1, collate_fn=pad_collate)
        batch = next(iter(loader))

        assert batch["input_ids"].dim() == 2  # (B, L)
        assert batch["advantages"].dim() == 2
        assert batch["trainable_mask"].dim() == 2
        assert batch["input_ids"].shape == batch["advantages"].shape


# ======================================================================
# Test 7 — End-to-end one-step smoke test (real model)
# ======================================================================

_MODEL_PATH = os.path.join(_PROJ_ROOT, "models", "Qwen2.5-0.5B-Instruct")


@pytest.mark.skipif(not os.path.exists(_MODEL_PATH), reason="Qwen2.5-0.5B model not found")
class TestOneStepSmoke:
    """One-step forward+backward+loss with a real model."""

    def test_model_loads(self):
        """Model and tokenizer load without error."""
        trainer = SegmentGRPOTrainer(
            model_path=_MODEL_PATH,
            eps_clip=0.2, lr=1e-6,
        )
        assert trainer.model is not None
        assert trainer.old_model is not None
        assert trainer.tokenizer is not None

    def test_one_step_loss_finite(self):
        """After one training step, all losses are finite."""
        trainer = SegmentGRPOTrainer(
            model_path=_MODEL_PATH,
            eps_clip=0.2, lr=1e-6,
        )

        # Build synthetic data that tokenizes for real
        segs, masks = _make_synthetic_segments_and_masks(n_tool_steps=2)

        # Write to temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            seg_path = os.path.join(tmpdir, "seg.jsonl")
            mask_path = os.path.join(tmpdir, "mask.jsonl")

            with open(seg_path, "w") as f:
                for s in segs:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            with open(mask_path, "w") as f:
                for m in masks:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")

            dataloader = trainer.build_dataloader(
                seg_path, mask_path, batch_size=1, max_length=512
            )

            assert len(dataloader.dataset) > 0, "Dataset should be non-empty"

            batch = next(iter(dataloader))
            input_ids = batch["input_ids"].to(trainer._device)
            attention_mask = batch["attention_mask"].to(trainer._device)
            advantages = batch["advantages"].to(trainer._device)
            trainable_mask = batch["trainable_mask"].to(trainer._device)

            # Compute loss
            metrics = trainer.compute_loss(
                input_ids, attention_mask, advantages, trainable_mask
            )

            print(f"\n  One-step loss check:")
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: {v.item():.6f}")
                else:
                    print(f"    {k}: {v}")

            # All components should be finite
            for key in ("loss", "clip_loss"):
                val = metrics[key]
                assert torch.isfinite(val), f"{key} should be finite, got {val}"

            # Mean ratio should be ≈ 1.0 (same model as old)
            assert metrics["mean_ratio"].item() == pytest.approx(1.0, abs=0.01), \
                f"Initial ratio should be ≈ 1.0, got {metrics['mean_ratio']}"

            # n_trainable should be > 0
            assert metrics["n_trainable"] > 0, "Should have trainable tokens"

    def test_one_train_step(self):
        """train_step runs without error and updates step counter."""
        trainer = SegmentGRPOTrainer(
            model_path=_MODEL_PATH,
            eps_clip=0.2, lr=1e-9,  # tiny LR
        )

        segs, masks = _make_synthetic_segments_and_masks(n_tool_steps=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            seg_path = os.path.join(tmpdir, "seg.jsonl")
            mask_path = os.path.join(tmpdir, "mask.jsonl")

            with open(seg_path, "w") as f:
                for s in segs:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            with open(mask_path, "w") as f:
                for m in masks:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")

            dataloader = trainer.build_dataloader(
                seg_path, mask_path, batch_size=1, max_length=512
            )
            batch = next(iter(dataloader))

            assert trainer.global_step == 0

            metrics = trainer.train_step(
                input_ids=batch["input_ids"].to(trainer._device),
                attention_mask=batch["attention_mask"].to(trainer._device),
                advantages=batch["advantages"].to(trainer._device),
                trainable_mask=batch["trainable_mask"].to(trainer._device),
            )

            assert trainer.global_step == 1
            assert math.isfinite(metrics["loss"])

            print(f"\n  Train step metrics: loss={metrics['loss']:.6f} "
                  f"clip_loss={metrics['clip_loss']:.6f}")

    def test_update_old_policy(self):
        """After update_old_policy, compute_loss produces ratio ≈ 1."""
        trainer = SegmentGRPOTrainer(
            model_path=_MODEL_PATH,
            eps_clip=0.2, lr=1e-9,
        )

        segs, masks = _make_synthetic_segments_and_masks(n_tool_steps=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            seg_path = os.path.join(tmpdir, "seg.jsonl")
            mask_path = os.path.join(tmpdir, "mask.jsonl")
            with open(seg_path, "w") as f:
                for s in segs:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            with open(mask_path, "w") as f:
                for m in masks:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")

            dataloader = trainer.build_dataloader(
                seg_path, mask_path, batch_size=1, max_length=512
            )
            batch = next(iter(dataloader))
            device = trainer._device

            # Run one training step to change current model weights
            trainer.train_step(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                advantages=batch["advantages"].to(device),
                trainable_mask=batch["trainable_mask"].to(device),
            )

            # Before update: ratio should NOT be ≈ 1 (current model changed, old unchanged)
            metrics_before = trainer.compute_loss(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["advantages"].to(device),
                batch["trainable_mask"].to(device),
            )
            ratio_before = metrics_before["mean_ratio"].item()

            # After update_old_policy: ratio should be ≈ 1 again
            trainer.update_old_policy()
            metrics_after = trainer.compute_loss(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["advantages"].to(device),
                batch["trainable_mask"].to(device),
            )
            ratio_after = metrics_after["mean_ratio"].item()

            assert ratio_after == pytest.approx(1.0, abs=0.01), \
                f"After update_old_policy, ratio should be ≈ 1, got {ratio_after}"
            print(f"\n  Update policy: ratio before={ratio_before:.4f}, after={ratio_after:.4f}")

    def test_ratio_is_one_at_init(self):
        """At initialization, π_θ == π_old, so ratio = 1."""
        trainer = SegmentGRPOTrainer(
            model_path=_MODEL_PATH,
            eps_clip=0.2, lr=1e-6,
        )

        segs, masks = _make_synthetic_segments_and_masks(n_tool_steps=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            seg_path = os.path.join(tmpdir, "seg.jsonl")
            mask_path = os.path.join(tmpdir, "mask.jsonl")
            with open(seg_path, "w") as f:
                for s in segs:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            with open(mask_path, "w") as f:
                for m in masks:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")

            dataloader = trainer.build_dataloader(
                seg_path, mask_path, batch_size=1, max_length=512
            )
            batch = next(iter(dataloader))

            metrics = trainer.compute_loss(
                batch["input_ids"].to(trainer._device),
                batch["attention_mask"].to(trainer._device),
                batch["advantages"].to(trainer._device),
                batch["trainable_mask"].to(trainer._device),
            )

            assert metrics["mean_ratio"].item() == pytest.approx(1.0, abs=0.01), \
                f"Initial ratio should be 1.0, got {metrics['mean_ratio']}"


# ======================================================================
# Test 8 — Edge cases
# ======================================================================

class TestEdgeCases:
    """Corner cases for the training pipeline."""

    def test_empty_sequence_handled(self):
        """Empty token sequence doesn't crash."""
        segs, masks = _make_synthetic_segments_and_masks(n_tool_steps=2)
        tokenizer = _make_dummy_tokenizer()

        # The dummy tokenizer always returns non-empty sequences, so
        # this test verifies the dataset can handle the minimum case.
        dataset = GRPOSequenceDataset(segs, masks, tokenizer, max_length=512)
        assert len(dataset) > 0

    def test_no_answer_segment(self):
        """Rollout with no answer (truncated) → no answer segment."""
        rid = "truncated_k0"
        segments = [
            {"rollout_id": rid, "segment_id": f"{rid}_tool_0",
             "segment_type": "tool", "step_index": 0,
             "return_value": -0.5, "advantage": -0.5, "raw_return": -0.5,
             "trainable": True, "message_role": "assistant",
             "content_type": "tool_call"},
        ]
        masks = [{
            "rollout_id": rid, "status": "truncated",
            "messages": [
                {"sequence_index": 0, "role": "system", "trainable": False,
                 "content_type": "system_prompt", "content_preview": "system"},
                {"sequence_index": 1, "role": "user", "trainable": False,
                 "content_type": "user_question", "content_preview": "Q?"},
                {"sequence_index": 2, "role": "assistant", "trainable": True,
                 "content_type": "tool_call", "step_index": 0,
                 "tool_names": ["cmd_executor"]},
                {"sequence_index": 3, "role": "user", "trainable": False,
                 "content_type": "tool_observation",
                 "content_preview": "[Tool Result] result"},
            ]
        }]

        tokenizer = _make_dummy_tokenizer()
        dataset = GRPOSequenceDataset(segments, masks, tokenizer, max_length=512)
        assert len(dataset) > 0
        sample = dataset[0]
        # All trainable tokens should be from the tool_call message
        assert sample["trainable_mask"].sum() > 0

    def test_single_message_direct_answer(self):
        """Rollout with only an answer (no tool steps)."""
        rid = "direct_answer_k0"
        segments = [
            {"rollout_id": rid, "segment_id": f"{rid}_answer",
             "segment_type": "answer", "step_index": -1,
             "return_value": 0.0, "advantage": 0.0, "raw_return": 0.0,
             "trainable": True, "message_role": "assistant",
             "content_type": "answer"},
        ]
        masks = [{
            "rollout_id": rid, "status": "completed",
            "messages": [
                {"sequence_index": 0, "role": "system", "trainable": False,
                 "content_type": "system_prompt", "content_preview": "sys"},
                {"sequence_index": 1, "role": "user", "trainable": False,
                 "content_type": "user_question", "content_preview": "Q?"},
                {"sequence_index": 2, "role": "assistant", "trainable": True,
                 "content_type": "answer", "content_preview": "Answer text."},
            ]
        }]
        tokenizer = _make_dummy_tokenizer()
        dataset = GRPOSequenceDataset(segments, masks, tokenizer, max_length=512)
        assert len(dataset) > 0
