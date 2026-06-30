"""
TEMPO-RL — Segment-aware GRPO-style Clipped Policy Optimization.

Supports both Phase 1 (subquestion-level) and Phase 2 (dialog-level) training
with three segment types: **tool**, **answer**, and **memory**.

Implements a group-relative clipped objective without a separate value model:

- Group-relative advantages: A = (R − μ_group) / (σ_group + ε)
  computed per segment type within the same prompt's K rollouts
- Token-level probability ratio ρ = π_θ / π_old
- Clipped surrogate objective: L = −min(ρ·A, clip(ρ)·A)
- Token masks (system / user / tool-observation excluded)
- No separate critic / value model
- No SFT replay loss (by design — first version excludes it per §11)

Usage::

    # Phase 1 (subquestion-level)
    python -m TEMPO_RL.train_phase1 \\
        --model_path models/Qwen2.5-0.5B-Instruct \\
        --segment_returns phase1_output/segment_returns.jsonl \\
        --conversation_masks phase1_output/segment_returns_conversation_masks.jsonl \\
        --output_dir train_output \\
        --batch_size 1 --lr 1e-6 --max_steps 1

    # Phase 2 (dialog-level, with memory segments)
    python -m TEMPO_RL.train_phase1 \\
        --model_path models/Qwen2.5-0.5B-Instruct \\
        --segment_returns phase2_output/phase2_segment_returns.jsonl \\
        --conversation_masks phase2_output/phase2_segment_returns_conversation_masks.jsonl \\
        --output_dir train_output \\
        --batch_size 1 --lr 1e-6 --max_steps 1
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)


# ======================================================================
# Dataset
# ======================================================================

class GRPOSequenceDataset(Dataset):
    """Dataset that produces tokenized conversation sequences with
    per-token advantage and trainable-mask labels.

    Each item is a single rollout's full conversation, tokenized so that
    every token position carries:

    - ``input_ids``: token ids for the full sequence
    - ``attention_mask``: 1 for all real tokens
    - ``advantages``: per-token advantage (0 where masked)
    - ``trainable_mask``: 1 for assistant tokens, 0 elsewhere
    - ``labels``: input_ids copy (unused by GRPO, kept for compatibility)
    """

    def __init__(
        self,
        segments: List[Dict[str, Any]],
        conversation_masks: List[Dict[str, Any]],
        tokenizer: Any,
        max_length: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._samples: List[Dict[str, Any]] = []

        # Index segments by rollout_id
        seg_by_rollout: Dict[str, Dict[str, Any]] = {}
        for seg in segments:
            rid = seg["rollout_id"]
            seg_by_rollout.setdefault(rid, {"tool": [], "answer": None, "memory": None})
            st = seg["segment_type"]
            if st == "tool":
                seg_by_rollout[rid]["tool"].append(seg)
            elif st == "answer":
                seg_by_rollout[rid]["answer"] = seg
            elif st == "memory":
                seg_by_rollout[rid]["memory"] = seg

        # Index conversation masks by rollout_id
        mask_by_rollout: Dict[str, Dict[str, Any]] = {}
        for cm in conversation_masks:
            mask_by_rollout[cm["rollout_id"]] = cm

        # Build samples
        for rid, seg_group in seg_by_rollout.items():
            cm = mask_by_rollout.get(rid)
            if cm is None:
                continue

            sample = self._build_sample(rid, seg_group, cm)
            if sample is not None:
                self._samples.append(sample)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._samples[idx]

    def _build_sample(
        self,
        rollout_id: str,
        seg_group: Dict[str, Any],
        conv_mask: Dict[str, Any],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Build a tokenized sample from one rollout.

        Strategy:
          1. Tokenize each message individually to know its token span.
          2. Assign advantage per span based on segment type.
          3. Assign trainable_mask per span based on role.
          4. Concatenate all token sequences.
        """
        messages = conv_mask.get("messages", [])
        tool_segs = seg_group["tool"]
        ans_seg = seg_group["answer"]
        mem_seg = seg_group.get("memory")

        # Build tool advantage lookup: step_index → return_value
        tool_adv: Dict[int, float] = {}
        for ts in tool_segs:
            si = ts.get("step_index", -1)
            tool_adv[si] = ts.get("return_value", 0.0)

        # Build answer advantage
        answer_adv = ans_seg.get("return_value", 0.0) if ans_seg else 0.0

        # Build memory advantage (for Phase 2 dialog-level rollouts)
        memory_adv = mem_seg.get("return_value", 0.0) if mem_seg else 0.0

        # We need to track which tool_step each assistant message belongs to
        # in the conversation.  Walk through messages and match them.
        tool_step_counter = 0  # counts tool_call assistant messages seen

        all_input_ids: List[int] = []
        all_advantages: List[float] = []
        all_trainable: List[float] = []

        for msg in messages:
            role = msg.get("role", "")
            content_type = msg.get("content_type", "")
            trainable = msg.get("trainable", False)

            # Build text representation
            text = self._message_to_text(msg)

            # Tokenize this message
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            token_ids = encoded  # list of ints
            n_tokens = len(token_ids)

            if n_tokens == 0:
                continue

            # Determine advantage for this span
            if content_type == "tool_call":
                step_idx = msg.get("step_index")
                adv = tool_adv.get(step_idx, 0.0)
                if step_idx is None:
                    # Fall back to counter
                    adv = tool_adv.get(tool_step_counter, 0.0)
                    tool_step_counter += 1
            elif content_type == "answer":
                adv = answer_adv
            elif content_type == "memory":
                adv = memory_adv
            else:
                adv = 0.0  # masked tokens get 0 advantage

            mask_val = 1.0 if trainable else 0.0

            all_input_ids.extend(token_ids)
            all_advantages.extend([adv] * n_tokens)
            all_trainable.extend([mask_val] * n_tokens)

        # Truncate to max_length
        if len(all_input_ids) > self.max_length:
            all_input_ids = all_input_ids[:self.max_length]
            all_advantages = all_advantages[:self.max_length]
            all_trainable = all_trainable[:self.max_length]

        if len(all_input_ids) == 0:
            return None

        # Build attention mask
        attention_mask = [1] * len(all_input_ids)

        # Labels for SFT replay: same as input_ids (causal LM shifts internally)
        labels = list(all_input_ids)

        return {
            "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "advantages": torch.tensor(all_advantages, dtype=torch.float32),
            "trainable_mask": torch.tensor(all_trainable, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    @staticmethod
    def _message_to_text(msg: Dict[str, Any]) -> str:
        """Convert a conversation mask entry to tokenizable text.

        For trainable assistant messages (tool_call, answer), uses the stored
        ``response_text`` (the actual model output).  For masked roles, uses
        ``content_preview`` as a fallback.
        """
        role = msg.get("role", "")
        content_type = msg.get("content_type", "")

        if role == "system":
            return msg.get("content") or msg.get("content_preview", "") or "You are a helpful assistant."
        elif role == "user":
            if content_type == "error_feedback":
                return "[ERROR] Multiple tool calls detected. Only ONE tool call per turn is allowed."
            return msg.get("content") or msg.get("content_preview", "")
        elif role == "assistant":
            # Use the actual model response text for trainable messages
            if content_type in ("tool_call", "answer", "memory"):
                return msg.get("response_text", "") or msg.get("content", "") or msg.get("content_preview", "")
            return msg.get("content") or msg.get("content_preview", "")
        elif role == "tool":
            return msg.get("content") or msg.get("content_preview", "") or "[Tool result]"
        return msg.get("content") or msg.get("content_preview", "")


# ======================================================================
# Collation
# ======================================================================

def pad_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad a batch of variable-length sequences to the same length."""
    if not batch:
        return {}

    keys = batch[0].keys()
    padded: Dict[str, torch.Tensor] = {}

    for key in keys:
        tensors = [item[key] for item in batch]
        if key == "attention_mask" or key == "trainable_mask" or key == "labels":
            # Pad with 0
            padded[key] = torch.nn.utils.rnn.pad_sequence(
                tensors, batch_first=True, padding_value=0
            )
        elif key == "advantages":
            # Pad with 0 (masked positions)
            padded[key] = torch.nn.utils.rnn.pad_sequence(
                tensors, batch_first=True, padding_value=0.0
            )
        else:
            # input_ids: pad with pad_token_id (or 0)
            padded[key] = torch.nn.utils.rnn.pad_sequence(
                tensors, batch_first=True, padding_value=0
            )

    return padded


# ======================================================================
# GRPO-style Trainer
# ======================================================================

class SegmentGRPOTrainer:
    """Segment-aware GRPO-style clipped policy optimization.

    Uses group-relative advantages (no value model / critic) and a clipped
    ratio objective.  Advantages are pre-computed by the segment-return
    builder as ``A = (R − μ_group) / (σ_group + ε)`` per segment type
    within the K rollouts of the same prompt.

    Parameters
    ----------
    model_path : str
        Path to HuggingFace model / SFT checkpoint.
    eps_clip : float = 0.2
        Clipping epsilon for the ratio objective.
    lr : float = 1e-6
        Learning rate.
    device : str = "auto"
        Device string or "auto".
    """

    def __init__(
        self,
        model_path: str,
        eps_clip: float = 0.2,
        lr: float = 1e-6,
        device: str = "auto",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.eps_clip = eps_clip

        if device == "auto":
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self._device = torch.device(device)

        print(f"[GRPO] Loading model from {model_path} ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map=None,
        )
        self.model.to(self._device)
        self.model.gradient_checkpointing_enable()
        self.model.train()

        # Frozen copy as π_old for ratio ρ = π_θ / π_old
        print(f"[GRPO] Creating π_old copy ...")
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self._global_step = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_dataloader(
        self,
        segment_path: str,
        mask_path: str,
        batch_size: int = 1,
        max_length: int = 4096,
    ) -> DataLoader:
        """Build a DataLoader from segment returns and conversation masks."""
        segments = _load_jsonl(segment_path)
        masks = _load_jsonl(mask_path)

        actual_masks: List[Dict[str, Any]] = []
        for m in masks:
            if "conversation_masks" in m:
                actual_masks.extend(m["conversation_masks"])
            elif "rollout_id" in m:
                actual_masks.append(m)

        dataset = GRPOSequenceDataset(
            segments=segments,
            conversation_masks=actual_masks,
            tokenizer=self.tokenizer,
            max_length=max_length,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=pad_collate,
        )

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        advantages: torch.Tensor,
        trainable_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute the GRPO-style clipped objective.

        L = −mean[ min(ρ·A, clip(ρ)·A) ] over trainable tokens

        where ρ = π_θ(a_t|s_t) / π_old(a_t|s_t).

        Returns dict with ``loss``, ``clip_loss``, ``mean_ratio``, ``n_trainable``.
        """
        B, L = input_ids.shape

        # --- log π_θ(a_t | s_t) ---
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        new_log_probs = self._gather_token_log_probs(logits, input_ids)  # (B, L)

        # --- log π_old(a_t | s_t) ---
        with torch.no_grad():
            old_outputs = self.old_model(input_ids=input_ids, attention_mask=attention_mask)
            old_log_probs = self._gather_token_log_probs(old_outputs.logits, input_ids)

        # --- Ratio ρ = π_θ / π_old ---
        ratio = torch.exp(new_log_probs - old_log_probs)  # (B, L)

        # --- Clipped surrogate ---
        clip_low = 1.0 - self.eps_clip
        clip_high = 1.0 + self.eps_clip
        ratio_clipped = torch.clamp(ratio, clip_low, clip_high)

        surr1 = ratio * advantages
        surr2 = ratio_clipped * advantages
        per_token_loss = -torch.min(surr1, surr2)  # (B, L)

        n_trainable = trainable_mask.sum() + self._eps()
        clip_loss = (per_token_loss * trainable_mask).sum() / n_trainable

        # --- Diagnostics ---
        with torch.no_grad():
            mean_ratio = (ratio * trainable_mask).sum() / n_trainable

        return {
            "loss": clip_loss,
            "clip_loss": clip_loss,
            "mean_ratio": mean_ratio.detach(),
            "n_trainable": int(n_trainable.item()),
        }

    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        advantages: torch.Tensor,
        trainable_mask: torch.Tensor,
    ) -> Dict[str, float]:
        """Run one training step: forward, backward, optimizer step."""
        self.model.train()
        self.optimizer.zero_grad()

        loss_dict = self.compute_loss(
            input_ids=input_ids,
            attention_mask=attention_mask,
            advantages=advantages,
            trainable_mask=trainable_mask,
        )

        if torch.isnan(loss_dict["loss"]) or torch.isinf(loss_dict["loss"]):
            print(f"  [WARN] step {self._global_step}: NaN/Inf loss detected, skipping update")
            self._global_step += 1
            return {k: float("nan") for k in loss_dict}

        loss_dict["loss"].backward()
        # Gradient clipping to prevent overflow → NaN
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self._global_step += 1

        return {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in loss_dict.items()
        }

    def update_old_policy(self) -> None:
        """Copy current model weights to π_old (call after each rollout epoch)."""
        self.old_model.load_state_dict(copy.deepcopy(self.model.state_dict()))

    @property
    def global_step(self) -> int:
        return self._global_step

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gather_token_log_probs(
        self, logits: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute log π(a_t|s_t) for each token position."""
        log_probs = F.log_softmax(logits, dim=-1)  # (B, L, V)
        B, L, V = log_probs.shape

        shifted_ids = input_ids[:, 1:].unsqueeze(-1)  # (B, L-1, 1)
        gathered = torch.gather(
            log_probs[:, :-1, :], dim=-1, index=shifted_ids
        ).squeeze(-1)  # (B, L-1)

        # Prepend zero so log_prob[j] = log P(token[j] | context[:j]),
        # aligning with trainable_mask[j] = 1 for assistant token at position j.
        pad = torch.zeros(B, 1, device=logits.device, dtype=gathered.dtype)
        return torch.cat([pad, gathered], dim=1)  # (B, L)

    @staticmethod
    def _eps() -> float:
        return 1e-8


# ======================================================================
# Helpers
# ======================================================================

from TEMPO_RL.io_utils import read_jsonl as _read_jsonl, load_json_file as _read_json

_load_jsonl = _read_jsonl  # backward-compat alias
_load_json = _read_json    # backward-compat alias


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TEMPO-RL Phase 1 — GRPO-style Training"
    )
    parser.add_argument(
        "--model_path", required=True,
        help="Path to HuggingFace model / SFT checkpoint"
    )
    parser.add_argument(
        "--segment_returns", required=True,
        help="Path to segment_returns.jsonl"
    )
    parser.add_argument(
        "--conversation_masks", required=True,
        help="Path to segment_returns_conversation_masks.jsonl"
    )
    parser.add_argument(
        "--output_dir", default="phase1_train_output",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Training batch size"
    )
    parser.add_argument(
        "--max_length", type=int, default=4096,
        help="Max sequence length"
    )
    parser.add_argument(
        "--max_steps", type=int, default=1,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-6,
        help="Learning rate"
    )
    parser.add_argument(
        "--eps_clip", type=float, default=0.2,
        help="Clipping epsilon for ratio objective"
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device: 'auto', 'cuda', 'cpu', or 'cuda:0' etc."
    )
    parser.add_argument(
        "--update_old_every", type=int, default=0,
        help="Update π_old every N steps (0 = never, update at end of training)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Init trainer ---
    trainer = SegmentGRPOTrainer(
        model_path=args.model_path,
        eps_clip=args.eps_clip,
        lr=args.lr,
        device=args.device,
    )

    # --- Build dataloader ---
    dataloader = trainer.build_dataloader(
        segment_path=args.segment_returns,
        mask_path=args.conversation_masks,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print(f"[GRPO] Dataset size: {len(dataloader.dataset)} sequences")
    print(f"[GRPO] Training for {args.max_steps} step(s) ...")

    # --- Training loop ---
    step = 0
    while step < args.max_steps:
        for batch in dataloader:
            input_ids = batch["input_ids"].to(trainer._device)
            attention_mask = batch["attention_mask"].to(trainer._device)
            advantages = batch["advantages"].to(trainer._device)
            trainable_mask = batch["trainable_mask"].to(trainer._device)

            metrics = trainer.train_step(
                input_ids=input_ids,
                attention_mask=attention_mask,
                advantages=advantages,
                trainable_mask=trainable_mask,
            )

            step = trainer.global_step
            print(
                f"  step {step}/{args.max_steps}: "
                f"loss={metrics['loss']:.4f} "
                f"clip_loss={metrics['clip_loss']:.4f} "
                f"ratio={metrics['mean_ratio']:.4f} "
                f"n_trainable={metrics['n_trainable']}"
            )
            if args.update_old_every > 0 and step % args.update_old_every == 0:
                trainer.update_old_policy()
                print(f"  [GRPO] Updated π_old at step {step}")
            if step >= args.max_steps:
                break
        # DataLoader exhausted, will cycle for next epoch

    # Final update to sync π_old
    if args.update_old_every > 0:
        trainer.update_old_policy()

    # Save checkpoint
    ckpt_path = os.path.join(args.output_dir, "phase1_step1")
    trainer.model.save_pretrained(ckpt_path)
    trainer.tokenizer.save_pretrained(ckpt_path)
    print(f"[GRPO] Saved checkpoint to {ckpt_path}")
    print("[GRPO] Done.")


if __name__ == "__main__":
    main()
