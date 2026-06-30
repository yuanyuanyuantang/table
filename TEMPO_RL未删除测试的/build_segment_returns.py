"""
TEMPO-RL Phase 1 — Segment Return Builder.

Converts ``phase1_rollouts.jsonl`` into segment-level training records suitable
for an RL trainer.  Each segment carries a computed return / advantage plus
token-mask metadata so the trainer knows which tokens are trainable.

Usage::

    python -m TEMPO_RL.build_segment_returns \\
        --input phase1_output/phase1_rollouts.jsonl \\
        --output segment_returns.jsonl \\
        --gamma_tool 0.95 --kappa_ans 1.0

Theory
------

Tool return-to-go (Eq. from TEMPO-RL §3):

    G_{i,t}^{tool} = Σ_{l=t}^{T_i} γ_tool^{l-t} · r_{i,l}^{tool}
                   + κ_ans · γ_tool^{T_i-t} · r_i^{ans}

Answer return:

    G_i^{answer} = r_i^{ans}

Group-relative advantage (GRPO-style):

    A_{z,n} = (G_{z,n} − μ_z^{group}) / (σ_z^{group} + ε)

where *group* = K rollouts of the same (sample, subquestion) prompt,
normalised per segment type z ∈ {tool, answer}.

    z ∈ {tool, answer}
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from TEMPO_RL.io_utils import read_jsonl, write_jsonl

# Regex to strip the _k<digits> rollout-index from rollout_ids, yielding the
# prompt-group key (sample + subquestion) for group-relative normalisation.
# Handles both Phase 1 format:  {task}_sq{sq}_k{N}
# and Phase 2 format:          {sample}_k{N}_sq{sq}
_GRPO_GROUP_RE = re.compile(r"_k\d+")


# ======================================================================
# Conversation reconstruction
# ======================================================================

def _reconstruct_conversation_messages(
    rollout: Dict[str, Any],
    system_template: str = "",
    question: str = "",
) -> List[Dict[str, Any]]:
    """Reconstruct the conversation message structure from a rollout record.

    Returns a list of message dicts with ``{role, content_preview, trainable,
    sequence_index, content_type}``.

    Message sequence (for a rollout with T tool steps + answer)::

        idx 0: system     → trainable=False  (masked)
        idx 1: user       → trainable=False  (masked)
        idx 2: assistant  → trainable=True   (tool_call 0)
        idx 3: tool       → trainable=False  (masked — observation)
        idx 4: assistant  → trainable=True   (tool_call 1)
        idx 5: tool       → trainable=False  (masked)
        ...
        idx N: assistant  → trainable=True   (answer)
    """
    messages: List[Dict[str, Any]] = []

    # System prompt (masked)
    system_content = system_template or rollout.get("system_prompt", "")
    question_text = question or rollout.get("question", "")

    messages.append({
        "sequence_index": len(messages),
        "role": "system",
        "trainable": False,
        "content_type": "system_prompt",
        "content": system_content,
        "content_preview": system_content[:200] if system_content else "(system prompt)",
    })

    # User question (masked)
    messages.append({
        "sequence_index": len(messages),
        "role": "user",
        "trainable": False,
        "content_type": "user_question",
        "content": question_text,
        "content_preview": question_text[:200] if question_text else "(question)",
    })

    agent_steps = rollout.get("agent_steps", [])
    for step in agent_steps:
        if not step.get("invalid_multi_tool"):
            # Normal tool step: assistant tool_call + tool observation
            tcs = step.get("tool_calls", [])
            tool_names = [tc.get("tool_name", "?") for tc in tcs]
            messages.append({
                "sequence_index": len(messages),
                "role": "assistant",
                "trainable": True,
                "content_type": "tool_call",
                "tool_names": tool_names,
                "content_preview": f"<tool_call> → {', '.join(tool_names)}",
                "response_text": step.get("response_text", ""),
                "step_index": step.get("step_index"),
            })
            # Tool observations (masked)
            for obs in step.get("observations", []):
                obs_content_str = str(obs.get("content", ""))
                # Reconstruct exactly as the model saw it during rollout
                if obs.get("success", False):
                    obs_text = f"[Tool Result] {obs_content_str}"
                else:
                    obs_text = f"[ERROR] {obs_content_str}"
                if len(obs_text) > 200:
                    obs_preview = obs_text[:200] + "..."
                else:
                    obs_preview = obs_text
                messages.append({
                    "sequence_index": len(messages),
                    "role": "user",
                    "trainable": False,
                    "content_type": "tool_observation",
                    "tool_name": obs.get("tool_name", ""),
                    "success": obs.get("success", False),
                    "content": obs_text,
                    "content_preview": obs_preview,
                    "step_index": step.get("step_index"),
                })
        else:
            # Multi-tool invalid step: assistant message only (no observation)
            tcs = step.get("tool_calls", [])
            tool_names = [tc.get("tool_name", "?") for tc in tcs]
            messages.append({
                "sequence_index": len(messages),
                "role": "assistant",
                "trainable": True,
                "content_type": "tool_call",
                "tool_names": tool_names,
                "invalid_multi_tool": True,
                "content_preview": f"<tool_call> x{len(tcs)} → INVALID",
                "response_text": step.get("response_text", ""),
                "step_index": step.get("step_index"),
            })
            # Error message (masked)
            messages.append({
                "sequence_index": len(messages),
                "role": "user",
                "trainable": False,
                "content_type": "error_feedback",
                "content_preview": "[ERROR] Multiple tool calls detected...",
                "step_index": step.get("step_index"),
            })

    # Final answer (trainable)
    answer = rollout.get("assistant_answer")
    if answer is not None:
        answer_text = answer.get("content", "")
        if not isinstance(answer_text, str):
            answer_text = str(answer_text)
        messages.append({
            "sequence_index": len(messages),
            "role": "assistant",
            "trainable": True,
            "content_type": "answer",
            "content": answer_text,
            "content_preview": answer_text[:200],
            "response_text": answer_text,
        })

    return messages


# ======================================================================
# Return-to-go computation
# ======================================================================

def compute_tool_return_to_go(
    r_tool_steps: List[float],
    r_answer: float,
    gamma_tool: float = 0.95,
    kappa_ans: float = 1.0,
) -> List[float]:
    """Compute tool return-to-go for each tool step.

    For a rollout with T tool steps::

        G_tool[t] = Σ_{l=t}^{T-1} γ^{l-t} · r_tool[l]
                  + κ_ans · γ^{T-t} · r_answer

    Parameters
    ----------
    r_tool_steps : list of float
        Per-step tool rewards, in execution order.
    r_answer : float
        Answer reward (0 if no answer).
    gamma_tool : float
        Discount factor for future tool rewards.
    kappa_ans : float
        Weight of the answer reward in the tool return-to-go.

    Returns
    -------
    list of float
        Return-to-go for each tool step (same length as r_tool_steps).
    """
    T = len(r_tool_steps)
    if T == 0:
        return []

    G = [0.0] * T

    # Precompute discounted tool rewards forward
    # G[t] = r_tool[t] + γ · r_tool[t+1] + γ² · r_tool[t+2] + ...
    running = 0.0
    for t in reversed(range(T)):
        running = r_tool_steps[t] + gamma_tool * running
        G[t] = running

    # Add discounted answer reward
    # For step t: add κ_ans · γ^{T-t} · r_answer
    for t in range(T):
        disc = gamma_tool ** (T - t)
        G[t] += kappa_ans * disc * r_answer

    return G


def compute_answer_return(r_answer: float) -> float:
    """Answer segment return is simply the answer reward."""
    return r_answer


# ======================================================================
# Normalisation
# ======================================================================

def _z_normalize(
    values: List[float],
    epsilon: float = 1e-8,
) -> Tuple[List[float], float, float]:
    """Z-score normalise a list of values.

    Returns ``(normalised, mean, std)``.
    """
    n = len(values)
    if n == 0:
        return [], 0.0, 0.0

    mean = sum(values) / n
    if n == 1:
        # Single value: std = 0, return 0 after normalisation
        return [0.0], mean, 0.0

    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)

    if std < epsilon:
        return [0.0] * n, mean, std

    return [(v - mean) / (std + epsilon) for v in values], mean, std


# ======================================================================
# Segment builder
# ======================================================================

class SegmentReturnBuilder:
    """Build segment-level training records from rollout trajectories.

    Parameters
    ----------
    gamma_tool : float = 0.95
        Discount factor for tool return-to-go.
    kappa_ans : float = 1.0
        Answer reward scaling in tool return-to-go.
    epsilon : float = 1e-8
        Numerical stability for normalisation.
    """

    def __init__(
        self,
        gamma_tool: float = 0.95,
        kappa_ans: float = 1.0,
        epsilon: float = 1e-8,
    ):
        self.gamma_tool = gamma_tool
        self.kappa_ans = kappa_ans
        self.epsilon = epsilon

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        rollouts: List[Dict[str, Any]],
        system_template: str = "",
    ) -> Dict[str, Any]:
        """Build segments from a list of rollout records.

        Returns a dict with ``segments`` (list), ``normalisation`` (dict),
        ``conversation_masks`` (list).
        """
        # Step 1: Build raw (unnormalised) segments
        raw_segments: List[Dict[str, Any]] = []
        conversation_masks: List[Dict[str, Any]] = []

        for rollout in rollouts:
            rollout_segs, conv_mask = self._build_rollout_segments(
                rollout, system_template
            )
            raw_segments.extend(rollout_segs)
            conversation_masks.append(conv_mask)

        # Step 2: Normalise by segment type
        all_segments, norm_stats = self._normalise_segments(raw_segments)

        return {
            "segments": all_segments,
            "normalisation": norm_stats,
            "conversation_masks": conversation_masks,
        }

    def build_and_save(
        self,
        rollouts: List[Dict[str, Any]],
        output_path: str,
        system_template: str = "",
    ) -> Dict[str, Any]:
        """Build segments and write to ``segment_returns.jsonl``.

        Also writes ``segment_norm_stats.json`` with normalisation parameters.
        """
        result = self.build(rollouts, system_template)

        # Write segments
        write_jsonl(output_path, result["segments"])

        # Write norm stats (useful for the trainer to apply the same normalisation)
        norm_path = output_path.replace(".jsonl", "_norm_stats.json")
        with open(norm_path, "w", encoding="utf-8") as f:
            json.dump(result["normalisation"], f, ensure_ascii=False, indent=2)

        mask_path = output_path.replace(".jsonl", "_conversation_masks.jsonl")
        write_jsonl(mask_path, result["conversation_masks"])

        n_tool = sum(1 for s in result["segments"] if s["segment_type"] == "tool")
        n_answer = sum(1 for s in result["segments"] if s["segment_type"] == "answer")
        print(
            f"[SegmentBuilder] Wrote {len(result['segments'])} segments "
            f"({n_tool} tool, {n_answer} answer) to {output_path}"
        )
        return result

    # ------------------------------------------------------------------
    # Per-rollout segment construction
    # ------------------------------------------------------------------

    def _build_rollout_segments(
        self,
        rollout: Dict[str, Any],
        system_template: str = "",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Build segment list + conversation mask for one rollout."""
        rollout_id = rollout.get("rollout_id", "")
        r_tool_steps = rollout.get("r_tool_steps", [])
        r_answer = rollout.get("r_answer", 0.0)
        agent_steps = rollout.get("agent_steps", [])
        status = rollout.get("status", "")

        segments: List[Dict[str, Any]] = []

        # Compute tool return-to-go
        G_tool = compute_tool_return_to_go(
            r_tool_steps=r_tool_steps,
            r_answer=r_answer,
            gamma_tool=self.gamma_tool,
            kappa_ans=self.kappa_ans,
        )

        # --- Tool segments ---
        for t, g in enumerate(G_tool):
            step = agent_steps[t] if t < len(agent_steps) else {}
            is_invalid = step.get("invalid_multi_tool", False)
            tcs = step.get("tool_calls", [])
            tool_call = tcs[0] if tcs else {}

            segments.append({
                "rollout_id": rollout_id,
                "segment_id": f"{rollout_id}_tool_{t}",
                "segment_type": "tool",
                "step_index": t,
                "return_value": g,  # raw, will be normalised later
                "advantage": g,      # raw, will be normalised later
                "tool_call": tool_call,
                "all_tool_calls_in_turn": tcs,
                "invalid_multi_tool": is_invalid,
                "r_tool_step": r_tool_steps[t] if t < len(r_tool_steps) else 0.0,
                "trainable": True,
                "message_role": "assistant",
                "content_type": "tool_call",
            })

        # --- Answer segment ---
        answer = rollout.get("assistant_answer")
        if answer is not None:
            segments.append({
                "rollout_id": rollout_id,
                "segment_id": f"{rollout_id}_answer",
                "segment_type": "answer",
                "step_index": -1,
                "return_value": r_answer,  # raw, will be normalised later
                "advantage": r_answer,
                "answer": answer,
                "r_answer": r_answer,
                "trainable": True,
                "message_role": "assistant",
                "content_type": "answer",
            })

        # --- Conversation mask ---
        question = rollout.get("question", "")
        conv_mask = {
            "rollout_id": rollout_id,
            "status": status,
            "messages": _reconstruct_conversation_messages(
                rollout, system_template, question
            ),
        }

        return segments, conv_mask

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def _normalise_segments(
        self,
        segments: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Normalise return_value and advantage with **group-relative** z-scores.

        Group = K rollouts of the same (sample, subquestion) prompt.
        Each segment type (tool, answer) is normalised *within* its group,
        producing ``A = (R − μ_group) / (σ_group + ε)``.

        This is the GRPO-style group-relative advantage: no separate value
        model / critic needed — advantages are computed relative to the
        other K−1 rollouts of the same prompt.

        ``raw_return`` preserves the unnormalised return-to-go.
        """

        # Parse group key: strip _k<digits> suffix from rollout_id
        group_by_rollout: Dict[str, str] = {}
        for seg in segments:
            rid = seg["rollout_id"]
            if rid not in group_by_rollout:
                group_by_rollout[rid] = _GRPO_GROUP_RE.sub("", rid)

        # Group (index, raw_return) by (prompt_group, segment_type)
        by_group_and_type: Dict[Tuple[str, str], List[Tuple[int, float]]] = (
            defaultdict(list)
        )

        for i, seg in enumerate(segments):
            group_key = group_by_rollout[seg["rollout_id"]]
            st = seg["segment_type"]
            by_group_and_type[(group_key, st)].append((i, seg["return_value"]))

        # Compute normalisation stats per (group, type)
        norm_stats: Dict[str, Dict[str, float]] = {}

        for (group_key, st), pairs in sorted(by_group_and_type.items()):
            vals = [p[1] for p in pairs]
            normed_vals, mean, std = _z_normalize(vals, self.epsilon)

            label = f"{group_key}/{st}"
            norm_stats[label] = {
                "group": group_key,
                "type": st,
                "count": len(vals),
                "mean": mean,
                "std": std,
                "min": min(vals) if vals else 0.0,
                "max": max(vals) if vals else 0.0,
            }

            # Apply normalisation back to segments
            for (idx, raw_val), norm_val in zip(pairs, normed_vals):
                segments[idx]["return_value"] = norm_val
                segments[idx]["advantage"] = norm_val
                segments[idx]["raw_return"] = raw_val

        return segments, norm_stats


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TEMPO-RL Phase 1 — Build Segment Returns"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to phase1_rollouts.jsonl"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for segment_returns.jsonl"
    )
    parser.add_argument(
        "--gamma_tool", type=float, default=0.95,
        help="Discount factor for tool return-to-go (default 0.95)"
    )
    parser.add_argument(
        "--kappa_ans", type=float, default=1.0,
        help="Answer reward weight in tool return-to-go (default 1.0)"
    )
    parser.add_argument(
        "--epsilon", type=float, default=1e-8,
        help="Numerical stability for normalisation"
    )
    parser.add_argument(
        "--no_normalise", action="store_true",
        help="Skip segment-type normalisation (output raw returns)"
    )
    args = parser.parse_args()

    # Load rollouts
    print(f"[SegmentBuilder] Loading rollouts from {args.input} ...")
    rollouts = read_jsonl(args.input)
    print(f"  Loaded {len(rollouts)} rollout records")

    # Build segments
    builder = SegmentReturnBuilder(
        gamma_tool=args.gamma_tool,
        kappa_ans=args.kappa_ans,
        epsilon=args.epsilon,
    )

    result = builder.build_and_save(
        rollouts=rollouts,
        output_path=args.output,
    )

    # If --no_normalise: overwrite normalised values with raw returns
    if args.no_normalise:
        for seg in result["segments"]:
            seg["return_value"] = seg.get("raw_return", seg["return_value"])
            seg["advantage"] = seg.get("raw_return", seg["advantage"])
        write_jsonl(args.output, result["segments"])
        print("[SegmentBuilder] --no_normalise: re-saved with raw returns")

    # Summary
    segs = result["segments"]
    stats = result["normalisation"]
    n_tool = sum(1 for s in segs if s["segment_type"] == "tool")
    n_answer = sum(1 for s in segs if s["segment_type"] == "answer")

    print(f"\n[SegmentBuilder] Summary:")
    print(f"  Segments: {n_tool} tool + {n_answer} answer = {len(segs)} total")
    for st, info in stats.items():
        print(
            f"  {st}: μ={info['mean']:.4f}, σ={info['std']:.4f}, "
            f"n={info['count']}, range=[{info['min']:.4f}, {info['max']:.4f}]"
        )


if __name__ == "__main__":
    main()
