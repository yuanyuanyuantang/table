"""
TEMPO-RL Phase 2 — Segment Return Builder.

Converts ``phase2_dialog_rollouts.jsonl`` into segment-level training records
with tool / answer / memory segment types.

Tool return-to-go (same as Phase 1):

    G_{i,t}^{tool} = Σ_{l=t}^{T_i} γ_tool^{l-t} · r_{i,l}^{tool}
                    + κ_ans · γ_tool^{T_i-t} · r_i^{ans}

Answer return:

    G_i^{answer} = r_i^{ans}

Memory return:

    G_i^{memory} = r_i^{mem}

Group-relative advantage (GRPO-style):

    A_{z,n} = (G_{z,n} − μ_z^{group}) / (σ_z^{group} + ε)

where *group* = K rollouts of the same (sample, subquestion) prompt,
normalised per segment type z ∈ {tool, answer, memory}.

Usage::

    python -m TEMPO_RL.build_segment_returns_phase2 \\
        --input phase2_output/phase2_dialog_rollouts.jsonl \\
        --output phase2_output/phase2_segment_returns.jsonl \\
        --gamma_tool 0.95 --kappa_ans 1.0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re as _re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from TEMPO_RL.io_utils import read_jsonl, write_jsonl

# Regex to strip the _k<digits> rollout-index from rollout_ids, yielding the
# prompt-group key (sample + subquestion) for group-relative normalisation.
# Handles both Phase 1 format:  {task}_sq{sq}_k{N}
# and Phase 2 format:          {sample}_k{N}_sq{sq}
_GRPO_GROUP_RE = _re.compile(r"_k\d+")


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
    """
    T = len(r_tool_steps)
    if T == 0:
        return []

    G = [0.0] * T
    running = 0.0
    for t in reversed(range(T)):
        running = r_tool_steps[t] + gamma_tool * running
        G[t] = running

    for t in range(T):
        disc = gamma_tool ** (T - t)
        G[t] += kappa_ans * disc * r_answer

    return G


def compute_answer_return(r_answer: float) -> float:
    return r_answer


def compute_memory_return(r_memory: float) -> float:
    return r_memory


# ======================================================================
# Normalisation
# ======================================================================

def _z_normalize(
    values: List[float],
    epsilon: float = 1e-8,
) -> Tuple[List[float], float, float]:
    """Z-score normalise a list of values."""
    n = len(values)
    if n == 0:
        return [], 0.0, 0.0

    mean = sum(values) / n
    if n == 1:
        return [0.0], mean, 0.0

    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)

    if std < epsilon:
        return [0.0] * n, mean, std

    return [(v - mean) / (std + epsilon) for v in values], mean, std


# ======================================================================
# Conversation reconstruction
# ======================================================================

def _reconstruct_conversation_messages(
    sq_rollout: Dict[str, Any],
    system_template: str = "",
    question: str = "",
) -> List[Dict[str, Any]]:
    """Reconstruct conversation messages for a single subquestion rollout.

    Message sequence (for T tool steps + answer + memory)::

        idx 0: system        → trainable=False (masked)
        idx 1: user          → trainable=False (masked)
        idx 2: assistant     → trainable=True  (tool_call 0)
        idx 3: tool          → trainable=False (masked)
        ...
        idx M: assistant     → trainable=True  (answer)
        idx N: assistant     → trainable=True  (memory)
    """
    messages: List[Dict[str, Any]] = []

    system_content = system_template or ""
    question_text = question or sq_rollout.get("question", "")

    messages.append({
        "sequence_index": len(messages),
        "role": "system",
        "trainable": False,
        "content_type": "system_prompt",
        "content": system_content,
        "content_preview": system_content[:200] if system_content else "(system prompt)",
    })

    # User question (may include memory hint)
    full_question = question_text
    messages.append({
        "sequence_index": len(messages),
        "role": "user",
        "trainable": False,
        "content_type": "user_question",
        "content": full_question,
        "content_preview": full_question[:200] if full_question else "(question)",
    })

    agent_steps = sq_rollout.get("agent_steps", [])
    for step in agent_steps:
        if not step.get("invalid_multi_tool"):
            tcs = step.get("tool_calls", [])
            tool_names = [tc.get("tool_name", "?") for tc in tcs]
            messages.append({
                "sequence_index": len(messages),
                "role": "assistant",
                "trainable": True,
                "content_type": "tool_call",
                "tool_names": tool_names,
                "content_preview": f"<tool_call> -> {', '.join(tool_names)}",
                "response_text": step.get("response_text", ""),
                "step_index": step.get("step_index"),
            })
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
            tcs = step.get("tool_calls", [])
            tool_names = [tc.get("tool_name", "?") for tc in tcs]
            messages.append({
                "sequence_index": len(messages),
                "role": "assistant",
                "trainable": True,
                "content_type": "tool_call",
                "tool_names": tool_names,
                "invalid_multi_tool": True,
                "content_preview": f"<tool_call> x{len(tcs)} -> INVALID",
                "response_text": step.get("response_text", ""),
                "step_index": step.get("step_index"),
            })
            messages.append({
                "sequence_index": len(messages),
                "role": "user",
                "trainable": False,
                "content_type": "error_feedback",
                "content_preview": "[ERROR] Multiple tool calls detected...",
                "step_index": step.get("step_index"),
            })

    # Answer (trainable)
    answer = sq_rollout.get("assistant_answer")
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

    # Memory (trainable)
    memory_output = sq_rollout.get("memory_output")
    if memory_output is not None:
        mem_str = json.dumps(memory_output, ensure_ascii=False) if isinstance(memory_output, dict) else str(memory_output)
        messages.append({
            "sequence_index": len(messages),
            "role": "assistant",
            "trainable": True,
            "content_type": "memory",
            "content": mem_str,
            "response_text": mem_str,
            "content_preview": mem_str[:200],
        })

    return messages


# ======================================================================
# Segment builder
# ======================================================================

class SegmentReturnBuilderPhase2:
    """Build segment-level training records from dialog rollout trajectories.

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
        dialog_rollouts: List[Dict[str, Any]],
        system_template: str = "",
    ) -> Dict[str, Any]:
        """Build segments from a list of dialog rollout records.

        Returns a dict with ``segments`` (list), ``normalisation`` (dict),
        ``conversation_masks`` (list).
        """
        raw_segments: List[Dict[str, Any]] = []
        conversation_masks: List[Dict[str, Any]] = []

        for rollout in dialog_rollouts:
            rollout_segs, conv_masks = self._build_dialog_segments(
                rollout, system_template
            )
            raw_segments.extend(rollout_segs)
            conversation_masks.extend(conv_masks)

        all_segments, norm_stats = self._normalise_segments(raw_segments)

        return {
            "segments": all_segments,
            "normalisation": norm_stats,
            "conversation_masks": conversation_masks,
        }

    def build_and_save(
        self,
        dialog_rollouts: List[Dict[str, Any]],
        output_path: str,
        system_template: str = "",
    ) -> Dict[str, Any]:
        """Build segments and write output files."""
        result = self.build(dialog_rollouts, system_template)

        write_jsonl(output_path, result["segments"])

        norm_path = output_path.replace(".jsonl", "_norm_stats.json")
        with open(norm_path, "w", encoding="utf-8") as f:
            json.dump(result["normalisation"], f, ensure_ascii=False, indent=2)

        mask_path = output_path.replace(".jsonl", "_conversation_masks.jsonl")
        write_jsonl(mask_path, result["conversation_masks"])

        n_tool = sum(1 for s in result["segments"] if s["segment_type"] == "tool")
        n_answer = sum(1 for s in result["segments"] if s["segment_type"] == "answer")
        n_memory = sum(1 for s in result["segments"] if s["segment_type"] == "memory")
        print(
            f"[SegmentBuilderPhase2] Wrote {len(result['segments'])} segments "
            f"({n_tool} tool, {n_answer} answer, {n_memory} memory) to {output_path}"
        )
        return result

    # ------------------------------------------------------------------
    # Per-dialog segment construction
    # ------------------------------------------------------------------

    def _build_dialog_segments(
        self,
        dialog_rollout: Dict[str, Any],
        system_template: str = "",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build segment list + conversation masks for one dialog rollout."""
        all_segments: List[Dict[str, Any]] = []
        all_masks: List[Dict[str, Any]] = []

        rollout_id = dialog_rollout.get("rollout_id", "")
        sq_rollouts = dialog_rollout.get("subquestion_rollouts", [])

        for sq_rollout in sq_rollouts:
            sq_id = sq_rollout.get("sq_id", 0)
            sq_rollout_id = f"{rollout_id}_sq{sq_id}"

            segments, conv_mask = self._build_subquestion_segments(
                sq_rollout, sq_rollout_id, system_template
            )
            all_segments.extend(segments)
            all_masks.append(conv_mask)

        return all_segments, all_masks

    def _build_subquestion_segments(
        self,
        sq_rollout: Dict[str, Any],
        sq_rollout_id: str,
        system_template: str = "",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Build segments for one subquestion within a dialog."""
        r_tool_steps = sq_rollout.get("r_tool_steps", [])
        r_answer = sq_rollout.get("r_answer", 0.0)
        r_memory = sq_rollout.get("r_memory", 0.0)
        agent_steps = sq_rollout.get("agent_steps", [])
        status = sq_rollout.get("status", "")
        memory_output = sq_rollout.get("memory_output")
        memory_severe_failure = sq_rollout.get("memory_severe_failure", False)

        segments: List[Dict[str, Any]] = []

        # --- Tool return-to-go ---
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
                "rollout_id": sq_rollout_id,
                "segment_id": f"{sq_rollout_id}_tool_{t}",
                "segment_type": "tool",
                "step_index": t,
                "return_value": g,
                "advantage": g,
                "tool_call": tool_call,
                "all_tool_calls_in_turn": tcs,
                "invalid_multi_tool": is_invalid,
                "r_tool_step": r_tool_steps[t] if t < len(r_tool_steps) else 0.0,
                "trainable": True,
                "message_role": "assistant",
                "content_type": "tool_call",
            })

        # --- Answer segment ---
        answer = sq_rollout.get("assistant_answer")
        if answer is not None:
            segments.append({
                "rollout_id": sq_rollout_id,
                "segment_id": f"{sq_rollout_id}_answer",
                "segment_type": "answer",
                "step_index": -1,
                "return_value": r_answer,
                "advantage": r_answer,
                "answer": answer,
                "r_answer": r_answer,
                "trainable": True,
                "message_role": "assistant",
                "content_type": "answer",
            })

        # --- Memory segment ---
        if memory_output is not None:
            segments.append({
                "rollout_id": sq_rollout_id,
                "segment_id": f"{sq_rollout_id}_memory",
                "segment_type": "memory",
                "step_index": -2,
                "return_value": r_memory,
                "advantage": r_memory,
                "memory_output": memory_output,
                "r_memory": r_memory,
                "severe_failure": memory_severe_failure,
                "trainable": True,
                "message_role": "assistant",
                "content_type": "memory",
            })

        # --- Conversation mask ---
        question = sq_rollout.get("question", "")
        conv_mask = {
            "rollout_id": sq_rollout_id,
            "sq_id": sq_rollout.get("sq_id", 0),
            "status": status,
            "messages": _reconstruct_conversation_messages(
                sq_rollout, system_template, question
            ),
        }

        return segments, conv_mask

    def merge_counterfactual_audit(
        self,
        segments: List[Dict[str, Any]],
        audit_path: str,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Overwrite memory segment rewards with counterfactual ``r_mem_final``.

        Reads ``phase3_counterfactual_audit.jsonl`` and updates matching
        memory segments in-place.  Memory segment ``rollout_id`` is
        ``{dialog_rollout_id}_sq{i}`` and the audit record key is
        ``{dialog_rollout_id}`` + ``boundary`` (i.e. *i*).

        Returns ``(segments, n_updated)``.
        """
        audit_records = read_jsonl(audit_path)
        if not audit_records:
            return segments, 0

        # Index audit records by (rollout_id, boundary)
        audit_index: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for rec in audit_records:
            key = (rec.get("rollout_id", ""), rec.get("boundary", 0))
            audit_index[key] = rec

        n_updated = 0
        for seg in segments:
            if seg.get("segment_type") != "memory":
                continue
            # Segment rollout_id is e.g. "sample_k0_sq1" →
            #   dialog rollout id = "sample_k0", boundary (i) = 1
            seg_rollout_id = seg.get("rollout_id", "")
            # Extract SQ number from the rollout_id suffix
            m = _re.search(r'_sq(\d+)$', seg_rollout_id)
            if not m:
                continue
            boundary = int(m.group(1))
            # Dialog rollout id is everything before _sqN
            dialog_rollout_id = seg_rollout_id[:m.start()]
            audit_key = (dialog_rollout_id, boundary)
            if audit_key not in audit_index:
                continue

            audit_entry = audit_index[audit_key]
            r_mem_final = audit_entry.get("r_mem_final")
            if r_mem_final is None:
                continue

            seg["return_value"] = r_mem_final
            seg["advantage"] = r_mem_final
            seg["r_memory"] = r_mem_final
            seg["cf_adjusted"] = True
            seg["cf_contribution"] = audit_entry.get("cf_contribution", 0.0)
            seg["cf_delta_u"] = audit_entry.get("delta_u", 0.0)
            n_updated += 1

        return segments, n_updated

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def _normalise_segments(
        self,
        segments: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Normalise return_value and advantage with **group-relative** z-scores.

        Group = K rollouts of the same (sample, subquestion) prompt.
        Each segment type (tool, answer, memory) is normalised *within* its
        group, producing ``A = (R − μ_group) / (σ_group + ε)``.

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
        description="TEMPO-RL Phase 2 — Build Segment Returns"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to phase2_dialog_rollouts.jsonl"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for phase2_segment_returns.jsonl"
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
        "--counterfactual_audit", default="",
        help="Path to phase3_counterfactual_audit.jsonl (optional merge)"
    )
    args = parser.parse_args()

    print(f"[SegmentBuilderPhase2] Loading dialog rollouts from {args.input} ...")
    rollouts = read_jsonl(args.input)
    print(f"  Loaded {len(rollouts)} dialog rollout records")

    builder = SegmentReturnBuilderPhase2(
        gamma_tool=args.gamma_tool,
        kappa_ans=args.kappa_ans,
        epsilon=args.epsilon,
    )

    result = builder.build_and_save(
        dialog_rollouts=rollouts,
        output_path=args.output,
    )

    # --- Merge counterfactual audit if provided ---
    if args.counterfactual_audit:
        segs, n_cf = builder.merge_counterfactual_audit(
            result["segments"], args.counterfactual_audit
        )
        # Re-normalise after merge so CF-adjusted memory rewards are
        # consistent with the group-relative advantage scale
        segs, new_stats = builder._normalise_segments(segs)
        result["segments"] = segs
        result["normalisation"] = new_stats
        write_jsonl(args.output, segs)
        print(f"[SegmentBuilderPhase2] Merged {n_cf} counterfactual "
              f"memory rewards from {args.counterfactual_audit}")

    segs = result["segments"]
    stats = result["normalisation"]
    n_tool = sum(1 for s in segs if s["segment_type"] == "tool")
    n_answer = sum(1 for s in segs if s["segment_type"] == "answer")
    n_memory = sum(1 for s in segs if s["segment_type"] == "memory")
    n_cf_adjusted = sum(1 for s in segs if s.get("cf_adjusted"))

    print(f"\n[SegmentBuilderPhase2] Summary:")
    print(f"  Segments: {n_tool} tool + {n_answer} answer + {n_memory} memory "
          f"= {len(segs)} total"
          + (f" ({n_cf_adjusted} cf-adjusted)" if n_cf_adjusted else ""))
    for st, info in stats.items():
        print(
            f"  {st}: mu={info['mean']:.4f}, sigma={info['std']:.4f}, "
            f"n={info['count']}, range=[{info['min']:.4f}, {info['max']:.4f}]"
        )


if __name__ == "__main__":
    main()
