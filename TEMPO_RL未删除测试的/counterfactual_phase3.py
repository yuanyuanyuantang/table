"""
TEMPO-RL Phase 3 — Sparse Counterfactual Memory RL.

Estimates the marginal utility of model-generated memory for future
topologically-related subquestions by running paired continuations:

    Continuation A: uses M_i^{gen} (the model's generated memory)
    Continuation B: uses M_{i-1}   (previous subquestion's memory)

    ΔU_i = r_{j*}^{ans}(M_i^{gen}) - r_{j*}^{ans}(M_{i-1})

    r_i^{mem-final} = r_i^{mem} + λ_cf * (
        1[F_i >= τ_f] * [ΔU_i]_+ + [ΔU_i]_-
    )

Usage::

    python -m TEMPO_RL.counterfactual_phase3 \\
        --dialog_rollouts phase2_output/phase2_dialog_rollouts.jsonl \\
        --samples dataset/train不含val的.json \\
        --target_evidence output/target_evidence.jsonl \\
        --future_dependencies output/future_dependencies.jsonl \\
        --table_root dataset/table \\
        --output_dir phase3_output \\
        --sparse_rate 0.25 --lambda_cf 0.2 --tau_f 0.8
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from TEMPO_RL.schemas import TargetEvidenceSet, FutureDependencySet, FutureDependency
from TEMPO_RL.evidence_ledger import EvidenceLedger
from TEMPO_RL.reward_calculator import RewardCalculator
from TEMPO_RL.rollout_phase2 import (
    PolicyWrapper,
    ToolExecutor,
    FakeToolExecutor,
    MockPolicy,
)
from TEMPO_RL.io_utils import (
    read_jsonl,
    write_jsonl,
    load_json_file,
    try_parse_json,
    extract_tool_calls_from_response,
    extract_answer_from_response,
    extract_memory_from_response,
    get_sample_id,
    DEFAULT_SYSTEM_TEMPLATE,
)


# ======================================================================
# Dependency Topology
# ======================================================================

def _parse_sq_id(sq_str: str) -> int:
    """Parse subquestion ID from string like 'sq2' -> 2."""
    if isinstance(sq_str, int):
        return sq_str
    if not isinstance(sq_str, str):
        return 0
    digits = ''.join(c for c in sq_str if c.isdigit())
    return int(digits) if digits else 0


def _get_dependency_topology(
    fds_lookup: Dict[Tuple[str, str], FutureDependencySet],
    sample_id: str,
    n_subquestions: int,
) -> Dict[int, Set[int]]:
    """Build ρ_ij matrix: for each subquestion i, the set of j>i that depend on it.

    Returns ``{i: {j1, j2, ...}}`` where j > i and ρ_ij = 1.
    """
    topology: Dict[int, Set[int]] = {}
    for i in range(1, n_subquestions):
        boundary = f"after_sq{i}"
        fds = fds_lookup.get((sample_id, boundary))
        if fds is None:
            continue

        dependent_sqs: Set[int] = set()
        for dep in fds.future_dependencies:
            j = _parse_sq_id(dep.needed_by)
            if j > i:
                dependent_sqs.add(j)
        if dependent_sqs:
            topology[i] = dependent_sqs

    return topology


def _get_first_dependent_sq(
    topology: Dict[int, Set[int]],
    i: int,
) -> Optional[int]:
    """Get j* = min { j > i : ρ_ij = 1 }."""
    if i not in topology:
        return None
    candidates = [j for j in topology[i] if j > i]
    return min(candidates) if candidates else None


# ======================================================================
# Counterfactual Estimator
# ======================================================================

class CounterfactualEstimator:
    """Estimate marginal utility of memory via paired continuations.

    Parameters
    ----------
    policy : PolicyWrapper
        Policy model for running continuations.
    tool_executor : ToolExecutor
        Tool execution backend.
    calculator : RewardCalculator
        Reward computation (for answer rewards in continuations).
    tes_lookup : dict
        ``(sample_id, subquestion_index) -> TargetEvidenceSet``.
    fds_lookup : dict
        ``(sample_id, boundary) -> FutureDependencySet``.
    lambda_cf : float = 0.2
        Weight of counterfactual delta in final memory reward.
    a : float = 1.0
        Clipping bound for ΔU_i.
    tau_f : float = 0.8
        Faithfulness gate threshold.
    sparse_rate : float = 0.25
        Fraction of eligible boundaries to actually run (0.2-0.3).
    temperature : float = 0.3
        Low temperature for deterministic continuations.
    top_p : float = 0.9
    max_tool_steps_per_turn : int = 6
    seed : int = 42
        Random seed for reproducibility and sparse sampling.
    """

    def __init__(
        self,
        policy: PolicyWrapper,
        tool_executor: ToolExecutor,
        calculator: RewardCalculator,
        tes_lookup: Dict[Tuple[str, int], TargetEvidenceSet],
        fds_lookup: Dict[Tuple[str, str], FutureDependencySet],
        lambda_cf: float = 0.2,
        a: float = 1.0,
        tau_f: float = 0.8,
        sparse_rate: float = 0.25,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tool_steps_per_turn: int = 6,
        seed: int = 42,
    ):
        self.policy = policy
        self.tool_executor = tool_executor
        self.calculator = calculator
        self.tes_lookup = tes_lookup
        self.fds_lookup = fds_lookup
        self.lambda_cf = lambda_cf
        self.a = a
        self.tau_f = tau_f
        self.sparse_rate = sparse_rate
        self.temperature = temperature
        self.top_p = top_p
        self.max_tool_steps_per_turn = max_tool_steps_per_turn
        self.seed = seed
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(
        self,
        dialog_rollouts: List[Dict[str, Any]],
        samples: List[Dict[str, Any]],
        tools_schema: List[Dict[str, Any]],
        system_template: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Run counterfactual estimation on eligible dialog rollouts.

        Returns ``(audit_records, summary)``.
        """
        audit_records: List[Dict[str, Any]] = []
        n_eligible = 0
        n_executed = 0
        n_skipped = 0

        sample_index = _index_samples_by_task(samples)

        for rollout in dialog_rollouts:
            sample_id = get_sample_id(rollout)
            n_sq = rollout.get("n_subquestions", 0)
            if n_sq < 2:
                continue

            # Get dependency topology
            topology = _get_dependency_topology(
                self.fds_lookup, sample_id, n_sq
            )

            # Find eligible boundaries
            sq_rollouts = rollout.get("subquestion_rollouts", [])
            for sq_rollout in sq_rollouts:
                i = sq_rollout.get("sq_id", 0)
                j_star = _get_first_dependent_sq(topology, i)
                if j_star is None:
                    continue

                n_eligible += 1

                # Sparse: only run with probability sparse_rate
                if self._rng.random() > self.sparse_rate:
                    n_skipped += 1
                    continue

                n_executed += 1

                # Run counterfactual estimation
                audit = self._estimate_one_boundary(
                    rollout=rollout,
                    sq_rollout=sq_rollout,
                    i=i,
                    j_star=j_star,
                    sample_index=sample_index,
                    tools_schema=tools_schema,
                    system_template=system_template,
                )
                audit_records.append(audit)

        summary = {
            "n_dialog_rollouts": len(dialog_rollouts),
            "n_eligible_boundaries": n_eligible,
            "n_executed": n_executed,
            "n_skipped": n_skipped,
            "sparse_rate": self.sparse_rate,
            "lambda_cf": self.lambda_cf,
            "tau_f": self.tau_f,
            "a": self.a,
        }

        return audit_records, summary

    def estimate_and_save(
        self,
        dialog_rollouts: List[Dict[str, Any]],
        samples: List[Dict[str, Any]],
        tools_schema: List[Dict[str, Any]],
        output_dir: str,
        system_template: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Run estimation and write ``phase3_counterfactual_audit.jsonl``."""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "phase3_counterfactual_audit.jsonl")

        audit_records, summary = self.estimate(
            dialog_rollouts, samples, tools_schema, system_template
        )

        write_jsonl(output_path, audit_records)

        # Write summary
        summary_path = os.path.join(output_dir, "phase3_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(
            f"[Phase3] Wrote {len(audit_records)} counterfactual audits to {output_path}"
        )
        print(
            f"[Phase3] Eligible={summary['n_eligible_boundaries']} "
            f"Executed={summary['n_executed']} "
            f"Skipped={summary['n_skipped']} "
            f"(sparse_rate={summary['sparse_rate']})"
        )
        return audit_records, summary

    # ------------------------------------------------------------------
    # Per-boundary estimation
    # ------------------------------------------------------------------

    def _estimate_one_boundary(
        self,
        rollout: Dict[str, Any],
        sq_rollout: Dict[str, Any],
        i: int,
        j_star: int,
        sample_index: Dict[str, Dict[str, Any]],
        tools_schema: List[Dict[str, Any]],
        system_template: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run paired continuations for boundary i → j*.

        Continuation A: uses M_i^{gen} (model's generated memory)
        Continuation B: uses M_{i-1} (previous memory, or None for first)
        """
        sample_id = get_sample_id(rollout)
        sample = sample_index.get(sample_id, {})
        subquestions = sample.get("design", {}).get("checkout_list", sample.get("subquestions", []))

        # Memory from current subquestion i (generated)
        memory_gen = sq_rollout.get("memory_output")
        if memory_gen is not None and isinstance(memory_gen, str):
            memory_gen = try_parse_json(memory_gen)

        # Memory from previous subquestion i-1
        memory_prev = sq_rollout.get("memory_before")

        # Get table path
        table_path = sample.get("table_path", sample.get("table_root", ""))

        # Get faithfulness score from existing memory reward detail
        mem_detail = sq_rollout.get("memory_reward_detail", {})
        F_i = float(mem_detail.get("F_i", 0.0))

        # Run Continuation A (with M_i^{gen})
        r_ans_gen, audit_gen = self._run_continuation(
            sample_id=sample_id,
            sample=sample,
            subquestions=subquestions,
            start_sq=i,
            end_sq=j_star,
            memory_to_use=memory_gen,
            table_path=table_path,
            tools_schema=tools_schema,
            system_template=system_template,
            label="A",
        )

        # Run Continuation B (with M_{i-1})
        r_ans_prev, audit_prev = self._run_continuation(
            sample_id=sample_id,
            sample=sample,
            subquestions=subquestions,
            start_sq=i,
            end_sq=j_star,
            memory_to_use=memory_prev,
            table_path=table_path,
            tools_schema=tools_schema,
            system_template=system_template,
            label="B",
        )

        # Compute ΔU_i
        delta_u = r_ans_gen - r_ans_prev
        clipped_delta_u = max(-self.a, min(self.a, delta_u))

        # Faithfulness gate
        faithfulness_gate = F_i >= self.tau_f
        positive_part = max(0.0, clipped_delta_u) if faithfulness_gate else 0.0
        negative_part = min(0.0, clipped_delta_u)

        # Counterfactual contribution
        cf_contribution = self.lambda_cf * (positive_part + negative_part)

        # Final memory reward
        r_mem_original = sq_rollout.get("r_memory", 0.0)
        r_mem_final = r_mem_original + cf_contribution

        # Clip final reward to [-1, 1]
        r_mem_final = max(-1.0, min(1.0, r_mem_final))

        return {
            "sample_id": sample_id,
            "rollout_id": rollout.get("rollout_id", ""),
            "boundary": i,
            "j_star": j_star,
            "F_i": F_i,
            "faithfulness_gate": faithfulness_gate,
            "r_mem_original": r_mem_original,
            "r_ans_gen": r_ans_gen,
            "r_ans_prev": r_ans_prev,
            "delta_u": delta_u,
            "clipped_delta_u": clipped_delta_u,
            "positive_part": positive_part,
            "negative_part": negative_part,
            "cf_contribution": cf_contribution,
            "r_mem_final": r_mem_final,
            "lambda_cf": self.lambda_cf,
            "tau_f": self.tau_f,
            "a": self.a,
            "continuation_A_audit": audit_gen,
            "continuation_B_audit": audit_prev,
        }

    # ------------------------------------------------------------------
    # Continuation execution
    # ------------------------------------------------------------------

    def _run_continuation(
        self,
        sample_id: str,
        sample: Dict[str, Any],
        subquestions: List[Dict[str, Any]],
        start_sq: int,
        end_sq: int,
        memory_to_use: Optional[Dict[str, Any]],
        table_path: str,
        tools_schema: List[Dict[str, Any]],
        system_template: Optional[str] = None,
        label: str = "",
    ) -> Tuple[float, Dict[str, Any]]:
        """Run subquestions from (start_sq+1) to (end_sq) using the given memory.

        Returns ``(r_answer_for_end_sq, continuation_audit)``.
        """
        current_memory = memory_to_use
        subq_results: List[Dict[str, Any]] = []
        r_answer_target = 0.0

        for sq_idx in range(start_sq, end_sq):
            sq = subquestions[sq_idx] if sq_idx < len(subquestions) else {}
            sq_id = sq_idx + 1  # 1-indexed

            question = sq.get("question", sq.get("cq", ""))
            if not question:
                question = sq.get("checkout_item", {}).get("checkout_text", "")

            score_points = sq.get("score_points", sq.get("checkout_item", {}).get("score_points", []))
            if isinstance(score_points, str):
                try:
                    score_points = json.loads(score_points)
                except (json.JSONDecodeError, TypeError):
                    score_points = [score_points]
            if not isinstance(score_points, list):
                score_points = []

            # Look up TES
            tes = self.tes_lookup.get((sample_id, sq_id))
            if tes is None:
                tes = self.tes_lookup.get((sample_id, sq_idx))

            # Look up FDS
            boundary_key = f"after_sq{sq_id - 1}" if sq_idx > 0 else "root"
            fds = self.fds_lookup.get((sample_id, boundary_key))

            # Run this subquestion
            sq_result = self._run_one_subquestion(
                sample_id=sample_id,
                sq_id=sq_id,
                question=question,
                score_points=score_points,
                tes=tes,
                fds=fds,
                table_path=table_path,
                tools_schema=tools_schema,
                previous_memory=current_memory,
                system_template=system_template,
            )

            subq_results.append(sq_result)

            # Update memory for next subquestion
            mem_out = sq_result.get("memory_output")
            if mem_out is not None:
                parsed = try_parse_json(
                    json.dumps(mem_out, ensure_ascii=False)
                    if isinstance(mem_out, dict) else str(mem_out)
                )
                if parsed is not None:
                    current_memory = parsed
                # else: keep current_memory as fallback
            # else: keep current_memory

            # If this is the target subquestion, capture the answer reward
            if sq_id == end_sq:
                r_answer_target = sq_result.get("r_answer", 0.0)

        return r_answer_target, {
            "label": label,
            "start_sq": start_sq,
            "end_sq": end_sq,
            "subq_results": subq_results,
            "r_answer_target": r_answer_target,
        }

    def _run_one_subquestion(
        self,
        sample_id: str,
        sq_id: int,
        question: str,
        score_points: List[str],
        tes: Optional[TargetEvidenceSet],
        fds: Optional[FutureDependencySet],
        table_path: str,
        tools_schema: List[Dict[str, Any]],
        previous_memory: Optional[Dict[str, Any]] = None,
        system_template: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a single subquestion within a continuation.

        This is a stripped-down version of DialogRolloutRunner._run_subquestion_rollout
        optimized for counterfactual continuations (lower temperature, deterministic).
        """
        task_id = f"{sample_id}_sq{sq_id}"

        system_msg = (system_template or DEFAULT_SYSTEM_TEMPLATE).format(
            table_path=table_path,
        )

        memory_hint = ""
        if previous_memory is not None:
            memory_str = json.dumps(previous_memory, ensure_ascii=False, indent=2)
            memory_hint = (
                f"\n\n**Memory from previous subquestion (use this information!):**\n"
                f"```json\n{memory_str}\n```"
            )

        if tes is not None:
            ledger = EvidenceLedger(tes)
            ledger.initialize_from_memory(previous_memory or {})
        else:
            ledger = EvidenceLedger(
                TargetEvidenceSet(
                    sample_id=sample_id, subquestion_id=sq_id, question=question,
                )
            )
            ledger.initialize_from_memory({})

        full_question = question + memory_hint
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": full_question},
        ]

        agent_steps: List[Dict[str, Any]] = []
        observations_list: List[Dict[str, Any]] = []
        code_outputs_list: List[str] = []
        r_tool_steps: List[float] = []

        assistant_answer: Optional[Dict[str, Any]] = None
        memory_output: Optional[Dict[str, Any]] = None
        status = "in_progress"

        try:
            for step_idx in range(self.max_tool_steps_per_turn):
                response = self.policy.call(
                    messages=messages,
                    tools=tools_schema,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                content = response.get("content", "")

                answer_text = extract_answer_from_response(content)
                tool_calls_in_turn = extract_tool_calls_from_response(content)
                has_answer = answer_text is not None
                has_tools = len(tool_calls_in_turn) > 0

                if has_answer and not has_tools:
                    assistant_answer = {"content": answer_text, "full_response": content}
                    mem = extract_memory_from_response(content)
                    if mem is not None:
                        memory_output = mem
                    status = "completed"
                    break

                if not has_tools and not has_answer:
                    status = "no_tool_or_answer"
                    break

                if len(tool_calls_in_turn) > 1:
                    status = "invalid_multi_tool"
                    agent_steps.append({
                        "step_index": step_idx,
                        "type": "tool_call",
                        "tool_calls": tool_calls_in_turn,
                        "observations": [],
                        "invalid_multi_tool": True,
                    })
                    r_tool = -(self.calculator.lambda_invalid)
                    r_tool_steps.append(r_tool)
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": "[ERROR] Multiple tool calls detected. "
                                   "Only ONE tool call per turn is allowed.",
                    })
                    continue

                tool_call = tool_calls_in_turn[0]
                tool_name = tool_call.get("tool_name", "")
                arguments = tool_call.get("arguments", {})

                obs = self.tool_executor.execute(tool_name, arguments)
                observations_list.append(obs)

                code_output = obs.get("content", "") if tool_name in (
                    "python_code_executor", "code_executor"
                ) else ""
                code_outputs_list.append(code_output)

                metadata = self.tool_executor.extract_metadata(tool_name, arguments, obs)
                ledger_result = ledger.update(
                    tool_call=tool_call,
                    observation=obs,
                    observation_metadata=metadata,
                    code_output=code_output,
                )

                tool_reward_result = self.calculator.compute_tool_reward(
                    tool_call=tool_call,
                    ledger_update_result=ledger_result,
                    subquestion_id=str(sq_id),
                )
                r_tool = tool_reward_result["r_tool"]
                r_tool_steps.append(r_tool)

                agent_steps.append({
                    "step_index": step_idx,
                    "type": "tool_call",
                    "tool_calls": tool_calls_in_turn,
                    "observations": [obs],
                    "invalid_multi_tool": False,
                })

                messages.append({"role": "assistant", "content": content})
                if obs.get("success"):
                    messages.append({
                        "role": "user",
                        "content": f"[Tool Result] {obs.get('content', '')}",
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": f"[ERROR] {obs.get('content', '')}",
                    })

                if has_answer:
                    assistant_answer = {"content": answer_text, "full_response": content}
                    mem = extract_memory_from_response(content)
                    if mem is not None:
                        memory_output = mem
                    status = "completed"
                    break

            if status == "in_progress":
                status = "truncated"
                response = self.policy.call(
                    messages=messages, tools=tools_schema,
                    temperature=self.temperature, top_p=self.top_p,
                )
                content = response.get("content", "")
                answer_text = extract_answer_from_response(content)
                if answer_text:
                    assistant_answer = {"content": answer_text, "full_response": content}
                mem = extract_memory_from_response(content)
                if mem is not None:
                    memory_output = mem

        except Exception as e:
            status = f"error: {e}"

        r_answer = 0.0
        grounded_answer_claims = None
        if assistant_answer is not None and tes is not None:
            answer_reward_result = self.calculator.compute_answer_reward(
                answer_json=assistant_answer.get("content", ""),
                score_points=score_points,
                ledger=ledger,
                memory_before=previous_memory,
                observations=observations_list,
                code_outputs=code_outputs_list,
            )
            r_answer = answer_reward_result.get("r_answer", 0.0)
            grounded_answer_claims = answer_reward_result.get("claim_results")

        r_memory = 0.0
        if memory_output is not None:
            memory_reward_result = self.calculator.compute_memory_reward(
                memory_after=memory_output,
                memory_before=previous_memory,
                ledger=ledger,
                observations=observations_list,
                code_outputs=code_outputs_list,
                future_dependency_set=fds,
                grounded_answer_claims=grounded_answer_claims,
            )
            r_memory = memory_reward_result.get("r_memory", 0.0)
            if memory_reward_result.get("severe_failure", False):
                r_memory = -1.0

        return {
            "sample_id": sample_id,
            "sq_id": sq_id,
            "task_id": task_id,
            "question": question,
            "status": status,
            "agent_steps": agent_steps,
            "assistant_answer": assistant_answer,
            "memory_before": previous_memory,
            "memory_output": memory_output,
            "r_tool_steps": r_tool_steps,
            "r_answer": r_answer,
            "r_memory": r_memory,
            "coverage_final": ledger.coverage if tes is not None else 0.0,
            "n_tool_steps": len(agent_steps),
        }


# ======================================================================
# Helpers
# ======================================================================

def _index_samples_by_task(samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index samples by task field."""
    idx: Dict[str, Dict[str, Any]] = {}
    for s in samples:
        key = get_sample_id(s)
        if key:
            idx[key] = s
    return idx


# DEFAULT_SYSTEM_TEMPLATE is imported from TEMPO_RL.io_utils (shared)


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TEMPO-RL Phase 3 — Sparse Counterfactual Memory RL"
    )
    parser.add_argument(
        "--dialog_rollouts", required=True,
        help="Path to phase2_dialog_rollouts.jsonl"
    )
    parser.add_argument(
        "--samples", required=True,
        help="Path to benchmark samples JSON file"
    )
    parser.add_argument(
        "--target_evidence", required=True,
        help="Path to target_evidence.jsonl"
    )
    parser.add_argument(
        "--future_dependencies", required=True,
        help="Path to future_dependencies.jsonl"
    )
    parser.add_argument(
        "--table_root", default="",
        help="Root directory for table files"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for phase3_counterfactual_audit.jsonl"
    )
    parser.add_argument(
        "--lambda_cf", type=float, default=0.2,
        help="Counterfactual delta weight (default 0.2)"
    )
    parser.add_argument(
        "--a", type=float, default=1.0,
        help="Clipping bound for ΔU_i (default 1.0)"
    )
    parser.add_argument(
        "--tau_f", type=float, default=0.8,
        help="Faithfulness gate threshold (default 0.8)"
    )
    parser.add_argument(
        "--sparse_rate", type=float, default=0.25,
        help="Fraction of eligible boundaries to execute (default 0.25)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sparse sampling"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="Temperature for counterfactual continuations (low)"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use MockPolicy for testing"
    )
    args = parser.parse_args()

    # --- Load inputs ---
    dialog_rollouts = read_jsonl(args.dialog_rollouts)
    samples = load_json_file(args.samples)
    if not isinstance(samples, list):
        samples = [samples]

    print(f"[Phase3] Loaded {len(dialog_rollouts)} dialog rollouts, "
          f"{len(samples)} samples")

    # --- Load TES ---
    tes_records = read_jsonl(args.target_evidence)
    tes_lookup: Dict[Tuple[str, int], TargetEvidenceSet] = {}
    for rec in tes_records:
        sid = get_sample_id(rec)
        sq = rec.get("subquestion_id", rec.get("sq_id", 0))
        if isinstance(sq, str):
            try:
                sq = int(sq)
            except ValueError:
                sq = 0
        tes_lookup[(sid, sq)] = TargetEvidenceSet.from_dict(rec)

    # --- Load FDS ---
    fds_records = read_jsonl(args.future_dependencies)
    fds_lookup: Dict[Tuple[str, str], FutureDependencySet] = {}
    for rec in fds_records:
        sid = get_sample_id(rec)
        boundary = rec.get("boundary", "")
        fds_lookup[(sid, boundary)] = FutureDependencySet.from_dict(rec)

    print(f"[Phase3] Loaded {len(tes_lookup)} TES, {len(fds_lookup)} FDS")

    # --- Init components ---
    if args.mock:
        policy = MockPolicy(["<answer>Mock.</answer><memory>{}</memory>"] * 100)
        tool_executor = FakeToolExecutor()
        print("[Phase3] Using MockPolicy + FakeToolExecutor")
    else:
        from TEMPO_RL.rollout_phase1 import ChatClientPolicy
        from src.utils.chat_api import ChatClient
        chat_client = ChatClient(provider="openai", config_key="mimo")
        policy = ChatClientPolicy(chat_client)
        tool_executor = ToolExecutor(table_root=args.table_root)
        print("[Phase3] Using ChatClientPolicy (real LLM)")

    calculator = RewardCalculator()

    # --- Build tools schema ---
    try:
        from src.tools.base import get_tools_schema
        tools_schema = get_tools_schema()
    except Exception:
        tools_schema = []

    # --- Run ---
    estimator = CounterfactualEstimator(
        policy=policy,
        tool_executor=tool_executor,
        calculator=calculator,
        tes_lookup=tes_lookup,
        fds_lookup=fds_lookup,
        lambda_cf=args.lambda_cf,
        a=args.a,
        tau_f=args.tau_f,
        sparse_rate=args.sparse_rate,
        temperature=args.temperature,
        seed=args.seed,
    )

    t_start = time.time()
    audit_records, summary = estimator.estimate_and_save(
        dialog_rollouts=dialog_rollouts,
        samples=samples,
        tools_schema=tools_schema,
        output_dir=args.output_dir,
    )
    elapsed = time.time() - t_start

    print(f"\n[Phase3] Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  elapsed: {elapsed:.1f}s")

    if audit_records:
        print(f"\n[Phase3] Sample audit record:")
        sample = audit_records[0]
        for k in ("sample_id", "boundary", "j_star", "r_ans_gen", "r_ans_prev",
                   "delta_u", "clipped_delta_u", "faithfulness_gate",
                   "r_mem_original", "r_mem_final", "cf_contribution"):
            print(f"  {k}: {sample.get(k)}")


if __name__ == "__main__":
    main()
