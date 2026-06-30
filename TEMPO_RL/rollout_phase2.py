"""
TEMPO-RL Phase 2 — Dialog-level Rollout Runner.

Runs K complete dialog rollouts per sample. Each dialog processes all
subquestions in sequence::

    q1 -> tools -> answer1 -> memory1
    q2 uses memory1 -> tools -> answer2 -> memory2
    ...

The model generates its own memory after each answer, and that raw memory
is passed verbatim to the next subquestion (no verifier correction).

Usage::

    python -m TEMPO_RL.rollout_phase2 \\
        --samples dataset/train不含val的.json \\
        --target_evidence output/target_evidence.jsonl \\
        --future_dependencies output/future_dependencies.jsonl \\
        --table_root dataset/table \\
        --output_dir phase2_output \\
        --max_samples 2 --K 2 --max_tool_steps_per_turn 6
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from TEMPO_RL.schemas import TargetEvidenceSet, FutureDependencySet
from TEMPO_RL.evidence_ledger import EvidenceLedger
from TEMPO_RL.reward_calculator import RewardCalculator
from TEMPO_RL.build_target_evidence import TargetEvidenceBuilder

from src.tools.base import get_tools_schema, execute_tool, TOOL_REGISTRY
from TEMPO_RL.io_utils import (
    read_jsonl,
    write_jsonl,
    load_json_file,
    try_parse_json,
    extract_tool_calls_from_response,
    extract_answer_from_response,
    extract_memory_from_response,
    count_tool_calls,
    get_sample_id,
    DEFAULT_SYSTEM_TEMPLATE,
)


# ======================================================================
# Policy Interface
# ======================================================================

class PolicyWrapper:
    """Abstract policy interface."""

    def call(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class LocalModelPolicy(PolicyWrapper):
    """Policy backed by a local HuggingFace model (e.g. Phase 1 checkpoint)."""

    def __init__(self, model_path: str, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[LocalModelPolicy] Loading model from {model_path} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        self._model.to(device)
        self._model.eval()
        self._device = device
        print("[LocalModelPolicy] Model loaded.")

    def call(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        import torch

        if hasattr(self._tokenizer, "chat_template") and self._tokenizer.chat_template:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tools=tools if tools else None,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(
                f"{m['role']}: {m['content']}" for m in messages
            )

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = outputs[0, inputs.input_ids.shape[1]:]
        content = self._tokenizer.decode(generated, skip_special_tokens=True)

        return {"content": content, "finish_reason": "stop"}


# ======================================================================
# Tool Executor
# ======================================================================

class ToolExecutor:
    """Executes real tools and returns observations."""

    def __init__(self, table_root: str = ""):
        self._table_root = os.path.abspath(table_root) if table_root else ""

    @staticmethod
    def _is_within_root(path: str, root: str) -> bool:
        """Check whether *path* is inside *root* (or equals it)."""
        if not root:
            return True
        try:
            resolved = os.path.abspath(path)
            return resolved == root or resolved.startswith(root + os.sep)
        except (ValueError, TypeError):
            return False

    def _sanitize_path(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Force any path-like argument to stay within ``_table_root``."""
        if not self._table_root:
            return arguments
        for key in ("folder_path", "file_path", "path", "table_path", "directory"):
            val = arguments.get(key)
            if isinstance(val, str) and val.strip():
                if not self._is_within_root(val, self._table_root):
                    # Try joining with table_root — if the model passed a relative
                    # sub-path it may still be valid after joining
                    fixed = os.path.normpath(os.path.join(self._table_root, val.lstrip(os.sep)))
                    if self._is_within_root(fixed, self._table_root):
                        arguments = dict(arguments)
                        arguments[key] = fixed
                    else:
                        # Still outside — force back to table_root
                        arguments = dict(arguments)
                        arguments[key] = self._table_root
        return arguments

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        arguments = self._sanitize_path(arguments)
        try:
            result = execute_tool(tool_name, **arguments)
        except Exception as e:
            return {
                "tool_name": tool_name,
                "success": False,
                "content": f"[ERROR] Tool execution failed: {str(e)}",
            }

        success = result.success
        content = (
            f"[SUCCESS] {result.data}"
            if success
            else f"[ERROR] {result.message}"
        )
        # Truncate long tool results
        max_len = 4000
        if len(content) > max_len:
            content = (
                f"[Truncated, original length {len(content)}]\n"
                + content[-max_len:]
            )
        return {
            "tool_name": tool_name,
            "success": success,
            "content": content,
        }

    def extract_metadata(
        self, tool_name: str, arguments: Dict[str, Any], observation: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        meta: Dict[str, Any] = {}
        for key in ("file_path", "table_name", "table_path", "path", "filename", "file"):
            if key in arguments:
                val = arguments[key]
                if isinstance(val, str):
                    meta["file"] = val
                    meta["table_name"] = os.path.splitext(os.path.basename(val))[0]
                    break
        if not meta:
            content = observation.get("content", "")
            for line in content.split("\n"):
                if "file" in line.lower() or "table" in line.lower():
                    meta["source_hint"] = line.strip()[:200]
                    break
        return meta if meta else None


# ======================================================================
# Dialog Rollout Runner
# ======================================================================

class DialogRolloutRunner:
    """Run K complete dialog rollouts per sample.

    Each dialog processes all subquestions sequentially. After each
    subquestion's answer, the model generates memory that is passed
    verbatim to the next subquestion.

    Parameters
    ----------
    policy : PolicyWrapper
    tool_executor : ToolExecutor
    calculator : RewardCalculator
    tes_lookup : dict
        Mapping ``(sample_id, subquestion_index) -> TargetEvidenceSet``.
    fds_lookup : dict
        Mapping ``(sample_id, boundary) -> FutureDependencySet``.
    K : int = 2
        Number of dialog rollouts per sample.
    temperature : float = 0.7
    top_p : float = 0.9
    max_tool_steps_per_turn : int = 6
        Max tool-call turns per subquestion.
    """

    def __init__(
        self,
        policy: PolicyWrapper,
        tool_executor: ToolExecutor,
        calculator: RewardCalculator,
        tes_lookup: Dict[Tuple[str, int], TargetEvidenceSet],
        fds_lookup: Optional[Dict[Tuple[str, str], FutureDependencySet]] = None,
        K: int = 2,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tool_steps_per_turn: int = 6,
    ):
        self.policy = policy
        self.tool_executor = tool_executor
        self.calculator = calculator
        self.tes_lookup = tes_lookup
        self.fds_lookup = fds_lookup or {}
        self.K = K
        self.temperature = temperature
        self.top_p = top_p
        self.max_tool_steps_per_turn = max_tool_steps_per_turn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        samples: List[Dict[str, Any]],
        tools_schema: List[Dict[str, Any]],
        output_dir: str,
        max_samples: Optional[int] = None,
        system_template: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run dialog rollouts and write ``phase2_dialog_rollouts.jsonl``."""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "phase2_dialog_rollouts.jsonl")
        all_records: List[Dict[str, Any]] = []

        samples_to_run = samples[:max_samples] if max_samples else samples
        n_samples = len(samples_to_run)

        print(f"[Phase2] Running {n_samples} samples x K={self.K} dialog rollouts ...")
        t_start = time.time()

        for s_idx, sample in enumerate(samples_to_run):
            sample_id = get_sample_id(sample, s_idx)
            subquestions = sample.get("design", {}).get("checkout_list", sample.get("subquestions", []))

            if not subquestions:
                print(f"  [{s_idx+1}/{n_samples}] {sample_id}: no subquestions, skipping")
                continue

            for k in range(self.K):
                rollout_id = f"{sample_id}_k{k}"
                try:
                    record = self._run_dialog_rollout(
                        sample_id=sample_id,
                        rollout_id=rollout_id,
                        sample=sample,
                        subquestions=subquestions,
                        tools_schema=tools_schema,
                        system_template=system_template,
                    )
                    all_records.append(record)
                    write_jsonl(output_path, all_records)

                    n_sq = record.get("n_subquestions", 0)
                    statuses = [sq.get("status", "?") for sq in record.get("subquestion_rollouts", [])]
                    print(f"  [{s_idx+1}/{n_samples}] {rollout_id}: {n_sq} subquestions, "
                          f"statuses={statuses}")
                except Exception as e:
                    print(f"  [{s_idx+1}/{n_samples}] {rollout_id}: ERROR — {e}")
                    import traceback
                    traceback.print_exc()
                    all_records.append({
                        "sample_id": sample_id,
                        "rollout_id": rollout_id,
                        "error": str(e),
                    })

        elapsed = time.time() - t_start
        print(f"[Phase2] Done. {len(all_records)} dialog rollouts in {elapsed:.1f}s")
        print(f"  Output: {output_path}")
        return all_records

    # ------------------------------------------------------------------
    # Dialog rollout
    # ------------------------------------------------------------------

    def _run_dialog_rollout(
        self,
        sample_id: str,
        rollout_id: str,
        sample: Dict[str, Any],
        subquestions: List[Dict[str, Any]],
        tools_schema: List[Dict[str, Any]],
        system_template: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run one complete dialog rollout: process all subquestions in sequence.

        Memory flows: after each subquestion, the model's raw memory output
        is passed to the next subquestion. Verifier does NOT correct memory.
        """
        subquestion_rollouts: List[Dict[str, Any]] = []
        previous_raw_memory: Optional[Dict[str, Any]] = None
        dialog_messages: List[Dict[str, Any]] = []  # Accumulated for logging

        # Get table path from sample
        table_path = sample.get("table_path", sample.get("table_root", ""))

        print(f"  [{rollout_id}] {len(subquestions)} subquestions", flush=True)

        n_sq = len(subquestions)
        for sq_idx, sq in enumerate(subquestions):
            sq_id = sq_idx + 1  # 1-indexed

            question = sq.get("question", sq.get("cq", ""))
            if not question:
                question = sq.get("checkout_item", {}).get("checkout_text", "")

            score_points = sq.get("score_points", sq.get("checkout_item", {}).get("score_points", []))

            # Resolve score_points format
            if isinstance(score_points, str):
                try:
                    score_points = json.loads(score_points)
                except (json.JSONDecodeError, TypeError):
                    score_points = [score_points]
            if not isinstance(score_points, list):
                score_points = []

            # Look up TargetEvidenceSet
            tes = self.tes_lookup.get((sample_id, sq_id))
            if tes is None:
                tes = self.tes_lookup.get((sample_id, sq_idx))

            # Look up FutureDependencySet for boundary
            boundary_key = f"after_sq{sq_id - 1}" if sq_idx > 0 else "root"
            fds = self.fds_lookup.get((sample_id, boundary_key))

            print(f"    [sq {sq_id}/{n_sq}] {question[:60]}...", flush=True)

            # Run subquestion rollout
            sq_rollout = self._run_subquestion_rollout(
                sample_id=sample_id,
                rollout_id=rollout_id,
                sq_id=sq_id,
                question=question,
                score_points=score_points,
                tes=tes,
                fds=fds,
                table_path=table_path,
                tools_schema=tools_schema,
                previous_memory=previous_raw_memory,
                system_template=system_template,
            )

            subquestion_rollouts.append(sq_rollout)

            # Extract memory for next subquestion
            mem_output = sq_rollout.get("memory_output")
            if mem_output is not None:
                # Parse to check if it's valid JSON
                parsed = try_parse_json(json.dumps(mem_output, ensure_ascii=False)
                                         if isinstance(mem_output, dict)
                                         else str(mem_output))
                if parsed is not None:
                    previous_raw_memory = parsed
                    sq_rollout["memory_passed"] = parsed
                    sq_rollout["memory_fallback_used"] = False
                else:
                    # Unparseable → keep previous memory as fallback
                    sq_rollout["memory_passed"] = previous_raw_memory
                    sq_rollout["memory_fallback_used"] = True
                    sq_rollout["memory_parse_error"] = True
            else:
                # No memory generated → keep previous
                sq_rollout["memory_passed"] = previous_raw_memory
                sq_rollout["memory_fallback_used"] = previous_raw_memory is not None

            dialog_messages.append({
                "sq_id": sq_id,
                "question": question,
                "memory_before": sq_rollout.get("memory_before"),
                "memory_after": sq_rollout.get("memory_output"),
                "memory_passed": sq_rollout.get("memory_passed"),
            })

        return {
            "sample_id": sample_id,
            "rollout_id": rollout_id,
            "n_subquestions": len(subquestions),
            "subquestion_rollouts": subquestion_rollouts,
            "dialog_messages": dialog_messages,
        }

    # ------------------------------------------------------------------
    # Single subquestion rollout
    # ------------------------------------------------------------------

    def _run_subquestion_rollout(
        self,
        sample_id: str,
        rollout_id: str,
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
        """Run one subquestion within a dialog.

        Includes tool steps, answer extraction, memory generation,
        and reward computation.
        """
        task_id = f"{sample_id}_sq{sq_id}"

        # --- Build system prompt with optional previous memory ---
        system_msg = (system_template or DEFAULT_SYSTEM_TEMPLATE).format(
            table_path=table_path,
        )

        # Inject previous memory if available
        memory_before = previous_memory
        memory_hint = ""
        if previous_memory is not None:
            memory_str = json.dumps(previous_memory, ensure_ascii=False, indent=2)
            memory_hint = (
                f"\n\n**Memory from previous subquestion (use this information!):**\n"
                f"```json\n{memory_str}\n```"
            )

        # --- Init ledger ---
        if tes is not None:
            ledger = EvidenceLedger(tes)
            ledger.initialize_from_memory(memory_before or {})
        else:
            # No TES for this subquestion — create empty ledger
            ledger = EvidenceLedger(
                TargetEvidenceSet(
                    sample_id=sample_id,
                    subquestion_id=sq_id,
                    question=question,
                )
            )
            ledger.initialize_from_memory({})

        # --- Build conversation ---
        full_question = question + memory_hint
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": full_question},
        ]

        # --- Per-step tracking ---
        agent_steps: List[Dict[str, Any]] = []
        ledger_trace: List[Dict[str, Any]] = [ledger.to_dict()]
        tool_calls_list: List[Dict[str, Any]] = []
        ledger_updates: List[Dict[str, Any]] = []
        observations_list: List[Dict[str, Any]] = []
        code_outputs_list: List[str] = []
        r_tool_steps: List[float] = []

        assistant_answer: Optional[Dict[str, Any]] = None
        memory_output: Optional[Dict[str, Any]] = None
        status = "in_progress"

        try:
            for step_idx in range(self.max_tool_steps_per_turn):
                print(f"      [step {step_idx+1}/{self.max_tool_steps_per_turn}] calling LLM...", flush=True)
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

                # --- Case 1: Answer with no tool calls → done with tools ---
                if has_answer and not has_tools:
                    assistant_answer = {"content": answer_text, "full_response": content}
                    mem = extract_memory_from_response(content)
                    if mem is not None:
                        memory_output = mem
                    status = "completed"
                    break

                # --- Case 2: No tool calls and no answer → error ---
                if not has_tools and not has_answer:
                    status = "no_tool_or_answer"
                    break

                # --- Save answer as fallback when both answer + tool_call present ---
                if has_answer and assistant_answer is None:
                    assistant_answer = {"content": answer_text, "full_response": content}

                # --- Case 3: Multiple tool calls → invalid ---
                if len(tool_calls_in_turn) > 1:
                    status = "invalid_multi_tool"
                    agent_steps.append({
                        "step_index": step_idx,
                        "type": "tool_call",
                        "tool_calls": tool_calls_in_turn,
                        "observations": [],
                        "invalid_multi_tool": True,
                        "response_text": content,
                    })
                    fake_update = {
                        "coverage_before": ledger.coverage,
                        "coverage_after": ledger.coverage,
                        "new_evidence_ids": [],
                        "ledger": ledger.to_dict(),
                        "audit": {"error": "multi_tool_call_invalid"},
                    }
                    ledger_updates.append(fake_update)
                    tool_calls_list.append(tool_calls_in_turn[0] if tool_calls_in_turn else {})
                    observations_list.append({})
                    code_outputs_list.append("")

                    is_invalid = True
                    is_repeat = False
                    r_tool = -(self.calculator.lambda_invalid)
                    r_tool_steps.append(r_tool)

                    # Error feedback to model
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": "[ERROR] Multiple tool calls detected. "
                                   "Only ONE tool call per turn is allowed. "
                                   "Please retry with a single tool call.",
                    })
                    ledger_trace.append(ledger.to_dict())
                    continue

                # --- Case 4: Single tool call → execute ---
                tool_call = tool_calls_in_turn[0]
                tool_name = tool_call.get("tool_name", "")
                arguments = tool_call.get("arguments", {})

                coverage_before = ledger.coverage

                # Execute tool
                obs = self.tool_executor.execute(tool_name, arguments)
                observations_list.append(obs)

                # Extract metadata and code output
                metadata = self.tool_executor.extract_metadata(tool_name, arguments, obs)
                code_output = obs.get("content", "") if tool_name in (
                    "python_code_executor", "code_executor"
                ) else ""
                code_outputs_list.append(code_output)

                # Update ledger
                ledger_result = ledger.update(
                    tool_call=tool_call,
                    observation=obs,
                    observation_metadata=metadata,
                    code_output=code_output,
                )
                ledger_updates.append(ledger_result)
                tool_calls_list.append(tool_call)

                # Compute tool reward
                tool_reward_result = self.calculator.compute_tool_reward(
                    tool_call=tool_call,
                    ledger_update_result=ledger_result,
                    subquestion_id=f"sq{sq_id}",
                )
                r_tool = tool_reward_result["r_tool"]
                r_tool_steps.append(r_tool)

                # Record step
                agent_steps.append({
                    "step_index": step_idx,
                    "type": "tool_call",
                    "tool_calls": tool_calls_in_turn,
                    "observations": [obs],
                    "invalid_multi_tool": False,
                    "response_text": content,
                    "tool_reward": tool_reward_result,
                    "ledger_result": ledger_result,
                })

                # Append to conversation
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

                ledger_trace.append(ledger.to_dict())

                # If answer was also present with tool calls, extract it
                if has_answer:
                    assistant_answer = {"content": answer_text, "full_response": content}
                    mem = extract_memory_from_response(content)
                    if mem is not None:
                        memory_output = mem
                    status = "completed"
                    break

            # --- After tool loop: check if truncated ---
            if status == "in_progress":
                status = "truncated"
                # Try one more call to get answer + memory
                response = self.policy.call(
                    messages=messages,
                    tools=tools_schema,
                    temperature=self.temperature,
                    top_p=self.top_p,
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
            import traceback
            traceback.print_exc()

        # --- Compute answer reward ---
        r_answer = 0.0
        answer_reward_result: Dict[str, Any] = {}
        if assistant_answer is not None and tes is not None:
            answer_reward_result = self.calculator.compute_answer_reward(
                answer_json=assistant_answer.get("content", ""),
                score_points=score_points,
                ledger=ledger,
                memory_before=memory_before,
                observations=observations_list,
                code_outputs=code_outputs_list,
            )
            r_answer = answer_reward_result.get("r_answer", 0.0)

        # --- Compute memory reward ---
        r_memory = 0.0
        memory_reward_result: Dict[str, Any] = {}
        if memory_output is not None:
            memory_reward_result = self.calculator.compute_memory_reward(
                memory_after=memory_output,
                memory_before=memory_before,
                ledger=ledger,
                observations=observations_list,
                code_outputs=code_outputs_list,
                future_dependency_set=fds,
                grounded_answer_claims=answer_reward_result.get("claim_results"),
            )
            r_memory = memory_reward_result.get("r_memory", 0.0)

        # --- Detect severe memory failure ---
        memory_severe_failure = memory_reward_result.get("severe_failure", False)
        memory_failure_reason = memory_reward_result.get("failure_reason", "")

        if memory_severe_failure:
            # r_memory should already be -1 from the calculator
            r_memory = -1.0

        return {
            "sample_id": sample_id,
            "rollout_id": rollout_id,
            "sq_id": sq_id,
            "task_id": task_id,
            "question": question,
            "score_points": score_points,
            "status": status,
            "agent_steps": agent_steps,
            "assistant_answer": assistant_answer,
            "memory_before": memory_before,
            "memory_output": memory_output,
            "memory_severe_failure": memory_severe_failure,
            "memory_failure_reason": memory_failure_reason,
            "r_tool_steps": r_tool_steps,
            "r_answer": r_answer,
            "r_memory": r_memory,
            "memory_reward_detail": memory_reward_result,
            "answer_reward_detail": answer_reward_result,
            "ledger_trace": ledger_trace,
            "ledger_final": ledger.to_dict() if tes is not None else {},
            "coverage_final": ledger.coverage if tes is not None else 0.0,
            "n_tool_steps": len(agent_steps),
        }


# ======================================================================
# System Template
# ======================================================================

# DEFAULT_SYSTEM_TEMPLATE is imported from TEMPO_RL.io_utils (shared)


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TEMPO-RL Phase 2 — Dialog Rollout Runner"
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
        "--future_dependencies", default="",
        help="Path to future_dependencies.jsonl"
    )
    parser.add_argument(
        "--table_root", default="",
        help="Root directory for table files"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for phase2_dialog_rollouts.jsonl"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit number of samples to process"
    )
    parser.add_argument(
        "--K", type=int, default=2,
        help="Number of dialog rollouts per sample (default 2)"
    )
    parser.add_argument(
        "--max_tool_steps_per_turn", type=int, default=6,
        help="Max tool calls per subquestion (default 6)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9,
    )
    parser.add_argument(
        "--eta", type=float, default=1.0,
        help="Tool evidence gain scaling"
    )
    parser.add_argument(
        "--alpha_f", type=float, default=0.5,
        help="Memory faithfulness weight"
    )
    parser.add_argument(
        "--alpha_s", type=float, default=0.4,
        help="Future dependency coverage weight"
    )
    parser.add_argument(
        "--lambda_comp", type=float, default=0.1,
        help="Compression penalty weight"
    )
    parser.add_argument(
        "--B", type=int, default=512,
        help="Memory budget in tokens"
    )
    parser.add_argument(
        "--provider", type=str, default="openai",
        help="LLM provider (openai, vllm, azure)"
    )
    parser.add_argument(
        "--config_key", type=str, default="mimo",
        help="Config key in api_key.json (e.g. mimo, qwen3-8b, sft-local)"
    )
    parser.add_argument(
        "--model_path", type=str, default="",
        help="Path to local HF model/checkpoint (overrides provider/config_key)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for local model (default cuda, or cuda:0, cuda:2, etc.)"
    )
    args = parser.parse_args()

    # --- Load inputs ---
    samples = load_json_file(args.samples)
    if not isinstance(samples, list):
        samples = [samples]

    print(f"[Phase2] Loaded {len(samples)} samples")

    # --- Build / load TargetEvidence ---
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

    print(f"[Phase2] Loaded {len(tes_lookup)} target evidence sets")

    # --- Load FutureDependencies ---
    fds_lookup: Dict[Tuple[str, str], FutureDependencySet] = {}
    if args.future_dependencies:
        fds_records = read_jsonl(args.future_dependencies)
        for rec in fds_records:
            sid = get_sample_id(rec)
            boundary = rec.get("boundary", "")
            fds_lookup[(sid, boundary)] = FutureDependencySet.from_dict(rec)
        print(f"[Phase2] Loaded {len(fds_lookup)} future dependency sets")

    # --- Init components ---
    if args.model_path:
        policy = LocalModelPolicy(args.model_path, device=args.device)
        print(f"[Phase2] Using LocalModelPolicy ({args.model_path}, {args.device})")
    else:
        from TEMPO_RL.rollout_phase1 import ChatClientPolicy
        from src.utils.chat_api import ChatClient
        chat_client = ChatClient(provider=args.provider, config_key=args.config_key)
        policy = ChatClientPolicy(chat_client)
        print("[Phase2] Using ChatClientPolicy (real LLM)")

    tool_executor = ToolExecutor(table_root=args.table_root)
    tool_registry = set(TOOL_REGISTRY.keys()) if TOOL_REGISTRY else None

    calculator = RewardCalculator(
        eta=args.eta,
        alpha_f=args.alpha_f,
        alpha_s=args.alpha_s,
        lambda_comp=args.lambda_comp,
        B=args.B,
        tool_registry=tool_registry,
    )

    # --- Build tools schema ---
    try:
        tools_schema = get_tools_schema()
    except Exception:
        tools_schema = []

    # --- Run ---
    runner = DialogRolloutRunner(
        policy=policy,
        tool_executor=tool_executor,
        calculator=calculator,
        tes_lookup=tes_lookup,
        fds_lookup=fds_lookup,
        K=args.K,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tool_steps_per_turn=args.max_tool_steps_per_turn,
    )

    runner.run(
        samples=samples,
        tools_schema=tools_schema,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
