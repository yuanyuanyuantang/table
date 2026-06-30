"""
TEMPO-RL Phase 1 — Rollout Runner.

Samples K trajectories per subquestion from a policy model, executes real
tools, updates the evidence ledger, and computes tool/answer rewards.
No training — just data collection.

Usage::

    python -m TEMPO_RL.rollout_phase1 \\
        --samples dataset/train不含val的.json \\
        --target_evidence output/target_evidence.jsonl \\
        --table_root dataset/table \\
        --output_dir phase1_output \\
        --max_samples 2 --K 4 --max_tool_steps 8
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

# Import Phase 0 components
from TEMPO_RL.schemas import TargetEvidenceSet, FutureDependencySet
from TEMPO_RL.evidence_ledger import EvidenceLedger
from TEMPO_RL.reward_calculator import RewardCalculator
from TEMPO_RL.build_target_evidence import TargetEvidenceBuilder
from TEMPO_RL.io_utils import (
    read_jsonl,
    write_jsonl,
    load_json_file,
    extract_tool_calls_from_response,
    extract_answer_from_response,
    count_tool_calls,
    get_sample_id,
    DEFAULT_SYSTEM_TEMPLATE_PHASE1,
)

# Import tool infrastructure
from src.tools.base import get_tools_schema, execute_tool, TOOL_REGISTRY


def _index_by_task(samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for s in samples:
        task = s.get("task", "")
        if task:
            idx[task] = s
    return idx


# ======================================================================
# Policy Wrappers
# ======================================================================

class PolicyWrapper:
    """Abstract policy interface.

    Subclasses implement ``call()`` to return model responses.
    """

    def call(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """Call the policy model.

        Returns a dict with:
          - ``content``: raw text response
          - ``finish_reason``: e.g. "stop", "tool_calls"
        """
        raise NotImplementedError


class MockPolicy(PolicyWrapper):
    """Mock policy that returns pre-configured responses in order.

    Used for smoke testing.

    Parameters
    ----------
    responses : list of str
        Raw response strings to return on each call, in order.
        Each response can contain ``<tool_call>`` or ``<answer>`` tags.
    """

    def __init__(self, responses: Optional[List[str]] = None):
        self._responses = responses or []
        self._index = 0
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def set_responses(self, responses: List[str]) -> None:
        self._responses = list(responses)
        self._index = 0

    def call(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        self._call_count += 1
        if self._index < len(self._responses):
            content = self._responses[self._index]
            self._index += 1
        else:
            # Default: return answer if exhausted
            content = "<answer>Fallback answer.</answer>"
        return {"content": content, "finish_reason": "stop"}


class ChatClientPolicy(PolicyWrapper):
    """Policy backed by a real LLM via ``ChatClient``.

    Parameters
    ----------
    chat_client : ChatClient
        The LLM client.
    enable_thinking : bool
        Whether to enable thinking mode.
    """

    def __init__(self, chat_client: Any, enable_thinking: bool = True):
        self._client = chat_client
        self._enable_thinking = enable_thinking

    def call(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        response = self._client.chat(
            message=messages,
            tools=tools,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=self._enable_thinking,
        )

        # Build normalized content (like TableAgent._get_llm_response)
        content = response.get("content", "")
        reasoning = response.get("reasoning_content", "")

        raw = ""
        if reasoning:
            raw += f"<think>\n{reasoning}\n</think>\n\n"
        raw += content or ""

        # If the model returned native tool_calls but not in content,
        # inject them as <tool_call> tags
        native_tool_calls = response.get("tool_calls") or []
        if native_tool_calls and "<tool_call>" not in (content or ""):
            raw += "\n\n"
            for tc in native_tool_calls:
                fn = tc.get("function", tc)
                info = json.dumps(
                    {
                        "tool": fn.get("name", ""),
                        "params": json.loads(fn.get("arguments", "{}"))
                        if isinstance(fn.get("arguments"), str)
                        else fn.get("arguments", {}),
                        "call_id": tc.get("id", ""),
                    },
                    ensure_ascii=False,
                )
                raw += f"<tool_call>{info}</tool_call>\n"

        return {
            "content": raw,
            "finish_reason": response.get("finish_reason", "stop"),
        }


# ======================================================================
# Tool Executor
# ======================================================================

class ToolExecutor:
    """Execute tools and return normalized observation dicts.

    Mirrors the format used in SFT trajectories so ledger update works.
    """

    def __init__(self, table_root: str = ""):
        self._table_root = table_root
        self._step_counter = 0

    def execute(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single tool call.

        Returns an observation dict with ``tool_call_id``, ``tool_name``,
        ``content``, ``success``.
        """
        self._step_counter += 1

        try:
            result = execute_tool(tool_name, **arguments)
            success = result.success
            content = (
                f"[SUCCESS] {result.data}"
                if success
                else f"[ERROR] {result.message}"
            )
        except Exception as e:
            success = False
            content = f"[ERROR] Tool execution exception: {str(e)}"

        # Truncate long tool results
        max_len = 4000
        if len(content) > max_len:
            content = (
                f"[Truncated, original length {len(content)}]\n"
                + content[-max_len:]
            )

        return {
            "tool_call_id": f"tc_{self._step_counter}",
            "tool_name": tool_name,
            "content": content,
            "success": success,
        }

    def extract_metadata(
        self, tool_name: str, arguments: Dict[str, Any], observation: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract observation metadata for ledger verification.

        Tries to find file/table name from tool arguments or observation content.
        """
        metadata: Dict[str, Any] = {}

        # From arguments
        for k in ("file_path", "path", "table", "table_name", "filename", "file"):
            if k in arguments:
                metadata["file"] = arguments[k]
                return metadata

        # From observation content
        content = observation.get("content", "")
        if isinstance(content, str):
            m = re.search(r'(\S+\.(?:xlsx|csv|xls))', content)
            if m:
                metadata["file"] = m.group(1)

        return metadata if metadata else None


# ======================================================================
# Rollout Runner
# ======================================================================

class RolloutRunner:
    """Run K rollouts per subquestion and collect reward data.

    Parameters
    ----------
    policy : PolicyWrapper
        The policy to sample trajectories from.
    tool_executor : ToolExecutor
        Tool execution backend.
    calculator : RewardCalculator
        Phase 0 reward calculator.
    tes_lookup : dict mapping ``(sample_id, subquestion_id)`` → TargetEvidenceSet
        Pre-built target evidence.
    K : int
        Number of rollouts per subquestion. Default 4.
    temperature : float
        Sampling temperature. Default 0.7.
    top_p : float
        Nucleus sampling top-p. Default 0.9.
    max_tool_steps : int
        Maximum tool calls before truncation. Default 8.
    """

    def __init__(
        self,
        policy: PolicyWrapper,
        tool_executor: ToolExecutor,
        calculator: RewardCalculator,
        tes_lookup: Dict[Tuple[str, int], TargetEvidenceSet],
        K: int = 4,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tool_steps: int = 8,
    ):
        self.policy = policy
        self.tool_executor = tool_executor
        self.calculator = calculator
        self._tes_lookup = tes_lookup
        self.K = K
        self.temperature = temperature
        self.top_p = top_p
        self.max_tool_steps = max_tool_steps

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
        """Run all rollouts and write ``phase1_rollouts.jsonl``.

        Returns the list of rollout records.
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "phase1_rollouts.jsonl")
        all_records: List[Dict[str, Any]] = []

        task_index = _index_by_task(samples)
        processed = 0

        for sample in samples:
            task_id = get_sample_id(sample)
            if not task_id:
                continue

            checkout_list = sample.get("design", {}).get("checkout_list", [])
            if not checkout_list:
                continue

            for cq in checkout_list:
                sq_id = cq.get("idx", 1)  # idx is 1-based in benchmark data
                tes_key = (task_id, sq_id)
                if tes_key not in self._tes_lookup:
                    continue
                tes = self._tes_lookup[tes_key]

                question = cq.get("info_item", "")
                score_points = cq.get("score_points", [])
                table_path = os.path.join(
                    self.tool_executor._table_root,
                    os.path.relpath(
                        sample.get("file_path", ""), "dataset/table"
                    ),
                ) if self.tool_executor._table_root else ""

                # Reset call history for this subquestion
                self.calculator.reset_call_history(f"sq{sq_id}")

                for k in range(self.K):
                    rollout_id = f"{task_id[:30]}_sq{sq_id}_k{k}"
                    record = self._run_one_rollout(
                        task_id=task_id,
                        sq_id=sq_id,
                        rollout_id=rollout_id,
                        question=question,
                        score_points=score_points,
                        tes=tes,
                        table_path=table_path,
                        tools_schema=tools_schema,
                        system_template=system_template,
                    )
                    all_records.append(record)

                processed += 1
                if max_samples and processed >= max_samples:
                    break

            if max_samples and processed >= max_samples:
                break

        write_jsonl(output_path, all_records)
        print(
            f"[Phase 1] Wrote {len(all_records)} rollout records to {output_path}"
        )
        return all_records

    # ------------------------------------------------------------------
    # Single rollout
    # ------------------------------------------------------------------

    def _run_one_rollout(
        self,
        task_id: str,
        sq_id: int,
        rollout_id: str,
        question: str,
        score_points: List[str],
        tes: TargetEvidenceSet,
        table_path: str,
        tools_schema: List[Dict[str, Any]],
        system_template: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a single rollout for one (sample, subquestion) pair."""

        status = "completed"
        error_msg: Optional[str] = None

        # --- Build system prompt ---
        system_msg = (system_template or DEFAULT_SYSTEM_TEMPLATE_PHASE1).format(
            table_path=table_path,
        )

        # --- Init ledger ---
        ledger = EvidenceLedger(tes)
        # Phase 1: memory_before is empty for now (no memory system yet)
        ledger.initialize_from_memory({})

        # --- Conversation ---
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question},
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

        try:
            for step_idx in range(self.max_tool_steps):
                # Get model response
                response = self.policy.call(
                    messages=messages,
                    tools=tools_schema,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                content = response.get("content", "")

                # Check for answer
                answer_text = extract_answer_from_response(content)
                if answer_text is not None and extract_tool_calls_from_response(content):
                    # Contains both answer and tool_call → save answer as
                    # fallback in case the tool loop ends without a pure answer
                    if assistant_answer is None:
                        assistant_answer = {"content": answer_text}
                elif answer_text is not None:
                    # Pure answer → stop
                    assistant_answer = {"content": answer_text}
                    messages.append({"role": "assistant", "content": content})
                    break

                # Check for tool calls
                tool_calls_in_turn = extract_tool_calls_from_response(content)

                if len(tool_calls_in_turn) == 0:
                    # No tool call and no answer → error, stop
                    status = "error"
                    error_msg = "No tool call or answer found in response"
                    break

                if len(tool_calls_in_turn) > 1:
                    # Multiple tool calls → invalid, record penalty, skip execution
                    status = "invalid_multi_tool"
                    # Record as agent step with invalid marker
                    agent_steps.append({
                        "step_index": step_idx,
                        "type": "tool_call",
                        "tool_calls": tool_calls_in_turn,
                        "observations": [],
                        "invalid_multi_tool": True,
                        "response_text": content,
                    })
                    # Create a fake ledger update with zero evidence gain
                    fake_update = {
                        "coverage_before": ledger.coverage,
                        "coverage_after": ledger.coverage,
                        "new_evidence_ids": [],
                        "ledger": ledger.to_dict(),
                        "audit": {"error": "multi_tool_call_invalid"},
                    }
                    tool_calls_list.append(
                        {"tool_name": "__multi_tool_invalid__", "arguments": {}}
                    )
                    ledger_updates.append(fake_update)
                    r_tool_steps.append(
                        self.calculator.compute_tool_reward(
                            {"tool_name": "__multi_tool_invalid__", "arguments": {}},
                            fake_update,
                            f"sq{sq_id}",
                        )["r_tool"]
                    )
                    ledger_trace.append(ledger.to_dict())
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": "[ERROR] Multiple tool calls detected. "
                                   "Only ONE tool call per turn is allowed. "
                                   "Please re-issue a single tool call.",
                    })
                    continue  # Let model retry (counts toward max_tool_steps)

                # Exactly one tool call — execute it
                tc = tool_calls_in_turn[0]
                tool_name = tc["tool_name"]
                arguments = tc["arguments"]

                # Execute tool
                obs = self.tool_executor.execute(tool_name, arguments)

                # Extract code output
                code_output = ""
                if tool_name in ("python_code_executor", "python_exec", "calculator",
                                 "code_exec"):
                    code_output = obs.get("content", "")
                    if code_output:
                        code_outputs_list.append(code_output)

                # Extract metadata
                obs_metadata = self.tool_executor.extract_metadata(
                    tool_name, arguments, obs
                )

                # Update ledger
                update_result = ledger.update(
                    tool_call=tc,
                    observation=obs,
                    observation_metadata=obs_metadata,
                    code_output=code_output,
                )

                # Track
                agent_steps.append({
                    "step_index": step_idx,
                    "type": "tool_call",
                    "tool_calls": [tc],
                    "observations": [obs],
                    "invalid_multi_tool": False,
                    "response_text": content,
                })
                tool_calls_list.append(tc)
                ledger_updates.append(update_result)
                observations_list.append(obs)
                r_tool = self.calculator.compute_tool_reward(
                    tc, update_result, f"sq{sq_id}"
                )
                r_tool_steps.append(r_tool["r_tool"])
                ledger_trace.append(ledger.to_dict())

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

            else:
                # Exhausted max_tool_steps without answer
                status = "truncated"

        except Exception as e:
            status = "error"
            error_msg = str(e)

        # --- Compute answer reward ---
        r_answer = 0.0
        answer_audit: Dict[str, Any] = {}
        if assistant_answer is not None:
            ans_result = self.calculator.compute_answer_reward(
                answer_json=assistant_answer,
                score_points=score_points,
                ledger=ledger,
                memory_before={},
                observations=observations_list,
                code_outputs=code_outputs_list,
            )
            r_answer = ans_result["r_answer"]
            answer_audit = {
                "claims_total": len(score_points),
                "claims_correct_and_grounded": ans_result["audit"].get(
                    "claims_correct_and_grounded", 0
                ),
                "format_error": ans_result["format_error"],
                "unsupported_extra_count": ans_result["unsupported_extra_count"],
                "grounded_score": ans_result["audit"].get("grounded_score", 0.0),
                "claim_results": ans_result.get("claim_results", []),
            }

        avg_r_tool = (
            sum(r_tool_steps) / max(len(r_tool_steps), 1) if r_tool_steps else 0.0
        )

        return {
            "sample_id": task_id,
            "subquestion_id": sq_id,
            "rollout_id": rollout_id,
            "agent_steps": agent_steps,
            "assistant_answer": assistant_answer,
            "ledger_trace": ledger_trace,
            "r_tool_steps": r_tool_steps,
            "r_tool": avg_r_tool,
            "r_answer": r_answer,
            "answer_audit": answer_audit,
            "status": status,
            "error": error_msg,
            "num_tool_steps": len(agent_steps),
            "evidence_coverage": ledger.coverage,
            "total_evidence_items": len(tes.evidence_items),
            "verified_evidence_ids": sorted(ledger.verified_ids),
        }


# ======================================================================
# Default system prompt
# ======================================================================

# DEFAULT_SYSTEM_TEMPLATE is imported from TEMPO_RL.io_utils (shared)


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TEMPO-RL Phase 1 — Rollout Runner"
    )
    parser.add_argument(
        "--samples", required=True,
        help="Path to benchmark samples JSON file"
    )
    parser.add_argument(
        "--target_evidence", required=True,
        help="Path to target_evidence.jsonl (or will auto-build if --build_te)"
    )
    parser.add_argument(
        "--table_root", default="",
        help="Root directory for table files"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for phase1_rollouts.jsonl"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Max benchmark samples to process (for debugging)"
    )
    parser.add_argument(
        "--K", type=int, default=4,
        help="Number of rollouts per subquestion"
    )
    parser.add_argument(
        "--max_tool_steps", type=int, default=8,
        help="Maximum tool steps before truncation"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9,
        help="Nucleus sampling top-p"
    )
    parser.add_argument(
        "--build_te", action="store_true",
        help="Auto-build target evidence if target_evidence.jsonl doesn't exist"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock policy for testing (returns pre-canned responses)"
    )
    args = parser.parse_args()

    # --- Load samples ---
    print(f"[Phase 1] Loading samples from {args.samples} ...")
    benchmark = load_json_file(args.samples)
    if not isinstance(benchmark, list):
        benchmark = [benchmark]

    # --- LLM client (used for target evidence building and policy) ---
    try:
        from src.utils.chat_api import ChatClient
        llm_client = ChatClient(provider="openai", config_key="mimo")
        llm_enabled = True
    except Exception:
        llm_client = None
        llm_enabled = False

    # --- Load or build target evidence ---
    te_path = args.target_evidence
    if os.path.exists(te_path) and not args.build_te:
        print(f"[Phase 1] Loading target evidence from {te_path} ...")
        records = read_jsonl(te_path)
        all_tes = [TargetEvidenceSet.from_dict(r) for r in records]
    else:
        print(f"[Phase 1] Building target evidence ...")
        tes_builder = TargetEvidenceBuilder(llm_client=llm_client, llm_enabled=llm_enabled)
        all_tes = []
        for sample in benchmark:
            all_tes.extend(tes_builder.build_one_sample(sample))

    # Build TES lookup
    tes_lookup: Dict[Tuple[str, int], TargetEvidenceSet] = {}
    for tes in all_tes:
        tes_lookup[(tes.sample_id, tes.subquestion_id)] = tes
    print(f"  {len(tes_lookup)} target evidence sets loaded")

    # --- Tools schema ---
    tools_schema = get_tools_schema()

    # --- Policy ---
    if args.mock:
        policy = MockPolicy()
    elif llm_client is not None:
        policy = ChatClientPolicy(llm_client, enable_thinking=True)
    else:
        print("[Phase 1] ERROR: LLM client unavailable and --mock not set. Cannot create policy.")
        sys.exit(1)

    # --- Tool executor ---
    tool_executor = ToolExecutor(table_root=args.table_root)

    # --- Calculator ---
    calculator = RewardCalculator(tool_registry=set(TOOL_REGISTRY.keys()))

    # --- Runner ---
    runner = RolloutRunner(
        policy=policy,
        tool_executor=tool_executor,
        calculator=calculator,
        tes_lookup=tes_lookup,
        K=args.K,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tool_steps=args.max_tool_steps,
    )

    # --- Run ---
    records = runner.run(
        samples=benchmark,
        tools_schema=tools_schema,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )

    # --- Summary ---
    completed = sum(1 for r in records if r["status"] == "completed")
    truncated = sum(1 for r in records if r["status"] == "truncated")
    invalid = sum(1 for r in records if "invalid" in (r.get("status") or ""))
    errors = sum(1 for r in records if r["status"] == "error")

    print(f"\n[Phase 1] Summary:")
    print(f"  Total rollouts: {len(records)}")
    print(f"  Completed: {completed}")
    print(f"  Truncated: {truncated}")
    print(f"  Invalid (multi-tool): {invalid}")
    print(f"  Errors: {errors}")

    if records:
        avg_r_tool = sum(r["r_tool"] for r in records) / len(records)
        avg_r_answer = sum(r["r_answer"] for r in records) / len(records)
        avg_cov = sum(r["evidence_coverage"] for r in records) / len(records)
        print(f"  Mean r_tool: {avg_r_tool:.3f}")
        print(f"  Mean r_answer: {avg_r_answer:.3f}")
        print(f"  Mean coverage: {avg_cov:.2%}")


if __name__ == "__main__":
    main()
