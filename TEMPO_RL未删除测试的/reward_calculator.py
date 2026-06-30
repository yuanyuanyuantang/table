"""
TEMPO-RL Phase 0 — Reward Calculator.

Computes tool / answer / memory rewards for a single subquestion rollout.
Reads from the EvidenceLedger (read-only) and FutureDependencySet;
does NOT modify ledger state.

Usage::

    from TEMPO_RL.reward_calculator import RewardCalculator

    calc = RewardCalculator()  # all defaults
    # Or with custom weights:
    calc = RewardCalculator(eta=1.0, lambda_call=0.02, alpha_f=0.5)

    # Per tool step
    tool_r = calc.compute_tool_reward(tool_call, ledger_result)

    # After answer
    ans_r = calc.compute_answer_reward(
        answer_json, score_points, ledger,
        memory_before, observations, code_outputs,
    )

    # After memory
    mem_r = calc.compute_memory_reward(
        memory_after, memory_before, ledger,
        observations, code_outputs, future_deps,
        grounded_answer_claims=ans_r.get("claim_results"),
    )

    # Or all at once
    result = calc.compute_all(
        subquestion_id, tool_calls, ledger_updates,
        answer_json, score_points, ledger,
        memory_before, memory_after,
        observations, code_outputs, future_deps,
    )
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .schemas import (
    EvidenceItem,
    FutureDependency,
    FutureDependencySet,
    required_fields_for_type,
)
from .verifier import (
    verify_value_match,
    verify_binding_match,
    verify_source_match,
    _normalize_number,
    _normalize_text_for_match,
    _extract_numbers_from_text,
)
from .evidence_ledger import EvidenceLedger, _parse_memory_facts


# ======================================================================
# RewardCalculator
# ======================================================================

class RewardCalculator:
    """Compute TEMPO-RL tool / answer / memory rewards.

    Parameters
    ----------
    eta : float = 1.0
        Tool evidence gain scaling factor.
    lambda_call : float = 0.02
        Per-tool-call cost.
    lambda_invalid : float = 1.0
        Invalid tool call penalty.
    lambda_repeat : float = 0.2
        Repeat call penalty.
    lambda_format : float = 1.0
        Answer format error penalty.
    lambda_extra : float = 0.5
        Unsupported extra claim penalty.
    alpha_f : float = 0.5
        Memory faithfulness weight.
    alpha_s : float = 0.4
        Future dependency coverage weight.
    lambda_comp : float = 0.1
        Compression penalty weight.
    B : int = 512
        Memory budget in tokens.
    epsilon : float = 1e-9
        Numerical stability constant.
    tool_registry : set of str or None
        Set of valid tool names. If None, all tool names pass validity check.
    tokenizer : optional
        Tokenizer for computing memory length in tokens. If None, uses
        character count of json.dumps(memory_after).
    """

    def __init__(
        self,
        eta: float = 1.0,
        lambda_call: float = 0.02,
        lambda_invalid: float = 1.0,
        lambda_repeat: float = 0.2,
        lambda_format: float = 1.0,
        lambda_extra: float = 0.5,
        alpha_f: float = 0.5,
        alpha_s: float = 0.4,
        lambda_comp: float = 0.1,
        B: int = 512,
        epsilon: float = 1e-9,
        tool_registry: Optional[Set[str]] = None,
        tokenizer: Any = None,
    ):
        # Tool reward params
        self.eta = eta
        self.lambda_call = lambda_call
        self.lambda_invalid = lambda_invalid
        self.lambda_repeat = lambda_repeat

        # Answer reward params
        self.lambda_format = lambda_format
        self.lambda_extra = lambda_extra

        # Memory reward params
        self.alpha_f = alpha_f
        self.alpha_s = alpha_s
        self.lambda_comp = lambda_comp
        self.B = B

        # General
        self.epsilon = epsilon
        self.tool_registry = tool_registry
        self.tokenizer = tokenizer

        # Per-subquestion call history for repeat detection
        # key: subquestion_id (str like "sq1"), value: list of canonical call strings
        self._call_history: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_tool_reward(
        self,
        tool_call: Dict[str, Any],
        ledger_update_result: Dict[str, Any],
        subquestion_id: str = "sq1",
    ) -> Dict[str, Any]:
        """Compute tool reward for one tool step.

        ``r_tool = η * max(0, ΔΦ) - λ_call - λ_invalid * I_invalid - λ_repeat * I_repeat``

        Parameters
        ----------
        tool_call : dict
            The tool call dict with ``tool_name`` and ``arguments`` keys.
        ledger_update_result : dict
            The return value of ``EvidenceLedger.update()`` for this step.
        subquestion_id : str
            Identifier for the current subquestion (for repeat tracking).

        Returns
        -------
        dict with keys ``r_tool``, ``delta_phi``, ``is_invalid``, ``is_repeat``,
        ``audit``.
        """
        tc = tool_call or {}
        tool_name = tc.get("tool_name", "")
        tool_args = tc.get("arguments", {})

        # Compute delta_phi
        cov_before = ledger_update_result.get("coverage_before", 0.0)
        cov_after = ledger_update_result.get("coverage_after", 0.0)
        delta_phi = EvidenceLedger.compute_delta_phi(cov_before, cov_after)

        # Check invalid
        is_invalid, invalid_reason = self._detect_invalid_tool_call(tc)

        # Check has new evidence
        new_ids = ledger_update_result.get("new_evidence_ids", [])
        has_new_evidence = len(new_ids) > 0

        # Check repeat
        is_repeat = self._is_repeat_call(tc, subquestion_id, has_new_evidence)

        # Compute reward
        r_tool = (
            self.eta * delta_phi
            - self.lambda_call
            - self.lambda_invalid * (1.0 if is_invalid else 0.0)
            - self.lambda_repeat * (1.0 if is_repeat else 0.0)
        )

        return {
            "r_tool": r_tool,
            "delta_phi": delta_phi,
            "is_invalid": is_invalid,
            "is_repeat": is_repeat,
            "audit": {
                "tool_name": tool_name,
                "canonical_args": self._canonicalize_arguments(tool_args, tool_name),
                "invalid_reason": invalid_reason if is_invalid else None,
                "has_new_evidence": has_new_evidence,
                "new_evidence_ids": new_ids,
            },
        }

    def compute_answer_reward(
        self,
        answer_json: Any,
        score_points: List[str],
        ledger: EvidenceLedger,
        memory_before: Any = None,
        observations: Optional[List[Dict[str, Any]]] = None,
        code_outputs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute answer reward for a subquestion.

        ``r_ans = sum_c w_c * C_correct(c) * G(c) / max(sum_c w_c, ε)
                - λ_format * I_format - λ_extra * I_unsupported_extra``

        Parameters
        ----------
        answer_json : dict or str
            The model's ANSWER block (parsed dict or JSON string).
        score_points : list of str
            Gold score point strings for this subquestion.
        ledger : EvidenceLedger
            Current ledger state (read-only).
        memory_before : dict or str or None
            M_{i-1} memory JSON.
        observations : list of dict or None
            All tool observations for this subquestion.
        code_outputs : list of str or None
            All code execution outputs.

        Returns
        -------
        dict with keys ``r_answer``, ``claim_results``, ``format_error``,
        ``format_issues``, ``unsupported_extra_count``, ``audit``.
        """
        obs_list = observations or []
        codes = code_outputs or []

        # 1. Parse answer
        parsed, format_error, format_issues = self._parse_answer_json(answer_json)

        # Completely illegal format → -1.0
        if format_error and parsed is None:
            return {
                "r_answer": -1.0,
                "claim_results": [],
                "format_error": True,
                "format_issues": format_issues,
                "unsupported_extra_count": 0,
                "audit": {"reason": "completely illegal answer format"},
            }

        # 2. Extract claims from answer
        answer_claims = self._extract_claims_from_answer(parsed or {})
        answer_full_text = json.dumps(parsed, ensure_ascii=False) if parsed else ""

        # 3. For each score_point, find best answer claim, check C_correct and G
        claim_results: List[Dict[str, Any]] = []
        total_weight = 0.0
        numerator = 0.0
        covered_score_points: Set[int] = set()

        for sp_idx, sp in enumerate(score_points):
            sp_weight = 1.0  # default weight per score point
            total_weight += sp_weight

            # Find best matching answer claim
            best_correct = False
            best_confidence = 0.0
            best_grounded = False
            best_grounded_audit: Dict[str, Any] = {}
            best_claim_idx = -1

            for ac_idx, ac in enumerate(answer_claims):
                correct, conf = self._check_answer_claim_correct(
                    ac, [sp], answer_full_text
                )
                if correct and conf > best_confidence:
                    best_confidence = conf
                    best_correct = True
                    best_claim_idx = ac_idx
                    # Check grounding
                    grounded, g_audit = self._check_answer_claim_grounded(
                        ac, ledger, obs_list, codes, memory_before
                    )
                    best_grounded = grounded
                    best_grounded_audit = g_audit

            if best_correct:
                covered_score_points.add(sp_idx)
                numerator += sp_weight * (1.0 if best_correct else 0.0) * (1.0 if best_grounded else 0.0)

            claim_results.append({
                "score_point": sp,
                "weight": sp_weight,
                "C_correct": best_correct,
                "C_correct_confidence": best_confidence,
                "G": best_grounded,
                "G_audit": best_grounded_audit,
                "matched_claim_idx": best_claim_idx,
                "contribution": (
                    sp_weight * (1.0 if best_correct else 0.0) * (1.0 if best_grounded else 0.0)
                    / max(total_weight, self.epsilon)
                ) if best_correct else 0.0,
            })

        # 4. Compute grounded score
        denom = max(total_weight, self.epsilon)
        grounded_score = numerator / denom

        # 5. Check unsupported extras
        unsupported_count = self._detect_unsupported_extras(
            answer_claims, score_points, claim_results
        )

        # 6. Compute final answer reward
        r_answer = (
            grounded_score
            - self.lambda_format * (1.0 if format_error else 0.0)
            - self.lambda_extra * unsupported_count
        )
        r_answer = max(-1.0, min(1.0, r_answer))

        return {
            "r_answer": r_answer,
            "claim_results": claim_results,
            "format_error": format_error,
            "format_issues": format_issues,
            "unsupported_extra_count": unsupported_count,
            "audit": {
                "grounded_score": grounded_score,
                "claims_total": len(score_points),
                "claims_correct_and_grounded": len(covered_score_points),
                "format_error": format_error,
                "unsupported_extra_count": unsupported_count,
            },
        }

    def compute_memory_reward(
        self,
        memory_after: Any,
        memory_before: Any,
        ledger: EvidenceLedger,
        observations: List[Dict[str, Any]],
        code_outputs: List[str],
        future_dependency_set: Optional[FutureDependencySet] = None,
        grounded_answer_claims: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Compute memory reward for a subquestion.

        ``r_mem = α_f * F_i + α_s * S_i - λ_comp * P_comp``
        (if H_i^keep is empty, drops the S_i term)

        Parameters
        ----------
        memory_after : dict or str
            The model's M_i memory JSON.
        memory_before : dict or str
            M_{i-1} memory JSON.
        ledger : EvidenceLedger
            Current ledger state (read-only).
        observations : list of dict
            All tool observations.
        code_outputs : list of str
            All code execution outputs.
        future_dependency_set : FutureDependencySet or None
            H_i^future for this memory boundary.
        grounded_answer_claims : list of dict or None
            Pre-computed claim results from compute_answer_reward
            (claims with C_correct=True and G=True).

        Returns
        -------
        dict with keys ``r_memory``, ``F_i``, ``S_i``, ``P_comp``,
        ``H_keep_ids``, ``memory_items_parsed``, ``item_audits``,
        ``severe_failure``, ``failure_reason``.
        """
        # 1. Parse memory items
        memory_items = self._parse_memory_items(memory_after)

        # Severe failure: unparseable
        if not memory_items and self._is_memory_unparseable(memory_after):
            return self._severe_failure_result(
                memory_items, "memory JSON completely unparseable"
            )

        # 2. Compute H_i^keep
        if future_dependency_set is not None:
            h_keep = self._compute_h_keep(future_dependency_set, ledger)
        else:
            h_keep = []

        h_keep_ids = [h.dependency_id for h in h_keep]

        # Severe failure: H_keep non-empty but memory empty
        if h_keep and not memory_items:
            return self._severe_failure_result(
                memory_items,
                "H_i^keep non-empty but memory is empty or completely unrelated",
            )

        # Severe failure: majority conflict
        has_majority_conflict, conflict_reasons = self._check_memory_conflicts(
            memory_items, ledger
        )
        if has_majority_conflict:
            return self._severe_failure_result(
                memory_items,
                f"majority of memory items conflict with verified evidence: {conflict_reasons[:3]}",
            )

        # 3. Compute F_i (faithfulness)
        memory_before_items = _parse_memory_facts(memory_before)
        grounded_claims = grounded_answer_claims or []

        item_audits: List[Dict[str, Any]] = []
        total_item_weight = 0.0
        faithfulness_sum = 0.0

        for item in memory_items:
            q_mem, item_audit = self._compute_single_memory_faithfulness(
                item,
                memory_before_items,
                observations,
                code_outputs,
                grounded_claims,
            )
            item_weight = 1.0  # Default equal weight per memory item
            total_item_weight += item_weight
            faithfulness_sum += item_weight * q_mem
            item_audits.append({
                "item_text": item.get("text", "")[:120],
                "q_mem": q_mem,
                "weight": item_weight,
                "audit": item_audit,
            })

        F_i = faithfulness_sum / max(total_item_weight, self.epsilon)

        # 4. Compute S_i (future dependency coverage)
        S_i: Optional[float] = None
        h_keep_covered = 0
        if h_keep:
            total_dep_weight = 0.0
            dep_sum = 0.0
            for dep in h_keep:
                retain = self._compute_retain(dep, memory_items)
                dw = dep.weight
                total_dep_weight += dw
                dep_sum += dw * retain
                if retain > 0.5:
                    h_keep_covered += 1
            S_i = dep_sum / max(total_dep_weight, self.epsilon)

        # 5. Compute P_comp
        P_comp = self._compute_compression_penalty(memory_after)

        # 6. Compute r_mem
        if h_keep:
            r_memory = self.alpha_f * F_i + self.alpha_s * (S_i or 0.0) - self.lambda_comp * P_comp
        else:
            r_memory = self.alpha_f * F_i - self.lambda_comp * P_comp

        r_memory = max(-1.0, min(1.0, r_memory))

        return {
            "r_memory": r_memory,
            "F_i": F_i,
            "S_i": S_i,
            "P_comp": P_comp,
            "H_keep_ids": h_keep_ids,
            "H_keep_size": len(h_keep),
            "H_keep_covered": h_keep_covered,
            "memory_items_parsed": len(memory_items),
            "item_audits": item_audits,
            "severe_failure": False,
            "failure_reason": None,
        }

    def compute_all(
        self,
        subquestion_id: str,
        tool_calls: List[Dict[str, Any]],
        ledger_updates: List[Dict[str, Any]],
        answer_json: Any,
        score_points: List[str],
        ledger: EvidenceLedger,
        memory_before: Any = None,
        memory_after: Any = None,
        observations: Optional[List[Dict[str, Any]]] = None,
        code_outputs: Optional[List[str]] = None,
        future_dependency_set: Optional[FutureDependencySet] = None,
    ) -> Dict[str, Any]:
        """Compute all three reward components for one subquestion.

        This is the main entry point — it calls compute_tool_reward for each
        tool step, then compute_answer_reward, then compute_memory_reward,
        and produces the unified output dict.

        Parameters
        ----------
        subquestion_id : str
            Identifier like ``"sq1"``.
        tool_calls : list of dict
            Each tool call dict for this subquestion.
        ledger_updates : list of dict
            Each return value from ``EvidenceLedger.update()`` (same order as tool_calls).
        answer_json : dict or str
            Model ANSWER output.
        score_points : list of str
            Gold score point strings.
        ledger : EvidenceLedger
            Current ledger state (read-only, after all tool steps).
        memory_before : dict or str or None
            M_{i-1} memory.
        memory_after : dict or str or None
            M_i memory (if this is the last subquestion).
        observations : list of dict or None
            All tool observations.
        code_outputs : list of str or None
            All code execution outputs.
        future_dependency_set : FutureDependencySet or None
            H_i^future for this boundary.

        Returns
        -------
        dict with ``r_tool``, ``r_answer``, ``r_memory``,
        ``reward_summary_for_logging``, ``audit``.
        """
        obs_list = observations or []
        codes = code_outputs or []

        # --- Tool rewards ---
        tool_rewards: List[float] = []
        tool_details: List[Dict[str, Any]] = []
        total_invalid = 0
        total_steps = max(len(tool_calls), 1)  # avoid div-by-zero

        for idx, (tc, lu) in enumerate(zip(tool_calls, ledger_updates)):
            result = self.compute_tool_reward(tc, lu, subquestion_id)
            tool_rewards.append(result["r_tool"])
            tool_details.append({
                "step": idx,
                "r_tool": result["r_tool"],
                "delta_phi": result["delta_phi"],
                "is_invalid": result["is_invalid"],
                "is_repeat": result["is_repeat"],
                "audit": result["audit"],
            })
            if result["is_invalid"]:
                total_invalid += 1

        avg_tool_reward = sum(tool_rewards) / max(len(tool_rewards), 1)

        # --- Answer reward ---
        ans_result = self.compute_answer_reward(
            answer_json, score_points, ledger,
            memory_before, obs_list, codes,
        )

        # --- Memory reward ---
        # Build grounded claims list from answer result
        grounded_claims = [
            cr for cr in ans_result.get("claim_results", [])
            if cr.get("C_correct") and cr.get("G")
        ]

        mem_result = self.compute_memory_reward(
            memory_after, memory_before, ledger,
            obs_list, codes,
            future_dependency_set, grounded_claims,
        )

        # --- Unified audit ---
        coverage = ledger.coverage
        tool_valid_count = total_steps - total_invalid
        tool_valid_rate = tool_valid_count / max(total_steps, 1)

        reward_summary_for_logging = (
            avg_tool_reward + ans_result["r_answer"] + mem_result["r_memory"]
        ) / 3.0

        return {
            "r_tool": avg_tool_reward,
            "r_answer": ans_result["r_answer"],
            "r_memory": mem_result["r_memory"],
            "reward_summary_for_logging": reward_summary_for_logging,
            "audit": {
                "tool_valid_rate": tool_valid_rate,
                "evidence_coverage": coverage,
                "memory_faithfulness": mem_result["F_i"],
                "tool_details": tool_details,
                "answer_details": {
                    "claims_total": len(score_points),
                    "claims_correct_and_grounded": ans_result["audit"].get(
                        "claims_correct_and_grounded", 0
                    ),
                    "format_error": ans_result["format_error"],
                    "unsupported_extra_count": ans_result["unsupported_extra_count"],
                    "grounded_score": ans_result["audit"].get("grounded_score", 0.0),
                },
                "memory_details": {
                    "memory_items_parsed": mem_result["memory_items_parsed"],
                    "F_i": mem_result["F_i"],
                    "S_i": mem_result["S_i"],
                    "P_comp": mem_result["P_comp"],
                    "H_keep_size": mem_result.get("H_keep_size", 0),
                    "H_keep_covered": mem_result.get("H_keep_covered", 0),
                    "severe_failure": mem_result["severe_failure"],
                },
            },
        }

    def reset_call_history(self, subquestion_id: Optional[str] = None) -> None:
        """Reset call history for a specific subquestion or all."""
        if subquestion_id is not None:
            self._call_history.pop(subquestion_id, None)
        else:
            self._call_history.clear()

    # ======================================================================
    # Private — Tool Reward Helpers
    # ======================================================================

    def _canonicalize_arguments(
        self,
        args: Any,
        tool_name: str = "",
    ) -> str:
        """Produce a canonical string representation of tool arguments.

        Normalizes: path (basename), JSON key order, table range casing,
        search keyword ordering, float precision.
        """
        def _canonicalize_value(v: Any, key_hint: str = "") -> Any:
            if isinstance(v, dict):
                return {
                    k: _canonicalize_value(vv, k)
                    for k, vv in sorted(v.items())
                }
            if isinstance(v, (list, tuple)):
                return [_canonicalize_value(vv, key_hint) for vv in v]
            if isinstance(v, float):
                return round(v, 6)
            if isinstance(v, bool):
                return v
            if isinstance(v, int):
                return v
            if isinstance(v, str):
                kh = key_hint.lower()
                # Path-like: extract basename
                if kh in ("file_path", "path", "filename", "file", "table_path", "csv_path"):
                    return os.path.basename(v.replace("\\", "/"))
                # Range-like: uppercase, strip whitespace
                if kh in ("range", "region", "cell_range", "sheet_range"):
                    return v.strip().upper()
                # Search query: sort whitespace/comma-separated terms
                if kh in ("query", "keyword", "search", "keywords", "search_term"):
                    terms = sorted(
                        t.strip() for t in re.split(r"[,\s]+", v) if t.strip()
                    )
                    return " ".join(terms)
                return v.strip()
            return v

        if not isinstance(args, dict):
            return json.dumps(args, sort_keys=True, ensure_ascii=False)

        normalized = {
            k: _canonicalize_value(v, k)
            for k, v in sorted(args.items())
        }
        return json.dumps(normalized, sort_keys=True, ensure_ascii=False)

    def _detect_invalid_tool_call(
        self,
        tool_call: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Check if a tool call is structurally invalid.

        Returns (is_invalid, reason).
        """
        if not isinstance(tool_call, dict):
            return True, "tool_call is not a dict"

        tool_name = tool_call.get("tool_name", "")
        if not tool_name or not isinstance(tool_name, str):
            return True, "missing or invalid tool_name"

        # Check against registry if provided
        if self.tool_registry is not None:
            if tool_name not in self.tool_registry:
                return True, f"tool_name '{tool_name}' not in registry"

        # Check arguments
        args = tool_call.get("arguments", None)
        if args is not None and not isinstance(args, dict):
            return True, "arguments is not a dict"

        # Check JSON serializability of the full call
        try:
            json.dumps(tool_call, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            return True, f"tool_call not JSON-serializable: {e}"

        return False, ""

    def _is_repeat_call(
        self,
        tool_call: Dict[str, Any],
        subquestion_id: str,
        has_new_evidence: bool,
    ) -> bool:
        """Check if this call is a repeat.

        Repeat = same (tool_name, canonical_args) seen before AND no new evidence.
        Side effect: records the call in _call_history.
        """
        tool_name = tool_call.get("tool_name", "")
        tool_args = tool_call.get("arguments", {})
        canonical = self._canonicalize_arguments(tool_args, tool_name)
        signature = f"{tool_name}::{canonical}"

        if subquestion_id not in self._call_history:
            self._call_history[subquestion_id] = []

        is_repeat = signature in self._call_history[subquestion_id] and not has_new_evidence
        self._call_history[subquestion_id].append(signature)

        return is_repeat

    # ======================================================================
    # Private — Answer Reward Helpers
    # ======================================================================

    def _parse_answer_json(
        self,
        answer_raw: Any,
    ) -> Tuple[Optional[Dict[str, Any]], bool, List[str]]:
        """Parse the model's answer into a structured dict.

        Returns (parsed_dict_or_None, is_format_error, issues_list).
        If completely unparseable, parsed_dict is None and format_error is True.
        """
        issues: List[str] = []

        if answer_raw is None:
            return None, True, ["answer is None"]

        if isinstance(answer_raw, dict):
            # Already a dict — check it has at least some content
            if not answer_raw:
                return None, True, ["answer dict is empty"]
            return answer_raw, False, []

        if isinstance(answer_raw, str):
            s = answer_raw.strip()
            if not s:
                return None, True, ["answer string is empty"]

            # Try direct JSON parse first
            try:
                parsed = json.loads(s)
                from_direct_json = True
            except json.JSONDecodeError:
                from_direct_json = False
                # Try extracting from markdown code blocks
                fenced = re.findall(r"```(?:json)?\s*\n?(.*?)\n?```", s, re.DOTALL)
                if fenced:
                    for block in fenced:
                        try:
                            parsed = json.loads(block.strip())
                            break
                        except json.JSONDecodeError:
                            continue
                    else:
                        return None, True, ["cannot parse answer as JSON"]
                else:
                    return None, True, ["answer string is not valid JSON"]

            if isinstance(parsed, dict):
                if not parsed:
                    return None, True, ["parsed answer dict is empty"]
                if from_direct_json:
                    # Direct JSON parse → model followed the required format
                    return parsed, False, []
                else:
                    # Extracted from markdown → still a format warning
                    return parsed, True, ["answer extracted from markdown code block (format warning)"]
            else:
                # Parsed but not a dict — wrap it, still a format warning
                return {"_raw": parsed}, True, ["answer parsed to non-dict, wrapped"]

        # Unsupported type
        return None, True, [f"unsupported answer type: {type(answer_raw)}"]

    def _extract_claims_from_answer(
        self,
        answer_json: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Extract individual claims from the answer dict.

        Handles the SFT answer format:
          {"score_points": [{"scorepoint": "...", "table": "..."}, ...]}

        Also handles flat structures and other reasonable formats.
        Returns list of claim dicts, each with at minimum {"text": str}.
        """
        claims: List[Dict[str, Any]] = []

        # SFT format: score_points array
        sp_list = answer_json.get("score_points")
        if isinstance(sp_list, list):
            for sp in sp_list:
                if isinstance(sp, dict):
                    text = sp.get("scorepoint") or sp.get("text") or sp.get("content") or ""
                    claims.append({
                        "text": str(text),
                        "table": sp.get("table") or sp.get("data_source") or "",
                    })
                elif isinstance(sp, str):
                    claims.append({"text": sp, "table": ""})
            if claims:
                return claims

        # Flat answer: try other common keys
        for key in ("answer", "result", "conclusion", "claims"):
            val = answer_json.get(key)
            if isinstance(val, str):
                claims.append({"text": val, "table": ""})
            elif isinstance(val, list):
                for v in val:
                    if isinstance(v, dict):
                        claims.append({
                            "text": str(v.get("text", v.get("value", json.dumps(v, ensure_ascii=False)))),
                            "table": v.get("table", v.get("data_source", "")),
                        })
                    elif isinstance(v, str):
                        claims.append({"text": v, "table": ""})

        if claims:
            return claims

        # Fallback: treat the whole answer as one text claim
        claims.append({
            "text": json.dumps(answer_json, ensure_ascii=False),
            "table": "",
        })
        return claims

    def _check_answer_claim_correct(
        self,
        answer_claim: Dict[str, Any],
        score_points: List[str],
        answer_full_text: str = "",
    ) -> Tuple[bool, float]:
        """Check C_correct(c): did the answer correctly express the score point?

        Returns (is_correct, confidence_score in [0, 1]).
        """
        claim_text = answer_claim.get("text", "")

        if not claim_text:
            return False, 0.0

        norm_claim = _normalize_text_for_match(claim_text)

        for sp in score_points:
            norm_sp = _normalize_text_for_match(sp)

            if not norm_sp:
                continue

            # 1. Exact normalized match
            if norm_claim == norm_sp:
                return True, 1.0

            # 2. Substring match (claim in score_point or vice versa)
            if len(norm_claim) >= 4 and norm_claim in norm_sp:
                return True, 0.9
            if len(norm_sp) >= 4 and norm_sp in norm_claim:
                return True, 0.8

            # 3. Numeric comparison
            claim_nums = _extract_numbers_from_text(claim_text)
            sp_nums = _extract_numbers_from_text(sp)
            if claim_nums and sp_nums:
                claim_vals = {v for _, v in claim_nums}
                sp_vals = {v for _, v in sp_nums}
                common = claim_vals & sp_vals
                if len(common) >= len(sp_vals):
                    return True, 0.95
                if common:
                    return True, 0.7

            # 4. Partial text overlap (for longer texts)
            if len(norm_sp) >= 8:
                # Check if core segments of the score point appear in the claim
                core = norm_sp[:min(len(norm_sp), 12)]
                if core in norm_claim:
                    return True, 0.6

        return False, 0.0

    def _check_answer_claim_grounded(
        self,
        answer_claim: Dict[str, Any],
        ledger: EvidenceLedger,
        observations: List[Dict[str, Any]],
        code_outputs: List[str],
        memory_before: Any,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check G(c): can this claim be grounded in evidence?

        Checks: observation texts, memory, code outputs, verified evidence.
        """
        claim_text = answer_claim.get("text", "")
        claim_table = answer_claim.get("table", "")

        # Build combined text pools
        obs_text = " ".join(
            o.get("content", "") if isinstance(o, dict) else str(o)
            for o in observations
        )
        code_text = " ".join(code_outputs)

        # Memory text: parse memory and extract all text + entity/time/metric/unit
        mem_facts = _parse_memory_facts(memory_before)
        mem_parts: List[str] = []
        for f in mem_facts:
            for k in ("entity", "time", "metric", "value", "unit", "text", "provenance"):
                v = f.get(k)
                if v:
                    mem_parts.append(str(v))
        mem_text = " ".join(mem_parts)

        # Verified evidence values
        verified_text = " ".join(
            ei.value for ei in ledger.verified_items
        )

        # Build a temporary evidence item for the claim
        source_tables = [claim_table] if claim_table else []
        temp_ei = EvidenceItem(
            sample_id="_",
            subquestion_id=0,
            evidence_id="_temp",
            type="text_fact",
            value=claim_text,
            source_tables=source_tables,
        )

        # Check value match against all sources
        combined_text = f"{obs_text} {mem_text} {code_text} {verified_text}"
        v_ok, v_audit = verify_value_match(
            temp_ei,
            observation_text=combined_text,
            memory_text="",
            code_output="",
        )

        # Check source match if table is specified
        source_ok = True
        s_audit: Dict[str, Any] = {"note": "no table specified"}
        if claim_table:
            source_ok = False
            # Check against observation metadata
            for obs in observations:
                if isinstance(obs, dict):
                    meta = obs.get("metadata", obs.get("observation_metadata"))
                    if meta:
                        source_ok, s_audit = verify_source_match(
                            temp_ei,
                            observation_metadata=meta,
                        )
                        if source_ok:
                            break
                    # Also check observation content for table name
                    content = obs.get("content", "")
                    if claim_table.lower() in content.lower():
                        source_ok = True
                        s_audit = {"note": "table found in observation content"}
                        break

        grounded = v_ok and source_ok
        audit = {
            "value_match": v_audit,
            "source_match": s_audit if claim_table else {"note": "no table constraint"},
            "grounded": grounded,
        }

        return grounded, audit

    def _detect_unsupported_extras(
        self,
        answer_claims: List[Dict[str, Any]],
        score_points: List[str],
        claim_results: List[Dict[str, Any]],
    ) -> int:
        """Count answer claims that match no score_point AND have no grounding.

        Since claim_results maps score_points to answer claims, extra claims
        are those NOT referenced by any claim_result with C_correct=True.

        We count answer claims that:
        1. Were not matched as C_correct for any score_point
        2. And are not substantive (non-empty, non-trivial text)
        """
        matched_claim_indices: Set[int] = set()
        for cr in claim_results:
            idx = cr.get("matched_claim_idx", -1)
            if idx >= 0 and cr.get("C_correct"):
                matched_claim_indices.add(idx)

        extra_count = 0
        for idx, ac in enumerate(answer_claims):
            if idx in matched_claim_indices:
                continue
            text = ac.get("text", "").strip()
            # Only count substantive extra claims (non-trivial length)
            if len(text) >= 8:
                extra_count += 1

        return extra_count

    # ======================================================================
    # Private — Memory Reward Helpers
    # ======================================================================

    def _parse_memory_items(self, memory_json: Any) -> List[Dict[str, Any]]:
        """Parse memory_after into individual memory items.

        Delegates to evidence_ledger._parse_memory_facts for structured parsing.
        """
        return _parse_memory_facts(memory_json)

    def _is_memory_unparseable(self, memory_json: Any) -> bool:
        """Check if memory is completely unparseable."""
        if memory_json is None:
            return True
        if isinstance(memory_json, str):
            s = memory_json.strip()
            if not s:
                return True
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                return True
            return not isinstance(parsed, dict)
        if isinstance(memory_json, dict):
            return not bool(memory_json)
        return True

    def _compute_single_memory_faithfulness(
        self,
        memory_item: Dict[str, Any],
        memory_before_items: List[Dict[str, Any]],
        observations: List[Dict[str, Any]],
        code_outputs: List[str],
        grounded_claims: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute q_mem for one memory item.

        q_mem = C_value * C_binding * C_provenance (each 0 or 1).
        """
        item_text = memory_item.get("text", "")
        item_value = memory_item.get("value", "")
        item_entity = memory_item.get("entity")
        item_time = memory_item.get("time")
        item_metric = memory_item.get("metric")
        item_unit = memory_item.get("unit")

        # Build combined source text from allowed provenances
        # 1. memory_before items
        mem_before_text = " ".join(f.get("text", "") for f in memory_before_items)

        # 2. observation contents
        obs_text = " ".join(
            o.get("content", "") if isinstance(o, dict) else str(o)
            for o in observations
        )

        # 3. code outputs
        code_text = " ".join(code_outputs)

        # 4. grounded answer claims
        claims_text = " ".join(
            cr.get("score_point", "") if isinstance(cr, dict) else str(cr)
            for cr in grounded_claims
        )

        # Build a temporary evidence item for verification
        temp_ei = EvidenceItem(
            sample_id="_",
            subquestion_id=0,
            evidence_id="_mem",
            type="raw_value" if item_value else "text_fact",
            value=item_value or item_text,
            entity=item_entity,
            time=item_time,
            metric=item_metric,
            unit=item_unit,
            source_tables=[],
        )

        all_source_text = f"{mem_before_text} {obs_text} {code_text} {claims_text}"

        # C_value: does the item's value/text appear in any allowed source?
        v_ok, v_audit = verify_value_match(
            temp_ei,
            observation_text=all_source_text,
            memory_text="",
            code_output="",
        )

        # C_binding: check entity/time/metric/unit in context
        b_ok, b_audit = verify_binding_match(
            temp_ei,
            observation_text=all_source_text,
            memory_text="",
            tool_arguments=None,
        )

        # C_provenance: does the item have a traceable source?
        provenance = memory_item.get("provenance", "")
        prov_ok = False
        if provenance:
            prov_lower = provenance.lower()
            # Check in memory_before provenance
            for f in memory_before_items:
                fp = (f.get("provenance") or "").lower()
                if prov_lower in fp or fp in prov_lower:
                    prov_ok = True
                    break
            # Check in observation metadata
            if not prov_ok:
                for obs in observations:
                    if isinstance(obs, dict):
                        meta = obs.get("metadata", obs.get("observation_metadata"))
                        if meta:
                            file = (meta.get("file") or meta.get("filename") or "").lower()
                            if prov_lower in file or file in prov_lower:
                                prov_ok = True
                                break
                        # Also check raw content
                        content = (obs.get("content") or "").lower()
                        if prov_lower in content:
                            prov_ok = True
                            break
            # Check if item text appears in claims (indirect provenance via answer)
            if not prov_ok and item_text:
                if any(
                    item_text[:20] in (cr.get("score_point", "") if isinstance(cr, dict) else str(cr))
                    for cr in grounded_claims
                ):
                    prov_ok = True
            # Check code outputs — if value matches code output, provenance is satisfied
            if not prov_ok:
                item_val = memory_item.get("value", "")
                if item_val:
                    for code in code_outputs:
                        if item_val in code:
                            prov_ok = True
                            break
        else:
            # No provenance field — check if the value/text can be found in
            # any source at all (if C_value passed, provenance is weakly satisfied)
            prov_ok = v_ok

        # Combine: all three must pass
        q_mem = (1.0 if v_ok else 0.0) * (1.0 if b_ok else 0.0) * (1.0 if prov_ok else 0.0)

        audit = {
            "C_value": {"passed": v_ok, "detail": v_audit},
            "C_binding": {"passed": b_ok, "detail": b_audit},
            "C_provenance": {"passed": prov_ok, "provenance": provenance if provenance else "(none)"},
        }

        return q_mem, audit

    def _check_memory_conflicts(
        self,
        memory_items: List[Dict[str, Any]],
        ledger: EvidenceLedger,
    ) -> Tuple[bool, List[str]]:
        """Check if majority of memory items conflict with verified evidence.

        A conflict: same (entity, time, metric) but substantially different value (>5%).
        """
        if not memory_items:
            return False, []

        verified_items = ledger.verified_items
        if not verified_items:
            return False, []

        conflict_count = 0
        conflict_reasons: List[str] = []

        for item in memory_items:
            item_entity = (item.get("entity") or "").strip()
            item_time = (item.get("time") or "").strip()
            item_metric = (item.get("metric") or "").strip()
            item_value_str = item.get("value") or ""

            if not item_value_str:
                continue  # Can't check conflicts without a value

            item_val = _normalize_number(item_value_str)
            if item_val is None:
                continue

            for ei in verified_items:
                ei_entity = (ei.entity or "").strip()
                ei_time = (ei.time or "").strip()
                ei_metric = (ei.metric or "").strip()

                # Check if same entity+time+metric (at least 2 of 3 match)
                matches = 0
                if item_entity and ei_entity and item_entity == ei_entity:
                    matches += 1
                if item_time and ei_time and item_time == ei_time:
                    matches += 1
                if item_metric and ei_metric and item_metric == ei_metric:
                    matches += 1

                if matches >= 2:
                    # Compare values
                    ei_val = _normalize_number(ei.value or "")
                    if ei_val is not None and abs(ei_val) > self.epsilon:
                        rel_diff = abs(item_val - ei_val) / abs(ei_val)
                        if rel_diff > 0.05:  # 5% tolerance
                            conflict_count += 1
                            conflict_reasons.append(
                                f"mem value {item_val} vs verified {ei_val} "
                                f"(entity={item_entity}, time={item_time}, metric={item_metric})"
                            )
                            break

        # Majority conflict if >50% of items are in conflict
        has_majority = (
            conflict_count > 0
            and len(memory_items) > 0
            and conflict_count / len(memory_items) > 0.5
        )
        return has_majority, conflict_reasons

    def _compute_support(
        self,
        dependency: FutureDependency,
        ledger: EvidenceLedger,
    ) -> bool:
        """Compute Support(h, L): can this dependency be satisfied by the ledger?"""
        dep = dependency
        verified_ids = ledger.verified_ids
        verified_items = ledger.verified_items

        if dep.type == "numeric_fact":
            if dep.source_evidence_id:
                return dep.source_evidence_id in verified_ids
            # Without source_evidence_id, check by fields
            entity = dep.fields.get("entity", "")
            time = dep.fields.get("time", "")
            metric = dep.fields.get("metric", "")
            for ei in verified_items:
                if (entity and ei.entity and entity in ei.entity) or \
                   (time and ei.time and time in ei.time) or \
                   (metric and ei.metric and metric in ei.metric):
                    # At least one field matches → likely supported
                    if entity and ei.entity and entity in ei.entity:
                        return True
            return False

        elif dep.type == "entity_set":
            entities_needed = set(dep.fields.get("entities", []))
            if not entities_needed:
                return False
            entities_in_ledger = set()
            for ei in verified_items:
                if ei.entity:
                    entities_in_ledger.add(ei.entity)
            # At least one matching entity = supported
            return len(entities_needed & entities_in_ledger) > 0

        elif dep.type == "reference":
            # References are semantic — always considered supported by the ledger
            return True

        elif dep.type == "constraint":
            constraint = dep.fields.get("constraint_content", "")
            if not constraint:
                return False
            norm_c = _normalize_text_for_match(constraint)
            for ei in verified_items:
                if norm_c in _normalize_text_for_match(ei.value or ""):
                    return True
            return False

        elif dep.type == "table_ref":
            table_name = dep.fields.get("table_name", "")
            if not table_name:
                return False
            for ei in verified_items:
                for tbl in ei.source_tables:
                    if table_name.lower() in tbl.lower() or tbl.lower() in table_name.lower():
                        return True
            return False

        return False

    def _compute_h_keep(
        self,
        future_dependency_set: FutureDependencySet,
        ledger: EvidenceLedger,
    ) -> List[FutureDependency]:
        """Filter H_i^future to H_i^keep: only deps supported by current ledger."""
        return [
            dep for dep in future_dependency_set.future_dependencies
            if self._compute_support(dep, ledger)
        ]

    def _compute_retain(
        self,
        dependency: FutureDependency,
        memory_items: List[Dict[str, Any]],
    ) -> float:
        """Compute I_retain(h, M_i): product over RequiredFields.

        Returns 0.0 or 1.0 (binary product — all required fields must be present).
        """
        required = required_fields_for_type(dependency.type)
        if not required:
            return 0.0

        # Build combined text from all memory items
        combined_text = " ".join(
            item.get("text", "") for item in memory_items
        )
        norm_combined = _normalize_text_for_match(combined_text)

        all_pass = True

        for field_name in required:
            field_val = dependency.fields.get(field_name)
            if field_val is None:
                all_pass = False
                continue

            if field_name == "entities":
                # Check each entity appears in memory
                entities = field_val if isinstance(field_val, list) else [field_val]
                for ent in entities:
                    ent_str = str(ent)
                    norm_ent = _normalize_text_for_match(ent_str)
                    found = False
                    for item in memory_items:
                        item_text = _normalize_text_for_match(
                            item.get("text", "") + " " + (item.get("entity") or "")
                        )
                        if norm_ent in item_text:
                            found = True
                            break
                    if not found:
                        all_pass = False
            elif field_name == "target_sq":
                # target_sq is structural (e.g. "sq1") — lenient: always pass
                # since memory implicitly references its own subquestion
                pass
            else:
                # Single field value — check if it appears in memory
                field_str = str(field_val)
                norm_field = _normalize_text_for_match(field_str)
                found = False

                # For numeric_fact fields, check specific item keys first
                for item in memory_items:
                    if field_name in item and item[field_name]:
                        item_field = _normalize_text_for_match(str(item[field_name]))
                        if norm_field in item_field or item_field in norm_field:
                            found = True
                            break

                # Fallback to checking combined text
                if not found and norm_field in norm_combined:
                    found = True

                if not found:
                    all_pass = False

        return 1.0 if all_pass else 0.0

    def _compute_compression_penalty(self, memory_after: Any) -> float:
        """Compute P_comp = max(0, len(M_i) - B) / B."""
        if self.tokenizer is not None:
            # Use tokenizer to count tokens
            try:
                mem_str = json.dumps(memory_after, ensure_ascii=False) \
                    if not isinstance(memory_after, str) else memory_after
                tokens = self.tokenizer.encode(mem_str)
                mem_len = len(tokens)
            except Exception:
                mem_len = len(
                    json.dumps(memory_after, ensure_ascii=False)
                    if not isinstance(memory_after, str) else memory_after
                )
        else:
            # Character count proxy
            if isinstance(memory_after, str):
                mem_len = len(memory_after)
            else:
                mem_len = len(json.dumps(memory_after, ensure_ascii=False))

        return max(0.0, (mem_len - self.B) / self.B)

    def _severe_failure_result(
        self,
        memory_items: List[Dict[str, Any]],
        reason: str,
    ) -> Dict[str, Any]:
        """Build a standardized severe failure result dict."""
        return {
            "r_memory": -1.0,
            "F_i": 0.0,
            "S_i": 0.0,
            "P_comp": 0.0,
            "H_keep_ids": [],
            "H_keep_size": 0,
            "H_keep_covered": 0,
            "memory_items_parsed": len(memory_items),
            "item_audits": [],
            "severe_failure": True,
            "failure_reason": reason,
        }
