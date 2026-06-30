#!/usr/bin/env python3
"""
TEMPO-RL Phase 0 — Offline Audit Script.

Given benchmark samples and cleaned SFT trajectories, runs the full Phase 0
reward infrastructure pipeline and produces audit artefacts:

  output_dir/
    target_evidence.jsonl
    future_dependencies.jsonl
    ledger_audit.jsonl
    reward_audit.jsonl
    phase0_report.md

Usage::

    python TEMPO_RL/run_phase0_audit.py \\
        --samples dataset/train不含val的.json \\
        --sft SFTbuild/output/memory_verified_subquestions.jsonl \\
        --output_dir TEMPO_RL/output/phase0_audit \\
        --max_samples 5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from TEMPO_RL.schemas import (
    EvidenceItem,
    FutureDependency,
    FutureDependencySet,
    TargetEvidenceSet,
    required_fields_for_type,
)
from TEMPO_RL.build_target_evidence import TargetEvidenceBuilder
from TEMPO_RL.build_future_dependencies import FutureDependencyBuilder
from TEMPO_RL.evidence_ledger import EvidenceLedger, _parse_memory_facts
from TEMPO_RL.reward_calculator import RewardCalculator
from TEMPO_RL.io_utils import write_jsonl, read_jsonl, load_json_file


# ======================================================================
# SFT memory format adaptation
# ======================================================================

def adapt_sft_memory_for_ledger(sft_memory: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert SFT memory format to the format expected by ``_parse_memory_facts``.

    SFT format uses ``facts`` (list of strings) and ``derived`` (list of strings),
    while ``_parse_memory_facts`` expects ``key_facts`` (list of dicts) and
    ``derived_results`` (list of dicts).  SFT ``tables`` entries also use
    ``content`` instead of ``description``.

    Returns a dict suitable for passing to ``EvidenceLedger.initialize_from_memory()``.
    """
    if not sft_memory or not isinstance(sft_memory, dict):
        return {}

    converted: Dict[str, Any] = {}

    # goal — pass through
    if "goal" in sft_memory:
        converted["goal"] = sft_memory["goal"]

    # pitfalls — pass through
    if "pitfalls" in sft_memory:
        converted["pitfalls"] = sft_memory["pitfalls"]

    # facts → key_facts (wrap strings in dicts with text field)
    facts = sft_memory.get("facts", [])
    if facts:
        converted["key_facts"] = [
            {"text": f} if isinstance(f, str) else f
            for f in facts
        ]

    # derived → derived_results (wrap strings in dicts)
    derived = sft_memory.get("derived", [])
    if derived:
        converted["derived_results"] = [
            {"text": d, "value": d} if isinstance(d, str) else d
            for d in derived
        ]

    # tables — rename content → description
    tables = sft_memory.get("tables", [])
    if tables:
        converted_tables = []
        for t in tables:
            if isinstance(t, dict):
                ct = dict(t)
                if "content" in ct and "description" not in ct:
                    ct["description"] = ct.pop("content")
                converted_tables.append(ct)
            else:
                converted_tables.append(t)
        converted["tables"] = converted_tables

    # constraints — pass through
    if "constraints" in sft_memory:
        converted["constraints"] = sft_memory["constraints"]

    return converted


def adapt_sft_memory_for_reward(sft_memory: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Same as adapt_sft_memory_for_ledger but also includes facts as-is for reward calc."""
    # The reward calculator calls _parse_memory_facts internally which uses
    # the same format, so we use the same adapter
    return adapt_sft_memory_for_ledger(sft_memory)


# ======================================================================
# Helpers
# ======================================================================

# _load_json / _load_jsonl replaced by io_utils.read_jsonl / load_json_file
_load_json = load_json_file
_load_jsonl = read_jsonl


def _index_by_task(samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index benchmark samples by task string."""
    idx: Dict[str, Dict[str, Any]] = {}
    for s in samples:
        task = s.get("task", "")
        if task:
            idx[task] = s
    return idx


# ======================================================================
# Main audit pipeline
# ======================================================================

def run_audit(
    samples_path: str,
    sft_path: str,
    output_dir: str,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the full Phase 0 audit pipeline.

    Returns a summary dict for the report.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"[1/6] Loading benchmark samples from {samples_path} ...")
    benchmark_samples = _load_json(samples_path)
    if not isinstance(benchmark_samples, list):
        benchmark_samples = [benchmark_samples]
    task_index = _index_by_task(benchmark_samples)

    print(f"[1/6] Loading SFT trajectories from {sft_path} ...")
    sft_records = _load_jsonl(sft_path)

    # Group SFT records by sample_id (task string)
    sft_by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in sft_records:
        sid = rec.get("sample_id", "")
        if sid and rec.get("_trajectory_cleaned"):
            sft_by_sample[sid].append(rec)

    # Only process samples that exist in benchmark AND have SFT data
    common_ids = sorted(set(task_index.keys()) & set(sft_by_sample.keys()))
    if max_samples and max_samples > 0:
        common_ids = common_ids[:max_samples]

    print(f"  Found {len(common_ids)} samples with both benchmark and SFT data")

    if not common_ids:
        print("ERROR: No overlapping samples found!")
        return {"error": "no_overlap"}

    # ------------------------------------------------------------------
    # 2. Build target evidence
    # ------------------------------------------------------------------
    try:
        from src.utils.chat_api import ChatClient
        llm_client = ChatClient(provider="openai", config_key="mimo")
        llm_enabled = True
    except Exception:
        llm_client = None
        llm_enabled = False
        print("  (LLM client unavailable — will use rule-only extraction)")

    print(f"[2/6] Building target evidence ...")
    tes_builder = TargetEvidenceBuilder(llm_client=llm_client, llm_enabled=llm_enabled)
    all_target_evidence: List[TargetEvidenceSet] = []
    evidence_stats: Dict[str, int] = Counter()

    for task_id in common_ids:
        sample = task_index[task_id]
        # Build TES for each subquestion separately
        tes_list = tes_builder.build_one_sample(sample)
        all_target_evidence.extend(tes_list)
        for tes in tes_list:
            evidence_stats["total_items"] += len(tes.evidence_items)
            for ei in tes.evidence_items:
                evidence_stats[f"type_{ei.type}"] += 1

    # Save target_evidence.jsonl
    te_path = os.path.join(output_dir, "target_evidence.jsonl")
    write_jsonl(te_path, [tes.to_dict() for tes in all_target_evidence])
    print(f"  Wrote {len(all_target_evidence)} target evidence sets to {te_path}")

    # Build TES lookup: (sample_id, subquestion_id) → TargetEvidenceSet
    tes_lookup: Dict[tuple, TargetEvidenceSet] = {}
    for tes in all_target_evidence:
        tes_lookup[(tes.sample_id, tes.subquestion_id)] = tes

    # ------------------------------------------------------------------
    # 3. Build future dependencies
    # ------------------------------------------------------------------
    print(f"[3/6] Building future dependencies ...")
    fd_builder = FutureDependencyBuilder(d_fdc=2, llm_client=llm_client, llm_enabled=llm_enabled)
    all_future_deps: List[FutureDependencySet] = []

    # Build per-sample target evidence index
    te_index: Dict[Tuple[str, int], List[EvidenceItem]] = defaultdict(list)
    for tes in all_target_evidence:
        te_index[(tes.sample_id, tes.subquestion_id)].extend(tes.evidence_items)

    for task_id in common_ids:
        sample = task_index[task_id]
        fds_list = fd_builder.build_one_sample(sample, te_index=te_index)
        all_future_deps.extend(fds_list)

    # Save future_dependencies.jsonl
    fd_path = os.path.join(output_dir, "future_dependencies.jsonl")
    write_jsonl(fd_path, [fds.to_dict() for fds in all_future_deps])
    total_deps = sum(len(fds.future_dependencies) for fds in all_future_deps)
    print(f"  Wrote {len(all_future_deps)} dependency sets ({total_deps} deps) to {fd_path}")

    # Build FDS lookup: (sample_id, boundary) → FutureDependencySet
    fds_lookup: Dict[tuple, FutureDependencySet] = {}
    for fds in all_future_deps:
        fds_lookup[(fds.sample_id, fds.boundary)] = fds

    # ------------------------------------------------------------------
    # 4. Process trajectories — ledger + rewards
    # ------------------------------------------------------------------
    print(f"[4/6] Processing trajectories ...")
    calculator = RewardCalculator()
    ledger_records: List[Dict[str, Any]] = []
    reward_records: List[Dict[str, Any]] = []

    # Stats accumulators
    all_tool_rewards: List[float] = []
    all_answer_rewards: List[float] = []
    all_memory_rewards: List[float] = []
    all_coverages: List[float] = []
    warnings: Counter = Counter()
    total_tool_steps = 0
    total_invalid_steps = 0
    total_repeat_steps = 0

    for task_id in common_ids:
        sample = task_index[task_id]
        sq_records = sorted(sft_by_sample[task_id], key=lambda r: r.get("subquestion_id", 0))

        # Track memory_after from previous subquestion (for memory_before of next)
        # The first subquestion's memory_before is used as-is
        # For subsequent subquestions, the previous memory_after is the memory_before

        for sq_idx, sft_rec in enumerate(sq_records):
            sq_id = sft_rec.get("subquestion_id", 0)

            # Target evidence for this subquestion
            tes_key = (task_id, sq_id)
            if tes_key not in tes_lookup:
                warnings["missing_target_evidence"] += 1
                continue
            tes = tes_lookup[tes_key]

            # Future dependencies for this boundary
            boundary = f"after_sq{sq_id}"
            fds_key = (task_id, boundary)
            fds = fds_lookup.get(fds_key)

            # Adapt SFT memory format
            sft_mem_before = sft_rec.get("memory_before", {})
            adapted_mem_before = adapt_sft_memory_for_ledger(sft_mem_before)

            sft_mem_after = sft_rec.get("memory_after", {})
            adapted_mem_after = adapt_sft_memory_for_ledger(sft_mem_after)

            # --- Initialize ledger ---
            ledger = EvidenceLedger.from_target_evidence_set(
                tes,
                memory_before=adapted_mem_before,
            )

            # --- Process tool steps ---
            agent_steps = sft_rec.get("agent_steps", [])
            tool_calls_list: List[Dict[str, Any]] = []
            ledger_updates: List[Dict[str, Any]] = []
            all_observations: List[Dict[str, Any]] = []
            all_code_outputs: List[str] = []

            for step in agent_steps:
                if step.get("type") != "tool_call":
                    continue

                # Each step may have multiple tool calls with multiple observations
                tcs = step.get("tool_calls", [])
                obss = step.get("observations", [])

                # Pair tool_calls with observations by tool_call_id
                obs_by_id: Dict[str, Dict[str, Any]] = {}
                for obs in obss:
                    obs_by_id[obs.get("tool_call_id", "")] = obs

                for tc in tcs:
                    tc_id = tc.get("tool_call_id", "")
                    obs = obs_by_id.get(tc_id, {})

                    total_tool_steps += 1

                    # Build observation metadata from tool args and obs
                    obs_metadata: Dict[str, Any] = {}
                    args = tc.get("arguments", {})
                    for k in ("file_path", "path", "table", "table_name", "filename"):
                        if k in args:
                            obs_metadata["file"] = args[k]
                            break
                    # Also try to extract file from observation content
                    content = obs.get("content", "")
                    if isinstance(content, str) and not obs_metadata.get("file"):
                        m = re.search(r'(\S+\.(?:xlsx|csv|xls))', content)
                        if m:
                            obs_metadata["file"] = m.group(1)

                    # Extract code output for python_exec / calculator tools
                    code_output = ""
                    if tc.get("tool_name") in ("python_exec", "calculator", "code_exec"):
                        code_output = obs.get("content", "")
                        if isinstance(code_output, str):
                            all_code_outputs.append(code_output)

                    # Update ledger
                    update_result = ledger.update(
                        tool_call=tc,
                        observation=obs,
                        observation_metadata=obs_metadata if obs_metadata else None,
                        code_output=code_output,
                    )

                    tool_calls_list.append(tc)
                    ledger_updates.append(update_result)
                    all_observations.append(obs)

                    if not update_result.get("new_evidence_ids"):
                        pass  # no new evidence

            # --- Compute tool rewards ---
            tool_details = []
            tool_valid_count = 0
            for idx, (tc, lu) in enumerate(zip(tool_calls_list, ledger_updates)):
                tr = calculator.compute_tool_reward(tc, lu, f"sq{sq_id}")
                tool_details.append(tr)
                all_tool_rewards.append(tr["r_tool"])
                if not tr["is_invalid"]:
                    tool_valid_count += 1
                if tr["is_repeat"]:
                    total_repeat_steps += 1
                if tr["is_invalid"]:
                    total_invalid_steps += 1

            avg_tool_reward = (
                sum(d["r_tool"] for d in tool_details) / max(len(tool_details), 1)
                if tool_details else 0.0
            )

            # --- Compute answer reward ---
            assistant_answer = sft_rec.get("assistant_answer", {})
            score_points = []
            checkout = sample.get("design", {}).get("checkout_list", [])
            for c in checkout:
                if c.get("idx") == sq_id:  # both idx and sq_id are 1-based
                    score_points = c.get("score_points", [])
                    break

            ans_result = calculator.compute_answer_reward(
                answer_json=assistant_answer,
                score_points=score_points,
                ledger=ledger,
                memory_before=adapted_mem_before,
                observations=all_observations,
                code_outputs=all_code_outputs,
            )
            all_answer_rewards.append(ans_result["r_answer"])

            # --- Compute memory reward ---
            grounded_claims = [
                cr for cr in ans_result.get("claim_results", [])
                if cr.get("C_correct") and cr.get("G")
            ]

            mem_result = calculator.compute_memory_reward(
                memory_after=adapted_mem_after,
                memory_before=adapted_mem_before,
                ledger=ledger,
                observations=all_observations,
                code_outputs=all_code_outputs,
                future_dependency_set=fds,
                grounded_answer_claims=grounded_claims,
            )
            all_memory_rewards.append(mem_result["r_memory"])

            if mem_result.get("severe_failure"):
                warnings["memory_severe_failure"] += 1

            # --- Collect coverage ---
            all_coverages.append(ledger.coverage)

            # --- Build ledger audit record ---
            ledger_records.append({
                "sample_id": task_id[:120],
                "subquestion_id": sq_id,
                "tool_steps": len(tool_calls_list),
                "coverage_initial": 0.0,
                "coverage_final": ledger.coverage,
                "verified_count": len(ledger.verified_ids),
                "target_count": len(tes.evidence_items),
                "verified_ids": sorted(ledger.verified_ids),
                "ledger_snapshot": ledger.to_dict(),
            })

            # --- Build reward audit record ---
            reward_records.append({
                "sample_id": task_id[:120],
                "subquestion_id": sq_id,
                "r_tool": avg_tool_reward,
                "r_answer": ans_result["r_answer"],
                "r_memory": mem_result["r_memory"],
                "reward_summary": (avg_tool_reward + ans_result["r_answer"] + mem_result["r_memory"]) / 3.0,
                "evidence_coverage": ledger.coverage,
                "tool_valid_rate": tool_valid_count / max(len(tool_calls_list), 1) if tool_calls_list else 1.0,
                "memory_faithfulness": mem_result["F_i"],
                "memory_fdc": mem_result.get("S_i"),
                "tool_details": [
                    {
                        "step": i,
                        "tool_name": td["audit"]["tool_name"],
                        "r_tool": td["r_tool"],
                        "delta_phi": td["delta_phi"],
                        "is_invalid": td["is_invalid"],
                        "is_repeat": td["is_repeat"],
                    }
                    for i, td in enumerate(tool_details)
                ],
                "answer_audit": {
                    "claims_total": len(score_points),
                    "claims_correct_and_grounded": ans_result["audit"].get("claims_correct_and_grounded", 0),
                    "format_error": ans_result["format_error"],
                    "unsupported_extra": ans_result["unsupported_extra_count"],
                },
                "memory_audit": {
                    "F_i": mem_result["F_i"],
                    "S_i": mem_result.get("S_i"),
                    "P_comp": mem_result["P_comp"],
                    "H_keep_size": mem_result.get("H_keep_size", 0),
                    "H_keep_covered": mem_result.get("H_keep_covered", 0),
                    "severe_failure": mem_result["severe_failure"],
                },
            })

    # ------------------------------------------------------------------
    # 5. Save audit files
    # ------------------------------------------------------------------
    print(f"[5/6] Writing audit outputs ...")

    ledger_path = os.path.join(output_dir, "ledger_audit.jsonl")
    write_jsonl(ledger_path, ledger_records)
    print(f"  Wrote {len(ledger_records)} ledger audit records to {ledger_path}")

    reward_path = os.path.join(output_dir, "reward_audit.jsonl")
    write_jsonl(reward_path, reward_records)
    print(f"  Wrote {len(reward_records)} reward audit records to {reward_path}")

    # ------------------------------------------------------------------
    # 6. Generate report
    # ------------------------------------------------------------------
    print(f"[6/6] Generating report ...")

    n_tool = len(all_tool_rewards)
    n_ans = len(all_answer_rewards)
    n_mem = len(all_memory_rewards)
    n_cov = len(all_coverages)

    report = _build_report(
        output_dir=output_dir,
        num_samples=len(common_ids),
        num_subquestions=len(ledger_records),
        total_tool_steps=total_tool_steps,
        total_invalid_steps=total_invalid_steps,
        total_repeat_steps=total_repeat_steps,
        evidence_stats=evidence_stats,
        total_deps=total_deps,
        all_tool_rewards=all_tool_rewards,
        all_answer_rewards=all_answer_rewards,
        all_memory_rewards=all_memory_rewards,
        all_coverages=all_coverages,
        warnings=warnings,
    )

    report_path = os.path.join(output_dir, "phase0_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Wrote report to {report_path}")

    return {"status": "ok", "report_path": report_path}


def _build_report(
    output_dir: str,
    num_samples: int,
    num_subquestions: int,
    total_tool_steps: int,
    total_invalid_steps: int,
    total_repeat_steps: int,
    evidence_stats: Dict[str, int],
    total_deps: int,
    all_tool_rewards: List[float],
    all_answer_rewards: List[float],
    all_memory_rewards: List[float],
    all_coverages: List[float],
    warnings: Counter,
) -> str:
    """Build the markdown report string."""

    def _stats(vals: List[float]) -> str:
        if not vals:
            return "N/A"
        return (
            f"min={min(vals):.3f}, max={max(vals):.3f}, "
            f"mean={sum(vals)/len(vals):.3f}, median={sorted(vals)[len(vals)//2]:.3f}"
        )

    lines = [
        "# TEMPO-RL Phase 0 Audit Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Output Directory**: `{output_dir}`",
        "",
        "## 1. Overview",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Processed samples | {num_samples} |",
        f"| Processed subquestions | {num_subquestions} |",
        f"| Total tool steps | {total_tool_steps} |",
        f"| Total evidence items | {evidence_stats.get('total_items', 0)} |",
        f"| Total future dependencies | {total_deps} |",
        "",
        "## 2. Evidence Statistics",
        "",
        f"| Type | Count |",
        f"|---|---|",
    ]

    for etype in ("raw_value", "derived_value", "text_fact"):
        cnt = evidence_stats.get(f"type_{etype}", 0)
        if cnt > 0:
            lines.append(f"| {etype} | {cnt} |")

    lines += [
        "",
        "## 3. Ledger Coverage",
        "",
        f"| Stat | Value |",
        f"|---|---|",
        f"| Coverage stats | {_stats(all_coverages)} |",
        f"| Mean coverage | {sum(all_coverages)/max(len(all_coverages),1):.1%} |",
        "",
        "## 4. Reward Statistics",
        "",
        f"| Component | Stats |",
        f"|---|---|",
        f"| r_tool (avg over {len(all_tool_rewards)} steps) | {_stats(all_tool_rewards)} |",
        f"| r_answer ({len(all_answer_rewards)} subquestions) | {_stats(all_answer_rewards)} |",
        f"| r_memory ({len(all_memory_rewards)} subquestions) | {_stats(all_memory_rewards)} |",
        "",
        "## 5. Tool Efficiency",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total tool calls | {total_tool_steps} |",
        f"| Invalid calls | {total_invalid_steps} |",
        f"| Repeat calls (no new evidence) | {total_repeat_steps} |",
        f"| Invalid rate | {total_invalid_steps/max(total_tool_steps,1):.1%} |",
        f"| Repeat rate | {total_repeat_steps/max(total_tool_steps,1):.1%} |",
        "",
        "## 6. Warnings & Issues",
        "",
    ]

    if warnings:
        lines.append("| Category | Count |")
        lines.append("|---|---|")
        for cat, cnt in warnings.most_common(15):
            lines.append(f"| {cat} | {cnt} |")
    else:
        lines.append("No warnings detected.")

    lines += [
        "",
        "## 7. Output Files",
        "",
        f"| File | Description |",
        f"|---|---|",
        f"| `target_evidence.jsonl` | Target evidence sets per subquestion |",
        f"| `future_dependencies.jsonl` | Future dependency sets per memory boundary |",
        f"| `ledger_audit.jsonl` | Per-subquestion ledger state and coverage |",
        f"| `reward_audit.jsonl` | Per-subquestion reward breakdowns |",
        f"| `phase0_report.md` | This report |",
    ]

    return "\n".join(lines)


# ======================================================================
# Validation
# ======================================================================

def validate_outputs(output_dir: str) -> Dict[str, Any]:
    """Validate all output files for structural correctness.

    Returns a dict with validation results.
    """
    results: Dict[str, Any] = {"pass": True, "checks": []}

    def _check(desc: str, ok: bool, detail: str = ""):
        results["checks"].append({"check": desc, "pass": ok, "detail": detail})
        if not ok:
            results["pass"] = False

    # 1. All JSONL files parseable
    for fname in ["target_evidence.jsonl", "future_dependencies.jsonl",
                   "ledger_audit.jsonl", "reward_audit.jsonl"]:
        fpath = os.path.join(output_dir, fname)
        if not os.path.exists(fpath):
            _check(f"{fname} exists", False, "file not found")
            continue
        try:
            records = _load_jsonl(fpath)
            _check(f"{fname} parseable", True, f"{len(records)} records")
        except Exception as e:
            _check(f"{fname} parseable", False, str(e))

    # 2. Each subquestion has target evidence
    te_path = os.path.join(output_dir, "target_evidence.jsonl")
    tes_records = None
    if os.path.exists(te_path):
        try:
            tes_records = _load_jsonl(te_path)
            sq_ids = set()
            for r in tes_records:
                sq_ids.add((r.get("sample_id"), r.get("subquestion_id")))
            _check("target evidence per subquestion", len(sq_ids) > 0,
                   f"{len(sq_ids)} unique (sample, sq) pairs")
        except Exception as e:
            _check("target evidence per subquestion", False, str(e))

    # 3. Future dependencies don't reference future info
    fd_path = os.path.join(output_dir, "future_dependencies.jsonl")
    if os.path.exists(fd_path):
        try:
            fd_records = _load_jsonl(fd_path)
            bad_refs = 0
            for r in fd_records:
                boundary = r.get("boundary", "")
                # Extract sq number from boundary like "after_sq1"
                m = re.match(r"after_sq(\d+)", boundary)
                if m:
                    boundary_sq = int(m.group(1))
                    for dep in r.get("future_dependencies", []):
                        needed = dep.get("needed_by", "")
                        nm = re.match(r"sq(\d+)", needed)
                        if nm:
                            needed_sq = int(nm.group(1))
                            if needed_sq <= boundary_sq:
                                bad_refs += 1
            if bad_refs > 0:
                _check("future deps don't reference past", False,
                       f"{bad_refs} deps reference past or current sq")
            else:
                _check("future deps don't reference past", True)
        except Exception as e:
            _check("future deps don't reference past", False, str(e))

    # 4. Derived evidence has input_evidence_ids
    if tes_records is not None:
        derived_missing_inputs = 0
        for r in tes_records:
            for ei in r.get("evidence_items", []):
                if ei.get("type") == "derived_value":
                    if not ei.get("input_evidence_ids"):
                        derived_missing_inputs += 1
        if derived_missing_inputs > 0:
            _check("derived evidence has input_evidence_ids", False,
                   f"{derived_missing_inputs} derived items missing inputs")
        else:
            _check("derived evidence has input_evidence_ids", True)

    # 5. Reward audit contains r_tool / r_answer / r_memory
    ra_path = os.path.join(output_dir, "reward_audit.jsonl")
    if os.path.exists(ra_path):
        try:
            ra_records = _load_jsonl(ra_path)
            all_have_keys = all(
                all(k in r for k in ("r_tool", "r_answer", "r_memory"))
                for r in ra_records
            )
            _check("reward audit has r_tool/r_answer/r_memory", all_have_keys)
        except Exception as e:
            _check("reward audit has r_tool/r_answer/r_memory", False, str(e))

    # 6. phase0_report.md exists
    report_path = os.path.join(output_dir, "phase0_report.md")
    _check("phase0_report.md exists", os.path.exists(report_path))

    return results


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TEMPO-RL Phase 0 — Offline Audit Script"
    )
    parser.add_argument(
        "--samples", required=True,
        help="Path to benchmark samples JSON file"
    )
    parser.add_argument(
        "--sft", required=True,
        help="Path to SFT trajectories JSONL file (memory_verified_subquestions.jsonl)"
    )
    parser.add_argument(
        "--output_dir", default="TEMPO_RL/output/phase0_audit",
        help="Output directory for audit files (default: TEMPO_RL/output/phase0_audit)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Max number of samples to process (default: all)"
    )
    parser.add_argument(
        "--validate_only", action="store_true",
        help="Only validate existing outputs, don't rebuild"
    )
    args = parser.parse_args()

    if args.validate_only:
        print("=== Validating existing outputs ===")
        results = validate_outputs(args.output_dir)
        for c in results["checks"]:
            status = "PASS" if c["pass"] else "FAIL"
            print(f"  [{status}] {c['check']}: {c.get('detail', '')}")
        if results["pass"]:
            print("\nAll validations passed.")
        else:
            print("\nSome validations FAILED.")
        return 0 if results["pass"] else 1

    # Run full audit
    result = run_audit(
        samples_path=args.samples,
        sft_path=args.sft,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )

    if result.get("status") != "ok":
        print(f"ERROR: {result}")
        return 1

    # Validate outputs
    print("\n=== Validating outputs ===")
    validation = validate_outputs(args.output_dir)
    for c in validation["checks"]:
        status = "PASS" if c["pass"] else "FAIL"
        print(f"  [{status}] {c['check']}: {c.get('detail', '')}")

    if validation["pass"]:
        print("\nAll validations passed.")
    else:
        print("\nSome validations FAILED.")

    # Print report summary
    report_path = os.path.join(args.output_dir, "phase0_report.md")
    if os.path.exists(report_path):
        print(f"\n=== Report ===\n")
        with open(report_path, "r", encoding="utf-8") as f:
            print(f.read())

    return 0 if validation["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
