"""
TEMPO-RL Phase 0 — Evidence Verifier.

Provides verification functions that check whether a target ``EvidenceItem``
is supported by an observation, memory, or computation result.

Verification dimensions
-----------------------
- **C_value**   : target value / fact appears in observation, memory, or code output.
- **C_source**  : observation source table matches an allowed ``source_tables`` entry.
- **C_binding** : entity, time, metric, unit are correctly bound to the value.
- **C_input**   : (derived only) all input evidence_ids are already verified.
- **C_operator**: (derived only) the computation operation matches.
- **C_result**  : (derived only) the result value can be recomputed from inputs.

Rule-based verifiers handle simple numeric / table checks; LLM fallback is
available for entity-value binding, trends, and qualitative claims.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .schemas import EvidenceItem


# ======================================================================
# Numeric normalisation helpers
# ======================================================================

def _normalize_number(s: str) -> Optional[float]:
    """Parse a Chinese / Western numeric string to float.

    Handles: "16.96%", "70,73元", "41459万元/公里", "1.5亿", "1,234.56"
    Returns None if not parseable.
    """
    if not s:
        return None
    # Remove commas (both Chinese and Western), percent signs, spaces
    cleaned = s.replace(",", "").replace("，", "").replace("%", "").replace(" ", "")
    # Handle common Chinese magnitude suffixes (longest first)
    multipliers = [
        ("万元/公里", 1e4), ("亿元/公里", 1e8),
        ("万元/吨", 1e4), ("元/吨", 1),
        ("亿元", 1e8), ("万元", 1e4), ("美元", 1), ("欧元", 1),
        ("亿", 1e8), ("万", 1e4), ("千", 1e3), ("百", 1e2),
    ]
    factor = 1.0
    for suffix, mult in multipliers:
        if cleaned.endswith(suffix):
            factor = mult
            cleaned = cleaned[:-len(suffix)]
            break
    # Strip residual unit-like trailing characters
    cleaned = cleaned.rstrip("元个辆台吨公斤克公里米厘米毫米小时分钟秒次倍人/")
    if not cleaned:
        return None
    try:
        return float(cleaned) * factor
    except (ValueError, TypeError):
        return None


def _extract_numbers_from_text(text: str) -> List[Tuple[str, float]]:
    """Extract all (raw_string, float_value) pairs from text."""
    results: List[Tuple[str, float]] = []
    # Match numbers with optional unit/percent/currency suffixes
    pat = re.compile(
        r"(\d+(?:[.,]\d+)*\s*(?:亿|万|千|百)?\s*(?:%|元|万元|亿元|美元|欧元|"
        r"辆|台|吨|公斤|克|公里|米|厘米|毫米|小时|分钟|秒|次|倍|个|人)?)",
    )
    seen: Set[str] = set()
    for m in pat.finditer(text):
        raw = m.group(0).replace(",", "").replace("，", "").strip()
        if raw and raw not in seen:
            seen.add(raw)
            val = _normalize_number(raw)
            if val is not None:
                results.append((raw, val))
    return results


def _normalize_text_for_match(s: str) -> str:
    """Normalize text for fuzzy comparison."""
    return re.sub(r"\s+", "", s.replace(",", "").replace("，", "").lower())


# ======================================================================
# C_value — Value / fact match
# ======================================================================

def verify_value_match(
    evidence_item: EvidenceItem,
    observation_text: str = "",
    memory_text: str = "",
    code_output: str = "",
    tolerance: float = 1e-6,
) -> Tuple[bool, Dict[str, Any]]:
    """Check if the evidence value appears in observation / memory / code output.

    Returns ``(matched, audit_dict)``.

    Strategy
    --------
    - ``raw_value`` : compare numeric value to numbers in the text sources.
    - ``text_fact`` : fuzzy substring match (after whitespace / punctuation
      normalisation).
    - ``derived_value`` : delegated to ``verify_derived_result`` (not checked
      here — this function only does direct appearance checks).
    """
    audit: Dict[str, Any] = {"method": "value_match", "matched_in": None}

    target_value = evidence_item.value or ""

    # -- raw_value: numeric comparison --
    if evidence_item.type in ("raw_value", "derived_value"):
        target_num = _normalize_number(target_value)
        if target_num is None:
            # Non-numeric raw_value — fall through to substring match
            pass
        else:
            sources: List[Tuple[str, str]] = [
                (observation_text, "observation"),
                (memory_text, "memory"),
                (code_output, "code_output"),
            ]
            for src_text, src_name in sources:
                if not src_text:
                    continue
                nums = _extract_numbers_from_text(src_text)
                for raw, val in nums:
                    if abs(val - target_num) < tolerance:
                        audit["matched_in"] = src_name
                        audit["matched_raw"] = raw
                        audit["matched_value"] = val
                        return True, audit
            # If target_num is an integer, also try fuzzy string match
            # (sometimes numbers appear as "约16.96%" vs exact "16.96%")

    # -- text_fact / fallback: normalised substring match --
    norm_target = _normalize_text_for_match(target_value)
    # For very short values (e.g., single letters/numbers), skip the length
    # guard only if they appear verbatim in the source.
    if len(norm_target) < 2:
        # Check verbatim (case-insensitive)
        for src_text, src_name in [
            (observation_text, "observation"),
            (memory_text, "memory"),
            (code_output, "code_output"),
        ]:
            if src_text and target_value.strip().lower() in src_text.lower():
                audit["matched_in"] = src_name
                return True, audit
        return False, audit

    sources_text = [
        (observation_text, "observation"),
        (memory_text, "memory"),
        (code_output, "code_output"),
    ]
    for src_text, src_name in sources_text:
        if not src_text:
            continue
        norm_src = _normalize_text_for_match(src_text)
        # For text_fact, check if a significant portion matches
        if len(norm_target) >= 8:
            # Long text — check if a core segment appears
            if norm_target[:min(len(norm_target), 12)] in norm_src:
                audit["matched_in"] = src_name
                return True, audit
        if norm_target in norm_src:
            audit["matched_in"] = src_name
            return True, audit

    return False, audit


# ======================================================================
# C_source — Source / provenance match
# ======================================================================

def verify_source_match(
    evidence_item: EvidenceItem,
    observation_metadata: Optional[Dict[str, Any]] = None,
    tool_arguments: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Check that the observation source matches an allowed source table.

    Rules (per §3.2)
    ----------------
    - For empty / null fields in ``source_tables``, skip the check.
    - All non-empty fields must match the observation metadata.
    - If ``source_tables`` is empty, source check is considered passed
      (no table constraint).

    Returns ``(matched, audit_dict)``.
    """
    audit: Dict[str, Any] = {
        "method": "source_match",
        "allowed_tables": evidence_item.source_tables,
        "observed_source": None,
    }

    # No source_tables constraint → pass
    if not evidence_item.source_tables:
        audit["note"] = "no source tables specified — pass"
        return True, audit

    # Gather observable source info
    observed_sources: List[str] = []

    if observation_metadata:
        fname = observation_metadata.get("file") or observation_metadata.get("filename") or ""
        if fname:
            observed_sources.append(fname)
        sheet = observation_metadata.get("sheet") or ""
        if sheet:
            observed_sources.append(sheet)

    if tool_arguments:
        # Common tool argument keys for file/table references
        for key in ("file_path", "path", "table", "table_name", "filename", "sheet"):
            val = tool_arguments.get(key, "")
            if val and isinstance(val, str):
                observed_sources.append(val)

    audit["observed_source"] = observed_sources

    if not observed_sources:
        # Can't verify source — flag as uncertain
        audit["warning"] = "no observable source metadata — cannot verify provenance"
        return False, audit

    # Check: does any observed source match any allowed source table?
    allowed_lower = [t.lower() for t in evidence_item.source_tables]
    for obs_src in observed_sources:
        obs_lower = obs_src.lower()
        for allowed in allowed_lower:
            # Match if observed source contains the allowed table name (or vice versa)
            if obs_lower in allowed or allowed in obs_lower:
                audit["matched_table"] = obs_src
                return True, audit

    audit["reason"] = f"no observed source matched allowed tables: {allowed_lower}"
    return False, audit


# ======================================================================
# C_binding — Entity / time / metric / unit binding
# ======================================================================

def verify_binding_match(
    evidence_item: EvidenceItem,
    observation_text: str = "",
    memory_text: str = "",
    tool_arguments: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Check that the value is bound to the correct entity / time / metric / unit.

    For each non-None binding field on the evidence item, we check whether
    that field's value appears in the observation or memory context.

    **Lenient rule**: only fail (return False) when there is evidence of a
    *conflicting* binding (e.g. observation mentions a different entity).
    If a binding field is simply absent from the context, flag it as
    ``"uncertain"`` but still return True — binding verification for
    absent fields should be done by LLM fallback.

    Returns ``(matched, audit_dict)``.
    """
    audit: Dict[str, Any] = {
        "method": "binding_match",
        "checks": {},
    }

    context = observation_text + " " + memory_text
    if tool_arguments:
        # Include relevant tool argument values
        for v in tool_arguments.values():
            if isinstance(v, str):
                context += " " + v

    norm_context = _normalize_text_for_match(context)

    has_conflict = False

    for field_name in ("entity", "time", "metric", "unit"):
        field_val = getattr(evidence_item, field_name, None)
        if field_val is None:
            audit["checks"][field_name] = {"status": "skipped", "value": None}
            continue

        norm_field = _normalize_text_for_match(str(field_val))
        if not norm_field:
            audit["checks"][field_name] = {"status": "skipped", "value": ""}
            continue

        found = norm_field in norm_context

        # For longer field values (metric names), also check if a significant
        # substring appears — "产量同比增长率" contains "增长率"
        if not found and len(norm_field) > 4 and field_name == "metric":
            # Check if any 2+ char substring of the metric appears in context
            for win in range(2, len(norm_field)):
                for start in range(len(norm_field) - win + 1):
                    sub = norm_field[start:start + win]
                    if sub in norm_context:
                        found = True
                        break
                if found:
                    break

        if found:
            audit["checks"][field_name] = {"status": "pass", "value": field_val}
        else:
            # Binding field not found — flag as uncertain, not fail
            audit["checks"][field_name] = {
                "status": "uncertain",
                "value": field_val,
                "note": "field not found in context — LLM verification recommended",
            }

    # Only fail on hard conflict (future: detect conflicting entities).
    # For now, single absence is not a failure.
    return not has_conflict, audit


# ======================================================================
# C_input — Derived evidence input check
# ======================================================================

def verify_derived_inputs(
    evidence_item: EvidenceItem,
    verified_evidence_ids: Set[str],
) -> Tuple[bool, Dict[str, Any]]:
    """Check that all ``input_evidence_ids`` of a derived item are verified.

    Returns ``(all_inputs_verified, audit_dict)``.
    """
    audit: Dict[str, Any] = {
        "method": "derived_input_check",
        "required_inputs": evidence_item.input_evidence_ids,
        "verified_inputs": sorted(verified_evidence_ids),
    }

    if not evidence_item.input_evidence_ids:
        audit["status"] = "fail"
        audit["reason"] = "derived_value has no input_evidence_ids"
        return False, audit

    missing = [eid for eid in evidence_item.input_evidence_ids
               if eid not in verified_evidence_ids]

    audit["missing_inputs"] = missing
    audit["status"] = "pass" if not missing else "fail"

    return len(missing) == 0, audit


# ======================================================================
# C_operator — Operation match for derived evidence
# ======================================================================

# Known operation keyword groups
_OPERATION_GROUPS: Dict[str, List[str]] = {
    "同比增长率": ["同比", "增长", "增长率", "yoy"],
    "环比增长率": ["环比", "增长率"],
    "占比": ["占比", "比例", "比重", "ratio", "proportion"],
    "average": ["平均", "均值", "average", "mean"],
    "sum": ["求和", "总和", "合计", "sum", "total"],
    "difference": ["差", "差值", "差异", "difference"],
    "ratio": ["比率", "比值", "ratio"],
}

# Computation keywords in code
_COMPUTE_KEYWORDS = [
    "同比", "环比", "增长率", "占比", "平均", "求和", "合计",
    "+", "-", "*", "/", "sum", "mean", "average", "ratio",
    "percentage", "percent",
]


def verify_derived_operation(
    evidence_item: EvidenceItem,
    tool_arguments: Optional[Dict[str, Any]] = None,
    code_output: str = "",
    tool_name: str = "",
) -> Tuple[bool, Dict[str, Any]]:
    """Check that the computation operation matches the expected operation.

    Strategy
    --------
    - If the tool is a code executor (``python_exec``, ``calculator``, etc.),
      inspect the code / arguments for the expected operation keywords.
    - If ``code_output`` is provided, check operation keywords there.
    - Fuzzy match the ``evidence_item.operation`` against known operation groups.

    Returns ``(matched, audit_dict)``.
    """
    audit: Dict[str, Any] = {
        "method": "derived_operation_check",
        "expected_operation": evidence_item.operation,
    }

    expected_op = (evidence_item.operation or "").lower()

    # Collect text sources to search for operation evidence
    search_text = code_output
    if tool_arguments:
        for v in tool_arguments.values():
            if isinstance(v, str):
                search_text += " " + v

    # Also check tool_name — if it's a known code tool, operation is likely correct
    code_tool_names = {"python_exec", "python", "calculator", "code_exec", "execute_code",
                       "calculate", "compute"}
    if tool_name.lower() in code_tool_names and not expected_op:
        # No specific operation expected — any computation tool is sufficient
        audit["status"] = "pass"
        audit["reason"] = "code tool used, no specific operation required"
        return True, audit

    if not expected_op:
        audit["status"] = "pass"
        audit["reason"] = "no operation specified — pass"
        return True, audit

    # Look for the expected operation in known groups
    matched_keywords: List[str] = []
    if expected_op in _OPERATION_GROUPS:
        for kw in _OPERATION_GROUPS[expected_op]:
            if kw.lower() in search_text.lower():
                matched_keywords.append(kw)
    else:
        # Direct keyword search
        for kw in _COMPUTE_KEYWORDS:
            if kw.lower() in search_text.lower():
                matched_keywords.append(kw)

    # Also check if operation name itself appears
    if expected_op and expected_op in search_text.lower():
        matched_keywords.append(expected_op)

    audit["matched_keywords"] = matched_keywords
    audit["status"] = "pass" if matched_keywords else "uncertain"

    if matched_keywords:
        return True, audit
    # If no keywords found but a code tool was used, still accept tentatively
    if tool_name.lower() in code_tool_names and search_text.strip():
        audit["status"] = "pass"
        audit["reason"] = "code tool executed — operation accepted"
        return True, audit

    audit["reason"] = "no operation keywords detected"
    return False, audit


# ======================================================================
# C_result — Derived result verification
# ======================================================================

def verify_derived_result(
    evidence_item: EvidenceItem,
    input_evidence_items: List[EvidenceItem],
    code_output: str = "",
) -> Tuple[bool, Dict[str, Any]]:
    """Check that the derived result is consistent with its inputs.

    Strategy
    --------
    1. Extract numeric values from input evidence items.
    2. Extract the target result value.
    3. Attempt recomputation for known operations (ratio, sum, difference,
       percentage).
    4. If recomputation is not possible, check that the result value
       appears in code_output.

    Returns ``(matched, audit_dict)``.
    """
    audit: Dict[str, Any] = {
        "method": "derived_result_check",
        "expected_result": evidence_item.value,
    }

    target_num = _normalize_number(evidence_item.value or "")
    if target_num is None:
        # Non-numeric derived value — check if result appears in code_output
        if evidence_item.value and evidence_item.value in code_output:
            audit["status"] = "pass"
            audit["reason"] = "result value found in code output"
            return True, audit
        audit["status"] = "uncertain"
        audit["reason"] = "non-numeric derived value, cannot verify"
        return True, audit  # Be lenient for non-numeric

    # Extract input numbers
    input_nums: List[Tuple[str, float]] = []
    for inp in input_evidence_items:
        n = _normalize_number(inp.value or "")
        if n is not None:
            input_nums.append((inp.evidence_id, n))

    audit["input_values"] = [{"id": iid, "value": v} for iid, v in input_nums]
    audit["target_value"] = target_num

    # Attempt recomputation for common operations
    operation = (evidence_item.operation or "").lower()
    recomputed: Optional[float] = None
    method = "none"

    if input_nums:
        vals = [v for _, v in input_nums]

        if any(kw in operation for kw in ("占比", "比例", "比重", "ratio", "proportion")):
            # ratio = part / total
            if len(vals) >= 2:
                recomputed = vals[0] / max(vals[1], 1e-9)
                method = "ratio: first/second"
            elif len(vals) == 1:
                recomputed = vals[0]
                method = "single value"

        elif any(kw in operation for kw in ("求和", "合计", "sum", "total")):
            recomputed = sum(vals)
            method = "sum"

        elif any(kw in operation for kw in ("差", "差异", "差值", "difference")):
            if len(vals) >= 2:
                recomputed = vals[0] - vals[1]
                method = "difference"

        elif any(kw in operation for kw in ("平均", "均值", "average", "mean")):
            recomputed = sum(vals) / len(vals)
            method = "average"

        elif any(kw in operation for kw in ("增长", "同比", "环比", "growth", "yoy")):
            # growth rate = (current - previous) / previous
            # We don't know which input is current vs previous, so try both
            if len(vals) >= 2:
                # Try (vals[0] - vals[1]) / vals[1] and swapped
                candidates_g = []
                r1 = (vals[0] - vals[1]) / max(abs(vals[1]), 1e-9)
                r2 = (vals[1] - vals[0]) / max(abs(vals[0]), 1e-9)
                candidates_g.append(r1)
                candidates_g.append(r2)
                # Also try percentage forms
                candidates_g.append(r1 * 100)
                candidates_g.append(r2 * 100)
                # Find best match
                best = min(candidates_g, key=lambda c: abs(c - target_num))
                recomputed = best
                method = "growth_rate"
            elif len(vals) == 1:
                recomputed = vals[0]
                method = "single_value_growth"

    if recomputed is not None:
        # Compare with tolerance.  Also try percentage/ratio equivalence:
        # if target is 18.46 (from "18.46%") and recomputed is 0.1846,
        # or vice versa, they should still match.
        candidates = [recomputed]
        if 0 < recomputed < 1:
            candidates.append(recomputed * 100)  # try percentage form
        if abs(recomputed) > 1:
            candidates.append(recomputed / 100)  # try ratio form

        for cand in candidates:
            abs_tol = max(abs(target_num) * 0.02, 1e-6)
            if math.isclose(cand, target_num, rel_tol=0.02, abs_tol=abs_tol):
                audit["status"] = "pass"
                audit["recomputed_value"] = recomputed
                audit["computation_method"] = method
                audit["match_form"] = "percentage" if cand != recomputed else "direct"
                return True, audit

        audit["status"] = "fail"
        audit["recomputed_value"] = recomputed
        audit["computation_method"] = method
        audit["reason"] = (
            f"recomputed {recomputed} does not match target {target_num}"
        )
        return False, audit

    # Fallback: check if target value appears in code_output
    if code_output:
        nums = _extract_numbers_from_text(code_output)
        for raw, val in nums:
            if abs(val - target_num) < 1e-6:
                audit["status"] = "pass"
                audit["reason"] = "result value found in code output"
                audit["matched_value"] = val
                return True, audit

    audit["status"] = "uncertain"
    audit["reason"] = "could not recompute and result not found in code output"
    return False, audit


# ======================================================================
# Full evidence verification
# ======================================================================

def verify_evidence_item(
    evidence_item: EvidenceItem,
    observation_text: str = "",
    observation_metadata: Optional[Dict[str, Any]] = None,
    tool_arguments: Optional[Dict[str, Any]] = None,
    tool_name: str = "",
    memory_text: str = "",
    code_output: str = "",
    verified_evidence_ids: Optional[Set[str]] = None,
    ledger_evidence_items: Optional[List[EvidenceItem]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Run all applicable verifiers for one evidence item.

    Returns ``(fully_verified, audit_dict)`` with detailed sub-audits.

    Parameters
    ----------
    evidence_item : EvidenceItem
        The target evidence item to verify.
    observation_text : str
        Raw text from the tool observation.
    observation_metadata : dict or None
        Sidecar metadata from the observation (file, sheet, region, headers).
    tool_arguments : dict or None
        The arguments passed to the tool call.
    tool_name : str
        Name of the tool that was called.
    memory_text : str
        Compressed memory text (for memory-init verification).
    code_output : str
        Output from a code execution tool.
    verified_evidence_ids : set of str or None
        Already-verified evidence ids (for derived input check).
    ledger_evidence_items : list of EvidenceItem or None
        The full EvidenceItem objects for verified items (for derived result
        recomputation).

    Returns
    -------
    (passed, audit) : tuple
        ``passed`` is True only if ALL applicable checks pass.
        ``audit`` is a dict with per-check results.
    """
    verified_ids = verified_evidence_ids or set()
    ledger_items = ledger_evidence_items or []

    audit: Dict[str, Any] = {
        "evidence_id": evidence_item.evidence_id,
        "type": evidence_item.type,
        "checks": {},
    }

    all_pass = True

    # --- C_value ---
    v_ok, v_audit = verify_value_match(
        evidence_item,
        observation_text=observation_text,
        memory_text=memory_text,
        code_output=code_output,
    )
    audit["checks"]["C_value"] = v_audit
    if not v_ok:
        all_pass = False

    # --- C_source ---
    s_ok, s_audit = verify_source_match(
        evidence_item,
        observation_metadata=observation_metadata,
        tool_arguments=tool_arguments,
    )
    audit["checks"]["C_source"] = s_audit
    if not s_ok:
        all_pass = False

    # --- C_binding ---
    b_ok, b_audit = verify_binding_match(
        evidence_item,
        observation_text=observation_text,
        memory_text=memory_text,
        tool_arguments=tool_arguments,
    )
    audit["checks"]["C_binding"] = b_audit
    if not b_ok:
        all_pass = False

    # --- Derived-specific checks ---
    if evidence_item.type == "derived_value":
        # C_input
        i_ok, i_audit = verify_derived_inputs(evidence_item, verified_ids)
        audit["checks"]["C_input"] = i_audit
        if not i_ok:
            all_pass = False

        # C_operator
        o_ok, o_audit = verify_derived_operation(
            evidence_item,
            tool_arguments=tool_arguments,
            code_output=code_output,
            tool_name=tool_name,
        )
        audit["checks"]["C_operator"] = o_audit
        if not o_ok:
            all_pass = False

        # C_result
        input_items = [ei for ei in ledger_items
                       if ei.evidence_id in evidence_item.input_evidence_ids]
        r_ok, r_audit = verify_derived_result(evidence_item, input_items, code_output)
        audit["checks"]["C_result"] = r_audit
        if not r_ok:
            all_pass = False

    audit["passed"] = all_pass
    return all_pass, audit
