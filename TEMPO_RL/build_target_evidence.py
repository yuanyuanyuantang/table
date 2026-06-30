"""
TEMPO-RL Phase 0 — Target Evidence Builder.

Constructs ``target_evidence.jsonl`` from benchmark samples.

Strategy (rule-first, LLM-fallback)
------------------------------------
1. For each score_point in each subquestion, *at least one* EvidenceItem is
   produced — score points are never silently dropped.
2. Rule-based extraction detects numeric values, times, units, percentages,
   and computation keywords.  Items that cannot be parsed numerically become
   ``text_fact`` items carrying the full score_point text.
3. When an LLM client is available and ``llm_enabled=True``, the LLM is called
   to annotate entity / metric / time / unit / operation.  Rule extraction
   still runs first; LLM results enrich or supplement.
4. ``related_tables`` are carried into ``source_tables`` on every item.
5. Every item carries an ``audit`` block with parse confidence and warnings.

Usage::

    from TEMPO_RL.build_target_evidence import TargetEvidenceBuilder
    from TEMPO_RL.io_utils import load_benchmark_samples

    samples = load_benchmark_samples("dataset/val.json")
    builder = TargetEvidenceBuilder()
    evidence_sets = builder.build(samples)
    builder.save(evidence_sets, "TEMPO_RL/output/target_evidence.jsonl")
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .schemas import AuditInfo, EvidenceItem, TargetEvidenceSet
from .io_utils import write_jsonl, read_jsonl, get_sample_id


# ======================================================================
# Rule-based extraction helpers
# ======================================================================

# -- Numeric value with compound unit (longer units first) --
_RE_NUMERIC = re.compile(
    r"(\d+(?:[.,]\d+)*\s*(?:万元/公里|元/吨|元/公斤|"
    r"%|万元|亿元|美元|欧元|人民币|"
    r"辆|台|吨|公斤|克|公里|米|厘米|毫米|小时|分钟|秒|次|倍|个|人)?)",
)

# -- Percentage (captures the number only) --
_RE_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")

# -- Time expressions --
_RE_TIME = re.compile(
    r"(\d{4}\s*年\s*(?:\d{1,2}\s*月(?:\s*(?:上旬|中旬|下旬|上半月|下半月|初|底|末))?)?"
    r"|\d{1,2}\s*月(?:\s*(?:上旬|中旬|下旬|上半月|下半月|初|底|末))?"
    r"|\d{4}\s*年第[一二三四季度]"
    r"|\d{4}年(?:\d{1,2}月)?\d{1,2}日)",
)

# -- Computation / derivation context patterns --
# Match score_points that *describe a computation to be performed* (as opposed
# to merely stating a value).  "增长率为16.96%" is a raw value; "需根据
# 当月值和去年值计算同比增长率" is derived.
_COMPUTE_CONTEXT = re.compile(
    r"(?:需|需要|须|应|可|请).{0,15}(?:计算|算出|求和|求平均|推导|换算|折算)"
    r"|(?:计算|算出|求和|求平均|换算|折算).{0,15}(?:得出|得到|获得|的|后|来)"
    r"|(?:根据|基于|利用|通过).{0,20}(?:计算|算出|求和|求平均|换算|折算)"
    r"|[计核]算\s*(?:同比|环比|增长|平均|占比|总|差|加权)"
    r"|求\s*(?:和|平均值|平均|差|比值)"
)


def extract_numeric_values(text: str) -> List[str]:
    """Return all numeric-value-with-unit strings found in *text*.

    >>> extract_numeric_values("产量为16.96%，出口18.46%")
    ['16.96%', '18.46%']
    """
    # Use a fresh findall to avoid overlapping
    seen: set = set()
    results: List[str] = []
    for m in _RE_NUMERIC.finditer(text):
        val = m.group(0).replace(",", "").replace("，", "").strip()
        if val and val not in seen:
            seen.add(val)
            results.append(val)
    return results


def extract_percentage_values(text: str) -> List[str]:
    """Return percentage strings found in *text*.

    >>> extract_percentage_values("增长16.96%和18.46%")
    ['16.96%', '18.46%']
    """
    return [m.group(0) for m in _RE_PCT.finditer(text)]


def extract_time(text: str) -> Optional[str]:
    """Extract the first time expression from *text*.

    >>> extract_time("2010年1月德国汽车市场概况")
    '2010年1月'
    """
    m = _RE_TIME.search(text)
    return m.group(0) if m else None


def detect_computation(text: str) -> Tuple[bool, Optional[str]]:
    """Check whether *text* describes a derived / computed value.

    Returns (is_computed, operation_keyword_or_None).

    >>> detect_computation("需根据当月值和去年值计算同比增长率")
    (True, '需根据当月值和去年值计算')
    >>> detect_computation("乘用车产量同比增长率为16.96%")
    (False, None)
    """
    m = _COMPUTE_CONTEXT.search(text)
    if m:
        return True, m.group(0)
    return False, None


def strip_prefix_number(s: str) -> str:
    """Remove leading digits, dots, and whitespace from a string.

    Used to extract unit from "16.96%" → "%".

    >>> strip_prefix_number("16.96%")
    '%'
    """
    return re.sub(r"^[\d.,\s]+", "", s)


# ======================================================================
# LLM-assisted extraction  (placeholder interface)
# ======================================================================

_EVIDENCE_LLM_PROMPT = """\
You are a data annotation expert. Extract structured evidence items from a scoring point.

Scoring point: {score_point}
Related tables: {related_tables}
Context (question): {question}

For each distinct fact in the scoring point, output one evidence item with:
- "type": "raw_value" (direct numeric fact), "derived_value" (computed), or "text_fact" (qualitative claim)
- "value": the fact string (e.g. "16.96%", or the full claim for text_fact)
- "entity": entity name (company, product, project, ...) or null
- "time": time expression or null
- "metric": what is being measured or null
- "unit": measurement unit or null
- "operation": for derived_value, the computation name or null

Output a JSON object: {{"evidence_items": [{{"type": "...", "value": "...", ...}}, ...]}}
"""


def _parse_llm_response(raw: str) -> List[Dict[str, Any]]:
    """Parse LLM JSON response to list of item dicts.  Returns [] on failure."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                return []
        else:
            return []
    if isinstance(data, dict):
        items = data.get("evidence_items", [])
        if isinstance(items, list):
            return items
    return []


def _call_llm_extract(
    score_points: List[str],
    related_tables: List[str],
    question: str,
    llm_client: Any,
) -> List[List[Dict[str, Any]]]:
    """Call LLM to extract evidence items from each score point.

    Returns one list of item-dicts per score_point.
    """
    rt_str = ", ".join(related_tables) if related_tables else "(none)"
    prompts = [
        _EVIDENCE_LLM_PROMPT.format(
            score_point=sp, related_tables=rt_str, question=question
        )
        for sp in score_points
    ]
    try:
        responses = llm_client.batch_chat(
            prompts=prompts,
            system="You are a rigorous data annotation expert.",
            temperature=0.0,
            response_format={"type": "json_object"},
            threads=5,
            batch_size=10,
        )
    except Exception:
        return [[] for _ in score_points]

    return [_parse_llm_response(r.get("content", "")) for r in responses]


# ======================================================================
# Target Evidence Builder
# ======================================================================

class TargetEvidenceBuilder:
    """Build ``TargetEvidenceSet`` objects from benchmark samples.

    Parameters
    ----------
    llm_client : optional
        A ``ChatClient``-compatible object for LLM-assisted annotation.
    llm_enabled : bool
        Whether to actually call the LLM (default False — rule-only).
    """

    def __init__(self, llm_client: Any = None, llm_enabled: bool = False):
        self._llm = llm_client
        self._llm_enabled = llm_enabled and llm_client is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, benchmark_samples: List[Dict[str, Any]]) -> List[TargetEvidenceSet]:
        """Build target evidence for every subquestion in every sample.

        Returns a flat list with one ``TargetEvidenceSet`` per subquestion.
        """
        all_sets: List[TargetEvidenceSet] = []
        for sample in benchmark_samples:
            all_sets.extend(self.build_one_sample(sample))
        return all_sets

    def build_one_sample(self, sample: Dict[str, Any]) -> List[TargetEvidenceSet]:
        """Build target evidence for all subquestions in one benchmark sample."""
        sample_id = get_sample_id(sample)
        checkout_list = sample.get("design", {}).get("checkout_list", [])

        # Collect all score_points for potential LLM batch call
        all_sps: List[Tuple[int, str]] = []  # (sq_idx, score_point)
        all_rt: List[List[str]] = []
        sq_map: Dict[int, Dict[str, Any]] = {}  # sq_idx -> checkout item

        for item in checkout_list:
            idx = item.get("idx", 1)
            sq_id = idx  # idx is 1-based in benchmark data
            sq_map[sq_id] = item
            for sp in item.get("score_points", []):
                all_sps.append((sq_id, sp))
                all_rt.append(item.get("related_tables", []))

        # LLM batch call (if enabled) — call once per question
        llm_results: Dict[int, Dict[int, List[Dict[str, Any]]]] = {}  # sq_id -> sp_idx -> items
        if self._llm_enabled:
            for sq_id, item in sq_map.items():
                sps = item.get("score_points", [])
                rts = item.get("related_tables", [])
                q = item.get("info_item", "")
                if sps:
                    results = _call_llm_extract(sps, rts, q, self._llm)
                    llm_results[sq_id] = {i: r for i, r in enumerate(results)}

        # Build per subquestion
        sets: List[TargetEvidenceSet] = []
        for item in checkout_list:
            idx = item.get("idx", 1)
            sq_id = idx  # idx is 1-based in benchmark data
            question = item.get("info_item", "")
            score_points = item.get("score_points", [])
            related_tables = item.get("related_tables", [])

            evidence_items, sp_coverage = self._extract_subquestion(
                sample_id=sample_id,
                subquestion_id=sq_id,
                question=question,
                score_points=score_points,
                related_tables=related_tables,
                llm_sp_results=llm_results.get(sq_id, {}),
            )

            # Gate: verify every score_point produced at least one item
            for spi, sp in enumerate(score_points):
                if not sp_coverage.get(spi, False):
                    # Force-create a text_fact item so nothing is dropped
                    evidence_items.append(
                        EvidenceItem(
                            sample_id=sample_id,
                            subquestion_id=sq_id,
                            evidence_id=f"sq{sq_id}_e{len(evidence_items) + 1}",
                            type="text_fact",
                            value=sp,
                            source_tables=list(related_tables),
                            audit=AuditInfo(
                                parse_confidence=0.05,
                                warnings=["fallback: score point could not be parsed — kept as text_fact"],
                                source="rule_extraction",
                            ),
                        )
                    )
                    sp_coverage[spi] = True

            sets.append(
                TargetEvidenceSet(
                    sample_id=sample_id,
                    subquestion_id=sq_id,
                    question=question,
                    evidence_items=evidence_items,
                )
            )

        # Post-process: link derived_value items to source raw_value items
        self._link_derived_inputs(sets)

        return sets

    # ------------------------------------------------------------------
    # Per-subquestion extraction
    # ------------------------------------------------------------------

    def _extract_subquestion(
        self,
        sample_id: str,
        subquestion_id: int,
        question: str,
        score_points: List[str],
        related_tables: List[str],
        llm_sp_results: Dict[int, List[Dict[str, Any]]],
    ) -> Tuple[List[EvidenceItem], Dict[int, bool]]:
        """Extract evidence items for one subquestion.

        Returns (items, sp_coverage) where sp_coverage maps score_point index → covered.
        """
        items: List[EvidenceItem] = []
        sp_coverage: Dict[int, bool] = {}
        counter = 1  # for generating evidence_id

        for spi, sp in enumerate(score_points):
            sp_items = self._extract_one_score_point(
                sample_id=sample_id,
                subquestion_id=subquestion_id,
                question=question,
                score_point=sp,
                score_point_index=spi,
                related_tables=related_tables,
                counter_start=counter,
                llm_items=llm_sp_results.get(spi, []),
            )
            if sp_items:
                sp_coverage[spi] = True
                for it in sp_items:
                    it.evidence_id = f"sq{subquestion_id}_e{counter}"
                    counter += 1
                items.extend(sp_items)
            else:
                sp_coverage[spi] = False

        return items, sp_coverage

    def _extract_one_score_point(
        self,
        sample_id: str,
        subquestion_id: int,
        question: str,
        score_point: str,
        score_point_index: int,
        related_tables: List[str],
        counter_start: int,
        llm_items: List[Dict[str, Any]],
    ) -> List[EvidenceItem]:
        """Extract evidence items from a single score_point.

        Rule extraction runs first.  If *llm_items* is non-empty, LLM results
        are used to enrich (or replace) rule results.
        """
        # ---- Rule extraction ----
        rule_items = self._rule_extract(
            sample_id, subquestion_id, score_point, related_tables, counter_start
        )

        # ---- LLM enrichment (if available) ----
        if llm_items:
            return self._merge_llm_results(
                rule_items, llm_items,
                sample_id, subquestion_id, related_tables, counter_start,
            )

        return rule_items

    # ------------------------------------------------------------------
    # Rule extraction
    # ------------------------------------------------------------------

    def _rule_extract(
        self,
        sample_id: str,
        subquestion_id: int,
        score_point: str,
        related_tables: List[str],
        counter_start: int,
    ) -> List[EvidenceItem]:
        """Rule-based extraction from one score_point.

        Never returns an empty list — if nothing can be parsed numerically,
        a single ``text_fact`` item is returned.
        """
        source_tables = list(related_tables)
        warnings: List[str] = []

        # Detect computation keywords
        is_comp, comp_op = detect_computation(score_point)

        # Extract numeric values
        num_vals = extract_numeric_values(score_point)
        # Also try percentage-only
        pct_vals = extract_percentage_values(score_point)

        # Merge and deduplicate
        all_vals: List[str] = []
        seen: set = set()
        for v in num_vals + pct_vals:
            if v not in seen:
                seen.add(v)
                all_vals.append(v)

        time = extract_time(score_point)
        if not time:
            # also check the question text?
            pass

        # Build items
        items: List[EvidenceItem] = []

        if all_vals:
            for val in all_vals:
                unit = strip_prefix_number(val)

                w: List[str] = list(warnings)
                confidence = 0.7

                if not time:
                    w.append("time not detected by rule")
                    confidence -= 0.15
                # entity / metric are almost never parsable by rule
                w.append("entity not detected by rule")
                w.append("metric not detected by rule")
                confidence -= 0.1

                etype = "derived_value" if is_comp else "raw_value"

                items.append(EvidenceItem(
                    sample_id=sample_id,
                    subquestion_id=subquestion_id,
                    evidence_id="",  # filled by caller
                    type=etype,
                    value=val,
                    entity=None,
                    time=time,
                    metric=None,
                    unit=unit if unit != val else None,
                    source_tables=list(source_tables),
                    operation=comp_op if is_comp else None,
                    weight=1.0,
                    audit=AuditInfo(
                        parse_confidence=max(0.0, min(1.0, confidence)),
                        warnings=w,
                        source="rule_extraction",
                    ),
                ))

        # If no numeric values found → text_fact for the whole score_point
        if not items:
            w = list(warnings)
            w.append("no numeric values detected; stored as text_fact")
            items.append(EvidenceItem(
                sample_id=sample_id,
                subquestion_id=subquestion_id,
                evidence_id="",
                type="text_fact",
                value=score_point,
                source_tables=list(source_tables),
                audit=AuditInfo(
                    parse_confidence=0.3,
                    warnings=w,
                    source="rule_extraction",
                ),
            ))

        return items

    # ------------------------------------------------------------------
    # LLM merge
    # ------------------------------------------------------------------

    def _merge_llm_results(
        self,
        rule_items: List[EvidenceItem],
        llm_items: List[Dict[str, Any]],
        sample_id: str,
        subquestion_id: int,
        related_tables: List[str],
        counter_start: int,
    ) -> List[EvidenceItem]:
        """Merge LLM-extracted items with rule results.

        Strategy: if LLM produced items, prefer LLM (it has better entity/
        metric/time annotation).  If LLM failed to produce anything, keep
        rule items.
        """
        if not llm_items:
            return rule_items

        source_tables = list(related_tables)
        merged: List[EvidenceItem] = []

        for li in llm_items:
            etype = li.get("type", "raw_value")
            if etype not in ("raw_value", "derived_value", "text_fact"):
                etype = "text_fact"

            warnings: List[str] = []
            confidence = 0.75  # base for LLM

            # Check for missing fields
            for fld in ("entity", "time", "metric", "unit"):
                if not li.get(fld):
                    warnings.append(f"{fld} not annotated by LLM")
                    confidence -= 0.05

            if etype == "derived_value" and not li.get("operation"):
                warnings.append("derived_value missing operation")
                confidence -= 0.1

            merged.append(EvidenceItem(
                sample_id=sample_id,
                subquestion_id=subquestion_id,
                evidence_id="",
                type=etype,
                value=str(li.get("value", "")),
                entity=li.get("entity"),
                time=li.get("time"),
                metric=li.get("metric"),
                unit=li.get("unit"),
                source_tables=list(source_tables),
                operation=li.get("operation"),
                weight=float(li.get("weight", 1.0)),
                audit=AuditInfo(
                    parse_confidence=max(0.0, min(1.0, confidence)),
                    warnings=warnings,
                    source="llm_annotation",
                ),
            ))

        if not merged:
            return rule_items  # LLM produced nothing useful
        return merged

    # ------------------------------------------------------------------
    # Derived-value linking
    # ------------------------------------------------------------------

    @staticmethod
    def _link_derived_inputs(evidence_sets: List[TargetEvidenceSet]) -> None:
        """Link derived_value items to source raw_value items by entity/time.

        Called after all evidence items for a sample have been built and
        assigned ``evidence_id`` values.  Modifies items **in place**.

        Matching strategy (first non-empty tier wins per derived item):

        1. Same entity **and** same time (exact).
        2. Same entity (relaxed).
        3. Same time (looser).
        4. All raw_value items in the sample (conservative fallback).
        """
        # Collect all raw_value items across subquestions: (evidence_id, entity, time)
        raw_refs: List[Tuple[str, Optional[str], Optional[str]]] = []
        for es in evidence_sets:
            for item in es.evidence_items:
                if item.type == "raw_value":
                    raw_refs.append((item.evidence_id, item.entity, item.time))

        if not raw_refs:
            return  # nothing to link against

        for es in evidence_sets:
            for item in es.evidence_items:
                if item.type != "derived_value":
                    continue
                # Only auto-link when input_evidence_ids is empty
                if item.input_evidence_ids:
                    continue

                candidates: List[str] = []

                # Tier 1 — entity + time exact match
                if item.entity and item.time:
                    candidates = [
                        eid for eid, ent, t in raw_refs
                        if ent == item.entity and t == item.time
                    ]

                # Tier 2 — entity match
                if not candidates and item.entity:
                    candidates = [
                        eid for eid, ent, t in raw_refs
                        if ent == item.entity
                    ]

                # Tier 3 — time match
                if not candidates and item.time:
                    candidates = [
                        eid for eid, ent, t in raw_refs
                        if t == item.time
                    ]

                # Tier 4 — all raw_value items (fallback)
                if not candidates:
                    candidates = [eid for eid, _, _ in raw_refs]

                item.input_evidence_ids = sorted(set(candidates))
                item.audit.warnings.append(
                    f"auto-linked to {len(item.input_evidence_ids)} raw_value source(s)"
                )

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @staticmethod
    def save(
        sets: List[TargetEvidenceSet],
        output_path: str,
    ) -> None:
        """Write target evidence sets to a JSONL file."""
        records = [s.to_dict() for s in sets]
        write_jsonl(output_path, records)

    @staticmethod
    def load(input_path: str) -> List[TargetEvidenceSet]:
        """Read target evidence sets from a JSONL file."""
        records = read_jsonl(input_path)
        return [TargetEvidenceSet.from_dict(r) for r in records]
