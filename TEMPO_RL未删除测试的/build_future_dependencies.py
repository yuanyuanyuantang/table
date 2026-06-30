"""
TEMPO-RL Phase 0 — Future Dependency Builder.

Constructs ``future_dependencies.jsonl`` from benchmark samples and
target evidence, identifying what information from earlier subquestions
is needed by later subquestions (within topological distance D_FDC ≤ 2).

Strategy (rule-first, LLM-fallback)
------------------------------------
1. For each memory boundary ``after_sq_i``, examine future subquestions
   within D_FDC steps.
2. Rule-based detectors identify five dependency types:
   - **numeric_fact** : entity/time/metric/value overlap between sqs
   - **entity_set**   : entity names accumulated across sqs
   - **reference**    : referring expressions ("该公司", "前者", "以上分析"...)
   - **constraint**   : inherited constraints (year, region, scope)
   - **table_ref**    : shared table files between sqs
3. Each dependency links to ``target_evidence.jsonl`` via ``source_evidence_id``
   when the supporting evidence can be identified.
4. Dependencies whose supporting evidence was NOT available at boundary *i*
   are filtered out (H_i^{keep} rule).
5. LLM placeholder interface for enrichment.

Usage::

    from TEMPO_RL.build_future_dependencies import FutureDependencyBuilder
    from TEMPO_RL.build_target_evidence import TargetEvidenceBuilder
    from TEMPO_RL.io_utils import load_benchmark_samples

    samples = load_benchmark_samples("dataset/val.json")
    target_sets = TargetEvidenceBuilder.load("TEMPO_RL/output/target_evidence.jsonl")
    builder = FutureDependencyBuilder()
    dep_sets = builder.build(samples, target_sets)
    builder.save(dep_sets, "TEMPO_RL/output/future_dependencies.jsonl")
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .schemas import (
    AuditInfo,
    EvidenceItem,
    FutureDependency,
    FutureDependencySet,
    TargetEvidenceSet,
    required_fields_for_type,
)
from .io_utils import write_jsonl, read_jsonl, get_sample_id


# ======================================================================
# Referring-expression patterns (Chinese)
# ======================================================================

# Patterns that signal a later sq is *referring back* to content from an
# earlier sq.  Each pattern maps to a reference type hint.
_REFERRING_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Demonstrative + noun: "该公司", "该企业", "该产品", "该单位", "该部门"
    (re.compile(r"该(?:公司|企业|产品|单位|部门|项目|案例|方案|渠道|数据|指标|模式|系统|环节|阶段|周期)"), "entity_ref"),
    # "前者" / "后者"  — ordinal reference
    (re.compile(r"前者|后者"), "ordinal_ref"),
    # Summarising: "综合以上分析", "综上所述", "综上", "基于上述", "基于以上"
    (re.compile(r"综合(?:以上|上述)|综上所述|综上(?:所述)?|基于(?:上述|以上|此前)"), "summary_ref"),
    # Collective: "这些数据", "这几个", "这两个类别", "这三个案例", "以上"
    (re.compile(r"这(?:些|几个|两个|三个|四个|五个)(?:案例|方案|项目|产品|渠道|类别|数据)?|以上(?:数据|分析|案例|方案)?"), "collective_ref"),
    # Explicit cross-reference with named cases: "案例A", "方案B", etc.
    (re.compile(r"(?:案例|方案|项目|渠道|产品)[A-E]"), "named_ref"),
    # Transition with accumulated knowledge: "现在我们有了...", "很好，现在我们..."
    (re.compile(r"(?:很好|好的|现在|目前)(?:我们|，|,).{0,10}(?:有了|已经有|已获得)"), "transition_ref"),
    # "上述", "前述", "前面(?:提到|计算|分析)的"
    (re.compile(r"上述|前述|前面(?:提到|计算|分析|讨论)的"), "direct_ref"),
    # "对比以上" / "比较以上"
    (re.compile(r"(?:对比|比较|相比|相对于?)以上"), "comparison_ref"),
]


def detect_referring_expressions(text: str) -> List[Tuple[str, str]]:
    """Detect referring expressions in *text*.

    Returns list of ``(matched_text, ref_type)`` tuples.
    """
    results: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    for pattern, ref_type in _REFERRING_PATTERNS:
        for m in pattern.finditer(text):
            matched = m.group(0)
            if matched not in seen:
                seen.add(matched)
                results.append((matched, ref_type))
    return results


# ======================================================================
# Entity / time / metric extraction for overlap analysis
# ======================================================================

# Simple entity patterns — company names, project names, product names, etc.
_RE_ENTITY = re.compile(
    r"(?:凤鸣大道[一二三]?期?|皖江路|上海|北京|广州|天津|深圳|"
    r"直通车|钻展|淘客|"
    r"名称\d+|材料\d+|产品[A-Z]|"
    r"案例[A-E]|方案[A-E]|"
    r"乘用车|商用车|"
    r"女粉色反领夹棉|"
    r"唐山市|西南石油大学|"
    r"环球金融中心|中央电视台|银泰中心|"
    r"金蝶EAS|"
    r"天猫|淘宝)"
)

# Time patterns already in build_target_evidence.py
_RE_TIME = re.compile(
    r"(\d{4}\s*年\s*(?:\d{1,2}\s*月(?:\s*(?:上旬|中旬|下旬|上半月|下半月|初|底|末))?)?"
    r"|\d{1,2}\s*月(?:\s*(?:上旬|中旬|下旬|上半月|下半月|初|底|末))?"
    r"|\d{4}\s*年第[一二三四季度]"
    r"|\d{4}年(?:\d{1,2}月)?\d{1,2}日)",
)

# Metric patterns
_RE_METRIC = re.compile(
    r"(?:产量|销量|出口量|同比增长率|环比增长率|"
    r"月还款|月供|总利息|成本占比|"
    r"访客数|跳失率|转化率|ROI|"
    r"投资|总投资|平均投资|单方造价|每公里造价|"
    r"库存周转|响应时间|询单转化率|下单转化率|"
    r"最高分|最低分|差异度|未完成率|"
    r"平均价格|涨跌幅|"
    r"库存周转压力系数|可支持天数)"
)

# Constraint keywords
_RE_CONSTRAINT = re.compile(
    r"(?:19\d{2}|20\d{2})\s*年(?:之前|之后|以来|以来|间|之间|度|第[一二三四]季度)?"
    r"|(?:19\d{2}|20\d{2})[-—](?:19\d{2}|20\d{2})"
    r"|\d{1,2}\s*月(?:\s*(?:上旬|中旬|下旬|上半月|下半月))?"
    r"|(?:北京|上海|广州|深圳|天津|成都|武汉|杭州|南京|重庆)"
    r"|(?:不低于|不高于|不超过|至少|至少为|最少|最多|高于|低于|大于|小于|以上|以下)"
    r"\s*\d+(?:\.\d+)?(?:\s*%|\s*元|\s*万|\s*亿)?"
)


def extract_entities(text: str) -> List[str]:
    """Extract entity names from text."""
    seen: Set[str] = set()
    results: List[str] = []
    for m in _RE_ENTITY.finditer(text):
        val = m.group(0)
        if val not in seen:
            seen.add(val)
            results.append(val)
    return results


def extract_times(text: str) -> List[str]:
    """Extract time expressions from text."""
    return [m.group(0) for m in _RE_TIME.finditer(text)]


def extract_metrics(text: str) -> List[str]:
    """Extract metric names from text."""
    seen: Set[str] = set()
    results: List[str] = []
    for m in _RE_METRIC.finditer(text):
        val = m.group(0)
        if val not in seen:
            seen.add(val)
            results.append(val)
    return results


def extract_constraints(text: str) -> List[str]:
    """Extract constraint expressions from text."""
    seen: Set[str] = set()
    results: List[str] = []
    for m in _RE_CONSTRAINT.finditer(text):
        val = m.group(0)
        if val not in seen:
            seen.add(val)
            results.append(val)
    return results


# ======================================================================
# LLM placeholder
# ======================================================================

_FUTURE_DEP_LLM_PROMPT = """\
You are a data dependency analyst. Given the current subquestion and all previous
subquestions in a multi-turn dialog, identify what information from earlier turns
is needed by the current turn.

Current subquestion: {current_question}
Current score points: {current_score_points}

Previous subquestions (with their score points and evidence):
{previous_context}

For each piece of information the current subquestion depends on from earlier turns,
output one dependency item with:
- "type": one of "numeric_fact", "entity_set", "reference", "constraint", "table_ref"
- "fields": type-specific fields (see required fields per type)
- "needed_by": the current subquestion id
- "source_boundary": which earlier subquestion provides this info

Output a JSON object: {"dependencies": [{"type": "...", "fields": {...}, ...}, ...]}
"""


def _call_llm_detect_dependencies(
    current_sq_id: int,
    current_question: str,
    current_score_points: List[str],
    previous_sqs: List[Dict[str, Any]],
    llm_client: Any,
) -> List[Dict[str, Any]]:
    """Call LLM to detect dependencies of current_sq on previous_sqs."""
    prev_lines: List[str] = []
    for psq in previous_sqs:
        prev_lines.append(f"  SQ{psq['sq_id']}: {psq['question']}")
        for sp in psq.get("score_points", []):
            prev_lines.append(f"    - {sp}")
    previous_context = "\n".join(prev_lines)

    prompt = _FUTURE_DEP_LLM_PROMPT.format(
        current_question=current_question,
        current_score_points="\n".join(current_score_points),
        previous_context=previous_context,
    )
    try:
        resp = llm_client.chat(
            prompt=prompt,
            system="You are a rigorous data dependency analyst.",
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        import json
        data = json.loads(resp.get("content", "{}"))
        return data.get("dependencies", [])
    except Exception:
        return []


# ======================================================================
# Future Dependency Builder
# ======================================================================

class FutureDependencyBuilder:
    """Build ``FutureDependencySet`` objects from benchmark samples.

    Parameters
    ----------
    d_fdc : int
        Maximum topological distance for future dependency analysis
        (default 2, per TEMPO-RL.md §5.4).
    llm_client : optional
        A ``ChatClient``-compatible object for LLM-assisted annotation.
    llm_enabled : bool
        Whether to actually call the LLM (default False — rule-only).
    """

    def __init__(
        self,
        d_fdc: int = 2,
        llm_client: Any = None,
        llm_enabled: bool = False,
    ):
        self.d_fdc = d_fdc
        self._llm = llm_client
        self._llm_enabled = llm_enabled and llm_client is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        benchmark_samples: List[Dict[str, Any]],
        target_evidence_sets: Optional[List[TargetEvidenceSet]] = None,
    ) -> List[FutureDependencySet]:
        """Build future dependency sets for every memory boundary in every sample.

        Parameters
        ----------
        benchmark_samples : list[dict]
            Loaded benchmark samples (from val.json or similar).
        target_evidence_sets : list[TargetEvidenceSet] or None
            Pre-built target evidence sets for source_evidence_id linking.
            If None, source_evidence_id fields will be left unset.

        Returns
        -------
        list[FutureDependencySet]
            Flat list with one entry per memory boundary.
        """
        # Index target evidence by (sample_id, subquestion_id) for fast lookup
        te_index: Dict[Tuple[str, int], List[EvidenceItem]] = {}
        if target_evidence_sets:
            for tes in target_evidence_sets:
                key = (tes.sample_id, tes.subquestion_id)
                te_index.setdefault(key, []).extend(tes.evidence_items)

        all_sets: List[FutureDependencySet] = []
        for sample in benchmark_samples:
            all_sets.extend(self.build_one_sample(sample, te_index))
        return all_sets

    def build_one_sample(
        self,
        sample: Dict[str, Any],
        te_index: Optional[Dict[Tuple[str, int], List[EvidenceItem]]] = None,
    ) -> List[FutureDependencySet]:
        """Build future dependencies for all boundaries in one sample.

        Returns one ``FutureDependencySet`` per boundary (i.e. N-1 sets for
        N subquestions).
        """
        sample_id = get_sample_id(sample)
        checkout_list = sample.get("design", {}).get("checkout_list", [])

        if len(checkout_list) <= 1:
            return []  # no boundaries to analyse

        # Flatten subquestions
        sqs: List[Dict[str, Any]] = []
        for item in checkout_list:
            sq_id = item.get("idx", 1)  # idx is 1-based in benchmark data
            sqs.append({
                "sq_id": sq_id,
                "question": item.get("info_item", ""),
                "score_points": item.get("score_points", []),
                "related_tables": item.get("related_tables", []),
                "raw_item": item,
            })

        if te_index is None:
            te_index = {}

        N = len(sqs)
        boundary_sets: List[FutureDependencySet] = []

        # LLM batch pre-computation (if enabled)
        llm_deps: Dict[int, List[Dict[str, Any]]] = {}
        if self._llm_enabled:
            for j in range(1, N):  # sq index j (1-based)
                current = sqs[j]
                prev_sqs = sqs[:j]
                if prev_sqs:
                    llm_deps[j] = _call_llm_detect_dependencies(
                        current_sq_id=current["sq_id"],
                        current_question=current["question"],
                        current_score_points=current["score_points"],
                        previous_sqs=prev_sqs,
                        llm_client=self._llm,
                    )

        # For each boundary after_sq_i, i in 1..N-1
        for i in range(1, N):  # i is 1-based sq_id, boundary after sq_i
            boundary_name = f"after_sq{i}"
            deps: List[FutureDependency] = []
            dep_counter = 1

            # Future subquestions within D_FDC
            max_j = min(N, i + self.d_fdc)
            for j in range(i, max_j):  # j is 0-based index into sqs, j >= i
                sq_j = sqs[j]
                sq_j_id = sq_j["sq_id"]

                # Only scan sources *before* the boundary (k < i), since
                # information from sq_i+1 onward hasn't been produced yet.
                for k in range(i):  # sq_k is before the boundary
                    sq_k = sqs[k]
                    sq_k_id = sq_k["sq_id"]

                    # ---- Rule-based dependency detection ----
                    detected = self._detect_dependencies(
                        sample_id=sample_id,
                        from_sq=sq_k,
                        to_sq=sq_j,
                        te_index=te_index,
                    )

                    for dep in detected:
                        dep.dependency_id = f"dep_sq{i}_{dep_counter}"
                        dep_counter += 1
                        deps.append(dep)

                # ---- LLM enrichment (if enabled) for this future sq ----
                if self._llm_enabled and j in llm_deps:
                    llm_result = llm_deps[j]
                    if llm_result:
                        llm_items = self._llm_results_to_deps(
                            llm_result, sample_id, boundary_name, dep_counter,
                        )
                        deps.extend(llm_items)
                        dep_counter += len(llm_items)

            boundary_sets.append(
                FutureDependencySet(
                    sample_id=sample_id,
                    boundary=boundary_name,
                    future_dependencies=deps,
                )
            )

        return boundary_sets

    # ------------------------------------------------------------------
    # Dependency detection between two subquestions
    # ------------------------------------------------------------------

    def _detect_dependencies(
        self,
        sample_id: str,
        from_sq: Dict[str, Any],
        to_sq: Dict[str, Any],
        te_index: Dict[Tuple[str, int], List[EvidenceItem]],
    ) -> List[FutureDependency]:
        """Detect what *to_sq* depends on from *from_sq*."""
        deps: List[FutureDependency] = []
        needed_by = f"sq{to_sq['sq_id']}"
        from_sq_id = from_sq["sq_id"]

        to_question = to_sq["question"]
        to_sps = " ".join(to_sq["score_points"])
        to_text = to_question + " " + to_sps
        from_text = from_sq["question"] + " " + " ".join(from_sq["score_points"])

        # 1. Referring expressions in to_sq
        refs = detect_referring_expressions(to_text)
        for ref_text, ref_type in refs:
            # Try to link to source evidence
            source_ev = self._find_source_evidence(
                sample_id, from_sq_id, "reference", ref_text, te_index,
            )
            deps.append(FutureDependency(
                dependency_id="",  # filled by caller
                type="reference",
                needed_by=needed_by,
                source_evidence_id=source_ev,
                fields={
                    "reference_text": ref_text,
                    "target_sq": f"sq{from_sq_id}",
                    "ref_type": ref_type,
                },
                audit=AuditInfo(
                    parse_confidence=0.6 if source_ev else 0.4,
                    warnings=(
                        [] if source_ev
                        else ["reference not linked to source evidence"]
                    ),
                    source="rule_extraction",
                ),
            ))

        # 2. Entity overlap
        from_entities = set(extract_entities(from_text))
        to_entities = set(extract_entities(to_text))
        shared_entities = from_entities & to_entities
        if shared_entities:
            # Create entity_set dependency
            source_ev = self._find_source_evidence(
                sample_id, from_sq_id, "entity_set",
                ",".join(sorted(shared_entities)), te_index,
            )
            deps.append(FutureDependency(
                dependency_id="",
                type="entity_set",
                needed_by=needed_by,
                source_evidence_id=source_ev,
                fields={"entities": sorted(shared_entities)},
                audit=AuditInfo(
                    parse_confidence=0.7 if source_ev else 0.5,
                    warnings=(
                        [] if source_ev
                        else ["entity_set not linked to source evidence"]
                    ),
                    source="rule_extraction",
                ),
            ))

        # 3. Numeric fact overlap: time + metric combinations
        from_times = set(extract_times(from_text))
        to_times = set(extract_times(to_text))
        from_metrics = set(extract_metrics(from_text))
        to_metrics = set(extract_metrics(to_text))

        shared_times = from_times & to_times
        shared_metrics = from_metrics & to_metrics

        if shared_metrics:
            # Look for numeric_fact dependencies — only create when we have
            # a source_evidence link that can fill entity/time/value/unit.
            for metric in shared_metrics:
                time_val = list(shared_times)[0] if shared_times else None
                entity_val = list(shared_entities)[0] if shared_entities else None

                source_ev = self._find_source_evidence(
                    sample_id, from_sq_id, "numeric_fact", metric, te_index,
                )

                # numeric_fact has 5 required fields.  Skip if we can't link
                # to source evidence or don't have entity+time context.
                if not source_ev:
                    continue

                # Extract value/unit from the linked evidence item
                ev_value = ""
                ev_unit = ""
                ev_entity = entity_val or ""
                ev_time = time_val or ""
                key = (sample_id, from_sq_id)
                for ei in te_index.get(key, []):
                    if ei.evidence_id == source_ev:
                        ev_value = ei.value or ""
                        ev_unit = ei.unit or ""
                        ev_entity = ei.entity or ev_entity
                        ev_time = ei.time or ev_time
                        break

                deps.append(FutureDependency(
                    dependency_id="",
                    type="numeric_fact",
                    needed_by=needed_by,
                    source_evidence_id=source_ev,
                    fields={
                        "entity": ev_entity,
                        "time": ev_time,
                        "metric": metric,
                        "value": ev_value,
                        "unit": ev_unit,
                    },
                    audit=AuditInfo(
                        parse_confidence=0.7,
                        warnings=[],
                        source="rule_extraction",
                    ),
                ))

        # 4. Table reference overlap
        from_tables = set(from_sq.get("related_tables", []))
        to_tables = set(to_sq.get("related_tables", []))
        shared_tables = from_tables & to_tables
        for tbl in shared_tables:
            source_ev = self._find_source_evidence(
                sample_id, from_sq_id, "table_ref", tbl, te_index,
            )
            deps.append(FutureDependency(
                dependency_id="",
                type="table_ref",
                needed_by=needed_by,
                source_evidence_id=source_ev,
                fields={"table_name": tbl},
                audit=AuditInfo(
                    parse_confidence=0.8,
                    warnings=[],
                    source="rule_extraction",
                ),
            ))

        # 5. Constraint inheritance: detect if to_sq references constraints
        #    from from_sq (year ranges, thresholds, etc.)
        from_constraints = set(extract_constraints(from_text))
        to_constraints = set(extract_constraints(to_text))
        shared_constraints = from_constraints & to_constraints
        for c in shared_constraints:
            source_ev = self._find_source_evidence(
                sample_id, from_sq_id, "constraint", c, te_index,
            )
            deps.append(FutureDependency(
                dependency_id="",
                type="constraint",
                needed_by=needed_by,
                source_evidence_id=source_ev,
                fields={"constraint_content": c},
                audit=AuditInfo(
                    parse_confidence=0.6 if source_ev else 0.5,
                    warnings=(
                        [] if source_ev
                        else ["constraint not linked to source evidence"]
                    ),
                    source="rule_extraction",
                ),
            ))

        # Deduplicate by type + fields signature
        return self._deduplicate_deps(deps)

    # ------------------------------------------------------------------
    # Source evidence linking
    # ------------------------------------------------------------------

    def _find_source_evidence(
        self,
        sample_id: str,
        from_sq_id: int,
        dep_type: str,
        match_text: str,
        te_index: Dict[Tuple[str, int], List[EvidenceItem]],
    ) -> Optional[str]:
        """Try to find a target evidence item that supports this dependency.

        Returns the ``evidence_id`` if found, else None.
        """
        key = (sample_id, from_sq_id)
        items = te_index.get(key, [])
        if not items:
            return None

        match_lower = match_text.lower()

        for ei in items:
            if dep_type == "numeric_fact":
                # Match by metric or value substring
                if (
                    (ei.metric and match_lower in ei.metric.lower())
                    or (ei.value and match_lower in ei.value.lower())
                ):
                    return ei.evidence_id

            elif dep_type == "entity_set":
                # Match by entity name
                if ei.entity and (
                    match_lower in ei.entity.lower()
                    or ei.entity.lower() in match_lower
                ):
                    return ei.evidence_id

            elif dep_type == "reference":
                # Match by value substring (referring expression in text_fact)
                if ei.value and match_lower in ei.value.lower():
                    return ei.evidence_id

            elif dep_type == "constraint":
                # Match by time or value
                if ei.time and match_lower in ei.time:
                    return ei.evidence_id
                if ei.value and match_lower in ei.value:
                    return ei.evidence_id

            elif dep_type == "table_ref":
                # Match by source_tables
                if match_text in ei.source_tables:
                    return ei.evidence_id

        return None

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate_deps(deps: List[FutureDependency]) -> List[FutureDependency]:
        """Remove duplicate dependencies (same type + fields + needed_by)."""
        seen: Set[Tuple[str, str, str]] = set()
        unique: List[FutureDependency] = []
        for dep in deps:
            # Build a signature from type, needed_by, and stable fields repr
            fields_sig = str(sorted(dep.fields.items()))
            sig = (dep.type, dep.needed_by, fields_sig)
            if sig not in seen:
                seen.add(sig)
                unique.append(dep)
        return unique

    # ------------------------------------------------------------------
    # LLM merge
    # ------------------------------------------------------------------

    def _llm_results_to_deps(
        self,
        llm_items: List[Dict[str, Any]],
        sample_id: str,
        boundary_name: str,
        counter_start: int,
    ) -> List[FutureDependency]:
        """Convert LLM-extracted dependency dicts into FutureDependency objects."""
        deps: List[FutureDependency] = []
        for i, li in enumerate(llm_items):
            dep_type = li.get("type", "reference")
            if dep_type not in ("numeric_fact", "entity_set", "reference",
                                "constraint", "table_ref"):
                dep_type = "reference"

            needed_by = li.get("needed_by", "")
            fields = li.get("fields", {})

            warnings: List[str] = []
            for fname in required_fields_for_type(dep_type):
                if fname not in fields or not fields[fname]:
                    warnings.append(f"LLM missing field '{fname}'")

            deps.append(FutureDependency(
                dependency_id=f"dep_{boundary_name}_{counter_start + i + 1}",
                type=dep_type,
                needed_by=needed_by,
                source_evidence_id=li.get("source_evidence_id"),
                fields=fields,
                weight=float(li.get("weight", 1.0)),
                audit=AuditInfo(
                    parse_confidence=0.7,
                    warnings=warnings,
                    source="llm_annotation",
                ),
            ))
        return deps

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @staticmethod
    def save(
        sets: List[FutureDependencySet],
        output_path: str,
    ) -> None:
        """Write future dependency sets to a JSONL file."""
        records = [s.to_dict() for s in sets]
        write_jsonl(output_path, records)

    @staticmethod
    def load(input_path: str) -> List[FutureDependencySet]:
        """Read future dependency sets from a JSONL file."""
        records = read_jsonl(input_path)
        return [FutureDependencySet.from_dict(r) for r in records]
