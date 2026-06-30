"""
TEMPO-RL Phase 0 — Target Evidence Schema.

Defines the structured dataclasses for target evidence items, evidence sets,
and audit metadata.  All dataclasses support ``to_dict()`` / ``from_dict()``
for JSONL serialisation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

@dataclass
class AuditInfo:
    """Per-evidence-item audit metadata.

    Attributes
    ----------
    parse_confidence : float
        Confidence of the extraction (0.0 = rule-only, low confidence;
        1.0 = manually verified). LLM-assisted extractions typically sit
        in the 0.6–0.9 range.
    warnings : list[str]
        Human-readable warnings about missing fields, ambiguous parsing, etc.
    source : str
        How this item was produced — one of:
        ``"rule_extraction"``, ``"llm_annotation"``, ``"manual"``, ``"sft_trajectory"``.
    """
    parse_confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)
    source: str = "rule_extraction"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parse_confidence": self.parse_confidence,
            "warnings": self.warnings,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AuditInfo":
        return cls(
            parse_confidence=float(d.get("parse_confidence", 0.0)),
            warnings=list(d.get("warnings", [])),
            source=str(d.get("source", "rule_extraction")),
        )


# ---------------------------------------------------------------------------
# Evidence Item
# ---------------------------------------------------------------------------

@dataclass
class EvidenceItem:
    """One key fact the agent must obtain / compute for a subquestion.

    Fields
    ------
    sample_id : str
        Benchmark sample identifier (usually the ``task`` field).
    subquestion_id : int
        1-based index of the subquestion within the dialog.
    evidence_id : str
        Unique identifier within this subquestion, e.g. ``"sq1_e1"``.
    type : str
        One of ``"raw_value"``, ``"derived_value"``, ``"text_fact"``.
    value : str
        The fact string. For numeric types this is the number-with-unit
        string (e.g. ``"16.96%"``); for text_fact it is the free-text claim.
    entity : str or None
        The entity this fact describes (company, product, project, …).
    time : str or None
        Time expression (e.g. ``"2010年1月"``).
    metric : str or None
        What is being measured (e.g. ``"产量同比增长率"``).
    unit : str or None
        Unit of measurement (e.g. ``"%"``, ``"万元/公里"``).
    source_tables : list[str]
        Candidate table filenames this evidence likely comes from
        (derived from ``related_tables`` in the benchmark).
    input_evidence_ids : list[str]
        For ``derived_value`` only — the evidence_ids this computation
        depends on.
    operation : str or None
        For ``derived_value`` only — e.g. ``"同比增长率"``, ``"average"``.
    weight : float
        Importance weight (default 1.0).
    audit : AuditInfo
        Extraction audit trail.
    """

    sample_id: str
    subquestion_id: int
    evidence_id: str
    type: str                              # raw_value | derived_value | text_fact
    value: str
    entity: Optional[str] = None
    time: Optional[str] = None
    metric: Optional[str] = None
    unit: Optional[str] = None
    source_tables: List[str] = field(default_factory=list)
    input_evidence_ids: List[str] = field(default_factory=list)
    operation: Optional[str] = None
    weight: float = 1.0
    audit: AuditInfo = field(default_factory=AuditInfo)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def required_fields(self) -> List[str]:
        """Return the field names that *must* be non-None for this type."""
        common = ["sample_id", "subquestion_id", "evidence_id", "type", "value"]
        if self.type == "derived_value":
            return common + ["operation"]
        return common

    def missing_required_fields(self) -> List[str]:
        """Return which required fields are missing / empty."""
        missing: List[str] = []
        for fname in self.required_fields():
            val = getattr(self, fname, None)
            if val is None or (isinstance(val, str) and not val.strip()):
                missing.append(fname)
        return missing

    def validate(self) -> List[str]:
        """Run all validations and return a list of issues (empty = clean)."""
        issues: List[str] = []

        # Required fields
        for fname in self.missing_required_fields():
            issues.append(f"missing required field '{fname}'")

        # Type check
        if self.type not in ("raw_value", "derived_value", "text_fact"):
            issues.append(f"unknown type '{self.type}'")

        # Derived-value specific
        if self.type == "derived_value":
            if not self.input_evidence_ids:
                issues.append("derived_value missing input_evidence_ids")
            if not self.operation:
                issues.append("derived_value missing operation")

        # Text-fact specific
        if self.type == "text_fact":
            if self.value and len(self.value) < 2:
                issues.append("text_fact value too short")

        return issues

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "sample_id": self.sample_id,
            "subquestion_id": self.subquestion_id,
            "evidence_id": self.evidence_id,
            "type": self.type,
            "value": self.value,
            "weight": self.weight,
            "source_tables": self.source_tables,
            "audit": self.audit.to_dict(),
        }
        # Optional fields — only include if non-None / non-empty
        for k in ("entity", "time", "metric", "unit", "operation"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        if self.input_evidence_ids:
            d["input_evidence_ids"] = self.input_evidence_ids
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvidenceItem":
        audit = AuditInfo.from_dict(d.get("audit", {}))
        return cls(
            sample_id=d["sample_id"],
            subquestion_id=int(d["subquestion_id"]),
            evidence_id=d["evidence_id"],
            type=d.get("type", "raw_value"),
            value=d.get("value", ""),
            entity=d.get("entity"),
            time=d.get("time"),
            metric=d.get("metric"),
            unit=d.get("unit"),
            source_tables=list(d.get("source_tables", [])),
            input_evidence_ids=list(d.get("input_evidence_ids", [])),
            operation=d.get("operation"),
            weight=float(d.get("weight", 1.0)),
            audit=audit,
        )


# ---------------------------------------------------------------------------
# Evidence Set  (one per subquestion)
# ---------------------------------------------------------------------------

@dataclass
class TargetEvidenceSet:
    """All target evidence items for a single subquestion.

    Fields
    ------
    sample_id : str
    subquestion_id : int
    question : str
        The ``info_item`` text — what the user asked.
    evidence_items : list[EvidenceItem]
        One or more items; at least one per score_point in the benchmark.
    """
    sample_id: str
    subquestion_id: int
    question: str
    evidence_items: List[EvidenceItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "subquestion_id": self.subquestion_id,
            "question": self.question,
            "evidence_items": [ei.to_dict() for ei in self.evidence_items],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TargetEvidenceSet":
        items = [EvidenceItem.from_dict(e) for e in d.get("evidence_items", [])]
        return cls(
            sample_id=d["sample_id"],
            subquestion_id=int(d["subquestion_id"]),
            question=d.get("question", ""),
            evidence_items=items,
        )

    def validate_all(self) -> Dict[str, List[str]]:
        """Validate every evidence item; returns ``{evidence_id: [issues]}``."""
        result: Dict[str, List[str]] = {}
        for ei in self.evidence_items:
            issues = ei.validate()
            if issues:
                result[ei.evidence_id] = issues
        return result


# ==========================================================================
# Future Dependency  (Phase 0 — Part 2)
# ==========================================================================

# Required fields per dependency type
_REQUIRED_FIELDS: Dict[str, List[str]] = {
    "numeric_fact": ["entity", "time", "metric", "value", "unit"],
    "entity_set": ["entities"],
    "reference": ["reference_text", "target_sq"],
    "constraint": ["constraint_content"],
    "table_ref": ["table_name"],
}

_VALID_DEP_TYPES = frozenset(_REQUIRED_FIELDS.keys())


def required_fields_for_type(dep_type: str) -> List[str]:
    """Return the required field names for a given dependency type."""
    return _REQUIRED_FIELDS.get(dep_type, [])


@dataclass
class FutureDependency:
    """One piece of information that a *later* subquestion depends on from an
    earlier subquestion (or earlier memory boundary).

    Fields
    ------
    dependency_id : str
        Unique id, e.g. ``"dep_sq1_001"``.
    type : str
        One of ``"numeric_fact"``, ``"entity_set"``, ``"reference"``,
        ``"constraint"``, ``"table_ref"``.
    needed_by : str
        The subquestion id (as string ``"sq<N>"``) that needs this info.
    source_evidence_id : str or None
        If this dependency can be linked to a specific ``EvidenceItem`` in
        ``target_evidence.jsonl``, this is its ``evidence_id``.
    fields : dict
        Type-specific content. Required keys depend on ``type``:

        * ``numeric_fact`` → ``entity, time, metric, value, unit``
        * ``entity_set`` → ``entities`` (list of entity name strings)
        * ``reference`` → ``reference_text`` (the referring expression),
          ``target_sq`` (which subquestion it refers to)
        * ``constraint`` → ``constraint_content`` (the constraint text)
        * ``table_ref`` → ``table_name``
    weight : float
        Importance weight (default 1.0).
    audit : AuditInfo
        Extraction audit trail.
    """

    dependency_id: str
    type: str                           # numeric_fact | entity_set | reference | constraint | table_ref
    needed_by: str                      # "sq2", "sq3", ...
    source_evidence_id: Optional[str] = None
    fields: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    audit: AuditInfo = field(default_factory=AuditInfo)

    def validate(self) -> List[str]:
        """Validate this dependency item; returns a list of issues (empty = ok)."""
        issues: List[str] = []

        if not self.dependency_id:
            issues.append("missing dependency_id")
        if self.type not in _VALID_DEP_TYPES:
            issues.append(f"unknown type '{self.type}'")
        if not self.needed_by:
            issues.append("missing needed_by")

        # Required fields per type
        for fname in required_fields_for_type(self.type):
            if fname not in self.fields or not self.fields[fname]:
                issues.append(f"missing required field '{fname}' for type '{self.type}'")

        return issues

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "dependency_id": self.dependency_id,
            "type": self.type,
            "needed_by": self.needed_by,
            "fields": self.fields,
            "weight": self.weight,
            "audit": self.audit.to_dict(),
        }
        if self.source_evidence_id:
            d["source_evidence_id"] = self.source_evidence_id
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FutureDependency":
        audit = AuditInfo.from_dict(d.get("audit", {}))
        return cls(
            dependency_id=d["dependency_id"],
            type=d.get("type", "numeric_fact"),
            needed_by=d.get("needed_by", ""),
            source_evidence_id=d.get("source_evidence_id"),
            fields=dict(d.get("fields", {})),
            weight=float(d.get("weight", 1.0)),
            audit=audit,
        )


@dataclass
class FutureDependencySet:
    """The set of future dependencies visible from one memory boundary.

    A *boundary* is ``"after_sq< i >"`` — i.e. the point after subquestion *i*
    finishes and before subquestion *i+1* begins.

    Fields
    ------
    sample_id : str
    boundary : str
        e.g. ``"after_sq1"``, ``"after_sq2"``.
    future_dependencies : list[FutureDependency]
        Dependencies needed by subquestions *after* this boundary that can
        (in principle) be satisfied from evidence obtained up to this boundary.
    """

    sample_id: str
    boundary: str
    future_dependencies: List[FutureDependency] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "boundary": self.boundary,
            "future_dependencies": [fd.to_dict() for fd in self.future_dependencies],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FutureDependencySet":
        deps = [FutureDependency.from_dict(fd) for fd in d.get("future_dependencies", [])]
        return cls(
            sample_id=d["sample_id"],
            boundary=d.get("boundary", ""),
            future_dependencies=deps,
        )

    def validate_all(self) -> Dict[str, List[str]]:
        """Validate every dependency; returns ``{dependency_id: [issues]}``."""
        result: Dict[str, List[str]] = {}
        for fd in self.future_dependencies:
            issues = fd.validate()
            if issues:
                result[fd.dependency_id] = issues
        return result
