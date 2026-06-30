"""
TEMPO-RL Phase 0 — Evidence Ledger.

Maintains the verified evidence ledger :math:`L_{i,t}` for a single subquestion
during rollout.  The ledger is monotonic — evidence is only added, never removed,
and each evidence item is verified at most once.

Usage::

    from TEMPO_RL.evidence_ledger import EvidenceLedger
    from TEMPO_RL.schemas import TargetEvidenceSet

    tes = TargetEvidenceSet(sample_id="s1", subquestion_id=1, ...)
    ledger = EvidenceLedger(tes)

    # Initialize from memory_before
    result = ledger.initialize_from_memory(memory_before_json)

    # Update after each tool call
    result = ledger.update(
        tool_call={"tool_name": "search_table", "arguments": {...}},
        observation={"tool_name": "search_table", "content": "...", "success": True},
        observation_metadata={"file": "auto_2010.csv"},
    )
    print(result["coverage_before"], "→", result["coverage_after"])
    print("New evidence:", result["new_evidence_ids"])
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .schemas import AuditInfo, EvidenceItem, TargetEvidenceSet
from .verifier import (
    verify_value_match,
    verify_source_match,
    verify_binding_match,
    verify_derived_inputs,
    verify_derived_operation,
    verify_derived_result,
    verify_evidence_item,
)


# ======================================================================
# Memory parsing helpers
# ======================================================================

def _parse_memory_facts(memory_before: Any) -> List[Dict[str, Any]]:
    """Extract factual claims from a memory_before JSON object.

    Memory is expected to be a JSON object with keys like:
    ``goal``, ``tables``, ``key_facts``, ``derived_results``,
    ``constraints``, ``pitfalls``.

    Returns a flat list of fact dicts with keys:
    ``{text, entity, time, metric, value, unit, provenance}``.
    """
    if isinstance(memory_before, str):
        try:
            memory_before = json.loads(memory_before)
        except (json.JSONDecodeError, TypeError):
            pass

    if not isinstance(memory_before, dict):
        return []

    facts: List[Dict[str, Any]] = []

    # key_facts: list of fact objects or strings
    for fact in _iter_list(memory_before, "key_facts"):
        facts.append(_normalize_fact_entry(fact))

    # derived_results: list of computed values
    for fact in _iter_list(memory_before, "derived_results"):
        facts.append(_normalize_fact_entry(fact))

    # tables: list of table descriptions (treated as provenance references)
    for tbl in _iter_list(memory_before, "tables"):
        if isinstance(tbl, dict):
            facts.append({
                "text": tbl.get("description", json.dumps(tbl, ensure_ascii=False)),
                "table_name": tbl.get("name") or tbl.get("table_name") or tbl.get("file", ""),
                "provenance": tbl.get("name") or tbl.get("table_name") or tbl.get("file", ""),
            })
        elif isinstance(tbl, str):
            facts.append({"text": tbl, "table_name": tbl, "provenance": tbl})

    # constraints: list of constraint strings/objects
    for c in _iter_list(memory_before, "constraints"):
        if isinstance(c, dict):
            facts.append({
                "text": c.get("content", json.dumps(c, ensure_ascii=False)),
                "constraint_content": c.get("content", ""),
            })
        elif isinstance(c, str):
            facts.append({"text": c, "constraint_content": c})

    return facts


def _iter_list(obj: dict, key: str) -> List[Any]:
    """Safely iterate over a list-valued key in a dict."""
    val = obj.get(key, [])
    if isinstance(val, list):
        return val
    if isinstance(val, (str, dict)):
        return [val]
    return []


def _normalize_fact_entry(fact: Any) -> Dict[str, Any]:
    """Normalize a fact entry (dict or str) into a standard dict."""
    if isinstance(fact, dict):
        return {
            "text": fact.get("text") or fact.get("value") or fact.get("content",
                   json.dumps(fact, ensure_ascii=False)),
            "entity": fact.get("entity"),
            "time": fact.get("time"),
            "metric": fact.get("metric"),
            "value": fact.get("value"),
            "unit": fact.get("unit"),
            "provenance": fact.get("provenance") or fact.get("source") or "",
        }
    if isinstance(fact, str):
        return {"text": fact}
    return {"text": str(fact)}


# ======================================================================
# Memory verification
# ======================================================================

def _verify_memory_fact_against_evidence(
    fact: Dict[str, Any],
    evidence_item: EvidenceItem,
) -> Tuple[bool, Dict[str, Any]]:
    """Check whether a memory fact satisfies C_value, C_binding, C_provenance
    for a target evidence item.

    Returns ``(all_pass, audit)``.
    """
    fact_text = (fact.get("text") or "") + " " + (fact.get("value") or "")
    audit: Dict[str, Any] = {
        "fact": {k: v for k, v in fact.items() if v},
        "evidence_id": evidence_item.evidence_id,
    }

    all_pass = True

    # --- C_value: check if evidence value appears in memory fact ---
    v_ok, v_audit = verify_value_match(
        evidence_item,
        observation_text="",       # memory init — no observation
        memory_text=fact_text,
        code_output="",
    )
    audit["C_value"] = v_audit
    if not v_ok:
        all_pass = False

    # --- C_binding: check entity/time/metric/unit in memory fact ---
    b_ok, b_audit = verify_binding_match(
        evidence_item,
        observation_text="",
        memory_text=fact_text,
        tool_arguments=None,
    )
    audit["C_binding"] = b_audit
    if not b_ok:
        all_pass = False

    # --- C_provenance: does the memory fact have a traceable source? ---
    provenance = fact.get("provenance", "") or fact.get("source", "") or ""
    # Also check if fact references any of the evidence's source_tables
    prov_ok = bool(provenance) or bool(fact.get("table_name"))
    if evidence_item.source_tables and provenance:
        prov_lower = provenance.lower()
        prov_ok = any(
            tbl.lower() in prov_lower or prov_lower in tbl.lower()
            for tbl in evidence_item.source_tables
        )
    audit["C_provenance"] = {
        "provenance_found": provenance if provenance else "(none)",
        "status": "pass" if prov_ok else "fail",
    }
    if not prov_ok:
        all_pass = False

    audit["passed"] = all_pass
    return all_pass, audit


# ======================================================================
# Evidence Ledger
# ======================================================================

class EvidenceLedger:
    """Verified evidence ledger for one subquestion during rollout.

    Parameters
    ----------
    target_evidence_set : TargetEvidenceSet
        The ground-truth evidence items for this subquestion.
    llm_client : optional
        Reserved for LLM-based binding verification (not yet implemented).
    llm_enabled : bool
        Whether to use LLM fallback (default False).
    """

    def __init__(
        self,
        target_evidence_set: TargetEvidenceSet,
        llm_client: Any = None,
        llm_enabled: bool = False,
    ):
        self._tes = target_evidence_set
        self._llm = llm_client
        self._llm_enabled = llm_enabled and llm_client is not None

        # verified_items: evidence_id → EvidenceItem
        self._verified: Dict[str, EvidenceItem] = {}

        # audit trail: evidence_id → audit dict
        self._audit: Dict[str, Dict[str, Any]] = {}

        # Step counter
        self._step: int = 0

        # Total weight of all target evidence
        self._total_weight = max(
            sum(ei.weight for ei in target_evidence_set.evidence_items),
            1e-9,
        )

        # Pre-index evidence by id for fast lookup
        self._evidence_map: Dict[str, EvidenceItem] = {}
        for ei in target_evidence_set.evidence_items:
            self._evidence_map[ei.evidence_id] = ei

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def verified_ids(self) -> Set[str]:
        """Return the set of verified evidence_ids."""
        return set(self._verified.keys())

    @property
    def verified_items(self) -> List[EvidenceItem]:
        """Return the list of verified EvidenceItem objects."""
        return list(self._verified.values())

    @property
    def target_items(self) -> List[EvidenceItem]:
        """Return all target evidence items."""
        return list(self._tes.evidence_items)

    @property
    def step(self) -> int:
        """Current tool step (0 = pre-tool, memory-init)."""
        return self._step

    @property
    def coverage(self) -> float:
        """Current weighted evidence coverage (0.0 – 1.0)."""
        verified_weight = sum(
            self._evidence_map[eid].weight
            for eid in self._verified
            if eid in self._evidence_map
        )
        return verified_weight / self._total_weight

    # ------------------------------------------------------------------
    # Initialization from memory
    # ------------------------------------------------------------------

    def initialize_from_memory(
        self,
        memory_before: Any = None,
    ) -> Dict[str, Any]:
        """Initialize the ledger from ``memory_before`` (M_{i-1}).

        Implements::

            L_{i,0} = VerifyMem(M_{i-1}) ∩ E_i^{req}

        Each memory fact must pass C_value, C_binding, and C_provenance
        to be admitted into the ledger.

        Returns
        -------
        dict with keys ``coverage_before`` (always 0), ``coverage_after``,
        ``new_evidence_ids``, ``memory_facts_checked``, ``audit``.
        """
        coverage_before = self.coverage
        new_ids: List[str] = []
        memory_audit: List[Dict[str, Any]] = []

        if memory_before is None or memory_before == {} or memory_before == "":
            return {
                "coverage_before": coverage_before,
                "coverage_after": self.coverage,
                "new_evidence_ids": [],
                "memory_facts_checked": 0,
                "audit": {"note": "no memory_before provided — ledger remains empty"},
            }

        # Parse memory into facts
        facts = _parse_memory_facts(memory_before)
        if not facts:
            return {
                "coverage_before": coverage_before,
                "coverage_after": self.coverage,
                "new_evidence_ids": [],
                "memory_facts_checked": 0,
                "audit": {"note": "memory_before parsed to 0 facts"},
            }

        # For each unverified target evidence item, check against memory facts
        for ei in self._tes.evidence_items:
            if ei.evidence_id in self._verified:
                continue

            # Try each memory fact
            best_audit: Optional[Dict[str, Any]] = None
            for fact in facts:
                ok, audit = _verify_memory_fact_against_evidence(fact, ei)
                memory_audit.append(audit)
                if ok and best_audit is None:
                    best_audit = audit

            if best_audit is not None:
                self._verified[ei.evidence_id] = ei
                self._audit[ei.evidence_id] = {
                    "step": 0,
                    "source": "memory_init",
                    "detail": best_audit,
                }
                new_ids.append(ei.evidence_id)

        return {
            "coverage_before": coverage_before,
            "coverage_after": self.coverage,
            "new_evidence_ids": new_ids,
            "memory_facts_checked": len(facts),
            "audit": {
                "facts_parsed": len(facts),
                "evidence_from_memory": len(new_ids),
                "checks": memory_audit[:20],  # truncate for readability
            },
        }

    # ------------------------------------------------------------------
    # Update from tool call
    # ------------------------------------------------------------------

    def update(
        self,
        tool_call: Optional[Dict[str, Any]] = None,
        observation: Optional[Dict[str, Any]] = None,
        observation_metadata: Optional[Dict[str, Any]] = None,
        code_output: str = "",
    ) -> Dict[str, Any]:
        """Update the ledger after a tool execution.

        Implements::

            L_{i,t+1} = UpdateLedger(L_{i,t}, c_{i,t}, o_{i,t}, E_i^{req})

        Parameters
        ----------
        tool_call : dict or None
            The tool call with ``tool_name`` and ``arguments`` keys.
        observation : dict or None
            The observation result with ``tool_name``, ``content``, ``success``.
        observation_metadata : dict or None
            Sidecar metadata (file, sheet, region, headers).
        code_output : str
            Output from code execution, if applicable.

        Returns
        -------
        dict with keys ``coverage_before``, ``coverage_after``,
        ``new_evidence_ids``, ``ledger`` (snapshot), ``audit``,
        ``tool_info``.
        """
        self._step += 1
        coverage_before = self.coverage
        new_ids: List[str] = []
        update_audit: List[Dict[str, Any]] = []

        # Normalize inputs
        tc = tool_call or {}
        obs = observation or {}
        tool_name = tc.get("tool_name") or obs.get("tool_name", "")
        tool_args = tc.get("arguments", {})
        obs_text = obs.get("content", "")
        obs_success = obs.get("success", True)

        # Handle infrastructure errors gracefully
        if not obs_success:
            return {
                "coverage_before": coverage_before,
                "coverage_after": self.coverage,
                "new_evidence_ids": [],
                "ledger": self.to_dict(),
                "audit": {"error": "observation marked as failed (infrastructure)"},
                "tool_info": {
                    "tool_name": tool_name,
                    "step": self._step,
                },
            }

        # For each unverified target evidence item, run verification
        for ei in self._tes.evidence_items:
            if ei.evidence_id in self._verified:
                continue  # already verified — never re-verify

            ok, item_audit = verify_evidence_item(
                evidence_item=ei,
                observation_text=obs_text,
                observation_metadata=observation_metadata,
                tool_arguments=tool_args,
                tool_name=tool_name,
                memory_text="",
                code_output=code_output,
                verified_evidence_ids=self.verified_ids,
                ledger_evidence_items=list(self._verified.values()),
            )
            update_audit.append(item_audit)

            if ok:
                self._verified[ei.evidence_id] = ei
                self._audit[ei.evidence_id] = {
                    "step": self._step,
                    "source": "tool_observation",
                    "tool_name": tool_name,
                    "detail": item_audit,
                }
                new_ids.append(ei.evidence_id)

        coverage_after = self.coverage

        return {
            "coverage_before": coverage_before,
            "coverage_after": coverage_after,
            "new_evidence_ids": new_ids,
            "ledger": self.to_dict(),
            "audit": {
                "items_checked": len(update_audit),
                "items_verified": len(new_ids),
                "checks": update_audit[:20],
            },
            "tool_info": {
                "tool_name": tool_name,
                "step": self._step,
            },
        }

    # ------------------------------------------------------------------
    # Derived evidence helper
    # ------------------------------------------------------------------

    def verify_derived_item(
        self,
        evidence_item: EvidenceItem,
        code_output: str = "",
    ) -> Dict[str, Any]:
        """Attempt to verify a single derived evidence item using the current
        ledger state.

        This is useful after a code execution returns the computed result —
        the caller can re-check whether previously-unverifiable derived items
        are now satisfied.

        Returns an audit dict like ``update()``.
        """
        if evidence_item.evidence_id in self._verified:
            return {
                "evidence_id": evidence_item.evidence_id,
                "already_verified": True,
                "coverage_before": self.coverage,
                "coverage_after": self.coverage,
            }

        coverage_before = self.coverage

        ok, item_audit = verify_evidence_item(
            evidence_item=evidence_item,
            observation_text="",
            observation_metadata=None,
            tool_arguments=None,
            tool_name="",
            memory_text="",
            code_output=code_output,
            verified_evidence_ids=self.verified_ids,
            ledger_evidence_items=list(self._verified.values()),
        )

        if ok:
            self._verified[evidence_item.evidence_id] = evidence_item
            self._audit[evidence_item.evidence_id] = {
                "step": self._step,
                "source": "derived_verification",
                "detail": item_audit,
            }

        return {
            "evidence_id": evidence_item.evidence_id,
            "verified": ok,
            "coverage_before": coverage_before,
            "coverage_after": self.coverage,
            "new_evidence_ids": [evidence_item.evidence_id] if ok else [],
            "audit": item_audit,
        }

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def is_verified(self, evidence_id: str) -> bool:
        """Check if a specific evidence_id is in the ledger."""
        return evidence_id in self._verified

    def get_audit(self, evidence_id: str) -> Optional[Dict[str, Any]]:
        """Return the audit trail for a verified evidence item."""
        return self._audit.get(evidence_id)

    def delta_coverage(self, previous_coverage: float) -> float:
        """Return coverage change since *previous_coverage*."""
        return max(0.0, self.coverage - previous_coverage)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the ledger to a dict."""
        return {
            "sample_id": self._tes.sample_id,
            "subquestion_id": self._tes.subquestion_id,
            "question": self._tes.question[:120],
            "step": self._step,
            "coverage": self.coverage,
            "total_items": len(self._tes.evidence_items),
            "verified_items": len(self._verified),
            "verified_ids": sorted(self._verified.keys()),
            "items": {
                eid: {
                    "type": ei.type,
                    "value": ei.value,
                    "verified_at_step": self._audit.get(eid, {}).get("step"),
                    "verified_via": self._audit.get(eid, {}).get("source"),
                }
                for eid, ei in self._verified.items()
            },
        }

    def summary(self) -> str:
        """Return a one-line summary string."""
        return (
            f"EvidenceLedger(sq{self._tes.subquestion_id}, "
            f"step={self._step}, "
            f"coverage={self.coverage:.2%}, "
            f"verified={len(self._verified)}/{len(self._tes.evidence_items)})"
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_delta_phi(
        coverage_before: float,
        coverage_after: float,
    ) -> float:
        """Compute evidence progress ΔΦ for a tool step.

        ``ΔΦ = max(0, coverage_after - coverage_before)``
        """
        return max(0.0, coverage_after - coverage_before)

    @staticmethod
    def from_target_evidence_set(
        tes: TargetEvidenceSet,
        memory_before: Any = None,
        llm_client: Any = None,
        llm_enabled: bool = False,
    ) -> "EvidenceLedger":
        """Factory: create a ledger and initialise from memory in one call."""
        ledger = EvidenceLedger(tes, llm_client=llm_client, llm_enabled=llm_enabled)
        if memory_before is not None:
            ledger.initialize_from_memory(memory_before)
        return ledger
