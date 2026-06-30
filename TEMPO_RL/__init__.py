"""
TEMPO-RL: Evidence-Guided Tool and Memory Optimization for Multi-turn Table Agents.

Phase 0 — Reward Infrastructure
--------------------------------
- ``schemas``                   : EvidenceItem, TargetEvidenceSet, AuditInfo,
                                  FutureDependency, FutureDependencySet
- ``build_target_evidence``     : TargetEvidenceBuilder
- ``build_future_dependencies`` : FutureDependencyBuilder
- ``verifier``                  : verify_value_match, verify_source_match,
                                  verify_binding_match, verify_derived_*, etc.
- ``evidence_ledger``           : EvidenceLedger
- ``reward_calculator``         : RewardCalculator
- ``io_utils``                  : read_jsonl, write_jsonl, load_benchmark_samples,
                                  try_parse_json, DEFAULT_SYSTEM_TEMPLATE

Phase 1 — Single-turn GRPO
-----------------------------
- ``rollout_phase1``            : RolloutRunner, PolicyWrapper
- ``build_segment_returns``     : SegmentReturnBuilder
- ``train_phase1``              : SegmentGRPOTrainer, GRPOSequenceDataset

Phase 2 — On-policy Memory RL
------------------------------
- ``rollout_phase2``            : DialogRolloutRunner
- ``build_segment_returns_phase2`` : SegmentReturnBuilderPhase2

Phase 3 — Sparse Counterfactual Memory RL
------------------------------------------
- ``counterfactual_phase3``     : CounterfactualEstimator
"""

# --- Phase 0: Schemas (eager — no heavy deps) ---
from .schemas import (
    AuditInfo,
    EvidenceItem,
    FutureDependency,
    FutureDependencySet,
    TargetEvidenceSet,
    required_fields_for_type,
)

# --- Phase 0: Builders (eager — no heavy deps) ---
from .build_target_evidence import (
    TargetEvidenceBuilder,
    extract_numeric_values,
    extract_percentage_values,
    extract_time,
    detect_computation,
)

from .build_future_dependencies import (
    FutureDependencyBuilder,
    detect_referring_expressions,
    extract_entities,
    extract_times,
    extract_metrics,
    extract_constraints,
)

# --- Phase 0: Verifier (eager — no heavy deps) ---
from .verifier import (
    verify_value_match,
    verify_source_match,
    verify_binding_match,
    verify_derived_inputs,
    verify_derived_operation,
    verify_derived_result,
    verify_evidence_item,
)

# --- Phase 0: Ledger & Calculator (eager — no heavy deps) ---
from .evidence_ledger import EvidenceLedger
from .reward_calculator import RewardCalculator

# --- Phase 0: I/O & Shared Utilities (eager — no heavy deps) ---
from .io_utils import (
    read_jsonl,
    write_jsonl,
    load_benchmark_samples,
    load_json_file,
    try_parse_json,
    DEFAULT_SYSTEM_TEMPLATE,
    DEFAULT_SYSTEM_TEMPLATE_PHASE1,
)

# ---------------------------------------------------------------------------
# Lazy imports for heavy modules (torch, transformers, chat_api, etc.)
# These are resolved on first access via module-level __getattr__.
# ---------------------------------------------------------------------------

_LAZY_IMPORTS = {
    # Phase 1
    "RolloutRunner": ("TEMPO_RL.rollout_phase1", "RolloutRunner"),
    "PolicyWrapper": ("TEMPO_RL.rollout_phase1", "PolicyWrapper"),
    "SegmentReturnBuilder": ("TEMPO_RL.build_segment_returns", "SegmentReturnBuilder"),
    "SegmentGRPOTrainer": ("TEMPO_RL.train_phase1", "SegmentGRPOTrainer"),
    "GRPOSequenceDataset": ("TEMPO_RL.train_phase1", "GRPOSequenceDataset"),
    # Phase 2
    "DialogRolloutRunner": ("TEMPO_RL.rollout_phase2", "DialogRolloutRunner"),
    "SegmentReturnBuilderPhase2": (
        "TEMPO_RL.build_segment_returns_phase2",
        "SegmentReturnBuilderPhase2",
    ),
    # Phase 3
    "CounterfactualEstimator": (
        "TEMPO_RL.counterfactual_phase3",
        "CounterfactualEstimator",
    ),
    # Phase 0 audit (imports rollout_phase1 internally, so keep lazy)
    "run_phase0_audit": ("TEMPO_RL.run_phase0_audit", "run_audit"),
}

# Also expose submodules for direct import (e.g., ``from TEMPO_RL import rollout_phase1``)
_LAZY_MODULES = {
    "rollout_phase1",
    "build_segment_returns",
    "train_phase1",
    "rollout_phase2",
    "build_segment_returns_phase2",
    "counterfactual_phase3",
    "run_phase0_audit",
}


def __getattr__(name: str):
    # Check class/function-level lazy imports
    info = _LAZY_IMPORTS.get(name)
    if info is not None:
        mod_name, attr_name = info
        import importlib
        mod = importlib.import_module(mod_name)
        attr = getattr(mod, attr_name)
        # Cache in module globals for faster subsequent access
        globals()[name] = attr
        return attr

    # Check module-level lazy imports
    if name in _LAZY_MODULES:
        import importlib
        mod = importlib.import_module(f"TEMPO_RL.{name}")
        globals()[name] = mod
        return mod

    raise AttributeError(f"module 'TEMPO_RL' has no attribute {name!r}")
