"""
TEMPO-RL Phase 0 — Smoke test for offline audit pipeline.

Verifies that the full pipeline (target evidence → future dependencies →
ledger → reward) runs end-to-end without crashing, and that all output
files are structurally valid.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

# Import the audit module (not as __main__)
from TEMPO_RL.run_phase0_audit import (
    run_audit,
    validate_outputs,
    adapt_sft_memory_for_ledger,
)
from TEMPO_RL.io_utils import read_jsonl


# ======================================================================
# Paths to real data
# ======================================================================

_SAMPLES_PATH = os.path.join(_PROJ_ROOT, "dataset", "train不含val的.json")
_SFT_PATH = os.path.join(
    _PROJ_ROOT, "SFTbuild", "output", "memory_verified_subquestions.jsonl"
)


def _data_available():
    return os.path.exists(_SAMPLES_PATH) and os.path.exists(_SFT_PATH)


# ======================================================================
# Test 1 — Memory format adaptation
# ======================================================================

class TestMemoryAdaptation:
    """Test adapt_sft_memory_for_ledger conversion."""

    def test_empty_memory(self):
        result = adapt_sft_memory_for_ledger(None)
        assert result == {}

    def test_empty_dict(self):
        result = adapt_sft_memory_for_ledger({})
        assert result == {}

    def test_facts_conversion(self):
        sft_mem = {"facts": ["fact one", "fact two"]}
        result = adapt_sft_memory_for_ledger(sft_mem)
        assert "key_facts" in result
        assert len(result["key_facts"]) == 2
        assert result["key_facts"][0]["text"] == "fact one"

    def test_derived_conversion(self):
        sft_mem = {"derived": ["derived result 1"]}
        result = adapt_sft_memory_for_ledger(sft_mem)
        assert "derived_results" in result
        assert len(result["derived_results"]) == 1
        assert result["derived_results"][0]["text"] == "derived result 1"

    def test_tables_content_to_description(self):
        sft_mem = {
            "tables": [
                {"name": "table1.xlsx", "content": "This table contains data"}
            ]
        }
        result = adapt_sft_memory_for_ledger(sft_mem)
        assert "tables" in result
        assert result["tables"][0]["description"] == "This table contains data"
        assert "content" not in result["tables"][0]

    def test_goal_passthrough(self):
        sft_mem = {"goal": "find data"}
        result = adapt_sft_memory_for_ledger(sft_mem)
        assert result["goal"] == "find data"

    def test_constraints_passthrough(self):
        sft_mem = {"constraints": ["only 2010", "乘用车 only"]}
        result = adapt_sft_memory_for_ledger(sft_mem)
        assert result["constraints"] == sft_mem["constraints"]

    def test_mixed_facts(self):
        """Facts can be strings or dicts."""
        sft_mem = {
            "facts": [
                "string fact",
                {"text": "dict fact", "entity": "test"},
            ]
        }
        result = adapt_sft_memory_for_ledger(sft_mem)
        assert len(result["key_facts"]) == 2
        assert result["key_facts"][0]["text"] == "string fact"
        assert result["key_facts"][1]["text"] == "dict fact"


# ======================================================================
# Test 2 — Full pipeline smoke test (with real data)
# ======================================================================

@pytest.mark.skipif(not _data_available(), reason="SFT data not available")
class TestPipelineSmoke:
    """End-to-end pipeline smoke test using 1-2 real SFT records."""

    def test_pipeline_runs_without_crash(self):
        """Pipeline runs end-to-end without exceptions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_audit(
                samples_path=_SAMPLES_PATH,
                sft_path=_SFT_PATH,
                output_dir=tmpdir,
                max_samples=1,
            )
            assert result.get("status") == "ok", f"Pipeline failed: {result}"

    def test_all_output_files_created(self):
        """All four JSONL files + report are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_audit(
                samples_path=_SAMPLES_PATH,
                sft_path=_SFT_PATH,
                output_dir=tmpdir,
                max_samples=1,
            )
            for fname in [
                "target_evidence.jsonl",
                "future_dependencies.jsonl",
                "ledger_audit.jsonl",
                "reward_audit.jsonl",
                "phase0_report.md",
            ]:
                fpath = os.path.join(tmpdir, fname)
                assert os.path.exists(fpath), f"Missing: {fname}"

    def test_all_jsonl_parseable(self):
        """Every JSONL file contains parseable records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_audit(
                samples_path=_SAMPLES_PATH,
                sft_path=_SFT_PATH,
                output_dir=tmpdir,
                max_samples=1,
            )
            for fname in [
                "target_evidence.jsonl",
                "future_dependencies.jsonl",
                "ledger_audit.jsonl",
                "reward_audit.jsonl",
            ]:
                fpath = os.path.join(tmpdir, fname)
                records = read_jsonl(fpath)
                assert len(records) > 0, f"{fname} is empty"

    def test_reward_audit_has_required_fields(self):
        """Each reward audit record has r_tool, r_answer, r_memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_audit(
                samples_path=_SAMPLES_PATH,
                sft_path=_SFT_PATH,
                output_dir=tmpdir,
                max_samples=1,
            )
            fpath = os.path.join(tmpdir, "reward_audit.jsonl")
            records = read_jsonl(fpath)
            for r in records:
                assert "r_tool" in r, f"Missing r_tool in: {list(r.keys())}"
                assert "r_answer" in r
                assert "r_memory" in r
                assert isinstance(r["r_tool"], (int, float))
                assert isinstance(r["r_answer"], (int, float))
                assert isinstance(r["r_memory"], (int, float))

    def test_ledger_audit_has_required_fields(self):
        """Each ledger audit record has coverage info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_audit(
                samples_path=_SAMPLES_PATH,
                sft_path=_SFT_PATH,
                output_dir=tmpdir,
                max_samples=1,
            )
            fpath = os.path.join(tmpdir, "ledger_audit.jsonl")
            records = read_jsonl(fpath)
            for r in records:
                assert "coverage_final" in r
                assert "verified_count" in r
                assert "target_count" in r
                assert r["verified_count"] <= r["target_count"], \
                    "More verified than target items"

    def test_target_evidence_per_subquestion(self):
        """Each subquestion has at least one evidence item."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_audit(
                samples_path=_SAMPLES_PATH,
                sft_path=_SFT_PATH,
                output_dir=tmpdir,
                max_samples=1,
            )
            fpath = os.path.join(tmpdir, "target_evidence.jsonl")
            records = read_jsonl(fpath)
            for r in records:
                assert len(r.get("evidence_items", [])) > 0, \
                    f"Empty evidence items for sq {r.get('subquestion_id')}"

    def test_derived_evidence_has_inputs(self):
        """Derived evidence items have input_evidence_ids."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_audit(
                samples_path=_SAMPLES_PATH,
                sft_path=_SFT_PATH,
                output_dir=tmpdir,
                max_samples=1,
            )
            fpath = os.path.join(tmpdir, "target_evidence.jsonl")
            records = read_jsonl(fpath)
            for r in records:
                for ei in r.get("evidence_items", []):
                    if ei.get("type") == "derived_value":
                        assert ei.get("input_evidence_ids"), \
                            f"derived_value missing input_evidence_ids: {ei.get('evidence_id')}"

    def test_report_contains_key_sections(self):
        """phase0_report.md has all required sections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_audit(
                samples_path=_SAMPLES_PATH,
                sft_path=_SFT_PATH,
                output_dir=tmpdir,
                max_samples=1,
            )
            fpath = os.path.join(tmpdir, "report.md")
            # The report is named phase0_report.md by default
            fpath = os.path.join(tmpdir, "phase0_report.md")
            with open(fpath, "r") as f:
                report = f.read()

            required_sections = [
                "1. Overview",
                "2. Evidence Statistics",
                "3. Ledger Coverage",
                "4. Reward Statistics",
                "5. Tool Efficiency",
                "6. Warnings",
                "7. Output Files",
            ]
            for section in required_sections:
                assert section in report, f"Missing section: {section}"

    def test_validation_passes(self):
        """The validate_outputs function returns pass=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_audit(
                samples_path=_SAMPLES_PATH,
                sft_path=_SFT_PATH,
                output_dir=tmpdir,
                max_samples=1,
            )
            results = validate_outputs(tmpdir)
            assert results["pass"], \
                f"Validation failed: {[c for c in results['checks'] if not c['pass']]}"

    def test_pipeline_with_two_samples(self):
        """Pipeline handles multiple samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_audit(
                samples_path=_SAMPLES_PATH,
                sft_path=_SFT_PATH,
                output_dir=tmpdir,
                max_samples=2,
            )
            assert result.get("status") == "ok"
            # Should have processed at least 2 subquestions
            fpath = os.path.join(tmpdir, "reward_audit.jsonl")
            records = read_jsonl(fpath)
            assert len(records) >= 2, f"Expected >= 2 reward records, got {len(records)}"


# ======================================================================
# Test 3 — Validation function tests
# ======================================================================

class TestValidateOutputs:
    """Test the validate_outputs function directly."""

    def test_empty_dir(self):
        """Empty directory should fail most checks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = validate_outputs(tmpdir)
            # At least one check will fail (files missing)
            assert not results["pass"]
            assert len(results["checks"]) > 0

    def test_missing_files(self):
        """Missing files are reported."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only one file
            with open(os.path.join(tmpdir, "target_evidence.jsonl"), "w") as f:
                f.write('{"test": 1}\n')
            results = validate_outputs(tmpdir)
            checks_failed = [c for c in results["checks"] if not c["pass"]]
            assert len(checks_failed) > 0

    def test_bad_jsonl(self):
        """Malformed JSONL is caught."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file with bad JSON
            with open(os.path.join(tmpdir, "target_evidence.jsonl"), "w") as f:
                f.write("this is not json\n")
            results = validate_outputs(tmpdir)
            target_check = [c for c in results["checks"] if "target_evidence" in c["check"]]
            assert len(target_check) > 0
