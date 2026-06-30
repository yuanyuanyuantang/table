# TEMPO-RL Phase 0 Audit Report

**Generated**: 2026-06-23 23:53:48
**Output Directory**: `TEMPO_RL/output/phase0_audit`

## 1. Overview

| Metric | Value |
|---|---|
| Processed samples | 5 |
| Processed subquestions | 17 |
| Total tool steps | 32 |
| Total evidence items | 188 |
| Total future dependencies | 288 |

## 2. Evidence Statistics

| Type | Count |
|---|---|
| raw_value | 185 |
| text_fact | 3 |

## 3. Ledger Coverage

| Stat | Value |
|---|---|
| Coverage stats | min=0.000, max=0.762, mean=0.375, median=0.500 |
| Mean coverage | 37.5% |

## 4. Reward Statistics

| Component | Stats |
|---|---|
| r_tool (avg over 32 steps) | min=-0.020, max=0.623, mean=0.011, median=-0.020 |
| r_answer (17 subquestions) | min=0.000, max=1.000, mean=0.482, median=0.200 |
| r_memory (17 subquestions) | min=-0.068, max=0.668, mean=0.295, median=0.216 |

## 5. Tool Efficiency

| Metric | Value |
|---|---|
| Total tool calls | 32 |
| Invalid calls | 0 |
| Repeat calls (no new evidence) | 0 |
| Invalid rate | 0.0% |
| Repeat rate | 0.0% |

## 6. Warnings & Issues

| Category | Count |
|---|---|
| missing_target_evidence | 5 |

## 7. Output Files

| File | Description |
|---|---|
| `target_evidence.jsonl` | Target evidence sets per subquestion |
| `future_dependencies.jsonl` | Future dependency sets per memory boundary |
| `ledger_audit.jsonl` | Per-subquestion ledger state and coverage |
| `reward_audit.jsonl` | Per-subquestion reward breakdowns |
| `phase0_report.md` | This report |