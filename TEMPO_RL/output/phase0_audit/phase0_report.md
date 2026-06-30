# TEMPO-RL Phase 0 Audit Report

**Generated**: 2026-06-27 12:40:44
**Output Directory**: `TEMPO_RL/output/phase0_audit`

## 1. Overview

| Metric | Value |
|---|---|
| Processed samples | 5 |
| Processed subquestions | 22 |
| Total tool steps | 62 |
| Total evidence items | 102 |
| Total future dependencies | 343 |

## 2. Evidence Statistics

| Type | Count |
|---|---|
| raw_value | 88 |
| derived_value | 3 |
| text_fact | 11 |

## 3. Ledger Coverage

| Stat | Value |
|---|---|
| Coverage stats | min=0.000, max=1.000, mean=0.318, median=0.000 |
| Mean coverage | 31.8% |

## 4. Reward Statistics

| Component | Stats |
|---|---|
| r_tool (avg over 62 steps) | min=-0.020, max=0.980, mean=0.093, median=-0.020 |
| r_answer (22 subquestions) | min=0.000, max=1.000, mean=0.509, median=1.000 |
| r_memory (22 subquestions) | min=-0.151, max=0.616, mean=0.186, median=0.177 |

## 5. Tool Efficiency

| Metric | Value |
|---|---|
| Total tool calls | 62 |
| Invalid calls | 0 |
| Repeat calls (no new evidence) | 0 |
| Invalid rate | 0.0% |
| Repeat rate | 0.0% |

## 6. Warnings & Issues

No warnings detected.

## 7. Output Files

| File | Description |
|---|---|
| `target_evidence.jsonl` | Target evidence sets per subquestion |
| `future_dependencies.jsonl` | Future dependency sets per memory boundary |
| `ledger_audit.jsonl` | Per-subquestion ledger state and coverage |
| `reward_audit.jsonl` | Per-subquestion reward breakdowns |
| `phase0_report.md` | This report |