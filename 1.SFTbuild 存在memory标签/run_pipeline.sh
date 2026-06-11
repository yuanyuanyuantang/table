#!/bin/bash
# SFTbuild 流水线：按顺序执行 Step2 → Step8
# 用法:
#   bash SFTbuild/run_pipeline.sh                        # 完整运行
#   bash SFTbuild/run_pipeline.sh --dry-run              # 预览 LLM prompt（不调用 API）
#   bash SFTbuild/run_pipeline.sh --skip-llm             # 仅运行确定性审计 (step2/3/4，完成后退出)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

DRY_RUN=false
SKIP_LLM=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true ;;
        --skip-llm) SKIP_LLM=true ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

echo "=========================================="
echo "  SFTbuild Pipeline"
echo "  Dry run: $DRY_RUN  |  Skip LLM: $SKIP_LLM"
echo "=========================================="

# ---- Step 2: Decompose traces into sub-questions ----
echo ""
echo "[Step 2] Decomposing traces..."
python SFTbuild/step2_decompose.py -v
echo "[Step 2] Done → SFTbuild/output/aligned_subquestions.jsonl"

# ---- Step 3: Align evaluations ----
echo ""
echo "[Step 3] Aligning evaluations..."
python SFTbuild/step3_align_eval.py -v
echo "[Step 3] Done → SFTbuild/output/evaluated_subquestions.jsonl"

# ---- Step 4: Filter ----
echo ""
echo "[Step 4] Filtering sub-questions..."
python SFTbuild/step4_filter.py -v
echo "[Step 4] Done → SFTbuild/output/audit_report.jsonl, passed_subquestions.jsonl"

if $DRY_RUN; then
    echo ""
    echo "=== Dry-run: previewing LLM prompts ==="
    echo ""
    echo "[Step 5] Repair prompt preview:"
    python SFTbuild/step5_repair.py --dry-run
    echo ""
    echo "[Step 6] Memory generation prompt preview:"
    python SFTbuild/step6_memory_gen.py --dry-run
    echo ""
    echo "[Step 7] Memory verification prompt preview:"
    python SFTbuild/step7_memory_verify.py --dry-run
    exit 0
fi

if $SKIP_LLM; then
    echo ""
    echo "[Steps 5-8] Skipped (--skip-llm)"
    echo "[Done] Deterministic audit stages (Step 2-4) completed."
    echo "  Review audit outputs before running full pipeline:"
    echo "    SFTbuild/output/audit_report.jsonl"
    echo "    SFTbuild/output/passed_subquestions.jsonl"
    exit 0
fi

# ---- Step 5: Repair failed sub-questions ----
echo ""
echo "[Step 5] Repairing failed sub-questions..."
python SFTbuild/step5_repair.py -v
echo "[Step 5] Done → SFTbuild/output/repaired_subquestions.jsonl"

# ---- Step 5.5: Clean trajectories BEFORE memory generation ----
# 必须在 memory 生成之前清洗轨迹，否则 memory_after 可能引用
# 已被删除的 BFloat16 错误/重复调用/展示调用。
echo ""
echo "[Step 5.5] Cleaning trajectories..."
python SFTbuild/step55_clean_trajectory.py \
    --subquestions SFTbuild/output/repaired_subquestions.jsonl \
    --output SFTbuild/output/cleaned_subquestions.jsonl \
    --output_recovery_audit SFTbuild/output/recovery_audit.jsonl \
    --output_audit SFTbuild/output/cleaning_audit.jsonl
echo "[Step 5.5] Done → SFTbuild/output/cleaned_subquestions.jsonl"

# ---- Step 6: Generate compressed memory ----
# 使用 step55 清洗后的记录，确保 memory 基于干净轨迹生成
echo ""
echo "[Step 6] Generating compressed memory..."
python SFTbuild/step6_memory_gen.py \
    --subquestions SFTbuild/output/cleaned_subquestions.jsonl \
    --output SFTbuild/output/subquestions_with_memory.jsonl
echo "[Step 6] Done → SFTbuild/output/subquestions_with_memory.jsonl"

# ---- Step 7: Verify memory quality ----
echo ""
echo "[Step 7] Verifying memory quality..."
python SFTbuild/step7_memory_verify.py \
    --subquestions SFTbuild/output/subquestions_with_memory.jsonl \
    --output_audit SFTbuild/output/memory_audit.jsonl \
    --output_pass SFTbuild/output/memory_verified_subquestions.jsonl
echo "[Step 7] Done → SFTbuild/output/memory_audit.jsonl, memory_verified_subquestions.jsonl"

# ---- Step 8: Build final SFT data ----
echo ""
echo "[Step 8] Building final SFT data..."

# 必须使用 step7 输出（fail closed：无 memory 验证的数据不进入训练集）
if [ -f SFTbuild/output/memory_verified_subquestions.jsonl ] && [ "$(wc -l < SFTbuild/output/memory_verified_subquestions.jsonl)" -gt 0 ]; then
    SUBQ_INPUT="SFTbuild/output/memory_verified_subquestions.jsonl"
    echo "[Step 8] Using memory-verified subquestions"
else
    echo "[ERROR] memory_verified_subquestions.jsonl not found or empty — cannot build SFT data without memory verification"
    exit 1
fi

python SFTbuild/step8_build_sft.py --subquestions "$SUBQ_INPUT"
echo "[Step 8] Done → SFTbuild/output/trainable_sft.jsonl, trainable_sft_chat.jsonl"

# ---- Summary ----
echo ""
echo "=========================================="
echo "  Pipeline Complete"
echo "=========================================="
echo "Outputs:"
ls -lh SFTbuild/output/*.jsonl 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}' || echo "  (no output files)"
