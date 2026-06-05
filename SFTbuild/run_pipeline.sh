#!/bin/bash
# SFTbuild 流水线：按顺序执行 Step2 → Step8
# 用法:
#   bash SFTbuild/run_pipeline.sh              # 完整运行
#   bash SFTbuild/run_pipeline.sh --dry-run    # 预览 LLM prompt（不调用 API）
#   bash SFTbuild/run_pipeline.sh --skip-llm   # 仅运行确定性步骤 (step2/3/4/8)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

DRY_RUN=false
SKIP_LLM=false

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --skip-llm) SKIP_LLM=true ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

echo "=========================================="
echo "  SFTbuild Pipeline"
echo "  Dry run: $DRY_RUN  |  Skip LLM: $SKIP_LLM"
echo "=========================================="

# ---- Step 2: Decompose traces into sub-questions ----
echo ""
echo "[Step 2] Decomposing traces..."
python SFTbuild/step2_decompose.py
echo "[Step 2] Done → SFTbuild/output/aligned_subquestions.jsonl"

# ---- Step 3: Align evaluations ----
echo ""
echo "[Step 3] Aligning evaluations..."
python SFTbuild/step3_align_eval.py
echo "[Step 3] Done → SFTbuild/output/evaluated_subquestions.jsonl"

# ---- Step 4: Filter ----
echo ""
echo "[Step 4] Filtering sub-questions..."
python SFTbuild/step4_filter.py
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
    echo "[Step 5-7] Skipped (--skip-llm)"
else
    # ---- Step 5: Repair failed sub-questions ----
    echo ""
    echo "[Step 5] Repairing failed sub-questions..."
    python SFTbuild/step5_repair.py
    echo "[Step 5] Done → SFTbuild/output/repaired_subquestions.jsonl"

    # ---- Step 6: Generate compressed memory ----
    # 使用 step5 修复后的全部记录（含修复失败 + 原始通过）
    echo ""
    echo "[Step 6] Generating compressed memory..."
    python SFTbuild/step6_memory_gen.py \
        --subquestions SFTbuild/output/repaired_subquestions.jsonl \
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
fi

# ---- Step 8: Build final SFT data ----
echo ""
echo "[Step 8] Building final SFT data..."

# 优先使用 step7 输出（含 memory），其次 step6，最后 step4
if [ -f SFTbuild/output/memory_verified_subquestions.jsonl ] && [ "$(wc -l < SFTbuild/output/memory_verified_subquestions.jsonl)" -gt 0 ]; then
    SUBQ_INPUT="SFTbuild/output/memory_verified_subquestions.jsonl"
    echo "[Step 8] Using memory-verified subquestions"
elif [ -f SFTbuild/output/subquestions_with_memory.jsonl ] && [ "$(wc -l < SFTbuild/output/subquestions_with_memory.jsonl)" -gt 0 ]; then
    SUBQ_INPUT="SFTbuild/output/subquestions_with_memory.jsonl"
    echo "[Step 8] Using subquestions with memory"
else
    SUBQ_INPUT="SFTbuild/output/passed_subquestions.jsonl"
    echo "[Step 8] Using passed subquestions (no memory)"
fi

python SFTbuild/step8_build_sft.py --subquestions "$SUBQ_INPUT"
echo "[Step 8] Done → SFTbuild/output/trainable_sft.jsonl, trainable_sft_chat.jsonl"

# ---- Summary ----
echo ""
echo "=========================================="
echo "  Pipeline Complete"
echo "=========================================="
echo "Outputs:"
ls -lh SFTbuild/output/*.jsonl | awk '{print "  " $NF " (" $5 ")"}'
