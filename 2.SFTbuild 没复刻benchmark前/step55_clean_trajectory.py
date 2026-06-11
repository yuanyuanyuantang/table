"""
Step 5.5: 轨迹清洗。

在 Memory 生成之前清洗轨迹，确保 memory 基于干净数据生成。
清洗阶段：call_id 冲突检测 → 去重 → BFloat16 过滤 → 展示调用删除 →
         孤立内容清理 → call_id 重编号 → 轨迹校验

输入:
  --subquestions : step5 输出的 repaired_subquestions.jsonl

输出:
  output/cleaned_subquestions.jsonl   — 清洗后的子问题记录
  output/recovery_audit.jsonl         — IndexError/NameError 恢复标签
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SFTbuild.utils import (
    read_jsonl, write_jsonl,
    run_cleaning_pipeline,
)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description='Step 5.5: Clean trajectories before memory generation')
    parser.add_argument('--subquestions', type=str,
                        default=os.path.join(project_root, 'SFTbuild', 'output', 'repaired_subquestions.jsonl'),
                        help='Path to repaired_subquestions.jsonl from step5')
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'output', 'cleaned_subquestions.jsonl'),
                        help='Output cleaned sub-questions JSONL path')
    parser.add_argument('--output_recovery_audit', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'output', 'recovery_audit.jsonl'),
                        help='Output recovery audit JSONL path (IndexError/NameError tags)')
    parser.add_argument('--output_audit', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'output', 'cleaning_audit.jsonl'),
                        help='Output cleaning audit JSONL path (all dialogs, pass and fail)')
    parser.add_argument('--config_key', type=str, default='mimo',
                        help='LLM config key for evidence fallback verification')
    parser.add_argument('--verbose', '-v', action='store_true', default=True)
    args = parser.parse_args()

    # Initialize LLM client for evidence verification fallback (Tier 2)
    from src.utils.chat_api import ChatClient
    evidence_client = ChatClient(config_key=args.config_key)

    records = read_jsonl(args.subquestions)
    if not records:
        print(f"[ERROR] No records in {args.subquestions}")
        sys.exit(1)

    # 按 (sample_id, candidate_id) 分组
    dialogs = {}
    for rec in records:
        key = (rec.get('sample_id', ''), rec.get('candidate_id', ''))
        dialogs.setdefault(key, []).append(rec)

    print(f"Grouped {len(records)} sub-questions into {len(dialogs)} dialogs")

    cleaned_records = []
    skipped = 0
    recovery_audit = []
    cleaning_audit = []

    for dialog_idx, ((sample_id, candidate_id), sub_recs) in enumerate(dialogs.items()):
        # 排序确保子问题顺序
        sub_recs.sort(key=lambda r: r.get('subquestion_id', 0))

        if args.verbose:
            print(f"\n  [CLEANING] dialog {dialog_idx}: {candidate_id[:60]}...")

        clean_pass, cleaned_sub_recs, clean_report = run_cleaning_pipeline(
            sub_recs, dialog_idx=dialog_idx,
            include_tools=None,
            verbose=args.verbose,
            evidence_client=evidence_client,
            use_llm_evidence=True,
        )

        # Build audit entry for this dialog (pass or fail)
        audit_entry = {
            'sample_id': sample_id,
            'candidate_id': candidate_id,
            'subquestion_count': len(sub_recs),
            'pass': clean_pass,
            'stage1_conflicts': clean_report.get('stage1_conflicts'),
            'stage3_bf16_removed': clean_report.get('stage3_bf16_removed', 0),
            'stage4_presentation_removed': clean_report.get('stage4_presentation_removed', False),
            'stage7_validation_pass': clean_report.get('stage7_validation_pass'),
            'stage7_issues': clean_report.get('stage7_issues', []),
            'stage7_evidence_audit': clean_report.get('stage7_evidence_audit', []),
            'recovery_tags': clean_report.get('recovery_tags', []),
        }
        cleaning_audit.append(audit_entry)

        if not clean_pass:
            skipped += 1
            if args.verbose:
                if clean_report.get('stage1_conflicts'):
                    conflicts = clean_report['stage1_conflicts']
                    print(f"  [SKIP] stage1 call_id conflicts: {len(conflicts)} call_ids")
                if clean_report.get('stage7_issues'):
                    print(f"  [SKIP] stage7 validation failed: {len(clean_report['stage7_issues'])} issues")
                    for issue in clean_report['stage7_issues'][:5]:
                        print(f"    - {issue}")
            continue

        # Collect recovery audit tags
        for tag in clean_report.get('recovery_tags', []):
            tag['sample_id'] = sample_id
            tag['candidate_id'] = candidate_id
            recovery_audit.append(tag)

        # Mark records as cleaned for downstream tracking
        for r in cleaned_sub_recs:
            r['_trajectory_cleaned'] = True

        cleaned_records.extend(cleaned_sub_recs)

        if args.verbose:
            n_bf16 = clean_report.get('stage3_bf16_removed', 0)
            n_recovery = len(clean_report.get('recovery_tags', []))
            print(f"  [PASS] {len(cleaned_sub_recs)} sub-questions, "
                  f"BFloat16 removed: {n_bf16}, recovery tags: {n_recovery}")

    # 输出
    write_jsonl(args.output, cleaned_records)
    if recovery_audit:
        write_jsonl(args.output_recovery_audit, recovery_audit)
    if cleaning_audit:
        write_jsonl(args.output_audit, cleaning_audit)

    print(f"\nDone.")
    print(f"  Skipped (cleaning failed): {skipped} dialogs")
    print(f"  Cleaned records:    {len(cleaned_records)} → {args.output}")
    if recovery_audit:
        print(f"  Recovery audit:     {len(recovery_audit)} entries → {args.output_recovery_audit}")
    print(f"  Cleaning audit:     {len(cleaning_audit)} entries → {args.output_audit}")


if __name__ == '__main__':
    main()
