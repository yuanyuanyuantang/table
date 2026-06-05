"""
Step 4: 候选 Trace 筛选。

对每条子问题做硬性检查，再汇总为 dialog 级通过/不通过。

输入:
  --subquestions  : step3 输出的 evaluated_subquestions.jsonl
  --samples       : benchmark 样本 JSON 文件（用于获取 related_tables）
  --output_audit  : 输出 audit_report.jsonl 路径
  --output_pass   : 通过筛选的子问题输出路径（默认 output/passed_subquestions.jsonl）

筛选逻辑 —— 子问题级硬性条件:
  - accuracy.coverage_ratio == 1.0
  - table_depend.recall == 1.0
  - answer JSON 可解析、非空
  - data_source 覆盖 checkout_list[i].related_tables
  - unrecovered_error_count == 0

筛选逻辑 —— dialog 级:
  - 所有子问题均通过
  - 子问题数量与 checkout_list 一致
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SFTbuild.utils import read_jsonl, write_jsonl, load_samples


def check_data_source_coverage(model_tables: list, true_tables: list) -> dict:
    """
    检查 model 的 data_source 是否覆盖 gold related_tables。
    用 basename 比较。
    """
    model_basenames = set(os.path.basename(t) for t in (model_tables or []))
    true_basenames = set(os.path.basename(t) for t in (true_tables or []))

    if not true_basenames:
        return {'covered': True, 'missing': [], 'model_basenames': list(model_basenames)}

    missing = true_basenames - model_basenames
    return {
        'covered': len(missing) == 0,
        'missing': list(missing),
        'model_basenames': list(model_basenames),
        'true_basenames': list(true_basenames)
    }


def filter_subquestion(rec: dict, sample: dict = None) -> dict:
    """
    对单条子问题记录做筛选，返回 issue 列表和 pass 状态。
    """
    issues = []
    ev = rec.get('eval', {})

    # ---- 1. accuracy coverage ----
    accuracy = ev.get('accuracy') or {}
    is_missing = accuracy.get('is_missing', False)
    coverage = accuracy.get('coverage_ratio')
    if is_missing:
        issues.append('accuracy: is_missing')
    elif coverage is None:
        issues.append('accuracy: no coverage_ratio')
    elif coverage < 0.999:  # 允许浮点误差
        issues.append(f'accuracy: coverage_ratio={coverage} < 1.0')

    # ---- 2. table recall ----
    td = ev.get('table_depend') or {}
    recall = td.get('recall')
    if recall is None:
        issues.append('table_depend: no recall')
    elif recall < 0.999:
        issues.append(f'table_depend: recall={recall} < 1.0')

    # ---- 3. format audit ----
    fmt = ev.get('format_audit', {})
    if not fmt.get('answer_json_valid'):
        issues.append('format: answer JSON invalid')
    if not fmt.get('has_answer'):
        issues.append('format: answer is empty')
    if not fmt.get('has_data_source'):
        issues.append('format: data_source is empty')

    # ---- 4. data_source coverage ----
    answer = rec.get('assistant_answer', {})
    model_ds = answer.get('data_source', []) or []

    # Try to get related_tables from sample
    true_tables = []
    if sample:
        checkout_list = sample.get('design', {}).get('checkout_list', [])
        sub_idx = rec.get('subquestion_id', 1) - 1
        if sub_idx < len(checkout_list):
            true_tables = checkout_list[sub_idx].get('related_tables', []) or []

    ds_check = check_data_source_coverage(model_ds, true_tables)
    if not ds_check['covered']:
        issues.append(f'data_source: missing tables {ds_check["missing"]}')

    # ---- 5. tool errors ----
    tool_audit = ev.get('tool_audit', {})
    if tool_audit.get('unrecovered_error_count', 0) > 0:
        issues.append(f'tool: {tool_audit["unrecovered_error_count"]} unrecovered errors')

    # ---- 6. completeness ----
    agent_steps = rec.get('agent_steps', [])
    has_final = any(s.get('type') == 'final_answer' for s in agent_steps)
    has_evidence = any(s.get('type') == 'tool_call' for s in agent_steps)
    if not has_final:
        issues.append('completeness: no final_answer step')
    if not has_evidence:
        # 可接受：纯基于前文 memory 回答
        pass

    return {
        'pass': len(issues) == 0,
        'issues': issues,
        'data_source_check': ds_check
    }


def filter_dialogs(records: list, samples: list, verbose: bool = False) -> tuple:
    """
    对子问题 + dialog 级别做筛选。
    返回 (audit_records, passed_records)
    """
    # 建立 sample_id -> sample 的索引
    sample_map = {}
    for s in samples:
        task = (s.get('task', '') or '').strip()
        if task:
            sample_map[task] = s

    # 按 (sample_id, candidate_id) 分组
    dialogs = {}
    for rec in records:
        key = (rec.get('sample_id', ''), rec.get('candidate_id', ''))
        dialogs.setdefault(key, []).append(rec)

    audit_records = []
    passed_records = []

    for (sample_id, candidate_id), sub_recs in dialogs.items():
        sample = sample_map.get(sample_id)
        if sample is None:
            if verbose:
                print(f"  [WARN] {candidate_id}: sample '{sample_id}' not found in sample_map, using empty sample")
            sample = {}
        checkout_len = len(sample.get('design', {}).get('checkout_list', []))

        sub_audits = []
        all_pass = True
        for rec in sorted(sub_recs, key=lambda r: r.get('subquestion_id', 0)):
            audit = filter_subquestion(rec, sample)
            sub_audits.append({
                'subquestion_id': rec.get('subquestion_id'),
                'user': rec.get('user', ''),
                'pass': audit['pass'],
                'issues': audit['issues'],
                'data_source_check': audit['data_source_check']
            })
            # 标记每个子问题的独立通过状态（用于 step8 筛选）
            rec['_sq_pass'] = audit['pass']
            if not audit['pass']:
                all_pass = False

        # Dialog-level checks
        dialog_issues = []
        actual_sq_count = len(sub_recs)
        if checkout_len > 0 and actual_sq_count != checkout_len:
            dialog_issues.append(f'subquestion count mismatch: {actual_sq_count} vs checkout {checkout_len}')
            all_pass = False
        elif checkout_len == 0:
            dialog_issues.append(f'sample not found or checkout_list empty, cannot verify subquestion count')
            if verbose:
                print(f"  [WARN] {candidate_id}: checkout_len=0, count check skipped")

        dialog_audit = {
            'sample_id': sample_id,
            'candidate_id': candidate_id,
            'dialog_pass': all_pass,
            'dialog_issues': dialog_issues,
            'subquestion_count': actual_sq_count,
            'checkout_count': checkout_len,
            'subquestion_audits': sub_audits
        }
        audit_records.append(dialog_audit)

        if all_pass:
            # 将通过筛选的子问题标记为 passed
            for rec in sub_recs:
                rec['_pass'] = True
            passed_records.extend(sub_recs)

        if verbose:
            status = 'PASS' if all_pass else 'FAIL'
            n_pass = sum(1 for a in sub_audits if a['pass'])
            print(f"  {candidate_id}: {status} ({n_pass}/{len(sub_audits)} sub-questions)")

    return audit_records, passed_records


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description='Step 4: Filter candidates')
    parser.add_argument('--subquestions', type=str,
                        default=os.path.join(project_root, 'SFTbuild', 'output', 'evaluated_subquestions.jsonl'),
                        help='Path to evaluated_subquestions.jsonl from step3')
    parser.add_argument('--samples', type=str,
                        default=os.path.join(project_root, 'dataset', 'samples_normal_easy.json'),
                        help='Path to samples JSON file')
    parser.add_argument('--output_audit', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'output', 'audit_report.jsonl'),
                        help='Output audit report JSONL path')
    parser.add_argument('--output_pass', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'output', 'passed_subquestions.jsonl'),
                        help='Output passed sub-questions JSONL path')
    parser.add_argument('--verbose', '-v', action='store_true', default=True)
    args = parser.parse_args()

    records = read_jsonl(args.subquestions)
    if not records:
        print(f"[ERROR] No records in {args.subquestions}")
        sys.exit(1)

    samples = load_samples(args.samples)
    print(f"Loaded {len(records)} sub-question records, {len(samples)} samples")

    audit_records, passed_records = filter_dialogs(records, samples, verbose=args.verbose)

    write_jsonl(args.output_audit, audit_records)
    write_jsonl(args.output_pass, passed_records)

    n_pass_dialogs = sum(1 for a in audit_records if a['dialog_pass'])
    print(f"\nDone.")
    print(f"  Dialogs: {n_pass_dialogs}/{len(audit_records)} passed")
    print(f"  Sub-questions: {len(passed_records)}/{len(records)} passed")
    print(f"  Audit report: {args.output_audit}")
    print(f"  Passed: {args.output_pass}")


if __name__ == '__main__':
    main()
