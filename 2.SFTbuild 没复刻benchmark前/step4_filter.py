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
  - data_source 覆盖 checkout_list[i].related_tables（LLM 判断，复刻 benchmark TableDependMetric）
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

from SFTbuild.utils import read_jsonl, write_jsonl, load_samples, validate_assistant_answer


def _build_ds_coverage_prompts(dialogs: dict, sample_map: dict):
    """
    收集所有 data_source → related_tables 配对，构造 LLM batch 请求。

    Returns:
        prompts: 格式化后的 TABLE_COVERAGE_EVAL_PROMPT 列表
        keys: [(candidate_id, subquestion_id), ...]
        metadata: [{model_basenames, true_basenames, true_count}, ...]
    """
    from src.prompts.AgentEvalPrompt import TABLE_COVERAGE_EVAL_PROMPT

    prompts, keys, metadata = [], [], []

    for (sample_id, candidate_id), sub_recs in dialogs.items():
        sample = sample_map.get(sample_id, {})
        checkout_list = sample.get('design', {}).get('checkout_list', [])
        for rec in sorted(sub_recs, key=lambda r: r.get('subquestion_id', 0)):
            sq_id = rec.get('subquestion_id', 1)
            answer, _ = validate_assistant_answer(rec.get('assistant_answer', {}))
            model_ds = answer.get('data_source', []) or []
            sub_idx = sq_id - 1
            true_tables = []
            if sub_idx < len(checkout_list):
                true_tables = checkout_list[sub_idx].get('related_tables', []) or []

            # 与 benchmark TableDependMetric 一致：取 basename
            model_basenames = [os.path.basename(t) for t in model_ds
                               if isinstance(t, str) and t.strip()]
            true_basenames = [os.path.basename(t) for t in true_tables]

            pred_str = "\n".join(f"- {t}" for t in model_basenames) if model_basenames else "None"
            true_str = "\n".join(f"- {t}" for t in true_basenames) if true_basenames else "None"

            prompts.append(TABLE_COVERAGE_EVAL_PROMPT.format(
                true_tables=true_str, pred_tables=pred_str))
            keys.append((candidate_id, sq_id))
            metadata.append({
                'model_basenames': model_basenames,
                'true_basenames': true_basenames,
                'true_count': len(true_basenames),
            })

    return prompts, keys, metadata


def _batch_check_data_source_coverage(dialogs: dict, sample_map: dict,
                                       config_key: str = 'mimo',
                                       verbose: bool = False) -> dict:
    """
    使用 LLM 批量检查 data_source 覆盖率，完全复刻 benchmark TableDependMetric。

    Returns:
        {(candidate_id, subquestion_id): {
            'covered': bool,
            'missing': [...],
            'model_basenames': [...],
            'true_basenames': [...],
            'reasoning': str,
        }}
    """
    prompts, keys, meta_list = _build_ds_coverage_prompts(dialogs, sample_map)

    if not prompts:
        return {}

    from src.utils.chat_api import ChatClient
    client = ChatClient(config_key=config_key)

    if verbose:
        print(f"  [LLM] Checking data_source coverage for {len(prompts)} sub-questions...")

    responses = client.batch_chat(
        prompts, temperature=0.0,
        response_format={"type": "json_object"},
        threads=10, batch_size=20,
        verbose=verbose,
    )

    results = {}
    for (key, meta, resp) in zip(keys, meta_list, responses):
        try:
            result_json = json.loads(resp['content'])
            if not isinstance(result_json, dict):
                result_json = {"reasoning": "Parse Error", "covered_true_count": 0, "correct_pred_count": 0}
        except Exception:
            result_json = {"reasoning": "Exception Error", "covered_true_count": 0, "correct_pred_count": 0}

        covered_count = result_json.get("covered_true_count",
                                        result_json.get("covered_count", 0))
        total_true = meta['true_count']

        # 计算 missing（LLM 判定未覆盖的 true table）
        missing = []
        if covered_count < total_true:
            # LLM 不直接返回 missing 列表，保守标记全部为潜在缺失
            missing = meta['true_basenames']

        results[key] = {
            'covered': covered_count >= total_true,
            'missing': missing,
            'model_basenames': meta['model_basenames'],
            'true_basenames': meta['true_basenames'],
            'llm_reasoning': result_json.get('reasoning', ''),
            'covered_true_count': covered_count,
            'total_true': total_true,
        }

    return results


def filter_subquestion(rec: dict, sample: dict = None,
                       ds_coverage: dict = None) -> dict:
    """
    对单条子问题记录做筛选，返回 issue 列表和 pass 状态。

    ds_coverage: 预计算的 LLM data_source 覆盖结果（由 _batch_check_data_source_coverage 生成），
                 若为 None 则跳过 data_source 检查。
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
    elif coverage < 0.999:
        issues.append(f'accuracy: coverage_ratio={coverage} < 1.0')

    # ---- 2. table recall (benchmark LLM 判断) ----
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

    # ---- 4. data_source coverage (LLM 判断，复刻 benchmark) ----
    ds_check = {}
    if ds_coverage is not None:
        key = (rec.get('candidate_id', ''), rec.get('subquestion_id', 1))
        ds_check = ds_coverage.get(key)
        if ds_check is None:
            issues.append('data_source: no LLM coverage result')
        elif not ds_check.get('covered'):
            missing = ds_check.get('missing', [])
            if missing:
                issues.append(f'data_source: missing tables {missing}')
            else:
                issues.append('data_source: LLM judged incomplete coverage')
    else:
        # 回退：不做 data_source 检查（不应发生）
        ds_check = {'covered': True, 'missing': [], 'fallback': True}

    # ---- 5. tool errors ----
    tool_audit = ev.get('tool_audit', {})
    if tool_audit.get('unrecovered_error_count', 0) > 0:
        issues.append(f'tool: {tool_audit["unrecovered_error_count"]} unrecovered errors')

    # ---- 6. completeness (structural integrity) ----
    agent_steps = rec.get('agent_steps', [])
    has_final = any(s.get('type') == 'final_answer' for s in agent_steps)
    has_evidence = any(s.get('type') == 'tool_call' for s in agent_steps)

    if not has_final:
        issues.append('completeness: no final_answer step')
    else:
        fa_indices = [i for i, s in enumerate(agent_steps) if s.get('type') == 'final_answer']
        if len(fa_indices) > 1:
            issues.append(f'completeness: {len(fa_indices)} final_answer steps (expected exactly 1)')
        if fa_indices and fa_indices[-1] != len(agent_steps) - 1:
            issues.append(f'completeness: final_answer is not the last step (at index {fa_indices[-1]}, total {len(agent_steps)} steps)')

    if not has_evidence:
        pass

    return {
        'pass': len(issues) == 0,
        'issues': issues,
        'data_source_check': ds_check
    }


def filter_dialogs(records: list, samples: list, verbose: bool = False,
                   config_key: str = 'mimo') -> tuple:
    """
    对子问题 + dialog 级别做筛选。
    返回 (audit_records, passed_records)
    """
    sample_map = {}
    for s in samples:
        task = (s.get('task', '') or '').strip()
        if task:
            sample_map[task] = s

    dialogs = {}
    for rec in records:
        key = (rec.get('sample_id', ''), rec.get('candidate_id', ''))
        dialogs.setdefault(key, []).append(rec)

    # ---- 批量 LLM 检查 data_source 覆盖率 ----
    ds_coverage = _batch_check_data_source_coverage(dialogs, sample_map,
                                                     config_key=config_key,
                                                     verbose=verbose)

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
            audit = filter_subquestion(rec, sample, ds_coverage=ds_coverage)
            sub_audits.append({
                'subquestion_id': rec.get('subquestion_id'),
                'user': rec.get('user', ''),
                'pass': audit['pass'],
                'issues': audit['issues'],
                'data_source_check': audit['data_source_check']
            })
            rec['_sq_pass'] = audit['pass']
            if not audit['pass']:
                all_pass = False

        dialog_issues = []
        actual_sq_count = len(sub_recs)
        if checkout_len > 0 and actual_sq_count != checkout_len:
            dialog_issues.append(f'subquestion count mismatch: {actual_sq_count} vs checkout {checkout_len}')
            all_pass = False
        elif checkout_len == 0:
            dialog_issues.append('sample not found or checkout_list empty, cannot verify subquestion count')
            all_pass = False
            if verbose:
                print(f"  [WARN] {candidate_id}: checkout_len=0, count check skipped → dialog FAIL")

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

        for rec in sub_recs:
            rec['_dialog_pass'] = all_pass

        if all_pass:
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
    parser.add_argument('--config_key', type=str, default='mimo',
                        help='LLM config key for data_source coverage check')
    parser.add_argument('--verbose', '-v', action='store_true', default=False)
    args = parser.parse_args()

    records = read_jsonl(args.subquestions)
    if not records:
        print(f"[ERROR] No records in {args.subquestions}")
        sys.exit(1)

    samples = load_samples(args.samples)
    print(f"Loaded {len(records)} sub-question records, {len(samples)} samples")

    audit_records, passed_records = filter_dialogs(records, samples, verbose=args.verbose,
                                                    config_key=args.config_key)

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
