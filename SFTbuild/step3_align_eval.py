"""
Step 3: 将 evaluation 结果对齐到子问题。

输入:
  --subquestions  : step2 输出的 aligned_subquestions.jsonl
  --trace_dir     : 带 evaluation 的 trace JSON 文件目录
  --output        : 输出 JSONL 路径（默认 output/evaluated_subquestions.jsonl）

输出: evaluated_subquestions.jsonl
在 step2 记录基础上增加 eval 字段:
{
  ...
  "eval": {
    "accuracy": {...},
    "quality": {...},
    "table_depend": {...},
    "tool_audit": {...},
    "format_audit": {...}
  }
}
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SFTbuild.utils import (
    load_trace, read_jsonl, write_jsonl, audit_tool_errors
)


def align_evaluation(records: list, trace_dir: str, verbose: bool = False) -> list:
    """
    将 trace 中的 evaluation 数据合并到 sub-question 记录中。
    """
    # 预先加载所有 trace 的 evaluation 数据
    trace_evals = {}
    for fname in os.listdir(trace_dir):
        if fname.startswith('trace_') and fname.endswith('.json'):
            trace_id = fname.replace('.json', '')
            trace_data = load_trace(os.path.join(trace_dir, fname))
            evaluation = trace_data.get('evaluation', {})
            trace_evals[trace_id] = evaluation

    enriched = []
    for rec in records:
        candidate_id = rec.get('candidate_id', '')
        sub_idx = rec.get('subquestion_id', 1) - 1  # 1-based → 0-based
        ev = trace_evals.get(candidate_id, {})

        # 提取对应子问题的 evaluation
        accuracy = None
        quality = None
        table_depend = None

        acc_steps = ev.get('accuracy_steps', [])
        if sub_idx < len(acc_steps):
            accuracy = acc_steps[sub_idx].get('accuracy', {})

        qual_steps = ev.get('quality_steps', [])
        if sub_idx < len(qual_steps):
            quality = qual_steps[sub_idx].get('quality', {})

        td_steps = ev.get('table_depend_steps', [])
        if sub_idx < len(td_steps):
            table_depend = td_steps[sub_idx].get('table_depend', {})

        # 工具审计：从 agent_steps 重新解析
        tool_audit = audit_tool_errors(rec.get('agent_steps', []))

        # 格式审计
        answer = rec.get('assistant_answer', {})
        answer_text = answer.get('answer', '')
        data_source = answer.get('data_source', [])
        format_audit = {
            'answer_json_valid': isinstance(answer, dict) and 'answer' in answer,
            'has_answer': bool(answer_text),
            'has_data_source': bool(data_source)
        }

        rec['eval'] = {
            'accuracy': accuracy,
            'quality': quality,
            'table_depend': table_depend,
            'tool_audit': tool_audit,
            'format_audit': format_audit
        }

        enriched.append(rec)

        if verbose:
            acc_str = f"acc={accuracy.get('coverage_ratio', '?')}" if accuracy else "acc=?"
            td_str = f"recall={table_depend.get('recall', '?')}" if table_depend else "td=?"
            print(f"  {candidate_id}/sq{sub_idx+1}: {acc_str}, {td_str}")

    return enriched


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description='Step 3: Align evaluation to sub-questions')
    parser.add_argument('--subquestions', type=str,
                        default=os.path.join(project_root, 'SFTbuild', 'output', 'aligned_subquestions.jsonl'),
                        help='Path to aligned_subquestions.jsonl from step2')
    parser.add_argument('--trace_dir', type=str,
                        default=os.path.join(project_root, 'traces_output'),
                        help='Directory containing trace JSON files with evaluation')
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'output', 'evaluated_subquestions.jsonl'),
                        help='Output JSONL path')
    parser.add_argument('--verbose', '-v', action='store_true', default=True)
    args = parser.parse_args()

    records = read_jsonl(args.subquestions)
    if not records:
        print(f"[ERROR] No records found in {args.subquestions}")
        sys.exit(1)

    print(f"Loaded {len(records)} sub-question records from {args.subquestions}")
    records = align_evaluation(records, args.trace_dir, verbose=args.verbose)

    write_jsonl(args.output, records)
    print(f"\nDone. {len(records)} records written to {args.output}")


if __name__ == '__main__':
    main()
