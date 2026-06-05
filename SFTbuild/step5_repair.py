"""
Step 5: Repair 失败子问题。

对筛选未通过的子问题，用强 LLM 修复生成专家级轨迹。

输入:
  --subquestions  : step3 输出的 evaluated_subquestions.jsonl（含 eval 反馈）
  --audit         : step4 输出的 audit_report.jsonl（标记哪些失败）
  --config_key    : LLM 配置 key（默认 mimo）

输出:
  output/repaired_subquestions.jsonl  — 修复后的子问题记录

注意:
  - repair 可以看 score_points / related_tables / evaluation feedback
  - 但最终输出不能泄漏这些内容
  - 修复成功时 agent_steps 被替换并标记 _repaired=True
  - 修复失败时保留原始记录（原始 agent_steps 未被修改），避免 dialog 出现缺口
  - 修复失败的记录可通过 _repair_error 字段识别
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SFTbuild.utils import read_jsonl, write_jsonl, extract_json_from_response

# Default dataset table root
_DEFAULT_DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     'dataset', 'tables')

# Max chars per table in repair prompt (keep under token budget)
_MAX_TABLE_CHARS = 3000


def _find_table_path(filename: str, dataset_root: str = _DEFAULT_DATASET_ROOT) -> str | None:
    """
    Find the actual file path for a table filename by walking dataset_root.

    Handles two input formats:
      - Bare filename: "2010年产量.xlsx"
      - Relative path:  "Chinese/.../2010年产量.xlsx"

    Returns the first match, or None if not found.
    When multiple files share the same basename, the path prefix (if provided)
    is used to disambiguate.
    """
    if not os.path.isdir(dataset_root):
        return None

    basename = os.path.basename(filename)
    path_hint = os.path.dirname(filename) if '/' in filename else ''

    candidates = []
    for root, _dirs, files in os.walk(dataset_root):
        for f in files:
            if f == basename:
                full_path = os.path.join(root, f)
                candidates.append(full_path)

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # 多个同名文件：用输入中的路径前缀消歧义
    if path_hint:
        for cp in candidates:
            if path_hint in cp:
                return cp
    # 无法消歧义时返回第一个（记录警告由调用方处理）
    return candidates[0]


def _read_table_content(file_path: str, max_chars: int = _MAX_TABLE_CHARS) -> str:
    """
    Read a table file (xlsx/xls/csv) and return a formatted text representation.
    Returns empty string on failure.
    """
    if not os.path.exists(file_path):
        return ''
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    try:
        if ext == '.csv':
            with open(file_path, 'r', encoding='utf-8-sig') as fh:
                lines = fh.readlines()
        elif ext in ('.xlsx', '.xls'):
            import pandas as pd
            engine = 'openpyxl' if ext == '.xlsx' else 'xlrd'
            sheet_names = pd.ExcelFile(file_path, engine=engine).sheet_names
            all_rows = []
            for sn in sheet_names:
                df = pd.read_excel(file_path, sheet_name=sn, engine=engine, header=None)
                all_rows.append(f'[Sheet: {sn}]')
                all_rows.append(df.to_csv(index=False, header=False))
            # Join and split back to lines
            text = '\n'.join(all_rows)
            lines = text.split('\n')
        else:
            return ''

        # Format as table_head_reader style output
        total = len(lines)
        header = f'[SUCCESS] [{filename}] Total {total} lines\n{"─" * 40}'
        body_lines = []
        chars = len(header)
        for i, line in enumerate(lines):
            prefix = f'  {i+1}| '
            new_line = prefix + line.rstrip()
            if chars + len(new_line) > max_chars:
                body_lines.append(f'  ... (truncated, {total - i} remaining lines)')
                break
            body_lines.append(new_line)
            chars += len(new_line)
        return header + '\n' + '\n'.join(body_lines)

    except Exception as e:
        return f'[ERROR reading {filename}: {e}]'


def _build_table_context(true_tables: list, dataset_root: str = _DEFAULT_DATASET_ROOT) -> str:
    """
    Build a text block containing the real content of all required tables,
    for injection into the repair prompt.
    """
    if not true_tables:
        return '（无关联表格信息）'

    parts = []
    for tbl in true_tables:
        path = _find_table_path(tbl, dataset_root)
        if path:
            content = _read_table_content(path)
            parts.append(f'### 表格文件: {tbl}\n实际路径: {path}\n\n{content}')
        else:
            parts.append(f'### 表格文件: {tbl}\n[WARNING] 未在 dataset 中找到此文件，请根据文件名推断数据结构')
    return '\n\n'.join(parts)


# repair prompt 模板（子问题级局部修复）
REPAIR_SUBQUESTION_PROMPT = """你是一个专业的表格数据分析智能体。请修复以下未通过质检的子问题轨迹。

## 原始问题
{user_question}

## 原始智能体轨迹（可能包含错误）
{original_trajectory}

## 失败原因
{failure_reasons}

## 真实表格数据（你必须基于这些真实数据生成轨迹）
以下是从实际表格文件中读取的真实数据。你在 observation 中只能使用这些数据，严禁编造任何数字、日期、实体名称。
如果某个 observation 内容与下面提供的表格数据不一致，以这里的数据为准。

{table_context}

## 任务
为这个子问题生成一条修正后的智能体轨迹。轨迹必须满足以下要求：
1. 基于上面提供的**真实表格数据**进行推理，阅读并解析数据
2. 执行必要的计算（数值必须来自真实表格数据）
3. 生成包含 "answer" 和 "data_source" 字段的 JSON 答案
4. data_source 必须包含用到的表格文件名（使用上面表格数据中给出的文件名和后缀）

注意：
- 步骤规划（step_plan）使用中文
- 工具调用参数按实际工具要求填写
- 最终答案用中文表述
- **关键：observation 中的工具返回内容必须与上面提供的真实表格数据完全一致**
- 如果原始轨迹使用了错误的工具或路径，应在修复后的轨迹中体现正确的探索和定位过程

输出格式（JSON，每个 observation 必须是包含 content 字段的 dict）：
{{
  "agent_steps": [
    {{
      "agent_step_id": 1,
      "type": "tool_call",
      "step_plan": "本次操作的意图说明",
      "tool_calls": [
        {{
          "tool_call_id": "call_1",
          "tool_name": "工具名称",
          "arguments": {{"参数名": "参数值"}}
        }}
      ],
      "observations": [
        {{
          "tool_call_id": "call_1",
          "tool_name": "工具名称",
          "content": "工具返回的数据内容（必须与真实表格数据一致）",
          "success": true
        }}
      ]
    }},
    {{
      "agent_step_id": N,
      "type": "final_answer",
      "assistant_answer": {{
        "answer": "用中文表述的答案",
        "data_source": ["用到的文件名"]
      }}
    }}
  ]
}}

严禁在输出中包含修复指令或标注。"""


def build_repair_prompt(rec: dict, audit: dict, dataset_root: str = _DEFAULT_DATASET_ROOT) -> str:
    """构造修复 prompt，注入真实表格数据以保证跨轮数据一致性。"""
    # 找对应的 audit 信息
    sub_audit = None
    for sa in audit.get('subquestion_audits', []):
        if sa.get('subquestion_id') == rec.get('subquestion_id'):
            sub_audit = sa
            break

    failure_reasons = '\n'.join(f'- {i}' for i in (sub_audit.get('issues', []) if sub_audit else []))

    # 还原原始轨迹文本
    trajectory_parts = [f"User: {rec.get('user', '')}"]
    for step in rec.get('agent_steps', []):
        if step['type'] == 'tool_call':
            plan = step.get('step_plan', '')
            if plan:
                trajectory_parts.append(f"Plan: {plan}")
            for tc in step.get('tool_calls', []):
                if isinstance(tc, dict):
                    trajectory_parts.append(f"Tool Call: {tc.get('tool_name', 'unknown')}({json.dumps(tc.get('arguments', {}), ensure_ascii=False)})")
                else:
                    trajectory_parts.append(f"Tool Call: {tc}")
            for obs in step.get('observations', []):
                if isinstance(obs, dict):
                    trajectory_parts.append(f"Observation: {obs.get('content', '')[:500]}")
                else:
                    trajectory_parts.append(f"Observation: {str(obs)[:500]}")
        elif step['type'] == 'final_answer':
            ans = step.get('assistant_answer', {})
            trajectory_parts.append(f"Answer: {json.dumps(ans, ensure_ascii=False)}")

    original_trajectory = '\n'.join(trajectory_parts)

    # 从 evaluation 提取真实表格名，读取真实表格数据注入 prompt
    ev = rec.get('eval', {}) or {}
    td = ev.get('table_depend', {}) or {}
    true_tables = td.get('true_tables', [])

    # 如果 eval 里没有 true_tables，回退到 memory_before 中的 tables
    if not true_tables:
        mb = rec.get('memory_before', {}) or {}
        true_tables = [t.get('name', '') for t in mb.get('tables', []) if t.get('name')]

    table_context = _build_table_context(true_tables, dataset_root)

    return REPAIR_SUBQUESTION_PROMPT.format(
        user_question=rec.get('user', ''),
        original_trajectory=original_trajectory,
        failure_reasons=failure_reasons or 'Unknown',
        table_context=table_context
    )


def validate_agent_steps(steps: list) -> list:
    """校验 LLM 返回的 agent_steps 基本结构，返回问题列表。"""
    issues = []
    if not isinstance(steps, list):
        return ['agent_steps is not a list']
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            issues.append(f'step[{i}]: not a dict')
            continue
        stype = step.get('type', '')
        if stype == 'tool_call':
            tcs = step.get('tool_calls', [])
            if not isinstance(tcs, list):
                issues.append(f'step[{i}]: tool_calls is not a list')
            else:
                for j, tc in enumerate(tcs):
                    if not isinstance(tc, dict):
                        issues.append(f'step[{i}].tool_calls[{j}]: not a dict')
                    elif 'tool_name' not in tc:
                        issues.append(f'step[{i}].tool_calls[{j}]: missing tool_name')
            obs = step.get('observations', [])
            if not isinstance(obs, list):
                issues.append(f'step[{i}]: observations is not a list')
            else:
                for j, o in enumerate(obs):
                    if not isinstance(o, dict):
                        # 自动包装字符串 observation 为 dict
                        obs[j] = {'content': str(o), 'success': True}
                    elif 'content' not in o:
                        issues.append(f'step[{i}].observations[{j}]: missing content')
        elif stype == 'final_answer':
            ans = step.get('assistant_answer', {})
            if not isinstance(ans, dict):
                issues.append(f'step[{i}]: assistant_answer is not a dict')
        else:
            issues.append(f'step[{i}]: unknown type "{stype}"')
    return issues


def repair_subquestion(rec: dict, audit: dict, client, dataset_root: str = _DEFAULT_DATASET_ROOT,
                       verbose: bool = False) -> dict:
    """
    修复单条子问题。返回修复后的 record。
    """
    prompt = build_repair_prompt(rec, audit, dataset_root)

    try:
        response = client.chat(
            prompt=prompt,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        repaired = extract_json_from_response(response)

        # 用修复后的 agent_steps 替换
        new_steps = repaired.get('agent_steps', [])
        if new_steps:
            # 先校验结构再接受
            validation_issues = validate_agent_steps(new_steps)
            if validation_issues:
                rec['_repair_error'] = f'Structure validation failed: {"; ".join(validation_issues)}'
                if verbose:
                    print(f"    Repair validation failed: {validation_issues}")
                return rec

            rec['agent_steps'] = new_steps
            # 提取 final_answer，并验证其存在
            has_final = False
            for step in reversed(new_steps):
                if step.get('type') == 'final_answer':
                    rec['assistant_answer'] = step.get('assistant_answer', rec.get('assistant_answer', {}))
                    has_final = True
                    break
            if has_final:
                rec['_sq_pass'] = True
                rec['_repaired'] = True
                if verbose:
                    print(f"    Repaired: {len(new_steps)} steps")
            else:
                rec['_repair_error'] = 'LLM returned agent_steps without final_answer'
                if verbose:
                    print(f"    Repair incomplete: no final_answer in agent_steps")
        else:
            rec['_repair_error'] = 'LLM returned empty agent_steps'
            if verbose:
                print(f"    Repair returned empty agent_steps")

    except Exception as e:
        if verbose:
            print(f"    Repair failed: {e}")
        rec['_repair_error'] = str(e)

    return rec


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description='Step 5: Repair failed sub-questions')
    parser.add_argument('--subquestions', type=str,
                        default=os.path.join(project_root, 'SFTbuild', 'output', 'evaluated_subquestions.jsonl'),
                        help='Path to evaluated_subquestions.jsonl from step3')
    parser.add_argument('--audit', type=str,
                        default=os.path.join(project_root, 'SFTbuild', 'output', 'audit_report.jsonl'),
                        help='Path to audit_report.jsonl from step4')
    parser.add_argument('--config_key', type=str, default='mimo',
                        help='LLM config key for repair')
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'output', 'repaired_subquestions.jsonl'),
                        help='Output JSONL path')
    parser.add_argument('--dataset_root', type=str,
                        default=_DEFAULT_DATASET_ROOT,
                        help='Root directory of the dataset tables')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only print repair prompts without calling LLM')
    parser.add_argument('--verbose', '-v', action='store_true', default=True)
    args = parser.parse_args()

    records = read_jsonl(args.subquestions)
    audit_records = read_jsonl(args.audit)

    # 从 audit report 中收集失败子问题的 (sample_id, candidate_id, subquestion_id)
    failed_ids = set()
    for audit in audit_records:
        sid = audit.get('sample_id', '')
        cid = audit.get('candidate_id', '')
        for sa in audit.get('subquestion_audits', []):
            if not sa.get('pass'):
                failed_ids.add((sid, cid, sa.get('subquestion_id')))

    if not failed_ids:
        # 全部通过，标记 _sq_pass
        for rec in records:
            rec['_sq_pass'] = True
        print("No failed sub-questions found.")
        write_jsonl(args.output, records)
        print(f"All {len(records)} records passed through → {args.output}")
        return

    print(f"Found {len(failed_ids)} failed sub-questions to repair")

    # 标记每条记录的 step4 子问题级通过状态（供 step8 使用）
    for rec in records:
        key = (rec.get('sample_id', ''), rec.get('candidate_id', ''), rec.get('subquestion_id'))
        rec['_sq_pass'] = key not in failed_ids

    if args.dry_run:
        # 打印首个失败子问题的 repair prompt
        for rec in records:
            key = (rec.get('sample_id', ''), rec.get('candidate_id', ''), rec.get('subquestion_id'))
            if key in failed_ids:
                for audit in audit_records:
                    if audit.get('sample_id') == rec.get('sample_id') and audit.get('candidate_id') == rec.get('candidate_id'):
                        prompt = build_repair_prompt(rec, audit, args.dataset_root)
                        print(f"=== Repair Prompt for {rec['candidate_id']}/sq{rec['subquestion_id']} ===")
                        print(prompt[:2000])
                        print("...")
                        return
        print("No failed sub-questions found.")
        return

    # 初始化 LLM client
    from src.utils.chat_api import ChatClient
    client = ChatClient(config_key=args.config_key)

    repaired_count = 0
    repair_failed_count = 0
    for rec in records:
        key = (rec.get('sample_id', ''), rec.get('candidate_id', ''), rec.get('subquestion_id'))
        if key not in failed_ids:
            continue
        # 找对应的 audit
        audit_found = False
        for audit in audit_records:
            if audit.get('sample_id') == rec.get('sample_id') and audit.get('candidate_id') == rec.get('candidate_id'):
                audit_found = True
                if args.verbose:
                    print(f"Repairing {rec['candidate_id']}/sq{rec['subquestion_id']}...")
                repair_subquestion(rec, audit, client, args.dataset_root, verbose=args.verbose)
                repaired_count += 1
                if rec.get('_repair_error'):
                    repair_failed_count += 1
                break
        if not audit_found:
            rec['_repair_error'] = 'No matching audit record found'
            repair_failed_count += 1
            repaired_count += 1
            if args.verbose:
                print(f"  [WARN] {rec['candidate_id']}/sq{rec['subquestion_id']}: no matching audit → skipped repair")

    # 修复失败时保留原始记录（原始 agent_steps 未被修改），仅打标记供下游参考
    # 避免因 repair 失败导致 dialog 出现缺口和 memory 链断裂
    if repair_failed_count > 0:
        print(f"\n  [WARN] {repair_failed_count} sub-questions failed repair → kept original records (see _repair_error field)")

    write_jsonl(args.output, records)
    n_repaired = repaired_count - repair_failed_count
    print(f"Done. Repaired: {n_repaired}/{repaired_count} attempted")
    print(f"  {len(records)} records → {args.output}")


if __name__ == '__main__':
    main()
