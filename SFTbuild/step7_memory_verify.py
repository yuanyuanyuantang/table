"""
Step 7: Memory Verifier。

检查 memory_before / memory_after 的四个维度:
  1. Faithfulness  — 事实必须来自 agent_steps 中的 observation / answer
  2. Sufficiency   — 下一轮问题中的指代能否被 memory 解释
  3. Continuity    — 跨轮 memory 不丢失已确认的关键信息
  4. Compression   — 不是完整历史的复制，只保存后续有用的状态

输入:
  --subquestions  : step6 输出的 subquestions_with_memory.jsonl
  --config_key    : LLM 配置 key

输出:
  output/memory_audit.jsonl  — 每条子问题的 memory 验证结果
  output/memory_verified_subquestions.jsonl  — 通过验证的记录
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SFTbuild.utils import read_jsonl, write_jsonl, extract_json_from_response

MEMORY_VERIFIER_PROMPT = """你是一个记忆质量审核专家。你的任务是审计从智能体执行轨迹中生成的压缩记忆。

## 之前的记忆（memory_before）
{memory_before}

## 当前问题
{user_question}

## 本轮的智能体执行步骤
{agent_trace}

## 最终答案
{final_answer}

## 生成的记忆（memory_after）
{memory_after}

## 任务
从以下四个维度评估 memory_after。对每个维度，给出是否通过（true/false）和具体问题。

1. **忠实性（Faithfulness）**：memory_after 中的每一条事实必须来自以下来源之一：
   - memory_before（上一轮的已有记忆）
   - agent_steps 中的工具返回结果（Observation）
   - final_answer（最终答案）
   严禁包含凭空捏造的数字、实体名称、年份或表名。
   注意：如果 memory_after 新增了关键数据却遗漏了，这也是忠实性问题。

2. **充分性（Sufficiency）**：
   - 如果 **未提供** 下一个问题（Next Question 为空），则该维度自动通过：sufficiency_pass=true, sufficiency_issues=[]。
   - 如果 **提供了** 下一个问题，只检查一个核心问题：**下一问中的指代词能否在 memory_after 中找到对应的实体？**

   **充分性只检查"指代解析"（referential resolution），不检查"数据完整性"**：
   - ✅ 检查：下一问提到"这两个类别"，memory_after 是否记录了当前讨论的是哪两个类别？
   - ✅ 检查：下一问提到"同期"或"上一轮的XX"，memory_after 是否能确定时间范围或具体实体？
   - ✅ 检查：下一问提到"该数据"，memory_after 是否能确定指的是哪个数据？
   - ❌ 不检查：下一问需要的具体数值（如某年某类的指数值）是否已存入 memory_after

   **重要判断原则**：
   - Memory 是上下文指针，不是数据缓存。agent 后续会自己查表获取数值。
   - 只要下一问涉及的实体名称、类别名称、表格位置、时间范围在 memory_after 中有记录，就应判定为通过。
   - **下一问要查询的新数据（本轮未查过的表格/子类），memory 不需要提前存储其数值**——只需要让 agent 知道去哪张表找即可。
   - 只有当指代词（如"它"、"这两个"、"上述"等）在 memory_after 中完全找不到对应实体时，才判定为不通过。

   **判定示例**：
   - 当前查了医疗保健子类数据，下一问要查衣着子类数据 → **通过**，因为下一问明确说了"衣着类"，没有歧义指代，agent 知道去哪里查
   - 当前查了2015年数据，下一问要查2019年数据 → **通过**，因为下一问明确说了"2019年"，没有歧义指代
   - 下一问说"这两个类别的价格恢复情况"，memory_after 没有记录当前讨论的是哪两个类别 → **不通过**

3. **连续性（Continuity）**：memory_after 是否保留了 memory_before 中的关键信息？
   - 已确认的表格不应消失
   - 时间范围不应变更
   - 单位约定应保持一致
   - 之前计算出的、未来可能用到的关键结果应保留
   - 如果本轮的答案更正了之前的某个信息，更新是允许的

4. **压缩性（Compression）**：memory_after 不应是历史的完整复制。
   - 不应包含长段的推理链
   - 不应复制完整的工具返回内容
   - 只保留对后续轮次有用的信息
   - 如果本轮有新发现，memory_after 应有相应的更新（不能和 memory_before 完全相同）

输出格式（JSON）：
{{
  "faithfulness_pass": true/false,
  "faithfulness_issues": ["具体的忠实性问题"],
  "sufficiency_pass": true/false,
  "sufficiency_issues": ["具体的充分性问题"],
  "continuity_pass": true/false,
  "continuity_issues": ["具体的连续性问题"],
  "compression_pass": true/false,
  "compression_issues": ["具体的压缩性问题"],
  "overall_pass": true/false,
  "rewrite_suggestion": "如果任一维度未通过，给出具体的修复建议。如果全部通过，留空。"
}}

任一维度未通过，overall_pass 应为 false，且 rewrite_suggestion 中给出可操作的修改指导。"""


def verify_memory(rec: dict, next_rec: dict = None, client=None, verbose: bool = False) -> dict:
    """
    验证单条子问题的 memory_after。
    """
    agent_trace = _build_agent_trace_text(rec.get('agent_steps', []))
    memory_before = json.dumps(rec.get('memory_before', {}), ensure_ascii=False, indent=2)
    memory_after = json.dumps(rec.get('memory_after', {}), ensure_ascii=False, indent=2)
    final_answer = json.dumps(rec.get('assistant_answer', {}), ensure_ascii=False)

    # 如果有下一个子问题，用于 sufficiency 检查
    next_question = next_rec.get('user', '') if next_rec else ''

    prompt = MEMORY_VERIFIER_PROMPT.format(
        memory_before=memory_before,
        user_question=rec.get('user', ''),
        agent_trace=agent_trace,
        final_answer=final_answer,
        memory_after=memory_after
    )
    if next_question:
        prompt += f"\n\n## Next Question (for sufficiency check)\n{next_question}"

    max_retries = 2
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat(
                prompt=prompt,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            result = extract_json_from_response(response)
            return result
        except Exception as e:
            last_error = e
            if verbose:
                print(f"    Verifier error (attempt {attempt+1}/{max_retries+1}): {e}")
    # 重试耗尽，返回失败
    if verbose:
        print(f"    Verifier failed after {max_retries+1} attempts")
    return {
        'faithfulness_pass': False,
        'faithfulness_issues': [f'Verifier failed after retries: {last_error}'],
        'sufficiency_pass': False,
        'sufficiency_issues': ['Verifier failed, unable to check sufficiency'],
        'continuity_pass': False,
        'continuity_issues': ['Verifier failed, unable to check continuity'],
        'compression_pass': False,
        'compression_issues': ['Verifier failed, unable to check compression'],
        'overall_pass': False,
        'rewrite_suggestion': 'Verifier failed after retries, re-run step7'
    }


def _build_agent_trace_text(agent_steps: list) -> str:
    parts = []
    for step in agent_steps:
        if step['type'] == 'tool_call':
            plan = step.get('step_plan', '')
            if plan:
                parts.append(f"[Plan] {plan}")
            for tc in step.get('tool_calls', []):
                if isinstance(tc, dict):
                    parts.append(f"[Tool] {tc.get('tool_name', 'unknown')}({json.dumps(tc.get('arguments', {}), ensure_ascii=False)})")
                else:
                    parts.append(f"[Tool] {tc}")
            for obs in step.get('observations', []):
                if isinstance(obs, dict):
                    parts.append(f"[Obs] {obs.get('content', '')[:800]}")
                else:
                    parts.append(f"[Obs] {str(obs)[:800]}")
        elif step['type'] == 'final_answer':
            ans = step.get('assistant_answer', {})
            parts.append(f"[Answer] {json.dumps(ans, ensure_ascii=False)}")
    return '\n'.join(parts)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description='Step 7: Verify compressed memory')
    parser.add_argument('--subquestions', type=str,
                        default=os.path.join(project_root, 'SFTbuild', 'output', 'subquestions_with_memory.jsonl'),
                        help='Path to subquestions_with_memory.jsonl from step6')
    parser.add_argument('--config_key', type=str, default='mimo',
                        help='LLM config key')
    parser.add_argument('--output_audit', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'output', 'memory_audit.jsonl'),
                        help='Output memory audit JSONL path')
    parser.add_argument('--output_pass', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'output', 'memory_verified_subquestions.jsonl'),
                        help='Output verified sub-questions JSONL path')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print prompt without calling LLM')
    parser.add_argument('--verbose', '-v', action='store_true', default=True)
    args = parser.parse_args()

    records = read_jsonl(args.subquestions)
    if not records:
        print(f"[ERROR] No records in {args.subquestions}")
        sys.exit(1)

    if args.dry_run:
        rec = records[0]
        agent_trace = _build_agent_trace_text(rec.get('agent_steps', []))
        prompt = MEMORY_VERIFIER_PROMPT.format(
            memory_before=json.dumps(rec.get('memory_before', {}), ensure_ascii=False, indent=2),
            user_question=rec.get('user', ''),
            agent_trace=agent_trace[:2000],
            final_answer=json.dumps(rec.get('assistant_answer', {}), ensure_ascii=False),
            memory_after=json.dumps(rec.get('memory_after', {}), ensure_ascii=False, indent=2)
        )
        print("=== Memory Verifier Prompt ===")
        print(prompt[:3000])
        return

    from src.utils.chat_api import ChatClient
    client = ChatClient(config_key=args.config_key)

    print(f"Loaded {len(records)} records")

    # 确保按 (sample_id, candidate_id, subquestion_id) 排序，用于 sufficiency 检查
    records.sort(key=lambda r: (r.get('sample_id', ''), r.get('candidate_id', ''), r.get('subquestion_id', 0)))

    audit_records = []
    passed_records = []

    for i, rec in enumerate(records):
        # 找下一个子问题（同 dialog 内）
        next_rec = records[i + 1] if i + 1 < len(records) else None
        if next_rec and (next_rec.get('sample_id') != rec.get('sample_id') or
                         next_rec.get('candidate_id') != rec.get('candidate_id')):
            next_rec = None

        if args.verbose:
            print(f"  Verifying {rec['candidate_id']}/sq{rec['subquestion_id']}...")

        result = verify_memory(rec, next_rec, client, verbose=args.verbose)

        audit_entry = {
            'sample_id': rec.get('sample_id'),
            'candidate_id': rec.get('candidate_id'),
            'subquestion_id': rec.get('subquestion_id'),
            **result
        }
        audit_records.append(audit_entry)

        if result.get('overall_pass'):
            rec['_memory_verified'] = True
            passed_records.append(rec)
        else:
            rec['_memory_verified'] = False
            rec['_memory_issues'] = result

    # 全部记录写入 pass 文件（含验证失败的），避免 dialog 出现缺口
    # 下游 step8 按 _memory_verified 筛选完整 dialog
    write_jsonl(args.output_audit, audit_records)
    write_jsonl(args.output_pass, records)

    n_pass = sum(1 for a in audit_records if a.get('overall_pass'))
    n_verified_only = len(passed_records)
    print(f"\nDone. Memory verification: {n_pass}/{len(audit_records)} passed")
    print(f"  All records ({len(records)}) → {args.output_pass} (verified-only subset: {n_verified_only})")
    print(f"  Audit: {args.output_audit}")


if __name__ == '__main__':
    main()
