"""
Step 8: 构造最终 SFT 样本并导出 trainable_sft.jsonl。

输入:
  --subquestions  : step7 输出的 memory_verified_subquestions.jsonl
  --samples       : benchmark 样本 JSON（用于获取 task / table_path 等元信息）

输出:
  output/trainable_sft.jsonl  — 最终训练数据

结构:
{
  "sample_id": "...",
  "task": "metadata.query",
  "table_path": "...",
  "dialog_turns": [
    {
      "subquestion_id": 1,
      "user": "干净的 checkout_list[i].info_item",
      "memory_before": {...},
      "agent_steps": [...],
      "memory_after": {...}
    }
  ]
}

关键原则:
- score_points / related_tables / evaluation feedback 不进入最终 SFT 样本
- assistant_plan 只保留短动作意图
- memory_before 从上一轮 memory_after 继承
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SFTbuild.utils import read_jsonl, write_jsonl, load_samples


def clean_agent_steps(agent_steps: list) -> list:
    """
    清洗 agent_steps，确保不含 gold 信息。
    """
    cleaned = []
    for step in agent_steps:
        if step['type'] == 'tool_call':
            cleaned_step = {
                'agent_step_id': step.get('agent_step_id'),
                'type': 'tool_call',
                'step_plan': step.get('step_plan', ''),
                'tool_calls': [],
                'observations': []
            }
            for tc in step.get('tool_calls', []):
                if isinstance(tc, dict):
                    cleaned_step['tool_calls'].append({
                        'tool_call_id': tc.get('tool_call_id', ''),
                        'tool_name': tc.get('tool_name', ''),
                        'arguments': tc.get('arguments', {})
                    })
                else:
                    cleaned_step['tool_calls'].append({
                        'tool_call_id': '',
                        'tool_name': str(tc),
                        'arguments': {}
                    })
            for obs in step.get('observations', []):
                if isinstance(obs, dict):
                    cleaned_step['observations'].append({
                        'tool_call_id': obs.get('tool_call_id', ''),
                        'tool_name': obs.get('tool_name', ''),
                        'content': obs.get('content', ''),
                        'success': obs.get('success', True)
                    })
                else:
                    cleaned_step['observations'].append({
                        'tool_call_id': '',
                        'tool_name': 'unknown',
                        'content': str(obs),
                        'success': True
                    })
            cleaned.append(cleaned_step)
        elif step['type'] == 'final_answer':
            ans = step.get('assistant_answer', {})
            cleaned.append({
                'agent_step_id': step.get('agent_step_id'),
                'type': 'final_answer',
                'assistant_answer': {
                    'answer': ans.get('answer', ''),
                    'data_source': ans.get('data_source', [])
                }
            })
    return cleaned


def build_sft_sample(sample_id: str, task: str, table_path: str,
                     sub_records: list) -> dict:
    """
    将同一样本的所有子问题记录组装为一条 SFT dialog。
    """
    # 按 subquestion_id 排序
    sub_records.sort(key=lambda r: r.get('subquestion_id', 0))

    dialog_turns = []
    for rec in sub_records:
        turn = {
            'subquestion_id': rec.get('subquestion_id'),
            'user': rec.get('user', ''),
            'memory_before': rec.get('memory_before', {}),
            'agent_steps': clean_agent_steps(rec.get('agent_steps', [])),
            'memory_after': rec.get('memory_after', {})
        }
        dialog_turns.append(turn)

    return {
        'sample_id': sample_id,
        'task': task,
        'table_path': table_path,
        'dialog_turns': dialog_turns
    }


def build_chat_format(sft_sample: dict, system_prompt_template: str = None) -> dict:
    """
    将 SFT 样本转为 chat format（messages 列表），用于训练。

    每轮格式:
      system: 表格 agent 规则 + 工具规则 + JSON answer 约束
      user: <MEMORY_BEFORE>...</MEMORY_BEFORE>\n<QUESTION>...</QUESTION>
      assistant: <PLAN>...</PLAN>\ntool_call(...)
      tool: 工具返回
      ...
      assistant: <ANSWER>{...}</ANSWER>\n<MEMORY_AFTER>{...}</MEMORY_AFTER>
    """
    if system_prompt_template is None:
        system_prompt_template = (
            "你是一个专业的表格数据分析智能体，能够通过工具调用逐步完成复杂的数据查询任务。\n\n"
            "## 工作流程\n"
            "1. 接收用户问题后，先阅读压缩记忆（<MEMORY_BEFORE>）了解已有上下文\n"
            "2. 制定执行计划（<PLAN>），明确本步要做什么\n"
            "3. 通过工具调用（tool_call）查找表格、读取数据、执行计算\n"
            "4. 根据工具返回结果，继续调用工具或给出最终答案\n"
            "5. 最终答案以 JSON 格式输出在 <ANSWER> 中，包含 'answer' 和 'data_source' 两个字段\n"
            "6. 更新压缩记忆（<MEMORY_AFTER>），供后续对话轮次使用\n\n"
            "## 注意事项\n"
            "- 每次工具调用前先写 <PLAN> 说明意图\n"
            "- 利用压缩记忆避免重复查询已确认的表格和事实\n"
            "- 计算时应仔细核对数值，避免单位换算错误"
        )

    messages = [{"role": "system", "content": system_prompt_template}]

    for turn in sft_sample.get('dialog_turns', []):
        # User message: memory_before + question
        mem_before = json.dumps(turn.get('memory_before', {}), ensure_ascii=False, indent=2)
        user_content = (
            f"<MEMORY_BEFORE>\n{mem_before}\n</MEMORY_BEFORE>\n\n"
            f"<QUESTION>\n{turn.get('user', '')}\n</QUESTION>"
        )
        messages.append({"role": "user", "content": user_content})

        # Assistant + Tool messages
        for step in turn.get('agent_steps', []):
            if step['type'] == 'tool_call':
                # Assistant: PLAN + tool_calls
                plan = step.get('step_plan', '')
                assistant_content = f"<PLAN>{plan}</PLAN>" if plan else ""
                assistant_msg = {
                    "role": "assistant",
                    "content": assistant_content
                }
                # tool_calls 以 OpenAI format 存储
                tc_list = []
                for tc in step.get('tool_calls', []):
                    if isinstance(tc, dict):
                        tc_list.append({
                            "id": tc.get('tool_call_id', ''),
                            "type": "function",
                            "function": {
                                "name": tc.get('tool_name', ''),
                                "arguments": json.dumps(tc.get('arguments', {}), ensure_ascii=False)
                            }
                        })
                    else:
                        tc_list.append({
                            "id": str(tc),
                            "type": "function",
                            "function": {
                                "name": "unknown",
                                "arguments": "{}"
                            }
                        })
                if tc_list:
                    assistant_msg['tool_calls'] = tc_list
                messages.append(assistant_msg)

                # Tool results
                for obs in step.get('observations', []):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": obs.get('tool_call_id', ''),
                        "content": obs.get('content', '')
                    })

            elif step['type'] == 'final_answer':
                ans = step.get('assistant_answer', {})
                answer_json = json.dumps(ans, ensure_ascii=False)
                mem_after = json.dumps(turn.get('memory_after', {}), ensure_ascii=False, indent=2)
                assistant_content = (
                    f"<ANSWER>\n{answer_json}\n</ANSWER>\n"
                    f"<MEMORY_AFTER>\n{mem_after}\n</MEMORY_AFTER>"
                )
                messages.append({"role": "assistant", "content": assistant_content})

    return {"messages": messages}


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description='Step 8: Build and export SFT data')
    parser.add_argument('--subquestions', type=str,
                        default=os.path.join(project_root, 'SFTbuild', 'output', 'passed_subquestions.jsonl'),
                        help='Path to passed_subquestions.jsonl from step4 (or step7 if memory enabled)')
    parser.add_argument('--samples', type=str,
                        default=os.path.join(project_root, 'dataset', 'samples_normal_easy.json'),
                        help='Path to samples JSON (optional, for task/table_path metadata)')
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'output', 'trainable_sft.jsonl'),
                        help='Output trainable SFT JSONL path')
    parser.add_argument('--output_chat', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'output', 'trainable_sft_chat.jsonl'),
                        help='Output chat-format SFT JSONL path')
    parser.add_argument('--verbose', '-v', action='store_true', default=True)
    args = parser.parse_args()

    records = read_jsonl(args.subquestions)
    if not records:
        print(f"[ERROR] No records in {args.subquestions}")
        sys.exit(1)

    # 加载 samples 获取元信息
    sample_map = {}
    if args.samples:
        samples = load_samples(args.samples)
        for s in samples:
            task = (s.get('task', '') or '').strip()
            if task:
                sample_map[task] = s

    # 按 (sample_id, candidate_id) 分组
    dialogs = {}
    for rec in records:
        key = (rec.get('sample_id', ''), rec.get('candidate_id', ''))
        dialogs.setdefault(key, []).append(rec)

    print(f"Grouped {len(records)} sub-questions into {len(dialogs)} dialogs")

    sft_samples = []
    chat_samples = []
    skipped = 0

    for (sample_id, candidate_id), sub_recs in dialogs.items():
        # 只保留全部子问题质量达标的 dialog：
        #   _sq_pass: step4 子问题级通过
        #   _repaired: step5 修复成功
        #   _memory_verified: step7 记忆验证通过
        if not all((r.get('_sq_pass') or r.get('_repaired')) and r.get('_memory_verified', True) for r in sub_recs):
            skipped += 1
            if args.verbose:
                n_bad = sum(1 for r in sub_recs if not ((r.get('_sq_pass') or r.get('_repaired')) and r.get('_memory_verified', True)))
                print(f"  [SKIP] {candidate_id[:50]}... : {n_bad} sub-questions unrepaired or unverified")
            continue

        sample = sample_map.get(sample_id, {})
        task = sample.get('task', sample_id)
        table_path = sample.get('table_path', sample.get('file_path', ''))

        sft_sample = build_sft_sample(sample_id, task, table_path, sub_recs)
        sft_samples.append(sft_sample)

        chat_sample = build_chat_format(sft_sample)
        chat_samples.append(chat_sample)

        if args.verbose:
            n_turns = len(sft_sample['dialog_turns'])
            n_steps = sum(len(t['agent_steps']) for t in sft_sample['dialog_turns'])
            print(f"  {sample_id[:60]}... : {n_turns} turns, {n_steps} agent_steps")

    # ==============================
    # ========= 最终输出 ============
    write_jsonl(args.output, sft_samples)
    write_jsonl(args.output_chat, chat_samples)

    print(f"\nDone.")
    if skipped:
        print(f"  Skipped: {skipped} dialogs (sub-questions not all passed/repaired)")
    print(f"  SFT dialogs:     {len(sft_samples)} → {args.output}")
    print(f"  Chat format:     {len(chat_samples)} → {args.output_chat}")


if __name__ == '__main__':
    main()
