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

from SFTbuild.utils import (
    read_jsonl, write_jsonl, load_samples,
    normalize_paths_in_messages, validate_tool_calls, has_unresolved_absolute_paths,
)


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
            "You are a professional table data analysis expert. Please strictly follow the process and rules below to accurately respond to user table data questions.\n\n"
            "# I. Role and Core Goal\n"
            "- **Role**: Professional Table Data Analysis Agent, proficient in table preprocessing, tool combination, and pandas programming.\n"
            "- **Goal**: Extract accurate information from tables and answer user questions through rigorous thinking and correct tool invocation.\n"
            "- **Stateless Tools**: The code executor is stateless; each execution is independent and does not retain previous execution results!\n"
            "- **Thinking + Tool Parallelism**: Think before acting. Output a short, verifiable action plan. Call multiple tools in parallel only when they are independent of each other (i.e., none depends on the output of another).\n\n"
            "# II. Task Environment\n"
            "> **Note: Please perform all operations within the environment directory and use absolute paths.**\n"
            "**Current Working Environment Path**: <TABLE_ROOT>\n"
            "- <TABLE_ROOT> is the table data root directory assigned for the current task.\n"
            "- All file operations must use absolute paths starting with <TABLE_ROOT>/.\n"
            "- Do not guess or fabricate file paths. Use table_selector, grep_search, or cmd_executor to locate tables first.\n"
            "- Only use file paths confirmed by tool results or compressed memory.\n"
            "- Do not add extra directory layers like dataset/tables after <TABLE_ROOT>.\n"
            "File reading/writing and operations outside the <TABLE_ROOT> directory are prohibited.\n\n"
            "# III. Output Requirements\n"
            "- **Tool-call responses**: When calling tools, output a short, verifiable plan in <PLAN>...</PLAN> to describe your intent, followed by the tool calls. Keep plans concise — describe what you intend to do, why, and how (which tools/tables to use).\n"
            "- **Final response**: Output only <ANSWER>...</ANSWER> followed by <MEMORY_AFTER>...</MEMORY_AFTER>. Do NOT include <PLAN> in the final response.\n"
            "- The final answer is fixed in JSON format within <ANSWER>...</ANSWER>, including two fields: `answer` providing the answer, and `data_source` explaining the source table(s) of the answer:\n"
            "```json\n"
            '{"answer": "Answer to the user question, providing the answer in text form.", "data_source": ["Table Name 1", "Table Name 2", ...]}\n'
            "```\n"
            "- After the final answer, output compressed memory in <MEMORY_AFTER>...</MEMORY_AFTER> to preserve confirmed tables and key findings for subsequent sub-questions.\n"
            "- Read <MEMORY_BEFORE>...</MEMORY_BEFORE> at the start of each sub-question to understand previously confirmed context.\n"
            "- The answer should be concise, directly providing key data and conclusions."
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

    # Normalize hardcoded paths to <TABLE_ROOT> placeholder
    messages = normalize_paths_in_messages(messages)

    # Build tools schema (OpenAI function calling format)
    from src.tools.base import get_tools_schema
    tools_schema = get_tools_schema()
    if not tools_schema:
        raise RuntimeError(
            "Tool schema is empty — cannot build SFT training data. "
            "Ensure all tool modules are importable and @register_tool decorators have executed."
        )
    tools = json.dumps(tools_schema, ensure_ascii=False)

    return {"messages": messages, "tools": tools}


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description='Step 8: Build and export SFT data')
    parser.add_argument('--subquestions', type=str,
                        default=os.path.join(project_root, 'SFTbuild', 'output', 'memory_verified_subquestions.jsonl'),
                        help='Path to memory_verified_subquestions.jsonl from step7')
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

    # ---- Early check: tools registry must be populated ----
    # If tool modules fail to import, get_tools_schema() returns [] and every
    # dialog will be silently skipped.  Fail early with a clear error.
    from src.tools.base import get_tools_schema
    tools_schema = get_tools_schema()
    if not tools_schema:
        print("[FATAL] Tool registry is empty — no tools are registered.")
        print("  Check that all tool modules under src/tools/ are importable")
        print("  and that @register_tool decorators have executed.")
        sys.exit(2)

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
    cleaned_skipped = 0

    for dialog_idx, ((sample_id, candidate_id), sub_recs) in enumerate(dialogs.items()):
        # 只保留全部子问题质量达标的 dialog：
        #   _dialog_pass: step4/5 dialog 级完整（子问题数一致等）
        #   _sq_pass: step4 子问题级通过
        #   _repaired: step5 修复成功
        #   _memory_verified: step7 记忆验证通过
        # ---- 质量门禁 1: Dialog 级完整性 ----
        # _dialog_pass from step4/5 is based on the original dialog. After step7
        # drops records (memory verification failed), the flag can be stale.
        # Re-verify completeness against the sample's checkout_list.
        sample = sample_map.get(sample_id, {})
        checkout_list = sample.get('design', {}).get('checkout_list', [])
        expected_count = len(checkout_list)
        actual_count = len(sub_recs)

        if expected_count <= 0:
            skipped += 1
            if args.verbose:
                print(f"  [SKIP] {candidate_id[:50]}... : "
                      f"sample_id not found in samples JSON — cannot verify dialog completeness")
            continue

        actual_ids = {r.get('subquestion_id') for r in sub_recs}
        expected_ids = set(range(1, expected_count + 1))
        id_ok = (actual_ids == expected_ids)
        if not id_ok:
            skipped += 1
            if args.verbose:
                missing = expected_ids - actual_ids
                extra = actual_ids - expected_ids
                detail = []
                if missing:
                    detail.append(f'missing sq={sorted(missing)}')
                if extra:
                    detail.append(f'extra sq={sorted(extra)}')
                print(f"  [SKIP] {candidate_id[:50]}... : "
                      f"dialog incomplete — {', '.join(detail)}")
            continue

        # ---- 质量门禁 2: 子问题级通过状态 ----
        if not all((r.get('_sq_pass') or r.get('_repaired')) and r.get('_memory_verified', False) for r in sub_recs):
            skipped += 1
            if args.verbose:
                n_bad = sum(1 for r in sub_recs if not ((r.get('_sq_pass') or r.get('_repaired')) and r.get('_memory_verified', False)))
                print(f"  [SKIP] {candidate_id[:50]}... : {n_bad} sub-questions unrepaired or unverified")
            continue

        # ---- 质量门禁 3: 不允许有任何 _repair_error ----
        repair_errors = [r.get('_repair_error') for r in sub_recs if r.get('_repair_error') is not None]
        if repair_errors:
            skipped += 1
            if args.verbose:
                print(f"  [SKIP] {candidate_id[:50]}... : _repair_error in sub-questions: {repair_errors[:3]}")
            continue

        # ============================================================
        # Trajectory must be cleaned before memory generation (step55).
        # Step8 only verifies the flag — no secondary modification.
        # ============================================================
        if not all(r.get('_trajectory_cleaned', False) for r in sub_recs):
            cleaned_skipped += 1
            if args.verbose:
                n_bad = sum(1 for r in sub_recs if not r.get('_trajectory_cleaned', False))
                print(f"  [SKIP] {candidate_id[:50]}... : {n_bad} sub-questions not trajectory-cleaned")
            continue

        # ---- 质量门禁 4: 清洗后工具调用校验 ----
        all_tool_issues = []
        for r in sub_recs:
            steps = r.get('agent_steps', [])
            issues = validate_tool_calls(steps)
            if issues:
                all_tool_issues.extend([f"sq{r.get('subquestion_id', '?')}: {i}" for i in issues])
        if all_tool_issues:
            skipped += 1
            if args.verbose:
                print(f"  [SKIP] {candidate_id[:50]}... : tool_call validation failed after cleaning:")
                for issue in all_tool_issues[:5]:
                    print(f"    - {issue}")
            continue

        # ============================================================

        task = sample.get('task', sample_id)
        table_path = sample.get('table_path', sample.get('file_path', ''))

        sft_sample = build_sft_sample(sample_id, task, table_path, sub_recs)
        chat_sample = build_chat_format(sft_sample)

        # ---- 质量门禁 5: chat format 中不允许残留未归一化路径 ----
        unresolved = has_unresolved_absolute_paths(chat_sample['messages'])
        if unresolved:
            skipped += 1
            if args.verbose:
                print(f"  [SKIP] {candidate_id[:50]}... : unresolved absolute paths:")
                for issue in unresolved[:3]:
                    print(f"    - {issue}")
            continue

        sft_samples.append(sft_sample)
        chat_samples.append(chat_sample)

        if args.verbose:
            n_turns = len(sft_sample['dialog_turns'])
            n_steps = sum(len(t['agent_steps']) for t in sft_sample['dialog_turns'])
            print(f"  {sample_id[:60]}... : {n_turns} turns, {n_steps} agent_steps")

    # ==============================
    # ========= 最终输出 ============
    write_jsonl(args.output, sft_samples)
    write_jsonl(args.output_chat, chat_samples)

    total_dialogs = len(dialogs)
    produced = len(sft_samples)
    skip_rate = (total_dialogs - produced) / max(total_dialogs, 1)

    print(f"\nDone.")
    if skipped:
        print(f"  Skipped (quality gates): {skipped} dialogs")
    if cleaned_skipped:
        print(f"  Skipped (trajectory not cleaned): {cleaned_skipped} dialogs")
    print(f"  SFT dialogs:     {produced}/{total_dialogs} → {args.output}")
    print(f"  Chat format:     {len(chat_samples)} → {args.output_chat}")

    # ---- Exit code: fail if no SFT data was produced from non-empty input ----
    if produced == 0:
        print(f"\n[FATAL] 0 SFT dialogs produced from {total_dialogs} input dialogs "
              f"({len(records)} sub-questions).")
        if skipped == total_dialogs:
            print("  All dialogs were rejected by quality gates. Check:")
            print("    - _dialog_pass flag (set by step4/5)")
            print("    - _sq_pass / _repaired flags (set by step4/5)")
            print("    - _memory_verified flag (set by step7)")
            print("    - Tool call validation (validate_tool_calls)")
            print("    - Unresolved absolute paths in chat format")
        sys.exit(3)

    # ---- Warning: abnormally high skip rate ----
    if skip_rate > 0.8:
        print(f"\n[WARN] High skip rate: {skip_rate:.0%} of dialogs rejected. "
              f"Only {produced} SFT samples produced.")


if __name__ == '__main__':
    main()
