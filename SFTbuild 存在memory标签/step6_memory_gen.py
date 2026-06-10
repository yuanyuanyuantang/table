"""
Step 6: 从子问题 Trace 后处理生成压缩记忆。

输入:
  --subquestions  : step5 输出的 repaired_subquestions.jsonl（或 step4 的 passed_subquestions.jsonl）
  --config_key    : LLM 配置 key

输出:
  output/subquestions_with_memory.jsonl

记忆结构:
{
  "goal": "总任务目标",
  "tables": [{"name": "表格文件名", "content": "数据内容简述"}],
  "facts": ["已验证的原始数值、年份、单位、对象"],
  "derived": ["本轮计算出的结果"],
  "constraints": ["统计口径、时间范围"],
  "pitfalls": ["易错点"]
}

更新粒度: 一个子问题结束后更新一次 memory
memory_before_{i+1} = memory_after_i
"""
import copy
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SFTbuild.utils import read_jsonl, write_jsonl, extract_json_from_response

# memory 初始化模板
INITIAL_MEMORY = {
    "goal": "",
    "tables": [],
    "facts": [],
    "derived": [],
    "constraints": [],
    "pitfalls": []
}

# memory 标注 prompt
MEMORY_ANNOTATION_PROMPT = """你是一个专业的记忆压缩专家。你需要将智能体的执行轨迹压缩为结构化的记忆，供后续对话轮次使用。

## 总体任务目标
{goal}

## 之前的记忆（本轮子问题开始前的状态）
{memory_before}

## 当前问题
{user_question}

## 本轮的智能体执行步骤
{agent_trace}

## 最终答案
{final_answer}

## 任务
基于以上信息，生成本轮求解后的更新记忆。这个记忆将被后续子问题用来理解上下文。

**重要：你的记忆应该为可能的后续子问题提供足够的上文信息。** 请确保记忆包含：
- 本轮已确认的实体、类别名称、表格文件及其位置
- 时间范围、单位、统计口径等约束条件
- 对后续可能有用的计算结果和关键发现
- 容易出错的地方（陷阱）

但注意：记忆不是全文缓存，不要把所有原始数值复制进去。后续所需的详细数据应该由 agent 从表格中读取，记忆只需要让它知道"去哪里找、找什么"。

记忆结构及填写规则：

1. **goal（任务目标）**：保持与 memory_before 一致，不要修改。

2. **tables（已确认的数据表）**：
   - 保留 memory_before 中仍然有用的表格信息
   - 如果本轮发现了新的有用表格，追加进来
   - 如果本轮的答案更正了之前某个表格的信息，更新该条目
   - 每条包含表格名称和内容的简要说明

3. **facts（已验证的事实）**：
   - 保留 memory_before 中仍然有用的关键事实
   - 记录本轮从工具返回结果中确认的原始数值、年份、实体名称、单位
   - 只记录能在 agent_steps 的 observation 或 final_answer 中找到来源的事实
   - 严禁凭空编造任何数字、年份、实体名

4. **derived（推导结果）**：
   - 保留 memory_before 中未来可能用到的推导结果
   - 记录本轮计算得出的增长率、排名、对比结果等
   - 只记录本轮已验证的计算结果
   - 如果本轮的结论与之前某个推导矛盾，以本轮为准并更新
   - **严禁推测性因果分析**：不记录需要外部知识或因果推理的结论
     禁止示例: "可能与经济环境有关"、"表明XX导致了YY"、"可能预示着ZZ"、"反映出XX趋势"
     允许示例: "2023年增长率15.2%"、"A类价格103.5高于B类99.5"、"XX排名第3"

5. **constraints（约束条件）**：
   - 统计口径、时间范围、对比基准
   - 单位约定（如"单位：万辆"、"上年=100"）
   - 如果本轮没有新约束，保留 memory_before 的约束

6. **pitfalls（注意事项）**：
   - 容易出错的地方（如"累计值不可用于环比计算"、"该表有两列名为'车型'"）
   - 如果本轮发现了新的陷阱，追加进来

核心原则：
- **动态更新**：memory_before 中已有的信息可以被本轮更正或合并。不要机械地保留所有旧信息——如果旧信息已被本轮的新发现取代，应更新而非保留旧版。
- **有据可查**：所有新增内容必须能在 agent_steps 或 final_answer 中找到来源
- **面向未来**：记录对后续子问题可能有用的信息，不要全量复制历史
- **适度压缩**：合并相似条目，删除已失去后续价值的信息，避免 memory 持续膨胀
- **禁止泄露**：不要包含 score_points、evaluation feedback、gold answer

输出格式（JSON）：
{{
  "memory_after": {{
    "goal": "任务目标（从 memory_before 继承）",
    "tables": [
      {{"name": "文件名", "content": "数据内容简述"}}
    ],
    "facts": ["事实1", "事实2"],
    "derived": ["计算结果1", "计算结果2"],
    "constraints": ["约束条件1", "约束条件2"],
    "pitfalls": ["注意事项1"]
  }}
}}"""


def build_agent_trace_text(agent_steps: list) -> str:
    """将 agent_steps 转为文本描述供 LLM 参考"""
    parts = []
    for step in agent_steps:
        if step['type'] == 'tool_call':
            plan = step.get('step_plan', '')
            if plan:
                parts.append(f"[规划] {plan}")
            for tc in step.get('tool_calls', []):
                if isinstance(tc, dict):
                    parts.append(f"[工具调用] {tc.get('tool_name', 'unknown')}({json.dumps(tc.get('arguments', {}), ensure_ascii=False)})")
                else:
                    parts.append(f"[工具调用] {tc}")
            for obs in step.get('observations', []):
                if isinstance(obs, dict):
                    parts.append(f"[工具返回] {obs.get('content', '')[:800]}")
                else:
                    parts.append(f"[工具返回] {str(obs)[:800]}")
        elif step['type'] == 'final_answer':
            ans = step.get('assistant_answer', {})
            parts.append(f"[最终答案] {json.dumps(ans, ensure_ascii=False)}")
    return '\n'.join(parts)


def generate_memory(records: list, client, goal: str = "", verbose: bool = False) -> list:
    """
    为按 (sample_id, candidate_id) 排序后的记录生成 memory_before / memory_after。
    """
    # 按 (sample_id, candidate_id, subquestion_id) 排序
    records.sort(key=lambda r: (r.get('sample_id', ''), r.get('candidate_id', ''), r.get('subquestion_id', 0)))

    def _fresh_memory():
        """Deep-copy INITIAL_MEMORY so list values are not shared references."""
        return {k: list(v) if isinstance(v, list) else v
                for k, v in INITIAL_MEMORY.items()}

    # 按 dialog 分组
    current_dialog = None
    memory_after = _fresh_memory()

    for i, rec in enumerate(records):
        dialog_key = (rec.get('sample_id', ''), rec.get('candidate_id', ''))

        # 新的 dialog：重置 memory, goal 取自 --goal 参数或 sample_id
        if dialog_key != current_dialog:
            current_dialog = dialog_key
            memory_after = _fresh_memory()
            memory_after['goal'] = goal or rec.get('sample_id', '')

        # 当前轮 memory_before = 上一轮 memory_after
        rec['memory_before'] = copy.deepcopy(memory_after)

        # 构造 agent trace 文本
        agent_trace = build_agent_trace_text(rec.get('agent_steps', []))
        final_answer = json.dumps(rec.get('assistant_answer', {}), ensure_ascii=False)
        memory_before_str = json.dumps(memory_after, ensure_ascii=False, indent=2)

        prompt = MEMORY_ANNOTATION_PROMPT.format(
            goal=memory_after.get('goal', rec.get('sample_id', '')),
            memory_before=memory_before_str,
            user_question=rec.get('user', ''),
            agent_trace=agent_trace,
            final_answer=final_answer,
        )

        prev_memory = memory_after  # 保存旧值，用于异常回退
        try:
            response = client.chat(
                prompt=prompt,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            result = extract_json_from_response(response)
            if 'memory_after' not in result:
                raise ValueError(f"LLM response missing 'memory_after' key. "
                                 f"Got keys: {list(result.keys())}")
            memory_after = result['memory_after']
            if not isinstance(memory_after, dict):
                raise ValueError(f"memory_after is not a dict: {type(memory_after).__name__}")
            rec['memory_after'] = copy.deepcopy(memory_after)
            rec['_memory_generated'] = True
        except Exception as e:
            if verbose:
                print(f"  [ERROR] Memory generation failed for {dialog_key[1]}/sq{rec['subquestion_id']}: {e}")
            memory_after = prev_memory  # 恢复旧值，保证后续轮次链不断
            rec['memory_after'] = copy.deepcopy(memory_after)
            rec['_memory_generated'] = False
            rec['_memory_error'] = str(e)

        if verbose:
            n_tables = len(rec['memory_after'].get('tables', []))
            n_facts = len(rec['memory_after'].get('facts', []))
            print(f"  {rec['candidate_id']}/sq{rec['subquestion_id']}: memory tables={n_tables}, facts={n_facts}")

    return records


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description='Step 6: Generate compressed memory')
    parser.add_argument('--subquestions', type=str,
                        default=os.path.join(project_root, 'SFTbuild', 'output', 'cleaned_subquestions.jsonl'),
                        help='Path to cleaned sub-question records from step55')
    parser.add_argument('--config_key', type=str, default='mimo',
                        help='LLM config key')
    parser.add_argument('--goal', type=str, default='',
                        help='Overall task goal (if not set, uses sample_id)')
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'output', 'subquestions_with_memory.jsonl'),
                        help='Output JSONL path')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print prompt template without calling LLM')
    parser.add_argument('--verbose', '-v', action='store_true', default=True)
    args = parser.parse_args()

    records = read_jsonl(args.subquestions)
    if not records:
        print(f"[ERROR] No records in {args.subquestions}")
        sys.exit(1)

    print(f"Loaded {len(records)} sub-question records")

    if args.dry_run:
        rec = records[0]
        agent_trace = build_agent_trace_text(rec.get('agent_steps', []))
        prompt = MEMORY_ANNOTATION_PROMPT.format(
            goal=rec.get('sample_id', ''),
            memory_before=json.dumps(INITIAL_MEMORY, ensure_ascii=False, indent=2),
            user_question=rec.get('user', ''),
            agent_trace=agent_trace[:2000],
            final_answer=json.dumps(rec.get('assistant_answer', {}), ensure_ascii=False),
        )
        print("=== Memory Annotation Prompt (truncated) ===")
        print(prompt[:3000])
        return

    from src.utils.chat_api import ChatClient
    client = ChatClient(config_key=args.config_key)

    records = generate_memory(records, client, goal=args.goal, verbose=args.verbose)

    write_jsonl(args.output, records)
    print(f"\nDone. {len(records)} records → {args.output}")


if __name__ == '__main__':
    main()
