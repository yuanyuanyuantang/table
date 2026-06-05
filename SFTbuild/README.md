# SFTbuild — SFT 数据构造流水线

从强模型 benchmark rollout 的原始 trace 出发，经拆解、评估对齐、筛选、修复、记忆生成、记忆验证，最终输出可直接训练 SFT 的轨迹数据。

## 流程概览

```
traces_output/  →  step2(decompose)  →  step3(align eval)  →  step4(filter)
                      ↓                        ↓                    ↓
              子问题切分+agent_steps      挂载evaluation        dialog/子问题筛选
                                                            ↓
                                              step5(repair) ← 失败样本
                                              step6(memory) ← 通过+修复后样本
                                              step7(verify) ← 记忆质检
                                                            ↓
                                                   step8(build)
                                                        ↓
                                               trainable_sft.jsonl
                                               trainable_sft_chat.jsonl
```

## 一键运行

```bash
# 完整流水线 (Step2 → Step8)
bash SFTbuild/run_pipeline.sh

# 仅确定性步骤，跳过 LLM (Step5/6/7)
bash SFTbuild/run_pipeline.sh --skip-llm

# 预览 LLM prompt，不调用 API
bash SFTbuild/run_pipeline.sh --dry-run
```

也可以单独运行每一步：

```bash
python SFTbuild/step2_decompose.py    # 10条 trace → 43条子问题
python SFTbuild/step3_align_eval.py   # 挂载 accuracy/table_depend/等评估
python SFTbuild/step4_filter.py       # 筛选 → 4/10 dialog 通过
python SFTbuild/step5_repair.py       # LLM 修复失败子问题
python SFTbuild/step6_memory_gen.py   # LLM 生成压缩记忆
python SFTbuild/step7_memory_verify.py # LLM 验证记忆质量
python SFTbuild/step8_build_sft.py    # 导出 trainable_sft.jsonl
```

## 文件说明

| 文件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `utils.py` | 公共工具库（加载/切分/解析） | — | — |
| `run_pipeline.sh` | 一键运行脚本，支持 `--dry-run` / `--skip-llm` | — | — |
| `step2_decompose.py` | 从累计 trace 按 checkout_list 切分子问题，解析 agent_steps | traces_output/, dataset/samples_*.json | output/aligned_subquestions.jsonl |
| `step3_align_eval.py` | 将 evaluation 结果对齐到子问题 | step2输出, traces_output/ | output/evaluated_subquestions.jsonl |
| `step4_filter.py` | 子问题级 + dialog 级筛选 | step3输出, samples | output/audit_report.jsonl, passed_subquestions.jsonl |
| `step5_repair.py` | 用强LLM修复失败子问题（注入dataset真实表格数据保证跨轮一致性） | step3输出, step4审计, dataset/tables/ | output/repaired_subquestions.jsonl |
| `step6_memory_gen.py` | 用强LLM生成压缩记忆（memory_before/after），按 dialog 逐轮累积 | step5输出（或 step4 通过样本） | output/subquestions_with_memory.jsonl |
| `step7_memory_verify.py` | 验证记忆质量（faithfulness/sufficiency/continuity/compression），含健壮 JSON 解析 | step6输出 | output/memory_audit.jsonl, memory_verified_subquestions.jsonl |
| `step8_build_sft.py` | 构造最终 SFT 样本（结构化 + chat 格式），自动选择最优输入源 | step4/6/7输出 | output/trainable_sft.jsonl, trainable_sft_chat.jsonl |

## 参数覆盖

所有步骤都有合理的默认值，也可手动指定：

```bash
python SFTbuild/step2_decompose.py \
  --trace_dir traces_output \
  --samples dataset/samples_normal_easy.json \
  --output SFTbuild/output/aligned_subquestions.jsonl

python SFTbuild/step5_repair.py \
  --subquestions SFTbuild/output/evaluated_subquestions.jsonl \
  --audit SFTbuild/output/audit_report.jsonl \
  --config_key mimo \
  --dataset_root dataset/tables

python SFTbuild/step6_memory_gen.py \
  --subquestions SFTbuild/output/passed_subquestions.jsonl \
  --config_key mimo \
  --output SFTbuild/output/subquestions_with_memory.jsonl
```

## 输出目录结构

```
SFTbuild/output/
  aligned_subquestions.jsonl        # Step2: 子问题 + agent_steps
  evaluated_subquestions.jsonl      # Step3: 子问题 + eval
  audit_report.jsonl                # Step4: 筛选审计（按 dialog 汇总）
  passed_subquestions.jsonl         # Step4: 通过筛选的子问题
  repaired_subquestions.jsonl       # Step5: 修复结果（agent_steps 含真实表格数据）
  subquestions_with_memory.jsonl    # Step6: 带压缩记忆
  memory_audit.jsonl                # Step7: 记忆验证结果
  memory_verified_subquestions.jsonl # Step7: 验证后的子问题
  trainable_sft.jsonl               # Step8: 结构化 SFT 数据
  trainable_sft_chat.jsonl          # Step8: Chat 格式 SFT 数据
```

## 筛选标准

**子问题级（硬性）：**
- accuracy.coverage_ratio = 1.0
- table_depend.recall = 1.0
- answer JSON 可解析且非空
- data_source 覆盖 checkout_list[i].related_tables
- 无未恢复的工具错误

**Dialog 级：**
- 所有子问题均通过
- 子问题数与 checkout_list 一致

## 记忆结构

Step6 生成的记忆包含六个维度，按 dialog 逐轮累积更新：

```json
{
  "goal": "总任务目标",
  "tables": [{"name": "表格文件名", "content": "数据内容简述"}],
  "facts": ["已验证的原始数值、年份、单位、对象"],
  "derived": ["本轮计算出的结果（增长率、排名等）"],
  "constraints": ["统计口径、时间范围、比较基线"],
  "pitfalls": ["易错点（如累计值不可用于环比）"]
}
```

`memory_before_{i+1} = memory_after_i`，第一轮 `memory_before` 为空。

## 记忆验证维度 (Step7)

| 维度 | 说明 |
|------|------|
| Faithfulness | 事实必须来自 agent_steps 中的 observation / answer，禁止捏造 |
| Sufficiency | 下一轮问题的指代能否被当前 memory 解释（有下轮时检查） |
| Continuity | 跨轮不丢失已确认的关键信息（表格、年份范围、单位） |
| Compression | 不是完整历史的复制，仅保存后续有用的状态 |

注意：Step5/6/7 的 LLM 调用使用 `utils.extract_json_from_response()` 健壮解析（3种 fallback 策略），同时从 `content` 和 `reasoning_content` 中提取 JSON，确保无论模型将输出放在哪个字段都能正确解析。

## 最终 SFT 样本结构

### trainable_sft.jsonl（结构化格式，4 条 dialog）
```json
{
  "sample_id":    "任务描述文本",
  "task":         "任务描述文本（同上）",
  "table_path":   "dataset/table/chinese_table/...",
  "dialog_turns": [
    {
      "subquestion_id": 1,           // 第几轮
      "user":           "...",       // 干净的子问题文本
      "memory_before":  { goal, tables[], facts[], derived[], constraints[], pitfalls[] },
      "agent_steps": [
        {                              // type= tool_call
          "agent_step_id": 1,
          "type": "tool_call",
          "step_plan": "...",         // 动作意图（一句话）
          "tool_calls": [
            { tool_call_id, tool_name, arguments }
          ],
          "observations": [
            { tool_call_id, tool_name, content, success }
          ]
        },
        {                              // type= final_answer
          "agent_step_id": 10,
          "type": "final_answer",
          "assistant_answer": {
            "answer": "...",
            "data_source": ["file.xlsx"]
          }
        }
      ],
      "memory_after":   { goal, tables[], facts[], derived[], constraints[], pitfalls[] }
    }
  ]
}
```
memory_before_{i+1} = memory_after_i，首轮 memory_before 为空。

### trainable_sft_chat.jsonl（Chat 格式，4 条 dialog）
```json
{ "messages": [
    { "role": "system",   "content": "You are a professional table data analysis agent..." },

    { "role": "user",     "content": "<MEMORY_BEFORE>{...}</MEMORY_BEFORE>\n<QUESTION>子问题文本</QUESTION>" },

    { "role": "assistant", "content": "<PLAN>动作意图</PLAN>",
      "tool_calls": [{ "id", "type":"function", "function": { "name", "arguments" } }] },

    { "role": "tool",     "tool_call_id": "...", "content": "[SUCCESS] 工具返回结果..." },

    ... (多轮 tool_call/tool 交替) ...

    { "role": "assistant", "content": "<ANSWER>{...}</ANSWER>\n<MEMORY_AFTER>{...}</MEMORY_AFTER>" },

    { "role": "user",     "content": "<MEMORY_BEFORE>{上一轮memory_after}</MEMORY_BEFORE>\n<QUESTION>下一问</QUESTION>" },
    ...
  ]
}
```


**Chat 格式** 额外包含 `<MEMORY_BEFORE>`, `<QUESTION>`, `<PLAN>`, `<ANSWER>`, `<MEMORY_AFTER>` 标签，用于训练。

**关键原则：score_points / related_tables / evaluation feedback 不进入最终 SFT 样本，只留在 audit_report.jsonl 中用于审计。**
每轮用户消息包含上一轮的压缩记忆 (<MEMORY_BEFORE>)，模型需在此基础上推理
每轮最终 assistant 输出 <ANSWER> 和 <MEMORY_AFTER>（更新后的记忆）
tool_call 格式兼容 OpenAI function calling 标准
