# SFTbuild — SFT 数据构造流水线

从强模型 benchmark rollout 的原始 trace 出发，经拆解、评估对齐、筛选、修复、轨迹清洗、记忆生成、记忆验证，最终输出可直接训练 SFT 的轨迹数据。

## 流程概览

```
traces_output/  →  step2(decompose)  →  step3(align eval)  →  step4(filter)
                      ↓                        ↓                    ↓
              子问题切分+agent_steps      挂载evaluation        dialog/子问题筛选
                                                            ↓
                                              step5(repair)  ← 失败样本
                                              step5.5(clean) ← 9阶段轨迹清洗（确定性）
                                              step6(memory)  ← 清洗后样本
                                              step7(verify)  ← 记忆质检
                                                            ↓
                                                   step8(build)
                                                        ↓
                                               trainable_sft.jsonl
                                               trainable_sft_chat.jsonl
```

**Step5.5 轨迹清洗（9 阶段，全部确定性，无 LLM 调用）：**

```
Stage 1: detect_call_id_conflicts   → 同一 call_id 不同结果 → dialog 拒绝
Stage 2: deduplicate_log_calls      → 同一 call_id 完全重复 → 保留首次
Stage 3: filter_bf16_errors         → BFloat16/ScalarType 错误 → 按 call 粒度删除
Stage 4: remove_presentation_calls  → AST 判定纯展示调用 → 按 call 粒度删除
Stage 5: cleanup_orphaned_content   → 清理孤立 plan / 空步骤
Stage 6: reindex_tool_call_ids      → 全局重编号，确保 dialog 内 call_id 唯一
Stage 7: revalidate_trajectory      → 13 项最终校验
Stage 8: evidence verifier          → 答案数值可追溯到保留的 observation
Stage 9: build_chat_format          → 导出最终训练数据（在 step8 中执行）
```

**核心原则：按 call 粒度删除，而非按 step 粒度。** 一个 agent_step 可含多个并行 tool_call，仅删除命中条件的 call+observation，保留同 step 内其他有效 call。

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
python SFTbuild/step2_decompose.py       # 10条 trace → 43条子问题
python SFTbuild/step3_align_eval.py      # 挂载 accuracy/table_depend/等评估
python SFTbuild/step4_filter.py          # 筛选 → 4/10 dialog 通过
python SFTbuild/step5_repair.py          # LLM 修复失败子问题
python SFTbuild/step55_clean_trajectory.py # 轨迹清洗（BFloat16/重复/展示调用）
python SFTbuild/step6_memory_gen.py      # LLM 生成压缩记忆（基于清洗后轨迹）
python SFTbuild/step7_memory_verify.py   # LLM 验证记忆质量
python SFTbuild/step8_build_sft.py       # 导出 trainable_sft.jsonl
```

## 文件说明

| 文件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `utils.py` | 公共工具库：9阶段清洗流水线、语义数值提取（`extract_semantic_numeric_claims`）、证据验证（`_verify_evidence`）、答案格式校验（`validate_assistant_answer`）、LLM响应解析（`extract_json_from_response`） | — | — |
| `run_pipeline.sh` | 一键运行脚本，支持 `--dry-run` / `--skip-llm` | — | — |
| `step2_decompose.py` | 从累计 trace 按 checkout_list 切分子问题，解析 agent_steps | traces_output/, dataset/samples_*.json | output/aligned_subquestions.jsonl |
| `step3_align_eval.py` | 将 evaluation 结果对齐到子问题 | step2输出, traces_output/ | output/evaluated_subquestions.jsonl |
| `step4_filter.py` | 子问题级 + dialog 级筛选 | step3输出, samples | output/audit_report.jsonl, passed_subquestions.jsonl |
| `step5_repair.py` | 用强LLM迭代修复失败子问题（逐步决策+真实工具执行+答案验证） | step3输出, step4审计, dataset/tables/ | output/repaired_subquestions.jsonl |
| `step55_clean_trajectory.py` | 轨迹清洗：冲突检测→去重→BFloat16过滤→展示调用删除→孤立清理→重编号→13项校验→证据验证。全确定性，不涉及LLM。**必须在 memory 生成前运行** | step5输出 | output/cleaned_subquestions.jsonl, recovery_audit.jsonl |
| `step6_memory_gen.py` | 用强LLM生成压缩记忆（memory_before/after），按 dialog 逐轮累积，基于**清洗后**轨迹 | step55输出 | output/subquestions_with_memory.jsonl |
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

python SFTbuild/step55_clean_trajectory.py \
  --subquestions SFTbuild/output/repaired_subquestions.jsonl \
  --output SFTbuild/output/cleaned_subquestions.jsonl

python SFTbuild/step6_memory_gen.py \
  --subquestions SFTbuild/output/cleaned_subquestions.jsonl \
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
  repaired_subquestions.jsonl       # Step5: 修复结果（迭代执行+答案验证）
  cleaned_subquestions.jsonl        # Step5.5: 清洗后轨迹（BFloat16/重复/展示已删）
  recovery_audit.jsonl              # Step5.5: IndexError/NameError 恢复标签
  subquestions_with_memory.jsonl    # Step6: 带压缩记忆（基于清洗后轨迹）
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

## 轨迹清洗详解 (Step5.5)

### 9 个阶段

| 阶段 | 函数 | 说明 |
|------|------|------|
| 1 | `detect_call_id_conflicts()` | 同一子问题内相同 call_id 但不同内容 → dialog 拒绝 |
| 2 | `deduplicate_log_calls()` | 同一子问题内相同 call_id 完全相同 → 保留首次，删除后续 |
| 3 | `filter_bf16_errors()` | observation 含 `BFloat16` / `ScalarType` → 按 call 粒度删除 |
| 4 | `remove_presentation_calls()` | Python AST 判定纯展示调用（静态 print 最终答案）→ 删除 |
| 5 | `cleanup_orphaned_content()` | 清理空 step、被删除错误影响的 plan 引用 |
| 6 | `reindex_tool_call_ids()` | 全局重编号为 `call_{dialog}_{sq}_{seq}`，确保 dialog 内唯一 |
| 7 | `revalidate_trajectory()` | 13 项校验（tool_call↔observation 一一对应、无空 step、答案一致性等） |
| 8 | evidence verifier | 语义数值证据：答案中的数值声明必须可追溯到 observation |
| 9 | `build_chat_format()` | 导出最终训练格式（在 step8 中执行） |

### 核心原则

- **按 call 粒度删除**：一个 step 可含多个并行 tool_call，仅删除命中条件的 call+observation，保留同 step 内其他 call
- **阶段 1 失败 → dialog 直接跳过**（fail-closed，不可自动修复）
- **IndexError / NameError**：保留在训练数据中（含完整恢复链），标签写入独立的 `recovery_audit.jsonl`

### 13 项校验 (Stage 7)

1. tool_call step 至少有 1 个 tool_call 和 1 个 observation
2. 每个 tool_call_id 恰好对应 1 个 observation（双向一一对应）
3. tool_call 和 observation 数量一致
4. 不存在 `_dedup_conflicts`
5. 不存在空 agent_step
6. 不存在纯 JSON 展示型 Python 调用（阶段 4 二次确认）
7. 清洗前后 final_answer 的 (answer, sorted(data_source)) 不变
8. 保留的 IndexError/NameError 后面均存在成功恢复
9. 清洗后答案仍可由剩余 observation 支撑
10. 每个保留的错误恢复使用新的 call_id（阶段 6 保证）
11. IndexError/NameError 后存在 `[SUCCESS]` tool observation（完整恢复链）
12. 所有工具名称和参数通过当前 tools Schema 校验
13. 整个 dialog 的 tool_call_id 全局唯一

## 语义证据验证

证据验证使用逐子句语义数值提取（`extract_semantic_numeric_claims()`），按优先级从数字前缀判断语义符号：

```
优先级 1: 显式 '- ' 号  → 信任
优先级 2: "至/到/为" 目标水平  → 保持正数（回落至20.8% → +20.8）
优先级 3: "X幅/X幅度" 幅度描述  → 保持正数（降幅为82.1% → +82.1）
优先级 4: 负向变化（下降/减少/…）→ 取负（下降5.79% → -5.79）
优先级 5: 正向变化（增长/增加/…）→ 保持正数（增长5.79% → +5.79）
优先级 6: 无方向词            → 保持原始数值，不推断符号
```

关键特性：
- **逐子句隔离**："库存下降，但产量增长5.79%" 中 5.79 不受前句 "下降" 影响
- **双向一致**：答案文本和 observation 文本使用同一套语义提取，确保符号比较一致
- **多修饰词兼容**："下降了约5.79%"（了+约）、"下降了大约5.79%"（了+大约）均正确识别

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
