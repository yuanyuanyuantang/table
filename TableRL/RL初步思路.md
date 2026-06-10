# TableAgent 完整 Dialog 细粒度过程奖励 GRPO

## 核心创新

```
工具调用 → 服务当前子问题（微观决策）
压缩记忆 → 服务后续子问题（宏观决策）

分层信用分配：当前工具步骤的局部质量 + 压缩记忆对后续子问题的长期价值
→ 同时纳入完整 Dialog 强化学习
```

只有完整 Dialog rollout 才能真实观察：

```
当前子问题生成的 memory_after
→ 是否帮助模型理解后续问题
→ 是否提高剩余子问题的完成率
```

---

# 一、总体目标

模型从 SFT checkpoint 初始化，在完整多轮任务中自主执行：

```
子问题 1 → 多步工具调用 → answer_1 + memory_1
子问题 2 → 仅用 memory_1 理解上下文 → 多步工具调用 → answer_2 + memory_2
...
子问题 N → answer_N + memory_N
```

**RL rollout 时必须在子问题边界真正裁剪历史**：下一子问题只接收 `memory_before`，不接收前序完整消息和 observation。否则模型仍能从完整历史中获取信息，无法证明 memory 有效。

训练目标同时包含：当前子问题工具调用质量、答案正确性、memory 对后续子问题的实际帮助、完整 Dialog 任务完成率。

---

# 二、分层 MDP 建模

## 微观决策：工具交互

第 `i` 个子问题中的第 `t` 个工具步骤：

| 元素 | 内容 |
|------|------|
| 状态 s_{i,t} | 当前总任务 + memory_before_i + 当前子问题 q_i + 子问题内已有 PLAN/tool_call/observation |
| 动作 a_{i,t} | PLAN + tool_call |
| 环境反馈 o_{i,t} | tool observation |

## 宏观决策：回答与记忆更新

子问题完成后：

| 元素 | 内容 |
|------|------|
| 宏观动作 m_i | assistant_answer_i + memory_after_i |

其中 `memory_before_{i+1} = memory_after_i`。前序子问题的完整消息、工具调用和 observation 不再传入下一子问题。

---

# 三、数据：必须先做扩充

当前 9 个 Dialog 只能验证 pipeline 是否跑通。完整 Dialog GRPO 的训练单位是 Dialog 数量，不是子问题数量。

```
当前 9 个 Dialog    → 仅验证 pipeline
Pilot 实验          → 至少 30-50 个完整 Dialog
正式实验            → 建议 100+ 个 Dialog，保留独立评测集
```

## 扩充方法

```
1. 用 SFT 模型在 dataset/samples_normal_easy.json 全量 samples 上做 rollout（temperature=0.0）
2. 跑 benchmark evaluator（accuracy/table_depend/format），筛选通过的 dialog
3. 通过的 dialog 按 step2-step55-step6 流水线切子问题 + 清洗 + 生成 memory
4. SFT 数据管线中的 quality gates（_trajectory_cleaned + _memory_verified + tool_call validation）二次筛选
5. 最终得到扩充后的 dialog 池
```

目标：Pilot 阶段 30-50 个高质量 Dialog，正式训练 100+ 个。

> 训练与评测不能使用相同 Dialog。建议按表格领域（家电/汽车/保险/工业材料/物价）划分 train/eval split。

---

# 四、完整 Dialog Group 构造

对于同一个 benchmark dialog，生成 K 条完整候选轨迹：

```
同一个 Dialog
├── Branch 1: sq1 → memory1 → sq2 → memory2 → ... → sqN
├── Branch 2: sq1 → memory1 → sq2 → memory2 → ... → sqN
├── Branch 3: sq1 → memory1 → sq2 → memory2 → ... → sqN
└── Branch 4: sq1 → memory1 → sq2 → memory2 → ... → sqN
```

每个 Branch：
- 从相同初始状态开始（相同 memory_before=∅、相同系统 prompt、相同 tools schema）
- 按相同子问题顺序执行
- 使用自己生成的 memory_after 作为下一轮的 memory_before
- 工具调用和 observation 均真实执行
- **子问题边界处裁剪历史**：下一子问题只看到 memory_before + 新问题 + tools schema

起步参数：

```
K = 4
temperature = 0.7-0.8
top_p = 0.9
max_tool_steps_per_subquestion = 8
```

---

# 五、失败与截断处理

某个子问题达到 `max_tool_steps` 但未生成 ANSWER 时：

```
该子问题：answer reward = 0, failure_penalty = -1
memory_after = memory_before（不更新，避免传播错误信息）
继续执行下一子问题
```

这样不同 Branch 始终按相同子问题顺序执行，既惩罚失败又不会丢失后续训练信号。

终止整个 Dialog 的条件：
- 超过 Dialog 总工具预算（如 40 步）
- 连续 2 个子问题无法生成合法输出
- 模型输出完全无法解析且无法继续

---

# 六、工具步骤奖励

## 6.1 规则奖励

每个工具调用得到局部奖励 p_{i,t}：

| 条件 | reward |
|------|--------|
| 非法工具名称或参数 | -1.0 |
| 重复调用相同工具与参数 | -0.3 |
| Python 仅用于格式化最终 JSON（AST 判定） | -0.3 |
| 模型导致的工具执行错误 | -0.3 |
| 工具错误后未在后续步骤恢复 | 额外 -0.4 |
| 合理恢复了模型导致的错误 | +0.1 |
| observation 为空或不可用 | -0.2 |
| 正常合法调用 | 0 |

基础设施错误不惩罚模型：

```
BFloat16、服务超时、环境故障 → reward = 0，记录异常并忽略
```

虽然 SFT 数据中错误步骤少（2/103），但 RL 探索期间模型会生成大量新错误参数、Python 异常和无效调用，因此**不能删除错误奖励**。

## 6.2 表格相关性

不在每次访问时加分（防止反复读取刷奖励）。子问题结束时，根据访问过的唯一表格集合计算：

```
table_recall = |accessed_tables ∩ gold_tables| / |gold_tables|
table_precision = |accessed_tables ∩ gold_tables| / |accessed_tables|
table_f1 = 2 × recall × precision / (recall + precision)
```

`table_f1` 作为子问题级奖励的一部分。

---

# 七、Groupwise LLM Judge

## 7.1 为什么必须组内比较

只评规则分最高的 2 条会导致：

```
高规则分轨迹 → 有 Judge 分 → reward 偏高
低规则分轨迹 → 无 Judge 分 → reward 偏低
→ Group 内 reward 不可比较
→ GRPO advantage 出现系统性偏差
```

## 7.2 组内比较方案

对同一子问题的 K 个候选做一次组内比较：

```
输入：
  同一个子问题 + K 条候选轨迹
  （每条含：PLAN + tool_call + observation + answer + memory_after）
  + 下一子问题文本（用于 sufficiency 判断，若为最后一个子问题则跳过）

输出：每条候选的分数
```

```json
{
  "candidates": [
    {
      "candidate_id": 1,
      "tool_necessity": 0.9,
      "execution_quality": 0.8,
      "overall_efficiency": 0.7,
      "memory_faithfulness": 1.0,
      "memory_sufficiency": 0.9
    }
  ]
}
```

**最后一个子问题**：没有 next question，sufficiency 维度跳过（或置为 1.0），只评 faithfulness + compression。

### Judge 信息边界（防 reward 泄露）

Judge 可用的信息：

```
✓ 当前子问题文本
✓ K 条 candidate 轨迹（PLAN + tool_call + observation + answer + memory_after）
✓ 下一子问题文本（仅用于判断 sufficiency，不包含其 gold answer）
✓ memory_before
```

Judge **不能**使用的信息：

```
✗ 下一子问题的 gold answer / score_points / evaluation
✗ 任何 benchmark 标注（related_tables、checkout_list 等）
✗ 完整 Dialog 的最终 gold 结果
```

这确保 Judge 判断"memory 是否足够支撑下一问"时，只能通过问题文本来推断，不能通过对比 gold answer 来作弊。如果 reviewer 质疑，可以明确展示 Judge prompt 中只包含下一问文本、不包含任何 gold 信息。

每个子问题只需一次 Judge 调用。假设每次更新采样 4 个 Dialog，每个 Dialog 平均 4 个子问题：

```
每次更新 = 4 × 4 = 16 次 Judge 调用
```

成本优化：
- 批量调用 Judge（一次 API call 处理多个子问题）
- 缓存相同轨迹评分（policy 收敛后轨迹重复率高）
- K 条轨迹完全相同时跳过 Judge（用规则分近似）

---

# 八、Memory 奖励

完整 Dialog rollout 中 memory 对后续子问题的影响可直接观察，不需要额外 continuation。

## 8.1 即时 Memory 质量

每个 `memory_after_i` 在两个维度上评估：

### Faithfulness（确定性规则）

证据范围：

```
memory_before_i + 当前子问题 observations + Python 执行结果 + assistant_answer_i
```

检查：
- 数值、年份、单位、表名是否有证据支撑
- 是否添加未经支持的结论
- 是否错误修改前序记忆中的事实

数值声明用确定性规则验证（复用 `utils.py` 中的 evidence verifier），非数值结论由 Judge 判断。

### Sufficiency（LLM Judge）

Judge 查看 `candidate memory_after_i + 下一子问题文本`，判断：

```
下一问中的指代词能否在 memory 中找到对应实体？
年份、单位、类别名称是否在 memory 中可获取？
memory 是否包含了下一问需要的表格位置信息？
```

注意：sufficiency 不是"memory 是否包含下一问需要的数值"，而是"memory 是否让 agent 知道去哪里找"。

**最后一个子问题**：没有 next question，sufficiency 评分跳过（置为 1.0）。

**信息边界**：Judge 只能看下一问文本，**不能看下一问的 gold answer、score_points 或任何 benchmark 标注**。如果 Judge 能够通过对比 gold 答案来判断 memory 是否充分，就会造成 reward 泄露 — reviewer 会质疑模型并非学会了更好的 memory 策略，而是奖励函数偷看了答案。

### Compression

不奖励"越短越好"，而是设置长度预算：

```
faithfulness ✓ 且 sufficiency ✓
且 memory token 数 ≤ 预算
→ compression 通过，无惩罚

超过预算 → 小幅惩罚（-0.1）
过短导致 sufficiency 失败 → 已经在 sufficiency 中惩罚
```

防止模型通过输出空 memory 投机。

## 8.2 延迟 Memory 价值

memory_after_i 的训练优势 = 即时 memory 质量 + 后续子问题累计奖励。

不需要单独为 memory 生成候选并跑 continuation。后续子问题的实际完成情况就是最好的 memory 评估。

---

# 九、Reward 汇总

## 9.1 子问题级奖励

第 `i` 个子问题：

```
R_i = 0.35 × answer_coverage
    + 0.15 × table_f1
    + 0.15 × evidence_consistency
    + 0.15 × tool_process_quality（规则 + Judge 归一化）
    + 0.15 × memory_quality（即时）
    + 0.05 × gated_efficiency
    + rule_penalties（非法工具、重复调用等）
```

各分量说明：

| 分量 | 来源 | 范围 |
|------|------|------|
| answer_coverage | benchmark evaluator 的 accuracy.coverage_ratio | 0-1 |
| table_f1 | 访问表格 vs gold related_tables（来自 benchmark sample） | 0-1 |
| evidence_consistency | 答案中数值声明在 observation/Python 结果中可找到的比例（确定性规则，复用 utils.py evidence verifier） | 0-1 |
| tool_process_quality | 规则分 + Judge 分归一化到 0-1 | 0-1 |
| memory_quality | faithfulness（规则）+ sufficiency（Judge）+ compression | 0-1 |
| gated_efficiency | 见下文 | 0-1 |
| rule_penalties | 非法工具 -1.0、重复 -0.3 等，直接累加（可为负） | - |

效率奖励必须通过正确性门控：

```
if answer_coverage < 0.8 or evidence_consistency < 0.8:
    gated_efficiency = 0
else:
    gated_efficiency = 1.0 - (工具步骤数 / max_tool_steps)
```

否则模型可能通过不调用工具、快速输出错误答案获得效率奖励。

## 9.2 Dialog 级额外奖励

`R_i` 已覆盖子问题级质量。Dialog 级只奖励子问题级无法捕捉的跨轮信号：

```
R_dialog_extra = 0.6 × all_subquestions_pass
               + 0.4 × cross_turn_consistency
```

- `all_subquestions_pass`：所有子问题均 answer_coverage ≥ 阈值 → 1，否则 → 0。鼓励模型稳定完成整个长任务。
- `cross_turn_consistency`（LLM Judge）：对象/年份/单位跨轮一致、后续回答正确继承前序事实、memory 无事实漂移、指代正确解析。

---

# 十、分层信用分配

不能把 `R_dialog` 直接赋给所有模型 token，否则无法区分哪个子问题/哪个步骤做得好。

## 10.1 子问题级未来回报

Branch `k` 的第 `i` 个子问题：

```
G_i^k = R_i^k + γ × R_{i+1}^k + γ² × R_{i+2}^k + ... + γ^{N-i} × λ_dialog × R_dialog_extra^k
```

γ = 0.9，λ_dialog = 0.3。

**为什么 R_dialog_extra 不再包含平均 R_i**：`G_i` 的前 N-i 项已经把每个子问题的 R 直接加进去了，Dialog 级的 `平均 R_i` 会在 G 中重复计算。`R_dialog_extra` 只放 Dialog 独有的信号（全部通过 + 跨轮一致），通过缩小的 λ_dialog 权重加入，避免干扰子问题级信用分配。

## 10.2 子问题级 Group Advantage

同一 Dialog 的 K 个 Branch，在相同子问题位置归一化：

```
A_i^k = (G_i^k - mean(G_i^{1..K})) / (std(G_i^{1..K}) + ε)
```

这样：
- 第一个子问题生成的 memory 受后续所有子问题结果影响
- 后面的子问题主要受当前和剩余任务结果影响
- Group 内比较的是同一子问题位置，更稳定

## 10.3 工具步骤 Advantage

每个工具步骤的 token 使用：

```
A_tool_{i,t}^k = A_i^k + η × clip(p_{i,t}^k, -1, 1)
```

其中 `p_{i,t}` 是规则 + Judge 给出的当前工具步骤质量，η = 0.2。

## 10.4 Token-Level Advantage 三类分派

不能把整个 assistant 动作的 token 共用同一个 advantage。至少分三类：

```
PLAN + tool_call JSON tokens → A_tool_{i,t}^k
  当前工具步骤质量 + 所在子问题的子问题级 advantage

ANSWER tokens → A_answer_i^k
  当前子问题答案质量（answer_coverage + evidence_consistency + table_f1）
  不接收后续子问题回报（答案不需要为未来负责）

MEMORY_AFTER tokens → A_memory_i^k
  memory 即时质量 + 后续子问题回报（γ R_{i+1} + γ² R_{i+2} + ...）
  这是论文核心主张：memory 为后续服务
```

其中：

```
A_answer_i^k = (R_answer_i^k - mean(R_answer_i^{1..K})) / (std(...) + ε)

R_answer_i = answer_coverage + evidence_consistency + table_f1

A_memory_i^k = A_i^k
  （即第 10.2 节的子问题级 group advantage，因为 G_i 已经包含了后续回报）
```

如果框架不支持 token-level mask 分类，至少实现 **segment-level**：把每个 assistant 消息拆成三段分别算 loss：

```
msg[assistant] → segment_1: content (PLAN text)
               → segment_2: tool_calls JSON
               → segment_3: ANSWER + MEMORY_AFTER（第一版可合并，后续再分开）
```

**第一版最低要求**：PLAN+tool_call 和 ANSWER+MEMORY_AFTER 至少分开。后续升级为三类分派。

## 10.5 完整训练目标

policy π_θ 初始化为 SFT checkpoint，reference policy π_ref 固定为同一 SFT checkpoint（冻结）。

```
L = L_GRPO + β × KL(π_θ || π_ref)

L_GRPO = -E[A^k × log π_θ(a^k | s)]
KL(π_θ || π_ref) = E[log π_θ(a | s) - log π_ref(a | s)]
```

β = 0.01 起步，动态调整：

```
工具合法率 < 85% 或 answer_coverage 连续下降 → β × 2
β 调整后稳定 10 步 → 尝试 β × 0.5 恢复
```

KL 惩罚防止 RL 破坏 SFT 已经学好的工具调用格式和 memory 结构。没有 KL 的情况下，policy 可能为追求更高 reward 学会生成格式错误但碰巧得分高的输出。

---

# 十一、分段 Logprob Replay + Tool-Call Token Gate

## 11.1 核心问题

当前 SFT 数据使用 OpenAI-style 格式：

```json
{
  "role": "assistant",
  "content": "<PLAN>读取相关表格。</PLAN>",
  "tool_calls": [{"id": "...", "type": "function", "function": {"name": "table_head_reader", "arguments": "{...}"}}]
}
```

`content` 和 `tool_calls` 是两个独立字段。RL 训练时，**框架必须将 tool_call JSON 序列化为 chat template 文本 token，并纳入 logprob 计算**。

如果框架只对 `content` 文本算 logprob，把 `tool_calls` 当作外部结构化字段跳过 → tool_call 选择不会收到梯度 → **工具调用策略无法被 RL 优化**。

## 11.2 硬性 Gate：阶段 0 必须先验证

这是整个 RL 方案的最高优先级前置条件。**不通过则不能开始 reward/GRPO 实验**。

测试用当前 SFT 数据中的 2 个 Dialog（共 ~8 个子问题）：

### Gate 1：Tool-Call Token 可训练

```
1. 加载 SFT checkpoint 作为 policy
2. 取一个子问题的 SFT state_snapshot（system + user + tools_schema + memory_before）
3. 让模型生成 assistant 输出（PLAN + tool_call）
4. 保存：
   - 模型生成的原始 token_ids
   - 每个 token 的 logprob（old_logprob）
   - 解析后的 tool_call JSON（用于执行工具）
5. 用同一 state_snapshot 重跑 forward pass（replay）
6. 检查：
   ✓ content token（PLAN 文本）的 logprob 有限且非 None
   ✓ tool_call JSON token（tool_name + arguments）的 logprob 有限且非 None
   ✓ token 序列对齐（old_token_ids == new_token_ids）
   ✓ 同一 backend 下 |old_logprob - new_logprob| < 1e-5
   ✓ 跨 backend（如 vLLM rollout → HF replay）允许 < 1e-2
   
如果 tool_call JSON token 没有 logprob 或为 None → **FAIL**，框架不支持。
```

**logprob 一致性阈值说明**：vLLM 和 HuggingFace 的 attention 实现、CUDA kernel 版本、模板序列化细节可能产生微小数值差异。重要的是 token 对齐、mask 正确、logprob 有限且可复现，而不是追求跨 backend 的精度一致。如果跨 backend 误差超过 1e-2，应排查注意力实现或模板序列化差异。

### Gate 2：Token Mask 正确

```
1. 对完整的 assistant + tool + assistant + ... 序列做 forward pass
2. 检查 loss mask：
   ✓ PLAN token                 → mask=1（训练）
   ✓ tool_call JSON token       → mask=1（训练）
   ✓ ANSWER token               → mask=1（训练）
   ✓ MEMORY_AFTER token         → mask=1（训练）
   ✓ system prompt token        → mask=0（不训练）
   ✓ user message token         → mask=0（不训练）
   ✓ tools schema token         → mask=0（不训练）
   ✓ tool observation token     → mask=0（不训练）
   
如果 observation token 的 mask=1 → **FAIL**，模型会学习复制 observation。
```

### Gate 3：多轮上下文裁剪一致

```
1. 取 Dialog 的 sq1 → sq2 边界
2. 用 sq1 的 memory_after 作为 sq2 的 memory_before
3. 裁剪掉 sq1 的全部工具调用和 observation
4. 检查 sq2 的 state_snapshot 是否正确：
   ✓ 包含 memory_before（来自 sq1 的 memory_after）
   ✓ 包含 sq2 的问题文本
   ✓ 不包含 sq1 的工具调用和 observation
   ✓ tools schema 完整
5. 用裁剪后的 state 做一次 forward pass
6. 检查 logprob 与完整历史时的 logprob 关系：
   ✓ 两者可能不同（上下文变了），但应都是合法的 logprob 值
   ✓ 裁剪后的 logprob 不应退化到随机水平（说明模型确实在用 memory）
```

### Gate 4：完整 Dialog Rollout 端到端

```
1. 用 SFT 模型跑一个完整 2-3 轮 Dialog rollout
2. 使用真实工具执行
3. 子问题边界处裁剪历史（只传 memory_before）
4. 检查：
   ✓ 工具调用成功执行（解析正常、参数合法、返回 observation）
   ✓ 所有子问题均生成 ANSWER（完成 Dialog）
   ✓ 每个 assistant 动作的 logprob 成功保存
   ✓ 分段 replay 后的 logprob 与 rollout 时一致
5. 对比：完整历史 vs 仅 memory 的 answer_coverage
   → 如果裁剪历史后 answer_coverage 大幅下降，说明 SFT 模型尚未学会依赖 memory
```

## 11.3 Gate 判定

| 结果 | 行动 |
|------|------|
| 4 个 Gate 全部通过 | 可以开始 reward 设计 + GRPO |
| Gate 1/2 失败（logprob 不可得） | 框架不支持，需换框架（如改自建文本化 tool-call 格式） |
| Gate 3 失败（裁剪后退化严重） | SFT 模型需要先做 memory-only fine-tune，再进入 RL |
| Gate 4 失败（端到端跑不通） | 排查工具执行环境，修复后重新验证 |

## 11.4 分段 Replay 方案

Gate 通过后，训练时使用：

```
1. Rollout 时保存每个 assistant 动作的 state_snapshot（模型当时看到的完整上下文）
2. 训练时，用 state_snapshot 重算该动作的 logprob
3. 使用该动作的 advantage（含后续回报）作为权重
4. system/user/tools schema/tool observation tokens 全部 mask
5. 只优化 PLAN + tool_call JSON + ANSWER + MEMORY_AFTER tokens
```

因为子问题边界裁剪了历史，完整 Dialog 不能简单拼成线性序列训练——每个 assistant 动作只能看到它生成时的 state_snapshot。

---

# 十二、RL 框架建议

LLaMAFactory 适合 SFT，做在线工具 GRPO 大概率不够。建议调研：

| 框架 | 优势 | 劣势 |
|------|------|------|
| **veRL** | 字节开源，支持 tool-integrated GRPO，社区活跃 | 需要适配自定义工具环境 |
| **OpenRLHF** | 支持自定义 reward function + KL，成熟稳定 | tool 集成需自行扩展 |
| 自建 | 完全可控，适合研究 | 开发成本高 |

推荐 veRL 起步：其 hybrid-flow 模式支持 rollout 时调用真实工具，且已支持分段 replay 和 token-level mask。

---

# 十三、完整训练流程

```
阶段 0：基础设施验证
  1. 验证历史裁剪后 SFT 模型是否还能完成 Dialog
     （对比 full-history vs memory-only）
  2. 验证 tool-call token logprob 可正确计算
  3. 验证所有 reward 函数范围稳定
  4. 确认分段 replay 可行

阶段 1：短 Dialog 预热（2-3 个子问题的 Dialog）
  Outcome + Rule Tool + Memory Reward
  K = 4, γ = 0.9
  不加入 Judge（降低复杂度）
  验证完整 Dialog rollout 和分层信用分配稳定

阶段 2：加入 Judge + 完整奖励
  Outcome + Rule + Groupwise Judge + Memory + Dialog Reward
  扩展到 3-5 个子问题的 Dialog
  验证细粒度过程奖励增量

阶段 3：扩展到完整 Dialog 池
  使用所有训练 Dialog（含 5+ 个子问题的长任务）
  测试模型能否泛化到比训练更长的任务
```

每 N 步做 checkpoint 评估：
- 在留出的 eval Dialog 上跑完整 rollout
- 对比 SFT 基线：answer_coverage、memory sufficiency、工具合法率

Early stop：连续 20 步 outcome 不提升 或 工具合法率下降到 < 85%。

---

# 十四、实验设计

## 核心实验

| # | 实验组 | 奖励组成 |
|---|--------|---------|
| 1 | SFT | 无 RL |
| 2 | Outcome-only Dialog GRPO | 仅答案 + Dialog 完成奖励 |
| 3 | Rule Process Dialog GRPO | Outcome + 规则过程奖励 |
| 4 | Hybrid Process Dialog GRPO | Outcome + 规则 + Judge + Memory（完整版） |

核心假设：**组 4 > 组 2**，证明细粒度过程奖励比只看最终答案更适合表格 agent。

## 关键消融

| 消融 | 对比 |
|------|------|
| Memory reward 有效性 | 组 4 vs 组 4 无 memory reward |
| 后续回报分配给 memory | G_i 含后续 vs G_i 只含当前 R_i |
| Full-history vs Memory-only | rollout 时是否裁剪历史 |
| 工具过程奖励有效性 | 组 3 vs 组 2 |
| Groupwise Judge 增量 | 组 4 vs 组 3 |
| KL 正则必要性 | 有 KL vs 无 KL |

## 关键指标

- 完整 Dialog 全部通过率
- 平均子问题 answer_coverage
- 后续依赖型子问题准确率（需要前序 memory 才能回答的子问题）
- 相关表格 F1
- 证据可复现率
- 无效/重复工具调用率
- Memory faithfulness / sufficiency
- Memory token 数量分布
- 不同 Dialog 长度下的性能变化

---

# 最终路线

```
V1 完整 Dialog K=4 Rollout
  + 每个 Branch 独立维护压缩记忆
  + 子问题内真实工具执行
  + 历史在子问题边界真实裁剪
  + 规则奖励 + Groupwise LLM Judge
  + 子问题级 Return-to-Go（G_i = R_i + γ R_{i+1} + ...）
  + Memory 接收后续任务奖励
  + 分段 Logprob Replay
  + KL 正则（β = 0.01，防止 policy 偏离 SFT 太远）

V1 仍不做：
  ✗ 不训练独立 QRM
  ✗ 不做额外 continuation rollout
  ✗ 不生成大量 memory 候选离线比较
```
