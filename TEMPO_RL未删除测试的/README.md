# TEMPO-RL：多轮表格智能体的证据引导强化学习

TEMPO-RL 是 TableAgentBench 项目的强化学习训练系统。它在 SFT 模型的基础上，通过 GRPO 风格的策略优化，分别提升模型的**工具证据获取**、**答案证据支撑**和**跨轮记忆更新**三项能力。

---

## 目录

- [核心思想](#核心思想)
- [项目结构](#项目结构)
- [四阶段流水线](#四阶段流水线)
- [文件速查表](#文件速查表)
- [快速开始](#快速开始)
- [命令行参考](#命令行参考)
- [数据格式](#数据格式)
- [配置与超参数](#配置与超参数)
- [测试](#测试)

---

## 核心思想

TableAgentBench 的一个完整 dialog 包含 N 个按顺序依赖的子问题。每个子问题内部又有多步工具调用 → 工具返回 → 计算 → 回答。模型还需要在子问题之间维护一段压缩记忆（memory），把前面找到的关键信息传递给后面。

TEMPO-RL 将这个过程建模为**双时间尺度**的决策：

| 尺度 | 做什么 | 优化什么 |
|------|--------|----------|
| **Micro（工具层）** | 子问题内部：生成 plan → 调用工具 → 读 observation → 积累证据 | 每一步工具调用是否带来了新证据 |
| **Macro（记忆层）** | 子问题之间：回答后生成 memory，传给下一个子问题 | memory 是否忠实、压缩、包含后续真正需要的信息 |

强化学习不改变模型的基本行为格式（SFT 已经教会了），而是在此基础上用奖励信号引导模型做得更好。

### 奖励公式概览

**工具奖励**（每一步）：`r_tool = η × ΔΦ - λ_call - λ_invalid - λ_repeat`
- `ΔΦ`：本次工具调用带来的证据账本覆盖率增量
- 扣减项：工具调用成本、无效调用惩罚、重复调用惩罚

**答案奖励**（每个子问题）：`r_answer = Σ(w_c × C_correct × G) / Σw_c - λ_format × I_format - λ_extra × N_extra`
- `C_correct`：答案 claim 与 score_point 是否一致
- `G`（grounded）：答案 claim 是否能由 observation / memory / 计算结果支撑
- 完全无法解析的非法答案格式 → `r_answer = -1.0`

**记忆奖励**（Phase 2 起）：`r_mem = α_f × F_i + α_s × S_i - λ_comp × P_comp`
- `F_i`（faithfulness）：memory 每一项是否忠实于原始 observation
- `S_i`（future dependency coverage）：memory 保留了多少后续子问题需要的信息
- `P_comp`（compression penalty）：memory 过长时的惩罚

**反事实记忆奖励**（Phase 3）：`r_mem_final = r_mem + λ_cf × (1[F_i ≥ τ_f] × [ΔU]_+ + [ΔU]_-)`
- 额外跑一次"用上一轮 memory"的对照实验，看模型自己生成的 memory 是否真的带来了后续回答质量的提升
- `ΔU = r_ans(M_i^gen) - r_ans(M_{i-1})`，`[x]_+ = max(x, 0)`，`[x]_- = min(x, 0)`
- 正向 ΔU 需要 faithfulness gate（`F_i ≥ τ_f`），负向 ΔU 始终计入（不忠实的 memory 对未来有害必须惩罚）

---

## 项目结构

```
TEMPO_RL/
├── __init__.py                       # 包入口，Phase 0 立即导入，Phase 1-3 延迟导入
│
│   # ===== Phase 0：奖励基础设施 =====
├── schemas.py                        # 数据类：EvidenceItem, FutureDependency, AuditInfo 等
├── build_target_evidence.py          # 从 benchmark 样本构建 target_evidence.jsonl
├── build_future_dependencies.py      # 构建跨子问题信息依赖关系
├── verifier.py                       # 证据验证函数（C_value, C_source, C_binding 等）
├── evidence_ledger.py                # 证据账本，跟踪已验证证据和覆盖率
├── reward_calculator.py              # 奖励计算器（工具/答案/记忆奖励）
├── io_utils.py                       # 共享工具：JSONL 读写、响应解析、prompt 模板
├── run_phase0_audit.py               # 离线审计脚本，验证 Phase 0 输出
│
│   # ===== Phase 1：工具 + 答案 RL（单轮） =====
├── rollout_phase1.py                 # 对每个子问题采样 K 条轨迹，执行工具，计算奖励
├── build_segment_returns.py          # 将 rollout 转为 segment 级训练数据（tool + answer）
├── train_phase1.py                   # GRPO 训练器（工具段 + 答案段）
│
│   # ===== Phase 2：记忆 RL（多轮 dialog） =====
├── rollout_phase2.py                 # 完整 dialog rollout（模型自己写 memory）
├── build_segment_returns_phase2.py   # Segment 训练数据（tool + answer + memory）
│
│   # ===== Phase 3：反事实记忆 RL =====
├── counterfactual_phase3.py          # 稀疏反事实评估，估计 memory 边际效用
│
│   # ===== 测试 =====
├── tests/
│   ├── test_schemas.py               # (schemas 无独立测试，通过其他测试覆盖)
│   ├── test_target_evidence.py       # 目标证据构建测试
│   ├── test_future_dependencies.py   # 未来依赖检测测试
│   ├── test_evidence_ledger.py       # 证据账本测试
│   ├── test_reward_calculator.py     # 奖励计算器测试（~84 tests）
│   ├── test_phase0_audit.py          # Phase 0 审计测试
│   ├── test_rollout_phase1.py        # Phase 1 rollout 测试
│   ├── test_rollout_phase2.py        # Phase 2 rollout 测试
│   ├── test_build_segment_returns.py # Phase 1 segment builder 测试
│   ├── test_build_segment_returns_phase2.py  # Phase 2 segment builder 测试
│   ├── test_train_phase1.py          # 训练器测试（概率比、clip 损失、mask 等）
│   └── test_counterfactual_phase3.py # Phase 3 反事实测试
│
│   # ===== 规格文档 =====
└── TEMPO-RL.md                       # 完整方法规格（中英双语，1238 行）
```

---

## 四阶段流水线

### 阶段关系图

```
benchmark samples  ──→ Phase 0 ──→ target_evidence.jsonl
                    │             └→ future_dependencies.jsonl
                    │
                    ├──→ Phase 1 ──→ phase1_rollouts.jsonl
                    │               └→ segment_returns.jsonl ──→ 训练（工具+答案）
                    │
                    ├──→ Phase 2 ──→ phase2_dialog_rollouts.jsonl
                    │               └→ segment_returns.jsonl ──→ 训练（+记忆）
                    │
                    └──→ Phase 3 ──→ 反事实记忆奖励调整
                                    （在 Phase 2 rollout 基础上）
```

### Phase 0：奖励基础设施（离线，不需要 GPU）

**做什么**：从 benchmark 样本中提取每个子问题的"目标证据"和"跨子问题信息依赖"。

1. `build_target_evidence.py`：读取 benchmark 样本，规则提取 + LLM 补充，生成 `target_evidence.jsonl`
   - 每个子问题的每个 score_point 至少产生一个 EvidenceItem
   - 类型：`raw_value` / `derived_value` / `text_fact`
   - 附带 source_tables、entity、time、metric、unit 等字段

2. `build_future_dependencies.py`：分析子问题间的信息依赖
   - 在每个 memory boundary（子问题 i 回答后），预测后续子问题 j 需要之前哪些信息
   - 依赖类型：`numeric_fact` / `entity_set` / `table_ref` / `constraint` / `reference`
   - 窗口大小由 `D_FDC` 控制（默认 2，即看未来 2 步）

3. `run_phase0_audit.py`：验证 Phase 0 输出，生成审计报告

**输出文件**：
| 文件 | 内容 |
|------|------|
| `target_evidence.jsonl` | 每个子问题的目标证据集合 |
| `future_dependencies.jsonl` | 每个 memory boundary 的未来依赖集合 |
| `phase0_report.md` | 审计报告 |

### Phase 1：工具 + 答案 RL（子问题级 GRPO）

**做什么**：独立采样每个子问题，训练模型更好地调用工具和给出有据可依的答案。

1. `rollout_phase1.py`：
   - 对每个子问题采样 K 条轨迹（默认 K=4）
   - 每条轨迹：模型生成 tool_call → 执行工具 → 更新证据账本 → 循环直到给出 answer 或达到 max_tool_steps
   - 记录每一步的 evidence coverage 变化、tool reward、answer reward

2. `build_segment_returns.py`：
   - 将 rollout 转为 segment 级训练数据
   - 工具段：tool return-to-go（每个工具步骤的累积折扣奖励，discount γ_tool）
   - 答案段：answer return（直接使用 rollout 的 r_answer）
   - 按 segment 类型（tool / answer）分别做 group-relative 归一化，计算 advantage

3. `train_phase1.py`：
   - GRPO 风格 clipped policy optimization，无独立 value model
   - 优势函数：`A = (R - μ_group) / (σ_group + ε)`（同 segment 类型内归一化）
   - 目标函数：`L = -min(ρ × A, clip(ρ, 1-ε, 1+ε) × A)`，其中 ρ 是 token 级概率比
   - 只有 assistant 的 tool_call 和 answer token 参与训练（system/user/tool observation 被 mask）

**特点**：
- 不需要 critic / value model
- 不需要 KL 散度正则项
- Segment 类型归一化避免 tool 和 answer 的 reward scale 差异污染训练

### Phase 2：记忆 RL（dialog 级 GRPO）

**做什么**：在完整 dialog 上做 rollout，模型自己写 memory，训练记忆忠实度和信息保留能力。

1. `rollout_phase2.py`：
   - 一个 dialog 包含全部 N 个子问题，按顺序执行
   - 模型在每个子问题 answer 后生成 `<memory>...</memory>`，作为下一个子问题的上下文
   - 第一个子问题的 memory_before 为空，后续子问题使用模型上一轮生成的 memory

2. `build_segment_returns_phase2.py`：
   - Segment 类型增加为三种：tool / answer / memory
   - Memory reward 包含三项：faithfulness（忠实度）、future dependency coverage（覆盖度）、compression penalty（压缩惩罚）

**与 Phase 1 的关键区别**：
| | Phase 1 | Phase 2 |
|---|---|---|
| 粒度 | 单子问题 | 完整 dialog |
| Memory 来源 | 固定/无 | 模型自己生成 |
| Segment 类型 | tool, answer | tool, answer, memory |
| 记忆奖励 | 无 | F_i + S_i − compression |

### Phase 3：反事实记忆 RL（稀疏辅助）

**做什么**：估计模型自己写的 memory 对未来子问题的"边际效用"。

1. `counterfactual_phase3.py`：
   - 选取有拓扑依赖的子问题对（i ← j），做两组对照实验：
     - Continuation A：使用模型生成的 M_i^gen 作为 memory，继续回答子问题 j
     - Continuation B：使用上一轮 memory M_{i-1}（或空），继续回答子问题 j
   - 计算 ΔU = r_ans(A) − r_ans(B)，衡量 M_i^gen 相比旧 memory 带来的答案质量提升
   - 最终 memory reward 在 Phase 2 基础上加上反事实调整项

**特点**：
- 只在 20-30% 的 dialog 上触发（稀疏采样）
- 只在 memory 忠实度 F_i ≥ τ_f 时计入正向 ΔU
- 标注为"Phase 1+2 稳定后再加入"

---

## 文件速查表

### 你想了解某个概念去哪看

| 问题 | 看这个文件 | 关键函数/类 |
|------|-----------|------------|
| EvidenceItem 有哪些字段 | `schemas.py` | `EvidenceItem`, `required_fields_for_type()` |
| 目标证据怎么构建 | `build_target_evidence.py` | `TargetEvidenceBuilder.build_one_sample()` |
| 未来依赖怎么检测 | `build_future_dependencies.py` | `FutureDependencyBuilder.build_one_sample()` |
| 验证逻辑怎么写 | `verifier.py` | `verify_value_match()`, `verify_evidence_item()` |
| 证据账本怎么更新 | `evidence_ledger.py` | `EvidenceLedger.update()` |
| 奖励怎么算 | `reward_calculator.py` | `RewardCalculator.compute_tool_reward()` 等 |
| rollout 消息格式 | `rollout_phase1.py:569` / `rollout_phase2.py:604` | 工具 observation 的 role 和 content 格式 |
| segment 怎么构建 | `build_segment_returns.py` | `SegmentReturnBuilder.build()` |
| segment 的 mask 怎么重建文本 | `train_phase1.py:217` | `_message_to_text()` |
| 训练 loss 公式 | `train_phase1.py` | `SegmentGRPOTrainer.compute_loss()` |
| 概率比怎么算 | `train_phase1.py:471` | `_gather_token_log_probs()` |
| prompt 模板 | `io_utils.py` | `DEFAULT_SYSTEM_TEMPLATE`, `DEFAULT_SYSTEM_TEMPLATE_PHASE1` |
| sample_id 怎么统一提取 | `io_utils.py:199` | `get_sample_id()` |

### 你想改某件事去哪改

| 想改什么 | 改哪个文件 | 改什么位置 |
|----------|-----------|-----------|
| 奖励权重 | `reward_calculator.py` | `RewardCalculator.__init__()` 参数 |
| 训练超参 | `train_phase1.py` | CLI 参数 / `SegmentGRPOTrainer.__init__()` |
| 工具 step 上限 | `rollout_phase1.py:673` | `--max_tool_steps` |
| Rollout 采样数 K | `rollout_phase1.py:669` | `--K` |
| Discount factor | `build_segment_returns.py:530` | `--gamma_tool` |
| FDC 窗口大小 | `build_future_dependencies.py` | `FutureDependencyBuilder.__init__(d_fdc=2)` |
| 证据验证阈值 | `verifier.py` | 各 verify 函数内的模糊匹配阈值 |
| Prompt 模板 | `io_utils.py` | `DEFAULT_SYSTEM_TEMPLATE` 系列 |

---

## 快速开始

### 完整训练流程

以下命令假设你在项目根目录 `/data/zenghaoyang/TableAgentBench/` 下运行。

#### Step 1：Phase 0 构建奖励基础设施

```bash
# 运行 Phase 0 审计（会一并生成 target_evidence.jsonl + future_dependencies.jsonl）
python -m TEMPO_RL.run_phase0_audit \
    --samples dataset/train不含val的.json \
    --sft path/to/sft_trajectories.jsonl \
    --output_dir TEMPO_RL/output/phase0_audit \
    --max_samples 10
```

> `build_target_evidence.py` 和 `build_future_dependencies.py` 是库文件（无 CLI），由 `run_phase0_audit.py` 内部调用。

#### Step 2：Phase 1 Rollout（采样数据）

```bash
python -m TEMPO_RL.rollout_phase1 \
    --samples dataset/train不含val的.json \
    --target_evidence TEMPO_RL/output/target_evidence.jsonl \
    --table_root dataset/table \
    --output_dir phase1_output \
    --K 4 \
    --max_tool_steps 8 \
    --temperature 0.7 \
    --max_samples 10          # 先跑少量样本调试
```

#### Step 3：Phase 1 构建 Segment Returns

```bash
python -m TEMPO_RL.build_segment_returns \
    --input phase1_output/phase1_rollouts.jsonl \
    --output phase1_output/segment_returns.jsonl \
    --gamma_tool 0.95
```

#### Step 4：Phase 1 训练

```bash
python -m TEMPO_RL.train_phase1 \
    --model_path /path/to/sft_checkpoint \
    --segment_returns phase1_output/segment_returns.jsonl \
    --conversation_masks phase1_output/segment_returns_conversation_masks.jsonl \
    --output_dir phase1_train_output \
    --batch_size 1 \
    --max_steps 100 \
    --lr 1e-6 \
    --eps_clip 0.2
```

#### Step 5：Phase 2 Dialog Rollout

```bash
python -m TEMPO_RL.rollout_phase2 \
    --samples dataset/train不含val的.json \
    --target_evidence TEMPO_RL/output/target_evidence.jsonl \
    --future_dependencies TEMPO_RL/output/future_dependencies.jsonl \
    --table_root dataset/table \
    --output_dir phase2_output \
    --K 2 \
    --max_tool_steps_per_turn 6 \
    --temperature 0.7
```

> Phase 2 rollout 当前通过 `ChatClient`（API）调用模型，不支持本地 checkpoint 推理。如需用 Phase 1 训练出的 checkpoint 做 rollout，需要先将其部署为 API 或在 `rollout_phase2.py` 中添加本地推理支持。

#### Step 6：Phase 2 构建 Segment Returns

```bash
python -m TEMPO_RL.build_segment_returns_phase2 \
    --input phase2_output/phase2_dialog_rollouts.jsonl \
    --output phase2_output/segment_returns.jsonl \
    --gamma_tool 0.95
```

#### Step 7：Phase 2 训练（复用 Phase 1 训练器）

```bash
python -m TEMPO_RL.train_phase1 \
    --model_path phase1_train_output/checkpoint-100 \
    --segment_returns phase2_output/segment_returns.jsonl \
    --conversation_masks phase2_output/segment_returns_conversation_masks.jsonl \
    --output_dir phase2_train_output \
    --max_steps 100
```

#### Step 8（可选）：Phase 3 反事实

```bash
python -m TEMPO_RL.counterfactual_phase3 \
    --dialog_rollouts phase2_output/phase2_dialog_rollouts.jsonl \
    --samples dataset/train不含val的.json \
    --target_evidence TEMPO_RL/output/target_evidence.jsonl \
    --future_dependencies TEMPO_RL/output/future_dependencies.jsonl \
    --table_root dataset/table \
    --output_dir phase3_output \
    --sparse_rate 0.25 \
    --lambda_cf 0.2 \
    --tau_f 0.8
```

---

## 命令行参考

### Phase 0 Audit

| 参数 | 必需 | 说明 |
|------|------|------|
| `--samples` | 是 | benchmark 样本 JSON |
| `--sft` | 是 | SFT 轨迹 JSONL |
| `--output_dir` | 否 | 输出目录（默认 `TEMPO_RL/output/phase0_audit`） |
| `--max_samples` | 否 | 最多处理样本数 |
| `--validate_only` | 否 | 仅验证不重建 |

### Phase 1 Rollout

| 参数 | 必需 | 说明 |
|------|------|------|
| `--samples` | 是 | benchmark 样本 JSON |
| `--target_evidence` | 是 | target_evidence.jsonl 路径 |
| `--table_root` | 否 | 表格文件根目录 |
| `--output_dir` | 是 | 输出目录 |
| `--K` | 否 | 每子问题采样数（默认 4） |
| `--max_tool_steps` | 否 | 最大工具步数（默认 8） |
| `--temperature` | 否 | 采样温度（默认 0.7） |
| `--top_p` | 否 | nucleus sampling（默认 0.9） |
| `--max_samples` | 否 | 最多处理样本数 |
| `--build_te` | 否 | 自动构建 target evidence |
| `--mock` | 否 | 使用 mock policy 测试 |

### Phase 1 Segment Builder

| 参数 | 必需 | 说明 |
|------|------|------|
| `--input` | 是 | phase1_rollouts.jsonl 路径 |
| `--output` | 是 | segment_returns.jsonl 输出路径 |
| `--gamma_tool` | 否 | 工具折扣因子（默认 0.95） |
| `--kappa_ans` | 否 | 答案奖励权重（默认 1.0） |
| `--no_normalise` | 否 | 跳过归一化，使用原始 reward |

### Phase 1 Training

| 参数 | 必需 | 说明 |
|------|------|------|
| `--model_path` | 是 | SFT checkpoint 路径 |
| `--segment_returns` | 是 | segment_returns.jsonl 路径 |
| `--conversation_masks` | 是 | conversation_masks.jsonl 路径 |
| `--output_dir` | 否 | checkpoint 输出目录 |
| `--batch_size` | 否 | 训练 batch size（默认 1） |
| `--max_length` | 否 | 最大序列长度（默认 4096） |
| `--max_steps` | 否 | 最大训练步数 |
| `--lr` | 否 | 学习率（默认 1e-6） |
| `--eps_clip` | 否 | clip epsilon（默认 0.2） |
| `--update_old_every` | 否 | 每 N 步更新 π_old（0=不更新） |

---

## 数据格式

### target_evidence.jsonl

每行是一个 `TargetEvidenceSet`：

```json
{
  "sample_id": "task_001",
  "subquestion_id": 1,
  "question": "2020年产量增长了百分之多少？",
  "evidence_items": [
    {
      "evidence_id": "ev_sq1_001",
      "type": "raw_value",
      "value": "16.96%",
      "entity": "产量",
      "time": "2020",
      "metric": "增长率",
      "unit": "%",
      "source_tables": ["auto_2010.csv"],
      "weight": 1.0,
      "audit": {"parse_confidence": 0.7, "warnings": ["entity not detected by rule"], "source": "rule_extraction"}
    }
  ]
}
```

### future_dependencies.jsonl

每行是一个 `FutureDependencySet`：

```json
{
  "sample_id": "task_001",
  "boundary": "after_sq1",
  "future_dependencies": [
    {
      "dependency_id": "dep_sq1_1",
      "type": "numeric_fact",
      "source_evidence_id": "ev_sq1_001",
      "needed_by": "sq3",
      "fields": {
        "entity": "产量",
        "time": "2020",
        "metric": "增长率",
        "value": "16.96%",
        "unit": "%"
      },
      "weight": 1.0
    }
  ]
}
```

### phase1_rollouts.jsonl

**每行一条** rollout 记录。一个子问题对应 K 行（K 条独立采样轨迹），每条包含：
- `rollout_id`, `sample_id`, `subquestion_id`, `question`
- `agent_steps`：每一步的工具调用、observation、ledger 更新
- `assistant_answer`：模型最终答案
- `r_tool_steps`：每一步的工具奖励
- `r_answer`：答案奖励
- `ledger_trace`：证据覆盖率变化轨迹

### segment_returns.jsonl

每个 rollout 的 segment 和 conversation mask **分别存放在两个文件中**，通过 `rollout_id` 关联：

**segment_returns.jsonl** — 每行一个训练 segment（纯 advantage 数据，不含消息文本）：

```json
{
  "rollout_id": "task_001_sq1_k0",
  "segment_id": "task_001_sq1_k0_tool_0",
  "segment_type": "tool",
  "step_index": 0,
  "return_value": -0.02,
  "advantage": 0.35,
  "raw_return": -0.02,
  "trainable": true,
  "message_role": "assistant",
  "content_type": "tool_call"
}
```

**segment_returns_conversation_masks.jsonl** — 每行一个 rollout 的完整对话消息序列，每条消息标记 `trainable` 和 `content_type`：

```json
{
  "rollout_id": "task_001_sq1_k0",
  "status": "completed",
  "messages": [
    {"sequence_index": 0, "role": "system", "trainable": false, "content_type": "system_prompt", "content_preview": "You are..."},
    {"sequence_index": 1, "role": "user", "trainable": false, "content_type": "user_question", "content_preview": "2020年..."},
    {"sequence_index": 2, "role": "assistant", "trainable": true, "content_type": "tool_call", "response_text": "<tool_call>..."},
    {"sequence_index": 3, "role": "user", "trainable": false, "content_type": "tool_observation", "content_preview": "[Tool Result]..."},
    {"sequence_index": 4, "role": "assistant", "trainable": true, "content_type": "answer", "response_text": "<answer>..."},
    {"sequence_index": 5, "role": "assistant", "trainable": true, "content_type": "memory", "response_text": "<memory>..."}
  ]
}
```

训练时 trainer 两边加载，按 `rollout_id` 匹配，从 segment 取 advantage，从 mask 取消息文本和 token 是否训练。

Segment 类型：
- `tool`：一个工具步骤的 return-to-go
- `answer`：一个子问题的最终答案 return
- `memory`：一个子问题的 memory return（Phase 2 起）

### 对话消息的 role 约定

Rollout 中的 tool observation 使用统一格式：

```python
# 成功
{"role": "user", "content": "[Tool Result] [SUCCESS] actual data..."}

# 失败
{"role": "user", "content": "[ERROR] [ERROR] error message..."}
```

Segment builder 重建训练文本时必须与 rollout 完全一致，否则 policy ratio 计算上下文不匹配。

---

## 配置与超参数

### RewardCalculator 默认权重

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `eta` | 1.0 | 工具证据增益缩放 |
| `lambda_call` | 0.02 | 每次工具调用成本 |
| `lambda_invalid` | 1.0 | 无效工具调用惩罚 |
| `lambda_repeat` | 0.2 | 重复调用惩罚 |
| `lambda_format` | 1.0 | 答案格式错误惩罚 |
| `lambda_extra` | 0.5 | 额外无据 claim 惩罚 |
| `alpha_f` | 0.5 | 记忆忠实度权重 |
| `alpha_s` | 0.4 | 未来依赖覆盖度权重 |
| `lambda_comp` | 0.1 | 记忆压缩惩罚权重 |
| `B` | 512 | 记忆预算（token 数） |

### 训练超参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `gamma_tool` | 0.95 | 工具 return-to-go 折扣因子 |
| `kappa_ans` | 1.0 | 答案奖励在 return-to-go 中的权重 |
| `eps_clip` | 0.2 | GRPO clip ε |
| `lr` | 1e-6 | 学习率 |
| `K` | 4 | 每子问题 rollout 数 |

---

## 测试

```bash
# 运行全部测试（需要项目依赖）
python -m pytest TEMPO_RL/tests/ -v

# 只跑不依赖外部工具包的测试
python -m pytest TEMPO_RL/tests/ -v --ignore=TEMPO_RL/tests/test_rollout_phase1.py \
    --ignore=TEMPO_RL/tests/test_rollout_phase2.py \
    --ignore=TEMPO_RL/tests/test_counterfactual_phase3.py

# 按模块运行
python -m pytest TEMPO_RL/tests/test_reward_calculator.py -v
python -m pytest TEMPO_RL/tests/test_train_phase1.py -v
```

三个 rollout 测试（`test_rollout_phase1.py`、`test_rollout_phase2.py`、`test_counterfactual_phase3.py`）依赖 `src.tools.base` 包，只在完整服务器环境下可收集。
