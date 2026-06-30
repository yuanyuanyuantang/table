# TEMPO\-RL

**TEMPO\-RL: Evidence\-Guided Tool and Memory Optimization for Multi\-turn Table Agents**

**面向多轮表格智能体的证据引导工具调用与记忆策略优化**

## **0\. 核心目标**

TableAgentBench 的任务具有两个关键特点：

1. 一个完整 dialog 包含多个按顺序依赖的子问题。

2. 每个子问题内部又包含多步工具调用、工具返回、计算和最终回答。

SFT 阶段已经让模型学会了基本轨迹格式、工具调用方式、最终 JSON 回答和压缩记忆更新。RL 阶段不再从零教模型，而是进一步优化三类能力：

1\. **工具证据获取能力**：当前工具调用是否真正带来了当前子问题需要的新证据。

2\. **答案证据支撑能力**：最终答案中的 claim 是否被 observation、memory 或计算结果支撑。

3\. **跨轮记忆更新能力**：memory 是否忠实、压缩，并保留后续子问题真正需要的信息。

最终方法主线为：

> We formulate multi\-turn table\-agent reasoning as a dual\-timescale decision process and optimize it with a lightweight evidence ledger, grounded answer reward, and future\-aware memory reward\.
> 
> 我们将多轮表格智能体推理建模为双时间尺度决策过程，并借助轻量证据账本、基于事实的答案奖励以及具备前瞻意识的记忆奖励对其进行优化。
> 
> 

## **1\. 方法总览**

TEMPO\-RL 将多轮表格智能体建模为两个时间尺度上的联合决策过程：

\- **Micro\-level tool decision**：在一个子问题内部，模型不断生成 plan、调用工具、读取 observation，并逐步获得当前问题需要的证据。

\- **Macro\-level memory decision**：一个子问题结束后，模型生成 answer，并将已验证且未来可能有用的信息压缩为 \`memory\_after\`，作为下一子问题的长期上下文。

整体流程包含一个奖励基础设施阶段和三个渐进式 RL 阶段：

```Plain Text
Phase 0: Reward Infrastructure
Phase 1: Tool + Answer RL
Phase 2: On-policy Memory RL
Phase 3: Sparse Counterfactual Memory RL
```

第一阶段主实现完成 Phase 0、Phase 1、Phase 2。  

Phase 3 不进入第一阶段主训练循环，但如果最终论文将 counterfactual memory utility 作为主要贡献，Phase 3 必须进入 Full TEMPO\-RL 主实验。



## **2\. 双时间尺度建模**

设一个 benchmark dialog 包含 `N` 个子问题：

$D = (q_1, q_2, ..., q_N)$

第 `i` 个子问题开始前的压缩记忆为：

$M_{i-1}$

第 `i` 个子问题内部第 `t` 个工具步骤的状态定义为：

$s_{i,t} = (q_i, M_{i-1}, h_{i,t})$

其中：

- $q_i$：当前子问题。

- $M_{i-1}$：是策略模型上一轮原始生成的 memory。Verifier 只用于计算奖励和初始化已验证证据账本，不修改模型实际传递给下一轮的 memory。

- $h_{i,t}$：当前子问题内部截至第 `t` 步的局部历史，包括 plan、tool call 和 observation。

模型在状态 $s_{i,t}$ 下生成工具动作：

$u_{i,t} = (p_{i,t}, c_{i,t})$

其中：

- $p_{i,t}$：简短、可监督的行动计划。

- $c_{i,t}$：一个工具调用。

> 本实现中，每个 assistant tool turn 只解析和执行一个 tool call。模型理论上可以生成多个工具调用，但训练协议通过 prompt 和 tool schema 约束其每轮只输出一个调用；多工具并行执行留作后续工程扩展。
> 
> 

工具环境返回 observation：

$o_{i,t} = E(c_{i,t}; D_{table})$

当前子问题内部历史更新为：

$h_{i,t+1} = h_{i,t} ∪ {(p_{i,t}, c_{i,t}, o_{i,t})}$

当模型认为证据充分后，生成当前子问题的最终答案：

$y_i \sim π_θ(. | q_i, M_{i-1}, τ_i, segment=answer)$

随后生成新的压缩记忆：

$M_i \sim π_θ(. | M_{i-1}, q_i, τ_i, y_i, segment=memory)$

下一个子问题只接收新的压缩记忆，而不接收完整历史：

$s_{i+1,0} = (q_{i+1}, M_i, ∅)$

因此，完整系统自然形成一个双时间尺度的 Semi\-MDP：

- 子问题内部是可变长度工具交互。

- 子问题之间由 memory 进行长期状态转移。

## **3\. Phase 0：奖励基础设施**

Phase 0 不训练模型，只构建 RL 所需的 verifier 和 reward 组件。

### **3\.1 Target Evidence 构造**

对每个子问题构造目标证据集合：

$E_i^{req} = {e_{i,1}, e_{i,2}, ..., e_{i,n_i}}$

每个 evidence item 表示当前子问题需要获得、绑定或计算出的一个关键事实。

推荐 schema：

```JSON
{
  "sample_id": "...",
  "subquestion_id": 1,
  "evidence_id": "sq1_e1",
  "type": "raw_value",
  "value": "16.96%",
  "entity": "乘用车",
  "time": "2010年1月",
  "metric": "产量同比增长率",
  "unit": "%",
  "source_anchors": [
  {
    "file": "xxx.xlsx",
    "sheet": null,
    "region": null,
    "header_path": null
  }
],
  "operation": null,
  "weight": 1.0
}
```

对于计算结果：

```JSON
{
  "sample_id": "...",
  "subquestion_id": 1,
  "evidence_id": "sq1_e3",
  "type": "derived_value",
  "value": "18.46%",
  "entity": "乘用车",
  "time": "2010年1月",
  "metric": "出口量同比增长率",
  "unit": "%",
  "input_evidence_ids": [
    "sq1_export_current",
    "sq1_export_previous"
  ],
  "operation": "同比增长率",
  "weight": 1.0
}
```

derived evidence 进入 ledger 的条件为：

$C_{input}(e)=1,$

$C_{operator}(e)=1,$

$C_{result}(e)=1.$其中：

- $C_{input}(e)$: 所需输入来自已经验证的 evidence。

- $C_{operator}(e)$: 使用的计算操作与目标 operation 一致。

- $C_{result}(e)$: 输出结果可以由输入 evidence 重新计算得到。

Target evidence 可以从以下信息中构造：

- benchmark `score_points`

- gold answer

- `related_tables`

- SFT expert trajectory 中的 observation

- Python 计算结果

- 已验证 memory

- LLM 辅助标注的 entity、metric、time、unit、operation

第一版不要求完全自动化。可以使用规则抽取 \+ LLM 辅助 \+ 少量人工抽样校验。

### **3\.2 Evidence Ledger**

RL rollout 时，为每个子问题维护一个证据账本：

$L_{i,t}$

$L_{i,t}$ 表示截至第 $t$个工具步骤，已经被验证获得的 target evidence。

每个子问题开始时，ledger 不是空集，而是由上一轮策略模型原始生成的 memory 初始化：

$L_{i,0}
=
VerifyMem(M_{i-1}) ∩ E_i^{req}$

只有同时满足以下条件的 memory evidence 才能进入 L\_\{i,0\}：

$C_{value}(e)=1,$

$C_{binding}(e)=1,$

$C_{provenance}(e)=1.$

也就是说，memory 中的事实必须在数值、实体/时间/指标绑定、来源可追溯性上都通过验证，才能被视为当前子问题已经拥有的证据。

工具执行后，根据 observation 更新账本：

$L_{i,t+1} = UpdateLedger(L_{i,t}, c_{i,t}, o_{i,t}, E_i^{req})$

一个 evidence item 被认为获得，需要满足三类条件：

1\. **Value match**：目标数值或事实出现在 observation、memory 或代码执行结果中，或可由它们计算得到。

2\. **Source match**：工具 observation 的 sidecar metadata 与任一允许的 source anchor 匹配。对于 source anchor 中的空字段不进行检查；所有非空字段均匹配时，来源验证通过。

3\. **Binding match**：数值绑定到正确实体、时间、指标、单位和统计口径。

其中，简单数值和表格来源优先使用规则 verifier；实体\-数值绑定、趋势、比较、解释性 claim 使用 LLM grounding verifier 兜底。

### **3\.3 Evidence Coverage**

定义当前账本覆盖率：

$\operatorname{Cov}(L_{i,t}) = \frac{\sum_j w_j \cdot \mathbb{I}\left[e_{i,j} \in L_{i,t}\right]}{\max\left(\sum_j w_j,\, \varepsilon\right)}$

工具步骤带来的证据增量为：

$ΔΦ_{i,t} = Cov(L_{i,t+1}) - Cov(L_{i,t})$

只奖励首次获得的新证据。已经进入账本的 evidence 不重复计分。



## **4\. Phase 1：Tool \+ Answer RL**

Phase 1 的目标是优化当前子问题内部的工具调用和答案生成。  

这一阶段建议固定或使用 teacher/SFT memory，不训练 memory。

### **4\.1 Rollout 单位**

使用 subquestion\-level rollout。

每个 prompt 包含：

```Plain Text
system prompt
tools schema
memory_before
当前子问题 q_i
```

模型在线生成：

```Plain Text
PLAN + tool_call
tool observation
PLAN + tool_call
tool observation
...
ANSWER
```

达到 `max_tool_steps` 仍未输出 answer，则视为截断失败。

建议初始参数：

```Plain Text
K = 4
temperature = 0.7
top_p = 0.9
max_tool_steps = 6 或 8
```

### **4\.2 工具过程奖励**

第 `t` 个工具步骤的奖励：

$r_{i,t}^{tool}
= η * max(0, ΔΦ_{i,t})
- λ_{call}
- λ_{invalid} * I_{invalid}
- λ_{repeat} * I_{repeat}$

其中：

- $ΔΦ_{i,t}$：当前工具调用带来的 evidence coverage 增量。

- $I_{invalid}$：工具名不存在、参数 schema 错误、JSON 无法解析、调用格式不合法。

- $I_{repeat}$：重复调用且没有新增证据。

- $λ_{call}$：轻微工具调用成本。

建议初始权重：

```Plain Text
η = 1.0
λ_call = 0.02
λ_invalid = 1.0
λ_repeat = 0.2
```

合法工具调用本身不加分。只有当工具调用新增了目标证据时才获得正奖励。

### **4\.3 重复调用判定**

重复调用定义为：

$\text{same tool\_name} + \text{canonical}(\text{arguments}) + \text{no new evidence}$

其中 `canonical(arguments)` 包括：

- 路径归一化

- JSON 参数排序

- 表格范围归一化

- 搜索关键词规范化

- Python 代码 AST 级别的展示调用识别

如果一次工具调用重复读取同一表、同一区域，且账本无新增 evidence，则扣重复惩罚。

基础设施错误，例如工具服务异常或非策略导致的执行失败，不作为模型策略错误训练；对应 token 可以从 policy loss 中 mask。



### **4\.4 Answer Reward**

当前子问题结束后，计算 claim\-level grounded answer reward。  

设当前子问题的目标 claim 集合为：

$C_i^{req} = {c_{i,1}, c_{i,2}, ..., c_{i,m_i}}$

每个 claim $c$ 具有权重 $w_c$，并计算两个值：

- $C_i^{correct}(c)$：答案是否正确表达了该 claim。

- $G_i(c)$：该 claim 是否能由 observation、memory 或计算结果支撑。

Answer reward 定义为：

$r_i^{\mathrm{ans}}
= \frac{\sum_{c \in C_i^{\mathrm{req}}} w_c \cdot C_i^{\mathrm{correct}}(c) \cdot G_i(c)}{\max\left(\sum_{c \in C_i^{\mathrm{req}}} w_c,\, \varepsilon\right)}
- \lambda_{\mathrm{format}} \cdot \mathbb{I}_i^{\mathrm{format}}
- \lambda_{\mathrm{extra}} \cdot \mathbb{I}_i^{\mathrm{unsupported\_extra}}$

其中：

- $I_i^{format}$：ANSWER JSON 无法解析、字段缺失或字段类型非法。

- $I_i^{unsupported-extra}$：答案包含 score points 之外、且无法由证据支撑的额外解释、因果或推测性 claim。

- $G_i(c)$ 同时检查 value、source/provenance 和 binding。`data_source` 不再作为独立加分项，而是进入 claim grounding 或作为 hard gate。

如果最终答案格式完全非法：

$r_i^{ans} = -1.0$

如果某个 claim 答对但没有证据支撑，则 $G_i(c)=0$，该 claim 不得分。  

如果 `data_source` 或 evidence provenance 与 claim 不一致，则对应 claim 的 $G_i(c)=0$。

### **4\.5 Phase 1 信用分配**

Phase 1 不使用额外的加权总奖励作为正式训练目标，避免重复计算工具成本、效率项和格式项。

对第 `t` 个工具步骤，使用工具 return\-to\-go：

$G_{i,t}^{\mathrm{tool}}
= \sum_{l=t}^{T_i} \gamma_{\mathrm{tool}}^{\,l-t} \, r_{i,l}^{\mathrm{tool}}
+ \kappa_{\mathrm{ans}} \cdot \gamma_{\mathrm{tool}}^{\,T_i-t} \, r_i^{\mathrm{ans}}$

其中：

- $r_{i,l}^{tool}$ 已包含工具调用成本、非法调用惩罚和重复调用惩罚。

- $r_i^{ans}$ 通过 $κ_{ans}$ 向前传播给工具决策，用于奖励能够支持最终正确答案的工具路径。

- 不再额外加入独立 `efficiency_reward` 或 `format_reward`。

ANSWER token 使用：

$G_i^{answer} = r_i^{ans}$

因此 Phase 1 的正式 token\-level credit 为：

- PLAN/tool\-call token：使用对应步骤的 $G_{i,t}^{tool}$。

- ANSWER token：使用 $G_i^{answer}$。

- system、user、tool observation token：mask，不参与 policy loss。

## **5\. Phase 2：On\-policy Memory RL**

Phase 2 开始训练模型自己生成 memory，并让后续子问题真实使用该 memory。

### **5\.1 Rollout 单位**

使用 dialog\-level rollout。

流程：

```Plain Text
q1 -> tools -> answer1 -> memory1
q2 uses memory1 -> tools -> answer2 -> memory2
q3 uses memory2 -> ...
```

建议初始参数：

```Plain Text
K = 2
max_tool_steps_per_turn = 6
```

先从短 dialog 和中等难度样本开始，避免长程方差过大。

### **5\.2 Memory Reward 总体设计**

Memory reward 只回答三个问题：

1. 是否忠实。

2. 是否覆盖未来依赖。

3. 是否足够压缩。

定义：

$r_i^{mem}$

作用于 `<MEMORY_AFTER>...</MEMORY_AFTER>` token。

### **5\.3 Memory Faithfulness**

`memory_after` 中的事实必须来自：

- `memory_before` 中已经验证的事实

- 当前工具 observation

- 当前代码执行结果

- 当前 answer 中已经满足 $C_i^{correct}(c)G_i(c)=1$ 的 grounded claim

未被证据支撑的 answer claim 不能作为 memory provenance。

将 `memory_after` 拆成若干 memory item：

$M_i^{write} = {m_{i,1}, m_{i,2}, ..., m_{i,J_i}}$

每个 memory item 的忠实性定义为：

$q_{mem}(m)
= C_{value}(m)
× C_{binding}(m)
× C_{provenance}(m)$

整体 memory faithfulness 为：

$F_i = \frac{\sum_{j=1}^{J_i} w_{i,j} \cdot q_{\mathrm{mem}}(m_{i,j})}{\max\left(\sum_{j=1}^{J_i} w_{i,j},\, \varepsilon\right)}$

### **5\.4 Future Dependency Coverage**

对每个 memory 边界 `after q_i`，离线构造未来依赖集合：

```JSON
{
  "sample_id": "...",
  "boundary": "after_sq1",
  "future_dependencies": [
    {
      "dependency_id": "dep_sq1_001",
      "type": "numeric_fact",
      "needed_by": "sq2",
      "source_evidence_id": "sq1_e1",
      "fields": {
        "entity": "乘用车",
        "time": "2010年1月",
        "metric": "产量同比增长率",
        "value": "16.96%",
        "unit": "%"
      },
      "weight": 1.0
    }
  ]
}
```

future dependency 采用类型化结构：

$h = (id_h, type_h, fields_h, needed_{by_h}, w_h)$

不同依赖类型具有不同必需字段：

$RequiredFields(numeric\_fact) = \{entity, time, metric, value, unit\}$

$RequiredFields(entity\_set) = \{entities\}$

$RequiredFields(reference) = \{reference, target\}$

$RequiredFields(constraint) = \{constraint_content\}$

$RequiredFields(table\_ref) = \{table_name 或 sheet_name\}$

为了控制标注和验证成本，Future Dependency 只考虑后续拓扑相关问题。第一版可限制拓扑距离：

$D_{FDC} ≤ 2$

这里的 $D_{FDC}$ 表示未来依赖图中的最大拓扑距离，不表示物理上连续的后续 `H` 轮，也不等同于 Phase 3 的反事实评估范围。



注意：不能要求 memory 保存模型在第 `i` 轮尚未见过的事实。  

因此，真正参与奖励的集合不是完整未来依赖 $H_i^{future}$，而是：

$H_i^{keep}
=
{ h ∈ H_i^{future} : Support(h, L_{i,T_i}) = 1 }$

$Support(h, L_{i,T_i}) $表示依赖 h 所需的信息已经被当前 evidence ledger 验证获得。对于 numeric\_fact，可通过 source\_evidence\_id 与 ledger 对齐；对于 entity\_set、reference、constraint 和 table\_ref，则通过其 RequiredFields 是否由 ledger 或已验证 memory 支撑来判断。

其中：

- $H_i^{future}$：在 $D_{FDC}$ 范围内，后续拓扑相关问题需要依赖的历史信息。

- $L_{i,T_i}$：当前子问题结束时已经验证获得的 evidence ledger。

- $H_i^{keep}$：当前已经获得、且未来确实需要保留的信息。

对于 numeric\_fact，memory 只保留实体、年份或指标名，但没有保留对应数值和绑定关系时，不计为覆盖该 future dependency。对于 entity\_set、reference、constraint 和 table\_ref，则按照其类型对应的 RequiredFields 判断是否保留完整。



Future Dependency Coverage 定义为：

$\mathbb{I}_{\mathrm{retain}}(h, M_i)
= \prod_{f \in \mathrm{RequiredFields}(\mathrm{type}_h)} \mathbb{I}\big[\text{$f$ 在 $M_i$ 中被正确保留}\big]

$

$S_i
= \frac{\sum_{h \in H_i^{\mathrm{keep}}} w_h \cdot \mathbb{I}_{\mathrm{retain}}(h, M_i)}{\max\left(\sum_{h \in H_i^{\mathrm{keep}}} w_h,\, \varepsilon\right)}$

如果 $H_i^{keep}$ 为空，则当前 memory 不计算未来依赖项，只使用 faithfulness 和 compression。  

论文指标仍可称为 Future Dependency Coverage，但训练公式中使用更严格的 $S_i$，表示“已获得且应保留的未来相关信息覆盖率”。

### **5\.5 Compression Penalty**

memory 不应复制完整历史或大段 observation。

定义压缩惩罚：

$P_i^{comp} = max(0, len(M_i) - B) / B$

建议：

B = 512 tokens

`B` 是默认 memory budget，可作为实验超参数调整；一组正式实验中必须固定同一个 tokenizer 和同一个 `B`，但真的做的时候要看512够不够我们的平均长度。

memory 长度使用当前训练模型的 tokenizer 计算。不同语言、不同 tokenizer 下都以 token 数为准，不再同时使用中文字符预算。



### **5\.6 Memory Reward 公式**

严重失败时直接给强负奖励：

$r_i^{mem} = -1.0$

严重失败仅包括：

- memory JSON 完全无法解析。

- $H_i^{keep}$ 非空，但 memory 为空或完全无关。

- 大部分 memory item 与已验证证据冲突。

其他情况使用 item\-level faithfulness：

$r_i^{mem}
= alpha_f * F_i
+ alpha_s * S_i
- lambda\_comp * P_i^{comp}$

其中：

- $F_i$：memory faithfulness score。

- $S_i$： $H_i^{keep}$ 的覆盖率。

- $P_i^{comp}$：压缩惩罚。

建议初始权重：

```Plain Text
alpha_f = 0.5
alpha_s = 0.4
lambda_comp = 0.1
```

如果 $H_i^{keep}$ 为空，则使用：

$r_i^{mem}
= alpha_f * F_i
- lambda\_comp * P_i^{comp}$

其中 $F_i$ 是 memory item 的比例分数，而不是整体布尔值。



### **5\.7 Phase 2 信用分配**

Phase 2 不使用整体 turn reward 或 dialog reward 广播给所有 token。  

正式训练使用 segment\-wise credit：

```Plain Text
PLAN/tool-call token -> G_{i,t}^{tool}
ANSWER token         -> r_i^{ans}
MEMORY_AFTER token   -> r_i^{mem}
```

其中 $G_{i,t}^{tool}$ 沿用 Phase 1 的工具 return\-to\-go。

必须实现 token mask：

- system、user、tool observation token 不参与 policy loss。

- PLAN 和 tool\-call token 使用工具步骤对应的 advantage。

- ANSWER token 使用 answer advantage。

- MEMORY\_AFTER token 使用 memory advantage。

整条 dialog 的平均 reward 可以作为日志指标、早期 smoke test 或 checkpoint selection，但不能作为正式训练时广播到所有 token 的唯一奖励。否则双时间尺度信用分配无法成立。







## **6\. Phase 3：Sparse Counterfactual Memory RL**

Phase 3 不进入第一阶段主训练循环，等 Phase 1 和 Phase 2 跑通后再补最小实现。  

但如果论文贡献中保留“memory 对未来推理的边际效用”这一点，最终实验版本应包含本节的稀疏反事实模块。

### **6\.1 目标**

估计模型生成的 memory 是否真的提升了未来子问题表现。

Phase 3 只从存在后续依赖的 memory 边界中采样：

$i \sim { i : ∃ j>i, ρ_{ij}=1 }$

其中 $ρ_{ij}=1$ 表示第 `j` 个子问题依赖 `after q_i` 时写入的历史信息。

构造两个 continuation：

```Plain Text
A: 使用模型生成的 M_i^{gen}
B: 使用 previous memory M_{i-1}
```

定义第一个依赖当前 memory 边界的目标后续问题：

$j^{\star}
= min \{ j>i : ρ_{ij}=1 \}$

两个 continuation 都按照原始 dialog 顺序执行，直到完成 $q_{j^{\star}}$。中间子问题不能跳过，因为它们也会更新 memory。

### **6\.2 Sparse Counterfactual Memory Utility**

定义：

$ΔU_i =r_{j^{\star}}^{ans}(M_i^{gen})
- r_{j^{\star}}^{ans}(M_{i-1})$

其中：

$j^{\star} = min \{ j>i : ρ_{ij}=1 \}$

表示第一个依赖当前 memory 边界的后续子问题。反事实评估只比较 $q_{j^{\star}}$ 这个目标后续问题，而不是物理上紧邻的下一轮。两个 continuation 共享相同 policy、相同原始 dialog 顺序、相同工具环境、相同 checkpoint 和相同预算，并都执行到 $q_{j^{\star}}$ 完成。

如果 $ΔU_i > 0$，说明当前 memory 对未来子问题有帮助。  

如果 $ΔU_i < 0$，说明当前 memory 误导或损害了未来推理。

### **6\.3 使用方式**

只对 `<MEMORY_AFTER>` token 施加 counterfactual reward。先裁剪反事实效用：

$\tilde{ΔU_i}
=
clip(ΔU_i, -a, a)$

最终 memory reward 为：

$r_i^{mem-final}
=
r_i^{mem}
+
λ_{cf}
(
1[F_i ≥ τ_f] [\tilde{ΔU_i}]_+
+
[\tilde{ΔU_i}]_-
)$

其中：

$[x]_+ = max(x, 0)$

$[x]_- = min(x, 0)$

建议初始参数：

$λ_cf = 0.2$

$a = 1.0$

$τ_f = 0.8$

含义是：只有足够忠实的 memory 才能获得正向未来效用奖励；有害 memory 的负向效用始终保留。

两个 continuation 使用相同 policy checkpoint、相同原始问题顺序、相同工具环境和预算、相同或配对随机种子，并采用低温或确定性解码。

只对 20%\-30% dialog 做 sparse counterfactual，控制成本。

## **7\. 优化目标**

使用 SFT checkpoint 初始化 policy：

$π_θ ← π_{SFT}$

冻结一份 reference model：

$π_{ref} = π_{SFT}$

采用 GRPO\-style relative policy optimization，但 advantage 必须按 segment 类型计算，不能把整条 dialog 的同一个 reward 广播给所有 assistant token。

由于不同 rollout 的工具步数不同，且中间状态会在 observation 后分叉，第一版不假设每个工具步骤都构成严格 same\-state group。正式训练使用 segment\-type minibatch normalization：

$A_{z,n}
= (G_{z,n} - μ_z^{batch})
/ (σ_z^{batch} + ε),
z ∈ \{tool, answer, memory\}$

其中：

```Plain Text
tool segment   -> G_{i,t}^{tool}
answer segment -> r_i^{ans}
memory segment -> r_i^{mem} 或 r_i^{mem-final}
```

分别维护 tool return pool、answer return pool 和 memory return pool。同一 prompt 的 `K` 条 rollout 仍用于提升采样多样性，但论文表述使用：

> segment\-aware relative policy optimization with process\-shaped rewards
> 
> 

如果工程上需要先做 smoke test，可以临时记录整条轨迹 reward，但正式训练和论文结果必须使用 segment\-aware advantage。

对每个参与训练的 assistant token，定义 token\-level probability ratio：

$ρ_{z,n,r}(θ)
=
π_θ(a_{z,n,r} | s_{z,n,r})
/
π_{old}(a_{z,n,r} | s_{z,n,r})$

其中：

- $z∈{tool,answer,memory}$ 表示 segment 类型；

- $n$ 表示当前 rollout batch 中的一个 segment；

- $r$ 表示该 segment 内参与训练的 assistant token；

- $π_{old}$ 是采样该 rollout 时的冻结旧策略；

- system、user 和 tool observation token 通过 mask 排除，不参与 $L_{RL}$。

正式 clipped policy loss 定义为：

$L_{RL}
=
-\mathbb E_{z,n,r}
\left[
\min
\left(
\rho_{z,n,r}(\theta)A_{z,n},
\operatorname{clip}
\left(
\rho_{z,n,r}(\theta),
1-\epsilon_{clip},
1+\epsilon_{clip}
\right)
A_{z,n}
\right)
\right].$

总体目标：

$L =
  L_{RL}
+ β_{KL} * KL(π_θ || π_{ref})
+ λ_{SFT} * L_{replay}$

建议初始参数：

```Plain Text
β_KL = 0.01
λ_SFT = 0.05 或 0.1
```

$L_{replay}$ 用于保持工具调用格式、ANSWER JSON、MEMORY JSON 和基础表格推理能力。

论文表述建议：

> We use a GRPO\-style clipped objective with process\-shaped rewards\.
> 
> 

不要声称所有中间工具步骤都构成严格的 same\-state group。更稳的说法是：

> segment\-aware relative policy optimization
> 
> 

或者：

> relative policy optimization with process\-shaped rewards
> 
> 

## **8\. 工程实现模块**

### **8\.1 Rollout Environment**

功能：

- 拼接 system prompt、tools schema、memory\_before 和当前问题。

- 调用 policy model。

- 解析 tool\_call。

- 执行真实 TableAgentBench 工具。

- 拼接 observation。

- 判断是否生成 answer。

- 限制 `max_tool_steps`。

- 输出完整 trajectory。

输出格式：

```JSON
{
  "sample_id": "...",
  "subquestion_id": 1,
  "trajectory": [],
  "assistant_answer": {},
  "memory_after": {},
  "status": "completed"
}
```

### **8\.2 Target Evidence Builder**

输入：

- benchmark samples

- score\_points

- related\_tables

- SFT expert trajectory

- gold answer

输出：

"source\_anchors": \[\.\.\.\]

```Plain Text
target_evidence.jsonl
```

### **8\.3 Evidence Ledger**

核心函数：

```Python
update_ledger(
    ledger,
    tool_call,
    observation,
    target_evidence
)
```

返回：

```Python
{
    "coverage_before": 0.25,
    "coverage_after": 0.50,
    "new_evidence_ids": ["sq1_e2"],
    "audit": {}
}
```

### **8\.4 Reward Calculator**

输入：

- rollout trajectory

- target evidence

- score\_points

- related\_tables

- future dependencies

输出：

```JSON
{
  "r_tool": 0.72,
  "r_answer": 1.0,
  "r_memory": 0.81,
  "reward_summary_for_logging": 0.87,
  "audit": {
    "tool_valid_rate": 1.0,
    "evidence_coverage": 0.75,
    "memory_faithfulness": 0.92
  }
}
```

`reward_summary_for_logging` 仅用于日志、调试和 checkpoint selection，不参与正式 token\-level policy loss。`memory_faithfulness` 输出比例分数，而不是布尔值。



### **8\.5 Trainer**

第一版建议：

```Plain Text
same prompt -> sample K rollouts -> execute tools -> compute segment returns -> build tool/answer/memory return pools -> segment-type minibatch normalization -> update
```

先跑通 subquestion\-level Tool \+ Answer RL，再进入 dialog\-level Memory RL。

## **9\. 实验设计**

### **9\.1 主实验组**

\| 实验组 \| 设置 \|

\|\-\-\-\|\-\-\-\|

\| SFT \| 当前 SFT checkpoint \|

\| Outcome\-only RL \| 只使用最终 answer reward \|

\| Tool Process RL \| answer reward \+ evidence progress reward \|

\| Tool \+ Memory RL \| 加 memory faithfulness \+ FDC \|

\| Full TEMPO\-RL \| 加 sparse counterfactual memory gain \|

核心假设：

```Plain Text
Tool Process RL > Outcome-only RL
Tool + Memory RL > Tool Process RL
Full TEMPO-RL 在长 dialog 上进一步提升
```

### **9\.2 消融实验**

\| 消融项 \| 验证目的 \|

\|\-\-\-\|\-\-\-\|

\| 去掉 evidence progress reward \| 验证工具过程奖励是否有效 \|

\| 去掉 binding check \| 验证实体\-数值绑定的重要性 \|

\| 去掉 computation verifier \| 验证计算验证的重要性 \|

\| offline SFT memory vs on\-policy memory \| 验证自生成 memory 的影响 \|

\| 去掉 FDC reward \| 验证未来依赖感知 memory reward \|

\| 无 counterfactual vs sparse counterfactual \| 验证 `ΔU_i` 是否带来增益 \|

\| 去掉 SFT replay \| 验证格式保持和能力保持是否必要 \|

## **10\. 评估指标**

### **10\.1 主任务指标**

- final answer accuracy

- score point coverage

- table dependence precision / recall / F1

- tool valid rate

- answer grounding rate

- evidence coverage

- repeated call rate

- average tool steps

- average token cost

### **10\.2 Memory 指标**

- memory faithfulness

- future dependency coverage

- memory compression ratio

- memory parse success rate

- downstream answer accuracy conditioned on generated memory

### **10\.3 新增论文指标**

#### **Tool Marginal Utility**

```Plain Text
TMU =
  平均每次工具调用带来的新增 target evidence coverage
```

#### **Memory Faithfulness**

```Plain Text
MF =
  memory 中可追溯事实比例
```

#### **Future Dependency Coverage**

```Plain Text
FDC =
  memory 覆盖未来依赖项比例
```

#### **Sparse Counterfactual Memory Utility**

```Plain Text
ΔU =
  U(M_gen)
  - U(M_prev)
```

## **11\. 第一版不实现的内容**

为控制工程量，第一版不实现以下内容：

- 完整在线 evidence graph。

- 每个工具步骤的 sibling branching。

- 每个 memory 边界都做 counterfactual；最终只保留稀疏采样的 `M_i^{gen}` vs `M_{i-1}`。

- 独立训练 reward model。

- 多种 counterfactual baseline 同时训练。

- 自适应 memory budget。

- 复杂图算法，例如 SCC 或全路径 proof search。

- 复杂 token\-level dense return；但基础的 segment\-wise token mask 与 segment\-wise advantage 是必做项。

- 独立的 efficiency reward、terminal evidence reward、continuity reward。

- stall、unsupported、recovery 等过细的独立错误恢复奖励。

这些可以作为 appendix discussion 或 future work。

## **12\. 推荐实现顺序**

建议严格按以下顺序推进：

```Plain Text
1. 固定 SFT checkpoint
2. 构造 target_evidence.jsonl
3. 实现 evidence ledger
4. 实现 subquestion-level rollout
5. 实现 Tool + Answer reward
6. 跑 SFT vs Outcome-only RL vs Tool Process RL
7. 实现 dialog-level rollout
8. 实现 memory faithfulness + FDC reward
9. 跑 Tool + Memory RL
10. 补最简 sparse counterfactual memory gain：
    从存在后续依赖的边界中采样，比较 M_i^{gen} vs M_{i-1} 在第一个依赖目标问题 q_{j^{\star}} 上的 answer reward 差异
```

不要在 Phase 1 和 Phase 2 跑通前实现 Phase 3。  

但最终 TPAMI 实验若要主张 memory utility，建议保留第 10 步的最简反事实结果。

## **13\. 论文贡献表述**



建议论文中将贡献写成三点：



### **Contribution 1: Dual\-timescale Table\-Agent Formulation**



We formulate multi\-turn table\-agent reasoning as a dual\-timescale decision process that separates fine\-grained tool interaction within each sub\-question from long\-horizon memory writing across sub\-questions\.



### **Contribution 2: Evidence\-Guided Process Reward**



We propose a lightweight target\-evidence ledger that rewards tool actions only when they introduce new, verifiable, and semantically grounded evidence required by the current sub\-question\.



### **Contribution 3: Future\-Aware Memory Optimization**



We introduce a future\-dependency\-aware memory reward and a sparse same\-prefix counterfactual utility estimate to evaluate the marginal utility of model\-generated memory for subsequent table reasoning\.



**\-\-\-**



## **14\. 最终定位**



TEMPO\-RL 不应被描述为一个庞大的完整 RL 系统，而应被描述为：



> A lightweight process optimization framework for multi\-turn table agents\.
> 
> 



核心是：



```Plain Text
证据账本
+ grounded answer reward
+ future-aware memory reward
+ sparse counterfactual memory utility
```



这个版本同时满足：



- 有明确方法创新。

- 能和现有 SFT pipeline 自然衔接。

- 工程量可控。

- 可以支撑 TPAMI 转刊新增实验。

- 后续还可以扩展为更复杂的 reward model 或 process reward model。

因此，本文档可作为后续 RL 实现与论文方法章节撰写的最终方案基准。



