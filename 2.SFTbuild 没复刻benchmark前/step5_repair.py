"""
Step 5: Repair 失败子问题（两阶段真实执行版）。

Stage A: Repair LLM 只生成 tool calls（不含 observations、不含 final answer）
Stage B: 在真实环境中执行 tool calls，记录真实 observations
Stage C: Repair LLM 基于真实 observations 生成 final answer
Stage D: 重新 evaluation，验证修复质量

输入:
  --subquestions  : step3 输出的 evaluated_subquestions.jsonl（含 eval 反馈）
  --audit         : step4 输出的 audit_report.jsonl（标记哪些失败）
  --samples       : benchmark 样本 JSON（用于 re-evaluation）
  --config_key    : LLM 配置 key（默认 mimo）

输出:
  output/repaired_subquestions.jsonl  — 修复后的子问题记录

关键变更（v2）:
  - observation 来自真实工具执行，而非 LLM 编造
  - 修复后重新 evaluation，不通过的不标记 _repaired
  - 修复失败保留原始记录，避免 dialog 缺口
"""
import os
import sys
import copy
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional

from SFTbuild.utils import (
    read_jsonl, write_jsonl, extract_json_from_response,
    get_tool_schema_constraints, validate_tool_calls, load_samples,
    validate_assistant_answer,
)

# Default dataset table root — used as <TABLE_ROOT> resolution target
_DEFAULT_DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     'dataset', 'tables')

# Max chars per table in repair prompt (keep under token budget)
_MAX_TABLE_CHARS = 3000

# Tool execution timeout (seconds)
_TOOL_EXEC_TIMEOUT = 60

# Max repair steps to prevent infinite loops
_MAX_REPAIR_STEPS = 10


# ============================================================
# Iterative repair prompts
# ============================================================

REPAIR_ITERATIVE_PROMPT = """你是一个专业的表格数据分析智能体。你需要通过逐步调用工具来回答用户的问题。

## 原始问题
{user_question}

## 原始智能体轨迹（可能包含错误，供参考）
{original_trajectory}

## 失败原因
{failure_reasons}

## 真实表格数据（你必须基于这些真实数据规划工具调用）
以下是从实际表格文件中读取的真实数据。请基于这些数据规划正确的工具调用。
如果工具调用的参数需要引用文件路径，请确保路径与下面提供的实际路径一致。

{table_context}

{tool_constraints}

## 已执行的步骤及真实返回结果
{execution_history}

## 任务
基于上述**已执行步骤的真实返回结果**，决定下一步操作：

- 如果还需要更多数据才能回答用户问题：生成下一步的**工具调用**（tool_call）
- 如果已有足够的数据回答用户问题：生成**最终答案**（final_answer）
- **重要**：如果历史中显示上一轮答案被拒绝（⚠️ 答案被拒绝），你必须根据列出的**具体问题**修正答案。不要重复生成相同的答案。

注意：
- **基于真实返回结果做决策**：仔细阅读已执行步骤中的 Observation，决定下一步需要什么数据
- 步骤规划（step_plan）使用中文，保持简洁（1-2句话，描述意图和原因）
- **工具调用必须严格遵守上面「可用工具」列表中的工具名称和参数定义**
- 所有文件路径使用 <TABLE_ROOT>/ 前缀。格式: <TABLE_ROOT>/子目录/文件名（例如 <TABLE_ROOT>/chinese_table/2010年产量.xlsx）
- 上面表格数据中给出的实际路径已标注了子目录结构，请根据它构造 <TABLE_ROOT>/... 路径
- 严禁使用真实绝对路径（如 /data/、/tmp/ 等前缀）
- python_code_executor 的代码中如有文件路径，也使用 <TABLE_ROOT>/ 前缀
- 答案必须基于已执行步骤中 Observation 的真实数据，严禁编造任何数字、日期、实体名称
- 最终答案用中文表述

输出格式（JSON）：

如果是工具调用：
{{
  "action": "tool_call",
  "step_plan": "本次操作的意图说明",
  "tool_calls": [
    {{
      "tool_call_id": "call_{call_seq}",
      "tool_name": "工具名称",
      "arguments": {{"参数名": "参数值"}}
    }}
  ]
}}

如果是最终答案：
{{
  "action": "final_answer",
  "assistant_answer": {{
    "answer": "用中文表述的答案",
    "data_source": ["用到的文件名"]
  }}
}}

严禁在输出中包含修复指令或标注。"""


REPAIR_INITIAL_PROMPT = """你是一个专业的表格数据分析智能体。你需要通过逐步调用工具来回答用户的问题。

## 原始问题
{user_question}

## 原始智能体轨迹（可能包含错误，供参考）
{original_trajectory}

## 失败原因
{failure_reasons}

## 真实表格数据（你必须基于这些真实数据规划工具调用）
以下是从实际表格文件中读取的真实数据。请基于这些数据规划正确的工具调用。
如果工具调用的参数需要引用文件路径，请确保路径与下面提供的实际路径一致。

{table_context}

{tool_constraints}

## 任务
这是修复的开始，还没有执行任何步骤。请生成**第一步工具调用**来开始回答用户问题。

注意：
- **一次只生成一步操作**：生成你觉得最合适的下一步
- 步骤规划（step_plan）使用中文，保持简洁（1-2句话，描述意图和原因）
- **工具调用必须严格遵守上面「可用工具」列表中的工具名称和参数定义**
- 所有文件路径使用 <TABLE_ROOT>/ 前缀。格式: <TABLE_ROOT>/子目录/文件名（例如 <TABLE_ROOT>/chinese_table/2010年产量.xlsx）
- 上面表格数据中给出的实际路径已标注了子目录结构，请根据它构造 <TABLE_ROOT>/... 路径
- 严禁使用真实绝对路径（如 /data/、/tmp/ 等前缀）
- python_code_executor 的代码中如有文件路径，也使用 <TABLE_ROOT>/ 前缀

输出格式（JSON）：
{{
  "action": "tool_call",
  "step_plan": "本次操作的意图说明",
  "tool_calls": [
    {{
      "tool_call_id": "call_1",
      "tool_name": "工具名称",
      "arguments": {{"参数名": "参数值"}}
    }}
  ]
}}

严禁在输出中包含修复指令或标注。"""


REPAIR_ANSWER_VERIFIER_PROMPT = """你是一个严格的答案验证专家。验证模型回答是否与工具返回的真实数据一致。

## 用户问题
{user_question}

## 所有工具 Observation（真实执行返回的数据）
{all_observations}

## 模型给出的答案
{answer}

## 任务
严格逐条验证答案中的事实性声明。重点检查：

1. **数字准确性**：答案中的每个数字（数量、百分比、排名、年份等）必须能在 Observation 中找到对应来源，且数值一致。
2. **实体名称**：答案中的实体名称（类别名、产品名、文件名等）必须与 Observation 中的名称一致。
3. **逻辑一致性**：答案的结论（如"XX最高"、"XX下降"）必须与 Observation 中的数据趋势吻合，不能反向或编造趋势。
4. **完整性**：答案不能选择性忽略 Observation 中与问题相关的核心数据。

输出 JSON：
{{
  "pass": true,
  "issues": [],
  "confidence": "high"
}}

或：

{{
  "pass": false,
  "issues": ["具体的数据不一致问题"],
  "confidence": "low"
}}

严格标准：只要有一条事实无法在 Observation 中找到来源，就应判定为不通过。"""


def verify_repair_answer(user_question: str, all_observations: list,
                         answer: dict, client) -> dict:
    """
    LLM-based verification that the repaired answer is semantically consistent
    with the real tool observations.

    Returns dict with 'pass' (bool), 'issues' (list), 'confidence' (str).
    """
    answer, _ = validate_assistant_answer(answer)

    # Flatten observations into a readable text block
    obs_lines = []
    for i, step_obs in enumerate(all_observations):
        for obs in step_obs:
            if isinstance(obs, dict):
                success = obs.get('success', False)
                tool = obs.get('tool_name', 'unknown')
                content = obs.get('content', '')
                status = '[OK]' if success else '[FAIL]'
                obs_lines.append(f'### Observation [{status}] {tool}')
                obs_lines.append(content[:1500])  # Truncate each obs
            else:
                obs_lines.append(str(obs)[:1500])
        obs_lines.append('')

    obs_text = '\n'.join(obs_lines)
    answer_text = json.dumps(answer, ensure_ascii=False, indent=2)

    prompt = REPAIR_ANSWER_VERIFIER_PROMPT.format(
        user_question=user_question,
        all_observations=obs_text[:6000],  # Keep within token budget
        answer=answer_text
    )

    try:
        response = client.chat(
            prompt=prompt,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        result = extract_json_from_response(response)
        return {
            'pass': result.get('pass', False) is True,
            'issues': result.get('issues', []),
            'confidence': result.get('confidence', 'unknown')
        }
    except Exception as e:
        return {
            'pass': False,
            'issues': [f'Verifier LLM call failed: {e}'],
            'confidence': 'unknown'
        }


REPAIR_ANSWER_JUDGE_PROMPT = """你是一个独立的答案评分专家。你的任务是评判模型回答是否完整覆盖了评分点（score_points）。

## 用户问题
{user_question}

## 评分点（gold score_points，一个或多个独立的事实陈述）
{score_points}

## 金标关联表格（gold related_tables）
{gold_tables}

## 模型答案
answer: {answer_text}
data_source: {data_source}

## 任务
逐条检查每个评分点是否被模型的 answer 覆盖。判断标准：
- **语义覆盖**：不需要逐字匹配，但关键事实（数字、实体、趋势）必须一致
- **数据来源**：data_source 应包含评分点所需的表格文件（按 basename 比较）

输出 JSON：
{{
  "pass": true,
  "covered_points": ["评分点1", "评分点2"],
  "missing_points": [],
  "table_coverage_pass": true,
  "table_issues": [],
  "summary": "简短评价"
}}

如果任一评分点缺失或数据表覆盖不完整，pass 应为 false。"""


def judge_repair_answer(user_question: str, score_points: list,
                        gold_tables: list, answer: dict, client) -> dict:
    """
    Independent evaluation of a repaired answer against gold score_points.

    Unlike verify_repair_answer (which checks consistency with observations),
    this judge checks whether the answer covers the benchmark's expected key
    facts (score_points). This catches answers that are internally consistent
    but miss critical information the benchmark requires.

    Returns dict with 'pass' (bool), 'covered_points', 'missing_points', etc.
    """
    if not score_points:
        return {
            'pass': True,
            'covered_points': [],
            'missing_points': [],
            'table_coverage_pass': True,
            'table_issues': [],
            'summary': 'No score_points to check'
        }

    answer, _ = validate_assistant_answer(answer)
    answer_text = answer.get('answer', '')
    data_source = json.dumps(answer.get('data_source', []), ensure_ascii=False)
    sp_text = '\n'.join(f'- {sp}' for sp in score_points)
    gt_text = '\n'.join(f'- {t}' for t in gold_tables) if gold_tables else '（无）'

    prompt = REPAIR_ANSWER_JUDGE_PROMPT.format(
        user_question=user_question,
        score_points=sp_text,
        gold_tables=gt_text,
        answer_text=answer_text,
        data_source=data_source
    )

    try:
        response = client.chat(
            prompt=prompt,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        result = extract_json_from_response(response)

        # ---- Deterministic cross-validation ----
        # Do NOT trust the LLM's "pass" field blindly. An LLM may return
        # pass=true while simultaneously listing missing_points or
        # table_coverage_pass=false. Cross-validate all fields.
        llm_pass = result.get('pass', False) is True
        covered = result.get('covered_points', [])
        missing = result.get('missing_points', [])
        table_ok = result.get('table_coverage_pass', False) is True  # default False!

        # Type safety: covered/missing must be lists of strings.
        # An LLM may return pass=true with covered_points="A" which would
        # satisfy len(covered)==len(score_points) when score_points has 1 item.
        # An LLM may also return covered_points=["A", "A"] to bypass the
        # length check with duplicates. Deduplicate covered set.
        covered_set = set(covered) if isinstance(covered, list) else set()
        types_ok = (
            isinstance(covered, list)
            and isinstance(missing, list)
            and all(isinstance(x, str) for x in covered)
            and all(isinstance(x, str) for x in missing)
            and len(covered) == len(covered_set)  # No duplicate points
        )

        det_pass = (
            types_ok
            and llm_pass
            and table_ok
            and len(missing) == 0
            and len(covered_set) == len(score_points)
        )

        return {
            'pass': det_pass,
            'covered_points': covered,
            'missing_points': missing,
            'table_coverage_pass': table_ok,
            'table_issues': result.get('table_issues', []),
            'summary': result.get('summary', ''),
            # Debug: show what the LLM claimed vs what we determined
            '_llm_pass': llm_pass,
            '_det_pass': det_pass,
        }
    except Exception as e:
        return {
            'pass': False,
            'covered_points': [],
            'missing_points': [f'Judge LLM call failed: {e}'],
            'table_coverage_pass': False,
            'table_issues': [f'Judge LLM call failed: {e}'],
            'summary': 'Judge failed'
        }


# ============================================================
# Table data helpers (unchanged from original)
# ============================================================

def _find_table_path(filename: str, dataset_root: str = _DEFAULT_DATASET_ROOT,
                      sample_file_path: str = '') -> Optional[str]:
    """Find the actual file path for a table filename by walking dataset_root.

    When multiple files share the same basename (e.g. 10.csv exists in both
    200-csv/ and 201-csv/), the sample's file_path is used to disambiguate.

    Args:
        filename: Table filename (basename or relative path).
        dataset_root: Root directory to search.
        sample_file_path: Optional sample file_path (e.g. "english_table/200-csv/10.csv")
                          used to break ties among candidates with the same basename.
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

    # Multiple candidates — try to disambiguate.

    # 1. Path hint from the filename itself (e.g. "200-csv/10.csv")
    if path_hint:
        for cp in candidates:
            if path_hint in cp:
                return cp

    # 2. Sample file_path hint — prefer the candidate that shares the
    #    same parent directory as the sample file.
    if sample_file_path:
        sample_dir = os.path.dirname(sample_file_path)
        if sample_dir:
            for cp in candidates:
                # Normalize for comparison: strip dataset_root prefix from candidate
                try:
                    cp_rel = os.path.relpath(cp, dataset_root)
                except ValueError:
                    continue
                cp_dir = os.path.dirname(cp_rel)
                if cp_dir == sample_dir or cp_dir.endswith('/' + sample_dir) or sample_dir.endswith('/' + cp_dir):
                    return cp
            # Weaker match: candidate path contains sample directory as substring
            for cp in candidates:
                if sample_dir in cp:
                    return cp

    # Cannot disambiguate — return first (conservative, may be wrong).
    # The caller can detect mismatches later through answer verification.
    return candidates[0]


def _read_table_content(file_path: str, max_chars: int = _MAX_TABLE_CHARS) -> str:
    """Read a table file and return a formatted text representation."""
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
            text = '\n'.join(all_rows)
            lines = text.split('\n')
        else:
            return ''

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


def _build_table_context(true_tables: list, dataset_root: str = _DEFAULT_DATASET_ROOT,
                         sample_file_path: str = '') -> str:
    """Build a text block containing the real content of all required tables."""
    if not true_tables:
        return '（无关联表格信息）'

    parts = []
    for tbl in true_tables:
        path = _find_table_path(tbl, dataset_root, sample_file_path=sample_file_path)
        if path:
            content = _read_table_content(path)
            # Show real path AND the <TABLE_ROOT> equivalent for LLM reference
            # The <TABLE_ROOT> path is: <TABLE_ROOT>/<relative path from dataset_root>
            try:
                rel_path = os.path.relpath(path, dataset_root)
            except ValueError:
                rel_path = os.path.basename(path)
            table_root_path = f'<TABLE_ROOT>/{rel_path}'
            parts.append(
                f'### 表格文件: {tbl}\n'
                f'实际路径: {path}\n'
                f'<TABLE_ROOT> 路径: {table_root_path}\n\n{content}'
            )
        else:
            parts.append(f'### 表格文件: {tbl}\n[WARNING] 未在 dataset 中找到此文件，请根据文件名推断数据结构')
    return '\n\n'.join(parts)


# ============================================================
# Stage B: Real tool execution
# ============================================================

def _resolve_tool_paths(arguments: dict, dataset_root: str) -> dict:
    """
    Resolve <TABLE_ROOT> placeholders in tool arguments to real paths.

    <TABLE_ROOT> represents the dataset table root (dataset_root).
    The LLM generates paths like:
      <TABLE_ROOT>/chinese_table/foo.xlsx
    Which resolves to:
      {dataset_root}/chinese_table/foo.xlsx

    Also handles the legacy format <TABLE_ROOT>/dataset/tables/... if the LLM
    was told to use that format by an older prompt.
    """
    # Normalize: ensure dataset_root has no trailing slash for consistent joining
    table_root = dataset_root.rstrip('/')

    # Try importing from src.tools.base first; fall back to local implementation
    # if the function doesn't exist (e.g. older version of the codebase).
    try:
        from src.tools.base import resolve_placeholder_paths
        resolved = resolve_placeholder_paths(arguments, table_root)
    except ImportError:
        # Local fallback: recursively replace <TABLE_ROOT> in all values.
        # Handles strings, nested dicts, lists, and the exact "<TABLE_ROOT>" value.
        table_root_slash = table_root + '/'

        def _resolve_value(val):
            """Recursively resolve <TABLE_ROOT> in any nested structure."""
            if isinstance(val, str):
                if val == '<TABLE_ROOT>':
                    return table_root
                if '<TABLE_ROOT>' in val:
                    return val.replace('<TABLE_ROOT>', table_root)
                return val
            elif isinstance(val, dict):
                return {k: _resolve_value(v) for k, v in val.items()}
            elif isinstance(val, (list, tuple)):
                return [_resolve_value(item) for item in val]
            else:
                return val

        resolved = {k: _resolve_value(v) for k, v in arguments.items()}

    # If the LLM used the legacy <TABLE_ROOT>/dataset/tables/... format,
    # the resolved path will contain /dataset/tables/dataset/tables/...
    # Detect and fix this by checking if any resolved path points to a non-existent
    # file that would exist if we strip the duplicate.
    legacy_prefix = os.path.join(table_root, 'dataset', 'tables')
    for k, v in resolved.items():
        if isinstance(v, str) and legacy_prefix in v:
            # Remove the duplicate: {root}/dataset/tables/dataset/tables/...
            # → {root}/dataset/tables/...
            fixed = v.replace(legacy_prefix + '/', table_root + '/', 1)
            resolved[k] = fixed
        elif isinstance(v, list):
            resolved[k] = [
                item.replace(legacy_prefix + '/', table_root + '/', 1)
                if isinstance(item, str) and legacy_prefix in item else item
                for item in v
            ]

    return resolved


def _execute_single_tool(tool_name: str, arguments: dict, dataset_root: str) -> dict:
    """
    Execute a single tool call and return the observation dict.
    Uses a subprocess with timeout for safety.
    """
    try:
        # Resolve <TABLE_ROOT> placeholders (moved inside try so ImportError
        # in _resolve_tool_paths is caught and converted to error observation).
        resolved_args = _resolve_tool_paths(arguments, dataset_root)
        from src.tools.base import execute_tool

        # Use signal-based timeout
        result = None

        def _exec():
            nonlocal result
            result = execute_tool(tool_name, **resolved_args)

        # Execute in a thread with timeout (signal only works on main thread)
        import threading
        thread = threading.Thread(target=_exec, daemon=True)
        thread.start()
        thread.join(timeout=_TOOL_EXEC_TIMEOUT)

        if thread.is_alive():
            return {
                'tool_call_id': '',
                'tool_name': tool_name,
                'content': f'[ERROR] Tool execution timed out after {_TOOL_EXEC_TIMEOUT}s',
                'success': False
            }

        if result is None:
            return {
                'tool_call_id': '',
                'tool_name': tool_name,
                'content': '[ERROR] Tool execution returned None',
                'success': False
            }

        success = getattr(result, 'success', False)
        data = getattr(result, 'data', '') or ''
        message = getattr(result, 'message', '')

        return {
            'tool_call_id': '',
            'tool_name': tool_name,
            'content': data if data else message,
            'success': success
        }

    except ImportError as e:
        return {
            'tool_call_id': '',
            'tool_name': tool_name,
            'content': f'[ERROR] Cannot import tool infrastructure: {e}',
            'success': False
        }
    except Exception as e:
        return {
            'tool_call_id': '',
            'tool_name': tool_name,
            'content': f'[ERROR] Tool execution failed: {e}',
            'success': False
        }


def execute_repair_tool_calls(tool_call_steps: list, dataset_root: str,
                               call_seq: int = 0, verbose: bool = False) -> list:
    """
    Execute all tool calls from repair LLM output and return complete agent_steps
    with real observations.

    Args:
        tool_call_steps: List of step dicts with type='tool_call', each containing
                         tool_calls list (no observations yet).
        dataset_root: Path to dataset tables for <TABLE_ROOT> resolution.
        call_seq: Iteration number for generating unique call_ids.

    Returns:
        List of agent_steps with real observations filled in.
    """
    executed_steps = []
    tool_counter = 0  # global index across all tool calls in this call batch

    for step in tool_call_steps:
        if step.get('type') != 'tool_call':
            continue

        tool_calls = step.get('tool_calls', [])
        if not tool_calls:
            continue

        observations = []
        for idx, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                continue
            tool_name = tc.get('tool_name', '')
            arguments = tc.get('arguments', {})

            if verbose:
                args_brief = json.dumps(arguments, ensure_ascii=False)[:120]
                print(f"      Executing: {tool_name}({args_brief})...")

            obs = _execute_single_tool(tool_name, arguments, dataset_root)

            # Assign deterministic unique call_id. The LLM-generated id is
            # ignored entirely — it can be empty or collide across parallel
            # calls.  Format: repair_call_{iteration}_{tool_index:03d}
            unique_id = f'repair_call_{call_seq}_{tool_counter:03d}'
            tool_counter += 1
            tc['tool_call_id'] = unique_id
            obs['tool_call_id'] = unique_id
            observations.append(obs)

            if verbose:
                status = 'OK' if obs['success'] else 'FAIL'
                preview = obs['content'][:100].replace('\n', ' ')
                print(f"        [{status}] {preview}...")

        executed_steps.append({
            'agent_step_id': step.get('agent_step_id'),
            'type': 'tool_call',
            'step_plan': step.get('step_plan', ''),
            'tool_calls': tool_calls,
            'observations': observations
        })

    return executed_steps


# ============================================================
# Stage D: Re-evaluation
# ============================================================

def re_evaluate_subquestion(rec: dict, sample: dict = None, client=None) -> dict:
    """
    Re-run evaluation checks on a repaired subquestion.

    Checks performed:
      1. Format: answer JSON valid, non-empty answer, non-empty data_source
      2. Tool errors: no unrecovered errors in agent_steps
      3. Completeness: has final_answer step, exactly one and at the end
      4. Tool schema: all tool names/args valid against TOOL_REGISTRY
      5. Data source coverage: model data_source covers gold related_tables
      6. Evidence verification: answer numeric claims traceable to observations

    Note: Full accuracy/table_depend judging requires LLM-based evaluation
    (step3). This function provides the deterministic subset that can be
    checked without additional LLM calls. For production, re-run step3
    evaluation on repaired trajectories after repair.

    Returns a dict with 'pass' and 'issues' keys.
    """
    issues = []

    agent_steps = rec.get('agent_steps', [])
    answer = rec.get('assistant_answer', {})

    # ---- 1. Format audit ----
    if not isinstance(answer, dict) or 'answer' not in answer:
        issues.append('format: answer JSON invalid')
    elif not isinstance(answer.get('answer'), str) or not answer['answer'].strip():
        issues.append('format: answer is empty or not a string')
    if not isinstance(answer.get('data_source'), list) or not answer['data_source']:
        issues.append('format: data_source is empty')
    elif not all(isinstance(d, str) for d in answer['data_source']):
        issues.append('format: data_source contains non-string entries')

    # ---- 2. Tool errors ----
    from SFTbuild.utils import audit_tool_errors
    tool_audit = audit_tool_errors(agent_steps)
    if tool_audit.get('unrecovered_error_count', 0) > 0:
        issues.append(f'tool: {tool_audit["unrecovered_error_count"]} unrecovered errors')

    # ---- 3. Completeness ----
    fa_indices = [i for i, s in enumerate(agent_steps) if s.get('type') == 'final_answer']
    if not fa_indices:
        issues.append('completeness: no final_answer step')
    elif len(fa_indices) > 1:
        issues.append(f'completeness: {len(fa_indices)} final_answer steps (expected 1)')
    elif fa_indices[0] != len(agent_steps) - 1:
        issues.append(f'completeness: final_answer not last step (at {fa_indices[0]}, total {len(agent_steps)})')

    # ---- 4. Tool schema validation ----
    tool_issues = validate_tool_calls(agent_steps)
    for ti in tool_issues:
        issues.append(f'tool_schema: {ti}')

    # ---- 5. Data source coverage against gold related_tables ----
    if sample:
        model_ds = answer.get('data_source', [])
        if not isinstance(model_ds, list):
            model_ds = []
        checkout_list = sample.get('design', {}).get('checkout_list', [])
        sub_idx = rec.get('subquestion_id', 1) - 1
        if sub_idx < len(checkout_list):
            gold_tables = checkout_list[sub_idx].get('related_tables', []) or []
            if gold_tables:
                model_basenames = set(os.path.basename(t) for t in model_ds if isinstance(t, str))
                gold_basenames = set(os.path.basename(t) for t in gold_tables)
                missing = gold_basenames - model_basenames
                if missing:
                    issues.append(f'data_source: missing tables {list(missing)}')

    # ---- 6. Evidence verification ----
    # Check that numeric claims in answer are traceable to tool observations.
    # Build tagged observations: List[Tuple[str, Dict]] — each tagged with
    # the tool_name that produced it, so _verify_evidence can apply tiered trust.
    answer_text = answer.get('answer', '')
    if answer_text:
        tagged_obs = []
        for step in agent_steps:
            if step.get('type') != 'tool_call':
                continue
            # Build tool_call_id → tool_name map for this step
            cid_to_tool = {}
            for tc in step.get('tool_calls', []):
                if isinstance(tc, dict):
                    cid = tc.get('tool_call_id', '')
                    if cid:
                        cid_to_tool[cid] = tc.get('tool_name', 'unknown')
            for obs in step.get('observations', []):
                oid = obs.get('tool_call_id', '') if isinstance(obs, dict) else ''
                tool_name = cid_to_tool.get(oid, 'unknown')
                tagged_obs.append((tool_name, obs))
        from SFTbuild.utils import verify_evidence_with_fallback, _verify_evidence
        if client is not None:
            ev_pass, ev_missing = verify_evidence_with_fallback(
                answer_text, tagged_obs, client,
                user_question=rec.get('user', ''),
                goal=rec.get('sample_id', ''))
        else:
            ev_pass, ev_missing = _verify_evidence(answer_text, tagged_obs)
        if not ev_pass:
            issues.append(f'evidence: {len(ev_missing)} unsupported claims: {ev_missing[:5]}')

    return {
        'pass': len(issues) == 0,
        'issues': issues
    }


# ============================================================
# Validate structure (unchanged)
# ============================================================

def _reindex_call_ids(agent_steps: list) -> list:
    """Reindex tool_call_ids in agent_steps to ensure uniqueness.

    During iterative repair, each iteration may generate call_ids using the
    same template (e.g., call_1), leading to duplicates across steps.
    This reindexes all call_ids in sequential order so validate_tool_calls
    doesn't flag them as duplicates. Final reindexing is done by step8 stage 6.
    """
    seq = 1
    for step in agent_steps:
        if step.get('type') != 'tool_call':
            continue
        # Build old→new call_id mapping for this step
        mapping = {}
        for tc in step.get('tool_calls', []):
            old_id = tc.get('tool_call_id', '')
            new_id = f'call_{seq:03d}'
            mapping[old_id] = new_id
            tc['tool_call_id'] = new_id
            seq += 1
        # Update observations to match
        for obs in step.get('observations', []):
            if not isinstance(obs, dict):
                continue
            old_id = obs.get('tool_call_id', '')
            if old_id in mapping:
                obs['tool_call_id'] = mapping[old_id]
    return agent_steps


def validate_agent_steps(steps: list) -> list:
    """校验 agent_steps 基本结构，返回问题列表。"""
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
            # Observations should now be present (from real execution)
            obs = step.get('observations', [])
            if not isinstance(obs, list):
                issues.append(f'step[{i}]: observations is not a list')
        elif stype == 'final_answer':
            ans = step.get('assistant_answer', {})
            if not isinstance(ans, dict):
                issues.append(f'step[{i}]: assistant_answer is not a dict')
        else:
            issues.append(f'step[{i}]: unknown type "{stype}"')
    return issues


# ============================================================
# Iterative repair helpers
# ============================================================

def _build_failure_reasons(rec: dict, audit: dict) -> str:
    """Extract failure reasons from the audit record for a given subquestion."""
    for sa in audit.get('subquestion_audits', []):
        if sa.get('subquestion_id') == rec.get('subquestion_id'):
            return '\n'.join(f'- {i}' for i in sa.get('issues', []))
    return 'Unknown'


def _build_original_trajectory_text(rec: dict) -> str:
    """Build a text summary of the original (failed) agent trajectory."""
    parts = [f"User: {rec.get('user', '')}"]
    for step in rec.get('agent_steps', []):
        if step['type'] == 'tool_call':
            plan = step.get('step_plan', '')
            if plan:
                parts.append(f"Plan: {plan}")
            for tc in step.get('tool_calls', []):
                if isinstance(tc, dict):
                    parts.append(
                        f"Tool Call: {tc.get('tool_name', 'unknown')}({json.dumps(tc.get('arguments', {}), ensure_ascii=False)})")
                else:
                    parts.append(f"Tool Call: {tc}")
            for obs in step.get('observations', []):
                if isinstance(obs, dict):
                    parts.append(f"Observation: {obs.get('content', '')[:300]}")
                else:
                    parts.append(f"Observation: {str(obs)[:300]}")
        elif step['type'] == 'final_answer':
            ans = step.get('assistant_answer', {})
            parts.append(f"Answer: {json.dumps(ans, ensure_ascii=False)}")
    return '\n'.join(parts)


def _get_true_tables(rec: dict) -> list:
    """Get the list of true table filenames for this subquestion."""
    ev = rec.get('eval', {}) or {}
    td = ev.get('table_depend', {}) or {}
    true_tables = td.get('true_tables', [])
    if not true_tables:
        mb = rec.get('memory_before', {}) or {}
        for t in mb.get('tables', []):
            if isinstance(t, dict) and t.get('name'):
                true_tables.append(t['name'])
    return true_tables


def _format_execution_history(executed_steps: list) -> str:
    """Format executed steps as a readable history block for the iterative prompt."""
    if not executed_steps:
        return '（尚未执行任何步骤）'

    lines = []
    for i, entry in enumerate(executed_steps, 1):
        entry_type = entry.get('type', 'tool_call')

        if entry_type == 'rejected_answer':
            # Show the rejected answer and the verifier issues so the model
            # can fix them in the next attempt.
            answer = entry.get('answer', '')
            issues = entry.get('issues', [])
            lines.append(f'### 步骤 {i} — ⚠️ 答案被拒绝')
            lines.append(f'**被拒绝的答案**: {answer}')
            lines.append(f'**问题**:')
            for issue in issues:
                lines.append(f'  - {issue}')
            lines.append(f'**请修正以上问题后重新生成答案。**')
            lines.append('')
            continue

        plan = entry.get('plan', '')
        tool_call = entry.get('tool_call', '')
        observation = entry.get('observation', '')
        success = entry.get('success', False)

        lines.append(f'### 步骤 {i}')
        if plan:
            lines.append(f'**Plan**: {plan}')
        lines.append(f'**Tool Call**: {tool_call}')
        # Truncate observation to keep prompt within token budget
        obs_text = observation[:800] if observation else '（无返回）'
        lines.append(f'**Observation**: {obs_text}')
        status_label = '[SUCCESS]' if success else '[FAILED]'
        lines.append(f'**Status**: {status_label}')
        lines.append('')

    return '\n'.join(lines)


def build_iterative_repair_prompt(rec: dict, audit: dict,
                                   executed_steps: list, call_seq: int,
                                   dataset_root: str = _DEFAULT_DATASET_ROOT,
                                   sample: dict = None) -> str:
    """Build the repair prompt for the current iteration.

    If no steps have been executed yet, uses REPAIR_INITIAL_PROMPT.
    Otherwise uses REPAIR_ITERATIVE_PROMPT with full execution history.
    """
    failure_reasons = _build_failure_reasons(rec, audit)
    original_trajectory = _build_original_trajectory_text(rec)
    true_tables = _get_true_tables(rec)
    sample_file_path = (sample.get('file_path', '') or sample.get('table_path', '')) if sample else ''
    table_context = _build_table_context(true_tables, dataset_root,
                                          sample_file_path=sample_file_path)
    tool_constraints = get_tool_schema_constraints()
    execution_history = _format_execution_history(executed_steps)

    if not executed_steps:
        # First iteration — no history yet
        return REPAIR_INITIAL_PROMPT.format(
            user_question=rec.get('user', ''),
            original_trajectory=original_trajectory,
            failure_reasons=failure_reasons,
            table_context=table_context,
            tool_constraints=tool_constraints,
        )
    else:
        return REPAIR_ITERATIVE_PROMPT.format(
            user_question=rec.get('user', ''),
            original_trajectory=original_trajectory,
            failure_reasons=failure_reasons,
            table_context=table_context,
            tool_constraints=tool_constraints,
            execution_history=execution_history,
            call_seq=call_seq,
        )


def repair_subquestion(rec: dict, audit: dict, client,
                       dataset_root: str = _DEFAULT_DATASET_ROOT,
                       sample: dict = None,
                       verbose: bool = False) -> dict:
    """
    Iterative repair of a single subquestion.

    Loop:
      1. LLM decides next action (tool_call or final_answer)
      2. If tool_call → execute with real tools → append observation to history → loop
      3. If final_answer → validate and exit

    Each iteration the LLM sees the full execution history (Plan, Tool Call,
    Observation, [SUCCESS]/[FAILED]) and decides what to do next based on
    real observations rather than generating all steps in one shot.
    """
    execution_history = []  # list of {plan, tool_call, observation, success}
    agent_steps = []        # complete agent_steps being built
    call_seq = 1

    while call_seq <= _MAX_REPAIR_STEPS:
        # ---- Build prompt for current iteration ----
        prompt = build_iterative_repair_prompt(
            rec, audit, execution_history, call_seq, dataset_root,
            sample=sample
        )

        try:
            response = client.chat(
                prompt=prompt,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            result = extract_json_from_response(response)
        except Exception as e:
            rec['_repair_error'] = f'Iteration {call_seq}: LLM call failed: {e}'
            if verbose:
                print(f"    [Iter {call_seq}] LLM failed: {e}")
            return rec

        action = result.get('action', '')

        # ============================================================
        # Case 1: LLM outputs final_answer
        # ============================================================
        if action == 'final_answer':
            assistant_answer = result.get('assistant_answer', {})
            if not isinstance(assistant_answer, dict) or not assistant_answer.get('answer'):
                rec['_repair_error'] = f'Iteration {call_seq}: invalid assistant_answer'
                if verbose:
                    print(f"    [Iter {call_seq}] final_answer validation failed")
                return rec

            # ---- Gate: At least one successful tool call required ----
            # Since the repair prompt contains gold table content, the model
            # might skip tool calls and answer directly. Force at least one
            # successful tool execution before accepting final_answer.
            has_success = any(h.get('success') for h in execution_history)
            if not has_success:
                if verbose:
                    print(f"    [Iter {call_seq}] final_answer rejected: "
                          f"no successful tool call yet, forcing another iteration")
                # Feed rejection back into execution history so the model
                # sees why its answer was rejected and knows to execute tools.
                execution_history.append({
                    'type': 'rejected_answer',
                    'answer': json.dumps(assistant_answer, ensure_ascii=False)[:500],
                    'issues': ['No successful tool call executed. You must execute '
                               'at least one tool call and observe its result '
                               'before providing the final answer.']
                })
                call_seq += 1
                continue

            # ---- Gate: LLM-based Answer Verifier ----
            # Verify semantic consistency between answer and real observations
            if verbose:
                print(f"    [Iter {call_seq}] final_answer generated, running answer verifier...")

            all_obs_for_verify = [
                step.get('observations', [])
                for step in agent_steps if step.get('type') == 'tool_call'
            ]
            verify_result = verify_repair_answer(
                rec.get('user', ''), all_obs_for_verify,
                assistant_answer, client
            )

            if not verify_result['pass']:
                # Feed rejected answer + issues back into execution history so the
                # model sees what was wrong and can fix it. Without this, the model
                # at temperature=0 would just repeat the same rejected answer.
                answer_brief = json.dumps(assistant_answer, ensure_ascii=False)[:500]
                execution_history.append({
                    'type': 'rejected_answer',
                    'answer': answer_brief,
                    'issues': verify_result['issues']
                })
                if call_seq < _MAX_REPAIR_STEPS:
                    if verbose:
                        print(f"    [Iter {call_seq}] Answer verifier FAILED: "
                              f"{verify_result['issues'][:3]}, forcing another iteration")
                    call_seq += 1
                    continue
                else:
                    rec['_repair_error'] = (
                    f'Answer verifier failed at max steps: {"; ".join(verify_result["issues"])}'
                    )
                    if verbose:
                        print(f"    [Iter {call_seq}] Answer verifier FAILED at max steps")
                    return rec

            if verbose:
                print(f"    [Iter {call_seq}] Answer verifier PASSED "
                      f"(confidence: {verify_result['confidence']})")

            # ---- Gate: Independent Score-Point Judge ----
            # Unlike the answer verifier (which checks consistency with
            # observations), this judge evaluates the answer against the
            # benchmark's gold score_points using a separate verification
            # prompt. It catches answers that are internally consistent
            # but miss critical facts the benchmark expects.
            #
            # IMPORTANT: score_points are NOT fed back to the repair loop
            # if the judge fails — that would leak gold information into
            # the repair prompt. Instead, the repair is simply rejected.
            score_points = []
            gold_tables = []
            if sample:
                checkout_list = sample.get('design', {}).get('checkout_list', [])
                sub_idx = rec.get('subquestion_id', 1) - 1
                if sub_idx < len(checkout_list):
                    score_points = checkout_list[sub_idx].get('score_points', []) or []
                    gold_tables = checkout_list[sub_idx].get('related_tables', []) or []

            if score_points:
                if verbose:
                    print(f"    [Iter {call_seq}] Running independent score-point judge "
                          f"({len(score_points)} score points)...")

                judge_result = judge_repair_answer(
                    rec.get('user', ''), score_points, gold_tables,
                    assistant_answer, client
                )

                if not judge_result['pass']:
                    missing = judge_result.get('missing_points', [])
                    rec['_repair_error'] = (
                        f'Score-point judge: {len(missing)}/{len(score_points)} '
                        f'points missing'
                    )
                    rec['_judge_missing_points'] = missing
                    if verbose:
                        print(f"    [Iter {call_seq}] Score-point judge FAILED: "
                              f"{len(missing)}/{len(score_points)} points missing")
                    return rec

                if verbose:
                    n_covered = len(judge_result.get('covered_points', []))
                    print(f"    [Iter {call_seq}] Score-point judge PASSED "
                          f"({n_covered}/{len(score_points)} points covered)")
            else:
                if verbose:
                    print(f"    [Iter {call_seq}] No gold score_points available, "
                          f"skipping independent judge")

            # Assemble complete trajectory
            final_step_id = len(agent_steps) + 1
            full_agent_steps = agent_steps + [{
                'agent_step_id': final_step_id,
                'type': 'final_answer',
                'assistant_answer': assistant_answer
            }]

            # Evaluate on a deep copy so rec is never corrupted by an
            # exception inside re_evaluate_subquestion. Only commit on pass.
            candidate = copy.deepcopy(rec)
            candidate['agent_steps'] = _reindex_call_ids(full_agent_steps)
            candidate['assistant_answer'] = assistant_answer

            # ---- Re-evaluate (deterministic checks) ----
            try:
                eval_result = re_evaluate_subquestion(candidate, sample, client)
            except Exception as e:
                rec['_repair_error'] = (
                    f'Re-evaluation crashed: {e}'
                )
                if verbose:
                    print(f'    [Iter {call_seq}] CRASHED during re-eval: {e}')
                return rec

            if eval_result['pass']:
                # Commit: copy the validated fields back to rec
                rec['agent_steps'] = candidate['agent_steps']
                rec['assistant_answer'] = candidate['assistant_answer']
                rec['_sq_pass'] = True
                rec['_repaired'] = True
                rec['_repair_error'] = None  # Clear any error from previous iterations
                if verbose:
                    print(f"    [Iter {call_seq}] PASSED — repair successful ({len(agent_steps)} tool-call steps)")
            else:
                rec['_repair_error'] = f'Re-evaluation failed: {"; ".join(eval_result["issues"])}'
                if verbose:
                    issues = eval_result['issues']
                    print(f'    [Iter {call_seq}] FAILED: {issues}')
            return rec

        # ============================================================
        # Case 2: LLM outputs tool_call
        # ============================================================
        if action == 'tool_call':
            step_plan = result.get('step_plan', '')
            tool_calls = result.get('tool_calls', [])

            # Validate structure
            if not isinstance(tool_calls, list) or not tool_calls:
                rec['_repair_error'] = f'Iteration {call_seq}: tool_calls empty or not a list'
                if verbose:
                    print(f"    [Iter {call_seq}] tool_calls invalid: {tool_calls}")
                return rec

            for tc in tool_calls:
                if not isinstance(tc, dict) or 'tool_name' not in tc:
                    rec['_repair_error'] = f'Iteration {call_seq}: malformed tool_call entry'
                    if verbose:
                        print(f"    [Iter {call_seq}] malformed tool_call: {tc}")
                    return rec

            # Build a temporary step for execution
            step_for_exec = {
                'agent_step_id': len(agent_steps) + 1,
                'type': 'tool_call',
                'step_plan': step_plan,
                'tool_calls': tool_calls
            }

            # Execute with real tools
            if verbose:
                args_brief = json.dumps(tool_calls[0].get('arguments', {}), ensure_ascii=False)[:120]
                n_calls = len(tool_calls)
                print(f"    [Iter {call_seq}] Executing: {tool_calls[0].get('tool_name', '?')}({args_brief}){' ...' if n_calls > 1 else ''}")

            executed = execute_repair_tool_calls(
                [step_for_exec], dataset_root, call_seq=call_seq, verbose=False
            )

            if not executed:
                rec['_repair_error'] = f'Iteration {call_seq}: tool execution returned no steps'
                if verbose:
                    print(f"    [Iter {call_seq}] execution returned empty")
                return rec

            executed_step = executed[0]
            agent_steps.append(executed_step)

            # Append to execution history for next iteration
            for tc in executed_step.get('tool_calls', []):
                tc_id = tc.get('tool_call_id', '')
                tc_name = tc.get('tool_name', 'unknown')
                tc_args = json.dumps(tc.get('arguments', {}), ensure_ascii=False)

                # Find matching observation
                obs_content = ''
                obs_success = False
                for obs in executed_step.get('observations', []):
                    if obs.get('tool_call_id') == tc_id:
                        obs_content = obs.get('content', '')
                        obs_success = obs.get('success', False)
                        break

                execution_history.append({
                    'plan': step_plan,
                    'tool_call': f'{tc_name}({tc_args})',
                    'observation': obs_content,
                    'success': obs_success
                })

                if verbose:
                    status = 'OK' if obs_success else 'FAIL'
                    preview = obs_content[:100].replace('\n', ' ')
                    print(f"      [{status}] {preview}...")

            call_seq += 1
            continue

        # ============================================================
        # Unknown action
        # ============================================================
        rec['_repair_error'] = f'Iteration {call_seq}: unknown action "{action}"'
        if verbose:
            print(f"    [Iter {call_seq}] unknown action: {action}")
        return rec

    # Exceeded max steps
    rec['_repair_error'] = f'Exceeded max repair steps ({_MAX_REPAIR_STEPS}) without final_answer'
    if verbose:
        print(f"    Exceeded max steps ({_MAX_REPAIR_STEPS})")
    return rec


# ============================================================
# Main
# ============================================================

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description='Step 5: Repair failed sub-questions (real-execution v2)')
    parser.add_argument('--subquestions', type=str,
                        default=os.path.join(project_root, 'SFTbuild', 'output', 'evaluated_subquestions.jsonl'),
                        help='Path to evaluated_subquestions.jsonl from step3')
    parser.add_argument('--audit', type=str,
                        default=os.path.join(project_root, 'SFTbuild', 'output', 'audit_report.jsonl'),
                        help='Path to audit_report.jsonl from step4')
    parser.add_argument('--samples', type=str,
                        default=os.path.join(project_root, 'dataset', 'samples_normal_easy.json'),
                        help='Path to samples JSON (for re-evaluation)')
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
    parser.add_argument('--verbose', '-v', action='store_true', default=False)
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_root):
        parser.error(f"dataset_root does not exist: {args.dataset_root}")

    records = read_jsonl(args.subquestions)
    audit_records = read_jsonl(args.audit)

    if not audit_records:
        parser.error(
            f"Audit file is empty or missing: {args.audit}\n"
            f"  Step5 requires the step4 audit report to identify failed sub-questions.\n"
            f"  Run step4 first, or check that the audit file is correct."
        )

    # Load samples for re-evaluation
    sample_map = {}
    if args.samples and os.path.exists(args.samples):
        samples = load_samples(args.samples)
        for s in samples:
            task = (s.get('task', '') or '').strip()
            if task:
                sample_map[task] = s

    # Collect all audited keys and failed keys from audit
    audited_ids = set()
    failed_ids = set()
    for audit in audit_records:
        sid = audit.get('sample_id', '')
        cid = audit.get('candidate_id', '')
        for sa in audit.get('subquestion_audits', []):
            key = (sid, cid, sa.get('subquestion_id'))
            audited_ids.add(key)
            if not sa.get('pass'):
                failed_ids.add(key)

    if not audited_ids:
        print("[ERROR] Audit report contains no subquestion entries — all records marked as fail (fail closed)")
        for rec in records:
            rec['_sq_pass'] = False
            rec['_repair_error'] = 'Audit report has no subquestion entries — cannot verify pass status'
    else:
        n_failed = len(failed_ids)
        if n_failed == 0:
            print("No failed sub-questions found in audit.")
        else:
            print(f"Found {n_failed} failed sub-questions to repair")
        # Mark step4 pass status — per-record, fail closed for unaudited
        unaudited_count = 0
        for rec in records:
            key = (rec.get('sample_id', ''), rec.get('candidate_id', ''), rec.get('subquestion_id'))
            if key not in audited_ids:
                rec['_sq_pass'] = False
                rec['_repair_error'] = 'No matching audit record found'
                unaudited_count += 1
            else:
                rec['_sq_pass'] = key not in failed_ids
        if unaudited_count:
            print(f"  [WARN] {unaudited_count} records not found in audit → marked as fail (fail closed)")

    if args.dry_run:
        for rec in records:
            key = (rec.get('sample_id', ''), rec.get('candidate_id', ''), rec.get('subquestion_id'))
            if key in failed_ids:
                for audit in audit_records:
                    if audit.get('sample_id') == rec.get('sample_id') and audit.get('candidate_id') == rec.get('candidate_id'):
                        # Show initial prompt (no execution history yet)
                        sample = sample_map.get(rec.get('sample_id', ''))
                        prompt = build_iterative_repair_prompt(
                            rec, audit, executed_steps=[], call_seq=1,
                            dataset_root=args.dataset_root,
                            sample=sample
                        )
                        print(f"=== Initial Repair Prompt for {rec['candidate_id']}/sq{rec['subquestion_id']} ===")
                        print(prompt[:2000])
                        print("...")
                        return
        print("No failed sub-questions found.")
        return

    # Initialize LLM client
    from src.utils.chat_api import ChatClient
    client = ChatClient(config_key=args.config_key)

    repaired_count = 0
    repair_failed_count = 0
    for rec in records:
        key = (rec.get('sample_id', ''), rec.get('candidate_id', ''), rec.get('subquestion_id'))
        if key not in failed_ids:
            continue

        # Find matching audit
        audit_found = False
        for audit in audit_records:
            if audit.get('sample_id') == rec.get('sample_id') and audit.get('candidate_id') == rec.get('candidate_id'):
                audit_found = True
                if args.verbose:
                    print(f"\nRepairing {rec['candidate_id']}/sq{rec['subquestion_id']}...")

                sample = sample_map.get(rec.get('sample_id', ''))
                repair_subquestion(rec, audit, client, args.dataset_root,
                                   sample=sample, verbose=args.verbose)
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

    if repair_failed_count > 0:
        print(f"\n  [WARN] {repair_failed_count} sub-questions failed repair → kept original records")

    # ---- Recompute _dialog_pass after repair ----
    # Two conditions must hold for dialog-level pass:
    #   1. ALL existing sub-questions in this dialog are individually passing
    #      (_sq_pass or _repaired).
    #   2. The count/IDs of sub-questions match the checkout_list expectation.
    #      If checkout_list expects 4 sub-questions but we only have 3 records,
    #      the dialog is incomplete regardless of individual pass status.
    dialogs = {}
    for rec in records:
        key = (rec.get('sample_id', ''), rec.get('candidate_id', ''))
        dialogs.setdefault(key, []).append(rec)
    for (sample_id, candidate_id), sub_recs in dialogs.items():
        # Check 1: all existing sub-questions pass
        all_individual_pass = all(
            (r.get('_sq_pass') or r.get('_repaired'))
            for r in sub_recs
        )

        # Check 2: completeness — verify against sample's checkout_list
        sample = sample_map.get(sample_id)
        if sample is None:
            # Cannot verify checkout_list completeness without sample.
            # Still recompute _dialog_pass from individual sub-question status
            # — step4's value may be stale after repair.
            dialog_pass = all_individual_pass
            if args.verbose:
                print(f"  [DIALOG WARN] {candidate_id[:50]}: "
                      f"sample not found, _dialog_pass={dialog_pass} "
                      f"(based on individual pass only)")
            for r in sub_recs:
                r['_dialog_pass'] = dialog_pass
            continue

        checkout_list = sample.get('design', {}).get('checkout_list', [])
        expected_count = len(checkout_list)

        if expected_count == 0:
            # Empty checkout_list — cannot verify completeness. Fail closed
            # (consistent with step4_filter.py behavior).
            id_ok = False
            if args.verbose:
                print(f"  [DIALOG FAIL] {candidate_id[:50]}: "
                      f"checkout_list is empty, cannot verify completeness")
        else:
            actual_ids = {r.get('subquestion_id') for r in sub_recs}
            expected_ids = set(range(1, expected_count + 1))
            id_ok = (actual_ids == expected_ids)
            if not id_ok and args.verbose:
                missing = expected_ids - actual_ids
                extra = actual_ids - expected_ids
                detail = []
                if missing:
                    detail.append(f'missing sq={sorted(missing)}')
                if extra:
                    detail.append(f'extra sq={sorted(extra)}')
                print(f"  [DIALOG FAIL] {candidate_id[:50]}: "
                      f"ID mismatch (expected {expected_count}): {', '.join(detail)}")

        dialog_pass = all_individual_pass and id_ok
        for r in sub_recs:
            r['_dialog_pass'] = dialog_pass

    write_jsonl(args.output, records)
    n_repaired = repaired_count - repair_failed_count
    print(f"Done. Repaired: {n_repaired}/{repaired_count} attempted")
    print(f"  {len(records)} records → {args.output}")


if __name__ == '__main__':
    main()
