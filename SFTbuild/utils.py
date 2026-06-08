"""
SFTbuild 共享工具函数
"""
import json
import re
import os
from typing import List, Dict, Any, Optional, Tuple


# ---------------- Text normalization ----------------

def normalize_text(text: str) -> str:
    """去空格、去标点、统一全半角，用于模糊匹配"""
    text = text.strip()
    # 全角转半角
    result = []
    for ch in text:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        elif code == 0x3000:  # 全角空格
            result.append(' ')
        else:
            result.append(ch)
    text = ''.join(result)
    # 去掉所有空白和常见标点
    text = re.sub(r'[\s\u3000，。！？、；：“”‘’（）—…《》\.\,\!\?;:\"\'\(\)\-\[\]]+', '', text)
    return text.lower()


def normalize_text_light(text: str) -> str:
    """仅去多余空格、统一全半角"""
    text = text.strip()
    result = []
    for ch in text:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        elif code == 0x3000:
            result.append(' ')
        else:
            result.append(ch)
    text = ''.join(result)
    text = re.sub(r'\s+', ' ', text)
    return text


# ---------------- Trace loading ----------------

def load_trace(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_samples(filepath: str) -> List[Dict[str, Any]]:
    """Load benchmark samples from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_full_messages(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    取 conversation_trace[-1] 的 messages，并补上最后一个 response 作为 assistant message。
    """
    conv_trace = trace.get('conversation_trace', [])
    if not conv_trace:
        return []
    last_entry = conv_trace[-1]
    messages = list(last_entry.get('messages', []))
    response = last_entry.get('response', {})
    if response:
        assistant_msg = {'role': 'assistant', 'content': response.get('content', '')}
        if response.get('tool_calls'):
            assistant_msg['tool_calls'] = response['tool_calls']
        if response.get('reasoning_content'):
            assistant_msg['reasoning_content'] = response['reasoning_content']
        messages.append(assistant_msg)
    return messages


def find_sample(trace: Dict[str, Any], samples: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """根据 trace 的 metadata.query 匹配 samples 中的任务"""
    query = (trace.get('metadata', {}).get('query', '')).strip()
    if not query:
        return None
    # 优先精确匹配
    for sample in samples:
        task = (sample.get('task', '') or '').strip()
        if task == query:
            return sample
    # 无精确匹配时，再用子串匹配（取最长匹配以减少歧义）
    best = None
    best_len = 0
    for sample in samples:
        task = (sample.get('task', '') or '').strip()
        if task in query or query in task:
            if len(task) > best_len:
                best = sample
                best_len = len(task)
    return best


# ---------------- User anchor matching ----------------

def find_user_anchors(messages: List[Dict[str, Any]],
                      checkout_list: List[Dict[str, Any]],
                      accuracy_steps: Optional[List[Dict]] = None,
                      verbose: bool = False) -> List[Dict[str, Any]]:
    """
    在 messages 中定位每个 checkout_list[i].info_item 对应的 user message 位置。

    返回 anchors 列表，每个元素包含:
        - sub_idx: checkout_list 中的索引（0-based）
        - info_item: 原始 info_item 文本
        - msg_idx: 在 messages 中的位置
        - matched_text: 匹配到的 user message 的 content
    """
    # 提取所有 user message 的 (index, content)
    user_msgs = [(i, msg) for i, msg in enumerate(messages) if msg.get('role') == 'user']

    anchors = []
    for sub_idx, item in enumerate(checkout_list):
        info_item = item.get('info_item', '').strip()
        if not info_item:
            continue

        found_idx = None
        matched_text = None
        match_method = None

        # 策略1: 精确匹配
        for ui, um in user_msgs:
            if um['content'].strip() == info_item:
                found_idx = ui
                matched_text = um['content'].strip()
                match_method = 'exact'
                break

        # 策略2: normalize 后匹配
        if found_idx is None:
            norm_target = normalize_text(info_item)
            for ui, um in user_msgs:
                if normalize_text(um['content']) == norm_target:
                    found_idx = ui
                    matched_text = um['content'].strip()
                    match_method = 'normalized'
                    break

        # 策略3: substring 匹配（info_item 是 user message 的子串）
        if found_idx is None:
            for ui, um in user_msgs:
                if info_item in um['content']:
                    found_idx = ui
                    matched_text = um['content'].strip()
                    match_method = 'substring'
                    break

        # 策略4: 用 accuracy_steps 中的 query 匹配
        if found_idx is None and accuracy_steps:
            for step in accuracy_steps:
                step_query = step.get('query', '').strip()
                if step_query:
                    for ui, um in user_msgs:
                        if normalize_text(um['content']) == normalize_text(step_query):
                            found_idx = ui
                            matched_text = um['content'].strip()
                            match_method = 'accuracy_query'
                            break
                    if found_idx:
                        break

        anchors.append({
            'sub_idx': sub_idx,
            'info_item': info_item,
            'msg_idx': found_idx,
            'matched_text': matched_text,
            'match_method': match_method
        })

        if verbose and found_idx is None:
            print(f"  [WARN] Cannot match checkout[{sub_idx}]: {info_item[:80]}...")

    # ---- Resolve duplicate msg_idx conflicts ----
    # Two checkout items may match the same user message via different
    # strategies (e.g. exact and substring). Keep the best match and
    # mark the other as unmatched to prevent cross-contamination.
    METHOD_PRIORITY = {'exact': 0, 'normalized': 1, 'substring': 2, 'accuracy_query': 3}
    matched_anchors = [a for a in anchors if a['msg_idx'] is not None]
    by_idx = {}
    for a in matched_anchors:
        idx = a['msg_idx']
        if idx not in by_idx:
            by_idx[idx] = a
            continue
        # Conflict: same msg_idx claimed by two anchors
        existing = by_idx[idx]
        a_prio = METHOD_PRIORITY.get(a['match_method'], 99)
        e_prio = METHOD_PRIORITY.get(existing['match_method'], 99)
        if a_prio < e_prio:
            # New anchor wins — demote the existing one
            if verbose:
                print(f"  [WARN] Duplicate msg_idx={idx}: "
                      f"'{existing['info_item'][:60]}' ({existing['match_method']}) "
                      f"replaced by '{a['info_item'][:60]}' ({a['match_method']})")
            existing['msg_idx'] = None
            existing['match_method'] = f'unmatched_dup_of_{idx}'
            existing['matched_text'] = None
            by_idx[idx] = a
        else:
            # Existing anchor wins — demote this one
            if verbose:
                print(f"  [WARN] Duplicate msg_idx={idx}: "
                      f"'{a['info_item'][:60]}' ({a['match_method']}) "
                      f"overridden by '{existing['info_item'][:60]}' ({existing['match_method']})")
            a['msg_idx'] = None
            a['match_method'] = f'unmatched_dup_of_{idx}'
            a['matched_text'] = None

    return anchors


# ---------------- Agent step parsing ----------------

def parse_final_answer(content: str) -> Dict[str, Any]:
    """从 assistant content 中解析最终 JSON 答案"""
    if not content:
        return {'answer': '', 'data_source': []}

    # 尝试找 <answer> 标签
    answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    answers = answer_pattern.findall(content)
    if answers:
        try:
            parsed = json.loads(answers[0].strip())
            if isinstance(parsed, dict):
                return {
                    'answer': parsed.get('answer', answers[0].strip()),
                    'data_source': parsed.get('data_source', [])
                }
            # Bare int / list / string in <answer> tag — wrap it
            return {'answer': str(parsed), 'data_source': []}
        except json.JSONDecodeError:
            return {'answer': answers[0].strip(), 'data_source': []}

    # 尝试直接解析 JSON
    try:
        parsed = json.loads(content.strip())
        if isinstance(parsed, dict) and 'answer' in parsed:
            return {
                'answer': parsed.get('answer', ''),
                'data_source': parsed.get('data_source', [])
            }
    except json.JSONDecodeError:
        pass

    # 尝试找 JSON block
    json_pattern = re.compile(r'\{[^{}]*"answer"[^{}]*\}', re.DOTALL)
    match = json_pattern.search(content)
    if match:
        try:
            parsed = json.loads(match.group())
            return {
                'answer': parsed.get('answer', ''),
                'data_source': parsed.get('data_source', [])
            }
        except json.JSONDecodeError:
            pass

    # fallback: 整个 content 作为 answer
    return {'answer': content.strip(), 'data_source': []}


def validate_assistant_answer(answer: Any) -> Tuple[dict, list]:
    """Validate and normalize an assistant_answer to the canonical form.

    Canonical form:
        {"answer": "<non-empty str>", "data_source": ["<str>", ...]}

    Returns (normalized_answer, issues).
    If issues is non-empty, the answer is invalid — downstream consumers
    should treat the record as failed rather than proceeding to evidence
    verification or training data construction.
    """
    issues = []

    if not isinstance(answer, dict):
        issues.append('format: assistant_answer is not a dict')
        return {'answer': '', 'data_source': []}, issues

    ans_text = answer.get('answer', '')
    if not isinstance(ans_text, str) or not ans_text.strip():
        issues.append('format: answer is empty or not a string')

    ds = answer.get('data_source', [])
    if not isinstance(ds, list) or len(ds) == 0:
        issues.append('format: data_source is empty or not a list')
    elif not all(isinstance(d, str) for d in ds):
        issues.append('format: data_source contains non-string entries')

    return {
        'answer': ans_text if isinstance(ans_text, str) else str(ans_text),
        'data_source': ds if isinstance(ds, list) else [],
    }, issues


def extract_step_plan(reasoning_content: str) -> str:
    """从 reasoning_content 提取动作规划，压缩为 1-2 句简短意图。

    System Prompt 要求 "Keep plans concise — one or two sentences"。
    训练数据中的 plan 必须与 System Prompt 一致，否则模型会学到输出冗长推理链。
    """
    if not reasoning_content:
        return ''

    import re
    text = reasoning_content.strip()

    # Split by sentence boundaries (。！？! ?) and newlines
    sentences = re.split(r'(?<=[。！？!?\n])\s*', text)
    # Flatten: split on newlines in each chunk
    flat: list = []
    for s in sentences:
        parts = s.split('\n')
        flat.extend(p.strip() for p in parts if p.strip())

    # Keep first 2 non-empty sentences, cap total at ~200 chars
    kept: list = []
    total_chars = 0
    for s in flat:
        if total_chars + len(s) > 200:
            break
        kept.append(s)
        total_chars += len(s)
        if len(kept) >= 2:
            break

    if not kept:
        # Fallback: first 200 chars
        return text[:200]

    return ' '.join(kept)


def validate_agent_steps_integrity(steps: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Post-validate parsed agent_steps for structural integrity.

    Checks:
      1. Exactly one final_answer step (keep the last one if multiple)
      2. final_answer must be the last step (remove trailing tool_call steps after it)
      3. No empty tool_call steps (tool_calls + observations both empty)

    Returns:
      (cleaned_steps, warnings) — cleaned steps and human-readable warning messages.
    """
    warnings = []

    # ---- Check 1: multiple final_answer steps ----
    final_answer_indices = [
        i for i, s in enumerate(steps) if s.get('type') == 'final_answer'
    ]

    if len(final_answer_indices) == 0:
        warnings.append('no final_answer step found')
    elif len(final_answer_indices) > 1:
        # Keep only the last final_answer; remove earlier ones
        last_fa_idx = final_answer_indices[-1]
        removed_indices = final_answer_indices[:-1]
        steps = [s for i, s in enumerate(steps) if i not in removed_indices]
        # Recompute index after removal
        final_answer_indices = [
            i for i, s in enumerate(steps) if s.get('type') == 'final_answer'
        ]
        warnings.append(
            f'{len(final_answer_indices) + len(removed_indices)} final_answer steps found '
            f'(indices {removed_indices}), kept only the last one at index {final_answer_indices[0]}'
        )

    # ---- Check 2: final_answer must be the last step ----
    if final_answer_indices:
        last_fa_idx = final_answer_indices[-1]
        if last_fa_idx < len(steps) - 1:
            trailing_types = [s.get('type') for s in steps[last_fa_idx + 1:]]
            warnings.append(
                f'final_answer at index {last_fa_idx} is not the last step; '
                f'trailing steps: {trailing_types}. Removing trailing steps.'
            )
            steps = steps[:last_fa_idx + 1]

    # ---- Check 3: no empty tool_call steps ----
    cleaned = []
    for i, s in enumerate(steps):
        if s.get('type') == 'tool_call':
            tcs = s.get('tool_calls', [])
            obs = s.get('observations', [])
            if not tcs and not obs:
                warnings.append(f'empty tool_call step at index {i} removed')
                continue
        cleaned.append(s)

    # ---- Re-index agent_step_id ----
    for idx, s in enumerate(cleaned):
        s['agent_step_id'] = idx + 1

    return cleaned, warnings


def parse_agent_steps(span_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    从子问题 span 中解析 agent_steps。
    span_messages[0] 是 user question，从 span_messages[1:] 开始解析。

    解析后自动校验结构完整性：
      - 确保恰好一个 final_answer 且位于最后
      - 移除空的 tool_call 步骤
    """
    steps = []
    if not span_messages:
        return steps

    # 找到第一条 user message，从其下一条开始解析
    i = 0
    while i < len(span_messages) and span_messages[i].get('role') != 'user':
        i += 1
    i += 1  # 跳过 user message 本身
    while i < len(span_messages):
        msg = span_messages[i]
        if msg.get('role') == 'assistant':
            tool_calls = msg.get('tool_calls', [])
            if tool_calls:
                # ---- Tool call step ----
                step_id = len(steps) + 1
                tool_call_list = []
                tc_name_map = {}  # tool_call_id -> tool_name
                for tc in tool_calls:
                    func = tc.get('function', {})
                    args = func.get('arguments', '{}')
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                    tc_entry = {
                        'tool_call_id': tc.get('id', ''),
                        'tool_name': func.get('name', ''),
                        'arguments': args
                    }
                    tool_call_list.append(tc_entry)
                    tc_name_map[tc.get('id', '')] = func.get('name', '')

                # 收集后续 tool messages 作为 observations
                observations = []
                i += 1
                while i < len(span_messages) and span_messages[i].get('role') == 'tool':
                    tool_msg = span_messages[i]
                    tc_id = tool_msg.get('tool_call_id', '')
                    content = tool_msg.get('content', '')
                    observations.append({
                        'tool_call_id': tc_id,
                        'tool_name': tc_name_map.get(tc_id, 'unknown'),
                        'content': content,
                        'success': (
                            content and
                            '[ERROR]' not in content and
                            not content.split('\n')[0].startswith('Error')
                        )
                    })
                    i += 1

                steps.append({
                    'agent_step_id': step_id,
                    'type': 'tool_call',
                    'step_plan': extract_step_plan(msg.get('reasoning_content', '')),
                    'tool_calls': tool_call_list,
                    'observations': observations
                })
                continue  # i already advanced past tool messages

            else:
                # ---- Final answer step ----
                answer_data = parse_final_answer(msg.get('content', ''))
                steps.append({
                    'agent_step_id': len(steps) + 1,
                    'type': 'final_answer',
                    'assistant_answer': answer_data
                })
        i += 1

    # Post-validate structural integrity
    steps, _warnings = validate_agent_steps_integrity(steps)

    return steps


# ---------------- JSONL helpers ----------------

def write_jsonl(filepath: str, records: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def read_jsonl(filepath: str) -> List[Dict[str, Any]]:
    records = []
    if not os.path.exists(filepath):
        return records
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------- Robust JSON extraction ----------------

def extract_json_from_response(response: dict) -> dict:
    """从 LLM 响应中健壮地提取 JSON，同时检查 content 和 reasoning_content。"""
    text = response.get('content', '') or ''
    reasoning = response.get('reasoning_content') or ''

    for source in [text, reasoning]:
        if not source:
            continue
        source = source.strip()
        # 策略1: 直接解析
        try:
            return json.loads(source)
        except json.JSONDecodeError:
            pass
        # 策略2: 从 JSON code block 提取（greedy 匹配以支持嵌套 JSON）
        m = re.search(r'```(?:json)?\s*(\{.*\})\s*```', source, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # 策略3: 找到第一个 { 到最后一个 } 之间的内容
        start = source.find('{')
        end = source.rfind('}')
        if start >= 0 and end > start:
            try:
                return json.loads(source[start:end+1])
            except json.JSONDecodeError:
                pass

    raise ValueError(f"Failed to extract JSON from response. content='{text[:200]}', reasoning='{reasoning[:200]}'")


# ---------------- Tool error audit ----------------

def audit_tool_errors(agent_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Count tool errors and determine which are unrecovered.

    Recovery definition:
      An error is "recovered" only if there exists a subsequent tool_call step
      with a successful observation from the SAME tool, targeting the SAME
      resource (same file, table, folder, or query target).

      For python_code_executor: 'code' is excluded from key-arg comparison
      (broken code must change during recovery).  Recovery instead requires
      either (a) shared file references (pd.read_* / open() / Path()
      basenames) between the failed and recovery code, or (b) the recovery
      plan explicitly states fix/retry intent AND is within 3 steps of the
      failure.  This prevents unrelated successful Python calls from being
      counted as recovery.

      For file/table tools: the target path/folder must match between the
      failed and successful calls.

      An error at the last tool_call step (no later successful same-tool
      same-target call) is unrecovered, even if a final_answer exists.

    Non-dict observations (legacy format) are always unrecovered.
    """
    # Argument names that identify the "target" of a tool call.
    # Recovery requires at least one target value to match between the
    # failed call and the successful retry.
    #
    # NOTE: 'code' is deliberately excluded. python_code_executor recovery
    # involves FIXING broken code, so the code text will differ between the
    # failed and successful calls.  Instead, recovery requires shared file
    # references (pd.read_*/open/Path basenames) between the failed and
    # recovery code, or proximity + plan recovery intent (see second pass).
    _KEY_ARG_NAMES = {
        'file_path', 'path', 'filename', 'table_name', 'table_path',
        'directory', 'dir_path', 'search_path', 'folder_path',
        'query', 'pattern',
    }

    def _extract_key_values(tool_calls: list) -> set:
        """Extract key argument values from a list of tool_call dicts."""
        values = set()
        for tc in (tool_calls or []):
            if not isinstance(tc, dict):
                continue
            args = tc.get('arguments', {})
            if isinstance(args, dict):
                for k in _KEY_ARG_NAMES:
                    v = args.get(k)
                    if isinstance(v, str) and v.strip():
                        values.add(v.strip())
        return values

    def _extract_code_file_refs(code: str) -> set:
        """Extract file basenames referenced in Python code.

        Looks for pd.read_csv/read_excel/read_parquet/read_table,
        open(), and Path() calls with string literal arguments.
        """
        refs = set()
        if not code or not isinstance(code, str):
            return refs
        for m in re.finditer(r'pd\.read_\w+\(["\']([^"\']+)["\']', code):
            refs.add(os.path.basename(m.group(1)))
        for m in re.finditer(r'open\(["\']([^"\']+)["\']', code):
            refs.add(os.path.basename(m.group(1)))
        # Also catch Path(...) constructor calls
        for m in re.finditer(r'Path\(["\']([^"\']+)["\']', code):
            refs.add(os.path.basename(m.group(1)))
        return refs

    # IndexError recovery patterns: Chinese keywords for diagnosing IndexError
    _INDEXERROR_RECOVERY_PATTERNS = [
        r'检查列名', r'检查字段', r'检查索引',
        r'检查筛选', r'检查数据', r'检查表格',
        r'结果为空', r'数据为空', r'筛选结果',
        r'索引(超出|越界|错误)',
        r'列名(不|可能)', r'字段名(不|可能)',
    ]

    # NameError recovery patterns: Chinese/English keywords for diagnosing NameError
    _NAMEERROR_RECOVERY_PATTERNS = [
        r'变量未定义', r'变量丢失', r'变量(被清除|已清除)',
        r'重新读取数据', r'重新加载', r'重建变量',
        r'执行环境不保留', r'需要重新',
        r'丢失了', r'不存在',
        r'name.*not.*defined', r'is not defined',
    ]

    def _detect_error_type(obs_content: str) -> str:
        """Detect error type from observation content string."""
        if not isinstance(obs_content, str):
            return 'unknown'
        if 'IndexError' in obs_content:
            return 'IndexError'
        if 'NameError' in obs_content:
            return 'NameError'
        if 'KeyError' in obs_content:
            return 'KeyError'
        if 'TypeError' in obs_content:
            return 'TypeError'
        if 'ValueError' in obs_content:
            return 'ValueError'
        if 'AttributeError' in obs_content:
            return 'AttributeError'
        if 'FileNotFoundError' in obs_content:
            return 'FileNotFoundError'
        return 'unknown'

    def _match_any_pattern(text: str, patterns: list) -> bool:
        """Return True if any regex pattern matches text."""
        if not text or not isinstance(text, str):
            return False
        for pat in patterns:
            if re.search(pat, text):
                return True
        return False

    def _plan_has_recovery_intent(plan: str, strict: bool = False) -> bool:
        """Check if plan text indicates intent to fix/retry a previous error.

        When strict=True (python_code_executor): requires explicit mention of
        the error type, failure cause, or failed target — generic words like
        "重新"/"再次"/"调整" are NOT sufficient. This prevents unrelated
        successful Python calls from being counted as recovery.
        """
        if not plan or not isinstance(plan, str):
            return False

        if strict:
            # Must mention an error type, failure cause, or corrective action
            # targeting a specific previous failure — not just generic redo words.
            strict_patterns = [
                # Error type names
                r'\bNameError\b', r'\bIndexError\b', r'\bKeyError\b',
                r'\bTypeError\b', r'\bValueError\b', r'\bAttributeError\b',
                r'\bFileNotFoundError\b', r'\bModuleNotFoundError\b',
                r'\bKeyError\b', r'\bOSError\b', r'\bRuntimeError\b',
                # Chinese error descriptions
                r'变量未定义', r'索引超出', r'键错误', r'类型错误',
                r'值错误', r'找不到文件', r'模块未找到',
                r'名称错误', r'属性错误',
                # Specific corrective phrases (targeting a prior failure)
                r'修正(上述|前面|之前|刚才)的',
                r'修复(上述|前面|之前|刚才)的',
                r'纠正(上述|前面|之前|刚才)的',
                r'改正(上述|前面|之前|刚才)的',
                r'fix\s+(the\s+)?(above|previous|prior|earlier)',
                r'correct\s+(the\s+)?(above|previous|prior|earlier)',
            ]
            import re as _re
            for pat in strict_patterns:
                if _re.search(pat, plan, _re.IGNORECASE):
                    return True
            return False

        # Non-strict: general recovery/retry keywords
        keywords = [
            'fix', '修正', '修复', '纠正', '改正',
            'retry', '重试', '再次', '重新',
            'correct', '调整', '修改', '改用',
        ]
        plan_lower = plan.lower()
        return any(kw.lower() in plan_lower for kw in keywords)

    error_count = 0
    unrecovered_count = 0
    infrastructure_error_count = 0
    tool_details = []

    # First pass: collect error positions with failed call details
    # {step_index: [(tool_name, key_arg_values), ...]}
    error_steps: dict = {}
    for si, step in enumerate(agent_steps):
        if step['type'] != 'tool_call':
            continue
        tool_calls = step.get('tool_calls', [])
        for obs in step.get('observations', []):
            if not isinstance(obs, dict):
                error_count += 1
                unrecovered_count += 1
                error_steps.setdefault(si, []).append(('unknown', set()))
                tool_details.append({
                    'tool_name': 'unknown',
                    'success': False,
                    'error': str(obs)[:200]
                })
                continue
            if not obs.get('success', True):
                error_count += 1
                tool_name = obs.get('tool_name', 'unknown')

                # ---- Infrastructure error gate: BFloat16 ----
                # BFloat16 is a file-format issue, not a model mistake.
                # The model recovers by switching to cmd_executor / grep_search.
                # Don't treat these as unrecovered errors.
                content = obs.get('content', '')
                is_infra = (isinstance(content, str) and
                            'Got unsupported ScalarType BFloat16' in content)
                if is_infra:
                    infrastructure_error_count += 1
                    tool_details.append({
                        'tool_name': tool_name,
                        'success': False,
                        'error': content[:200],
                        'infrastructure': True
                    })
                    continue  # Skip error_steps → won't be counted as unrecovered

                # Match this observation to its tool_call to get arguments
                obs_call_id = obs.get('tool_call_id', '')
                key_vals = set()
                for tc in tool_calls:
                    if isinstance(tc, dict) and tc.get('tool_call_id') == obs_call_id:
                        args = tc.get('arguments', {})
                        if isinstance(args, dict):
                            for k in _KEY_ARG_NAMES:
                                v = args.get(k)
                                if isinstance(v, str) and v.strip():
                                    key_vals.add(v.strip())
                        break
                # Fallback: if no call_id match, extract from all calls in step
                if not key_vals:
                    key_vals = _extract_key_values(tool_calls)
                error_steps.setdefault(si, []).append((tool_name, key_vals))
                tool_details.append({
                    'tool_name': tool_name,
                    'success': False,
                    'error': obs.get('content', '')[:200]
                })

    # Second pass: for each error, check if a later step recovers
    # Recovery requires: same tool + at least one shared key argument value
    for err_si, failed_calls in error_steps.items():
        for tool_name, failed_key_vals in failed_calls:
            recovered = False
            for later_si in range(err_si + 1, len(agent_steps)):
                later_step = agent_steps[later_si]
                if later_step.get('type') != 'tool_call':
                    continue
                for obs in later_step.get('observations', []):
                    if not (isinstance(obs, dict) and obs.get('success') is True):
                        continue
                    if obs.get('tool_name', '') != tool_name:
                        continue
                    # Same tool succeeded — now check if same target
                    obs_call_id = obs.get('tool_call_id', '')
                    recovery_key_vals = set()
                    for tc in later_step.get('tool_calls', []):
                        if isinstance(tc, dict) and tc.get('tool_call_id') == obs_call_id:
                            args = tc.get('arguments', {})
                            if isinstance(args, dict):
                                for k in _KEY_ARG_NAMES:
                                    v = args.get(k)
                                    if isinstance(v, str) and v.strip():
                                        recovery_key_vals.add(v.strip())
                            break
                    if not recovery_key_vals:
                        recovery_key_vals = _extract_key_values(
                            later_step.get('tool_calls', [])
                        )
                    # Recovery: at least one target matches
                    if failed_key_vals and recovery_key_vals:
                        if failed_key_vals & recovery_key_vals:
                            recovered = True
                            break
                    elif not failed_key_vals:
                        # No key args to compare.
                        if tool_name == 'python_code_executor':
                            # 'code' is excluded from _KEY_ARG_NAMES, so
                            # same-tool success alone is too weak — any
                            # unrelated successful Python call would match.
                            # Require: (a) shared file refs between failed
                            # and recovery code, OR (b) recovery plan
                            # explicitly states fix intent + proximity.
                            failed_code = ''
                            recovery_code = ''
                            for tc in agent_steps[err_si].get('tool_calls', []):
                                if isinstance(tc, dict):
                                    failed_code = tc.get('arguments', {}).get('code', '')
                                    if failed_code:
                                        break
                            for tc in later_step.get('tool_calls', []):
                                if isinstance(tc, dict) and tc.get('tool_call_id') == obs_call_id:
                                    recovery_code = tc.get('arguments', {}).get('code', '')
                                    break

                            failed_refs = _extract_code_file_refs(failed_code)
                            recovery_refs = _extract_code_file_refs(recovery_code)
                            shared_refs = failed_refs & recovery_refs

                            proximity = (later_si - err_si) <= 3

                            # Get the failed observation content for error type detection
                            failed_content = ''
                            for o in agent_steps[err_si].get('observations', []):
                                if isinstance(o, dict) and not o.get('success', True):
                                    failed_content = o.get('content', '')
                                    break

                            error_type = _detect_error_type(failed_content)
                            plan = later_step.get('step_plan', '')

                            if error_type == 'IndexError':
                                plan_ok = _match_any_pattern(plan, _INDEXERROR_RECOVERY_PATTERNS)
                            elif error_type == 'NameError':
                                plan_ok = _match_any_pattern(plan, _NAMEERROR_RECOVERY_PATTERNS)
                            else:
                                plan_ok = _plan_has_recovery_intent(plan, strict=True)

                            if shared_refs or (proximity and plan_ok):
                                recovered = True
                                break
                        else:
                            # Non-python tool with no key args —
                            # fall back to same-tool check (legacy format)
                            recovered = True
                            break
                if recovered:
                    break

            if not recovered:
                unrecovered_count += 1

    return {
        'tool_call_count': sum(1 for s in agent_steps if s['type'] == 'tool_call'),
        'tool_error_count': error_count,
        'unrecovered_error_count': unrecovered_count,
        'infrastructure_error_count': infrastructure_error_count,
        'tools': tool_details
    }


# ---------------- Tool Schema Utilities ----------------


def get_tool_schema_constraints() -> str:
    """
    Generate a text description of all registered tools for use in repair prompts.
    Dynamically reads from TOOL_REGISTRY to ensure consistency with the actual tools.

    Raises RuntimeError if the tool registry cannot be imported or is empty,
    since repair prompts without tool definitions would produce unusable data.
    """
    from src.tools.base import TOOL_REGISTRY

    if not TOOL_REGISTRY:
        raise RuntimeError(
            "TOOL_REGISTRY is empty — cannot generate tool schema constraints. "
            "Ensure all tool modules are imported before calling this function."
        )

    lines = ["以下是可以使用的工具列表，每个工具的调用参数必须严格匹配其定义：", ""]
    for name, tool in sorted(TOOL_REGISTRY.items()):
        # Build parameter list
        param_parts = []
        required_params = []
        for pname, pinfo in tool.parameters.items():
            if isinstance(pinfo, dict):
                ptype = pinfo.get("type", "string")
                preq = "必填" if pinfo.get("required", False) else "可选"
                param_parts.append(f"{pname}: {ptype} ({preq})")
                if pinfo.get("required", False):
                    required_params.append(pname)
            else:
                param_parts.append(f"{pname}: string (可选)")

        params_str = ", ".join(param_parts) if param_parts else "无参数"
        lines.append(f"### {name}")
        lines.append(f"描述: {tool.description}")
        lines.append(f"参数: {params_str}")
        if required_params:
            lines.append(f"必填参数: {', '.join(required_params)}")
        lines.append("")

    lines.append("重要提醒：")
    lines.append("- file_viewer 工具不存在，读取表格文件请使用 table_head_reader")
    lines.append("- table_head_reader 的参数是 file_path/start/n，不存在 head 参数")
    lines.append("- 所有路径必须使用 <TABLE_ROOT>/ 作为前缀。格式: <TABLE_ROOT>/子目录/文件名（例如 <TABLE_ROOT>/chinese_table/example.xlsx）")

    return "\n".join(lines)


def validate_tool_calls(agent_steps: list, include_tools: list = None,
                        require_observations: bool = True) -> list:
    """
    Validate all tool calls in agent_steps against the tool schema actually enabled.

    Checks performed:
      1. tool_name exists in the enabled tool set
      2. arguments is a dict (or parseable JSON string)
      3. No extra undeclared parameters
      4. Required parameters are present
      5. Parameter types match the schema
      6. Enum values are valid (if defined)
      7. Each tool_call_id has a corresponding observation (when require_observations=True)
      8. No duplicate tool_call_id within the same sub-question

    Args:
        agent_steps: List of agent step dicts.
        include_tools: Optional list of tool names actually enabled.
                       When None, validates against the full TOOL_REGISTRY.
        require_observations: When True (default), each tool_call_id must have
                              a matching observation. Set to False for Stage A
                              repair validation where tool calls have not yet
                              been executed.

    Returns a list of issue strings. Empty list means all tool calls are valid.
    """
    import json as _json

    try:
        from src.tools.base import TOOL_REGISTRY
    except ImportError:
        return ["Tool registry not available for validation"]

    if not TOOL_REGISTRY:
        return ["Tool registry is empty"]

    # Build the set of tool names allowed for this validation
    all_tool_names = set(TOOL_REGISTRY.keys())
    if include_tools is not None:
        allowed_names = [n for n in include_tools if n in all_tool_names]
        unknown = [n for n in include_tools if n not in all_tool_names]
        if unknown:
            return [f"include_tools 包含未注册的工具: {unknown}"]
    else:
        allowed_names = list(all_tool_names)

    issues = []

    # Collect all tool_call_ids for duplicate / correspondence checks
    seen_call_ids = {}  # call_id → step_index
    all_call_ids = set()
    all_obs_call_ids = set()

    for step_idx, step in enumerate(agent_steps):
        if step.get("type") != "tool_call":
            continue

        tool_calls = step.get("tool_calls", [])
        observations = step.get("observations", [])

        for tc_idx, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                issues.append(f"step[{step_idx}].tool_calls[{tc_idx}]: not a dict → {tc}")
                continue

            tool_name = tc.get("tool_name", "")
            call_id = tc.get("tool_call_id", "")

            # 1. Tool name check
            if not tool_name:
                issues.append(f"step[{step_idx}].tool_calls[{tc_idx}]: missing tool_name")
                continue
            if tool_name not in allowed_names:
                issues.append(
                    f"step[{step_idx}].tool_calls[{tc_idx}]: 工具 '{tool_name}' 不在可用列表中。"
                    f"可用工具: {', '.join(sorted(allowed_names))}"
                )
                continue

            # 2. call_id presence
            if not call_id:
                issues.append(
                    f"step[{step_idx}].tool_calls[{tc_idx}] ({tool_name}): "
                    f"missing tool_call_id"
                )
                continue

            # 3. Duplicate call_id check
            if call_id in seen_call_ids:
                issues.append(
                    f"step[{step_idx}].tool_calls[{tc_idx}]: "
                    f"tool_call_id '{call_id}' 重复（首次出现在 step[{seen_call_ids[call_id]}]）"
                )
            seen_call_ids[call_id] = step_idx
            all_call_ids.add(call_id)

            # 3. Arguments parsing & type/extra/enum validation
            tool = TOOL_REGISTRY[tool_name]
            args = tc.get("arguments", {})

            # Parse string arguments
            if isinstance(args, str):
                try:
                    args = _json.loads(args)
                except _json.JSONDecodeError:
                    issues.append(
                        f"step[{step_idx}].tool_calls[{tc_idx}] ({tool_name}): "
                        f"arguments 无法解析为 JSON: {args[:120]}"
                    )
                    continue
            elif not isinstance(args, dict):
                issues.append(
                    f"step[{step_idx}].tool_calls[{tc_idx}] ({tool_name}): "
                    f"arguments 应为 dict 或 JSON string，实际为 {type(args).__name__}"
                )
                continue

            valid_params = tool.parameters
            valid_param_names = set(valid_params.keys())

            # 4. Extra undeclared parameters
            for arg_name in args:
                if arg_name not in valid_param_names:
                    issues.append(
                        f"step[{step_idx}].tool_calls[{tc_idx}] ({tool_name}): "
                        f"不接受参数 '{arg_name}'。有效参数: {', '.join(sorted(valid_param_names))}"
                    )

            # 5. Required params + type + enum check
            for pname, pinfo in valid_params.items():
                if not isinstance(pinfo, dict):
                    continue  # skip legacy string-only param definitions

                ptype = pinfo.get("type", "string")
                preq = pinfo.get("required", False)
                penum = pinfo.get("enum")

                if preq and pname not in args:
                    issues.append(
                        f"step[{step_idx}].tool_calls[{tc_idx}] ({tool_name}): "
                        f"缺少必填参数 '{pname}'"
                    )
                    continue

                if pname not in args:
                    continue

                val = args[pname]

                # Type check
                type_ok, type_msg = _check_param_type(val, ptype)
                if not type_ok:
                    issues.append(
                        f"step[{step_idx}].tool_calls[{tc_idx}] ({tool_name}): "
                        f"参数 '{pname}' 类型错误: {type_msg}（期望 {ptype}）"
                    )

                # Enum check
                if penum and val not in penum:
                    issues.append(
                        f"step[{step_idx}].tool_calls[{tc_idx}] ({tool_name}): "
                        f"参数 '{pname}' 值 '{val}' 不在允许范围内: {penum}"
                    )

        # 6. Observation call_id checks
        step_obs_ids = set()
        for oi, obs in enumerate(observations):
            if not isinstance(obs, dict):
                issues.append(f"step[{step_idx}].observations[{oi}]: not a dict")
                continue
            oid = obs.get("tool_call_id", "")
            if not oid:
                issues.append(
                    f"step[{step_idx}].observations[{oi}]: missing tool_call_id"
                )
                continue
            if oid in step_obs_ids:
                issues.append(
                    f"step[{step_idx}].observations[{oi}]: "
                    f"duplicate observation tool_call_id '{oid}'"
                )
                continue
            step_obs_ids.add(oid)
            all_obs_call_ids.add(oid)

    # 7. Cross-check: each tool_call_id must have exactly one observation,
    #    and each observation must correspond to a known tool_call.
    if require_observations:
        for call_id in all_call_ids:
            if call_id not in all_obs_call_ids:
                issues.append(
                    f"tool_call_id '{call_id}' 没有对应的 observation"
                )
        for obs_id in all_obs_call_ids:
            if obs_id not in all_call_ids:
                issues.append(
                    f"observation tool_call_id '{obs_id}' 没有对应的 tool_call"
                )

    return issues


def _check_param_type(value, expected_type: str) -> tuple:
    """Check if value matches the expected JSON schema type. Returns (ok, message)."""
    if expected_type == "string":
        if not isinstance(value, str):
            return False, f"期望 string，实际 {type(value).__name__}"
    elif expected_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return False, f"期望 integer，实际 {type(value).__name__}"
    elif expected_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False, f"期望 number，实际 {type(value).__name__}"
    elif expected_type == "boolean":
        if not isinstance(value, bool):
            return False, f"期望 boolean，实际 {type(value).__name__}"
    elif expected_type == "array":
        if not isinstance(value, list):
            return False, f"期望 array，实际 {type(value).__name__}"
    elif expected_type == "object":
        if not isinstance(value, dict):
            return False, f"期望 object，实际 {type(value).__name__}"
    # Unrecognized types are accepted (don't block valid data)
    return True, ""


def validate_tool_call_ids(agent_steps: list) -> list:
    """
    Check that each tool_call step has observations covering all its tool_call_ids.
    This is done inside validate_tool_calls() already, but kept as a standalone
    for callers that only want the id-level checks.
    """
    issues = []
    for step_idx, step in enumerate(agent_steps):
        if step.get("type") != "tool_call":
            continue
        tool_calls = step.get("tool_calls", [])
        observations = step.get("observations", [])

        tc_ids = set()
        for tc in tool_calls:
            cid = tc.get("tool_call_id", "")
            if cid:
                tc_ids.add(cid)

        obs_ids = set()
        for obs in observations:
            if isinstance(obs, dict):
                oid = obs.get("tool_call_id", "")
                if oid:
                    obs_ids.add(oid)

        missing = tc_ids - obs_ids
        extra = obs_ids - tc_ids
        if missing:
            issues.append(
                f"step[{step_idx}]: tool_call_ids 无对应 observation: {missing}"
            )
        if extra:
            issues.append(
                f"step[{step_idx}]: observations 引用了不存在的 tool_call_ids: {extra}"
            )

    return issues


def normalize_paths_in_messages(messages: list) -> list:
    """
    Replace only known table-root paths with <TABLE_ROOT> placeholder.

    Replaces:
      - /tmp/data/task_<id>/dataset/tables/...  → <TABLE_ROOT>/...
      - /data/zenghaoyang/TableAgentBench/dataset/tables/... → <TABLE_ROOT>/...

    Does NOT replace:
      - Other absolute paths outside dataset/tables (these should be rejected or use <WORK_ROOT>)
    """
    import re

    replacements = [
        # Paths with trailing slash
        (re.compile(r'/tmp/data/task_\w+/dataset/tables/'), '<TABLE_ROOT>/'),
        (re.compile(r'/data/zenghaoyang/TableAgentBench/dataset/tables/'), '<TABLE_ROOT>/'),
        # Paths without trailing slash (directory references, JSON strings, etc.)
        (re.compile(r'/tmp/data/task_\w+/dataset/tables(?=[\"\'}\s，。；、！？\)\]\u3000]|$)'), '<TABLE_ROOT>/'),
        (re.compile(r'/data/zenghaoyang/TableAgentBench/dataset/tables(?=[\"\'}\s，。；、！？\)\]\u3000]|$)'), '<TABLE_ROOT>/'),
    ]

    def _replace(text):
        if not isinstance(text, str):
            return text
        for pattern, replacement in replacements:
            text = pattern.sub(replacement, text)
        return text

    for msg in messages:
        if "content" in msg and isinstance(msg["content"], str):
            msg["content"] = _replace(msg["content"])

        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            args_str = func.get("arguments", "")
            if isinstance(args_str, str) and args_str:
                if "/tmp/data/task_" in args_str or "/data/zenghaoyang" in args_str:
                    func["arguments"] = _replace(args_str)

    return messages


def has_unresolved_absolute_paths(messages: list) -> list:
    """
    Check if messages still contain absolute paths that were NOT normalized.
    Returns a list of issue strings (non-table absolute paths that need attention).
    """
    import re

    # Detect remaining absolute paths (but exclude <TABLE_ROOT> placeholder)
    abs_path_re = re.compile(r'(?<!<TABLE_ROOT>)/(?:tmp/data|data/zenghaoyang|home/\w+)/\S+')

    issues = []
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        if isinstance(content, str):
            matches = abs_path_re.findall(content)
            for m in matches:
                issues.append(f"msg[{i}] ({msg.get('role', '?')}): 未归一化的绝对路径: {m}")

        for tc in msg.get("tool_calls", []):
            args_str = tc.get("function", {}).get("arguments", "")
            if isinstance(args_str, str):
                matches = abs_path_re.findall(args_str)
                for m in matches:
                    issues.append(f"msg[{i}] tool_call args: 未归一化的绝对路径: {m}")

    return issues


# ============================================================================
# 9-Stage Cleaning Pipeline (Issues 4 & 5)
# ============================================================================

def _normalize_call_signature(tool_name: str, arguments: dict) -> str:
    """Generate a canonical string signature for a tool call to compare duplicates."""
    import json as _json
    args_str = _json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)
    return f"{tool_name}|{args_str}"


def _compare_answers(ans1: dict, ans2: dict, strict: bool = True) -> bool:
    """
    Compare two answer dicts for equality.

    When strict=True: exact match on answer text + sorted data_source.
    When strict=False: data_source must match exactly, answer text uses normalized comparison
      (numeric values must overlap significantly, text must be similar).
    """
    a1 = ans1.get('answer', '') if isinstance(ans1, dict) else ''
    a2 = ans2.get('answer', '') if isinstance(ans2, dict) else ''
    ds1 = sorted(ans1.get('data_source', [])) if isinstance(ans1, dict) else []
    ds2 = sorted(ans2.get('data_source', [])) if isinstance(ans2, dict) else []

    # Data source must always match
    if ds1 != ds2:
        return False

    if strict:
        return a1 == a2

    # Lenient comparison: normalize text and check numeric content overlap
    import re

    def _normalize(s):
        # Remove extra whitespace, unify to single spaces
        s = re.sub(r'\s+', ' ', s.strip())
        return s

    norm1 = _normalize(a1)
    norm2 = _normalize(a2)

    # If normalized texts match exactly → pass
    if norm1 == norm2:
        return True

    # Extract numeric values from both answers
    vals1, pcts1 = _extract_numeric_values(a1)
    vals2, pcts2 = _extract_numeric_values(a2)
    nums1 = vals1 | pcts1
    nums2 = vals2 | pcts2

    if not nums1 and not nums2:
        # No numeric content → fall back to substring check.
        # Guard: empty string is a substring of everything and would
        # produce false positives (e.g. empty presentation-call answer
        # matching an unrelated non-numeric final answer).
        if not norm1 or not norm2:
            return norm1 == norm2
        return norm1 in norm2 or norm2 in norm1

    # Check numeric overlap: at least 80% of values must match
    if nums1 and nums2:
        overlap = nums1 & nums2
        # Use the smaller set as baseline
        min_size = min(len(nums1), len(nums2))
        if min_size > 0 and len(overlap) / min_size >= 0.8:
            return True

    return False


def _compare_answers_strict(ans1: dict, ans2: dict) -> bool:
    """Exact answer comparison for stage 7 validation."""
    return _compare_answers(ans1, ans2, strict=True)


def _compare_answers_lenient(ans1: dict, ans2: dict) -> bool:
    """Lenient answer comparison for stage 4 presentation call removal."""
    return _compare_answers(ans1, ans2, strict=False)


# ---- Stage 1: detect_call_id_conflicts ----

def detect_call_id_conflicts(sub_records: list) -> tuple:
    """
    Scan each sub-question independently for call_id conflicts.

    Scope: per-subquestion — the same call_id (e.g. call_1) is expected to be
    reused across different sub-questions by the original system. Those are NOT
    conflicts; Stage 6 reindexes them to globally unique IDs.

    Within each sub-question, if the same call_id appears with DIFFERENT
    (tool_name, args, observation_content), it IS a real conflict.

    Observation comparison uses full content (not truncated), to avoid missing
    differences in the second half of long observations.

    Returns:
      (has_conflict: bool, conflicts: dict)
        conflicts maps call_id → list of {subquestion_id, step_idx, tool_name, args, obs_full}
    """
    from collections import defaultdict

    conflicts = {}

    for rec in sub_records:
        sq_id = rec.get('subquestion_id', '?')

        # Per-subquestion scope: call_id → occurrences within this subquestion only
        call_occurrences = defaultdict(list)

        for si, step in enumerate(rec.get('agent_steps', [])):
            if step.get('type') != 'tool_call':
                continue
            for tc in step.get('tool_calls', []):
                if not isinstance(tc, dict):
                    continue
                cid = tc.get('tool_call_id', '')
                if not cid:
                    continue
                tool_name = tc.get('tool_name', '')
                args = tc.get('arguments', {})
                sig = _normalize_call_signature(tool_name, args)

                # Find corresponding observation — use full content
                obs_full = ''
                for obs in step.get('observations', []):
                    if isinstance(obs, dict) and obs.get('tool_call_id') == cid:
                        obs_full = obs.get('content', '')
                        break

                call_occurrences[cid].append({
                    'subquestion_id': sq_id,
                    'step_idx': si,
                    'tool_name': tool_name,
                    'args': args,
                    'signature': sig,
                    'obs_full': obs_full
                })

        # Check for conflicts within this subquestion
        for cid, occurrences in call_occurrences.items():
            if len(occurrences) <= 1:
                continue

            first_sig = occurrences[0]['signature']
            first_obs = occurrences[0]['obs_full']
            all_same = all(
                occ['signature'] == first_sig and occ['obs_full'] == first_obs
                for occ in occurrences[1:]
            )

            if not all_same:
                # Prepend subquestion_id to key to avoid cross-sq key collisions
                conflict_key = f'sq{sq_id}_{cid}'
                conflicts[conflict_key] = occurrences

    return len(conflicts) > 0, conflicts


# ---- Stage 2: deduplicate_log_calls ----

def deduplicate_log_calls(agent_steps: list) -> list:
    """
    Remove duplicate tool calls within a single sub-question (entire scope, not just consecutive).

    For each call_id: keep the first occurrence, delete all subsequent duplicates.
    Deletion is per-call granularity (not per-step).
    If all calls in a step are deleted, the step is removed.

    Args:
        agent_steps: List of agent step dicts for one sub-question.

    Returns:
        Cleaned agent_steps list.
    """
    seen_call_ids = set()
    cleaned_steps = []

    for step in agent_steps:
        if step.get('type') != 'tool_call':
            cleaned_steps.append(step)
            continue

        # Filter tool_calls: keep only first occurrence of each call_id
        kept_calls = []
        kept_call_ids = set()
        for tc in step.get('tool_calls', []):
            cid = tc.get('tool_call_id', '') if isinstance(tc, dict) else ''
            if not cid:
                kept_calls.append(tc)
                continue
            if cid not in seen_call_ids:
                seen_call_ids.add(cid)
                kept_calls.append(tc)
                kept_call_ids.add(cid)
            # else: duplicate → skip

        if not kept_calls:
            # All calls in this step were duplicates → skip the step entirely
            continue

        # Filter observations: only keep those for kept calls
        kept_obs = []
        for obs in step.get('observations', []):
            oid = obs.get('tool_call_id', '') if isinstance(obs, dict) else ''
            if oid in kept_call_ids:
                kept_obs.append(obs)

        new_step = dict(step)
        new_step['tool_calls'] = kept_calls
        new_step['observations'] = kept_obs
        cleaned_steps.append(new_step)

    return cleaned_steps


# ---- Stage 3: filter_bf16_errors ----

def filter_bf16_errors(sub_records: list, max_bf16_keep: int = 0) -> tuple:
    """
    Remove BFloat16/ScalarType error calls from all sub-records.

    For each BFloat16 error:
      1. Delete the call + its observation (per-call granularity)
      2. Record the step index as "affected by BFloat16 removal"
      3. If all calls in a step are deleted → remove the step
      4. If BFloat16 error was recovered (has successor calls with different approach) →
         handle based on max_bf16_keep policy

    Args:
        sub_records: List of sub-question records.
        max_bf16_keep: Maximum BFloat16 traces to keep (0 = delete all).

    Returns:
        (cleaned_sub_records, removed_bf16_step_indices_per_sq)
        removed_bf16_step_indices_per_sq: {sq_idx: [step_indices_affected_by_bf16_removal]}
    """
    import json as _json

    cleaned_records = []
    removed_indices = {}  # sq_idx → [step_indices where BFloat16 calls were removed]

    # Clause-level BFloat16 cleanup: remove clauses referencing BFloat16/ScalarType
    # errors, tool execution errors, or "出现了错误". Operates at clause granularity
    # within sentences (splits on 。！？.!? then on ，,；;：:) so that action
    # descriptions are preserved even when they share a sentence with error text.
    # This handles real trajectory text like:
    #   "出现了一个工具执行错误，提示"Got unsupported ScalarType BFloat16"。现在使用cmd_executor列出文件。"
    #   "出现了错误："Got unsupported ScalarType BFloat16"。尝试另一种方式执行。"
    #   "出现了错误。可能是因为模型不支持BFloat16。接下来使用cmd_executor执行。"
    #   "出现错误，换一种方法：使用cmd_executor列出目录。"
    _bf16_clause_patterns = [
        re.compile(r'BFloat16', re.IGNORECASE),
        re.compile(r'ScalarType', re.IGNORECASE),
        re.compile(r'Got unsupported', re.IGNORECASE),
        re.compile(r'工具执行错误', re.IGNORECASE),
        re.compile(r'出现了错误', re.IGNORECASE),
        re.compile(r'出现错误', re.IGNORECASE),
        re.compile(r'不支持.*BFloat16', re.IGNORECASE),
    ]

    # Orphaned recovery-transition prefixes to strip after BFloat16 clauses are
    # removed. These only make sense when preceded by an error description.
    _orphaned_recovery_prefixes = [
        re.compile(r'^换[一另]种(方式|方法)[：:，,]*\s*', re.IGNORECASE),
        re.compile(r'^尝试[重另]新[执行]*[：:，,]*\s*', re.IGNORECASE),
        re.compile(r'^重试[：:，,]*\s*', re.IGNORECASE),
        re.compile(r'^改用其他方式[：:，,]*\s*', re.IGNORECASE),
    ]

    def _clean_bf16_from_plan(plan: str) -> str:
        """Remove clauses referencing BFloat16/ScalarType errors from plan text.

        Works at clause granularity within sentences so that an action
        description sharing a sentence with an error mention is preserved.
        """
        if not plan:
            return plan

        # Step 1: Split into sentences on major boundaries
        sentences = re.split(r'(?<=[。！？\.\!\?])\s*', plan)
        kept_sentences = []

        for s in sentences:
            stripped = s.strip()
            if not stripped:
                continue

            # Step 2: Split sentence into clauses on minor boundaries
            clauses = re.split(r'(?<=[，,；;：:])\s*', stripped)

            kept_clauses = []
            for clause in clauses:
                c = clause.strip()
                if not c:
                    continue
                if any(p.search(c) for p in _bf16_clause_patterns):
                    continue
                kept_clauses.append(clause)

            if kept_clauses:
                joined = ''.join(kept_clauses).strip()
                # Step 3: Strip orphaned recovery prefixes from the joined result
                for pat in _orphaned_recovery_prefixes:
                    joined = pat.sub('', joined)
                if joined:
                    kept_sentences.append(joined)

        result = ''.join(kept_sentences).strip()
        # Clean up doubled/leading punctuation from clause removal
        result = re.sub(r'[，,]\s*[，,]', '，', result)
        result = re.sub(r'^[，,；;：:]\s*', '', result)
        return result

    for rec_idx, rec in enumerate(sub_records):
        agent_steps = rec.get('agent_steps', [])
        cleaned_steps = []
        sq_removed_indices = []
        sq_fully_removed_count = 0
        # When a step is fully removed (all calls were BFloat16), the NEXT
        # step's plan may still reference the now-gone error. Track whether
        # the previous step was fully removed so we can clean the current
        # step's plan with sentence-level removal.
        prev_step_fully_removed = False

        for si, step in enumerate(agent_steps):
            if step.get('type') != 'tool_call':
                cleaned_steps.append(step)
                prev_step_fully_removed = False
                continue

            # Check each observation for BFloat16
            bf16_call_ids = set()
            tool_calls = step.get('tool_calls', [])
            for oi, obs in enumerate(step.get('observations', [])):
                content = obs.get('content', '') if isinstance(obs, dict) else str(obs)
                if 'BFloat16' in content or 'ScalarType' in content:
                    if isinstance(obs, dict):
                        oid = obs.get('tool_call_id', '')
                    else:
                        # Legacy string observation: try positional match
                        if oi < len(tool_calls) and isinstance(tool_calls[oi], dict):
                            oid = tool_calls[oi].get('tool_call_id', '')
                        else:
                            oid = ''
                    if oid:
                        bf16_call_ids.add(oid)

            if not bf16_call_ids:
                # This step has no BFloat16 errors, but if the previous step
                # was fully removed, this step's plan may still reference
                # the now-gone error. Do sentence-level cleanup.
                if prev_step_fully_removed:
                    plan = step.get('step_plan', '')
                    cleaned_plan = _clean_bf16_from_plan(plan)
                    if cleaned_plan != plan:
                        step = dict(step)
                        step['step_plan'] = cleaned_plan
                cleaned_steps.append(step)
                prev_step_fully_removed = False
                continue

            # Filter out BFloat16 calls
            kept_calls = []
            kept_call_ids = set()
            for tc in step.get('tool_calls', []):
                cid = tc.get('tool_call_id', '') if isinstance(tc, dict) else ''
                if cid in bf16_call_ids:
                    continue  # Delete BFloat16 call
                kept_calls.append(tc)
                kept_call_ids.add(cid)

            if not kept_calls:
                # All calls were BFloat16 → remove entire step.
                # The next step's plan may reference this error; flag it
                # so the next tool_call step gets its plan cleaned.
                prev_step_fully_removed = True
                sq_fully_removed_count += 1
                continue

            # Partial removal: keep remaining calls and their observations
            kept_obs = []
            for obs in step.get('observations', []):
                oid = obs.get('tool_call_id', '') if isinstance(obs, dict) else ''
                if oid in kept_call_ids:
                    kept_obs.append(obs)
                # BFloat16 observations are dropped

            new_step = dict(step)
            new_step['tool_calls'] = kept_calls
            new_step['observations'] = kept_obs

            # Clean the plan text inline for this partially-affected step.
            # Use sentence-level removal to handle cases like:
            #   "出现了错误：Got unsupported ScalarType BFloat16。换一种方法重试。"
            plan = new_step.get('step_plan', '')
            cleaned_plan = _clean_bf16_from_plan(plan)
            new_step['step_plan'] = cleaned_plan

            cleaned_steps.append(new_step)
            # Record the index in the CLEANED step list (not original),
            # so cleanup_orphaned_content can still reference it if needed.
            sq_removed_indices.append(len(cleaned_steps) - 1)
            prev_step_fully_removed = False

        new_rec = dict(rec)
        new_rec['agent_steps'] = cleaned_steps
        cleaned_records.append(new_rec)
        if sq_removed_indices or sq_fully_removed_count:
            removed_indices[rec_idx] = {
                'partial_removed': sq_removed_indices,
                'fully_removed_count': sq_fully_removed_count,
            }

    return cleaned_records, removed_indices


# ---- Stage 4: remove_presentation_calls ----

def _trace_variable_sources(tree: "ast.AST") -> dict:
    """
    Trace variable assignments in an AST to determine if all variable values
    ultimately come from Constant/Dict/List literals (static) or from
    external sources like function calls, BinOps, etc. (non-static).

    Returns: {var_name: 'static' | 'dynamic'}
    """
    import ast as _ast

    sources = {}

    def _is_static_node(node):
        """Check if a node represents a static/literal value."""
        if isinstance(node, _ast.Constant):
            return True
        if isinstance(node, (_ast.Dict, _ast.List, _ast.Tuple)):
            return all(_is_static_node(elt) for elt in (
                node.elts if isinstance(node, (_ast.List, _ast.Tuple))
                else list(node.keys) + list(node.values)
            ))
        if isinstance(node, _ast.Name):
            # If the variable was assigned from static sources
            return sources.get(node.id, 'dynamic') == 'static'
        if isinstance(node, _ast.JoinedStr):
            return all(
                _is_static_node(v) if isinstance(v, _ast.Constant) else False
                for v in node.values
            )
        return False

    for stmt in _ast.walk(tree):
        if isinstance(stmt, _ast.Assign):
            for target in stmt.targets:
                if isinstance(target, _ast.Name):
                    if _is_static_node(stmt.value):
                        sources[target.id] = 'static'
                    else:
                        sources[target.id] = 'dynamic'

    return sources


def _is_presentation_only_call(code: str) -> bool:
    """
    AST-based check: is this Python code purely for JSON presentation?

    Returns True only if ALL of:
      1. No file I/O (open, pd.read_*, DataFrame constructor)
      2. No computation — no arithmetic BinOp, AND no builtins that
         transform data (sum, min, max, sorted, abs, round, etc.)
      3. No control flow: no loops (while/for), no conditionals (if/elif/else),
         no comprehensions — these indicate logic, not pure formatting
      4. All variable values trace to Constant/Dict/List literals
      5. Final output is print(json.dumps(...)) with purely static arguments
    """
    import ast as _ast

    if not code or not isinstance(code, str):
        return False

    # Must have print of JSON content:
    #   print(json.dumps(...)) OR print('''{\n  "answer":...}''') (raw JSON string)
    has_json_dumps = 'json.dumps' in code
    has_raw_json_print = False
    if not has_json_dumps:
        # Match print('...') or print("...") or print('''...''') with JSON content
        for m in __import__('re').finditer(r'print\s*\(\s*["\']+\s*\{', code):
            has_raw_json_print = True
            break
    if not has_json_dumps and not has_raw_json_print:
        return False
    if 'print' not in code:
        return False

    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return False

    # Check 1: No control flow — no while/for loops, no if/elif/else,
    # no comprehensions. These indicate logic, not pure formatting.
    CONTROL_FLOW_NODES = (_ast.While, _ast.For, _ast.If,
                          _ast.ListComp, _ast.DictComp, _ast.SetComp,
                          _ast.GeneratorExp)
    for node in _ast.walk(tree):
        if isinstance(node, CONTROL_FLOW_NODES):
            return False

    # Check 2: No file I/O — no open(), no pd.read_*(), no DataFrame()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Call):
            if isinstance(node.func, _ast.Name) and node.func.id == 'open':
                return False
            if isinstance(node.func, _ast.Attribute):
                if isinstance(node.func.value, _ast.Name) and node.func.value.id == 'pd':
                    if node.func.attr.startswith('read_'):
                        return False
                if node.func.attr == 'DataFrame':
                    return False
                if node.func.attr == 'ExcelFile':
                    return False

    # Check 3: No computation — no BinOp, AND no computational builtins.
    # Only pure formatting/type-conversion builtins are allowed.
    FORMAT_ONLY_BUILTINS = {'print', 'str', 'int', 'float', 'bool',
                            'list', 'dict', 'tuple', 'set',
                            'type', 'isinstance'}
    for node in _ast.walk(tree):
        if isinstance(node, _ast.BinOp) and isinstance(node.op, (_ast.Add, _ast.Sub, _ast.Mult, _ast.Div)):
            return False
        if isinstance(node, _ast.UnaryOp) and isinstance(node.op, _ast.USub):
            return False
        if isinstance(node, _ast.Call):
            if isinstance(node.func, _ast.Name):
                if node.func.id not in FORMAT_ONLY_BUILTINS:
                    return False
            elif isinstance(node.func, _ast.Attribute):
                # Allow json.dumps, json.loads
                if isinstance(node.func.value, _ast.Name) and node.func.value.id == 'json':
                    if node.func.attr not in ('dumps', 'loads'):
                        return False
                else:
                    return False
            else:
                return False  # Complex call expression

    # Check 4: Trace variable sources — all must be static
    var_sources = _trace_variable_sources(tree)
    if any(src == 'dynamic' for src in var_sources.values()):
        return False

    return True


def _statically_extract_answer(code: str) -> Optional[dict]:
    """
    Statically evaluate the answer dict from presentation-only Python code.

    Since _is_presentation_only_call already verified all values trace to
    Constant/Dict/List literals, we can safely reconstruct the dict from the
    AST without executing the code.

    Handles patterns like:
      result = {"answer": "...", "data_source": [...]}
      print(json.dumps(result))

      answer_text = "..."
      data_source = [...]
      result = {"answer": answer_text, "data_source": data_source}
      print(json.dumps(result))

    Returns the answer dict if found, None otherwise.
    """
    import ast as _ast

    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return None

    # ---- Step 1: Collect all variable assignments ----
    # Build var_name → AST node (resolved recursively)
    var_map = {}

    def _eval_ast_node(node) -> any:
        """Recursively evaluate an AST node to a Python value.
        Only handles Constant, List, Dict, and Name (variable ref) nodes."""
        if isinstance(node, _ast.Constant):
            return node.value
        elif isinstance(node, _ast.List):
            return [_eval_ast_node(elt) for elt in node.elts]
        elif isinstance(node, _ast.Dict):
            result = {}
            for k, v in zip(node.keys, node.values):
                key = _eval_ast_node(k) if k else None
                val = _eval_ast_node(v)
                if key is not None:
                    result[key] = val
            return result
        elif isinstance(node, _ast.Name):
            # Variable reference — resolve from var_map
            if node.id in var_map:
                return _eval_ast_node(var_map[node.id])
            return None  # Unresolvable
        elif isinstance(node, _ast.JoinedStr):
            # f-string — try to resolve if all parts are constants
            parts = []
            for part in node.values:
                if isinstance(part, _ast.Constant):
                    parts.append(str(part.value))
                elif isinstance(part, _ast.FormattedValue):
                    val = _eval_ast_node(part.value)
                    if val is not None:
                        parts.append(str(val))
                    else:
                        return None
                else:
                    return None
            return ''.join(parts)
        else:
            return None  # Not statically evaluable

    # Collect assignments in order (later assignments override earlier)
    for stmt in tree.body:
        if isinstance(stmt, _ast.Assign):
            value = stmt.value
            for target in stmt.targets:
                if isinstance(target, _ast.Name):
                    var_map[target.id] = value

    # ---- Step 2: Find the answer dict ----
    # Look for:
    #   a) print(json.dumps(<dict>)) or print(<dict>)
    #   b) Any variable that holds a dict with "answer" key

    # Strategy a: find dict argument to print / json.dumps
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name) and node.func.id == 'print':
            for arg in node.args:
                # print(json.dumps(result))
                if isinstance(arg, _ast.Call):
                    func = arg.func
                    if isinstance(func, _ast.Attribute) and func.attr == 'dumps':
                        if isinstance(func.value, _ast.Name) and func.value.id == 'json':
                            # json.dumps(<var_or_dict>)
                            inner = arg.args[0] if arg.args else None
                            if inner is not None:
                                candidate = _eval_ast_node(inner)
                                if isinstance(candidate, dict) and 'answer' in candidate:
                                    if None in candidate.values():
                                        return None
                                    return candidate
                # print(<dict literal>)
                candidate = _eval_ast_node(arg)
                if isinstance(candidate, dict) and 'answer' in candidate:
                    if None in candidate.values():
                        return None
                    return candidate

    # Strategy b: scan var_map for dicts with "answer" key
    for var_name, node in var_map.items():
        candidate = _eval_ast_node(node)
        if isinstance(candidate, dict) and 'answer' in candidate:
            # Safety: if any value resolved to None, the static evaluation
            # is incomplete — return None to be conservative.
            if None in candidate.values():
                return None
            return candidate

    return None
def remove_presentation_calls(sub_records: list) -> list:
    """
    Remove pure JSON presentation calls (python_code_executor that only formats and prints).

    Uses _is_presentation_only_call() AST check with additional guard rails:
      - The corresponding <ANSWER> must exist in a later final_answer step
      - The answer content must match the final_answer
      - Later steps must not depend on this call's output

    Per-call granularity deletion.

    Args:
        sub_records: List of sub-question records.

    Returns:
        Cleaned sub_records.
    """
    import json as _json

    cleaned_records = []

    for rec in sub_records:
        agent_steps = rec.get('agent_steps', [])

        # First, find the final_answer for this sub-question
        final_answer = None
        for step in agent_steps:
            if step.get('type') == 'final_answer':
                final_answer = step.get('assistant_answer', {})
                break

        cleaned_steps = []
        for si, step in enumerate(agent_steps):
            if step.get('type') != 'tool_call':
                cleaned_steps.append(step)
                continue

            # Check each python_code_executor call
            kept_calls = []
            kept_call_ids = set()
            for tc in step.get('tool_calls', []):
                if not isinstance(tc, dict):
                    kept_calls.append(tc)
                    continue

                tool_name = tc.get('tool_name', '')
                if tool_name != 'python_code_executor':
                    kept_calls.append(tc)
                    if tc.get('tool_call_id'):
                        kept_call_ids.add(tc['tool_call_id'])
                    continue

                args = tc.get('arguments', {})
                code = args.get('code', '') if isinstance(args, dict) else ''

                if not _is_presentation_only_call(code):
                    kept_calls.append(tc)
                    if tc.get('tool_call_id'):
                        kept_call_ids.add(tc['tool_call_id'])
                    continue

                # Additional guard: must have final_answer and content must match
                if final_answer is None:
                    # No final answer → cannot verify, keep the call
                    kept_calls.append(tc)
                    if tc.get('tool_call_id'):
                        kept_call_ids.add(tc['tool_call_id'])
                    continue

                # Try to statically extract the answer dict from the AST.
                # Since _is_presentation_only_call already verified all values
                # are static literals, we can safely reconstruct the dict
                # without executing the code (no exec(), no hang risk).
                code_answer = _statically_extract_answer(code)

                if code_answer is None:
                    kept_calls.append(tc)
                    if tc.get('tool_call_id'):
                        kept_call_ids.add(tc['tool_call_id'])
                    continue

                # Compare with final_answer (lenient: same data sources + numeric overlap)
                if isinstance(code_answer, dict) and 'answer' in code_answer:
                    if _compare_answers_lenient(code_answer, final_answer):
                        # Match confirmed → this is truly a presentation call → delete
                        continue

                # No match → keep
                kept_calls.append(tc)
                if tc.get('tool_call_id'):
                    kept_call_ids.add(tc['tool_call_id'])

            if not kept_calls:
                # All calls were presentation → skip step
                continue

            # Filter observations for kept calls
            kept_obs = []
            for obs in step.get('observations', []):
                oid = obs.get('tool_call_id', '') if isinstance(obs, dict) else ''
                if oid in kept_call_ids:
                    kept_obs.append(obs)

            new_step = dict(step)
            new_step['tool_calls'] = kept_calls
            new_step['observations'] = kept_obs
            cleaned_steps.append(new_step)

        new_rec = dict(rec)
        new_rec['agent_steps'] = cleaned_steps
        cleaned_records.append(new_rec)

    return cleaned_records


# ---- Stage 5: cleanup_orphaned_content ----

def cleanup_orphaned_content(sub_records: list) -> list:
    """
    Clean up orphaned content after stages 2-4 removal.

    Single responsibility:
      Delete empty tool_call steps (no tool_calls AND no observations).
      Only clean up structural orphans — plan text cleanup for BFloat16
      is handled inline in filter_bf16_errors (stage 3).

    Does NOT:
      - Merge adjacent plans
      - Globally rewrite plan text
      - Modify IndexError/NameError recovery plans

    Args:
        sub_records: List of sub-question records.

    Returns:
        Cleaned sub_records.
    """
    cleaned_records = []

    for rec in sub_records:
        agent_steps = rec.get('agent_steps', [])
        cleaned_steps = []

        for step in agent_steps:
            if step.get('type') == 'final_answer':
                cleaned_steps.append(step)
                continue

            if step.get('type') != 'tool_call':
                cleaned_steps.append(step)
                continue

            tool_calls = step.get('tool_calls', [])
            observations = step.get('observations', [])

            # Delete empty tool_call steps
            if not tool_calls and not observations:
                continue

            cleaned_steps.append(step)

        new_rec = dict(rec)
        new_rec['agent_steps'] = cleaned_steps
        cleaned_records.append(new_rec)

    return cleaned_records


# ---- Stage 6: reindex_tool_call_ids ----

def reindex_tool_call_ids(sub_records: list, dialog_idx: int = 0) -> list:
    """
    Reassign globally unique call_ids across the entire dialog.

    Format: call_{dialog_idx}_{sq_idx}_{seq:03d}

    This ensures:
      - No duplicate call_ids across sub-questions in the same dialog
      - Continuous numbering after stages 2-4 removal

    Args:
        sub_records: List of sub-question records for one dialog.
        dialog_idx: Index of this dialog (for uniqueness across dialogs).

    Returns:
        Sub_records with reindexed call_ids.
    """
    cleaned_records = []
    global_seq = 0

    for rec_idx, rec in enumerate(sub_records):
        agent_steps = rec.get('agent_steps', [])
        cleaned_steps = []
        # Mapping old_call_id → new_call_id for this sub-question
        id_map = {}

        for step in agent_steps:
            if step.get('type') != 'tool_call':
                cleaned_steps.append(step)
                continue

            # Reindex tool_calls
            new_calls = []
            for tc in step.get('tool_calls', []):
                if not isinstance(tc, dict):
                    new_calls.append(tc)
                    continue

                old_id = tc.get('tool_call_id', '')
                new_id = f"call_{dialog_idx}_{rec_idx}_{global_seq:03d}"
                global_seq += 1

                new_tc = dict(tc)
                new_tc['tool_call_id'] = new_id
                if old_id:
                    id_map[old_id] = new_id
                new_calls.append(new_tc)

            # Reindex observations
            new_obs = []
            for obs in step.get('observations', []):
                if not isinstance(obs, dict):
                    new_obs.append(obs)
                    continue
                new_o = dict(obs)
                old_oid = obs.get('tool_call_id', '')
                if old_oid in id_map:
                    new_o['tool_call_id'] = id_map[old_oid]
                new_obs.append(new_o)

            new_step = dict(step)
            new_step['tool_calls'] = new_calls
            new_step['observations'] = new_obs
            cleaned_steps.append(new_step)

        new_rec = dict(rec)
        new_rec['agent_steps'] = cleaned_steps
        cleaned_records.append(new_rec)

    return cleaned_records


# ---- Stage 7+8: revalidate_trajectory + evidence verifier ----

# Clause boundaries: direction matching must never cross these.
_CLAUSE_BOUNDARY = re.compile(r'[，,。；;！？!?\n]')

# Number-with-unit pattern: captures standard %, full-width ％, and 个百分点.
_NUMBER_WITH_UNIT = re.compile(
    r'(?P<sign>-?)'
    r'(?P<number>\d+(?:\.\d+)?)\s*'
    r'(?P<unit>%|％|个百分点|个百分比点)'
)

# Priority-ordered patterns for semantic sign detection.
# Each is tested against the prefix (text from start of current clause up to
# the number) and must match at the END of the prefix, meaning the direction
# word directly modifies this number.

# Priority 1: Target level — "fell TO X%" → positive (read as is)
_TARGET_LEVEL_RE = re.compile(
    r'(?:下降|降低|减少|下滑|回落|下跌|增长|增加|提高|上升|上涨|提升)'
    r'(?:至|到|为)'
    r'(?:了|约|大约|超过|接近)*\s*$'
)

# Priority 2: Magnitude — "decline magnitude of X%" → positive
_MAGNITUDE_RE = re.compile(
    r'(?:降幅|跌幅|下降幅度|减少幅度|降低幅度|下滑幅度|回落幅度'
    r'|增幅|涨幅|增长幅度|增加幅度|提高幅度|上升幅度|上涨幅度|提升幅度)'
    r'(?:为|是|达到|约为|达|约|大约)*\s*$'
)

# Priority 3: Negative change — "down X%" → negate
_NEGATIVE_CHANGE_RE = re.compile(
    r'(?:同比|环比)?'
    r'(?:下降|减少|降低|下滑|回落|下跌|负增长|缩减|萎缩)'
    r'(?:了|约|大约|超过|接近)*\s*$'
)

# Priority 4: Positive change — "up X%" → keep positive
_POSITIVE_CHANGE_RE = re.compile(
    r'(?:同比|环比)?'
    r'(?:增长|增加|提高|上升|上涨|提升)'
    r'(?:了|约|大约|超过|接近)*\s*$'
)


def _match_end(pattern, text: str) -> bool:
    """Return True if *pattern* matches at the end of *text*."""
    m = pattern.search(text)
    return m is not None and m.end() == len(text)


def extract_semantic_numeric_claims(text: str) -> list:
    """Extract numeric claims with semantic sign from natural-language text.

    Operates clause-by-clause so that a direction word in one clause cannot
    affect a percentage in another (e.g. "库存下降，但产量增长5.79%" keeps
    5.79 positive because it is in a different clause).

    Uses the prefix immediately before each number (same clause only) and
    applies priority-ordered rules:
      1. Explicit '-' sign → trust it (already signed)
      2. "至/to" target-level pattern → read as-is (fell TO 20.8%)
      3. "幅度/magnitude" pattern → read as-is (decline magnitude of 82.1%)
      4. Negative-change pattern (下降/减少/...) → negate
      5. Positive-change pattern (增长/增加/...) → keep positive
      6. Ambiguous → keep raw sign, don't infer

    Returns a list of dicts:
        {"value": float, "unit": "percent"|"percentage_point",
         "kind": "change"|"level"|"magnitude"|"unknown",
         "source": "text_fragment"}

    Apply to BOTH answer text and observation text so that semantic signs
    are consistently resolved on both sides of evidence verification.
    """
    claims = []

    for clause in _CLAUSE_BOUNDARY.split(text):
        if not clause.strip():
            continue

        for m in _NUMBER_WITH_UNIT.finditer(clause):
            num_str = m.group('number')
            explicit_sign = m.group('sign')
            unit = 'percentage_point' if '百分点' in m.group('unit') else 'percent'
            raw_value = float(num_str)
            prefix = clause[:m.start()]  # text before number, same clause only

            # Priority-ordered semantic sign resolution
            if explicit_sign == '-':
                semantic_value = -raw_value
                kind = 'change'
            elif _match_end(_TARGET_LEVEL_RE, prefix):
                # "回落至20.8%" → target level, read as positive
                semantic_value = raw_value
                kind = 'level'
            elif _match_end(_MAGNITUDE_RE, prefix):
                # "降幅为82.1%" → magnitude, read as positive
                semantic_value = raw_value
                kind = 'magnitude'
            elif _match_end(_NEGATIVE_CHANGE_RE, prefix):
                # "下降5.79%" → negative change
                semantic_value = -raw_value
                kind = 'change'
            elif _match_end(_POSITIVE_CHANGE_RE, prefix):
                # "增长5.79%" → positive change
                semantic_value = raw_value
                kind = 'change'
            else:
                # Ambiguous — keep raw sign, don't infer
                semantic_value = raw_value
                kind = 'unknown'

            claims.append({
                'value': semantic_value,
                'unit': unit,
                'kind': kind,
                'source': m.group(0),
            })

    return claims


def _extract_numeric_values(text: str) -> tuple:
    """Extract numeric values from text. Returns (absolute_values, percentages).

    Percentages are tracked separately because they must always be verified
    regardless of magnitude (e.g. 6.8% is a claim about a ratio, not a small
    derived delta).

    Thousand separators (commas) are stripped before parsing so that
    "264,500" is parsed as 264500.0, not as 264 and 500.
    """
    import re

    # Pre-process: strip thousand separators from digit groups.
    # Only strips commas between digits where the right side has 3 digits
    # followed by a non-digit or end-of-string (e.g. 264,500 → 264500).
    # Does NOT strip commas in lists like "1,2,3".
    def _strip_commas(s: str) -> str:
        return re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', s)

    text = _strip_commas(text)

    values = set()
    percentages = set()

    # Match percentages (e.g. 14.36%, -36.03%, 6.8%)
    for m in re.finditer(r'(-?\d+\.?\d*)\s*%', text):
        pct = float(m.group(1))
        percentages.add(pct)

    # Match percentage points (e.g. 下降5.79个百分点, 增长3.2个百分点)
    for m in re.finditer(r'(-?\d+\.?\d*)\s*个百分点', text):
        pp = float(m.group(1))
        percentages.add(pp)

    # Match decimal numbers (but not dates like 2015). Preserve sign.
    for m in re.finditer(r'(?<!\d)(-?\d+\.\d+)(?!\d)', text):
        val = float(m.group(1))
        if abs(val) > 3000:  # Likely a year → skip
            continue
        values.add(val)

    # Match integers (2+ digits, not years 1900-2099, not after decimal point).
    # Preserve sign.
    for m in re.finditer(r'(?<![\d\.])(-?\d{2,})(?![\d\.])', text):
        val = int(m.group(1))
        if 1900 <= abs(val) <= 2099:  # Likely a year → skip
            continue
        values.add(float(val))

    return values, percentages


def _verify_evidence(answer_text: str, tagged_observations: list) -> tuple:
    """
    Verify that key numeric claims in the answer can be traced to observations.

    Uses extract_semantic_numeric_claims on BOTH answer and observation text
    so that Chinese directional words are consistently resolved on both sides.
    Observations also contribute raw numeric values (from _extract_numeric_values)
    to handle non-percentage numbers like "产量5000台".

    Returns:
        (pass: bool, missing_values: list)
    """
    # Extract semantic claims from answer (percentages + 个百分点)
    answer_claims = extract_semantic_numeric_claims(answer_text)

    # Also extract raw numeric values from answer for non-percentage claims
    answer_vals, answer_pcts = _extract_numeric_values(answer_text)
    significant_answer_vals = {v for v in answer_vals if abs(v) >= 10}

    if not answer_claims and not significant_answer_vals:
        return True, []  # No numeric claims to verify

    # Collect all observation values: semantic claims + raw numbers
    obs_values = set()

    for tool_name, obs in tagged_observations:
        content = obs.get('content', '') if isinstance(obs, dict) else str(obs)

        # Semantic claims from observation (handles 同比下降-5.79%, etc.)
        for claim in extract_semantic_numeric_claims(content):
            obs_values.add(claim['value'])

        # Raw numeric values for non-percentage numbers only.
        # Strip %/百分点 patterns first so that "同比下降5.79%" doesn't
        # contribute unsigned 5.79 alongside the semantic claim of -5.79.
        content_without_pct = re.sub(r'-?\d+\.?\d*\s*%', '', content)
        content_without_pct = re.sub(
            r'-?\d+\.?\d*\s*个百分点', '', content_without_pct)
        vals, _pcts = _extract_numeric_values(content_without_pct)
        obs_values.update(vals)

    def _fuzzy_match(value: float, candidates: set, tolerance: float = 0.01) -> bool:
        """Check if value approximately matches any candidate."""
        for c in candidates:
            if abs(value - c) / max(abs(value), abs(c), 1) < tolerance:
                return True
        return False

    missing = []

    # ---- Verify semantic claims (percentages / 个百分点) ----
    for claim in answer_claims:
        val = claim['value']
        if val in obs_values:
            continue
        if _fuzzy_match(val, obs_values):
            continue
        missing.append(f"{claim['source']} (semantic={val})")

    # ---- Verify significant raw values ----
    for m in significant_answer_vals:
        if m in obs_values:
            continue
        if _fuzzy_match(m, obs_values):
            continue
        missing.append(str(m))

    return len(missing) == 0, missing


# ---- LLM-based evidence verification (fallback for derived values) ----

_VERIFY_EVIDENCE_LLM_PROMPT = """你是一个严格的数据溯源审核专家。检查模型回答中的数值声明是否能从工具返回结果中找到来源。

## 模型回答
{answer_text}

## 工具返回结果（Observation）
{observations_text}

## 确定性验证未通过的数值声明
{missing_claims}

## 任务
对上述未通过确定性验证的数值声明，判断它们是否能从工具返回结果中找到支撑：

1. **直接匹配**：数值在某个 Observation 中直接出现（允许微小舍入误差 ≤ 0.1%）
2. **可推导**：数值可以通过 Observation 中的数据进行简单计算得出，例如：
   - 求和：Obs 中有 A=200000 和 B=64500，答案说 "总共 264500" → 200000+64500=264500 ✓
   - 求差：Obs 中有 103.5 和 99.5，答案说 "相差 4 个百分点" → 103.5-99.5=4.0 ✓
   - 比值：Obs 中有部分和整体，答案说 "占比 75.6%" → 可计算验证 ✓
   - 乘积：Obs 中有单价和数量，答案说 "总价 XXX" → 可计算验证 ✓
3. **无法溯源**：数值在 Observation 中完全找不到，也无法从 Observation 中的数值通过简单计算得出

注意：
- "上年=100"、"以100为基期" 这类统计口径说明中的数字不是需要验证的数据声明
- 答案中的年份（如 2015、2019）不需要在 Observation 中验证
- 如果答案是 "从2015年到2019年"，年份只是时间范围，不需要数值验证

输出 JSON：
{{
  "pass": true/false,
  "supported_claims": ["数值声明及其来源或推导过程"],
  "unsupported_claims": ["无法溯源的数值声明及原因"]
}}

如果所有数值声明都能找到来源或可推导，pass 为 true。任一数值无法溯源，pass 为 false。"""


def verify_evidence_with_fallback(answer_text: str, tagged_observations: list,
                                   client) -> tuple:
    """Verify that numeric claims in the answer are traceable to observations.

    Two-tier strategy:
      1. Deterministic: extract numeric claims from answer and observations,
         check for exact/fuzzy matches.
      2. LLM fallback: if deterministic verification fails, ask an LLM to
         judge whether the missing values can be derived (sum, difference,
         ratio, product, etc.) from the observation data.

    Args:
        answer_text: The answer string to verify.
        tagged_observations: List of (tool_name, obs_dict) tuples.
        client: LLM client for fallback verification.

    Returns:
        (pass: bool, missing: list) — pass is True if all numeric claims
        are supported, False otherwise. missing lists unsupported claims.
    """
    # Tier 1: Deterministic check
    det_pass, det_missing = _verify_evidence(answer_text, tagged_observations)
    if det_pass:
        return True, []

    # Tier 2: LLM fallback for derived values
    import json as _json

    # Build a compact observation summary for the LLM
    obs_lines = []
    for tool_name, obs in tagged_observations:
        content = obs.get('content', '') if isinstance(obs, dict) else str(obs)
        if content.strip():
            # Truncate very long observations to keep prompt manageable
            truncated = content[:2000] + ('...' if len(content) > 2000 else '')
            obs_lines.append(f"[{tool_name}] {truncated}")
    observations_text = '\n---\n'.join(obs_lines) if obs_lines else '(无 Observation)'

    prompt = _VERIFY_EVIDENCE_LLM_PROMPT.format(
        answer_text=answer_text,
        observations_text=observations_text,
        missing_claims=_json.dumps(det_missing, ensure_ascii=False),
    )

    try:
        response = client.chat(
            prompt=prompt,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        result = extract_json_from_response(response)
    except Exception:
        # LLM call failed — fall back to deterministic result
        return False, det_missing

    if result.get('pass') is True:
        return True, []

    # LLM confirmed some claims are unsupported
    unsupported = result.get('unsupported_claims', [])
    return False, unsupported if unsupported else det_missing


def revalidate_trajectory(sub_records: list, original_sub_records: list = None,
                           include_tools: list = None,
                           evidence_client=None,
                           use_llm_evidence: bool = False) -> tuple:
    """
    13-point final validation after stages 1-6 cleaning.

    Checks:
      1. Each tool_call step has >= 1 tool_call and >= 1 observation
      2. Each tool_call_id has exactly 1 observation (1:1 bidirectional)
      3. tool_call and observation counts match across dialog
      4. No _dedup_conflicts
      5. No empty agent_steps
      6. No residual pure JSON presentation calls (double-check stage 4)
      7. Answers unchanged from original
      8. IndexError/NameError all have successful recovery
      9. Answer values traceable to observations (evidence verifier)
      10. Recovery errors use new call_ids (guaranteed by stage 6)
      11. IndexError/NameError followed by [SUCCESS] observation → final_answer
      12. All tool names/params pass schema validation
      13. All tool_call_ids are globally unique

    Args:
        sub_records: Cleaned sub-records.
        original_sub_records: Original sub-records (pre-cleaning) for answer comparison.
        include_tools: Optional list of enabled tool names for schema validation.

    Returns:
        (pass: bool, issues: list)
    """
    import json as _json

    issues = []

    # Collect original answers per sub-question
    original_answers = {}
    if original_sub_records:
        for rec in original_sub_records:
            sq_id = rec.get('subquestion_id', 0)
            for step in rec.get('agent_steps', []):
                if step.get('type') == 'final_answer':
                    original_answers[sq_id] = step.get('assistant_answer', {})
                    break

    # Global call_id tracking
    all_call_ids = set()
    all_obs_call_ids = set()

    for rec_idx, rec in enumerate(sub_records):
        sq_id = rec.get('subquestion_id', rec_idx)
        agent_steps = rec.get('agent_steps', [])

        # Check 5: No empty agent_steps
        if not agent_steps:
            issues.append(f"sq={sq_id}: agent_steps is empty")
            continue

        has_final = any(s.get('type') == 'final_answer' for s in agent_steps)
        if not has_final:
            issues.append(f"sq={sq_id}: no final_answer step")

        # Collect observations for evidence verifier (tagged by tool name).
        # Each entry is (tool_name, obs_dict) so the verifier can distinguish
        # python_code_executor outputs (auto-trusted) from source-data outputs.
        sq_tagged_observations = []

        for si, step in enumerate(agent_steps):
            if step.get('type') == 'final_answer':
                # Check 7: Answer unchanged
                if sq_id in original_answers:
                    orig_ans = original_answers[sq_id]
                    new_ans = step.get('assistant_answer', {})
                    if not _compare_answers_strict(orig_ans, new_ans):
                        issues.append(
                            f"sq={sq_id}: final_answer changed during cleaning. "
                            f"Original: {str(orig_ans)[:100]}, New: {str(new_ans)[:100]}"
                        )
                continue

            if step.get('type') != 'tool_call':
                issues.append(f"sq={sq_id} step[{si}]: unknown type '{step.get('type')}'")
                continue

            tool_calls = step.get('tool_calls', [])
            observations = step.get('observations', [])

            # Check 1: tool_call step has >=1 call and >=1 observation
            if not tool_calls:
                issues.append(f"sq={sq_id} step[{si}]: tool_call step has no tool_calls")
            if not observations:
                issues.append(f"sq={sq_id} step[{si}]: tool_call step has no observations")

            # Build tool_call_id → tool_name mapping for this step
            cid_to_tool = {}
            for tc in tool_calls:
                if isinstance(tc, dict):
                    cid = tc.get('tool_call_id', '')
                    if cid:
                        cid_to_tool[cid] = tc.get('tool_name', 'unknown')

            # Tag each observation with its producing tool name
            for obs in observations:
                oid = obs.get('tool_call_id', '') if isinstance(obs, dict) else ''
                tool_name = cid_to_tool.get(oid, 'unknown')
                sq_tagged_observations.append((tool_name, obs))

            # Collect call_ids
            step_call_ids = set()
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                cid = tc.get('tool_call_id', '')
                if cid:
                    if cid in all_call_ids:
                        issues.append(
                            f"sq={sq_id} step[{si}]: duplicate tool_call_id '{cid}' (Check 13)"
                        )
                    all_call_ids.add(cid)
                    step_call_ids.add(cid)

                # Check 6: No residual pure presentation calls
                tool_name = tc.get('tool_name', '')
                if tool_name == 'python_code_executor':
                    args = tc.get('arguments', {})
                    code = args.get('code', '') if isinstance(args, dict) else ''
                    if _is_presentation_only_call(code):
                        issues.append(
                            f"sq={sq_id} step[{si}]: residual presentation-only "
                            f"python_code_executor call (Check 6)"
                        )

            # Collect observation call_ids
            step_obs_ids = set()
            for obs in observations:
                if not isinstance(obs, dict):
                    continue
                oid = obs.get('tool_call_id', '')
                if oid:
                    if oid in all_obs_call_ids:
                        issues.append(
                            f"sq={sq_id} step[{si}]: duplicate observation call_id "
                            f"'{oid}' — each tool_call_id must have exactly 1 observation"
                        )
                    all_obs_call_ids.add(oid)
                    step_obs_ids.add(oid)
                else:
                    issues.append(
                        f"sq={sq_id} step[{si}]: observation missing tool_call_id"
                    )

                # Check 8/11: IndexError/NameError recovery
                content = obs.get('content', '')
                if 'IndexError' in content or 'NameError' in content:
                    # Must be followed by a successful call to the SAME tool
                    # (corrected retry), not just any successful tool call.
                    # Reading B.xlsx after IndexError on A.xlsx is NOT recovery.
                    failed_tool = obs.get('tool_name', '')
                    recovered = False
                    for later_si in range(si + 1, len(agent_steps)):
                        later_step = agent_steps[later_si]
                        if later_step.get('type') != 'tool_call':
                            continue
                        for later_obs in later_step.get('observations', []):
                            if isinstance(later_obs, dict) and later_obs.get('success') is True:
                                if failed_tool and later_obs.get('tool_name', '') == failed_tool:
                                    recovered = True
                                    break
                            elif isinstance(later_obs, dict):
                                # Dict with success != True but may have legacy marker
                                lc = later_obs.get('content', '')
                                if ('[SUCCESS]' in lc or '[Success]' in lc):
                                    if failed_tool and later_obs.get('tool_name', '') == failed_tool:
                                        recovered = True
                                        break
                            else:
                                # Non-dict legacy string observation — accept as
                                # recovery since we cannot determine tool name
                                lc = str(later_obs)
                                if '[SUCCESS]' in lc or '[Success]' in lc:
                                    recovered = True
                                    break
                        if recovered:
                            break
                    if not recovered:
                        issues.append(
                            f"sq={sq_id} step[{si}]: {content[:50]}... "
                            f"not followed by same-tool successful recovery (Check 8/11)"
                        )

            # Check 2: 1:1 call_id ↔ observation correspondence within step
            if step_call_ids != step_obs_ids:
                missing = step_call_ids - step_obs_ids
                extra = step_obs_ids - step_call_ids
                if missing:
                    issues.append(
                        f"sq={sq_id} step[{si}]: call_ids without observation: {missing}"
                    )
                if extra:
                    issues.append(
                        f"sq={sq_id} step[{si}]: observations without call: {extra}"
                    )

        # Check 9: Evidence verifier (tiered by tool type)
        final_answer = None
        for step in agent_steps:
            if step.get('type') == 'final_answer':
                final_answer = step.get('assistant_answer', {})
                break
        if final_answer and sq_tagged_observations:
            answer_text = final_answer.get('answer', '') if isinstance(final_answer, dict) else str(final_answer)
            if use_llm_evidence and evidence_client is not None:
                evidence_ok, missing_nums = verify_evidence_with_fallback(
                    answer_text, sq_tagged_observations, evidence_client)
            else:
                evidence_ok, missing_nums = _verify_evidence(
                    answer_text, sq_tagged_observations)
            if not evidence_ok:
                issues.append(
                    f"sq={sq_id}: evidence verification failed. "
                    f"Values not in observations: {missing_nums[:5]} (Check 9)"
                )

    # Check 3: Global tool_call count matches observation count
    if len(all_call_ids) != len(all_obs_call_ids):
        issues.append(
            f"Global: tool_call_ids ({len(all_call_ids)}) != observation ids ({len(all_obs_call_ids)}) (Check 3)"
        )

    # Check 4: No _dedup_conflicts (guaranteed by stage 1 gate)
    # Check 10: Recovery uses new call_ids (guaranteed by stage 6)
    # Check 12: Schema validation — caller responsibility (validate_tool_calls)
    # Check 13: Global uniqueness already checked during scan

    return len(issues) == 0, issues


# ---- Convenience: Run full cleaning pipeline on one dialog ----

def run_cleaning_pipeline(sub_records: list, dialog_idx: int = 0,
                           include_tools: list = None,
                           verbose: bool = False,
                           **kwargs) -> tuple:
    """
    Run the full 9-stage cleaning pipeline on one dialog's sub_records.

    Args:
        sub_records: List of sub-question records for one dialog.
        dialog_idx: Dialog index for call_id reindexing.
        include_tools: Optional list of enabled tool names.
        verbose: Print per-stage details.

    Returns:
        (success: bool, cleaned_sub_records: list, report: dict)
        report contains per-stage results and any issues found.
    """
    report = {
        'dialog_idx': dialog_idx,
        'stage1_conflicts': None,
        'stage2_dedup': False,
        'stage3_bf16_removed': 0,
        'stage4_presentation_removed': 0,
        'stage5_cleanup': False,
        'stage6_reindex': False,
        'stage7_validation_pass': False,
        'stage7_issues': [],
        'recovery_tags': [],
    }

    original = [dict(r) for r in sub_records]  # Deep copy for answer comparison

    # Stage 1: Detect call_id conflicts
    has_conflict, conflicts = detect_call_id_conflicts(sub_records)
    if has_conflict:
        report['stage1_conflicts'] = conflicts
        if verbose:
            print(f"  [STAGE1] CONFLICT: {len(conflicts)} call_ids have conflicting content")
        return False, sub_records, report

    if verbose:
        print(f"  [STAGE1] No call_id conflicts")

    # Stage 2: Deduplicate log calls (per sub-question)
    deduped = []
    for rec in sub_records:
        new_rec = dict(rec)
        new_rec['agent_steps'] = deduplicate_log_calls(rec.get('agent_steps', []))
        deduped.append(new_rec)
    sub_records = deduped
    report['stage2_dedup'] = True
    if verbose:
        print(f"  [STAGE2] Dedup complete")

    # Stage 3: Filter BFloat16 errors
    sub_records, bf16_removed = filter_bf16_errors(sub_records, max_bf16_keep=0)
    total_bf16 = sum(
        len(v['partial_removed']) + v['fully_removed_count']
        for v in bf16_removed.values()
    )
    report['stage3_bf16_removed'] = total_bf16
    report['_bf16_removed_indices'] = bf16_removed
    if verbose:
        print(f"  [STAGE3] BFloat16: {total_bf16} steps affected "
              f"(partial={sum(len(v['partial_removed']) for v in bf16_removed.values())}, "
              f"fully_removed={sum(v['fully_removed_count'] for v in bf16_removed.values())})")

    # Stage 4: Remove presentation-only calls
    sub_records = remove_presentation_calls(sub_records)
    report['stage4_presentation_removed'] = True
    if verbose:
        print(f"  [STAGE4] Presentation call removal complete")

    # Stage 5: Cleanup orphaned content
    sub_records = cleanup_orphaned_content(sub_records)
    report['stage5_cleanup'] = True
    if verbose:
        print(f"  [STAGE5] Orphaned content cleanup complete")

    # Stage 6: Reindex tool_call_ids
    sub_records = reindex_tool_call_ids(sub_records, dialog_idx)
    report['stage6_reindex'] = True
    if verbose:
        print(f"  [STAGE6] Tool call IDs reindexed")

    # Collect recovery tags (IndexError/NameError)
    for rec in sub_records:
        for step in rec.get('agent_steps', []):
            for obs in step.get('observations', []):
                content = obs.get('content', '') if isinstance(obs, dict) else str(obs)
                if 'IndexError' in content:
                    report['recovery_tags'].append({
                        'sample_id': rec.get('sample_id', ''),
                        'subquestion_id': rec.get('subquestion_id', ''),
                        'trajectory_type': 'recovery',
                        'error_type': 'IndexError'
                    })
                    break
                if 'NameError' in content:
                    report['recovery_tags'].append({
                        'sample_id': rec.get('sample_id', ''),
                        'subquestion_id': rec.get('subquestion_id', ''),
                        'trajectory_type': 'recovery',
                        'error_type': 'NameError'
                    })
                    break

    # Stage 7+8: Revalidate trajectory
    # evidence_client/use_llm_evidence are forwarded when available
    validation_pass, issues = revalidate_trajectory(
        sub_records, original, include_tools,
        evidence_client=kwargs.get('evidence_client'),
        use_llm_evidence=kwargs.get('use_llm_evidence', False),
    )
    report['stage7_validation_pass'] = validation_pass
    report['stage7_issues'] = issues
    if verbose:
        if validation_pass:
            print(f"  [STAGE7] Validation PASSED")
        else:
            print(f"  [STAGE7] Validation FAILED: {len(issues)} issues")
            for issue in issues[:5]:
                print(f"    - {issue}")

    # Stage 9: build_chat_format is called separately in step8

    return validation_pass, sub_records, report
