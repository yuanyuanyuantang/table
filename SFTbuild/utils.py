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
            return {
                'answer': parsed.get('answer', answers[0].strip()),
                'data_source': parsed.get('data_source', [])
            }
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


def extract_step_plan(reasoning_content: str) -> str:
    """从 reasoning_content 提取动作规划，保留完整的显式推理链。"""
    if not reasoning_content:
        return ''
    return reasoning_content.strip()


def parse_agent_steps(span_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    从子问题 span 中解析 agent_steps。
    span_messages[0] 是 user question，从 span_messages[1:] 开始解析。
    """
    steps = []
    if not span_messages:
        return steps

    # 跳过第一条 user message
    i = 1
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
                        'success': not ('[ERROR]' in content or 'Error' in content.split('\n')[0])
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
    """统计子问题内部工具错误和恢复情况"""
    error_count = 0
    unrecovered_count = 0
    tool_details = []

    for step in agent_steps:
        if step['type'] != 'tool_call':
            continue
        for obs in step.get('observations', []):
            if not isinstance(obs, dict):
                # 兼容字符串 observation（如 repair 生成的旧格式）
                error_count += 1
                unrecovered_count += 1
                tool_details.append({
                    'tool_name': 'unknown',
                    'success': False,
                    'error': str(obs)[:200]
                })
                continue
            if not obs.get('success', True):
                error_count += 1
                detail = {
                    'tool_name': obs.get('tool_name', 'unknown'),
                    'success': False,
                    'error': obs.get('content', '')[:200]
                }
                tool_details.append(detail)

    # Check if final answer was reached — if yes, errors were recovered
    has_final_answer = any(s['type'] == 'final_answer' for s in agent_steps)
    if not has_final_answer:
        unrecovered_count = error_count  # All errors unrecovered if no answer

    return {
        'tool_call_count': sum(1 for s in agent_steps if s['type'] == 'tool_call'),
        'tool_error_count': error_count,
        'unrecovered_error_count': unrecovered_count,
        'tools': tool_details
    }
