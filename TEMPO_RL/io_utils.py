"""
TEMPO-RL — Shared I/O utilities.

Centralised read / write helpers for JSONL/JSON files, benchmark sample
loading, response parsing, and system templates.  All file paths are explicit
parameters — no hidden globals.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------

def write_jsonl(filepath: str, records: List[Dict[str, Any]]) -> None:
    """Write a list of dicts to a JSONL file.

    Creates parent directories if they don't exist.
    """
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts.

    Returns an empty list if the file does not exist.
    """
    if not os.path.exists(filepath):
        return []
    records: List[Dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Benchmark samples
# ---------------------------------------------------------------------------

def load_benchmark_samples(filepath: str) -> List[Dict[str, Any]]:
    """Load benchmark samples from a JSON file.

    Supports both a top-level JSON array (``[...]``) and a single object
    (wrapped into a one-element list).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(
        f"Expected JSON array or object in {filepath}, got {type(data).__name__}"
    )


def load_json_file(filepath: str) -> Any:
    """Load arbitrary JSON from a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def ensure_abs(path: str, relative_to: str) -> str:
    """Return *path* as absolute, resolved against *relative_to* if relative."""
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(relative_to, path))


# ---------------------------------------------------------------------------
# Response parsing (shared across Phase 1/2/3)
# ---------------------------------------------------------------------------

_TOOL_CALL_PATTERN = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
_ANSWER_PATTERN = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
_MEMORY_PATTERN = re.compile(r'<memory>(.*?)</memory>', re.DOTALL)
_THINK_PATTERN = re.compile(r'<think>(.*?)</think>', re.DOTALL)
# Fallback patterns for malformed XML parameter formats
_XML_PARAM_PATTERN = re.compile(r'<parameter\s+name\s*=\s*"([^"]*)"\s*>(.*?)</parameter\s*>', re.DOTALL)
# Malformed: <parameter=command>val</parameter> (missing "name=")
_XML_MALFORMED_PARAM = re.compile(r'<parameter\s*=\s*(\S+?)\s*>(.*?)</parameter\s*>', re.DOTALL)
# Malformed: <parameter cwd>val</parameter> (missing "name=", just a bare attribute)
_XML_BARE_PARAM = re.compile(r'<parameter\s+(\w+)\s*>(.*?)</parameter\s*>', re.DOTALL)
# Malformed: <tool=name> or <tool=name</tool> (missing closing >)
_XML_MALFORMED_TOOL = re.compile(r'<tool\s*=\s*(\S+?)\s*>', re.DOTALL)
# Alternative: <tool name="X"> and <param name="X">val</param>
_XML_TOOL_TAG = re.compile(r'<tool\s+name\s*=\s*"([^"]*)"\s*>', re.DOTALL)
_XML_PARAM_TAG = re.compile(r'<param\s+name\s*=\s*"([^"]*)"\s*>(.*?)</param\s*>', re.DOTALL)
_XML_FUNC_PATTERN = re.compile(r'<(\w+)\s*>', re.DOTALL)
# Unclosed tool_call: <tool_call>JSON... (truncated model output)
_UNCLOSED_TOOL_CALL_JSON = re.compile(r'<tool_call>\s*(\{.*\})\s*$', re.DOTALL)
_UNCLOSED_TOOL_CALL_XML = re.compile(r'<tool_call>(.*)', re.DOTALL)


def _parse_tool_call_content(raw: str) -> Optional[Dict[str, Any]]:
    """Parse the content between (or after) ``<tool_call>`` tags.

    Returns ``{"tool_name": str, "arguments": dict}`` or ``None``.
    Handles JSON, hybrid XML+JSON, and several malformed XML formats.
    """
    raw = raw.strip()
    if not raw:
        return None

    # --- Format 1: Pure JSON ---
    obj: Optional[Dict[str, Any]] = None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        try:
            obj, _ = json.JSONDecoder().raw_decode(raw)
        except json.JSONDecodeError:
            pass

    if obj is not None:
        tool_name = obj.get("tool") or obj.get("name") or ""
        tool_args = obj.get("params") or obj.get("arguments") or {}
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except (json.JSONDecodeError, TypeError):
                tool_args = {}
        if tool_name:
            return {"tool_name": tool_name, "arguments": tool_args}
        return None

    # --- Format 2: XML-like formats ---
    tool_name = ""
    tool_args: Dict[str, Any] = {}

    # Pattern B: <function tool_name>JSON</function> — hybrid XML+JSON
    json_inner = re.search(r'<function\s+(\w+)\s*>(.*?)</function\s*>', raw, re.DOTALL)
    if json_inner:
        tool_name = json_inner.group(1).strip()
        jt = json_inner.group(2).strip()
        try:
            tool_args = json.loads(jt)
        except json.JSONDecodeError:
            try:
                tool_args, _ = json.JSONDecoder().raw_decode(jt)
            except json.JSONDecodeError:
                tool_args = {}
        if tool_name:
            return {"tool_name": tool_name, "arguments": tool_args}

    # Pattern: <tool_name>JSON</tool_name> (generic tag+JSON)
    tag_json = re.search(r'<(\w+)\s*>(.*?)</\1\s*>', raw, re.DOTALL)
    if tag_json:
        tn = tag_json.group(1).strip()
        jt = tag_json.group(2).strip()
        if tn.lower() not in ("parameter", "function", "function_calls", "invoke", "tool_call"):
            try:
                tool_args = json.loads(jt)
                return {"tool_name": tn, "arguments": tool_args}
            except json.JSONDecodeError:
                # Not JSON — fall through to parameter parsing
                tool_name = tn

    # Handler: <function>tool_name>... or <function>tool_name</mismatch>...
    # Extract tool name from text immediately after <function> tag
    func_content_match = re.search(r'<function\s*>(.*)', raw, re.DOTALL)
    if func_content_match and not tool_name:
        inner = func_content_match.group(1).strip()
        # Tool name is the first word/token before '>', '<', '/', end of line, or space
        tn_match = re.match(r'(\w+)', inner)
        if tn_match:
            candidate = tn_match.group(1)
            if candidate.lower() not in ("parameter", "function", "function_calls"):
                tool_name = candidate

    # Pattern A: <parameter name="x">val</parameter> pairs (standard)
    params_match = _XML_PARAM_PATTERN.findall(raw)
    if not params_match:
        # Alternative: <param name="x">val</param> (used with <tool>)
        params_match = _XML_PARAM_TAG.findall(raw)
    if not params_match:
        # Malformed: <parameter=command>val</parameter> (missing "name=")
        params_match = _XML_MALFORMED_PARAM.findall(raw)
    if not params_match:
        # Malformed: <parameter cwd>val</parameter> (bare attribute, no =)
        params_match = _XML_BARE_PARAM.findall(raw)
    if params_match:
        for pname, pval in params_match:
            tool_args[pname.strip()] = pval.strip()

    # Malformed: <tool=name> or <tool=name</tool>
    if not tool_name:
        tool_match = _XML_MALFORMED_TOOL.search(raw)
        if tool_match:
            tool_name = tool_match.group(1).strip()

    # Alternative: <tool name="X">
    if not tool_name:
        tool_tag = _XML_TOOL_TAG.search(raw)
        if tool_tag:
            tool_name = tool_tag.group(1).strip()

    # Extract tool name from tags if not found yet
    if not tool_name:
        invoke_match = re.search(r'<invoke\s+name\s*=\s*"([^"]*)"', raw)
        func_name_match = re.search(r'<function\s+name\s*=\s*"([^"]*)"', raw)
        name_tag_match = re.search(r'<name>(.*?)</name>', raw, re.DOTALL)
        if invoke_match:
            tool_name = invoke_match.group(1).strip()
        elif func_name_match:
            tool_name = func_name_match.group(1).strip()
        elif name_tag_match:
            tool_name = name_tag_match.group(1).strip()
        else:
            # Find first meaningful tag
            func_match = _XML_FUNC_PATTERN.search(raw)
            if func_match:
                tn = func_match.group(1).strip()
                if tn.lower() not in ("parameter", "/parameter", "function",
                                      "function_calls", "invoke", "/invoke", "tool"):
                    tool_name = tn
                elif tn.lower() == "function":
                    full_tag = func_match.group(0)
                    parts = full_tag.lstrip("<").rstrip(">").split()
                    if len(parts) >= 2:
                        tool_name = parts[1].strip()

    if not tool_name and tool_args:
        # Heuristic: infer tool name from parameter keys
        param_keys = set(tool_args.keys())
        if "command" in param_keys or "cwd" in param_keys:
            tool_name = "cmd_executor"
        elif "file_path" in param_keys or "n" in param_keys or "start" in param_keys:
            tool_name = "table_head_reader"
        elif "query" in param_keys or "folder_path" in param_keys:
            tool_name = "table_selector"

    if tool_name:
        return {"tool_name": tool_name, "arguments": tool_args}

    return None


def extract_tool_calls_from_response(response: str) -> List[Dict[str, Any]]:
    """Extract tool calls from a model response.

    Handles:
    1. JSON inside ``<tool_call>`` tags (standard)
    2. Hybrid XML+JSON formats: ``<function cmd_executor>JSON</function>``
    3. Nested XML parameter format: ``<cmd_executor><parameter ...>``
    4. Malformed XML (missing ``name=``, truncated, etc.)
    5. Unclosed ``<tool_call>`` tags (truncated model output)

    Strips ``<think>`` blocks first (they may contain quoted tool_call tags).
    """
    tool_calls: List[Dict[str, Any]] = []

    # Remove think blocks first
    cleaned = _THINK_PATTERN.sub('', response)

    # --- Path 1: properly closed <tool_call>...</tool_call> ---
    for raw in _TOOL_CALL_PATTERN.findall(cleaned):
        parsed = _parse_tool_call_content(raw)
        if parsed:
            tool_calls.append(parsed)

    # --- Path 2: unclosed <tool_call> (truncated model output) ---
    if not tool_calls:
        # Try to find <tool_call> without matching closing tag
        tc_start = cleaned.find('<tool_call>')
        if tc_start >= 0:
            after_tc = cleaned[tc_start + len('<tool_call>'):]
            # Remove any trailing XML tags that might be present
            # Try extracting JSON from after <tool_call>
            json_match = re.search(r'\{.*', after_tc, re.DOTALL)
            if json_match:
                json_candidate = json_match.group(0)
                # Try to parse as much as possible
                try:
                    obj, _ = json.JSONDecoder().raw_decode(json_candidate)
                    tool_name = obj.get("tool") or obj.get("name") or ""
                    tool_args = obj.get("params") or obj.get("arguments") or {}
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except (json.JSONDecodeError, TypeError):
                            tool_args = {}
                    if tool_name:
                        tool_calls.append({"tool_name": tool_name, "arguments": tool_args})
                except json.JSONDecodeError:
                    pass
            if not tool_calls:
                # Try XML fallback for unclosed content
                parsed = _parse_tool_call_content(after_tc)
                if parsed:
                    tool_calls.append(parsed)

    return tool_calls


def extract_answer_from_response(response: str) -> Optional[str]:
    """Extract the ``<answer>`` content from a model response.

    Primary: looks for ``<answer>...</answer>`` XML tags.
    Fallback: looks for ``{"answer": ...}`` JSON objects when no XML tags
    are present (model outputs raw JSON without wrapping).
    """
    cleaned = _THINK_PATTERN.sub('', response)
    m = _ANSWER_PATTERN.search(cleaned)
    if m:
        return m.group(1).strip()

    # Fallback: model outputs raw JSON like {"answer": "...", "data_source": [...]}
    # but only when there's no <tool_call> in the response (avoid false positives)
    if '<tool_call>' not in cleaned and '</tool_call>' not in cleaned:
        # Try to find a JSON object with an "answer" key
        for pattern in [
            r'\{\s*"answer"\s*:\s*"[^"]*"[\s\S]*?\}',   # {"answer": "...", ...}
        ]:
            fm = re.search(pattern, cleaned)
            if fm:
                try:
                    obj = json.loads(fm.group(0))
                    if isinstance(obj, dict) and "answer" in obj:
                        return fm.group(0)
                except (json.JSONDecodeError, TypeError):
                    continue

    return None


def extract_memory_from_response(response: str) -> Optional[Dict[str, Any]]:
    """Extract the ``<memory>`` JSON from a model response.

    Returns the parsed dict, or ``None`` if the tag is missing or the
    content is not valid JSON.
    """
    cleaned = _THINK_PATTERN.sub('', response)
    m = _MEMORY_PATTERN.search(cleaned)
    if m:
        return try_parse_json(m.group(1).strip())
    return None


def count_tool_calls(response: str) -> int:
    """Count the number of ``<tool_call>`` blocks in a response."""
    cleaned = _THINK_PATTERN.sub('', response)
    return len(_TOOL_CALL_PATTERN.findall(cleaned))


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

_JSON_BRACE_PATTERN = re.compile(r'\{.*\}', re.DOTALL)


def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse *text* as a JSON object; return ``None`` on failure.

    Tries direct parse first, then falls back to extracting the first
    ``{...}`` span via regex.
    """
    if not text or not text.strip():
        return None
    try:
        result = json.loads(text.strip())
        if isinstance(result, dict):
            return result
        return None
    except (json.JSONDecodeError, TypeError):
        pass
    # Fallback: extract a JSON object from within the text
    m = _JSON_BRACE_PATTERN.search(text.strip())
    if m:
        try:
            result = json.loads(m.group(0))
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, TypeError):
            pass
    return None


# ---------------------------------------------------------------------------
# Sample ID extraction
# ---------------------------------------------------------------------------

def get_sample_id(data: Dict[str, Any], idx: Optional[int] = None) -> str:
    """Unified sample identifier extraction.

    Tries ``sample_id``, ``task``, and ``file_path`` keys in order.
    Falls back to ``"sample_{idx}"`` when *idx* is provided and no key
    yields a non-empty value.

    Works on both benchmark sample dicts and serialised records
    (TES / FDS / rollout JSONL entries).
    """
    for key in ("sample_id", "task", "file_path"):
        val = data.get(key, "")
        if val:
            return str(val)
    if idx is not None:
        return f"sample_{idx}"
    return ""


# ---------------------------------------------------------------------------
# Default system template
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_TEMPLATE = """You are a professional table data analysis agent.
Your task is to answer the user's question by calling tools to extract data from tables.

**CRITICAL RULES:**
1. You may call AT MOST ONE tool per response. Multiple tool calls in one response are INVALID.
2. Wrap each tool call in <tool_call>JSON</tool_call> tags.
3. When you have enough information to answer, respond with a JSON object inside <answer>...</answer>.
4. The final answer MUST cite specific data values found in the tables.
5. Think step by step before each tool call.

**Working directory**: {table_path}

**Tool call format** (exactly one per response):
<tool_call>
{{"tool": "TOOL_NAME", "params": {{"param1": "value1"}}}}
</tool_call>

**Answer format** (JSON object inside <answer> tags):
<answer>
{{"answer": "Your concise answer with supporting data.", "data_source": ["table_name.xlsx"]}}
</answer>

**IMPORTANT — Memory Generation:**
After providing your answer, you MUST generate a structured memory summarizing the key
information you found. This memory will be used to answer future related subquestions.
Wrap the memory in <memory>JSON</memory> tags.

<memory>
{{
  "goal": "What this subquestion asked about",
  "tables": ["list of tables examined"],
  "key_facts": [
    {{"entity": "...", "time": "...", "metric": "...", "value": "...", "unit": "..."}}
  ],
  "derived_results": [
    {{"text": "computed result description", "value": "...", "inputs": [...]}}
  ],
  "constraints": ["any important constraints discovered"],
  "pitfalls": ["data quality issues or caveats"]
}}
</memory>
"""

DEFAULT_SYSTEM_TEMPLATE_PHASE1 = """You are a professional table data analysis agent.
Your task is to answer the user's question by calling tools to extract data from tables.

**CRITICAL RULES:**
1. You may call AT MOST ONE tool per response. Multiple tool calls in one response are INVALID.
2. Wrap each tool call in <tool_call>JSON</tool_call> tags.
3. When you have enough information to answer, respond with a JSON object inside <answer>...</answer>.
4. The final answer MUST cite specific data values found in the tables.
5. Think step by step before each tool call.

**Working directory**: {table_path}

**Tool call format** (exactly one per response):
<tool_call>
{{"tool": "TOOL_NAME", "params": {{"param1": "value1"}}}}
</tool_call>

**Answer format** (JSON object inside <answer> tags):
<answer>
{{"answer": "Your concise answer with supporting data.", "data_source": ["table_name.xlsx"]}}
</answer>
"""
