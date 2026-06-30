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


def extract_tool_calls_from_response(response: str) -> List[Dict[str, Any]]:
    """Extract tool calls from a model response.

    Handles both XML ``<tool_call>`` tags and native function-calling format
    embedded in the response content.  Strips ``<think>`` blocks first (they
    may contain quoted tool_call tags).
    """
    tool_calls: List[Dict[str, Any]] = []

    # Remove think blocks first (they may contain quoted tool_call tags)
    cleaned = _THINK_PATTERN.sub('', response)

    for raw in _TOOL_CALL_PATTERN.findall(cleaned):
        raw = raw.strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            try:
                obj, _ = json.JSONDecoder().raw_decode(raw)
            except json.JSONDecodeError:
                continue

        tool_name = obj.get("tool") or obj.get("name") or ""
        tool_args = obj.get("params") or obj.get("arguments") or {}
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except (json.JSONDecodeError, TypeError):
                tool_args = {}

        if tool_name:
            tool_calls.append({"tool_name": tool_name, "arguments": tool_args})

    return tool_calls


def extract_answer_from_response(response: str) -> Optional[str]:
    """Extract the ``<answer>`` content from a model response."""
    cleaned = _THINK_PATTERN.sub('', response)
    m = _ANSWER_PATTERN.search(cleaned)
    if m:
        return m.group(1).strip()
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
