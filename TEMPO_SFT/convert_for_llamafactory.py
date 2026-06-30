"""
Convert OpenAI-format chat data (with tool_calls field on assistant messages)
to LLaMA-Factory Qwen sharegpt format.

In Qwen's format, tool calls are embedded in assistant content as:
    <tool_call>
    {"name": "...", "arguments": {...}}
    </tool_call>

Tool output messages keep role="tool" (mapped to observation in LLaMA-Factory).

Usage:
    python TEMPO_SFT/convert_for_llamafactory.py \
        --input SFTbuild/output/trainable_sft_chat.jsonl \
        --output TEMPO_SFT/trainable_sft_qwen.jsonl
"""

import json
import argparse
from typing import Dict, List


def format_tool_calls(tool_calls: List[Dict]) -> str:
    """Format OpenAI tool_calls as Qwen <tool_call> XML."""
    formatted = []
    for tc in tool_calls:
        func = tc["function"]
        try:
            arguments = json.loads(func["arguments"])
        except (json.JSONDecodeError, TypeError):
            arguments = func["arguments"]
        formatted.append(
            json.dumps(
                {"name": func["name"], "arguments": arguments},
                ensure_ascii=False,
            )
        )
    return "\n".join(f"<tool_call>\n{f}\n</tool_call>" for f in formatted)


def convert_message(msg: Dict) -> Dict:
    """Convert a single message from OpenAI format to Qwen sharegpt format."""
    msg = dict(msg)  # shallow copy

    if msg.get("role") == "assistant" and "tool_calls" in msg:
        tool_calls = msg.pop("tool_calls")
        tool_xml = format_tool_calls(tool_calls)
        content = msg.get("content", "") or ""
        msg["content"] = content + "\n" + tool_xml

    # Drop tool_call_id from tool messages (not needed in LLaMA-Factory format)
    msg.pop("tool_call_id", None)

    return msg


def convert_dialog(messages: List[Dict]) -> List[Dict]:
    """Convert all messages in a dialog.

    Steps:
    1. Embed tool_calls into assistant content as <tool_call> XML.
    2. Merge consecutive tool messages into one to satisfy LLaMA-Factory's
       odd/even role alternation check (parallel tool calls produce multiple
       consecutive tool outputs).
    """
    converted = [convert_message(m) for m in messages]

    # Merge consecutive tool messages (role="tool")
    merged = []
    for m in converted:
        if m["role"] == "tool" and merged and merged[-1]["role"] == "tool":
            # Append content to the previous tool message
            merged[-1]["content"] += "\n\n" + m["content"]
        else:
            merged.append(m)

    return merged


def main():
    parser = argparse.ArgumentParser(description="Convert OpenAI chat to LLaMA-Factory Qwen sharegpt")
    parser.add_argument("--input", required=True, help="Input JSONL file (OpenAI format)")
    parser.add_argument("--output", required=True, help="Output JSONL file (LLaMA-Factory format)")
    args = parser.parse_args()

    converted = 0
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            sample["messages"] = convert_dialog(sample["messages"])
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            converted += 1

    print(f"Converted {converted} dialogs → {args.output}")


if __name__ == "__main__":
    main()
