"""
Step 2: 从累计式 conversation_trace 中按子问题切分 agent_steps。

输入:
  --trace_dir   : 原始 trace JSON 文件目录（如 traces_output/）
  --samples     : benchmark 样本 JSON 文件（如 dataset/samples_normal_easy.json）
  --output      : 输出 JSONL 路径（默认 output/aligned_subquestions.jsonl）

输出: aligned_subquestions.jsonl
每行一条子问题记录:
{
  "sample_id": "...",
  "candidate_id": "trace_xxx",
  "subquestion_id": 1,
  "user": "checkout_list[i].info_item",
  "agent_steps": [...],
  "assistant_answer": {...},
  "match_method": "exact" | "normalized" | ...
}
"""
import os
import sys
import json
import argparse

# Allow importing from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SFTbuild.utils import (
    load_trace, load_samples, get_full_messages, find_sample,
    find_user_anchors, parse_agent_steps, parse_final_answer,
    write_jsonl, normalize_text
)


def decompose_trace(trace: dict, sample: dict, trace_id: str, verbose: bool = False) -> list:
    """
    将一条 trace 拆解为子问题级记录列表。
    """
    full_messages = get_full_messages(trace)
    if not full_messages:
        if verbose:
            print(f"  [SKIP] {trace_id}: empty messages")
        return []

    checkout_list = sample.get('design', {}).get('checkout_list', [])
    if not checkout_list:
        if verbose:
            print(f"  [SKIP] {trace_id}: no checkout_list in sample")
        return []

    # 提取已有的 evaluation 中的 accuracy_steps（用于 fallback 匹配）
    evaluation = trace.get('evaluation', {})
    accuracy_steps = evaluation.get('accuracy_steps', [])

    # 找 user anchor
    anchors = find_user_anchors(full_messages, checkout_list, accuracy_steps, verbose=verbose)

    # 构建相邻 anchor 之间的 span
    valid_anchors = [a for a in anchors if a['msg_idx'] is not None]
    if not valid_anchors:
        if verbose:
            print(f"  [SKIP] {trace_id}: no user anchors matched")
        return []

    records = []
    for i, anchor in enumerate(anchors):
        if anchor['msg_idx'] is None:
            # 未匹配到的子问题：生成一条空记录
            records.append({
                'sample_id': sample.get('task', ''),
                'candidate_id': trace_id,
                'subquestion_id': anchor['sub_idx'] + 1,
                'user': anchor['info_item'],
                'agent_steps': [],
                'assistant_answer': {'answer': '', 'data_source': []},
                'match_method': 'unmatched',
                '_missing': True
            })
            continue

        # 确定 span 范围
        start_idx = anchor['msg_idx']

        # 找下一个已匹配 anchor 作为 end
        end_idx = None
        for j in range(i + 1, len(anchors)):
            next_anchor = anchors[j]
            if next_anchor['msg_idx'] is not None:
                end_idx = next_anchor['msg_idx']
                break
        # end_idx 默认为消息列表末尾

        if end_idx is not None:
            span_messages = full_messages[start_idx:end_idx]
        else:
            span_messages = full_messages[start_idx:]

        # 解析 agent_steps
        agent_steps = parse_agent_steps(span_messages)

        # 提取 final_answer
        final_answer = {'answer': '', 'data_source': []}
        for step in reversed(agent_steps):
            if step['type'] == 'final_answer':
                final_answer = step.get('assistant_answer', final_answer)
                break

        # 如果没有 final_answer step，尝试从 span 最后一条 assistant 消息提取
        if final_answer['answer'] == '':
            for msg in reversed(span_messages):
                if msg.get('role') == 'assistant' and not msg.get('tool_calls'):
                    content = msg.get('content', '')
                    if content:
                        final_answer = parse_final_answer(content)
                        agent_steps.append({
                            'agent_step_id': len(agent_steps) + 1,
                            'type': 'final_answer',
                            'assistant_answer': final_answer
                        })
                    break

        records.append({
            'sample_id': sample.get('task', ''),
            'candidate_id': trace_id,
            'subquestion_id': anchor['sub_idx'] + 1,
            'user': anchor['info_item'],
            'agent_steps': agent_steps,
            'assistant_answer': final_answer,
            'match_method': anchor['match_method']
        })

    return records


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description='Step 2: Decompose traces into sub-questions')
    parser.add_argument('--trace_dir', type=str,
                        default=os.path.join(project_root, 'traces_output'),
                        help='Directory containing trace JSON files')
    parser.add_argument('--samples', type=str,
                        default=os.path.join(project_root, 'dataset', 'samples_normal_easy.json'),
                        help='Path to samples JSON file')
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'output', 'aligned_subquestions.jsonl'),
                        help='Output JSONL path')
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                        help='Verbose output')
    args = parser.parse_args()

    # 收集 trace 文件
    trace_files = sorted([
        f for f in os.listdir(args.trace_dir)
        if f.startswith('trace_') and f.endswith('.json')
    ])
    if not trace_files:
        print(f"[ERROR] No trace_*.json files found in {args.trace_dir}")
        sys.exit(1)

    print(f"Found {len(trace_files)} trace files in {args.trace_dir}")

    samples = load_samples(args.samples)
    print(f"Loaded {len(samples)} samples from {args.samples}")

    all_records = []
    stats = {'total': 0, 'matched': 0, 'unmatched': 0, 'no_sample': 0}

    for tf in trace_files:
        trace_path = os.path.join(args.trace_dir, tf)
        trace = load_trace(trace_path)
        trace_id = tf.replace('.json', '')

        sample = find_sample(trace, samples)
        if sample is None:
            if args.verbose:
                print(f"  [SKIP] {trace_id}: no matching sample found")
            stats['no_sample'] += 1
            continue

        records = decompose_trace(trace, sample, trace_id, verbose=args.verbose)
        n_matched = sum(1 for r in records if not r.get('_missing'))
        print(f"  {trace_id}: {len(records)} sub-questions, {n_matched} matched")
        all_records.extend(records)
        stats['total'] += 1

    stats['matched'] = sum(1 for r in all_records if not r.get('_missing'))
    stats['unmatched'] = sum(1 for r in all_records if r.get('_missing'))

    write_jsonl(args.output, all_records)
    print(f"\nDone. {stats['total']} traces → {len(all_records)} sub-question records")
    print(f"  Matched: {stats['matched']}, Unmatched: {stats['unmatched']}")
    print(f"  No sample found: {stats['no_sample']}")
    print(f"  Output: {args.output}")


if __name__ == '__main__':
    main()
