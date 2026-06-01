#!/usr/bin/env python3
"""Generate SFT training data from REVISE multi-round evaluation logs.

Reads REVISE multi-round eval logs (JSONL), reconstructs multi-turn
conversations, filters for quality, and outputs parquet files compatible
with MultiTurnSFTDataset.

IMPORTANT: The input log must come from the TRAIN split to avoid data
leakage. Use the dataset-specific `run_generate_teacher_data*.sh` helper
to generate train-split logs.

Usage:
    # Step 1: Generate teacher data on train split (requires GPU)
    ./revise/run_generate_teacher_data.sh

    # Step 2: Convert to SFT parquet
    python revise/generate_sft_data.py \
        --input outputs/nextqa_teacher_train_log.jsonl \
        --output outputs/sft_data/revise_sft.parquet \
        --val_ratio 0.05
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revise.pnp.utils import parse_strict_revise_action  # noqa: E402


def has_valid_summary(text: str) -> bool:
    """Check that text contains a well-formed <summarize>...</summarize> block."""
    return bool(re.search(r"<summarize>.+?</summarize>", text, re.DOTALL))


def has_valid_think(text: str) -> bool:
    """Check that text begins (after optional whitespace) with a <think>...</think> trace.

    The paper protocol requires every response to start with a reasoning trace.
    """
    return bool(re.match(r"\s*<think>.+?</think>", text, re.DOTALL))


def has_valid_answer(text: str) -> bool:
    """Check that the final-round output contains <answer>LETTER</answer>."""
    return bool(re.search(r"<answer>\s*[A-E]\s*</answer>", text))


def has_template_text(text: str) -> bool:
    """Detect if the model simply copied the template placeholder."""
    markers = [
        "I will summarize what has been shown so far",
        "I will record the key observations",
        "I will update my belief as new evidence arrives",
    ]
    return any(m in text for m in markers)


def strip_image_placeholders(text: str) -> str:
    """Replace <image> vision placeholders with a text-only marker."""
    return text.replace("<image>", "[frame]")


def first_assistant_action(messages: list[dict]) -> str:
    """Return the first REVISE assistant action in a conversation."""
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "")
        strict_action = parse_strict_revise_action(content)
        if strict_action is None:
            return "other"
        return str(strict_action["kind"])
    return "other"


def conversation_fingerprint(messages: list[dict]) -> str:
    """Stable content key for checking train/validation overlap after curation."""
    return json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def rebalance_first_select_ratio(
    conversations: list[list[dict]],
    *,
    min_first_select_ratio: float,
    seed: int,
) -> list[list[dict]]:
    """Oversample Select-first conversations until a minimum first-action ratio is reached.

    The teacher can be correct while still being a poor multi-round teacher if almost every
    valid trace answers immediately. For SFT, oversampling the scarce valid Select traces is
    preferable to changing the runtime loop or inventing a second prompt.
    """
    target = float(min_first_select_ratio or 0.0)
    if target <= 0.0 or not conversations:
        return conversations
    if not (0.0 < target < 1.0):
        raise ValueError("--min-first-select-ratio must be in [0, 1).")

    select_conversations = [conv for conv in conversations if first_assistant_action(conv) == "select"]
    select_count = len(select_conversations)
    total = len(conversations)
    current = select_count / max(1, total)
    if current >= target:
        return conversations
    if not select_conversations:
        print("  WARNING: no Select-first conversations found; cannot rebalance first-action ratio.")
        return conversations

    import math
    import random

    needed = int(math.ceil((target * total - select_count) / (1.0 - target)))
    rng = random.Random(seed)
    augmented = list(conversations)
    augmented.extend(rng.choices(select_conversations, k=max(0, needed)))
    rng.shuffle(augmented)
    return augmented


def teacher_answer_matches_ground_truth(entry: dict, strict_action: dict) -> bool:
    """Return False when a logged teacher answer contradicts available ground truth."""
    if "ground_truth_idx" not in entry:
        return True
    try:
        gold_idx = int(entry["ground_truth_idx"])
    except Exception:
        return True
    if gold_idx < 0:
        return True

    expected = chr(ord("A") + gold_idx)
    # Validate the actual assistant text that will be learned, not a cached
    # side-channel field that may be stale.
    answer = strict_action.get("answer")
    if not answer:
        return False
    answer = str(answer or "").strip().upper()
    return answer == expected


def build_conversation(rounds: list[dict], *, max_rounds: int | None = 4) -> list[dict] | None:
    """Build a chat-format conversation from a list of sorted round entries.

    Returns None if any quality check fails.
    """
    if not rounds:
        return None
    if max_rounds is not None:
        max_rounds = int(max_rounds)
        if len(rounds) > max_rounds:
            return None
        for entry in rounds:
            try:
                round_idx = int(entry.get("round_idx", entry.get("round", 0)))
            except Exception:
                return None
            if round_idx > max_rounds:
                return None

    messages = []

    # System prompt (same across rounds)
    system_prompt = rounds[0].get("system_prompt", "")
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for i, entry in enumerate(rounds):
        raw_output = entry.get("raw_output", "")
        user_text = entry.get("user_text", "")

        # Quality filter: every round must begin with a <think> reasoning trace (paper protocol)
        if not has_valid_think(raw_output):
            return None

        # Quality filter: reject template-copied outputs
        if has_template_text(raw_output):
            return None

        strict_action = parse_strict_revise_action(raw_output)
        if strict_action is None:
            return None

        is_last = i == len(rounds) - 1
        if is_last:
            # Answer round = <think> + <answer> only; no <summarize> required.
            if strict_action["kind"] != "answer" or not has_valid_answer(raw_output):
                return None
            if not teacher_answer_matches_ground_truth(entry, strict_action):
                return None
        else:
            # Select round = <think> + <summarize> + <select>.
            if strict_action["kind"] != "select" or not has_valid_summary(raw_output):
                return None

        # Strip image placeholders for text-only SFT
        user_content = strip_image_placeholders(user_text)
        assistant_content = raw_output.strip()

        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})

    return messages


def main():
    repo_root = Path(__file__).resolve().parents[1]
    asset_root = Path(os.getenv("REVISE_ASSET_ROOT", repo_root / "data" / "revise_assets"))
    parser = argparse.ArgumentParser(description="Generate SFT data from REVISE eval logs")
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/nextqa_teacher_train_log.jsonl",
        help="Path to eval log JSONL (should be from train split)",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default=os.getenv("REVISE_NEXTQA_VAL_CSV", str(asset_root / "NExT-QA" / "nextqa" / "val.csv")),
        help="Optional validation CSV to check for data leakage. Pass '' to disable.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/sft_data/revise_sft.parquet",
        help="Output parquet path (train split; val split auto-named)",
    )
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Fraction held out for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=4,
        help="Reject conversations that exceed the REVISE round budget.",
    )
    parser.add_argument(
        "--min-first-select-ratio",
        type=float,
        default=0.0,
        help=(
            "If >0, oversample valid conversations whose first assistant action is <select> until "
            "the SFT corpus reaches this minimum ratio. Useful when a local teacher collapses to "
            "one-round answers."
        ),
    )
    args = parser.parse_args()

    # 1. Read and group entries by sample_id
    print(f"Reading {args.input} ...")
    sample_rounds: dict[str, list[dict]] = defaultdict(list)
    with open(args.input) as f:
        for line in f:
            entry = json.loads(line)
            sample_rounds[entry["sample_id"]].append(entry)

    print(f"  Total entries: {sum(len(v) for v in sample_rounds.values())}")
    print(f"  Unique samples: {len(sample_rounds)}")

    # 1b. Check for data leakage against val split
    if args.val_csv and os.path.exists(args.val_csv):
        val_keys = set()
        with open(args.val_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                val_keys.add((row["video"], row["question"]))
        eval_keys = set()
        for entries in sample_rounds.values():
            e = entries[0]
            video_id = e.get("video_id")
            question = e.get("question")
            if video_id and question:
                eval_keys.add((video_id, question))
        overlap = eval_keys & val_keys
        if not eval_keys:
            print("  (Skipping leakage overlap stats: log entries do not expose video_id/question keys)")
            overlap = set()
        if overlap:
            pct = len(overlap) / len(eval_keys) * 100
            print(f"\n  WARNING: {len(overlap)}/{len(eval_keys)} samples ({pct:.0f}%) overlap with val split!")
            print("  This will cause data leakage if you evaluate on val.")
            print("  Use run_generate_teacher_data.sh to generate train-split teacher data.\n")
            if pct > 50:
                print("  ERROR: >50% overlap. Refusing to generate SFT data from val-split logs.")
                print("  Pass --val-csv '' to override (not recommended).")
                sys.exit(1)
    else:
        print(f"  (Skipping leakage check: {args.val_csv} not found)")

    # 2. Sort rounds within each sample and build conversations
    conversations = []
    skipped = 0
    for sample_id, entries in sample_rounds.items():
        # Sort by round_idx/round, then retry_idx (take retry_idx=0 only)
        entries = [e for e in entries if e.get("retry_idx", 0) == 0]
        entries.sort(key=lambda e: int(e.get("round_idx", e.get("round", 0))))
        conv = build_conversation(entries, max_rounds=args.max_rounds)
        if conv is not None:
            conversations.append(conv)
        else:
            skipped += 1

    print(f"  Valid conversations: {len(conversations)}")
    print(f"  Skipped (quality filter): {skipped}")

    if not conversations:
        print("ERROR: No valid conversations found. Check input file.")
        sys.exit(1)

    all_actions = Counter(first_assistant_action(conv) for conv in conversations)
    print(f"  First assistant action before split: {dict(all_actions)}")

    # 3. Split train / val before any oversampling so validation remains a held-out
    # quality check. Rebalancing duplicates training rows only.
    import random

    random.seed(args.seed)
    indices = list(range(len(conversations)))
    random.shuffle(indices)
    if args.val_ratio <= 0 or len(conversations) < 2:
        n_val = 0
    else:
        n_val = min(len(conversations) - 1, max(1, int(len(conversations) * args.val_ratio)))
    val_indices = set(indices[:n_val])
    train_convs = [conversations[i] for i in range(len(conversations)) if i not in val_indices]
    val_convs = [conversations[i] for i in val_indices]

    train_before_actions = Counter(first_assistant_action(conv) for conv in train_convs)
    val_actions = Counter(first_assistant_action(conv) for conv in val_convs)
    print(f"  Train first assistant action before rebalance: {dict(train_before_actions)}")
    print(f"  Val first assistant action: {dict(val_actions)}")
    train_convs = rebalance_first_select_ratio(
        train_convs,
        min_first_select_ratio=args.min_first_select_ratio,
        seed=args.seed,
    )
    train_after_actions = Counter(first_assistant_action(conv) for conv in train_convs)
    train_select_ratio = train_after_actions.get("select", 0) / max(1, len(train_convs))
    print(f"  Train first assistant action after rebalance: {dict(train_after_actions)}")
    print(f"  Train first-select ratio: {train_select_ratio:.3f}")

    train_keys = {conversation_fingerprint(conv) for conv in train_convs}
    val_keys = {conversation_fingerprint(conv) for conv in val_convs}
    overlap_count = len(train_keys & val_keys)
    if overlap_count:
        print(f"  WARNING: {overlap_count} exact conversations overlap train/val after curation.")

    print(f"  Train: {len(train_convs)}, Val: {len(val_convs)}")

    # 4. Write parquet files
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame({"messages": train_convs})
    val_path = output_path.parent / output_path.name.replace(".parquet", "_val.parquet")
    train_path = output_path.parent / output_path.name.replace(".parquet", "_train.parquet")

    train_df.to_parquet(str(train_path))
    val_df = pd.DataFrame({"messages": val_convs})
    val_df.to_parquet(str(val_path))

    print(f"  Wrote {train_path} ({len(train_df)} rows)")
    print(f"  Wrote {val_path} ({len(val_df)} rows)")

    # 5. Print stats
    written_convs = train_convs + val_convs
    turn_counts = [len([m for m in c if m["role"] == "assistant"]) for c in written_convs]
    print("\nStats:")
    print(f"  Avg assistant turns per conversation: {sum(turn_counts) / len(turn_counts):.1f}")
    print(f"  Min/Max turns: {min(turn_counts)}/{max(turn_counts)}")


if __name__ == "__main__":
    main()
