#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _compress_choice(text: str, max_chars: int) -> str:
    text = _normalize_text(text)
    if not text:
        return ""
    first_sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
    text = first_sentence or text
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rstrip(" ,;:")
    return cut + "..."


def _sample_id(video_path: str, question: str, answer: str) -> str:
    raw = json.dumps(
        {"video_path": video_path, "question": question, "answer": answer},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _pick_distractors(
    *,
    row_idx: int,
    correct: str,
    task_pool: list[str],
    global_pool: list[str],
    k: int,
) -> list[str]:
    seed = int(hashlib.sha1(f"{row_idx}:{correct}".encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    seen = {correct.lower()}
    chosen: list[str] = []
    for pool in (task_pool, global_pool):
        if not pool:
            continue
        if len(pool) <= max(64, k * 16):
            candidates = list(pool)
            rng.shuffle(candidates)
        else:
            sample_k = min(len(pool), max(64, k * 32))
            candidates = [pool[i] for i in rng.sample(range(len(pool)), k=sample_k)]
        for cand in candidates:
            if cand.lower() in seen:
                continue
            seen.add(cand.lower())
            chosen.append(cand)
            if len(chosen) >= k:
                return chosen
    return chosen


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    asset_root = Path(os.getenv("REVISE_ASSET_ROOT", repo_root / "data" / "revise_assets"))
    ap = argparse.ArgumentParser(description="Convert VideoEspresso open-ended train JSON to local MC JSON.")
    ap.add_argument(
        "--input",
        default=os.getenv(
            "REVISE_VIDEOESPRESSO_TRAIN_VIDEO_JSON",
            str(asset_root / "VideoEspresso" / "train_video" / "videoespresso_train_video.json"),
        ),
        help="Open-ended VideoEspresso train JSON.",
    )
    ap.add_argument(
        "--output",
        default=str(repo_root / "outputs" / "videoespresso_train_mc.json"),
        help="Output multiple-choice JSON path.",
    )
    ap.add_argument(
        "--num-options",
        type=int,
        default=4,
        help="Number of multiple-choice options. VideoEspresso close-ended evaluation uses A-D.",
    )
    ap.add_argument("--max-choice-chars", type=int, default=160)
    ap.add_argument("--max-rows", type=int, default=0, help="Optional cap for data-processing smoke runs.")
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Expected list in {args.input}, got {type(data)}")

    normalized_rows: list[dict] = []
    task_answers: dict[str, list[str]] = {}
    global_answers: list[str] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        question = _normalize_text(str(row.get("question") or ""))
        answer = _compress_choice(str(row.get("answer") or ""), args.max_choice_chars)
        video_path = _normalize_text(str(row.get("video_path") or ""))
        task = _normalize_text(str(row.get("task") or "general")) or "general"
        if not question or not answer or not video_path:
            continue
        normalized_rows.append({"question": question, "answer": answer, "video_path": video_path, "task": task})
        task_answers.setdefault(task, []).append(answer)
        global_answers.append(answer)

    def _dedupe(values: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for val in values:
            key = val.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(val)
        return out

    task_answers = {k: _dedupe(v) for k, v in task_answers.items()}
    global_answers = _dedupe(global_answers)

    out: list[dict] = []
    num_distractors = max(1, int(args.num_options) - 1)
    for idx, row in enumerate(normalized_rows):
        if args.max_rows > 0 and len(out) >= int(args.max_rows):
            break
        distractors = _pick_distractors(
            row_idx=idx,
            correct=row["answer"],
            task_pool=task_answers.get(row["task"], []),
            global_pool=global_answers,
            k=num_distractors,
        )
        if len(distractors) < num_distractors:
            continue
        options = [row["answer"], *distractors[:num_distractors]]
        seed = int(hashlib.sha1(f"shuffle:{idx}:{row['question']}".encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        rng.shuffle(options)
        answer_idx = options.index(row["answer"])
        out.append(
            {
                "id": _sample_id(row["video_path"], row["question"], row["answer"]),
                "question": row["question"],
                "options": options,
                "correct_answer": chr(ord("A") + answer_idx),
                "video_path": row["video_path"],
                "task": row["task"],
                "source": "videoespresso_open_train_synthetic_mc",
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(out), "output": str(output_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
