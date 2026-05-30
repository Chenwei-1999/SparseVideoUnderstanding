# Copyright 2026
# Licensed under the Apache License, Version 2.0

"""LVBench dataset loader for REVISE-style multi-round video QA."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


_OPT_RE = re.compile(r"^\(([A-Z])\)\s*(.*)$")


def _parse_options_from_lvbench_question(question: str) -> tuple[str, list[str]]:
    text = str(question or "").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "", []

    q_lines: list[str] = []
    opts: dict[str, str] = {}
    for ln in lines:
        m = _OPT_RE.match(ln)
        if m:
            opts[m.group(1).upper()] = m.group(2).strip()
        else:
            q_lines.append(ln)

    q_text = " ".join(q_lines).strip()
    if not opts:
        return q_text, []

    letters_sorted = sorted(opts.keys())
    options = [opts[k] for k in letters_sorted]
    return q_text, options


def _format_question(question: str, choices: list[str]) -> str:
    labels = [chr(ord("A") + i) for i in range(len(choices))]
    lines = [f"Question: {question}", "Options:"]
    for label, choice in zip(labels, choices, strict=False):
        lines.append(f"{label}. {choice}")
    if labels:
        lines.append(f"Answer with exactly one letter only: {', '.join(labels)}.")
    return "\n".join(lines)


def _stable_sample_id(dataset: str, video_key: str, uid: str, question: str, choices: list[str], answer_letter: str) -> str:
    payload = {
        "dataset": str(dataset),
        "video_key": str(video_key),
        "uid": str(uid),
        "question": str(question),
        "choices": [str(c) for c in (choices or [])],
        "answer": str(answer_letter),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return digest


@dataclass
class LVBenchSample:
    video_key: str
    video_path: str
    uid: str
    question: str
    choices: list[str]
    answer_idx: int
    answer_letter: str
    question_type: str
    time_reference: str


class LVBenchDataset(Dataset):
    """LVBench loader (HF metadata + local mp4 cache).

    Expected data config additions:
      data.lvbench.video_cache_dir: root dir containing a `lvbench/` folder with mp4s.
      data.lvbench.split: HF split name (default: "train"; LVBench currently exposes train only).
    """

    def __init__(
        self,
        data_files: str | Iterable[str],
        tokenizer=None,
        processor=None,
        config=None,
        max_samples: int = -1,
    ):
        self.config = config or {}
        self.tokenizer = tokenizer
        self.processor = processor

        # Allow limiting dataset size via config to build tiny smoke runs.
        cfg_max_samples = None
        if isinstance(config, dict):
            cfg_max_samples = config.get("max_samples")
        if cfg_max_samples is not None:
            max_samples = int(cfg_max_samples)

        lv_cfg = (config or {}).get("lvbench", {}) if isinstance(config, dict) else {}
        video_cache_dir = str(lv_cfg.get("video_cache_dir") or "/tmp/lvbench_video_cache")
        hf_split = str(lv_cfg.get("split") or "")

        # Interpret `data_files` as the split when it looks like a split name.
        split = ""
        if isinstance(data_files, (list, tuple)):
            parts = [str(x) for x in data_files if str(x).strip()]
            if len(parts) == 1:
                split = parts[0]
        else:
            split = str(data_files or "").strip()

        if not split or split.endswith(".csv") or split.endswith(".json") or split.endswith(".jsonl"):
            split = hf_split or "train"

        cache_dir = Path(video_cache_dir) / "lvbench"
        self.video_cache_dir = str(cache_dir)

        ds = load_dataset("lmms-lab/LVBench", split=split)
        rows = list(ds)

        if max_samples and max_samples > 0 and max_samples < len(rows):
            # Deterministic subset for reproducibility.
            import random

            rng = random.Random(42)
            rng.shuffle(rows)
            rows = rows[:max_samples]

        self.samples: list[LVBenchSample] = []
        missing = 0
        for ex in rows:
            video_key = str(ex.get("video_path") or "").strip()
            if not video_key:
                continue
            uid = str(ex.get("uid") or ex.get("key") or "").strip() or video_key
            q_raw = str(ex.get("question") or "").strip()
            q_text, options = _parse_options_from_lvbench_question(q_raw)
            answer_letter = str(ex.get("answer") or "").strip().upper()
            if not answer_letter:
                continue

            # Options are returned in sorted letter order (A,B,C,...).
            answer_idx = ord(answer_letter) - ord("A")
            if answer_idx < 0 or answer_idx >= len(options):
                continue

            video_path = str(cache_dir / video_key)
            if not os.path.exists(video_path) or os.path.getsize(video_path) <= 0:
                missing += 1
                continue

            self.samples.append(
                LVBenchSample(
                    video_key=video_key,
                    video_path=video_path,
                    uid=uid,
                    question=q_text if q_text else q_raw,
                    choices=[str(c) for c in options],
                    answer_idx=int(answer_idx),
                    answer_letter=answer_letter,
                    question_type=str(ex.get("question_type") or ex.get("type") or "").strip(),
                    time_reference=str(ex.get("time_reference") or "").strip(),
                )
            )

        if missing:
            # Keep a small hint for debugging, but do not fail hard.
            self.missing_videos = int(missing)
        else:
            self.missing_videos = 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        sample_id = _stable_sample_id(
            "lvbench", sample.video_key, sample.uid, sample.question, sample.choices, sample.answer_letter
        )
        prompt_text = _format_question(sample.question, sample.choices)
        raw_prompt = [{"role": "user", "content": prompt_text}]

        ground_truth = {
            "answer_idx": sample.answer_idx,
            "answer_text": sample.choices[sample.answer_idx],
            "choices": sample.choices,
            "uid": sample.uid,
            "type": sample.question_type,
            "sample_id": sample_id,
        }

        extra_info = {
            "sample_id": sample_id,
            "video_id": Path(sample.video_key).stem,
            "video_key": sample.video_key,
            "video_path": sample.video_path,
            # Let the agent loop infer video metadata; LVBench videos are long.
            "frame_count": 0,
            "question": sample.question,
            "choices": sample.choices,
            "time_reference": sample.time_reference,
            "type": sample.question_type,
        }

        return {
            "raw_prompt": raw_prompt,
            "data_source": "revise_lvbench",
            "reward_model": {"ground_truth": ground_truth},
            "extra_info": extra_info,
            "agent_name": "revise_agent",
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "index": idx,
        }

