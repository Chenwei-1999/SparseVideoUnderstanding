# Copyright 2026
# Licensed under the Apache License, Version 2.0

"""NExT-QA dataset loader for REVISE-style multi-round video QA."""

from __future__ import annotations

import json
import os
import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class NextQASample:
    video_id: str
    video_path: str
    frame_count: int
    width: int
    height: int
    question: str
    choices: list[str]
    answer_idx: int
    qid: str
    qtype: str


def _normalize_video_id(video_id: Any) -> str:
    """Ensure video_id is a string without decimals."""
    if isinstance(video_id, (int,)):
        return str(video_id)
    if isinstance(video_id, float):
        return str(int(video_id))
    return str(video_id)


def _load_map(map_path: str) -> dict[str, str]:
    with open(map_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): v for k, v in data.items()}


def _build_video_path(video_root: str, rel_path: str) -> str:
    if rel_path.endswith(".mp4"):
        return os.path.join(video_root, rel_path)
    return os.path.join(video_root, f"{rel_path}.mp4")


def _format_question(question: str, choices: list[str]) -> str:
    option_labels = [chr(ord("A") + i) for i in range(len(choices))]
    lines = [f"Question: {question}", "Options:"]
    for label, choice in zip(option_labels, choices, strict=False):
        lines.append(f"{label}. {choice}")
    if option_labels:
        lines.append(f"Answer with exactly one letter only: {', '.join(option_labels)}.")
    return "\n".join(lines)


def _stable_sample_id(video_id: str, question: str, choices: list[str], answer_idx: int) -> str:
    payload = {
        "video_id": str(video_id),
        "question": str(question),
        "choices": [str(c) for c in (choices or [])],
        "answer_idx": int(answer_idx),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return digest


class NextQADataset(Dataset):
    """Custom dataset for NExT-QA CSVs.

    Expected data config additions:
      data.nextqa.video_root
      data.nextqa.map_json
    """

    def __init__(
        self,
        data_files: str | Iterable[str],
        tokenizer=None,
        processor=None,
        config=None,
        max_samples: int = -1,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

        if isinstance(data_files, (list, tuple)):
            files = list(data_files)
        else:
            files = [data_files]

        # Allow limiting dataset size via config to build tiny smoke runs.
        cfg_max_samples = None
        if isinstance(config, dict):
            cfg_max_samples = config.get("max_samples")
        if cfg_max_samples is not None:
            max_samples = int(cfg_max_samples)

        nextqa_cfg = (config or {}).get("nextqa", {})
        video_root = nextqa_cfg.get("video_root")
        map_json = nextqa_cfg.get("map_json")
        if not video_root or not map_json:
            raise ValueError("data.nextqa.video_root and data.nextqa.map_json must be set for NextQADataset")

        self.video_root = video_root
        self.map_json = map_json
        self.video_map = _load_map(map_json)

        dfs = [pd.read_csv(path) for path in files]
        df = pd.concat(dfs, ignore_index=True)

        if max_samples and max_samples > 0 and max_samples < len(df):
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

        self.samples: list[NextQASample] = []
        for _, row in df.iterrows():
            video_id = _normalize_video_id(row["video"])
            rel_path = self.video_map.get(video_id)
            if rel_path is None:
                # Skip if no mapping available
                continue
            video_path = _build_video_path(video_root, rel_path)
            choices = [row[f"a{i}"] for i in range(5)]
            sample = NextQASample(
                video_id=video_id,
                video_path=video_path,
                frame_count=int(row.get("frame_count", 0)),
                width=int(row.get("width", 0)),
                height=int(row.get("height", 0)),
                question=str(row.get("question", "")),
                choices=[str(c) for c in choices],
                answer_idx=int(row.get("answer", 0)),
                qid=str(row.get("qid", "")),
                qtype=str(row.get("type", "")),
            )
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        sample_id = _stable_sample_id(sample.video_id, sample.question, sample.choices, sample.answer_idx)
        prompt_text = _format_question(sample.question, sample.choices)
        raw_prompt = [
            {
                "role": "user",
                "content": prompt_text,
            }
        ]

        ground_truth = {
            "answer_idx": sample.answer_idx,
            "answer_text": sample.choices[sample.answer_idx],
            "choices": sample.choices,
            "qid": sample.qid,
            "type": sample.qtype,
            "sample_id": sample_id,
        }

        extra_info = {
            "sample_id": sample_id,
            "video_id": sample.video_id,
            "video_path": sample.video_path,
            "frame_count": sample.frame_count,
            "width": sample.width,
            "height": sample.height,
            "question": sample.question,
            "choices": sample.choices,
        }

        return {
            "raw_prompt": raw_prompt,
            "data_source": "revise_nextqa",
            "reward_model": {"ground_truth": ground_truth},
            "extra_info": extra_info,
            "agent_name": "revise_agent",
            # Dummy tensor to satisfy DataProto batching requirements
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "index": idx,
        }
