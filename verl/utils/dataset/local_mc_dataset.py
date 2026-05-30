# Copyright 2026
# Licensed under the Apache License, Version 2.0

"""Generic local multiple-choice video QA dataset loader for REVISE."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

try:
    from omegaconf import DictConfig, OmegaConf
except Exception:  # pragma: no cover - OmegaConf is available in training envs
    DictConfig = None
    OmegaConf = None


def _format_question(question: str, choices: list[str]) -> str:
    labels = [chr(ord("A") + i) for i in range(len(choices))]
    lines = [f"Question: {question}", "Options:"]
    for label, choice in zip(labels, choices, strict=False):
        lines.append(f"{label}. {choice}")
    if labels:
        lines.append(f"Answer with exactly one letter only: {', '.join(labels)}.")
    return "\n".join(lines)


def _stable_sample_id(video_key: str, question: str, choices: list[str], answer_idx: int) -> str:
    payload = {
        "video_key": str(video_key),
        "question": str(question),
        "choices": [str(c) for c in choices],
        "answer_idx": int(answer_idx),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _normalize_answer_idx(raw_answer: Any, num_choices: int) -> int | None:
    if raw_answer is None:
        return None
    text = str(raw_answer).strip()
    if not text or text.lower() == "none":
        return None
    m = re.search(r"([A-E])", text.upper())
    if m:
        idx = ord(m.group(1)) - ord("A")
        return idx if 0 <= idx < num_choices else None
    try:
        idx = int(float(text))
    except Exception:
        return None
    if 0 <= idx < num_choices:
        return idx
    if 1 <= idx <= num_choices:
        return idx - 1
    return None


def _resolve_video_path(video_root: str, rel_video: str) -> str | None:
    candidates = []
    if os.path.isabs(rel_video):
        candidates.append(rel_video)
    else:
        norm_rel = rel_video.lstrip("./")
        candidates.extend(
            [
                os.path.join(video_root, norm_rel),
                os.path.join(video_root, "all_video", norm_rel),
                os.path.join(video_root, "train_video", "all_video", norm_rel),
                os.path.join(video_root, "test_video", "all_video", norm_rel),
                os.path.join(video_root, "videos", "videos", os.path.basename(norm_rel)),
            ]
        )
        if norm_rel.startswith("all_video/"):
            suffix = norm_rel[len("all_video/") :]
            candidates.extend(
                [
                    os.path.join(video_root, "train_video", norm_rel),
                    os.path.join(video_root, "test_video", norm_rel),
                    os.path.join(video_root, "train_video", "all_video", suffix),
                    os.path.join(video_root, "test_video", "all_video", suffix),
                ]
            )
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _to_mapping(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config
    if DictConfig is not None and isinstance(config, DictConfig) and OmegaConf is not None:
        container = OmegaConf.to_container(config, resolve=False)
        if isinstance(container, Mapping):
            return container
    return {}


class LocalMCDataset(Dataset):
    """Load local JSON multiple-choice video QA files for REVISE RL."""

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
        config_map = _to_mapping(config)

        if isinstance(data_files, (list, tuple)):
            files = [str(x) for x in data_files]
        else:
            files = [str(data_files)]

        cfg_max_samples = None
        if config_map:
            cfg_max_samples = config_map.get("max_samples")
        if cfg_max_samples is not None:
            max_samples = int(cfg_max_samples)

        local_cfg = config_map.get("local_mc", {}) if config_map else {}
        video_root = str(local_cfg.get("video_root") or "").strip()
        dataset_name = str(local_cfg.get("dataset_name") or "local_mc").strip() or "local_mc"

        rows: list[dict[str, Any]] = []
        for path in files:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise TypeError(f"Expected list in {path}, got {type(data)}")
            rows.extend(x for x in data if isinstance(x, dict))

        if max_samples and max_samples > 0 and max_samples < len(rows):
            import random

            rng = random.Random(42)
            rng.shuffle(rows)
            rows = rows[:max_samples]

        self.dataset_name = dataset_name
        self.samples: list[dict[str, Any]] = []
        for row in rows:
            question = str(row.get("question") or "").strip()
            choices = row.get("options") or row.get("choices") or []
            if not question or not isinstance(choices, list) or len(choices) < 2:
                continue
            choices = [str(x).strip() for x in choices]
            answer_idx = _normalize_answer_idx(row.get("correct_answer") or row.get("answer"), len(choices))
            if answer_idx is None:
                continue

            rel_video = str(row.get("video_path") or row.get("video") or row.get("path") or "").strip()
            if not rel_video:
                continue
            video_path = _resolve_video_path(video_root, rel_video)
            if not video_path:
                continue

            sample_id = _stable_sample_id(rel_video, question, choices, answer_idx)
            self.samples.append(
                {
                    "sample_id": sample_id,
                    "qid": str(row.get("qid") or row.get("id") or row.get("question_idx") or ""),
                    "video_path": video_path,
                    "video_key": rel_video,
                    "question": question,
                    "choices": choices,
                    "answer_idx": answer_idx,
                    "type": str(row.get("task") or row.get("type") or ""),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        prompt_text = _format_question(sample["question"], sample["choices"])
        raw_prompt = [{"role": "user", "content": prompt_text}]

        ground_truth = {
            "answer_idx": sample["answer_idx"],
            "answer_text": sample["choices"][sample["answer_idx"]],
            "choices": sample["choices"],
            "qid": sample["qid"],
            "type": sample["type"],
            "sample_id": sample["sample_id"],
        }

        extra_info = {
            "sample_id": sample["sample_id"],
            "video_id": Path(sample["video_key"]).stem,
            "video_key": sample["video_key"],
            "video_path": sample["video_path"],
            "frame_count": 0,
            "question": sample["question"],
            "choices": sample["choices"],
            "type": sample["type"],
        }

        return {
            "raw_prompt": raw_prompt,
            "data_source": f"revise_{self.dataset_name}",
            "reward_model": {"ground_truth": ground_truth},
            "extra_info": extra_info,
            "agent_name": "revise_agent",
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "index": idx,
        }
