"""Video-MME dataset loading and task adapter."""

from __future__ import annotations

from typing import Any

from revise.benchmarks.videomme_lvbench_vllm import MCVideoSample, VideoMMEDataset
from revise.benchmarks.videomme_lvbench_vllm import _load_videomme_samples as load_samples

__all__ = ["MCVideoSample", "VideoMMEDataset", "load_hf_samples", "load_samples"]


def load_hf_samples(split: str) -> list[Any]:
    from revise.datasets.lvbench import load_videomme_hf_samples

    return load_videomme_hf_samples(split)
