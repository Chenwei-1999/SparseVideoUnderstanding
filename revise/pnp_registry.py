"""Registry for plug-and-play dataset and backend adapters."""

from __future__ import annotations

from typing import Any

from revise.oneshot_local_mc_vllm import LocalMCDataset
from revise.plug_and_play_egoschema_vllm import EgoSchemaDataset
from revise.plug_and_play_lvbench_hf import HFInProcessBackend, LVBenchHFDataset
from revise.plug_and_play_nextqa_vllm import NextQADataset
from revise.plug_and_play_videomme_lvbench_vllm import LVBenchDataset, VideoMMEDataset
from revise.plug_and_play_videomme_lvbench_vllm import VllmHttpBackend


DATASETS: dict[str, type[Any]] = {
    "nextqa": NextQADataset,
    "egoschema": EgoSchemaDataset,
    "videoespresso": EgoSchemaDataset,
    "videomme": VideoMMEDataset,
    "lvbench": LVBenchDataset,
    "lvbench_hf": LVBenchHFDataset,
    "local_mc": LocalMCDataset,
}

BACKENDS: dict[str, type[Any]] = {
    "vllm_http": VllmHttpBackend,
    "hf_inprocess": HFInProcessBackend,
}


def resolve(kind: str, name: str) -> type[Any]:
    """Resolve a registered adapter class by kind and name."""
    registries = {
        "dataset": DATASETS,
        "datasets": DATASETS,
        "backend": BACKENDS,
        "backends": BACKENDS,
    }
    try:
        registry = registries[str(kind).strip().lower()]
    except KeyError as exc:
        raise KeyError(f"Unknown registry kind {kind!r}; expected one of {sorted(registries)}") from exc
    key = str(name).strip().lower()
    try:
        return registry[key]
    except KeyError as exc:
        raise KeyError(f"Unknown {kind} {name!r}; expected one of {sorted(registry)}") from exc
