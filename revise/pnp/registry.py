"""Lazy registry for plug-and-play dataset and backend adapters.

The shared ``revise.pnp`` package should not import concrete benchmark
launchers at module import time. Dataset adapters live under ``revise.datasets``
and inference runtimes live under ``revise.backends``. Keeping this registry
lazy preserves the dependency direction: core PnP orchestration knows adapter
names, but concrete adapters are loaded only at the CLI edge that asks for them.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AdapterSpec:
    module: str
    symbol: str


DATASETS: dict[str, AdapterSpec] = {
    "nextqa": AdapterSpec("revise.datasets.nextqa", "NextQADataset"),
    "egoschema": AdapterSpec("revise.datasets.egoschema", "EgoSchemaDataset"),
    "videoespresso": AdapterSpec("revise.datasets.videoespresso", "VideoEspressoDataset"),
    "videomme": AdapterSpec("revise.datasets.videomme", "VideoMMEDataset"),
    "lvbench": AdapterSpec("revise.datasets.lvbench", "LVBenchDataset"),
    "local_mc": AdapterSpec("revise.datasets.local_mc", "LocalMCDataset"),
}

BACKENDS: dict[str, AdapterSpec] = {
    "vllm_http": AdapterSpec("revise.backends.vllm_http", "VllmHttpBackend"),
    "hf_inprocess": AdapterSpec("revise.backends.hf_inprocess", "HFInProcessBackend"),
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
        spec = registry[key]
    except KeyError as exc:
        raise KeyError(f"Unknown {kind} {name!r}; expected one of {sorted(registry)}") from exc
    module = importlib.import_module(spec.module)
    return getattr(module, spec.symbol)
