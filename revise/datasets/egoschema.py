"""EgoSchema-style local multiple-choice dataset loading and task adapter."""

from __future__ import annotations

from revise.benchmarks.egoschema_vllm import EgoSchemaDataset, EgoSchemaSample
from revise.benchmarks.egoschema_vllm import _load_egoschema_hf_samples as load_hf_samples
from revise.benchmarks.egoschema_vllm import _load_egoschema_samples as load_samples

__all__ = ["EgoSchemaDataset", "EgoSchemaSample", "load_hf_samples", "load_samples"]
