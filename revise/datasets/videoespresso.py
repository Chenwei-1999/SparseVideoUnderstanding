"""VideoEspresso dataset adapter.

VideoEspresso uses the same JSON multiple-choice sample shape as the
EgoSchema-style local evaluator. Keeping this module separate prevents callers
from encoding the dataset identity as an inference backend detail.
"""

from __future__ import annotations

from revise.datasets.egoschema import EgoSchemaDataset, EgoSchemaSample, load_samples

VideoEspressoDataset = EgoSchemaDataset
VideoEspressoSample = EgoSchemaSample

__all__ = ["VideoEspressoDataset", "VideoEspressoSample", "load_samples"]
