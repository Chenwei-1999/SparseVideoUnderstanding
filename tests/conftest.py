"""Pytest bootstrap: make the repo root importable.

The ``verl`` distribution is normally installed editable (``pip install -e .``),
which puts the repo root on ``sys.path`` so ``import revise.*`` and ``import
verl.*`` resolve. On shared clusters that editable install can be stale or
absent (e.g. the metadata points at a path that no longer exists, or the conda
env is root-owned and cannot be re-installed). In that situation ``pytest
tests/`` would fail at collection with ``ModuleNotFoundError: No module named
'revise'`` unless the caller remembers to prefix ``PYTHONPATH=.``.

Prepending the repo root here makes the suite self-contained and invocation-dir
independent, without mutating the (possibly shared) Python environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
