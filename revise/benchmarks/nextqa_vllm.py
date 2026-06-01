#!/usr/bin/env python3
"""Compatibility entrypoint for NExT-QA REVISE PnP.

The implementation lives in ``revise.pnp_cli`` plus the dataset adapter in
``revise.datasets.nextqa``. Keeping this thin wrapper preserves older commands
without maintaining a second agent loop.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revise.datasets.nextqa import (  # noqa: E402
    _TIMELINE_CACHE,
    DEFAULT_SYSTEM_PROMPT,
    NextQADataset,
    NextQASample,
    _build_user_text,
    _format_question,
    _retry_feedback_text,
    _sample_unseen_frames,
    extract_frames_1fps,
    extract_video_info,
    load_progress_from_log,
    load_samples,
)
from revise.pnp.utils import build_vllm_serve_command, open_server_log_streams  # noqa: E402
from revise.pnp.utils import chat_once as _chat_once
from revise.pnp_cli import main as pnp_cli_main  # noqa: E402

_load_nextqa_samples = load_samples
_load_progress_from_log = load_progress_from_log


def _start_vllm_server(args: argparse.Namespace) -> subprocess.Popen[str]:
    image_limit = max(
        int(getattr(args, "max_frames_per_round", 1)),
        int(getattr(args, "caption_gen_max_frames", 1)),
    )
    cmd, env = build_vllm_serve_command(args, image_limit=image_limit, cuda_visible_default="0,1,2,3")
    server_stdout, server_stderr = open_server_log_streams(getattr(args, "server_log", None))
    return subprocess.Popen(cmd, env=env, stdout=server_stdout, stderr=server_stderr)


def main(argv: Sequence[str] | None = None) -> int | None:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    return pnp_cli_main(
        [
            "--dataset",
            "nextqa",
            "--backend",
            "vllm_http",
            "--setting",
            "multi_round_pnp",
            *forwarded,
        ]
    )


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "NextQADataset",
    "NextQASample",
    "_TIMELINE_CACHE",
    "_build_user_text",
    "_format_question",
    "_chat_once",
    "_load_nextqa_samples",
    "_load_progress_from_log",
    "_retry_feedback_text",
    "_sample_unseen_frames",
    "_start_vllm_server",
    "extract_frames_1fps",
    "extract_video_info",
    "load_progress_from_log",
    "load_samples",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
