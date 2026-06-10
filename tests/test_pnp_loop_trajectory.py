"""End-to-end loop-trajectory golden master for the NExT-QA agent loop.

This is the safety net that guards refactors of the ~700-line agent loop inside
``plug_and_play_nextqa_vllm.main``. Unit-testing extracted *pure* helpers is not
enough: the bugs a loop refactor produces live in the control flow itself —
``break`` vs ``continue`` (how many model calls happen), the seeded RNG stream
(which frames get sampled on fallback), and side-effect timing (what gets
logged). So this test drives the *real* loop end-to-end with a mocked transport
(scripted ``_chat_once`` outputs, stubbed frame extraction, fixed seed) and
freezes the program's own observable output — the ``--summary-json`` ``results``
block, which tallies model calls, retries, fallbacks, invalids, effective
rounds, and accuracy.

If a future refactor changes any of these counters, the trajectory changed, and
the change is NOT behavior-preserving. Run under the runtime env::

    python -m unittest tests.test_pnp_loop_trajectory -v
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import revise.backends.vllm_http as vllm_http
import revise.benchmarks.nextqa_vllm as nextqa
import revise.datasets.nextqa as nextqa_dataset
import revise.pnp_cli as pnp_cli

# A valid Select round: <think> + a P/O/H/U/R <summarize> + a <select> request.
SELECT_ROUND = (
    "<think>reasoning</think>\n"
    "<summarize>P: prior context; O: observed evidence; H: current belief; "
    "U: detail unclear; R: request more</summarize>\n"
    "<select>3, 5</select>"
)
ANSWER_A = "<think>done</think>\n<answer>A</answer>"
NO_TAGS = "no tags here"

# Volatile fields excluded from the frozen comparison (wall-clock / byte size of
# the log are not part of the trajectory).
_VOLATILE = ("elapsed_s", "prompt_log_bytes")


def _fake_frames(_video_path, indices, *args, **kwargs):
    from PIL import Image

    return [Image.new("RGB", (2, 2)) for _ in indices]


def _two_samples():
    return [
        nextqa.NextQASample("s1", "q1", "v1", "v1.mp4", "What?",
                            ["a", "b", "c", "d", "e"], 0, 10),
        nextqa.NextQASample("s2", "q2", "v2", "v2.mp4", "Why?",
                            ["a", "b", "c", "d", "e"], 1, 10),
    ]


def _run_loop(scripted_outputs, extra_argv=None):
    """Drive nextqa.main() end-to-end with a mocked transport; return results dict."""
    outputs = iter(scripted_outputs)

    def fake_chat_once(*_args, **_kwargs):
        try:
            return next(outputs)
        except StopIteration:  # safety: should not be reached by these scenarios
            return ANSWER_A

    with tempfile.TemporaryDirectory() as d:
        summary_path = os.path.join(d, "summary.json")
        log_path = os.path.join(d, "log.jsonl")
        argv = [
            "prog",
            "--model-path", "m", "--video-root", "vr", "--map-json", "mj", "--csv", "c",
            "--max-rounds", "3", "--max-frames", "2", "--seed", "0",
            "--progress-interval", "0", "--no-resume-from-log",
            "--summary-json", summary_path, "--log-jsonl", log_path,
        ]
        if extra_argv:
            argv += extra_argv
        # The loop now runs through the shared engine (revise.pnp_cli + engine),
        # so the seams live in the dataset adapter and the vLLM HTTP backend
        # rather than on the (compatibility) launcher module itself.
        with patch.object(sys, "argv", argv), \
                patch.object(nextqa_dataset, "load_samples", return_value=_two_samples()), \
                patch.object(pnp_cli, "get_model_id", return_value="test-model"), \
                patch.object(vllm_http, "get_model_id", return_value="test-model"), \
                patch.object(pnp_cli, "maybe_init_wandb", return_value=None), \
                patch.object(pnp_cli, "wandb_log", return_value=None), \
                patch.object(nextqa_dataset, "extract_frames_1fps", side_effect=_fake_frames), \
                patch.object(nextqa_dataset, "extract_video_info", return_value=(10, 1.0)), \
                patch.object(nextqa_dataset, "_get_video_fps", return_value=1.0), \
                patch.object(vllm_http, "chat_once", side_effect=fake_chat_once):
            rc = nextqa.main()
        results = json.load(open(summary_path, encoding="utf-8"))["results"]
    for k in _VOLATILE:
        results.pop(k, None)
    return rc, results


class NextqaLoopTrajectoryTest(unittest.TestCase):
    def test_strict_default_trajectory_golden(self):
        # Default strict-actions, no retries. Sample 1 answers on round 2;
        # sample 2 re-requests already-seen frames until the request is empty,
        # which strict-actions terminates -> invalid_action_terminated == 1.
        rc, results = _run_loop([SELECT_ROUND, ANSWER_A, SELECT_ROUND, SELECT_ROUND, ANSWER_A])
        self.assertEqual(rc, 0)
        self.assertEqual(results, {
            "samples": 2,
            "correct": 1,
            "accuracy": 0.5,
            "total_rounds": 4,
            "avg_rounds": 2.0,
            "total_frames_used": 8,
            "avg_frames_used": 4.0,
            "total_effective_rounds": 2,
            "avg_effective_rounds": 1.0,
            "failed": 0,
            "prompt_log_lines": 4,
            "invalid_outputs": 1,
            "invalid_action_terminated": 1,
            "total_retries": 0,
            "total_model_calls": 4,
            "fallback_frames_used": 0,
        })

    def test_retry_path_trajectory_golden(self):
        # max-retries=1, strict-actions off. Sample 1 round 1 emits untagged
        # output -> one retry -> valid Select; the retry is counted and the
        # sample is not hard-terminated.
        rc, results = _run_loop(
            [NO_TAGS, SELECT_ROUND, ANSWER_A, SELECT_ROUND, ANSWER_A],
            extra_argv=["--max-retries-per-round", "1", "--no-strict-actions"],
        )
        self.assertEqual(rc, 0)
        self.assertEqual(results, {
            "samples": 2,
            "correct": 1,
            "accuracy": 0.5,
            "total_rounds": 4,
            "avg_rounds": 2.0,
            "total_frames_used": 8,
            "avg_frames_used": 4.0,
            "total_effective_rounds": 2,
            "avg_effective_rounds": 1.0,
            "failed": 0,
            "prompt_log_lines": 5,
            "invalid_outputs": 1,
            "invalid_action_terminated": 0,
            "total_retries": 1,
            "total_model_calls": 5,
            "fallback_frames_used": 0,
        })


if __name__ == "__main__":
    unittest.main()
