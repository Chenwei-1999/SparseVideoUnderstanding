"""End-to-end loop-trajectory golden master for the EgoSchema agent loop.

This is the safety net for migrating ``plug_and_play_egoschema_vllm.main`` onto
the shared PnP engine. It drives the real CLI entry point with a mocked transport
and tiny samples, then freezes the script's own ``--summary-json`` ``results``
block. The goal is behavior preservation: if a later refactor changes these
counters, the loop trajectory changed.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import revise.benchmarks.egoschema_vllm as egoschema

SELECT_ROUND = (
    "<think>reasoning</think>\n"
    "<summarize>P: prior context; O: observed evidence; H: current belief; "
    "U: detail unclear; R: request more</summarize>\n"
    "<select>3, 5</select>"
)
UNSTRUCTURED_SELECT_ROUND = (
    "<think>reasoning</think>\n"
    "<summarize>The visible evidence suggests the person is preparing food, "
    "but one detail remains unclear.</summarize>\n"
    "<select>3, 5</select>"
)
ANSWER_A = "<think>done</think>\n<answer>A</answer>"
NO_TAGS = "no tags here"

_VOLATILE = ("elapsed_s", "prompt_log_bytes")


def _two_samples():
    return [
        egoschema.EgoSchemaSample("s1", "q1", "v1.mp4", "What?", ["a", "b", "c", "d", "e"], 0, 10),
        egoschema.EgoSchemaSample("s2", "q2", "v2.mp4", "Why?", ["a", "b", "c", "d", "e"], 1, 10),
    ]


def _run_loop(scripted_outputs, extra_argv=None, *, empty_frames=False):
    outputs = iter(scripted_outputs)

    def fake_chat_completions(*_args, **_kwargs):
        try:
            return next(outputs)
        except StopIteration:
            return ANSWER_A

    def fake_extract_frames(_video_path, indices):
        if empty_frames:
            return []
        return [Image.new("RGB", (2, 2), color=(int(i) % 255, 0, 0)) for i in indices]

    with tempfile.TemporaryDirectory() as d:
        summary_path = os.path.join(d, "summary.json")
        log_path = os.path.join(d, "log.jsonl")
        argv = [
            "prog",
            "--model-path", "m", "--video-root", "vr", "--json", "samples.json",
            "--max-rounds", "3", "--max-frames", "2", "--seed", "0",
            "--progress-interval", "999", "--max-retries-per-round", "0",
            "--no-resume-from-log",
            "--summary-json", summary_path, "--log-jsonl", log_path,
        ]
        if extra_argv:
            argv += extra_argv
        with patch.object(sys, "argv", argv), \
                patch.object(egoschema, "_load_egoschema_samples", return_value=_two_samples()), \
                patch.object(egoschema, "get_model_id", return_value="test-model"), \
                patch.object(egoschema, "maybe_init_wandb", return_value=None), \
                patch.object(egoschema, "wandb_log", return_value=None, create=True), \
                patch.object(egoschema, "extract_frames", side_effect=fake_extract_frames), \
                patch.object(egoschema, "_call_chat_completions", side_effect=fake_chat_completions):
            rc = egoschema.main()
        results = json.load(open(summary_path, encoding="utf-8"))["results"]
    for key in _VOLATILE:
        results.pop(key, None)
    return rc, results


class EgoSchemaLoopTrajectoryTest(unittest.TestCase):
    def test_strict_default_trajectory_golden(self):
        # Sample 1 selects frames, then answers correctly. Sample 2 emits an
        # illegal untagged response and strict-actions terminates the sample.
        rc, results = _run_loop([SELECT_ROUND, ANSWER_A, NO_TAGS])
        self.assertEqual(rc, 0)
        self.assertEqual(results, {
            "samples": 2,
            "accuracy": 0.5,
            "avg_rounds": 1.0,
            "failed": 0,
            "invalid_outputs": 1,
            "total_retries": 0,
            "total_model_calls": 3,
            "fallback_frames_used": 0,
        })

    def test_unstructured_summary_ablation_trajectory_golden(self):
        # With --ablate-structured-summary, an unstructured but meaningful
        # <summarize> is valid on the Select round.
        rc, results = _run_loop(
            [UNSTRUCTURED_SELECT_ROUND, ANSWER_A, NO_TAGS],
            extra_argv=["--ablate-structured-summary"],
        )
        self.assertEqual(rc, 0)
        self.assertEqual(results, {
            "samples": 2,
            "accuracy": 0.5,
            "avg_rounds": 1.0,
            "failed": 0,
            "invalid_outputs": 1,
            "total_retries": 0,
            "total_model_calls": 3,
            "fallback_frames_used": 0,
        })

    def test_empty_frame_extraction_fails_without_model_calls_golden(self):
        rc, results = _run_loop([ANSWER_A, ANSWER_A], empty_frames=True)
        self.assertEqual(rc, 2)
        self.assertEqual(results, {
            "samples": 2,
            "accuracy": 0.0,
            "avg_rounds": 0.0,
            "failed": 2,
            "invalid_outputs": 0,
            "total_retries": 0,
            "total_model_calls": 0,
            "fallback_frames_used": 0,
        })


if __name__ == "__main__":
    unittest.main()
