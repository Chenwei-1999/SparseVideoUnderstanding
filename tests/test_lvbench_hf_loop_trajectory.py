"""Golden trajectory test for the LVBench HF plug-and-play loop.

This is the pre-migration oracle for moving ``plug_and_play_lvbench_hf.main``
onto the shared PnP engine. It drives the real CLI entry point with mocked
dataset/video/model boundaries and freezes the script's own ``--summary-json``
counters.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

import revise.plug_and_play_lvbench_hf as lvbench_hf


SELECT_ROUND = (
    "<think>need more evidence</think>\n"
    "<summarize>P: prior context; O: observed evidence; H: current belief; "
    "U: detail unclear; R: request more</summarize>\n"
    "<select>1, 2</select>"
)
ANSWER_A = "<think>enough evidence</think>\n<answer>A</answer>"
ANSWER_B = "<think>enough evidence</think>\n<answer>B</answer>"
NO_TAGS = "no tags here"
INVALID_ANSWER = "<think>bad final</think>\n<answer>Z</answer>"

_VOLATILE = ("ts",)


def _samples():
    return [
        lvbench_hf.MCVideoSample(
            dataset="lvbench",
            uid="q1",
            video_key="lv1.mp4",
            question="What object appears?",
            options=["red cup", "blue book", "green bag"],
            answer_letter="A",
            time_reference="00:02-00:05",
            question_type="entity",
            video_type="movie",
        ),
        lvbench_hf.MCVideoSample(
            dataset="lvbench",
            uid="q2",
            video_key="lv2.mp4",
            question="What happens next?",
            options=["open", "close", "turn"],
            answer_letter="B",
            time_reference="00:02-00:05",
            question_type="action",
            video_type="movie",
        ),
    ]


def _materialize_cached_videos(cache_root, samples):
    dataset_cache = Path(cache_root) / "lvbench"
    dataset_cache.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        video_path = dataset_cache / sample.video_key
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"cached-video")


def _run_loop(scripted_outputs, *, sample_count=None, extra_argv=None):
    outputs = iter(scripted_outputs)
    samples = _samples()
    if sample_count is not None:
        samples = samples[:sample_count]

    def fake_chat_once_hf(*_args, **_kwargs):
        try:
            return next(outputs)
        except StopIteration:
            return ANSWER_A

    def fake_extract_frames_1fps(_video_path, indices):
        return [Image.new("RGB", (2, 2), color=(int(i) % 255, 0, 0)) for i in indices]

    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=32768))

    with tempfile.TemporaryDirectory() as d:
        cache_root = os.path.join(d, "cache")
        _materialize_cached_videos(cache_root, samples)
        summary_path = os.path.join(d, "summary.json")
        log_path = os.path.join(d, "log.jsonl")
        argv = [
            "prog",
            "--dataset", "lvbench",
            "--model-path", "m",
            "--video-cache-dir", cache_root,
            "--max-rounds", "3",
            "--max-frames-per-round", "2",
            "--candidate-k", "4",
            "--max-retries-per-round", "0",
            "--summary-json", summary_path,
            "--log-jsonl", log_path,
        ]
        if extra_argv:
            argv += extra_argv
        with patch.object(sys, "argv", argv), \
                patch.object(lvbench_hf, "_load_lvbench_samples", return_value=samples), \
                patch.object(lvbench_hf, "extract_video_info", return_value=(8, 1.0)), \
                patch.object(lvbench_hf, "extract_frames_1fps", side_effect=fake_extract_frames_1fps), \
                patch.object(lvbench_hf, "_load_model_and_processor", return_value=(model, object())), \
                patch.object(lvbench_hf, "maybe_init_wandb", return_value=None), \
                patch.object(lvbench_hf, "wandb_log", return_value=None), \
                patch.object(lvbench_hf, "_chat_once_hf", side_effect=fake_chat_once_hf):
            rc = lvbench_hf.main()
        summary = json.load(open(summary_path, encoding="utf-8"))
        records = []
        if os.path.exists(log_path):
            with open(log_path, encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
    for key in _VOLATILE:
        summary.pop(key, None)
    summary.pop("config", None)
    return rc, summary, records


class LVBenchHFLoopTrajectoryTest(unittest.TestCase):
    def test_select_answer_then_invalid_recovery_golden(self):
        rc, summary, records = _run_loop([SELECT_ROUND, ANSWER_A, NO_TAGS, ANSWER_B])
        self.assertIsNone(rc)
        self.assertEqual(summary, {
            "dataset": "lvbench",
            "split": "train",
            "model_path": "m",
            "num_samples": 2,
            "answered": 2,
            "correct": 2,
            "accuracy": 1.0,
            "failed": 0,
            "invalid_outputs": 1,
            "invalid_action_terminated": 0,
            "avg_rounds": 2.0,
            "avg_effective_rounds": 0.5,
            "avg_frames_used": 1.5,
            "think_present": 3,
            "total_model_calls": 4,
            "total_retries": 1,
            "max_len": 32768,
        })
        self.assertEqual(records[1]["answer_letter"], "A")
        self.assertEqual(records[3]["answer_letter"], "B")

    def test_invalid_forced_answer_trajectory_golden(self):
        rc, summary, records = _run_loop(
            [NO_TAGS, INVALID_ANSWER],
            sample_count=1,
            extra_argv=["--max-rounds", "1"],
        )
        self.assertIsNone(rc)
        self.assertEqual(summary, {
            "dataset": "lvbench",
            "split": "train",
            "model_path": "m",
            "num_samples": 1,
            "answered": 1,
            "correct": 0,
            "accuracy": 0.0,
            "failed": 0,
            "invalid_outputs": 1,
            "invalid_action_terminated": 1,
            "avg_rounds": 0.0,
            "avg_effective_rounds": 0.0,
            "avg_frames_used": 0.0,
            "think_present": 0,
            "total_model_calls": 2,
            "total_retries": 1,
            "max_len": 32768,
        })
        self.assertIsNone(records[0]["answer_letter"])


if __name__ == "__main__":
    unittest.main()
