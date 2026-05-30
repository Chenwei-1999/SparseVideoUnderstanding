"""Golden trajectory test for the Video-MME/LVBench vLLM plug-and-play loop.

This is the pre-migration oracle for moving
``plug_and_play_videomme_lvbench_vllm.main`` onto the shared PnP engine. It
drives the real CLI entry point with mocked dataset/video/model boundaries and
freezes the script's own ``--summary-json`` counters.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import revise.plug_and_play_videomme_lvbench_vllm as videomme_lvbench


SELECT_ROUND = (
    "<think>need more evidence</think>\n"
    "<summarize>P: prior context; O: observed evidence; H: current belief; "
    "U: detail unclear; R: request more</summarize>\n"
    "<select>1, 2</select>"
)
ANSWER_A = "<think>enough evidence</think>\n<answer>A</answer>"
NO_TAGS = "no tags here"
BARE_ANSWER_B = (
    "<summarize>P: seen; O: enough evidence; H: answer is B; "
    "U: none; R: answer now</summarize>\nB"
)
INVALID_CANDIDATE_SELECT = (
    "<think>need a different moment</think>\n"
    "<summarize>P: prior context; O: observed evidence; H: uncertain; "
    "U: need another candidate; R: request candidate</summarize>\n"
    "<select>999</select>"
)

_VOLATILE = ("elapsed_s", "prompt_log_bytes")


def _samples(dataset_name):
    if dataset_name == "videomme":
        return [
            videomme_lvbench.MCVideoSample(
                dataset="videomme",
                uid="q1",
                video_key="v1.mp4",
                video_url="https://example.test/v1",
                question="What object appears?",
                options=["red cup", "blue book", "green bag"],
                answer_letter="A",
            ),
            videomme_lvbench.MCVideoSample(
                dataset="videomme",
                uid="q2",
                video_key="v2.mp4",
                video_url="https://example.test/v2",
                question="What happens next?",
                options=["open", "close", "turn"],
                answer_letter="B",
            ),
        ]
    return [
        videomme_lvbench.MCVideoSample(
            dataset="lvbench",
            uid="q1",
            video_key="lv1.mp4",
            video_url="https://example.test/lv1",
            question="What object appears?",
            options=["red cup", "blue book", "green bag"],
            answer_letter="A",
            time_reference="00:02-00:05",
        ),
        videomme_lvbench.MCVideoSample(
            dataset="lvbench",
            uid="q2",
            video_key="lv2.mp4",
            video_url="https://example.test/lv2",
            question="What happens next?",
            options=["open", "close", "turn"],
            answer_letter="B",
            time_reference="00:02-00:05",
        ),
    ]


def _materialize_cached_videos(cache_root, dataset_name, samples):
    dataset_cache = Path(cache_root) / dataset_name
    dataset_cache.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        video_path = dataset_cache / sample.video_key
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"cached-video")


def _run_loop(dataset_name, scripted_outputs, *, sample_count=None):
    outputs = iter(scripted_outputs)
    samples = _samples(dataset_name)
    if sample_count is not None:
        samples = samples[:sample_count]

    def fake_chat_once(*_args, **_kwargs):
        try:
            return next(outputs)
        except StopIteration:
            return ANSWER_A

    def fake_extract_frames_1fps(_video_path, indices):
        return [Image.new("RGB", (2, 2), color=(int(i) % 255, 0, 0)) for i in indices]

    with tempfile.TemporaryDirectory() as d:
        cache_root = os.path.join(d, "cache")
        _materialize_cached_videos(cache_root, dataset_name, samples)
        summary_path = os.path.join(d, "summary.json")
        log_path = os.path.join(d, "log.jsonl")
        argv = [
            "prog",
            "--dataset", dataset_name,
            "--model-path", "m",
            "--video-cache-dir", cache_root,
            "--port", "18080",
            "--max-rounds", "3",
            "--max-frames-per-round", "2",
            "--candidate-k", "4",
            "--max-retries-per-round", "0",
            "--summary-json", summary_path,
            "--log-jsonl", log_path,
        ]
        with patch.object(sys, "argv", argv), \
                patch.object(videomme_lvbench, "_load_videomme_samples", return_value=samples), \
                patch.object(videomme_lvbench, "_load_lvbench_samples", return_value=samples), \
                patch.object(videomme_lvbench, "_download_youtube") as download_youtube, \
                patch.object(videomme_lvbench, "extract_video_info", return_value=(8, 1.0)), \
                patch.object(videomme_lvbench, "extract_frames_1fps", side_effect=fake_extract_frames_1fps), \
                patch.object(videomme_lvbench, "get_model_id", return_value="test-model"), \
                patch.object(videomme_lvbench, "maybe_init_wandb", return_value=None), \
                patch.object(videomme_lvbench, "wandb_log", return_value=None), \
                patch.object(videomme_lvbench, "_chat_once", side_effect=fake_chat_once):
            rc = videomme_lvbench.main()
        download_youtube.assert_not_called()
        results = json.load(open(summary_path, encoding="utf-8"))["results"]
        records = []
        if os.path.exists(log_path):
            with open(log_path, encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
    for key in _VOLATILE:
        results.pop(key, None)
    return rc, results, records


class VideoMMeLvbenchLoopTrajectoryTest(unittest.TestCase):
    def test_videomme_select_answer_then_invalid_bare_answer_golden(self):
        rc, results, records = _run_loop("videomme", [SELECT_ROUND, ANSWER_A, NO_TAGS, BARE_ANSWER_B])
        self.assertEqual(rc, 0)
        self.assertEqual(results, {
            "samples": 2,
            "answered": 2,
            "correct": 2,
            "accuracy": 1.0,
            "avg_rounds": 2.0,
            "avg_effective_rounds": 0.5,
            "avg_frames_used": 3.0,
            "avg_frames_used_all": 3.0,
            "failed": 0,
            "prompt_log_lines": 4,
            "invalid_outputs": 1,
            "invalid_action_terminated": 0,
            "total_retries": 1,
            "total_model_calls": 4,
            "think_present_rounds": 2,
            "missing_summary_rounds": 2,
        })
        self.assertEqual(records[1]["answer_letter"], "A")
        self.assertEqual(records[3]["answer_letter"], "B")

    def test_lvbench_time_window_select_answer_then_invalid_bare_answer_golden(self):
        rc, results, records = _run_loop("lvbench", [SELECT_ROUND, ANSWER_A, NO_TAGS, BARE_ANSWER_B])
        self.assertEqual(rc, 0)
        self.assertEqual(results, {
            "samples": 2,
            "answered": 2,
            "correct": 2,
            "accuracy": 1.0,
            "avg_rounds": 2.0,
            "avg_effective_rounds": 0.5,
            "avg_frames_used": 3.0,
            "avg_frames_used_all": 3.0,
            "failed": 0,
            "prompt_log_lines": 4,
            "invalid_outputs": 1,
            "invalid_action_terminated": 0,
            "total_retries": 1,
            "total_model_calls": 4,
            "think_present_rounds": 2,
            "missing_summary_rounds": 2,
        })
        self.assertEqual(records[1]["answer_letter"], "A")
        self.assertEqual(records[3]["answer_letter"], "B")

    def test_invalid_candidate_ids_fall_back_to_candidate_frames_golden(self):
        rc, results, records = _run_loop(
            "videomme",
            [INVALID_CANDIDATE_SELECT, ANSWER_A],
            sample_count=1,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(results, {
            "samples": 1,
            "answered": 1,
            "correct": 1,
            "accuracy": 1.0,
            "avg_rounds": 2.0,
            "avg_effective_rounds": 0.0,
            "avg_frames_used": 4.0,
            "avg_frames_used_all": 4.0,
            "failed": 0,
            "prompt_log_lines": 2,
            "invalid_outputs": 1,
            "invalid_action_terminated": 0,
            "total_retries": 0,
            "total_model_calls": 2,
            "think_present_rounds": 2,
            "missing_summary_rounds": 1,
        })
        self.assertEqual(records[1]["answer_letter"], "A")
        self.assertEqual(len(records[1]["seen_frames"]), 4)


if __name__ == "__main__":
    unittest.main()
