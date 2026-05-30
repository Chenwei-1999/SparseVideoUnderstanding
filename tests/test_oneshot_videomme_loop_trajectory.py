"""Golden trajectory test for the one-shot Video-MME/LVBench vLLM baseline.

This is the pre-migration oracle for moving
``oneshot_videomme_lvbench_vllm.main`` onto the shared PnP engine. It drives the
real CLI entry point with mocked dataset/video/model boundaries and freezes the
script's own ``--summary-json`` counter dict.

The single-round baseline samples frames once, performs one model chat per
sample, parses the answer letter, and scores. There is no select/summarize/retry
loop. The model boundaries are mocked in BOTH the ``oneshot_*`` namespace (the
pre-migration standalone call path) and the ``plug_and_play_*`` adapter/backend
namespace (the post-migration delegated call path), so this one test stays
byte-identical across the migration.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import revise.oneshot_videomme_lvbench_vllm as oneshot
import revise.plug_and_play_videomme_lvbench_vllm as videomme_lvbench


_VOLATILE = ("elapsed_s", "prompt_log_jsonl")


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
            return "A"

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
            "--max-frames", "2",
            "--summary-json", summary_path,
            "--log-jsonl", log_path,
        ]
        # The transport-ish seams live in two namespaces: the pre-migration
        # standalone oneshot call path (oneshot._*), and the post-migration
        # adapter/backend call path (plug_and_play.*). We patch both so this
        # test drives identically before and after the migration. The loaders
        # and get_model_id stay bare-name in oneshot.main() so they are patched
        # only on the oneshot module; _download_youtube is patched on
        # plug_and_play and asserted never called.
        with patch.object(sys, "argv", argv), \
                patch.object(oneshot, "_load_videomme_samples", return_value=samples), \
                patch.object(oneshot, "_load_lvbench_samples", return_value=samples), \
                patch.object(oneshot, "_get_model_id", return_value="test-model"), \
                patch.object(oneshot, "_extract_video_info", return_value=(8, 1.0)), \
                patch.object(oneshot, "_extract_frames_1fps", side_effect=fake_extract_frames_1fps), \
                patch.object(oneshot, "_chat_once", side_effect=fake_chat_once), \
                patch.object(videomme_lvbench, "_load_videomme_samples", return_value=samples), \
                patch.object(videomme_lvbench, "_load_lvbench_samples", return_value=samples), \
                patch.object(videomme_lvbench, "get_model_id", return_value="test-model"), \
                patch.object(videomme_lvbench, "extract_video_info", return_value=(8, 1.0)), \
                patch.object(videomme_lvbench, "extract_frames_1fps", side_effect=fake_extract_frames_1fps), \
                patch.object(videomme_lvbench, "_chat_once", side_effect=fake_chat_once), \
                patch.object(videomme_lvbench, "_download_youtube") as download_youtube:
            rc = oneshot.main()
        download_youtube.assert_not_called()
        results = json.load(open(summary_path, encoding="utf-8"))
        records = []
        if os.path.exists(log_path):
            with open(log_path, encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
    for key in _VOLATILE:
        results.pop(key, None)
    return rc, results, records


class OneshotVideoMMeLvbenchLoopTrajectoryTest(unittest.TestCase):
    def test_videomme_oneshot_answers_golden(self):
        rc, results, records = _run_loop("videomme", ["A", "B"])
        self.assertEqual(rc, 0)
        self.assertEqual(results, {
            "samples": 2,
            "answered": 2,
            "correct": 2,
            "accuracy": 1.0,
            "failed": 0,
            "total_model_calls": 2,
            "cached_only": False,
            "allow_missing_cached_videos": False,
            "cache_filter_total_samples": 2,
            "cache_filter_missing_videos": 0,
            "cache_filter_missing_examples": [],
        })
        self.assertEqual([r["pred_answer"] for r in records], ["A", "B"])
        self.assertEqual([r["correct"] for r in records], [True, True])

    def test_videomme_oneshot_one_wrong_golden(self):
        rc, results, records = _run_loop("videomme", ["A", "A"])
        self.assertEqual(rc, 0)
        self.assertEqual(results, {
            "samples": 2,
            "answered": 2,
            "correct": 1,
            "accuracy": 0.5,
            "failed": 0,
            "total_model_calls": 2,
            "cached_only": False,
            "allow_missing_cached_videos": False,
            "cache_filter_total_samples": 2,
            "cache_filter_missing_videos": 0,
            "cache_filter_missing_examples": [],
        })
        self.assertEqual([r["pred_answer"] for r in records], ["A", "A"])
        self.assertEqual([r["correct"] for r in records], [True, False])

    def test_lvbench_oneshot_time_window_answers_golden(self):
        rc, results, records = _run_loop("lvbench", ["A", "B"])
        self.assertEqual(rc, 0)
        self.assertEqual(results, {
            "samples": 2,
            "answered": 2,
            "correct": 2,
            "accuracy": 1.0,
            "failed": 0,
            "total_model_calls": 2,
            "cached_only": False,
            "allow_missing_cached_videos": False,
            "cache_filter_total_samples": 2,
            "cache_filter_missing_videos": 0,
            "cache_filter_missing_examples": [],
        })
        self.assertEqual([r["pred_answer"] for r in records], ["A", "B"])
        self.assertEqual([r["correct"] for r in records], [True, True])

    def test_videomme_oneshot_all_correct_golden(self):
        rc, results, records = _run_loop("videomme", ["A", "B"], sample_count=1)
        self.assertEqual(rc, 0)
        self.assertEqual(results, {
            "samples": 1,
            "answered": 1,
            "correct": 1,
            "accuracy": 1.0,
            "failed": 0,
            "total_model_calls": 1,
            "cached_only": False,
            "allow_missing_cached_videos": False,
            "cache_filter_total_samples": 1,
            "cache_filter_missing_videos": 0,
            "cache_filter_missing_examples": [],
        })
        self.assertEqual(records[0]["pred_answer"], "A")
        self.assertEqual(records[0]["correct"], True)


if __name__ == "__main__":
    unittest.main()
