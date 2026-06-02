"""Golden trajectory test for the one-shot LVBench HF (in-process) baseline.

This is the pre-migration oracle for moving ``oneshot_lvbench_hf.main`` onto the
shared PnP engine. It drives the real CLI entry point with mocked
dataset/video/model boundaries, a scripted in-process model, and freezes the
script's own ``--summary-json`` ``results`` counter dict.

The single-round baseline samples frames once, performs one ``model.generate``
per sample (in process, no vLLM server), decodes the answer letter, and scores.
There is no select/summarize/retry loop and no video download.

The model/processor boundary is faked so the SAME fake drives BOTH the
pre-migration inline ``model.generate`` path and the post-migration
``HFInProcessBackend.chat`` -> ``_chat_once_hf`` path: both go through
``apply_chat_template`` -> ``processor(...)`` -> ``model.generate`` ->
``batch_decode``. This keeps the test byte-identical across the migration.

NOTE (per the migration policy): the FROZEN contract is the ``--summary-json``
counter dict, NOT the per-sample JSONL record. The legacy standalone log carries
``input_len`` / ``pred_text`` / ``usable_frames`` fields that ``OneshotOutcome``
cannot supply; the shared path adopts the launcher log schema instead. The
shared one-shot path never downloads, never retries inference, and never
truncates the prompt, so ``invalid_outputs`` / ``total_retries`` land at 0 here
(no exceptions are scripted) and every cached video is materialized so the
existence-skip path matches.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

import revise.oneshot_lvbench_hf as oneshot
import revise.plug_and_play_lvbench_hf as lvbench_hf


_VOLATILE = ("ts",)
_VOLATILE_RESULTS = ("elapsed_s", "prompt_log_bytes", "prompt_log_lines")


class _FakeProcessor:
    """Minimal processor: a small fixed prompt length and scripted decode."""

    chat_template = "fake"

    def __init__(self, outputs):
        self._outputs = outputs

    def apply_chat_template(self, messages, **kwargs):
        return "PROMPT"

    def __call__(self, *, text, images, return_tensors=None):
        # A short fixed input length keeps input_len + max_new well under max_len,
        # so the script never drops frames or rejects the prompt as too long.
        n = max(1, len(images))
        return {"input_ids": torch.zeros((1, 4 + n), dtype=torch.long)}

    def batch_decode(self, gen_ids, skip_special_tokens=True):
        try:
            return [next(self._outputs)]
        except StopIteration:
            return ["A"]


class _FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(max_position_embeddings=32768)

    def generate(self, **kwargs):
        # Return input_ids + one generated token; gen_ids slice is decoded by
        # the fake processor (scripted), so the actual token value is irrelevant.
        input_ids = kwargs["input_ids"]
        extra = torch.zeros((input_ids.shape[0], 1), dtype=input_ids.dtype)
        return torch.cat([input_ids, extra], dim=-1)


def _samples():
    return [
        oneshot.MCVideoSample(
            dataset="lvbench",
            uid="q1",
            video_key="lv1.mp4",
            video_url="https://example.test/lv1",
            question="What object appears?",
            options=["red cup", "blue book", "green bag"],
            answer_letter="A",
            time_reference="00:02-00:05",
        ),
        oneshot.MCVideoSample(
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


def _materialize_cached_videos(cache_root, samples):
    dataset_cache = Path(cache_root) / "lvbench"
    dataset_cache.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        video_path = dataset_cache / sample.video_key
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"cached-video")


def _run_loop(scripted_outputs, *, sample_count=None):
    outputs = iter(scripted_outputs)
    samples = _samples()
    if sample_count is not None:
        samples = samples[:sample_count]

    def fake_extract_frames_1fps(_video_path, indices):
        return [Image.new("RGB", (2, 2), color=(int(i) % 255, 0, 0)) for i in indices]

    model = _FakeModel()
    processor = _FakeProcessor(outputs)

    with tempfile.TemporaryDirectory() as d:
        cache_root = os.path.join(d, "cache")
        _materialize_cached_videos(cache_root, samples)
        summary_path = os.path.join(d, "summary.json")
        log_path = os.path.join(d, "log.jsonl")
        argv = [
            "prog",
            "--split", "train",
            "--model-path", "m",
            "--video-cache-dir", cache_root,
            "--max-frames", "2",
            "--summary-json", summary_path,
            "--log-jsonl", log_path,
        ]
        # The loader, model/processor load, wandb, and yt-dlp download are called
        # by bare name in oneshot.main(), so they are patched on the oneshot
        # module. The migrated launcher reuses LVBenchHFDataset from
        # plug_and_play_lvbench_hf, whose frame probe / extraction are called by
        # bare name THERE, so those seams are patched on that module (matching the
        # multi-round HF golden's convention).
        with patch.object(sys, "argv", argv), \
                patch.object(oneshot, "_load_lvbench_samples", return_value=samples), \
                patch.object(oneshot, "_load_model_and_processor", return_value=(model, processor)), \
                patch.object(oneshot, "_maybe_init_wandb", return_value=None), \
                patch.object(oneshot, "_wandb_log", return_value=None), \
                patch.object(oneshot, "_download_youtube") as download_youtube, \
                patch.object(lvbench_hf, "extract_video_info", return_value=(8, 1.0)), \
                patch.object(lvbench_hf, "extract_frames_1fps", side_effect=fake_extract_frames_1fps):
            rc = oneshot.main()
        download_youtube.assert_not_called()
        summary = json.load(open(summary_path, encoding="utf-8"))
        records = []
        if os.path.exists(log_path):
            with open(log_path, encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
    results = summary["results"]
    for key in _VOLATILE_RESULTS:
        results.pop(key, None)
    for rec in records:
        for key in _VOLATILE:
            rec.pop(key, None)
    return rc, results, records


class OneshotLvbenchHfLoopTrajectoryTest(unittest.TestCase):
    def test_hf_oneshot_answers_golden(self):
        rc, results, records = _run_loop(["A", "B"])
        self.assertIsNone(rc)
        self.assertEqual(results, {
            "samples": 2,
            "correct": 2,
            "accuracy": 1.0,
            "total_rounds": 2,
            "avg_rounds": 1.0,
            "total_effective_rounds": 2,
            "avg_effective_rounds": 1.0,
            "total_frames_used": 4,
            "avg_frames_used": 2.0,
            "failed": 0,
            "invalid_outputs": 0,
            "invalid_action_terminated": 0,
            "total_retries": 0,
            "total_model_calls": 2,
        })
        self.assertEqual([r["pred_answer"] for r in records], ["A", "B"])

    def test_hf_oneshot_one_wrong_golden(self):
        rc, results, records = _run_loop(["A", "A"])
        self.assertIsNone(rc)
        self.assertEqual(results["correct"], 1)
        self.assertEqual(results["accuracy"], 0.5)
        self.assertEqual([r["pred_answer"] for r in records], ["A", "A"])

    def test_hf_oneshot_single_sample_golden(self):
        rc, results, records = _run_loop(["A", "B"], sample_count=1)
        self.assertIsNone(rc)
        self.assertEqual(results, {
            "samples": 1,
            "correct": 1,
            "accuracy": 1.0,
            "total_rounds": 1,
            "avg_rounds": 1.0,
            "total_effective_rounds": 1,
            "avg_effective_rounds": 1.0,
            "total_frames_used": 2,
            "avg_frames_used": 2.0,
            "failed": 0,
            "invalid_outputs": 0,
            "invalid_action_terminated": 0,
            "total_retries": 0,
            "total_model_calls": 1,
        })
        self.assertEqual(records[0]["pred_answer"], "A")


if __name__ == "__main__":
    unittest.main()
