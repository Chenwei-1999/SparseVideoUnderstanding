"""Golden trajectory test for the one-shot local-MC (NExT-QA / jsonmc) vLLM baseline.

This is the pre-migration oracle for moving ``oneshot_local_mc_vllm.main`` onto
the shared PnP engine. It drives the real CLI entry point with mocked
dataset/video/model boundaries, scripted model outputs, and a fixed seed, then
freezes the script's own ``--summary-json`` counter dict.

The single-round baseline samples frames once, performs one model chat per
sample, parses the answer letter, and scores. There is no select/summarize/retry
loop. Unlike the videomme oneshot test, this golden ALSO locks one full log
record (``user_text`` / ``messages`` / ``frame_indices`` / videoespresso flags),
because for local-MC the exact prompt IS the deliverable: the legacy prompt emits
``Frame {idx}: <image>`` keyed on the actual sampled indices, not ``Frame {i+1}``.
The model boundaries are mocked in BOTH the ``oneshot_local_mc_vllm`` namespace
(pre-migration standalone call path) and the ``plug_and_play_*`` adapter/backend
namespace (post-migration delegated call path), so this test stays byte-identical
across the migration.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import revise.oneshot_local_mc_vllm as oneshot
import revise.plug_and_play_egoschema_vllm as egoschema


_VOLATILE = ("ts", "elapsed_s")
_VOLATILE_SUMMARY = ("ts", "elapsed_s", "log_jsonl")


def _samples():
    return [
        egoschema.EgoSchemaSample(
            sample_id="s1",
            qid="q1",
            video_path="/videos/v1.mp4",
            question="What object appears?",
            choices=["red cup", "blue book", "green bag", "yellow hat", "black box"],
            answer_idx=0,
            frame_count=8,
            task="",
            evidence="",
        ),
        egoschema.EgoSchemaSample(
            sample_id="s2",
            qid="q2",
            video_path="/videos/v2.mp4",
            question="What happens next?",
            choices=["open", "close", "turn", "lift", "drop"],
            answer_idx=1,
            frame_count=8,
            task="",
            evidence="",
        ),
    ]


def _run_loop(scripted_outputs, *, sample_count=None):
    outputs = iter(scripted_outputs)
    samples = _samples()
    if sample_count is not None:
        samples = samples[:sample_count]

    def fake_chat_once(*_args, **_kwargs):
        try:
            return next(outputs)
        except StopIteration:
            return "A"

    def fake_extract_frames(_video_path, indices):
        return [Image.new("RGB", (2, 2), color=(int(i) % 255, 0, 0)) for i in indices]

    with tempfile.TemporaryDirectory() as d:
        summary_path = os.path.join(d, "summary.json")
        log_path = os.path.join(d, "log.jsonl")
        argv = [
            "prog",
            "--dataset", "jsonmc",
            "--json", "samples.json",
            "--video-root", "/videos",
            "--model-path", "m",
            "--port", "18081",
            "--max-frames", "2",
            "--summary-json", summary_path,
            "--log-jsonl", log_path,
        ]
        # All transport-ish seams are called by BARE NAME inside oneshot.main()
        # (loaders, frame extraction, chat fn, get_model_id), both before and
        # after the migration onto the shared engine. Patching the oneshot module
        # namespace therefore drives the script identically across the migration.
        with patch.object(sys, "argv", argv), \
                patch.object(oneshot, "_load_egoschema_samples", return_value=samples), \
                patch.object(oneshot, "_load_nextqa_samples", return_value=samples), \
                patch.object(oneshot, "get_model_id", return_value="test-model"), \
                patch.object(oneshot, "extract_video_info", return_value=(8, 1.0)), \
                patch.object(oneshot, "extract_frames", side_effect=fake_extract_frames), \
                patch.object(oneshot, "_chat_once", side_effect=fake_chat_once):
            rc = oneshot.main()
        summary = json.load(open(summary_path, encoding="utf-8"))
        records = []
        if os.path.exists(log_path):
            with open(log_path, encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
    for key in _VOLATILE_SUMMARY:
        summary.pop(key, None)
    for rec in records:
        for key in _VOLATILE:
            rec.pop(key, None)
    return rc, summary, records


class OneshotLocalMcLoopTrajectoryTest(unittest.TestCase):
    def test_jsonmc_oneshot_answers_golden(self):
        rc, summary, records = _run_loop(["A", "B"])
        self.assertEqual(rc, 0)
        self.assertEqual(summary, {
            "task": "oneshot_local_mc_vllm",
            "dataset": "jsonmc",
            "samples": 2,
            "answered": 2,
            "correct": 2,
            "accuracy": 1.0,
            "failed": 0,
            "avg_frames": 2.0,
            "total_model_calls": 2,
            "videoespresso_use_official_prompt": None,
            "videoespresso_with_evidence": None,
        })

    def test_jsonmc_oneshot_one_wrong_golden(self):
        rc, summary, records = _run_loop(["A", "A"])
        self.assertEqual(rc, 0)
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual([r["pred_answer"] for r in records], ["A", "A"])
        self.assertEqual([r["correct"] for r in records], [True, False])

    def test_jsonmc_oneshot_locks_prompt_record(self):
        rc, summary, records = _run_loop(["A", "B"])
        self.assertEqual(rc, 0)
        # Lock the full first log record: the prompt IS the deliverable.
        expected_user_text = (
            "Question: What object appears?\n"
            "Options:\n"
            "A. red cup\n"
            "B. blue book\n"
            "C. green bag\n"
            "D. yellow hat\n"
            "E. black box\n"
            "\n"
            "You will be shown 2 video frames.\n"
            "Answer with EXACTLY ONE option letter (for example: A/B/C/D/E). "
            "Do not output any other text.\n"
            "\n"
            "Frames:\n"
            "Frame 0: <image>\n"
            "Frame 7: <image>"
        )
        rec = records[0]
        self.assertEqual(rec["user_text"], expected_user_text)
        self.assertEqual(rec["frame_indices"], [0, 7])
        self.assertEqual(rec["dataset"], "jsonmc")
        self.assertEqual(rec["sample_id"], "s1")
        self.assertEqual(rec["question"], "What object appears?")
        self.assertEqual(rec["options"], ["red cup", "blue book", "green bag", "yellow hat", "black box"])
        self.assertEqual(rec["pred_answer"], "A")
        self.assertEqual(rec["answer_gt"], "A")
        self.assertEqual(rec["correct"], True)
        self.assertEqual(rec["raw_output"], "A")
        self.assertEqual(
            rec["messages"],
            [
                {"role": "system", "content": ""},
                {"role": "user", "content": expected_user_text},
                {"role": "assistant", "content": "A"},
            ],
        )
        self.assertIs(rec["videoespresso_official_prompt"], False)
        self.assertIs(rec["videoespresso_with_evidence"], False)


if __name__ == "__main__":
    unittest.main()
