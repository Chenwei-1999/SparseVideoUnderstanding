"""Characterization (golden-master) tests for the plug-and-play pipelines.

These tests pin the *current* observable behavior of helpers that are shared or
near-duplicated across the four ``plug_and_play_*`` evaluation scripts. They
exist to guard refactors (e.g. extracting a shared ``start_vllm_server``) against
silently changing benchmark behavior: the agent loop, frame sampling, and the
vLLM launch command all feed directly into reported paper numbers, yet were
previously untested.

Design notes:

* Written with stdlib ``unittest`` (not pytest fixtures) so they run under the
  runtime env that actually has the project dependencies::

      python -m unittest tests.test_pnp_characterization -v

  They are also collectable by pytest if it is available.
* Frame-sampling expectations are *golden values* captured from the live code.
  ``nextqa`` uses ``rng.sample`` and ``egoschema`` uses ``rng.shuffle``; for the
  same seed these intentionally draw different frames. The asserts below lock in
  that divergence so the two implementations are never accidentally unified
  (doing so would shift which frames each benchmark sees -> different numbers).
"""

import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import revise.plug_and_play_nextqa_vllm as nextqa
import revise.plug_and_play_egoschema_vllm as egoschema
import revise.plug_and_play_videomme_lvbench_vllm as videomme
import revise.plug_and_play_lvbench_hf as lvbench_hf


class SampleUnseenFramesTest(unittest.TestCase):
    """Pin both frame-sampling variants, including their intentional divergence."""

    def test_nextqa_sample_variant_golden(self):
        # nextqa uses sorted(rng.sample(candidates, k)).
        self.assertEqual(nextqa._sample_unseen_frames(10, set(), 3, random.Random(0)), [0, 6, 9])
        self.assertEqual(nextqa._sample_unseen_frames(10, set(), 3, random.Random(7)), [2, 5, 6])
        self.assertEqual(nextqa._sample_unseen_frames(10, {0, 1, 2}, 3, random.Random(0)), [6, 8, 9])

    def test_egoschema_shuffle_variant_golden(self):
        # egoschema uses sorted(rng.shuffle(candidates)[:k]) -> different draw.
        self.assertEqual(egoschema._sample_unseen_frames(10, set(), 3, random.Random(0)), [1, 7, 8])
        self.assertEqual(egoschema._sample_unseen_frames(10, set(), 3, random.Random(7)), [1, 3, 8])
        self.assertEqual(egoschema._sample_unseen_frames(10, {0, 1, 2}, 3, random.Random(0)), [4, 5, 7])

    def test_variants_diverge_for_same_seed(self):
        # This is the load-bearing fact: do NOT unify these two functions.
        seed_a = random.Random(0)
        seed_b = random.Random(0)
        self.assertNotEqual(
            nextqa._sample_unseen_frames(10, set(), 3, seed_a),
            egoschema._sample_unseen_frames(10, set(), 3, seed_b),
        )

    def test_shared_guard_branches(self):
        for fn in (nextqa._sample_unseen_frames, egoschema._sample_unseen_frames):
            self.assertEqual(fn(0, set(), 3, random.Random(0)), [])      # no frames
            self.assertEqual(fn(10, set(), 0, random.Random(0)), [])     # k == 0
            self.assertEqual(fn(3, {0, 1, 2}, 3, random.Random(0)), [])  # all seen
            self.assertEqual(fn(4, set(), 10, random.Random(0)), [0, 1, 2, 3])  # candidates <= k


class RetryFeedbackTextTest(unittest.TestCase):
    """Pin the exact retry-instruction strings (they are part of the prompt)."""

    NON_FORCE = "fb\nPlease respond with one of the required formats."

    def test_non_force_is_identical_everywhere(self):
        for mod in (nextqa, videomme, lvbench_hf):
            self.assertEqual(mod._retry_feedback_text("fb", force_answer=False), self.NON_FORCE)

    def test_nextqa_force_text_golden(self):
        self.assertEqual(
            nextqa._retry_feedback_text("fb", force_answer=True),
            "fb\n"
            "Output ONLY <think>...</think> then <answer>LETTER</answer>. "
            "In <summarize>, include P/O/H/U/R in that exact order. "
            "In <answer>, LETTER must be a single option letter (e.g., A/B/C/D/E).",
        )

    def test_videomme_and_lvbench_force_text_are_identical_golden(self):
        expected = "fb\nYou MUST answer now. Output <think>...</think> then <answer>LETTER</answer>."
        self.assertEqual(videomme._retry_feedback_text("fb", force_answer=True), expected)
        self.assertEqual(lvbench_hf._retry_feedback_text("fb", force_answer=True), expected)


def _launcher_args(**overrides):
    base = dict(
        model_path="/models/Qwen2.5-VL-3B-Instruct",
        model_id="Qwen2.5-VL-3B-Instruct",
        host="127.0.0.1",
        port=18000,
        dtype="bfloat16",
        max_model_len=8192,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.55,
        max_frames_per_round=3,
        caption_gen_max_frames=5,
        server_log=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class StartVllmServerTest(unittest.TestCase):
    """Snapshot the vLLM launch argv per script (pre-refactor baseline).

    The three launchers share a ~40-line command but differ in exactly two
    observable ways: the CUDA_VISIBLE_DEVICES default and the per-prompt image
    limit. These asserts pin both so an extraction that parameterizes them is
    proven behavior-preserving.
    """

    def _capture(self, mod, args):
        with patch.dict("os.environ", {}, clear=True), patch.object(mod.subprocess, "Popen") as popen:
            mod._start_vllm_server(args)
        kwargs = popen.call_args.kwargs
        return popen.call_args.args[0], kwargs["env"]

    def _flag(self, cmd, name):
        return cmd[cmd.index(name) + 1]

    def test_nextqa_uses_max_of_frame_caps_and_four_gpu_default(self):
        cmd, env = self._capture(nextqa, _launcher_args())
        # image limit = max(max_frames_per_round=3, caption_gen_max_frames=5) = 5
        self.assertEqual(self._flag(cmd, "--limit-mm-per-prompt"), '{"image": 5}')
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1,2,3")
        self.assertEqual(self._flag(cmd, "--served-model-name"), "Qwen2.5-VL-3B-Instruct")

    def test_egoschema_uses_frames_per_round_and_four_gpu_default(self):
        cmd, env = self._capture(egoschema, _launcher_args())
        self.assertEqual(self._flag(cmd, "--limit-mm-per-prompt"), '{"image": 3}')
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1,2,3")

    def test_videomme_uses_frames_per_round_and_single_gpu_default(self):
        cmd, env = self._capture(videomme, _launcher_args())
        self.assertEqual(self._flag(cmd, "--limit-mm-per-prompt"), '{"image": 3}')
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")

    def test_trust_remote_code_only_when_model_requires_it(self):
        # Qwen2.5-VL path does not trigger trust-remote-code.
        cmd, _ = self._capture(nextqa, _launcher_args())
        self.assertNotIn("--trust-remote-code", cmd)


class LoaderDivergenceTest(unittest.TestCase):
    """Document (and guard) why the videomme/lvbench loaders are NOT merged.

    The same-named ``_load_videomme_samples`` differs between the two scripts:
    the vLLM variant keeps every row and records a ``video_url``; the HF variant
    drops rows with an empty answer and carries ``question_type``/``video_type``
    instead. Unifying them would change which samples each pipeline scores.
    """

    ROWS = [
        {"videoID": "v1", "url": "http://x/1", "question_id": "q1",
         "question": "Q?", "options": ["A. a", "B. b"], "answer": "A", "domain": "d", "duration": "short"},
        {"videoID": "v2", "url": "http://x/2", "question_id": "q2",
         "question": "Q2?", "options": ["A. a", "B. b"], "answer": "", "domain": "d", "duration": "long"},
    ]

    def test_vllm_loader_keeps_all_rows_and_records_url(self):
        with patch.object(videomme, "load_dataset", return_value=list(self.ROWS)):
            samples = videomme._load_videomme_samples("test")
        self.assertEqual(len(samples), 2)  # empty-answer row retained
        self.assertEqual(samples[0].video_url, "http://x/1")
        self.assertTrue(hasattr(samples[0], "video_url"))

    def test_hf_loader_drops_empty_answer_and_has_no_url(self):
        with patch.object(lvbench_hf, "load_dataset", return_value=list(self.ROWS)):
            samples = lvbench_hf._load_videomme_samples("test")
        self.assertEqual(len(samples), 1)  # empty-answer row filtered out
        self.assertFalse(hasattr(samples[0], "video_url"))
        self.assertEqual(samples[0].question_type, "d")
        self.assertEqual(samples[0].video_type, "short")


if __name__ == "__main__":
    unittest.main()
