import json
import sys
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

import revise.backends.hf_inprocess as hf_inprocess
import revise.benchmarks.egoschema_vllm as egoschema_vllm
import revise.benchmarks.nextqa_vllm as nextqa_vllm
import revise.pnp.utils as pnp_utils
from revise.benchmarks.egoschema_vllm import _load_egoschema_samples
from revise.benchmarks.nextqa_vllm import _load_nextqa_samples
from revise.benchmarks.videomme_lvbench_vllm import _bare_answer_after_summary
from revise.pnp.utils import (
    apply_processor_chat_template,
    configure_llava_processor,
    format_videoespresso_question_block,
    format_videomme_question_block,
)
from scripts.extract_videoespresso_split_zip_subset import (
    DEFAULT_PREFIX,
    _extract_entry,
    _part_paths,
    _wanted_entries,
)


def test_videoespresso_loader_preserves_official_fields(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"not decoded in this loader test")
    rows = [
        {
            "question_idx": "ve-1",
            "video_path": "clip.mp4",
            "task": "temporal reasoning",
            "question": "What happens after the cup is lifted?",
            "options": ["(A): It spills", "(B): It is washed", "(C): It is placed down", "(D): It disappears"],
            "correct_answer": "C",
            "evidence": "The hand lowers the cup near the table.",
        }
    ]
    json_path = tmp_path / "bench_hard.json"
    json_path.write_text(json.dumps(rows), encoding="utf-8")

    samples = _load_egoschema_samples(str(json_path), str(tmp_path), max_samples=0, seed=0)

    assert len(samples) == 1
    assert samples[0].qid == "ve-1"
    assert samples[0].choices == ["It spills", "It is washed", "It is placed down", "It disappears"]
    assert samples[0].answer_idx == 2
    assert samples[0].task == "temporal reasoning"
    assert samples[0].evidence == "The hand lowers the cup near the table."


def test_videoespresso_prompt_matches_official_close_ended_shape():
    prompt = format_videoespresso_question_block(
        "What happens after the cup is lifted?",
        ["(A): It spills", "(B): It is washed", "(C): It is placed down", "(D): It disappears"],
        task="temporal reasoning",
        evidence="The hand lowers the cup near the table.",
        with_evidence=False,
        revise_answer_tags=True,
    )

    assert prompt.startswith("Please finish the temporal reasoning task. Question:")
    assert "Your inference evidence" not in prompt
    assert "(A) It spills" in prompt
    assert "(D) It disappears" in prompt
    assert "Select the answer and only give the option letters." in prompt
    assert "<answer>LETTER</answer>" in prompt


def test_component_ablation_can_omit_summary_state_from_prompt():
    prompt = egoschema_vllm._build_user_text(
        "Question: What happens?\nOptions:\nA. one\nB. two",
        "P: prior observation; O: current; H: belief; U: unclear; R: request evidence",
        frame_count=12,
        round_idx=2,
        frame_indices=[3, 4],
        seen_frames=[0, 1, 3, 4],
        carry_summary_state=False,
    )

    assert "state carryover is disabled" in prompt
    assert "P: prior observation" not in prompt
    assert "<summarize>" not in prompt


def test_component_ablation_can_validate_unstructured_summary():
    unstructured = "The visible evidence suggests the person is preparing food, but one detail remains unclear."

    assert egoschema_vllm._summary_is_valid_for_mode(unstructured, require_structured=False)
    assert not egoschema_vllm._summary_is_valid_for_mode(unstructured, require_structured=True)
    assert not egoschema_vllm._summary_is_valid_for_mode("...", require_structured=False)


def test_videoespresso_train_split_zip_subset_extractor(tmp_path):
    rel = "Moviechat/videos/1/AWG-5.mp4"
    payload = b"fake-video-bytes"
    zip_path = tmp_path / "VideoEspresso_train_video.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(DEFAULT_PREFIX + rel, payload)

    entries, missing, final_disk = _wanted_entries(zip_path, [rel], archive_prefix=DEFAULT_PREFIX)
    assert missing == []
    assert rel in entries

    out_path = tmp_path / "all_video" / rel
    result = _extract_entry(_part_paths(tmp_path, final_disk), entries[rel], out_path, overwrite=False, dry_run=False)

    assert result["status"] == "extracted"
    assert out_path.read_bytes() == payload


def test_videomme_prompt_matches_official_no_subtitle_shape():
    prompt = format_videomme_question_block(
        "Which object appears first?",
        ["A. Cup", "B. Book", "C. Bag", "D. Phone"],
    )

    assert prompt.startswith("Select the best answer to the following multiple-choice question based on the video.")
    assert "subtitles are listed below" not in prompt
    assert "A. Cup" in prompt
    assert "D. Phone" in prompt
    assert prompt.endswith("The best answer is:")


def test_videomme_bare_answer_recovery_ignores_frame_tags():
    raw = (
        "<think></think>\n"
        "<summarize>P: seen; O: unclear; H: unsure; U: need evidence; R: request frames</summarize>\n"
        "<select>1, 2, 3</select>"
    )

    assert _bare_answer_after_summary(raw) is None


def test_videomme_bare_answer_recovery_accepts_short_tail_letter():
    raw = "<summarize>P: seen; O: enough; H: answer; U: none; R: answered</summarize>\nC"

    assert _bare_answer_after_summary(raw) == "C"


def test_llava_processor_uses_tokenizer_chat_template_fallback():
    class DummyTokenizer:
        chat_template = "present"

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["tokenize"] is False
            assert kwargs["add_generation_prompt"] is True
            return messages[0]["content"]

    class DummyProcessor:
        chat_template = None
        tokenizer = DummyTokenizer()

        def apply_chat_template(self, messages, **kwargs):  # pragma: no cover - should not be called
            raise AssertionError("processor template should not be used when missing")

    text = apply_processor_chat_template(
        DummyProcessor(),
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Question?"}]}],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert text.startswith("<image>\n")
    assert text.endswith("Question?")


def test_configure_llava_processor_fills_missing_image_token_attrs():
    processor = SimpleNamespace(
        patch_size=None,
        vision_feature_select_strategy=None,
        num_additional_image_tokens=None,
    )
    config = SimpleNamespace(
        vision_config=SimpleNamespace(patch_size=14),
        vision_feature_select_strategy="default",
    )

    configure_llava_processor(processor, config)

    assert processor.patch_size == 14
    assert processor.vision_feature_select_strategy == "default"
    assert processor.num_additional_image_tokens == 0


def test_llava_qwen_hf_loader_routes_to_llava_next_runtime(tmp_path):
    model_dir = tmp_path / "LLaVA-OneVision-Qwen2-7B-OV"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlavaQwenForCausalLM"],
                "model_type": "llava",
                "vocab_size": 152064,
                "hidden_size": 3584,
            }
        ),
        encoding="utf-8",
    )
    runtime = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=32768))

    with patch.object(hf_inprocess, "load_llava_next_runtime", return_value=runtime) as runtime_mock:
        with patch.object(hf_inprocess, "_load_transformers_components") as components_mock:
            model, processor = hf_inprocess._load_model_and_processor(str(model_dir), "bfloat16", "cuda:0")

    assert model is runtime
    assert processor is None
    runtime_mock.assert_called_once()
    components_mock.assert_not_called()


def test_cached_video_reader_reuses_decord_for_metadata_and_frames(monkeypatch):
    import numpy as np

    constructed = []

    class DummyBatch:
        def __init__(self, frames):
            self.frames = frames

        def asnumpy(self):
            return np.stack(self.frames, axis=0)

    class DummyVideoReader:
        def __init__(self, path, ctx=None):
            self.path = path
            constructed.append(self)
            self.frames = [
                np.full((2, 2, 3), fill_value=i, dtype=np.uint8)
                for i in range(5)
            ]

        def __len__(self):
            return len(self.frames)

        def get_avg_fps(self):
            return 2.0

        def get_batch(self, indices):
            return DummyBatch([self.frames[i] for i in indices])

    monkeypatch.setitem(
        sys.modules,
        "decord",
        SimpleNamespace(VideoReader=DummyVideoReader, cpu=lambda _idx: "cpu"),
    )

    reader = pnp_utils.CachedVideoReader("clip.mp4")

    assert reader.info() == (5, 2.0)
    images = reader.extract_frames([1, 3])
    assert reader.extract_frames_1fps([0, 2])[1].getpixel((0, 0)) == (4, 4, 4)
    assert len(images) == 2
    assert images[0].getpixel((0, 0)) == (1, 1, 1)
    assert len(constructed) == 1


def test_cached_video_reader_maps_timeline_indices_with_cached_metadata(monkeypatch):
    import numpy as np

    batches = []

    class DummyBatch:
        def __init__(self, frames):
            self.frames = frames

        def asnumpy(self):
            return np.stack(self.frames, axis=0)

    class DummyVideoReader:
        def __init__(self, path, ctx=None):
            self.frames = [
                np.full((1, 1, 3), fill_value=i, dtype=np.uint8)
                for i in range(10)
            ]

        def __len__(self):
            return len(self.frames)

        def get_avg_fps(self):
            return 3.0

        def get_batch(self, indices):
            batches.append(list(indices))
            return DummyBatch([self.frames[i] for i in indices])

    monkeypatch.setitem(
        sys.modules,
        "decord",
        SimpleNamespace(VideoReader=DummyVideoReader, cpu=lambda _idx: "cpu"),
    )

    reader = pnp_utils.CachedVideoReader("clip.mp4")

    images = reader.extract_frames_1fps([0, 1, 9])

    assert batches == [[0, 3, 9]]
    assert [img.getpixel((0, 0))[0] for img in images] == [0, 3, 9]


def test_vllm_launcher_sets_served_model_name_when_model_id_is_explicit():
    args = SimpleNamespace(
        model_path="/models/Qwen2.5-VL-3B-Instruct",
        model_id="Qwen2.5-VL-3B-Instruct",
        host="127.0.0.1",
        port=18000,
        dtype="bfloat16",
        max_model_len=8192,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.55,
        max_frames_per_round=3,
        caption_gen_max_frames=1,
        server_log=None,
    )

    with patch.object(nextqa_vllm.subprocess, "Popen") as popen_mock:
        nextqa_vllm._start_vllm_server(args)

    cmd = popen_mock.call_args.args[0]
    assert "--served-model-name" in cmd
    idx = cmd.index("--served-model-name")
    assert cmd[idx + 1] == "Qwen2.5-VL-3B-Instruct"


def test_vllm_launcher_enables_trust_remote_code_for_auto_map(tmp_path):
    model_dir = tmp_path / "InternVL2-8B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"auto_map": {"AutoModel": "modeling.Custom"}}))
    args = SimpleNamespace(
        model_path=str(model_dir),
        model_id="OpenGVLab/InternVL2-8B",
        host="127.0.0.1",
        port=18000,
        dtype="bfloat16",
        max_model_len=8192,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.55,
        max_frames_per_round=3,
        caption_gen_max_frames=1,
        server_log=None,
    )

    with patch.object(nextqa_vllm.subprocess, "Popen") as popen_mock:
        nextqa_vllm._start_vllm_server(args)

    cmd = popen_mock.call_args.args[0]
    assert "--trust-remote-code" in cmd


def test_nextqa_loader_resolves_hf_mirror_video_layout(tmp_path):
    video_dir = tmp_path / "NExTVideo" / "NExTVideo"
    video_dir.mkdir(parents=True)
    (video_dir / "4010069381.mp4").write_bytes(b"not decoded in this loader test")

    map_path = tmp_path / "map_vid_vidorID.json"
    map_path.write_text(json.dumps({"4010069381": "0101/4010069381"}), encoding="utf-8")
    csv_path = tmp_path / "val.csv"
    csv_path.write_text(
        "\n".join(
            [
                "video,frame_count,width,height,question,answer,qid,type,a0,a1,a2,a3,a4",
                "4010069381,10,320,240,What happens?,2,q1,CW,one,two,three,four,five",
            ]
        ),
        encoding="utf-8",
    )

    samples = _load_nextqa_samples(
        csv_path=str(csv_path),
        map_json=str(map_path),
        video_root=str(tmp_path),
        max_samples=1,
        seed=0,
    )

    assert len(samples) == 1
    assert samples[0].video_path == str(video_dir / "4010069381.mp4")


def test_nextqa_loader_resolves_official_mapped_video_layout(tmp_path):
    video_dir = tmp_path / "NExTVideo" / "1106"
    video_dir.mkdir(parents=True)
    (video_dir / "4010069381.mp4").write_bytes(b"not decoded in this loader test")

    map_path = tmp_path / "map_vid_vidorID.json"
    map_path.write_text(json.dumps({"4010069381": "1106/4010069381"}), encoding="utf-8")
    csv_path = tmp_path / "val.csv"
    csv_path.write_text(
        "\n".join(
            [
                "video,frame_count,width,height,question,answer,qid,type,a0,a1,a2,a3,a4",
                "4010069381,10,320,240,What happens?,2,q1,CW,one,two,three,four,five",
            ]
        ),
        encoding="utf-8",
    )

    samples = _load_nextqa_samples(
        csv_path=str(csv_path),
        map_json=str(map_path),
        video_root=str(tmp_path / "NExTVideo"),
        max_samples=1,
        seed=0,
    )

    assert len(samples) == 1
    assert samples[0].video_path == str(video_dir / "4010069381.mp4")
