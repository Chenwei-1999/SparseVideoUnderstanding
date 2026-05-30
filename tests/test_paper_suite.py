from pathlib import Path

from scripts.repro.paper_suite import EXPERIMENTS, _cmd_egoschema_pnp, _egoschema_local_coverage


def test_egoschema_local_coverage_counts_present_videos(tmp_path):
    rows = tmp_path / "pnp_subset_500.json"
    rows.write_text(
        '[{"video_path":"present.mp4"},{"video_path":"missing.mp4"}]\n',
        encoding="utf-8",
    )
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "present.mp4").write_bytes(b"fake")

    assert _egoschema_local_coverage(str(rows), str(videos)) == {"total": 2, "present": 1, "missing": 1}


def test_full_egoschema_command_uses_hf_when_local_video_coverage_incomplete(tmp_path):
    rows = tmp_path / "pnp_subset_500.json"
    rows.write_text('[{"video_path":"missing.mp4"}]\n', encoding="utf-8")
    videos = tmp_path / "videos"
    videos.mkdir()
    assets = {
        "asset_root": str(tmp_path),
        "remote_api": {"base_url": "http://localhost:1", "model_id": "dummy"},
        "models": {},
        "datasets": {"egoschema": {"json": str(rows), "video_root": str(videos)}},
    }

    cmd = _cmd_egoschema_pnp(assets, smoke=False, out_dir=tmp_path)

    assert "--egoschema-source" in cmd
    assert cmd[cmd.index("--egoschema-source") + 1] == "hf"
    assert "--auto-download-egoschema-videos" in cmd


def test_smoke_egoschema_command_can_use_partial_local_cache(tmp_path):
    rows = tmp_path / "pnp_subset_500.json"
    rows.write_text('[{"video_path":"present.mp4"},{"video_path":"missing.mp4"}]\n', encoding="utf-8")
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "present.mp4").write_bytes(b"fake")
    assets = {
        "asset_root": str(tmp_path),
        "remote_api": {"base_url": "http://localhost:1", "model_id": "dummy"},
        "models": {},
        "datasets": {"egoschema": {"json": str(rows), "video_root": str(videos)}},
    }

    cmd = _cmd_egoschema_pnp(assets, smoke=True, out_dir=Path(tmp_path))

    assert cmd[cmd.index("--egoschema-source") + 1] == "local"


def _videoespresso_assets(tmp_path):
    return {
        "remote_api": {"base_url": "http://localhost:1", "model_id": "dummy-model"},
        "models": {},
        "datasets": {
            "videoespresso": {
                "test_json": str(tmp_path / "bench_hard.json"),
                "test_video_root": str(tmp_path / "test_video"),
            }
        },
    }


def _training_assets(tmp_path):
    return {
        "remote_api": {"base_url": None, "model_id": None},
        "models": {
            "qwen25_vl_3b": str(tmp_path / "models" / "Qwen2.5-VL-3B-Instruct"),
            "qwen25_vl_7b": str(tmp_path / "models" / "Qwen2.5-VL-7B-Instruct"),
        },
        "datasets": {
            "nextqa": {
                "video_root": str(tmp_path / "NExT-QA" / "NExTVideo"),
                "map_json": str(tmp_path / "NExT-QA" / "map_vid_vidorID.json"),
                "train_csv": str(tmp_path / "NExT-QA" / "nextqa" / "train.csv"),
                "val_csv": str(tmp_path / "NExT-QA" / "nextqa" / "val.csv"),
            },
            "videoespresso": {
                "root": str(tmp_path / "VideoEspresso"),
                "train_video_json": str(tmp_path / "VideoEspresso" / "train_video.json"),
                "test_json": str(tmp_path / "VideoEspresso" / "benchmark.json"),
            },
        },
    }


def test_full_sft_training_pipelines_use_paper_scale_teacher_cap(tmp_path):
    assets = _training_assets(tmp_path)

    nextqa_cmd = EXPERIMENTS["nextqa_train_pipeline"]["build"](assets, False, tmp_path)
    videoespresso_cmd = EXPERIMENTS["videoespresso_train_pipeline"]["build"](assets, False, tmp_path)

    assert "MAX_SAMPLES=8000" in nextqa_cmd
    assert "MAX_SAMPLES=0" not in nextqa_cmd
    assert "MAX_SAMPLES=8000" in videoespresso_cmd
    assert "MAX_SAMPLES=0" not in videoespresso_cmd


def test_smoke_sft_training_pipelines_keep_tiny_teacher_cap(tmp_path):
    assets = _training_assets(tmp_path)

    nextqa_cmd = EXPERIMENTS["nextqa_train_pipeline"]["build"](assets, True, tmp_path)
    videoespresso_cmd = EXPERIMENTS["videoespresso_train_pipeline"]["build"](assets, True, tmp_path)

    assert "MAX_SAMPLES=4" in nextqa_cmd
    assert "MAX_SAMPLES=4" in videoespresso_cmd


def test_sft_teacher_scripts_use_neutral_teacher_defaults():
    repo_root = Path(__file__).resolve().parents[1]
    nextqa_script = (repo_root / "examples/revise/run_generate_teacher_data.sh").read_text(encoding="utf-8")
    videoespresso_script = (repo_root / "examples/revise/run_generate_teacher_data_videoespresso.sh").read_text(
        encoding="utf-8"
    )
    sft_converter = (repo_root / "examples/revise/generate_sft_data.py").read_text(encoding="utf-8")

    assert 'MAX_SAMPLES="${MAX_SAMPLES:-8000}"' in nextqa_script
    assert 'MAX_SAMPLES="${MAX_SAMPLES:-8000}"' in videoespresso_script
    assert "nextqa_teacher_train_log.jsonl" in nextqa_script
    assert "videoespresso_teacher_train_log.jsonl" in videoespresso_script
    assert "nextqa_teacher_train_log.jsonl" in sft_converter
    assert "pnp_7b_train_log" not in nextqa_script
    assert "pnp_7b_train_log" not in videoespresso_script
    assert "pnp_7b_train_log" not in sft_converter


def test_budget_ablation_experiments_have_named_videoespresso_commands(tmp_path):
    expected = {
        "videoespresso_budget_10x2": (10, 2),
        "videoespresso_budget_2x10": (2, 10),
        "videoespresso_budget_4x5": (4, 5),
        "videoespresso_budget_5x4": (5, 4),
        "videoespresso_budget_7x4": (7, 4),
        "videoespresso_budget_9x4": (9, 4),
    }
    assets = _videoespresso_assets(tmp_path)

    for exp_id, (rounds, frames) in expected.items():
        cmd = EXPERIMENTS[exp_id]["build"](assets, False, tmp_path)
        assert cmd[cmd.index("--dataset-name") + 1] == "videoespresso"
        assert cmd[cmd.index("--max-rounds") + 1] == str(rounds)
        assert cmd[cmd.index("--max-frames-per-round") + 1] == str(frames)
        assert "--videoespresso-use-official-prompt" in cmd
        assert "--no-videoespresso-with-evidence" in cmd
        assert cmd[cmd.index("--summary-json") + 1].endswith(f"{exp_id}.summary.json")
        assert cmd[cmd.index("--log-jsonl") + 1].endswith(f"{exp_id}.jsonl")


def test_best_frame_round_experiments_have_named_videoespresso_commands(tmp_path):
    expected = {
        "videoespresso_best_01x06": (1, 6),
        "videoespresso_best_02x04": (2, 4),
        "videoespresso_best_03x06": (3, 6),
        "videoespresso_best_04x04": (4, 4),
    }
    assets = _videoespresso_assets(tmp_path)

    for exp_id, (rounds, frames) in expected.items():
        cmd = EXPERIMENTS[exp_id]["build"](assets, False, tmp_path)
        assert cmd[cmd.index("--dataset-name") + 1] == "videoespresso"
        assert cmd[cmd.index("--max-rounds") + 1] == str(rounds)
        assert cmd[cmd.index("--max-frames-per-round") + 1] == str(frames)
        assert "--videoespresso-use-official-prompt" in cmd
        assert "--no-videoespresso-with-evidence" in cmd
        assert cmd[cmd.index("--summary-json") + 1].endswith(f"{exp_id}.summary.json")
        assert cmd[cmd.index("--log-jsonl") + 1].endswith(f"{exp_id}.jsonl")


def test_component_ablation_experiments_toggle_carryover_and_structured_flags(tmp_path):
    # (ablate_carryover, ablate_structured) per the 4 rows of abaltion_components.tex.
    expected = {
        "videoespresso_components_full": (False, False),
        "videoespresso_components_no_carryover": (True, False),
        "videoespresso_components_no_structured": (False, True),
        "videoespresso_components_no_both": (True, True),
    }
    assets = _videoespresso_assets(tmp_path)

    for exp_id, (ablate_carryover, ablate_structured) in expected.items():
        cmd = EXPERIMENTS[exp_id]["build"](assets, False, tmp_path)
        assert cmd[cmd.index("--dataset-name") + 1] == "videoespresso"
        # Component ablation holds the budget fixed at the REVISE default 4x3.
        assert cmd[cmd.index("--max-rounds") + 1] == "4"
        assert cmd[cmd.index("--max-frames-per-round") + 1] == "3"
        assert ("--ablate-state-carryover" in cmd) is ablate_carryover
        assert ("--ablate-structured-summary" in cmd) is ablate_structured
        assert "--videoespresso-use-official-prompt" in cmd
        assert "--no-videoespresso-with-evidence" in cmd
        assert cmd[cmd.index("--summary-json") + 1].endswith(f"{exp_id}.summary.json")
        assert cmd[cmd.index("--log-jsonl") + 1].endswith(f"{exp_id}.jsonl")


def test_phase_a_teacher_log_override_replaces_inline_gen_for_nextqa(tmp_path, monkeypatch):
    # Without override: training pipeline emits the inline teacher-generation shell.
    monkeypatch.delenv("REVISE_NEXTQA_TEACHER_LOG_OVERRIDE", raising=False)
    from scripts.repro.common import discover_assets
    assets = discover_assets()
    if not assets["datasets"]["nextqa"].get("video_root"):
        # Cluster-shaped check fixture not available locally; skip silently.
        return
    cmd_default = EXPERIMENTS["nextqa_train_pipeline"]["build"](assets, False, tmp_path)
    assert "run_generate_teacher_data.sh" in cmd_default
    assert "ln -sfn" not in cmd_default

    # With override: same builder skips teacher gen and symlinks the Phase A JSONL.
    override = tmp_path / "phase_a" / "nextqa_teacher72b_train_log.jsonl"
    monkeypatch.setenv("REVISE_NEXTQA_TEACHER_LOG_OVERRIDE", str(override))
    cmd_override = EXPERIMENTS["nextqa_train_pipeline"]["build"](assets, False, tmp_path)
    assert "run_generate_teacher_data.sh" not in cmd_override
    assert "ln -sfn" in cmd_override
    # The validation snippet must reference the exact override path so an
    # accidentally-misspelled env var doesn't silently mask a missing file.
    assert str(override) in cmd_override
    assert "ERROR: Phase A teacher JSONL not found" in cmd_override


def test_phase_a_teacher_log_override_preserves_videoespresso_prep_and_extract(tmp_path, monkeypatch):
    # VE pipeline = prepare_mc + extract_videos + teacher_gen + sft_then_rl.
    # Override only replaces teacher_gen; prepare + extract are needed for GRPO.
    monkeypatch.delenv("REVISE_VIDEOESPRESSO_TEACHER_LOG_OVERRIDE", raising=False)
    from scripts.repro.common import discover_assets
    assets = discover_assets()
    if not assets["datasets"]["videoespresso"].get("train_video_json"):
        return

    override = tmp_path / "phase_a" / "videoespresso_teacher72b_train_log.jsonl"
    monkeypatch.setenv("REVISE_VIDEOESPRESSO_TEACHER_LOG_OVERRIDE", str(override))
    cmd_override = EXPERIMENTS["videoespresso_train_pipeline"]["build"](assets, False, tmp_path)

    # Teacher gen replaced; prepare + extract retained.
    assert "run_generate_teacher_data_videoespresso.sh" not in cmd_override
    assert "ln -sfn" in cmd_override
    assert "prepare_videoespresso_mc_train.py" in cmd_override
    assert "extract_videoespresso_split_zip_subset.py" in cmd_override
    assert str(override) in cmd_override
