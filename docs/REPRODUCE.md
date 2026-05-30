# Reproducing REVISE Paper Experiments

This document is the source-of-truth reproduction path for the REVISE paper numbers. Generated
datasets, model weights, logs, prompt traces, Slurm output, and paper drafts must stay outside Git
or under ignored directories such as `data/` and `outputs/`.

## Current Audit Status

- The original README result table is now treated as historical reported results, not verified
  rerun results.
- VideoEspresso evaluation has been corrected to preserve the official close-ended fields
  `task`, `options`, `correct_answer`, and optional `evidence`.
- GPU evaluation and training must be launched through Slurm. Local commands are for preflight,
  command generation, and metadata-only checks.
- Every experiment command writes JSON summaries plus JSONL prompt/conversation logs. Slurm job
  stdout/stderr and `doctor.py` snapshots are saved in the run directory.
- The Overleaf paper source is not tracked here. Update paper tables only after the corresponding
  run manifest is complete.

## Official Sources Checked

| Asset | Official source | Checked ref | Local use |
| --- | --- | --- | --- |
| VideoEspresso | <https://github.com/hshjerry/VideoEspresso> | `c865570f97b42a61d24c8cd16f6a728af64482db` | Official close-ended prompt and A-D scoring behavior |
| NExT-QA | <https://github.com/doc-doc/NExT-QA> | `2432e9724f88ed9f40010e2989f104570a91de4e` | Multi-choice QA annotations, video ID mapping, raw video link |
| EgoSchema | <https://github.com/egoschema/EgoSchema> | `505c787376b5e066d0ae406d0e0d41245cebba15` | Public 500-answer subset and download policy |
| Video-MME | <https://github.com/MME-Benchmarks/Video-MME> / <https://github.com/EvolvingLMMs-Lab/lmms-eval> | lmms-eval `247bebd8c6694f101fa076970d4b1cf9935897f8` | Official prompt/evaluation shape and HF dataset location |
| LVBench | <https://github.com/zai-org/LVBench> / <https://huggingface.co/datasets/lmms-lab/LVBench> | HF dataset updated 2025-09-16 | Official metadata plus HF video chunks |
| verl | <https://github.com/verl-project/verl> | `a104320336a56701f9bf911c1d14f6d795eb6d8c` | Upstream version target for the embedded `verl/` tree |
| Qwen2-VL-7B | <https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct> | HF model card / files | Exact local backbone for the Qwen2-VL paper row |
| InternVL2-8B | <https://huggingface.co/OpenGVLab/InternVL2-8B> | HF model card / vLLM instructions | Exact local backbone for the InternVL2 paper row |
| LLaVA-OV-7B | <https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov> | HF model card / LLaVA-NeXT loader `df179663ae8b83207df100a1f7af24caec633ff9` | Exact local backbone for Video-MME/LVBench paper rows |

## Environment

Use Python 3.10. The cluster already has a candidate environment at `verlrun`; otherwise create one:

```bash
conda create -n verlrun python=3.10 -y
conda activate verlrun
pip install -U pip
pip install -e .
```

Install exactly one inference backend in the same environment:

```bash
# vLLM
pip install -r requirements.txt

# or SGLang
pip install -r requirements_sglang.txt
```

The setup helper keeps the backend choice explicit:

```bash
ENV_NAME=verlrun INSTALL_BACKENDS=vllm bash scripts/repro/setup_env.sh
```

## Asset Download And Registration

Default ignored asset root:

```bash
export REVISE_ASSET_ROOT="$PWD/data/revise_assets"
export HF_HOME="$REVISE_ASSET_ROOT/.hf_home"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_XET_CACHE="$HF_HOME/xet"
```

On shared/HPC clusters, `data/revise_assets` is typically a symlink to a per-user
scratch path (so large videos, model snapshots, and video caches do not consume
home/project quota). The symlink target is host-specific — set `REVISE_ASSET_ROOT`
to bypass the symlink and use the scratch path directly:

```bash
export REVISE_ASSET_ROOT="/path/to/your/scratch/revise_assets"
```

The reproduction scripts also check these variables at runtime. If a cluster default
points to a read-only HF cache, they fall back to `$REVISE_ASSET_ROOT/hf_home`.

Dry-run the download/register plan:

```bash
python scripts/repro/download_assets.py --all --dry-run
```

Download Hugging Face-hosted metadata/assets and model snapshots. `--videomme` intentionally
excludes the 101GB official video chunks; those are handled separately below:

```bash
python scripts/repro/download_assets.py --videoespresso --videomme --models
python scripts/repro/prepare_downloaded_assets.py --videoespresso
source data/revise_assets/revise_env.sh
```

The default model download above only fetches the local Qwen2.5-VL snapshots used for the training
pipeline and efficient ablations. Some paper rows use different exact backbones. Download those
with the non-GPU Slurm helper before rerunning rows that name them:

```bash
python scripts/repro/submit_hf_model_download_slurm.py \
  --paper-models \
  --submit \
  --partition normal \
  --concurrency 2
```

This downloads/registers:

- `Qwen/Qwen2-VL-7B-Instruct` -> `REVISE_QWEN2_VL_7B_PATH`
- `OpenGVLab/InternVL2-8B` -> `REVISE_INTERNVL2_8B_PATH`
- `lmms-lab/llava-onevision-qwen2-7b-ov` -> `REVISE_LLAVA_OV_7B_PATH`

After completion, source the generated model env file or set `REVISE_LOCAL_MODEL_PATH` explicitly
for each row-specific rerun.

Video-MME annotations are loaded from Hugging Face. For reproducible full evaluation, use the
official `lmms-lab/Video-MME` chunked video zips rather than the public YouTube URLs, since the
official repo notes that broken links have had to be replaced. The helper below downloads chunks
`videos_chunked_01.zip` through `videos_chunked_20.zip`, flattens them into
`REVISE_VIDEO_CACHE_DIR/videomme`, and deletes each zip after extraction by default:

```bash
python scripts/repro/submit_videomme_official_download_slurm.py \
  --submit \
  --partition normal \
  --concurrency 4 \
  --overwrite-existing
```

LVBench's current Hugging Face dataset includes official video chunks. Prefer those chunks for full
evaluation instead of live YouTube downloads:

```bash
python scripts/repro/submit_lvbench_official_download_slurm.py \
  --submit \
  --partition normal \
  --concurrency 4
```

The older LVBench repository instructions provide YouTube IDs plus a download script. Use this path
only as a fallback or for diagnosing individual source-video availability:

```bash
pip install yt-dlp
python scripts/repro/cache_hf_video_benchmark.py --dataset lvbench --max-videos 1 --dry-run
python scripts/repro/cache_hf_video_benchmark.py --dataset lvbench
```

The full cached-only runs require the complete video cache, not only a smoke sample. Use dry-runs and
coverage checks to size the cache work first, then chunk LVBench with `--start-idx` / `--max-videos`
if needed:

```bash
python scripts/repro/cache_hf_video_benchmark.py --dataset lvbench --dry-run
python scripts/repro/cache_hf_video_benchmark.py --dataset lvbench --start-idx 0 --max-videos 25
```

Check cache coverage before submitting cached-only full evaluations:

```bash
python scripts/repro/check_video_cache_coverage.py \
  --json outputs/repro_runs/video_cache_coverage.json \
  --md outputs/repro_runs/video_cache_coverage.md
```

To run LVBench cache fill through Slurm's non-GPU partition:

```bash
python scripts/repro/submit_video_cache_slurm.py \
  --dataset lvbench \
  --chunk-size 25 \
  --submit
```

The Video-MME prompt follows the official no-subtitle multiple-choice template by default. Use a
separate, documented run if you enable subtitle-based evaluation.

`prepare_downloaded_assets.py --videoespresso` extracts the VideoEspresso test videos only. The
train-video archive is much larger; merge/extract it only before SFT/RL reproduction:

```bash
python scripts/repro/prepare_downloaded_assets.py --videoespresso --videoespresso-train
```

NExT-QA raw videos and some annotation bundles are distributed through Google Drive by the official
repo. Clone the official repo for annotations, download the raw videos from the official link, then
export:

```bash
export REVISE_NEXTQA_ROOT=/path/to/NExT-QA
export REVISE_NEXTQA_VIDEO_ROOT=/path/to/NExT-QA/NExTVideo
export REVISE_NEXTQA_MAP_JSON=/path/to/NExT-QA/map_vid_vidorID.json
export REVISE_NEXTQA_TRAIN_CSV=/path/to/NExT-QA/nextqa/train.csv
export REVISE_NEXTQA_VAL_CSV=/path/to/NExT-QA/nextqa/val.csv
```

If Google Drive access is unavailable, this repo can use the `rhymes-ai/NeXTVideo` Hugging Face
copy of the official raw-video archive while still pulling annotations from the official NExT-QA
GitHub repository:

```bash
python scripts/repro/download_assets.py --nextqa-raw-hf
python scripts/repro/prepare_downloaded_assets.py --nextqa
source data/revise_assets/revise_env.sh
```

The older `VLM2Vec/nextqa` Hugging Face mirror can be used for validation/test-only evaluation
probes, but it is not sufficient for training reproduction because it does not include the full
train-video set:

```bash
python scripts/repro/download_assets.py --nextqa-hf-mirror
python scripts/repro/prepare_downloaded_assets.py --nextqa
source data/revise_assets/revise_env.sh
```

The NExT-QA evaluation loaders resolve both the official mapped subdirectory layout and the older
HF mirror's nested `NExTVideo/NExTVideo/<video_id>.mp4` layout. `doctor.py --strict` probes one
validation row and one training row to ensure the configured CSV/map/root resolves to concrete local
video files before GPU jobs are submitted.

Other common overrides:

```bash
export REVISE_VIDEOESPRESSO_ROOT=/path/to/VideoEspresso
export REVISE_VIDEOESPRESSO_TEST_JSON=/path/to/VideoEspresso/test_video/bench_hard.json
export REVISE_VIDEOESPRESSO_TEST_VIDEO_ROOT=/path/to/VideoEspresso/test_video
export REVISE_VIDEOESPRESSO_TRAIN_VIDEO_JSON=/path/to/VideoEspresso/train_video/videoespresso_train_video.json
export REVISE_EGOSCHEMA_VIDEO_ROOT=/path/to/EgoSchema/videos
export REVISE_EGOSCHEMA_JSON=/path/to/EgoSchema/pnp_subset_500.json
export REVISE_VIDEO_CACHE_DIR=/path/to/video_cache
export REVISE_QWEN25_VL_3B_PATH=/path/to/Qwen2.5-VL-3B-Instruct
export REVISE_QWEN25_VL_7B_PATH=/path/to/Qwen2.5-VL-7B-Instruct
export REVISE_QWEN2_VL_7B_PATH=/path/to/Qwen2-VL-7B-Instruct
export REVISE_INTERNVL2_8B_PATH=/path/to/InternVL2-8B
export REVISE_LLAVA_OV_7B_PATH=/path/to/LLaVA-OneVision-Qwen2-7B-OV
```

The public EgoSchema 500-answer subset can be materialized locally from the Hugging Face mirror:

```bash
python scripts/repro/prepare_egoschema_subset.py
```

For EgoSchema, smoke runs may use a partial local video cache, but full runs use the HF Subset
auto-download path unless every video referenced by `REVISE_EGOSCHEMA_JSON` is present locally.

To rerun a paper row with an exact non-Qwen2.5 local backbone, point the launcher at that snapshot:

```bash
export REVISE_LOCAL_MODEL_PATH=/path/to/exact/backbone
export REVISE_LOCAL_MODEL_ID=optional-served-model-name
```

Unset `REVISE_LOCAL_MODEL_PATH` to return to the default Qwen2.5-VL asset discovery.

For an OpenAI-compatible hosted model instead of local checkpoints:

```bash
export REVISE_API_BASE_URL=http://host:port/v1
export REVISE_MODEL_ID=model-name
export REVISE_API_KEY=dummy-or-real-key
```

## Preflight

Run preflight before submitting jobs:

```bash
python scripts/repro/doctor.py
python scripts/repro/doctor.py --strict
python scripts/repro/paper_suite.py list
python scripts/repro/paper_suite.py check --all --smoke
python scripts/repro/paper_suite.py check --all
```

`doctor.py` must show dataset/model or remote API availability before full GPU jobs are submitted.
`paper_suite.py check --all --smoke` applies smoke-run rules, while the full check requires
complete cached-only video coverage for Video-MME and LVBench.
On login nodes, `nvidia-smi` may be unavailable; that is expected and is not itself a blocker for
Slurm jobs.

## Slurm Execution

Create a smoke-test Slurm run directory without submitting:

```bash
python scripts/repro/submit_paper_suite_slurm.py \
  --experiment videoespresso_pnp \
  --smoke \
  --run-name smoke-videoespresso \
  --output-root outputs/repro_runs
```

Submit the evaluation smoke GPU matrix. In this launcher, `--all` intentionally means the
evaluation matrix only; training pipelines must be requested explicitly:

```bash
python scripts/repro/submit_paper_suite_slurm.py \
  --all \
  --smoke \
  --submit \
  --output-root outputs/repro_runs
```

Submit the full evaluation matrix after smoke jobs pass:

```bash
python scripts/repro/submit_paper_suite_slurm.py \
  --all \
  --submit \
  --partition gengpu \
  --gres gpu:a100:1 \
  --time 02:00:00 \
  --output-root outputs/repro_runs
```

The paper's LLaVA-OV-7B rows should use the Hugging Face local-model path, not the vLLM launcher,
because the local `lmms-lab/llava-onevision-qwen2-7b-ov` snapshot is a legacy
`LlavaQwenForCausalLM` checkpoint. The model card's supported path is LLaVA-NeXT
`load_pretrained_model(..., model_name="llava_qwen")`; keep a local LLaVA-NeXT checkout at
`$REVISE_LLAVA_NEXT_PATH` or `data/revise_assets/third_party/LLaVA-NeXT`. The current checked
source commit is `df179663ae8b83207df100a1f7af24caec633ff9`. Smoke-test those rows explicitly:

```bash
python scripts/repro/submit_paper_suite_slurm.py \
  --experiment videomme_pnp_hf \
  --experiment lvbench_pnp_hf \
  --smoke \
  --submit \
  --partition gengpu \
  --gres gpu:a100:1 \
  --time 01:30:00 \
  --mem 96G \
  --output-root outputs/repro_runs
```

For training jobs, increase resources explicitly, for example:

SFT teacher generation is configurable and model-neutral. Full training commands cap teacher
generation at `MAX_SAMPLES=8000`, while smoke commands keep `MAX_SAMPLES=4`. By default the helper
starts a local vLLM teacher from `TEACHER_MODEL_PATH`, falling back to `REVISE_QWEN25_VL_7B_PATH`
when unset. To use an external OpenAI-compatible teacher later, set both `TEACHER_BASE_URL` and
`TEACHER_MODEL_ID`. GPT-4o-specific API plumbing is not required for the current local SFT pipeline.

Default teacher log names are:

- `outputs/nextqa_teacher_train_log.jsonl`
- `outputs/videoespresso_teacher_train_log.jsonl`

```bash
python scripts/repro/submit_paper_suite_slurm.py \
  --experiment nextqa_train_pipeline \
  --experiment videoespresso_train_pipeline \
  --submit \
  --partition gengpu \
  --gres gpu:a100:4 \
  --time 12:00:00 \
  --mem 192G \
  --cpus-per-task 16 \
  --output-root outputs/repro_runs
```

For a training pipeline smoke test, keep the job on one GPU first. The smoke command generates only
four teacher samples and writes teacher logs, SFT parquet files, SFT checkpoints, and GRPO outputs
inside that run directory:

```bash
python scripts/repro/submit_paper_suite_slurm.py \
  --experiment nextqa_train_pipeline \
  --experiment videoespresso_train_pipeline \
  --smoke \
  --submit \
  --partition gengpu \
  --gres gpu:a100:1 \
  --cpus-per-task 8 \
  --time 02:00:00 \
  --output-root outputs/repro_runs
```

### Phase E: consume Phase A teacher JSONL instead of regenerating

When Phase A (`scripts/repro/submit_phase_a_teacher72b_slurm.py`) has produced
the high-quality 72B teacher JSONL, Phase E SFT+GRPO must consume that JSONL
rather than re-run a 7B teacher inline. The `_manual_*_pipeline` builders
read two env vars at render time and, when set, replace the inline teacher
generation step with a validated symlink to the Phase A path:

- `REVISE_NEXTQA_TEACHER_LOG_OVERRIDE`
- `REVISE_VIDEOESPRESSO_TEACHER_LOG_OVERRIDE`

The rendered sbatch validates the override file exists at job start and
fails loudly with a clear error if it does not, so a typo in the env var
path cannot silently produce a no-data training run.

Submit the SFT+GRPO training pipelines with the Phase A handoff:

```bash
PHASE_A_RUN=outputs/repro_runs/phase-a-teacher72b-awq-20260527
REVISE_NEXTQA_TEACHER_LOG_OVERRIDE=$PHASE_A_RUN/results/nextqa_teacher72b_train_log.jsonl \
REVISE_VIDEOESPRESSO_TEACHER_LOG_OVERRIDE=$PHASE_A_RUN/results/videoespresso_teacher72b_train_log.jsonl \
python scripts/repro/submit_paper_suite_slurm.py \
  --experiment nextqa_train_pipeline \
  --experiment videoespresso_train_pipeline \
  --submit \
  --partition gengpu \
  --gres gpu:a100:4 \
  --time 48:00:00 \
  --mem 192G \
  --cpus-per-task 16 \
  --output-root outputs/repro_runs \
  --run-name phase-e-sft-grpo-$(date -u +%Y%m%d)
```

Then submit the LVBench reward-ablation rows (rows 1-6; row 7 deliberately
aliases row 2's checkpoint dir per `paper_suite.py`, so it is not
re-submitted as an independent training run):

```bash
python scripts/repro/submit_paper_suite_slurm.py \
  --experiment lvbench_reward_ablation_row1_full_beta1_tau2 \
  --experiment lvbench_reward_ablation_row2_no_conf_beta1_tau2 \
  --experiment lvbench_reward_ablation_row3_no_sum_beta1_tau2 \
  --experiment lvbench_reward_ablation_row4_no_stop_beta1_tau2 \
  --experiment lvbench_reward_ablation_row5_stop_design_beta0_tau1 \
  --experiment lvbench_reward_ablation_row6_stop_design_beta0_tau3 \
  --submit \
  --partition gengpu \
  --gres gpu:a100:4 \
  --time 48:00:00 \
  --mem 192G \
  --cpus-per-task 16 \
  --output-root outputs/repro_runs \
  --run-name phase-e-lvbench-reward-ablation-$(date -u +%Y%m%d)
```

The LVBench reward-ablation rows train directly from the
`lmms-lab/LLaVA-OneVision-1.5-4B-Instruct` HF snapshot and do NOT depend on
Phase A teacher data; they can be submitted in parallel with the SFT+GRPO
pipelines above. Row 0 (`Base no RL`) is a plug-and-play eval handled by
the Phase D matrix via `lvbench_pnp_hf`.

Each run directory contains:

- `manifest.json`: git snapshot, environment snapshot, asset discovery, blocked reasons, commands,
  Slurm job IDs.
- `jobs/*.sbatch`: exact Slurm scripts.
- `slurm/*.out` and `slurm/*.err`: scheduler logs.
- `logs/doctor.<experiment>.<jobid>.json`: per-job preflight snapshot.
- `results/*.summary.json`: metric summaries.
- `results/*.jsonl`: prompt/conversation traces and raw outputs.
- `results/*.server.log`: local vLLM/SGLang server logs when the job starts an inference server.

Collect and validate run-level summaries with:

```bash
python scripts/repro/collect_run_summaries.py outputs/repro_runs/<run-name> --write
```

Training pipeline runs can be summarized after the job finishes:

```bash
python scripts/repro/summarize_training_pipeline.py \
  outputs/repro_runs/<run-name> \
  --experiment nextqa_train_pipeline \
  --job-id <slurm-job-id>
python scripts/repro/collect_run_summaries.py outputs/repro_runs/<run-name> --write
```

The collector marks missing summaries, zero-sample runs, and zero-model-call runs as invalid so
failed orchestration runs do not get copied into paper tables.

After all relevant runs are valid, build the auditable result snapshot used for README and paper
table edits:

```bash
python scripts/repro/build_verified_result_tables.py \
  outputs/repro_runs/<full-eval-run> \
  --out-json outputs/repro_runs/verified_results.json \
  --out-md outputs/repro_runs/verified_results.md
```

Pass multiple run directories when a repaired rerun supersedes a cancelled or invalid job. The
snapshot keeps each reported number tied to its `summary.json` and JSONL trace path, and also emits
category breakdowns from the prompt/output logs when row-level metadata is available.

## VideoEspresso Evaluation Rules

The corrected VideoEspresso path follows the official close-ended evaluator:

- Keep the row-level `task`, `question`, `options`, `correct_answer`, and `evidence` fields.
- Format options as `(A)` through `(D)`.
- Ask the model to select only an option letter.
- Score by normalizing the predicted answer and ground truth to an option letter.
- Do not include `evidence` unless `--videoespresso-with-evidence` is explicitly enabled.

The PNP script appends the REVISE protocol requirement to the official benchmark query so that
`<think>`, `<summarize>`, and `<answer>` remain logged and parseable.

## Result Update Procedure

1. Run `doctor.py --strict` with all assets configured.
2. Run smoke jobs through Slurm and inspect JSONL prompt traces for every benchmark family.
3. Run the full matrix through `submit_paper_suite_slurm.py --all --submit`.
4. Verify every `manifest.json` entry is unblocked and every expected `summary.json` exists.
5. Aggregate the new metrics with `scripts/repro/collect_run_summaries.py`.
6. Generate `verified_results.json` / `verified_results.md` with
   `scripts/repro/build_verified_result_tables.py`.
7. Update README and Overleaf tables only from verified summaries.
8. Keep the run manifest path in the paper/repo notes so the numbers remain auditable.
