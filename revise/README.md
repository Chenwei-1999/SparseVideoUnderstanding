# REVISE: Sparse Video Understanding (NExT-QA)

This directory provides a REVISE-style multi-round agent loop, dataset loader, and configs for NExT-QA experiments.

For paper-wide reproduction across NExT-QA / VideoEspresso / EgoSchema / LVBench / Video-MME, use:

```bash
python scripts/doctor.py
python scripts/paper_suite.py check --all
```

## Pipeline map

Each benchmark has one self-contained plug-and-play script implementing the
REVISE multi-round loop (`<think>` → `<summarize>` P/O/H/U/R state →
`<select>` new frames / `<answer>`). They are CLI executables, driven by the
`run_*.sh` wrappers and orchestrated as subprocesses by `scripts/paper_suite.py`.

| Script | Benchmark(s) | Backend | Driven by |
|---|---|---|---|
| `plug_and_play_nextqa_vllm.py` | NExT-QA (+ SFT teacher data) | vLLM server | `run_generate_teacher_data.sh`, `paper_suite.py` |
| `plug_and_play_egoschema_vllm.py` | EgoSchema, VideoEspresso | vLLM server | `run_generate_teacher_data_videoespresso.sh`, `paper_suite.py` |
| `plug_and_play_videomme_lvbench_vllm.py` | Video-MME, LVBench | vLLM server | `paper_suite.py` |
| `plug_and_play_lvbench_hf.py` | Video-MME, LVBench | in-process HF `transformers` | `paper_suite.py` |

Shared code lives in `pnp_utils.py` (frame extraction, tag parsing, the vLLM
launch command, server lifecycle) and `pnp_prompts.py` (system prompts). The
two `videomme_lvbench` scripts keep separate `MCVideoSample`/loaders on purpose
— their row-filtering and fields differ per backend. The behavior of the
shared launch and frame-sampling helpers is pinned by
`tests/test_pnp_characterization.py` (run with
`python -m unittest tests.test_pnp_characterization`).

## Data
- Default ignored asset root: `$PWD/data/revise_assets`
- NExT-QA root: `REVISE_NEXTQA_ROOT`
- Videos: `REVISE_NEXTQA_VIDEO_ROOT`
- Mapping file: `REVISE_NEXTQA_MAP_JSON`
- CSVs: `REVISE_NEXTQA_TRAIN_CSV` and `REVISE_NEXTQA_VAL_CSV`

Run `python scripts/download_assets.py --all --dry-run` to generate the expected asset layout.

## Plug-and-play evaluation (REVISE multi-round)

```bash
ENGINE=sglang ./revise/run_revise_nextqa_eval.sh
```

vLLM backend:

```bash
ENGINE=vllm ./revise/run_revise_nextqa_eval.sh \
  --config-name revise_nextqa_eval_vllm
```

Notes:
- Uses `Qwen/Qwen2.5-VL-7B-Instruct` by default.
- Settings: `max_frames_per_round=3`, `max_rounds=4`, `temperature=0.2`, `top_p=0.9`, `max_response_length=256`.

## SFT teacher data

The SFT teacher-generation helpers use the configured teacher explicitly:

```bash
TEACHER_MODEL_PATH=/path/to/local/teacher \
MAX_SAMPLES=8000 \
./revise/run_generate_teacher_data.sh
```

Defaults are intentionally model-neutral:

- NExT-QA log: `outputs/nextqa_teacher_train_log.jsonl`
- VideoEspresso log: `outputs/videoespresso_teacher_train_log.jsonl`
- full teacher cap: `MAX_SAMPLES=8000`

If `TEACHER_MODEL_PATH` is not set, the scripts fall back to the local
`REVISE_QWEN25_VL_7B_PATH` snapshot. For an OpenAI-compatible external teacher, set
`TEACHER_BASE_URL` and `TEACHER_MODEL_ID`.

## Reinforcement fine-tuning (GRPO + EAGER-style reward)

```bash
ENGINE=sglang ./revise/run_revise_nextqa_grpo.sh
```

## Minimal smoke test (4 GPUs, 16 samples, 2 rounds)

```bash
ENGINE=sglang ./revise/run_revise_nextqa_smoke.sh
```

This uses 4 GPUs (CUDA_VISIBLE_DEVICES defaults to 0,1,2,3), `max_samples=16`, and short round/length settings to
validate the loop quickly. Set `ENGINE=vllm` to smoke-test vLLM.

Default training settings follow the paper:
- `lr=1e-6`, `kl_loss_coef=0.001`, `entropy_coeff=0`.
- `max_prompt_length=8192`, `max_response_length=512`, `train_batch_size=8`.
- `max_frames_per_round=3`, `max_rounds=4`.

## Customization
- Change VLM backbone via `actor_rollout_ref.model.path=...`.
- Override dataset paths or batch sizes via CLI flags.
- REVISE-specific settings live under `actor_rollout_ref.rollout.revise`.

## Notes on EAGER reward
The included `eager_videoqa` reward approximates EAGER when full margin signals are not available. If you compute per-round
confidence gains (margins) externally, pass them via `extra_info['revise']` to fully match the paper.
