# REVISE: Sparse Video Understanding (NExT-QA)

This directory provides a REVISE-style multi-round agent loop, dataset loader, and configs for NExT-QA experiments.

For paper-wide reproduction across NExT-QA / VideoEspresso / EgoSchema / LVBench / Video-MME, use:

```bash
python scripts/repro/doctor.py
python scripts/repro/paper_suite.py check --all
```

The paper-specific workflow is documented in [docs/REPRODUCE.md](../../docs/REPRODUCE.md).

## Data
- Default ignored asset root: `$PWD/data/revise_assets`
- NExT-QA root: `REVISE_NEXTQA_ROOT`
- Videos: `REVISE_NEXTQA_VIDEO_ROOT`
- Mapping file: `REVISE_NEXTQA_MAP_JSON`
- CSVs: `REVISE_NEXTQA_TRAIN_CSV` and `REVISE_NEXTQA_VAL_CSV`

Run `python scripts/repro/download_assets.py --all --dry-run` to generate the expected layout and
`docs/REPRODUCE.md` for full paper-wide setup.

## Plug-and-play evaluation (REVISE multi-round)

```bash
ENGINE=sglang ./examples/revise/run_revise_nextqa_eval.sh
```

vLLM backend:

```bash
ENGINE=vllm ./examples/revise/run_revise_nextqa_eval.sh \
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
./examples/revise/run_generate_teacher_data.sh
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
ENGINE=sglang ./examples/revise/run_revise_nextqa_grpo.sh
```

## Minimal smoke test (4 GPUs, 16 samples, 2 rounds)

```bash
ENGINE=sglang ./examples/revise/run_revise_nextqa_smoke.sh
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
