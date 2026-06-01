# REVISE: Sparse Video Understanding (NExT-QA)

This directory provides a REVISE-style multi-round agent loop, dataset loader, and configs for NExT-QA experiments.

For paper-wide reproduction across NExT-QA / VideoEspresso / EgoSchema / LVBench / Video-MME, use:

```bash
python scripts/doctor.py
python scripts/paper_suite.py check --all
```

## NExT-QA Table 4

The paper Table 4 rows are first-class `paper_suite.py` experiments:

```bash
python scripts/paper_suite.py check \
  --experiment nextqa_table4_direct \
  --experiment nextqa_table4_pnp \
  --experiment nextqa_table4_sft \
  --experiment nextqa_table4_rl_after_sft

python scripts/paper_suite.py report \
  --experiment nextqa_table4_pnp \
  --experiment nextqa_table4_rl_after_sft \
  --output-dir outputs/repro_runs/table4_nextqa \
  --output-json outputs/repro_runs/table4_nextqa/report.json

python scripts/paper_suite.py run --experiment nextqa_table4_pnp \
  --output-dir outputs/repro_runs/table4_nextqa

# Export WANDB_API_KEY in your shell before RL training if online logging is desired.
python scripts/paper_suite.py run --experiment nextqa_table4_rl_after_sft \
  --output-dir outputs/repro_runs/table4_nextqa
```

The report output separates readiness from evidence: `readiness_status` only
means the command has no known preflight blockers, `paper_target_metrics` are the
paper Table 4 reference numbers, and `observed_metrics` are paper-comparable
only when `verification_status` is `observed`. A summary collected while
preflight blockers remain is reported as `observed_with_blockers`.

Current audit status is documented in `docs/NEXTQA_TABLE4_REPRO.md`. In short:
the corrected RFT setting is runnable and observed but still below the paper
target; the direct/PnP rows require the original Table 4 baseline checkpoint via
`REVISE_NEXTQA_TABLE4_BASE_MODEL`. The current public Qwen2.5-VL-3B-Instruct
snapshot is much stronger than the paper direct row, so it is useful for audits
but not paper-comparable for those two rows.

Full Table 4 runs use the official NExT-QA split, 4 GPUs, `max_rounds=4`,
`max_frames_per_round=3`, `temperature=0.2`, `top_p=0.9`, and the strict paper
protocol: select rounds emit `<think><summarize><select>`, while answer rounds
emit `<think><answer>`. NExT-QA selection indices are interpreted on a 1-fps
timeline and mapped to raw video frames only for image extraction.

Run `python scripts/doctor.py --scope nextqa` before Table 4 jobs. Use
`--scope paper` only when checking every dataset in the full paper.

## Pipeline map

Use `pnp_cli.py` for plug-and-play and one-shot evaluation. The top-level
`revise/` directory should stay small; new code should separate the axes:

- `datasets/`: NExT-QA, EgoSchema, VideoEspresso, Video-MME, LVBench sample loading and task adapters.
- `backends/`: vLLM HTTP, HuggingFace in-process, and future inference runtimes.
- `pnp/`: shared REVISE loop, prompts, registry, harness, and utilities.
- `benchmarks/`: specialized evaluators that are not just dataset/backend/mode selection.

| Entry point | Benchmark(s) | Runtime / setting | Driven by |
|---|---|---|---|
| `pnp_cli.py --dataset nextqa --backend vllm_http` | NExT-QA | vLLM REVISE / one-shot | `paper_suite.py` |
| `pnp_cli.py --dataset videoespresso --backend vllm_http` | VideoEspresso | vLLM REVISE / one-shot | `paper_suite.py` |
| `pnp_cli.py --dataset {videomme,lvbench} --backend vllm_http` | Video-MME, LVBench | vLLM REVISE / one-shot | `paper_suite.py` |
| `pnp_cli.py --dataset {videomme,lvbench} --backend hf_inprocess` | Video-MME, LVBench | in-process `transformers` | `paper_suite.py` |
| `benchmarks/nextqa_caption_vllm.py` | NExT-QA caption baseline | vLLM caption-only | `paper_suite.py` |

Shared code lives in `pnp/utils.py` (frame extraction, tag parsing, the vLLM
launch command, server lifecycle) and `pnp/prompts.py` (system prompts). Use
`--dataset lvbench --backend hf_inprocess` when the runtime is HuggingFace; do
not introduce dataset names that encode the backend. The behavior of the shared
launch and frame-sampling helpers is pinned by
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
python scripts/paper_suite.py run --experiment nextqa_table4_pnp \
  --output-dir outputs/repro_runs/table4_nextqa
```

Notes:
- The Table 4 PnP row requires `REVISE_NEXTQA_TABLE4_BASE_MODEL`; the generic
  `nextqa_pnp` row can use public/local Qwen fallbacks for development.
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

When no teacher checkpoint/API is available, use the GPT-5-mini Batch bootstrap
utility to create train-split traces and convert them to the same teacher-log
schema. The bootstrap asks for variable-length traces: answer immediately when
the initial evidence is enough, otherwise use one or more `<select>` rounds and
stop at the first `<answer>` round. This matters for Table 4 because fixed
max-round SFT data teaches the RL policy to keep selecting. The default request
omits `service_tier`; add `--service-tier flex` only when Flex is available for
your selected model/project.

```bash
python scripts/nextqa_openai_teacher_batch.py prepare --max-samples 8000
python scripts/nextqa_openai_teacher_batch.py submit

# After the batch finishes and its output JSONL is downloaded:
python scripts/nextqa_openai_teacher_batch.py convert \
  --batch-output outputs/openai_batch/nextqa_teacher_batch_output.jsonl \
  --output-log outputs/nextqa_teacher_train_log.jsonl
```

The converted log can be passed directly to `revise/generate_sft_data.py` or
used through `revise/run_revise_nextqa_sft.sh`. For the recovered Table 4 RFT
setting, keep the first-action prior balanced enough to avoid one-round
collapse:

```bash
SFT_INPUT=outputs/nextqa_teacher_train_log.jsonl \
SFT_GENERATE_ARGS="--max-rounds 4 --min-first-select-ratio 0.45" \
./revise/run_revise_nextqa_sft.sh
```

## Reinforcement fine-tuning (GRPO + EAGER-style reward)

```bash
python scripts/paper_suite.py run --experiment nextqa_table4_rl_after_sft \
  --output-dir outputs/repro_runs/table4_nextqa
```

## Minimal smoke test (1 sample, 2 rounds)

```bash
python scripts/paper_suite.py run --experiment nextqa_table4_pnp --smoke
```

Smoke mode validates the loop quickly and intentionally reduces resource use.
Run without `--smoke` for the full 4-GPU paper setting.

Default training settings follow the audited NExT-QA RFT setting:
- `lr=1e-6`, `kl_loss_coef=0.001`, `entropy_coeff=0`.
- `max_prompt_length=8192`, `max_response_length=512`, `train_batch_size=8`.
- `max_frames_per_round=3`, `max_rounds=4`.
- after-SFT GRPO uses `stop_bonus_beta=0.0` because the positive early-stop
  bonus collapsed the policy toward one-round answers in local audits.

## Customization
- Change VLM backbone via `actor_rollout_ref.model.path=...`.
- Override dataset paths or batch sizes via CLI flags.
- REVISE-specific settings live under `actor_rollout_ref.rollout.revise`.

## Notes on EAGER reward
The included `eager_videoqa` reward approximates EAGER when full margin signals are not available. If you compute per-round
confidence gains (margins) externally, pass them via `extra_info['revise']` to fully match the paper.
