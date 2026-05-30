"""Shared outer evaluation driver for plug-and-play REVISE launchers."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Optional

import revise.pnp_engine as pnp_engine
from revise.pnp_protocols import Backend, Dataset, LoopConfig, RunStats
from revise.pnp_utils import maybe_log_jsonl, shard_by_video, wandb_log as _default_wandb_log


def _count_file_lines(path: str) -> int:
    if not path or not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _suffix_path(path: str, *, suffix: str) -> str:
    root, ext = os.path.splitext(path)
    if ext:
        return f"{root}{suffix}{ext}"
    return f"{path}{suffix}"


def _maybe_init_run(args: Any, run: Any, run_config: dict[str, Any] | None) -> Any:
    if run is not None or run_config is None:
        return run
    maybe_init = getattr(args, "_pnp_maybe_init_wandb", None)
    if maybe_init is None:
        return None
    return maybe_init(args, run_config)


def _wandb_log(args: Any, run: Any, metrics: dict[str, Any], *, step: int) -> None:
    logger = getattr(args, "_pnp_wandb_log", _default_wandb_log)
    logger(run, metrics, step=step)


def _auto_suffix_outputs(args: Any) -> None:
    num_shards = max(1, int(getattr(args, "num_shards", 1)))
    shard_idx = int(getattr(args, "shard_idx", 0))
    if num_shards <= 1:
        return
    suffix = f".shard{shard_idx}of{num_shards}"
    if getattr(args, "log_jsonl", None) and suffix not in args.log_jsonl:
        args.log_jsonl = _suffix_path(args.log_jsonl, suffix=suffix)
    if getattr(args, "summary_json", None) and suffix not in args.summary_json:
        args.summary_json = _suffix_path(args.summary_json, suffix=suffix)


def _index_shard(samples: list[Any], args: Any) -> list[Any]:
    num_shards = max(1, int(getattr(args, "num_shards", 1)))
    shard_idx = int(getattr(args, "shard_idx", 0))
    if num_shards <= 1:
        return samples
    if not (0 <= shard_idx < num_shards):
        raise ValueError(f"--shard-idx must be in [0, {num_shards}) (got {shard_idx}).")
    return [s for i, s in enumerate(samples) if (i % num_shards) == shard_idx]


def _video_shard(samples: list[Any], args: Any) -> list[Any]:
    return shard_by_video(samples, int(getattr(args, "num_shards", 1)), int(getattr(args, "shard_idx", 0)))


def _long_resume_completed(log_jsonl: str, *, key: str) -> int:
    if not log_jsonl or not os.path.exists(log_jsonl):
        return 0
    seen_samples: set[str] = set()
    with open(log_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            sid = obj.get("sample_id")
            if key == "raw_answer" and sid and "<answer>" in str(obj.get("raw_output", "")).lower():
                seen_samples.add(str(sid))
            elif key == "answer_letter" and sid and obj.get("answer_letter"):
                seen_samples.add(str(sid))
    return len(seen_samples)


def _run_nextqa(
    samples: list[Any],
    *,
    dataset: Dataset,
    backend: Backend,
    cfg: LoopConfig,
    stats: RunStats,
    rng: Any,
    base_url: str,
    model_id: str,
    run: Any,
    args: Any,
) -> dict[str, Any]:
    _auto_suffix_outputs(args)
    samples = _index_shard(samples, args)
    if not samples:
        raise RuntimeError(
            f"No samples selected for shard {int(getattr(args, 'shard_idx', 0))}/{int(getattr(args, 'num_shards', 1))} "
            "(check --max-samples / sharding)."
        )

    resume_completed = 0
    correct = 0
    total_rounds = 0
    if getattr(args, "resume_from_log", False) and getattr(args, "log_jsonl", None) and os.path.exists(args.log_jsonl):
        resume_loader = getattr(args, "_pnp_load_progress_from_log", None)
        if resume_loader is not None:
            resume_completed, correct, total_rounds = resume_loader(args.log_jsonl, max_rounds=args.max_rounds)
            resume_completed = min(resume_completed, len(samples))

    stats.processed = resume_completed
    stats.correct = correct
    stats.total_rounds = total_rounds

    run_config = getattr(args, "_pnp_run_config", None)
    if run_config is not None:
        run_config["initial_completed"] = resume_completed
        run_config["log_jsonl"] = args.log_jsonl
    run = _maybe_init_run(args, run, run_config)

    start_eval = time.time()
    for sample in samples[resume_completed:]:
        stats.processed += 1
        try:
            outcome = pnp_engine.run_sample(
                sample,
                dataset=dataset,
                backend=backend,
                cfg=cfg,
                stats=stats,
                rng=rng,
                base_url=base_url,
                model_id=model_id,
                run=run,
            )
            if outcome.answer_letter is not None and dataset.is_correct(sample, outcome.answer_letter):
                stats.correct += 1
            stats.total_frames_used += len(outcome.seen_frames)
        except Exception as e:
            stats.failed += 1
            stats.total_rounds += args.max_rounds
            restart_cb = getattr(args, "_pnp_restart_on_exception", None)
            restart_exc = getattr(args, "_pnp_restart_exception_type", None)
            if restart_cb is not None and restart_exc is not None and isinstance(e, restart_exc):
                new_model_id = restart_cb()
                if new_model_id:
                    model_id = new_model_id

        processed = stats.processed
        if args.progress_interval > 0 and processed % args.progress_interval == 0:
            elapsed = time.time() - start_eval
            acc = stats.correct / max(1, processed)
            avg_rounds = stats.total_rounds / max(1, processed)
            calls_per_sample = stats.total_model_calls / max(1, processed)
            avg_frames_used = stats.total_frames_used / max(1, processed)
            print(
                f"[{processed}/{len(samples)}] acc={acc:.4f} avg_rounds={avg_rounds:.3f} "
                f"failed={stats.failed} invalid={stats.invalid_outputs} retries={stats.total_retries} "
                f"calls={stats.total_model_calls} calls/sample={calls_per_sample:.2f} elapsed_s={elapsed:.1f}",
                flush=True,
            )
            _wandb_log(
                args,
                run,
                {
                    "eval/acc": acc,
                    "eval/avg_rounds": avg_rounds,
                    "eval/avg_effective_rounds": stats.effective_rounds_total / max(1, processed),
                    "eval/avg_frames_used": avg_frames_used,
                    "eval/failed": stats.failed,
                    "eval/processed": processed,
                    "eval/elapsed_s": elapsed,
                    "eval/invalid_outputs": stats.invalid_outputs,
                    "eval/invalid_action_terminated": stats.invalid_action_terminated,
                    "eval/total_retries": stats.total_retries,
                    "eval/total_model_calls": stats.total_model_calls,
                    "eval/fallback_frames_used": stats.fallback_frames_used,
                    "eval/calls_per_sample": calls_per_sample,
                },
                step=processed,
            )

    processed = stats.processed
    acc = stats.correct / max(1, processed)
    avg_rounds = stats.total_rounds / max(1, processed)
    elapsed = time.time() - start_eval
    prompt_log_lines = _count_file_lines(args.log_jsonl) if args.log_jsonl else 0
    prompt_log_bytes = os.path.getsize(args.log_jsonl) if args.log_jsonl and os.path.exists(args.log_jsonl) else 0
    avg_frames_used = stats.total_frames_used / max(1, processed)
    results = {
        "samples": processed,
        "correct": stats.correct,
        "accuracy": acc,
        "total_rounds": stats.total_rounds,
        "avg_rounds": avg_rounds,
        "total_frames_used": stats.total_frames_used,
        "avg_frames_used": avg_frames_used,
        "total_effective_rounds": stats.effective_rounds_total,
        "avg_effective_rounds": stats.effective_rounds_total / max(1, processed),
        "failed": stats.failed,
        "elapsed_s": elapsed,
        "prompt_log_lines": prompt_log_lines,
        "prompt_log_bytes": prompt_log_bytes,
        "invalid_outputs": stats.invalid_outputs,
        "invalid_action_terminated": stats.invalid_action_terminated,
        "total_retries": stats.total_retries,
        "total_model_calls": stats.total_model_calls,
        "fallback_frames_used": stats.fallback_frames_used,
    }
    print(json.dumps(results, indent=2))
    _wandb_log(
        args,
        run,
        {
            "eval/final_acc": acc,
            "eval/final_avg_rounds": avg_rounds,
            "eval/final_avg_effective_rounds": stats.effective_rounds_total / max(1, processed),
            "eval/final_avg_frames_used": avg_frames_used,
            "eval/final_failed": stats.failed,
            "eval/final_elapsed_s": elapsed,
            "eval/prompt_log_lines": prompt_log_lines,
            "eval/prompt_log_bytes": prompt_log_bytes,
            "eval/final_invalid_outputs": stats.invalid_outputs,
            "eval/final_invalid_action_terminated": stats.invalid_action_terminated,
            "eval/final_total_retries": stats.total_retries,
            "eval/final_total_model_calls": stats.total_model_calls,
            "eval/final_fallback_frames_used": stats.fallback_frames_used,
        },
        step=processed,
    )
    if run is not None:
        run.summary["prompt_log_jsonl"] = args.log_jsonl
        run.summary["prompt_log_lines"] = prompt_log_lines
        run.summary["prompt_log_bytes"] = prompt_log_bytes
        run.summary["final_acc"] = acc
        run.summary["final_avg_rounds"] = avg_rounds
        run.summary["final_avg_effective_rounds"] = stats.effective_rounds_total / max(1, processed)
        run.summary["final_avg_frames_used"] = avg_frames_used
        run.summary["final_failed"] = stats.failed
        run.summary["invalid_outputs"] = stats.invalid_outputs
        run.summary["invalid_action_terminated"] = stats.invalid_action_terminated
        run.summary["total_retries"] = stats.total_retries
        run.summary["total_model_calls"] = stats.total_model_calls
        run.summary["fallback_frames_used"] = stats.fallback_frames_used
        run.finish()

    if args.summary_json:
        summary_dir = os.path.dirname(args.summary_json)
        if summary_dir:
            os.makedirs(summary_dir, exist_ok=True)
        summary = dict(getattr(args, "_pnp_summary_payload", {}) or {})
        summary.update(
            {
                "results": results,
                "prompt_log_jsonl": args.log_jsonl,
                "prompt_log_lines": prompt_log_lines,
                "prompt_log_bytes": prompt_log_bytes,
                "wandb": {
                    "enabled": bool(getattr(args, "use_wandb", False)),
                    "mode": getattr(args, "wandb_mode", "")
                    or os.getenv("WANDB_MODE")
                    or ("online" if os.getenv("WANDB_API_KEY") else "offline"),
                    "project": getattr(args, "wandb_project", None),
                    "entity": getattr(args, "wandb_entity", None),
                    "name": getattr(args, "wandb_name", None),
                    "group": getattr(args, "wandb_group", None),
                    "id": getattr(run, "id", None),
                    "run_dir": getattr(run, "dir", None),
                    "url": getattr(run, "url", None),
                },
                "command": "python " + " ".join(os.sys.argv),
            }
        )
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    return results


def _run_egoschema(
    samples: list[Any],
    *,
    dataset: Dataset,
    backend: Backend,
    cfg: LoopConfig,
    stats: RunStats,
    rng: Any,
    base_url: str,
    model_id: str,
    run: Any,
    args: Any,
) -> dict[str, Any]:
    samples = _index_shard(samples, args)
    if not samples:
        raise RuntimeError("No samples loaded (check dataset source, JSON, video cache, or HF video availability).")

    log_jsonl = getattr(args, "log_jsonl", None)
    if log_jsonl:
        os.makedirs(os.path.dirname(log_jsonl) or ".", exist_ok=True)

    done_ids: set[str] = set()
    if log_jsonl and getattr(args, "resume_from_log", False) and os.path.exists(log_jsonl):
        with open(log_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                sid = str(rec.get("sample_id") or "")
                if sid and rec.get("done"):
                    done_ids.add(sid)

    summary = getattr(args, "_pnp_summary_payload", {}) or {}
    run = _maybe_init_run(args, run, summary)
    start_time = time.time()
    legacy_total_rounds = 0

    for sample in samples:
        if sample.sample_id in done_ids:
            continue
        stats.processed += 1
        try:
            pre_sample = getattr(args, "_pnp_pre_sample", None)
            if pre_sample is not None:
                new_model_id = pre_sample()
                if new_model_id:
                    model_id = new_model_id
            outcome = pnp_engine.run_sample(
                sample,
                dataset=dataset,
                backend=backend,
                cfg=cfg,
                stats=stats,
                rng=rng,
                base_url=base_url,
                model_id=model_id,
                run=run,
            )
        except Exception as e:
            stats.failed += 1
            if log_jsonl:
                with open(log_jsonl, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "sample_id": sample.sample_id,
                                "qid": sample.qid,
                                "video_path": sample.video_path,
                                "question": sample.question,
                                "options": sample.choices,
                                "task": sample.task,
                                "evidence": sample.evidence,
                                "dataset_name": getattr(args, "dataset_name", "egoschema"),
                                "done": True,
                                "error": f"{type(e).__name__}: {e}",
                                "error_stage": "shared_engine",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            restart_cb = getattr(args, "_pnp_restart_on_exception", None)
            restart_exc = getattr(args, "_pnp_restart_exception_type", None)
            if restart_cb is not None and restart_exc is not None and isinstance(e, restart_exc):
                new_model_id = restart_cb()
                if new_model_id:
                    model_id = new_model_id
            continue

        if outcome.answer_letter is None:
            if not outcome.terminated_invalid_action:
                stats.failed += 1
        else:
            if dataset.is_correct(sample, outcome.answer_letter):
                stats.correct += 1
            legacy_total_rounds += int(min(int(outcome.round_idx), int(args.max_rounds)))

        if log_jsonl:
            with open(log_jsonl, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "sample_id": sample.sample_id,
                            "qid": sample.qid,
                            "video_path": sample.video_path,
                            "question": sample.question,
                            "options": sample.choices,
                            "answer_gt": chr(ord("A") + int(sample.answer_idx)),
                            "task": sample.task,
                            "evidence": sample.evidence,
                            "dataset_name": getattr(args, "dataset_name", "egoschema"),
                            "round": int(min(int(outcome.round_idx), int(args.max_rounds))),
                            "done": True,
                            "final_answer": outcome.answer_letter,
                            "illegal_action": bool(outcome.terminated_invalid_action),
                            "terminated_reason": outcome.terminated_reason,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        total = stats.processed
        failed = stats.failed
        if args.progress_interval > 0 and total % args.progress_interval == 0:
            acc = stats.correct / max(1, total - failed)
            avg_r = legacy_total_rounds / max(1, total - failed)
            msg = (
                f"[progress] {total}/{len(samples)} done | acc={acc:.3f} avg_rounds={avg_r:.2f} "
                f"failed={failed} invalid={stats.invalid_outputs} calls={stats.total_model_calls}"
            )
            print(msg, flush=True)
            _wandb_log(
                args,
                run,
                {
                    "progress/samples": total,
                    "progress/failed": failed,
                    "metrics/accuracy": acc,
                    "metrics/avg_rounds": avg_r,
                    "debug/invalid_outputs": stats.invalid_outputs,
                    "debug/total_model_calls": stats.total_model_calls,
                },
                step=total,
            )

    elapsed_s = time.time() - start_time
    total = stats.processed
    failed = stats.failed
    acc = stats.correct / max(1, total - failed) if total > failed else 0.0
    avg_rounds = legacy_total_rounds / max(1, total - failed) if total > failed else 0.0
    results = {
        "samples": total,
        "accuracy": acc,
        "avg_rounds": avg_rounds,
        "failed": failed,
        "elapsed_s": elapsed_s,
        "invalid_outputs": stats.invalid_outputs,
        "total_retries": stats.total_retries,
        "total_model_calls": stats.total_model_calls,
        "fallback_frames_used": stats.fallback_frames_used,
    }
    summary["results"] = results
    summary["log_jsonl"] = log_jsonl

    if log_jsonl and os.path.exists(log_jsonl):
        try:
            prompt_lines = _count_file_lines(log_jsonl)
            prompt_bytes = os.path.getsize(log_jsonl)
        except Exception:
            prompt_lines = None
            prompt_bytes = None
        summary["prompt_log_lines"] = prompt_lines
        summary["prompt_log_bytes"] = prompt_bytes

    if run is not None:
        summary["wandb"] = {
            "enabled": True,
            "mode": run.settings.mode,
            "project": run.project,
            "entity": getattr(run, "entity", None),
            "name": run.name,
            "group": getattr(run, "group", None),
            "id": run.id,
            "run_dir": run.dir,
            "url": getattr(run, "url", None),
        }
        run.log(
            {
                "final/samples": total,
                "final/failed": failed,
                "final/accuracy": acc,
                "final/avg_rounds": avg_rounds,
                "final/invalid_outputs": stats.invalid_outputs,
                "final/total_model_calls": stats.total_model_calls,
                "final/fallback_frames_used": stats.fallback_frames_used,
            }
        )
        run.finish()

    if getattr(args, "summary_json", None):
        os.makedirs(os.path.dirname(args.summary_json) or ".", exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps({"results": results, "summary_json": getattr(args, "summary_json", None), "log_jsonl": log_jsonl}, indent=2))
    return results


def _run_long_vllm(
    samples: list[Any],
    *,
    dataset: Dataset,
    backend: Backend,
    cfg: LoopConfig,
    stats: RunStats,
    rng: Any,
    base_url: str,
    model_id: str,
    run: Any,
    args: Any,
) -> dict[str, Any]:
    samples = _video_shard(samples, args)
    samples.sort(key=lambda s: (s.video_key, s.uid))
    if not samples:
        raise SystemExit("No samples selected (check --split/--max-samples/--sharding).")
    _auto_suffix_outputs(args)

    resume_completed = _long_resume_completed(getattr(args, "log_jsonl", ""), key="raw_answer") if getattr(args, "resume_from_log", False) else 0
    if resume_completed > 0:
        print(f"[resume] detected {resume_completed} completed samples in {args.log_jsonl}")
        samples = samples[resume_completed:]

    run_config = getattr(args, "_pnp_run_config", None)
    if run_config is not None:
        run_config["log_jsonl"] = args.log_jsonl
        run_config["summary_json"] = args.summary_json
    run = _maybe_init_run(args, run, run_config)

    start_t = time.time()
    correct = 0
    total_rounds = 0
    total_effective_rounds = 0
    answered = 0
    total_frames_used_all = 0
    total_frames_used_answered = 0
    processed = 0

    for sample in samples:
        try:
            outcome = pnp_engine.run_sample(
                sample,
                dataset=dataset,
                backend=backend,
                cfg=cfg,
                stats=stats,
                rng=rng,
                base_url=base_url,
                model_id=model_id,
                run=run,
            )
        except Exception as e:
            stats.failed += 1
            maybe_log_jsonl(
                args.log_jsonl,
                {
                    "ts": time.time(),
                    "dataset": sample.dataset,
                    "split": getattr(args, "_pnp_split", getattr(args, "split", "")),
                    "sample_id": sample.sample_id,
                    "uid": sample.uid,
                    "video_key": sample.video_key,
                    "video_url": sample.video_url,
                    "video_path": str(getattr(args, "_pnp_video_path_for_error", lambda s: s.video_key)(sample)),
                    "error": f"{type(e).__name__}: {str(e)[:400]}",
                },
            )
        else:
            if outcome.answer_letter is not None:
                frames_used = len(outcome.seen_frames)
                total_frames_used_all += frames_used
                total_frames_used_answered += frames_used
                answered += 1
                total_rounds += min(outcome.round_idx, args.max_rounds)
                total_effective_rounds += outcome.effective_rounds
                if dataset.is_correct(sample, outcome.answer_letter):
                    correct += 1
            elif args.force_final_answer:
                stats.invalid_action_terminated += 1
                total_frames_used_all += len(outcome.seen_frames)

        processed += 1
        progress_interval = int(getattr(args, "progress_interval", 20) or 20)
        if processed % progress_interval == 0:
            acc = correct / max(1, processed)
            avg_rounds = total_rounds / max(1, processed)
            avg_frames = total_frames_used_answered / max(1, answered)
            print(
                f"[{processed}/{len(samples)}] acc={acc:.4f} avg_rounds={avg_rounds:.3f} avg_frames={avg_frames:.2f} "
                f"failed={stats.failed} invalid_term={stats.invalid_action_terminated} calls={stats.total_model_calls} "
                f"elapsed_s={time.time()-start_t:.1f}",
                flush=True,
            )
            _wandb_log(
                args,
                run,
                {
                    "eval/acc": acc,
                    "eval/avg_rounds": avg_rounds,
                    "eval/avg_frames_used": avg_frames,
                    "eval/failed": stats.failed,
                    "eval/invalid_action_terminated": stats.invalid_action_terminated,
                    "eval/invalid_outputs": stats.invalid_outputs,
                    "eval/total_calls": stats.total_model_calls,
                    "eval/think_present": getattr(dataset, "think_present", 0),
                    "eval/missing_summary": getattr(dataset, "missing_summary", 0),
                },
                step=processed,
            )

    elapsed = time.time() - start_t
    acc = correct / max(1, processed)
    avg_rounds = total_rounds / max(1, processed)
    avg_effective_rounds = total_effective_rounds / max(1, processed)
    avg_frames_used = total_frames_used_answered / max(1, answered)
    avg_frames_used_all = total_frames_used_all / max(1, processed)
    prompt_log_lines = 0
    prompt_log_bytes = 0
    if args.log_jsonl and os.path.exists(args.log_jsonl):
        prompt_log_bytes = os.path.getsize(args.log_jsonl)
        prompt_log_lines = _count_file_lines(args.log_jsonl)

    results = {
        "samples": processed,
        "answered": answered,
        "correct": correct,
        "accuracy": acc,
        "avg_rounds": avg_rounds,
        "avg_effective_rounds": avg_effective_rounds,
        "avg_frames_used": avg_frames_used,
        "avg_frames_used_all": avg_frames_used_all,
        "failed": stats.failed,
        "elapsed_s": elapsed,
        "prompt_log_lines": prompt_log_lines,
        "prompt_log_bytes": prompt_log_bytes,
        "invalid_outputs": stats.invalid_outputs,
        "invalid_action_terminated": stats.invalid_action_terminated,
        "total_retries": stats.total_retries,
        "total_model_calls": stats.total_model_calls,
        "think_present_rounds": getattr(dataset, "think_present", 0),
        "missing_summary_rounds": getattr(dataset, "missing_summary", 0),
    }
    print(json.dumps(results, indent=2), flush=True)

    wandb_info: Optional[dict[str, Any]] = None
    if run is not None:
        run.summary["answered"] = answered
        run.summary["final_acc"] = acc
        run.summary["final_avg_rounds"] = avg_rounds
        run.summary["final_avg_effective_rounds"] = avg_effective_rounds
        run.summary["final_avg_frames_used"] = avg_frames_used
        run.summary["final_avg_frames_used_all"] = avg_frames_used_all
        run.summary["failed"] = stats.failed
        run.summary["invalid_outputs"] = stats.invalid_outputs
        run.summary["invalid_action_terminated"] = stats.invalid_action_terminated
        run.summary["prompt_log_jsonl"] = args.log_jsonl
        run.summary["prompt_log_lines"] = prompt_log_lines
        run.summary["prompt_log_bytes"] = prompt_log_bytes
        run.summary["think_present_rounds"] = getattr(dataset, "think_present", 0)
        run.summary["missing_summary_rounds"] = getattr(dataset, "missing_summary", 0)
        run.finish()
        wandb_info = {
            "enabled": True,
            "mode": getattr(args, "wandb_mode", "") or os.getenv("WANDB_MODE"),
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "name": args.wandb_name,
            "group": args.wandb_group,
            "id": getattr(run, "id", None),
            "url": getattr(run, "url", None),
            "run_dir": getattr(run, "dir", None),
        }

    if args.summary_json:
        out_dir = os.path.dirname(args.summary_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        summary = dict(run_config or {})
        summary.update(
            {
                "results": results,
                "prompt_log_jsonl": args.log_jsonl,
                "wandb": wandb_info,
                "command": " ".join(["python", *sys.argv]),
            }
        )
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    return results


def _run_lvbench_hf(
    samples: list[Any],
    *,
    dataset: Dataset,
    backend: Backend,
    cfg: LoopConfig,
    stats: RunStats,
    rng: Any,
    base_url: str,
    model_id: str,
    run: Any,
    args: Any,
) -> dict[str, Any]:
    samples = _video_shard(samples, args)
    samples.sort(key=lambda s: (s.video_key, s.uid))
    if not samples:
        raise SystemExit("No samples selected (check --split/--start-idx/--max-samples/--sharding).")
    _auto_suffix_outputs(args)

    resume_completed = _long_resume_completed(getattr(args, "log_jsonl", ""), key="answer_letter") if getattr(args, "resume_from_log", False) else 0
    if resume_completed > 0:
        print(f"[resume] detected {resume_completed} completed samples in {args.log_jsonl}")
        samples = samples[resume_completed:]

    run = _maybe_init_run(args, run, getattr(args, "_pnp_run_config", None))

    correct = 0
    invalid_action_terminated = 0
    total_rounds = 0
    total_effective_rounds = 0
    total_frames_used = 0
    processed = 0

    for sample in samples:
        processed += 1
        processed_global = processed + resume_completed
        video_path = dataset.video_path(sample)

        if not os.path.exists(video_path) or os.path.getsize(video_path) <= 0:
            stats.failed += 1
            maybe_log_jsonl(
                args.log_jsonl,
                {
                    "ts": time.time(),
                    "dataset": sample.dataset,
                    "split": getattr(args, "_pnp_split", getattr(args, "split", "")),
                    "sample_id": sample.sample_id,
                    "uid": sample.uid,
                    "video_key": sample.video_key,
                    "video_path": video_path,
                    "error": "missing_video",
                },
            )
            continue

        try:
            outcome = pnp_engine.run_sample(
                sample,
                dataset=dataset,
                backend=backend,
                cfg=cfg,
                stats=stats,
                rng=rng,
                base_url=base_url,
                model_id=model_id,
                run=run,
            )
        except Exception as e:
            stats.failed += 1
            maybe_log_jsonl(
                args.log_jsonl,
                {
                    "ts": time.time(),
                    "dataset": sample.dataset,
                    "split": getattr(args, "_pnp_split", getattr(args, "split", "")),
                    "sample_id": sample.sample_id,
                    "uid": sample.uid,
                    "video_key": sample.video_key,
                    "video_path": video_path,
                    "error": f"{type(e).__name__}: {str(e)[:400]}",
                },
            )
            continue

        if outcome.answer_letter is not None:
            total_rounds += min(outcome.round_idx, int(args.max_rounds))
            total_effective_rounds += outcome.effective_rounds
            total_frames_used += int(outcome.answer_frame_count)
            if dataset.is_correct(sample, outcome.answer_letter):
                correct += 1
        elif args.force_final_answer:
            invalid_action_terminated += 1

        if run is not None:
            _wandb_log(
                args,
                run,
                {
                    "progress/processed": processed_global,
                    "metrics/acc_so_far": correct / max(1, processed_global - stats.failed),
                    "metrics/failed": stats.failed,
                    "metrics/invalid_outputs": stats.invalid_outputs,
                    "metrics/avg_rounds": total_rounds / max(1, processed_global - stats.failed),
                    "metrics/avg_effective_rounds": total_effective_rounds / max(1, processed_global - stats.failed),
                    "metrics/avg_frames_used": total_frames_used / max(1, processed_global - stats.failed),
                },
                step=processed_global,
            )

        if processed_global % 25 == 0 or processed_global == len(samples):
            acc_so_far = correct / max(1, processed_global - stats.failed)
            print(
                f"[{args.dataset}] processed {processed_global} / {len(samples)+resume_completed} | "
                f"acc={acc_so_far:.4f} failed={stats.failed} invalid={stats.invalid_outputs} calls={stats.total_model_calls}"
            )

    answered = max(0, len(samples) + resume_completed - stats.failed)
    summary = {
        "dataset": args.dataset,
        "split": getattr(args, "_pnp_split", getattr(args, "split", "")),
        "model_path": args.model_path,
        "num_samples": len(samples) + resume_completed,
        "answered": answered,
        "correct": correct,
        "accuracy": correct / max(1, answered),
        "failed": stats.failed,
        "invalid_outputs": stats.invalid_outputs,
        "invalid_action_terminated": invalid_action_terminated,
        "avg_rounds": total_rounds / max(1, answered),
        "avg_effective_rounds": total_effective_rounds / max(1, answered),
        "avg_frames_used": total_frames_used / max(1, answered),
        "think_present": getattr(dataset, "think_present", 0),
        "total_model_calls": stats.total_model_calls,
        "total_retries": stats.total_retries,
        "max_len": getattr(args, "_pnp_max_len", None),
        "config": getattr(args, "_pnp_run_config", None),
        "ts": time.time(),
    }

    if args.summary_json:
        os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))
    if run is not None:
        run.summary.update(summary)
        run.finish()
    return summary


def _oneshot_default_video_path(args: Any, sample: Any) -> str:
    """Video-existence path for the videomme/lvbench one-shot launcher."""
    cache_dir = getattr(args, "_pnp_oneshot_cache_dir")
    return str(cache_dir / sample.video_key)


def _oneshot_default_log_record(
    args: Any,
    sample: Any,
    *,
    outcome: Any,
    pred: Optional[str],
    is_correct: bool,
    video_path: str,
    split: str,
) -> dict[str, Any]:
    """Per-sample JSONL record for the videomme/lvbench one-shot launcher."""
    return {
        "ts": time.time(),
        "dataset": sample.dataset,
        "split": split,
        "uid": sample.uid,
        "video_key": sample.video_key,
        "video_path": video_path,
        "time_reference": sample.time_reference,
        "sample_id": sample.sample_id,
        "question": sample.question,
        "options": sample.options,
        "answer_gt": sample.answer_letter,
        "pred_answer": pred,
        "raw_output": outcome.raw_output,
        "correct": bool(is_correct),
    }


def _oneshot_default_build_summary(
    args: Any,
    *,
    samples_total: int,
    answered: int,
    correct: int,
    failed: int,
    frames_used: int,
    elapsed_s: float,
    stats: RunStats,
) -> dict[str, Any]:
    """Top-level summary dict for the videomme/lvbench one-shot launcher.

    Written directly as the summary-json payload (NOT nested under ``results``).
    """
    _ = frames_used
    return {
        "samples": samples_total,
        "answered": answered,
        "correct": correct,
        "accuracy": float(correct / max(1, answered)),
        "failed": failed,
        "elapsed_s": float(elapsed_s),
        "total_model_calls": stats.total_model_calls,
        "prompt_log_jsonl": args.log_jsonl,
        "cached_only": bool(getattr(args, "cached_only", False)),
        "allow_missing_cached_videos": bool(getattr(args, "allow_missing_cached_videos", False)),
        "cache_filter_total_samples": int(getattr(args, "_pnp_oneshot_cache_filter_total", 0)),
        "cache_filter_missing_videos": int(getattr(args, "_pnp_oneshot_cache_filter_missing", 0)),
        "cache_filter_missing_examples": list(getattr(args, "_pnp_oneshot_cache_filter_missing_examples", []) or []),
    }


def _run_oneshot(
    samples: list[Any],
    *,
    dataset: Dataset,
    backend: Backend,
    cfg: LoopConfig,
    stats: RunStats,
    rng: Any,
    base_url: str,
    model_id: str,
    run: Any,
    args: Any,
) -> dict[str, Any]:
    """Single-round baseline outer loop shared by every one-shot launcher.

    The default behavior reproduces the legacy
    ``oneshot_videomme_lvbench_vllm`` launcher byte-for-byte: video-existence
    skip (never downloads), one chat per sample via
    :func:`pnp_engine.run_sample_oneshot`, and the launcher's own top-level
    summary-json schema (NOT nested under ``results``).

    Launchers whose log record / summary schema / video-path resolution differ
    (lvbench_hf, local_mc) override the defaults via ``args`` hooks:

    * ``_pnp_oneshot_video_path(sample) -> str`` — existence path (default:
      ``cache_dir / sample.video_key``).
    * ``_pnp_oneshot_skip_missing_video: bool`` — pre-skip samples whose video is
      absent on disk (default ``True``); local-MC sets ``False`` so frame-probe
      failures inside the engine count the sample as failed instead.
    * ``_pnp_oneshot_log_record(sample, outcome, pred, is_correct, video_path,
      split) -> dict`` — per-sample JSONL record (default: videomme/lvbench).
    * ``_pnp_oneshot_build_summary(...) -> dict`` — final summary payload
      (default: the legacy videomme/lvbench top-level schema).
    """
    _ = run
    split = getattr(args, "_pnp_split", getattr(args, "split", ""))
    resume_completed = int(getattr(args, "_pnp_oneshot_resume_completed", 0) or 0)
    restart_cb = getattr(args, "_pnp_oneshot_restart_server", None)
    video_path_fn = getattr(args, "_pnp_oneshot_video_path", None)
    log_record_fn = getattr(args, "_pnp_oneshot_log_record", None)
    build_summary_fn = getattr(args, "_pnp_oneshot_build_summary", None)
    skip_missing = bool(getattr(args, "_pnp_oneshot_skip_missing_video", True))

    start_t = time.time()
    correct = 0
    failed = 0
    frames_used = 0
    printed_error = False

    for i, sample in enumerate(samples, start=1):
        if video_path_fn is not None:
            video_path = str(video_path_fn(sample))
        else:
            video_path = _oneshot_default_video_path(args, sample)
        if skip_missing and (not os.path.exists(video_path) or os.path.getsize(video_path) <= 0):
            failed += 1
            continue

        try:
            outcome = pnp_engine.run_sample_oneshot(
                sample,
                dataset=dataset,
                backend=backend,
                cfg=cfg,
                stats=stats,
                rng=rng,
                base_url=base_url,
                model_id=model_id,
                run=run,
            )
        except Exception as e:
            if not printed_error:
                printed_error = True
                print(f"[first_error] {type(e).__name__}: {e}", flush=True)
            if restart_cb is not None:
                new_model_id = restart_cb()
                if new_model_id:
                    model_id = new_model_id
                try:
                    outcome = pnp_engine.run_sample_oneshot(
                        sample,
                        dataset=dataset,
                        backend=backend,
                        cfg=cfg,
                        stats=stats,
                        rng=rng,
                        base_url=base_url,
                        model_id=model_id,
                        run=run,
                    )
                except Exception:
                    failed += 1
                    continue
            else:
                failed += 1
                continue

        if outcome.failed_reason is not None:
            failed += 1
            continue

        pred = outcome.answer_letter
        is_correct = pred is not None and dataset.is_correct(sample, pred)
        if is_correct:
            correct += 1
        frames_used += len(outcome.frame_indices)

        if log_record_fn is not None:
            record = log_record_fn(
                sample,
                outcome=outcome,
                pred=pred,
                is_correct=is_correct,
                video_path=video_path,
                split=split,
            )
        else:
            record = _oneshot_default_log_record(
                args,
                sample,
                outcome=outcome,
                pred=pred,
                is_correct=is_correct,
                video_path=video_path,
                split=split,
            )
        if record is not None:
            maybe_log_jsonl(args.log_jsonl, record)

        if i % 50 == 0:
            answered = i - failed
            acc = correct / max(1, answered)
            print(
                f"[{i}/{len(samples)}] acc={acc:.4f} failed={failed} calls={stats.total_model_calls} "
                f"elapsed_s={time.time()-start_t:.1f}",
                flush=True,
            )

    total = len(samples) + resume_completed
    answered = max(0, total - failed)
    elapsed_s = time.time() - start_t
    if build_summary_fn is not None:
        summary = build_summary_fn(
            samples_total=total,
            answered=answered,
            correct=correct,
            failed=failed,
            frames_used=frames_used,
            elapsed_s=elapsed_s,
            stats=stats,
        )
    else:
        summary = _oneshot_default_build_summary(
            args,
            samples_total=total,
            answered=answered,
            correct=correct,
            failed=failed,
            frames_used=frames_used,
            elapsed_s=elapsed_s,
            stats=stats,
        )
    payload = json.dumps(summary, indent=2, ensure_ascii=False)
    print(payload, flush=True)
    if args.summary_json:
        out_dir = os.path.dirname(args.summary_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
    return summary


def run_eval(
    samples: list[Any],
    *,
    dataset: Dataset,
    backend: Backend,
    cfg: LoopConfig,
    stats: RunStats,
    rng: Any,
    base_url: str,
    model_id: str,
    run: Any,
    args: Any,
) -> dict[str, Any]:
    """Run the legacy-compatible outer eval loop for a migrated PnP launcher."""
    setting = getattr(args, "_pnp_setting", "multi_round_pnp")
    if setting == "oneshot_baseline":
        return _run_oneshot(
            samples,
            dataset=dataset,
            backend=backend,
            cfg=cfg,
            stats=stats,
            rng=rng,
            base_url=base_url,
            model_id=model_id,
            run=run,
            args=args,
        )
    mode = getattr(args, "_pnp_harness_mode", getattr(dataset, "harness_mode", "nextqa"))
    if mode == "nextqa":
        return _run_nextqa(
            samples,
            dataset=dataset,
            backend=backend,
            cfg=cfg,
            stats=stats,
            rng=rng,
            base_url=base_url,
            model_id=model_id,
            run=run,
            args=args,
        )
    if mode == "egoschema":
        return _run_egoschema(
            samples,
            dataset=dataset,
            backend=backend,
            cfg=cfg,
            stats=stats,
            rng=rng,
            base_url=base_url,
            model_id=model_id,
            run=run,
            args=args,
        )
    if mode == "long_vllm":
        return _run_long_vllm(
            samples,
            dataset=dataset,
            backend=backend,
            cfg=cfg,
            stats=stats,
            rng=rng,
            base_url=base_url,
            model_id=model_id,
            run=run,
            args=args,
        )
    if mode == "lvbench_hf":
        return _run_lvbench_hf(
            samples,
            dataset=dataset,
            backend=backend,
            cfg=cfg,
            stats=stats,
            rng=rng,
            base_url=base_url,
            model_id=model_id,
            run=run,
            args=args,
        )
    raise ValueError(f"Unknown PnP harness mode: {mode}")
