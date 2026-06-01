#!/usr/bin/env python3
"""Prepare and convert GPT-5-mini Batch teacher traces for NExT-QA SFT.

This is a text-only bootstrap path for missing SFT teacher logs. It does not
change the REVISE agent loop. The generated batch output is converted into the
same JSONL schema produced by ``revise/run_generate_teacher_data.sh`` so
``revise/generate_sft_data.py`` and the SFT/RL scripts remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revise.pnp.prompts import SYSTEM_PROMPT
from revise.pnp.utils import (
    OPTION_LABELS,
    build_revise_user_text,
    dedupe_preserve_order,
    default_initial_summary,
    extract_video_info,
    format_revise_question_block,
    normalize_answer_letter,
    normalize_video_id,
    parse_int_list,
    parse_strict_revise_action,
    resolve_nextqa_video_path,
    sample_uniform_indices,
    stable_sample_id_nextqa,
    timeline_len_1fps,
)

OPENAI_BATCH_ENDPOINT = "/v1/chat/completions"

DEVELOPER_INSTRUCTIONS = """\
You generate supervised bootstrap traces for NExT-QA SFT.
The target learner is the canonical REVISE agent loop used by both plug-and-play evaluation and RL.
Return only JSON that matches the requested schema. Do not include markdown fences or commentary.
"""


@dataclass(frozen=True)
class BatchConfig:
    model: str = "gpt-5-mini"
    service_tier: Optional[str] = None
    endpoint: str = OPENAI_BATCH_ENDPOINT
    max_rounds: int = 4
    max_frames_per_round: int = 3
    max_completion_tokens: int = 2048
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    reasoning_effort: Optional[str] = "minimal"
    verbosity: Optional[str] = "low"
    response_format: bool = True
    seed: Optional[int] = None


@dataclass(frozen=True)
class NextQARow:
    custom_id: str
    sample_id: str
    qid: str
    video_id: str
    video_path: str
    question: str
    choices: list[str]
    answer_idx: int
    frame_count: int

    @property
    def answer_letter(self) -> str:
        if 0 <= self.answer_idx < len(OPTION_LABELS):
            return OPTION_LABELS[self.answer_idx]
        return "A"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_answer_idx(value: Any, num_choices: int) -> int:
    text = str(value).strip()
    if not text:
        return 0
    try:
        idx = int(text)
        if 0 <= idx < num_choices:
            return idx
    except Exception:
        pass
    letter = normalize_answer_letter(text, num_choices)
    if letter is not None:
        return ord(letter) - ord("A")
    return 0


def _parse_frame_count(row: dict[str, Any], default_frame_count: int) -> int:
    for key in ("frame_count", "num_frames", "total_frames"):
        value = row.get(key)
        if value is None or str(value).strip() == "":
            continue
        try:
            frame_count = int(float(str(value)))
        except Exception:
            continue
        if frame_count > 0:
            return frame_count
    return max(1, int(default_frame_count))


def _video_path_from_map(video_map: dict[str, Any], video_id: str, video_root: str | Path | None = None) -> str:
    rel = video_map.get(video_id)
    if rel is None:
        return ""
    if video_root:
        resolved = resolve_nextqa_video_path(str(video_root), str(rel), video_id)
        if resolved:
            return str(resolved)
    return str(rel)


def _timeline_frame_count(
    *,
    video_path: str,
    raw_frame_count: int,
    fallback_fps: float,
    use_1fps_timeline: bool,
) -> int:
    raw_frame_count = max(1, int(raw_frame_count or 1))
    if not use_1fps_timeline:
        return raw_frame_count
    try:
        total_frames, fps = extract_video_info(video_path) if video_path and Path(video_path).exists() else (0, 0.0)
    except Exception:
        total_frames, fps = 0, 0.0
    if total_frames <= 0:
        total_frames = raw_frame_count
    if fps <= 0:
        fps = float(fallback_fps or 30.0)
    return max(1, int(timeline_len_1fps(total_frames, fps)))


def load_nextqa_rows(
    csv_path: str | Path,
    map_json: str | Path | None,
    *,
    max_samples: int,
    seed: int,
    default_frame_count: int,
    video_root: str | Path | None = None,
    use_1fps_timeline: bool = False,
    fallback_fps: float = 30.0,
) -> list[NextQARow]:
    """Load NExT-QA rows in the same stable sample-id convention as PnP/RL."""
    csv_path = Path(csv_path)
    video_map = _read_json(Path(map_json)) if map_json and Path(map_json).exists() else {}
    video_map = {str(k): v for k, v in video_map.items()} if isinstance(video_map, dict) else {}

    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if max_samples > 0:
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:max_samples]

    out: list[NextQARow] = []
    for i, row in enumerate(rows):
        video_id = normalize_video_id(row.get("video", row.get("video_id", ""))).strip()
        question = str(row.get("question", "")).strip()
        choices = [str(row.get(f"a{idx}", "")).strip() for idx in range(5)]
        choices = [choice for choice in choices if choice]
        if not video_id or not question or not choices:
            continue
        answer_idx = _parse_answer_idx(row.get("answer", 0), len(choices))
        sample_id = stable_sample_id_nextqa(video_id, question, choices, answer_idx)
        custom_id = f"nextqa-{i:06d}-{sample_id[:12]}"
        raw_frame_count = _parse_frame_count(row, default_frame_count)
        video_path = _video_path_from_map(video_map, video_id, video_root=video_root)
        out.append(
            NextQARow(
                custom_id=custom_id,
                sample_id=sample_id,
                qid=str(row.get("qid", "")).strip(),
                video_id=video_id,
                video_path=video_path,
                question=question,
                choices=choices,
                answer_idx=answer_idx,
                frame_count=_timeline_frame_count(
                    video_path=video_path,
                    raw_frame_count=raw_frame_count,
                    fallback_fps=fallback_fps,
                    use_1fps_timeline=use_1fps_timeline,
                ),
            )
        )
    return out


def _target_system_prompt(config: BatchConfig) -> str:
    return SYSTEM_PROMPT.format(max_frames_per_round=config.max_frames_per_round)


def _response_schema(config: BatchConfig) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "nextqa_revise_teacher_trace",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["rounds"],
                "properties": {
                    "rounds": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": config.max_rounds,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["raw_output"],
                            "properties": {"raw_output": {"type": "string"}},
                        },
                    }
                },
            },
        },
    }


def _build_generation_prompt(row: NextQARow, config: BatchConfig) -> str:
    initial_frames = sample_uniform_indices(row.frame_count, config.max_frames_per_round)
    question_block = format_revise_question_block(row.question, row.choices)
    return "\n".join(
        [
            "Generate one supervised REVISE training trace for this NExT-QA training example.",
            "",
            "Target REVISE system prompt:",
            _target_system_prompt(config),
            "",
            "Question block:",
            question_block,
            "",
            f"Gold answer letter for the final round: {row.answer_letter}",
            f"Total frames L = {row.frame_count} (1 fps timeline)",
            f"Initial visible frame indices: {', '.join(str(i) for i in initial_frames)}",
            f"Maximum assistant rounds to return: {config.max_rounds}",
            "",
            "Target assistant outputs must use exactly this protocol:",
            "Select rounds: <think>...</think><summarize>...</summarize><select>...</select>",
            "Final answer round: <think>...</think><answer>LETTER</answer>",
            "",
            "Constraints:",
            f"- Return between 1 and {config.max_rounds} assistant rounds.",
            "- Use the shortest plausible trajectory. Most examples should answer in 1 or 2 rounds.",
            (
                f"- Reserve {config.max_rounds}-round traces only for questions that genuinely need several "
                "temporal checks."
            ),
            "- Answer as soon as the visible evidence is sufficient; do not select only to spend the budget.",
            "- The trace must end with exactly 1 answer round. All earlier rounds, if any, must be select rounds.",
            "- Each <summarize> must contain P:, O:, H:, U:, R: in that order.",
            f"- Each <select> must list 1 to {config.max_frames_per_round} comma-separated NEW frame indices.",
            "- Do not use ranges, repeated frame indices, or indices outside [0, L-1].",
            "- The final <answer> must be the gold answer letter and must not include words outside the tag.",
            "- Do not mention this prompt, the gold label, supervision, or JSON in the generated raw_output text.",
            "- Keep the trace concise and plausible; do not add any tags other than think, summarize, select, answer.",
        ]
    )


def build_batch_request(row: NextQARow, config: BatchConfig) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {"role": "developer", "content": DEVELOPER_INSTRUCTIONS},
            {"role": "user", "content": _build_generation_prompt(row, config)},
        ],
        "max_completion_tokens": config.max_completion_tokens,
    }
    if config.temperature is not None:
        body["temperature"] = config.temperature
    if config.top_p is not None:
        body["top_p"] = config.top_p
    if config.reasoning_effort:
        body["reasoning_effort"] = config.reasoning_effort
    if config.verbosity:
        body["verbosity"] = config.verbosity
    if config.service_tier:
        body["service_tier"] = config.service_tier
    if config.seed is not None:
        body["seed"] = config.seed
    if config.response_format:
        body["response_format"] = _response_schema(config)
    return {
        "custom_id": row.custom_id,
        "method": "POST",
        "url": config.endpoint,
        "body": body,
    }


def _manifest_record(row: NextQARow, config: BatchConfig) -> dict[str, Any]:
    record = asdict(row)
    record.update(
        {
            "model": config.model,
            "service_tier": config.service_tier,
            "endpoint": config.endpoint,
            "max_rounds": config.max_rounds,
            "max_frames_per_round": config.max_frames_per_round,
            "reasoning_effort": config.reasoning_effort,
            "verbosity": config.verbosity,
        }
    )
    return record


def write_batch_files(
    rows: Iterable[NextQARow],
    config: BatchConfig,
    requests_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    requests_path = Path(requests_path)
    manifest_path = Path(manifest_path)
    requests_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with requests_path.open("w", encoding="utf-8") as req_f, manifest_path.open("w", encoding="utf-8") as man_f:
        for row in rows:
            req_f.write(json.dumps(build_batch_request(row, config), ensure_ascii=False) + "\n")
            man_f.write(json.dumps(_manifest_record(row, config), ensure_ascii=False) + "\n")
            count += 1
    return {"requests": count}


def _load_manifest(path: str | Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            records[str(rec["custom_id"])] = rec
    return records


def _strip_json_fence(text: str) -> str:
    text = str(text or "").strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def _message_content_from_batch_line(line_obj: dict[str, Any]) -> str:
    body = ((line_obj.get("response") or {}).get("body") or {})
    choices = body.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _round_outputs_from_content(content: str) -> list[str]:
    payload = json.loads(_strip_json_fence(content))
    rounds = payload.get("rounds") if isinstance(payload, dict) else None
    if not isinstance(rounds, list):
        raise ValueError("Batch response JSON must contain a rounds array.")
    outputs: list[str] = []
    for item in rounds:
        if isinstance(item, str):
            raw = item
        elif isinstance(item, dict):
            raw = str(item.get("raw_output", ""))
        else:
            raw = ""
        if not raw.strip():
            raise ValueError("Round item is missing raw_output.")
        outputs.append(raw.strip())
    return outputs


def _coerce_row(record: dict[str, Any]) -> NextQARow:
    return NextQARow(
        custom_id=str(record["custom_id"]),
        sample_id=str(record["sample_id"]),
        qid=str(record.get("qid", "")),
        video_id=str(record.get("video_id", "")),
        video_path=str(record.get("video_path", "")),
        question=str(record["question"]),
        choices=[str(x) for x in record["choices"]],
        answer_idx=int(record["answer_idx"]),
        frame_count=max(1, int(record.get("frame_count") or 1)),
    )


def _log_entries_from_rounds(row: NextQARow, outputs: list[str], config: BatchConfig) -> list[dict[str, Any]]:
    if not 1 <= len(outputs) <= config.max_rounds:
        raise ValueError(f"Expected 1 to {config.max_rounds} rounds, got {len(outputs)}.")

    system_prompt = _target_system_prompt(config)
    question_block = format_revise_question_block(row.question, row.choices)
    summary_state = default_initial_summary()
    seen_frames: list[int] = []
    next_frames = sample_uniform_indices(row.frame_count, config.max_frames_per_round)
    entries: list[dict[str, Any]] = []

    for round_idx, raw in enumerate(outputs, start=1):
        frames_this_round = [idx for idx in next_frames if idx not in seen_frames]
        if not frames_this_round:
            frames_this_round = sample_uniform_indices(row.frame_count, 1)
        frames_this_round = frames_this_round[: config.max_frames_per_round]
        for idx in frames_this_round:
            if idx not in seen_frames:
                seen_frames.append(idx)

        summary_in = summary_state
        user_text = build_revise_user_text(
            question_block=question_block,
            summary=summary_in,
            frame_count=row.frame_count,
            round_idx=round_idx,
            frame_indices=frames_this_round,
            seen_frames=seen_frames,
            render_images=True,
            use_1fps_timeline=True,
        )
        if round_idx >= config.max_rounds:
            user_text = (
                f"{user_text}\n\n"
                "This is the final round. You MUST answer now using "
                "<think>...</think> then <answer>LETTER</answer>."
            )

        action = parse_strict_revise_action(raw)
        if action is None:
            raise ValueError(f"Invalid REVISE protocol for {row.custom_id} round {round_idx}.")
        is_trace_final_round = round_idx == len(outputs)
        if not is_trace_final_round and action["kind"] != "select":
            raise ValueError(
                f"Answer must terminate the teacher trace for {row.custom_id}; round {round_idx} is not final."
            )
        if is_trace_final_round and action["kind"] != "answer":
            raise ValueError(f"Final round must be an answer for {row.custom_id}.")

        requested_raw_frames: Optional[list[int]] = None
        requested_mapped_frames: Optional[list[int]] = None
        answer_letter = None
        if action["kind"] == "select":
            select_text = str(action["select"] or "")
            if not re.fullmatch(r"\s*\d+(\s*,\s*\d+)*\s*", select_text):
                raise ValueError(
                    f"Round {round_idx} select must be comma-separated integer frame indices for {row.custom_id}."
                )
            parsed_frames = parse_int_list(select_text)
            requested_raw_frames = dedupe_preserve_order(parsed_frames)
            if len(parsed_frames) != len(requested_raw_frames):
                raise ValueError(f"Round {round_idx} repeats frame indices for {row.custom_id}.")
            if len(requested_raw_frames) > config.max_frames_per_round:
                raise ValueError(
                    f"Round {round_idx} requests more than {config.max_frames_per_round} frames for {row.custom_id}."
                )
            out_of_range = [idx for idx in requested_raw_frames if idx < 0 or idx >= row.frame_count]
            if out_of_range:
                raise ValueError(
                    f"Round {round_idx} requested frame indices out of range for {row.custom_id}: {out_of_range}."
                )
            already_seen = [idx for idx in requested_raw_frames if idx in seen_frames]
            if already_seen:
                raise ValueError(
                    f"Round {round_idx} requested already-seen frame indices for {row.custom_id}: {already_seen}."
                )
            requested_mapped_frames = requested_raw_frames
            summary_state = str(action["summary"] or "").strip()
            next_frames = requested_mapped_frames
        else:
            answer_letter = normalize_answer_letter(str(action["answer"] or ""), len(row.choices))
            next_frames = []

        entries.append(
            {
                "ts": time.time(),
                "source": "openai_batch_gpt5mini_bootstrap",
                "sample_id": row.sample_id,
                "qid": row.qid,
                "video_id": row.video_id,
                "video_path": row.video_path,
                "round_idx": round_idx,
                "retry_idx": 0,
                "retry_feedback": None,
                "question": row.question,
                "choices": row.choices,
                "ground_truth_idx": row.answer_idx,
                "observation_mode": "image",
                "use_candidate_frames": False,
                "use_candidate_frame_ids": False,
                "candidate_unseen_frames": None,
                "captions_dir": None,
                "caption_include": "none",
                "caption_max_chars": 0,
                "shown_frame_captions": None,
                "candidate_id_captions": None,
                "seen_frames": list(seen_frames),
                "current_frames": frames_this_round,
                "requested_raw_frames": requested_raw_frames,
                "requested_mapped_frames": requested_mapped_frames,
                "summary_in": summary_in,
                "system_prompt": system_prompt,
                "user_text": user_text,
                "raw_output": raw,
                "answer_letter": answer_letter,
            }
        )

        if action["kind"] == "answer":
            break

    return entries


def convert_batch_output(
    batch_output_path: str | Path,
    manifest_path: str | Path,
    output_log_path: str | Path,
    config: BatchConfig,
    *,
    error_jsonl: str | Path | None = None,
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    output_log_path = Path(output_log_path)
    output_log_path.parent.mkdir(parents=True, exist_ok=True)
    error_path = Path(error_jsonl) if error_jsonl else None
    if error_path:
        error_path.parent.mkdir(parents=True, exist_ok=True)

    converted_samples = 0
    converted_rounds = 0
    failed = 0
    round_count_histogram: Counter[str] = Counter()
    first_action_histogram: Counter[str] = Counter()
    final_action_histogram: Counter[str] = Counter()
    with Path(batch_output_path).open("r", encoding="utf-8") as in_f, output_log_path.open(
        "w", encoding="utf-8"
    ) as out_f:
        err_f = error_path.open("w", encoding="utf-8") if error_path else None
        try:
            for line in in_f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                custom_id = str(obj.get("custom_id", ""))
                record = manifest.get(custom_id)
                try:
                    if record is None:
                        raise ValueError(f"Unknown custom_id: {custom_id}")
                    response = obj.get("response") or {}
                    status_code = int(response.get("status_code") or 0)
                    if status_code != 200:
                        raise ValueError(f"Batch request failed with status_code={status_code}: {obj.get('error')}")
                    outputs = _round_outputs_from_content(_message_content_from_batch_line(obj))
                    entries = _log_entries_from_rounds(_coerce_row(record), outputs, config)
                    for entry in entries:
                        out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    converted_samples += 1
                    converted_rounds += len(entries)
                    round_count_histogram[str(len(entries))] += 1
                    actions = [
                        (parse_strict_revise_action(str(entry.get("raw_output") or "")) or {}).get("kind", "invalid")
                        for entry in entries
                    ]
                    if actions:
                        first_action_histogram[str(actions[0])] += 1
                        final_action_histogram[str(actions[-1])] += 1
                except Exception as exc:
                    failed += 1
                    if err_f is not None:
                        err_f.write(
                            json.dumps(
                                {"custom_id": custom_id, "error": f"{type(exc).__name__}: {exc}", "line": obj},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
        finally:
            if err_f is not None:
                err_f.close()
    return {
        "converted_samples": converted_samples,
        "converted_rounds": converted_rounds,
        "failed": failed,
        "round_count_histogram": dict(sorted(round_count_histogram.items(), key=lambda item: int(item[0]))),
        "first_action_histogram": dict(sorted(first_action_histogram.items())),
        "final_action_histogram": dict(sorted(final_action_histogram.items())),
    }


def submit_batch(
    input_jsonl: str | Path,
    *,
    endpoint: str = OPENAI_BATCH_ENDPOINT,
    completion_window: str = "24h",
    metadata: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Upload a prepared JSONL file and create one OpenAI Batch job."""
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - import depends on user env
        raise RuntimeError(
            "The openai package is required for submit. Install it or upload the JSONL manually."
        ) from exc

    client = OpenAI()
    with Path(input_jsonl).open("rb") as f:
        input_file = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint=endpoint,
        completion_window=completion_window,
        metadata=metadata or {},
    )
    if hasattr(batch, "model_dump"):
        return batch.model_dump()
    if hasattr(batch, "to_dict"):
        return batch.to_dict()
    return json.loads(json.dumps(batch, default=lambda value: getattr(value, "__dict__", str(value))))


def _openai_client() -> Any:
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - import depends on user env
        raise RuntimeError("The openai package is required for OpenAI Batch operations.") from exc
    return OpenAI()


def _object_to_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return json.loads(json.dumps(obj, default=lambda value: getattr(value, "__dict__", str(value))))


def get_batch_status(batch_id: str, *, client: Any | None = None) -> dict[str, Any]:
    """Fetch a Batch object and return a plain JSON-serializable dict."""
    client = client or _openai_client()
    return _object_to_dict(client.batches.retrieve(batch_id))


def _write_openai_file_content(client: Any, file_id: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = client.files.content(file_id)
    if hasattr(content, "write_to_file"):
        content.write_to_file(path)
        return
    if hasattr(content, "read"):
        data = content.read()
    elif hasattr(content, "text"):
        data = content.text
    else:
        data = content
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(bytes(data))


def download_batch_outputs(
    batch: dict[str, Any],
    *,
    output_jsonl: str | Path,
    error_jsonl: str | Path,
    client: Any | None = None,
) -> dict[str, Any]:
    """Download output and error JSONL files referenced by a Batch object."""
    client = client or _openai_client()
    output_jsonl = Path(output_jsonl)
    error_jsonl = Path(error_jsonl)
    downloaded: dict[str, str] = {}
    missing: list[str] = []

    output_file_id = batch.get("output_file_id")
    if output_file_id:
        _write_openai_file_content(client, str(output_file_id), output_jsonl)
        downloaded["output"] = str(output_jsonl)
    else:
        missing.append("output_file_id")

    error_file_id = batch.get("error_file_id")
    if error_file_id:
        _write_openai_file_content(client, str(error_file_id), error_jsonl)
        downloaded["errors"] = str(error_jsonl)
    else:
        missing.append("error_file_id")

    return {
        "batch_id": str(batch.get("id", "")),
        "status": str(batch.get("status", "")),
        "downloaded": downloaded,
        "missing": missing,
    }


def _default_asset_root() -> Path:
    return Path(os.getenv("REVISE_ASSET_ROOT", REPO_ROOT / "data" / "revise_assets"))


def _config_from_args(args: argparse.Namespace) -> BatchConfig:
    return BatchConfig(
        model=args.model,
        service_tier=args.service_tier,
        endpoint=args.endpoint,
        max_rounds=args.max_rounds,
        max_frames_per_round=args.max_frames_per_round,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        reasoning_effort=args.reasoning_effort,
        verbosity=args.verbosity,
        response_format=not args.no_response_format,
        seed=args.seed_for_api,
    )


def _cmd_prepare(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    rows = load_nextqa_rows(
        args.csv,
        args.map_json,
        max_samples=args.max_samples,
        seed=args.seed,
        default_frame_count=args.default_frame_count,
        video_root=args.video_root,
        use_1fps_timeline=bool(args.use_1fps_timeline),
        fallback_fps=float(args.fallback_fps),
    )
    summary = write_batch_files(rows, config, args.output_jsonl, args.manifest)
    print(json.dumps({"requests_jsonl": str(args.output_jsonl), "manifest": str(args.manifest), **summary}, indent=2))
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    summary = convert_batch_output(
        args.batch_output,
        args.manifest,
        args.output_log,
        config,
        error_jsonl=args.error_jsonl,
    )
    print(json.dumps({"output_log": str(args.output_log), **summary}, indent=2))
    return 0 if summary["converted_samples"] > 0 else 1


def _cmd_submit(args: argparse.Namespace) -> int:
    metadata = {"dataset": "nextqa", "purpose": "revise_sft_teacher"}
    for item in args.metadata:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Metadata item must be KEY=VALUE, got: {item}")
        metadata[key] = value
    result = submit_batch(
        args.input_jsonl,
        endpoint=args.endpoint,
        completion_window=args.completion_window,
        metadata=metadata,
    )
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"batch_json": str(args.output_json), "id": result.get("id")}, indent=2))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    result = get_batch_status(args.batch_id)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    if args.batch_id:
        batch = get_batch_status(args.batch_id)
    else:
        batch = json.loads(Path(args.batch_json).read_text(encoding="utf-8"))
        batch_id = batch.get("id")
        if batch_id and not args.no_refresh:
            batch = get_batch_status(str(batch_id))

    if args.batch_json:
        Path(args.batch_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.batch_json).write_text(json.dumps(batch, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = download_batch_outputs(batch, output_jsonl=args.output_jsonl, error_jsonl=args.error_jsonl)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["downloaded"] else 1


def build_arg_parser() -> argparse.ArgumentParser:
    asset_root = _default_asset_root()
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    def add_config_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", default="gpt-5-mini")
        p.add_argument("--service-tier", default=None)
        p.add_argument("--endpoint", default=OPENAI_BATCH_ENDPOINT)
        p.add_argument("--max-rounds", type=int, default=4)
        p.add_argument("--max-frames-per-round", type=int, default=3)
        p.add_argument("--max-completion-tokens", type=int, default=2048)
        p.add_argument("--temperature", type=float, default=None)
        p.add_argument("--top-p", type=float, default=None)
        p.add_argument("--reasoning-effort", default="minimal")
        p.add_argument("--verbosity", default="low")
        p.add_argument("--seed-for-api", type=int, default=None)
        p.add_argument("--no-response-format", action="store_true")

    prepare = sub.add_parser("prepare", help="Write OpenAI Batch request JSONL and a conversion manifest.")
    add_config_flags(prepare)
    prepare.add_argument(
        "--csv",
        type=Path,
        default=Path(os.getenv("REVISE_NEXTQA_TRAIN_CSV", asset_root / "NExT-QA" / "nextqa" / "train.csv")),
    )
    prepare.add_argument(
        "--map-json",
        type=Path,
        default=Path(os.getenv("REVISE_NEXTQA_MAP_JSON", asset_root / "NExT-QA" / "map_vid_vidorID.json")),
    )
    prepare.add_argument(
        "--video-root",
        type=Path,
        default=Path(os.getenv("REVISE_NEXTQA_VIDEO_ROOT", asset_root / "NExT-QA" / "NExTVideo")),
    )
    prepare.add_argument("--max-samples", type=int, default=8000)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--default-frame-count", type=int, default=128)
    prepare.add_argument("--use-1fps-timeline", action=argparse.BooleanOptionalAction, default=True)
    prepare.add_argument("--fallback-fps", type=float, default=30.0)
    prepare.add_argument(
        "--output-jsonl",
        type=Path,
        default=REPO_ROOT / "outputs" / "openai_batch" / "nextqa_teacher_requests.jsonl",
    )
    prepare.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "outputs" / "openai_batch" / "nextqa_teacher_manifest.jsonl",
    )
    prepare.set_defaults(func=_cmd_prepare)

    submit = sub.add_parser("submit", help="Submit a prepared JSONL file to OpenAI Batch.")
    submit.add_argument(
        "--input-jsonl",
        type=Path,
        default=REPO_ROOT / "outputs" / "openai_batch" / "nextqa_teacher_requests.jsonl",
    )
    submit.add_argument("--endpoint", default=OPENAI_BATCH_ENDPOINT)
    submit.add_argument("--completion-window", default="24h")
    submit.add_argument("--metadata", action="append", default=[])
    submit.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "outputs" / "openai_batch" / "nextqa_teacher_batch.json",
    )
    submit.set_defaults(func=_cmd_submit)

    status = sub.add_parser("status", help="Fetch and print an OpenAI Batch status.")
    status.add_argument("--batch-id", required=True)
    status.add_argument("--output-json", type=Path, default=None)
    status.set_defaults(func=_cmd_status)

    download = sub.add_parser("download", help="Download Batch output/error JSONL files when file ids are available.")
    download.add_argument("--batch-id", default=None)
    download.add_argument(
        "--batch-json",
        type=Path,
        default=REPO_ROOT / "outputs" / "openai_batch" / "nextqa_teacher_batch.json",
    )
    download.add_argument("--no-refresh", action="store_true")
    download.add_argument(
        "--output-jsonl",
        type=Path,
        default=REPO_ROOT / "outputs" / "openai_batch" / "nextqa_teacher_batch_output.jsonl",
    )
    download.add_argument(
        "--error-jsonl",
        type=Path,
        default=REPO_ROOT / "outputs" / "openai_batch" / "nextqa_teacher_batch_errors.jsonl",
    )
    download.set_defaults(func=_cmd_download)

    convert = sub.add_parser("convert", help="Convert a completed Batch output JSONL to REVISE teacher log JSONL.")
    add_config_flags(convert)
    convert.add_argument(
        "--batch-output",
        type=Path,
        default=REPO_ROOT / "outputs" / "openai_batch" / "nextqa_teacher_batch_output.jsonl",
    )
    convert.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "outputs" / "openai_batch" / "nextqa_teacher_manifest.jsonl",
    )
    convert.add_argument(
        "--output-log",
        type=Path,
        default=REPO_ROOT / "outputs" / "nextqa_teacher_train_log.jsonl",
    )
    convert.add_argument(
        "--error-jsonl",
        type=Path,
        default=REPO_ROOT / "outputs" / "openai_batch" / "nextqa_teacher_batch_errors.jsonl",
    )
    convert.set_defaults(func=_cmd_convert)
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
