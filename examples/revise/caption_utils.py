from __future__ import annotations

import json
import os
import re
from typing import Any

import requests
from PIL import Image

from examples.revise.pnp_utils import (
    b64_jpeg,
    extract_frames_1fps,
    extract_video_info,
    get_api_headers,
    sample_uniform_indices_inclusive,
    timeline_len_1fps,
)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_CAPTION_CACHE: dict[tuple[str, str], dict[int, str]] = {}


def load_video_captions(captions_dir: str | None, video_id: str) -> dict[int, str]:
    root = os.path.abspath(captions_dir or "")
    cache_key = (root, str(video_id))
    cached = _CAPTION_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    if not captions_dir:
        _CAPTION_CACHE[cache_key] = {}
        return {}
    path = os.path.join(captions_dir, f"{video_id}_cap.json")
    if not os.path.exists(path):
        _CAPTION_CACHE[cache_key] = {}
        return {}
    try:
        raw = json.loads(open(path, "r", encoding="utf-8").read())
    except Exception:
        _CAPTION_CACHE[cache_key] = {}
        return {}
    captions: dict[int, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                idx = int(k)
            except Exception:
                continue
            if isinstance(v, str) and v.strip():
                captions[idx] = v.strip()
    _CAPTION_CACHE[cache_key] = dict(captions)
    return captions


def save_video_captions(captions_dir: str, video_id: str, captions: dict[int, str]) -> str:
    os.makedirs(captions_dir, exist_ok=True)
    path = os.path.join(captions_dir, f"{video_id}_cap.json")
    payload = {str(int(k)): str(v or "").strip() for k, v in sorted(captions.items()) if str(v or "").strip()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    _CAPTION_CACHE[(os.path.abspath(captions_dir), str(video_id))] = {int(k): str(v) for k, v in payload.items()}
    return path


def _select_caption_indices(video_path: str, *, max_frames: int, stride_s: int) -> list[int]:
    total_frames, fps = extract_video_info(video_path)
    tl_len = timeline_len_1fps(total_frames, fps)
    if tl_len <= 0:
        return []
    if stride_s > 0:
        indices = list(range(0, tl_len, stride_s))
        if (tl_len - 1) not in indices:
            indices.append(tl_len - 1)
    else:
        indices = sample_uniform_indices_inclusive(0, tl_len - 1, max_frames)
    if len(indices) > max_frames > 0:
        pick = sample_uniform_indices_inclusive(0, len(indices) - 1, max_frames)
        indices = [indices[i] for i in pick]
    return sorted({int(i) for i in indices if i >= 0})


def _build_caption_messages(indices: list[int], images: list[Image.Image]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Generate one short factual caption for each frame.\n"
                "Return ONLY a JSON object mapping each provided second index to its caption.\n"
                'Example: {"0":"a person cooking in a kitchen","4":"the person opens a cabinet"}'
            ),
        }
    ]
    for idx, image in zip(indices, images, strict=False):
        content.append({"type": "text", "text": f"\nFrame at {idx} seconds:\n"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_jpeg(image, max_edge=384, quality=85)}"}})
    return content


def _parse_caption_json(raw_text: str, expected_indices: list[int]) -> dict[int, str]:
    raw = (raw_text or "").strip()
    if not raw:
        return {}
    obj: Any = None
    try:
        obj = json.loads(raw)
    except Exception:
        m = _JSON_RE.search(raw)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        return {}
    parsed: dict[int, str] = {}
    for idx in expected_indices:
        val = obj.get(str(idx))
        if isinstance(val, str) and val.strip():
            parsed[int(idx)] = val.strip()
            continue
        val = obj.get(idx)
        if isinstance(val, str) and val.strip():
            parsed[int(idx)] = val.strip()
    return parsed


def generate_video_captions(
    *,
    video_path: str,
    base_url: str,
    model_id: str,
    max_frames: int = 16,
    stride_s: int = 4,
    timeout_s: int = 600,
    max_tokens: int = 512,
) -> dict[int, str]:
    indices = _select_caption_indices(video_path, max_frames=max_frames, stride_s=stride_s)
    if not indices:
        return {}
    images = extract_frames_1fps(video_path, indices)
    if not images:
        return {}
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a careful video captioning assistant."},
            {"role": "user", "content": _build_caption_messages(indices, images)},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": int(max_tokens),
    }
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        headers=get_api_headers(),
        json=payload,
        timeout=timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()
    raw = data["choices"][0]["message"]["content"]
    return _parse_caption_json(raw, indices)


def ensure_video_captions(
    *,
    captions_dir: str | None,
    generated_captions_dir: str | None,
    video_id: str,
    video_path: str,
    base_url: str,
    model_id: str,
    auto_generate: bool,
    max_frames: int,
    stride_s: int,
    timeout_s: int,
) -> dict[int, str]:
    captions = load_video_captions(captions_dir, video_id)
    if captions:
        return captions
    if generated_captions_dir:
        captions = load_video_captions(generated_captions_dir, video_id)
        if captions:
            return captions
    if not auto_generate:
        return {}
    captions = generate_video_captions(
        video_path=video_path,
        base_url=base_url,
        model_id=model_id,
        max_frames=max_frames,
        stride_s=stride_s,
        timeout_s=timeout_s,
    )
    if captions and generated_captions_dir:
        save_video_captions(generated_captions_dir, video_id, captions)
    return captions
