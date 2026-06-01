"""Shared video download helpers for public video benchmarks."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def ensure_yt_dlp(py_bin: str) -> list[str]:
    """Return command prefix to invoke yt-dlp."""
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    return [py_bin, "-m", "yt_dlp"]


def download_youtube(url: str, out_mp4: str, *, py_bin: str, timeout_s: int) -> None:
    out_mp4_path = Path(out_mp4)
    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(out_mp4_path.with_suffix("")) + ".%(ext)s"

    node_path = shutil.which("node")
    js_runtime_args: list[str] = []
    if node_path:
        js_runtime_args = ["--js-runtimes", f"node:{node_path}"]

    cmd = [
        *ensure_yt_dlp(py_bin),
        *js_runtime_args,
        "--no-playlist",
        "--merge-output-format",
        "mp4",
        "--extractor-args",
        "youtube:player_client=android",
        "-f",
        "best[ext=mp4][height<=480]/best[ext=mp4]/best",
        "-o",
        out_tmpl,
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp failed ({proc.returncode}): {proc.stderr.strip()[:500]}")

    if out_mp4_path.exists() and out_mp4_path.stat().st_size > 0:
        return
    candidates = list(out_mp4_path.parent.glob(out_mp4_path.stem + ".*"))
    for candidate in candidates:
        if candidate.suffix.lower() == ".mp4" and candidate.stat().st_size > 0:
            candidate.rename(out_mp4_path)
            return
    raise FileNotFoundError(f"Downloaded file not found for {url} (expected {out_mp4_path})")
