#!/usr/bin/env python3

from __future__ import annotations

import argparse
import binascii
import json
import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREFIX = "home/work/hsh/MLLM_data/VideoEspresso_train_video/"
EOCD_SIG = b"PK\x05\x06"
CENTRAL_SIG = b"PK\x01\x02"
LOCAL_SIG = b"PK\x03\x04"
ZIP64_EXTRA_ID = 0x0001


@dataclass(frozen=True)
class SplitZipEntry:
    name: str
    disk: int
    local_header_offset: int
    compress_type: int
    compress_size: int
    file_size: int
    crc: int


def _part_path(archive_dir: Path, disk: int, final_disk: int) -> Path:
    if disk == final_disk:
        return archive_dir / "VideoEspresso_train_video.zip"
    return archive_dir / f"VideoEspresso_train_video.z{disk + 1:02d}"


def _part_paths(archive_dir: Path, final_disk: int) -> list[Path]:
    return [_part_path(archive_dir, disk, final_disk) for disk in range(final_disk + 1)]


def _find_eocd(final_zip: Path) -> dict[str, int]:
    max_eocd = 65557
    size = final_zip.stat().st_size
    with final_zip.open("rb") as f:
        f.seek(max(0, size - max_eocd))
        tail = f.read()
    pos = tail.rfind(EOCD_SIG)
    if pos < 0:
        raise RuntimeError(f"Could not find ZIP end-of-central-directory in {final_zip}")
    (
        _sig,
        disk_no,
        central_disk,
        entries_this_disk,
        entries_total,
        central_size,
        central_offset,
        _comment_len,
    ) = struct.unpack_from("<4s4H2LH", tail, pos)
    if entries_this_disk != entries_total:
        raise RuntimeError(
            f"Unsupported split ZIP central directory layout: entries_this_disk={entries_this_disk}, "
            f"entries_total={entries_total}"
        )
    return {
        "final_disk": int(disk_no),
        "central_disk": int(central_disk),
        "entries_total": int(entries_total),
        "central_size": int(central_size),
        "central_offset": int(central_offset),
    }


def _read_zip64_values(extra: bytes, needed: list[bool]) -> list[int | None]:
    out: list[int | None] = [None] * len(needed)
    pos = 0
    while pos + 4 <= len(extra):
        header_id, data_size = struct.unpack_from("<HH", extra, pos)
        pos += 4
        data = extra[pos : pos + data_size]
        pos += data_size
        if header_id != ZIP64_EXTRA_ID:
            continue
        data_pos = 0
        for idx, is_needed in enumerate(needed):
            if not is_needed:
                continue
            if idx < 3:
                if data_pos + 8 > len(data):
                    return out
                out[idx] = struct.unpack_from("<Q", data, data_pos)[0]
                data_pos += 8
            else:
                if data_pos + 4 > len(data):
                    return out
                out[idx] = struct.unpack_from("<L", data, data_pos)[0]
                data_pos += 4
        return out
    return out


def _iter_central_entries(final_zip: Path) -> Iterable[SplitZipEntry]:
    eocd = _find_eocd(final_zip)
    with final_zip.open("rb") as f:
        f.seek(eocd["central_offset"])
        for _ in range(eocd["entries_total"]):
            header = f.read(46)
            if len(header) != 46:
                raise RuntimeError("Unexpected EOF while reading central directory")
            values = struct.unpack("<4s6H3L5H2L", header)
            if values[0] != CENTRAL_SIG:
                raise RuntimeError(f"Bad central directory signature at offset {f.tell() - 46}")
            compress_type = int(values[4])
            crc = int(values[7])
            compress_size = int(values[8])
            file_size = int(values[9])
            name_len = int(values[10])
            extra_len = int(values[11])
            comment_len = int(values[12])
            disk = int(values[13])
            local_header_offset = int(values[16])
            name = f.read(name_len).decode("utf-8")
            extra = f.read(extra_len)
            f.seek(comment_len, os.SEEK_CUR)

            zip64 = _read_zip64_values(
                extra,
                [
                    file_size == 0xFFFFFFFF,
                    compress_size == 0xFFFFFFFF,
                    local_header_offset == 0xFFFFFFFF,
                    disk == 0xFFFF,
                ],
            )
            if zip64[0] is not None:
                file_size = int(zip64[0])
            if zip64[1] is not None:
                compress_size = int(zip64[1])
            if zip64[2] is not None:
                local_header_offset = int(zip64[2])
            if zip64[3] is not None:
                disk = int(zip64[3])

            yield SplitZipEntry(
                name=name,
                disk=disk,
                local_header_offset=local_header_offset,
                compress_type=compress_type,
                compress_size=compress_size,
                file_size=file_size,
                crc=crc,
            )


def _normalize_rel_path(value: Any) -> str:
    rel = str(value or "").strip().replace("\\", "/")
    while rel.startswith("/"):
        rel = rel[1:]
    return rel


def _load_requested_rel_paths(json_path: Path, *, max_rows: int = 0) -> list[str]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Expected list in {json_path}, got {type(data)}")
    rels: list[str] = []
    seen: set[str] = set()
    for row in data:
        if max_rows > 0 and len(rels) >= max_rows:
            break
        if not isinstance(row, dict):
            continue
        rel = _normalize_rel_path(row.get("video_path") or row.get("video"))
        if not rel or rel in seen:
            continue
        seen.add(rel)
        rels.append(rel)
    return rels


def _wanted_entries(
    final_zip: Path,
    requested_rels: list[str],
    *,
    archive_prefix: str,
) -> tuple[dict[str, SplitZipEntry], list[str], int]:
    rel_to_entry: dict[str, SplitZipEntry] = {}
    prefix = archive_prefix.rstrip("/") + "/"
    requested = set(requested_rels)
    for entry in _iter_central_entries(final_zip):
        if entry.name.endswith("/"):
            continue
        rel = entry.name[len(prefix) :] if entry.name.startswith(prefix) else entry.name
        if rel in requested and rel not in rel_to_entry:
            rel_to_entry[rel] = entry
    missing = [rel for rel in requested_rels if rel not in rel_to_entry]
    final_disk = _find_eocd(final_zip)["final_disk"]
    return rel_to_entry, missing, final_disk


def _advance_position(parts: list[Path], disk: int, offset: int, nbytes: int) -> tuple[int, int]:
    remaining = nbytes
    cur_disk = disk
    cur_offset = offset
    while remaining > 0:
        if cur_disk >= len(parts):
            raise RuntimeError("Read position advanced beyond the final split ZIP part")
        part_size = parts[cur_disk].stat().st_size
        available = part_size - cur_offset
        if remaining < available:
            return cur_disk, cur_offset + remaining
        remaining -= max(available, 0)
        cur_disk += 1
        cur_offset = 0
    return cur_disk, cur_offset


def _read_from_parts(parts: list[Path], disk: int, offset: int, nbytes: int) -> bytes:
    chunks: list[bytes] = []
    for chunk in _iter_part_bytes(parts, disk, offset, nbytes, chunk_size=1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _iter_part_bytes(
    parts: list[Path],
    disk: int,
    offset: int,
    nbytes: int,
    *,
    chunk_size: int = 8 * 1024 * 1024,
) -> Iterable[bytes]:
    remaining = nbytes
    cur_disk = disk
    cur_offset = offset
    while remaining > 0:
        if cur_disk >= len(parts):
            raise RuntimeError("Read request extends beyond the final split ZIP part")
        part = parts[cur_disk]
        with part.open("rb") as f:
            f.seek(cur_offset)
            while remaining > 0:
                to_read = min(chunk_size, remaining)
                chunk = f.read(to_read)
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk
        cur_disk += 1
        cur_offset = 0


def _compressed_data_position(parts: list[Path], entry: SplitZipEntry) -> tuple[int, int]:
    header = _read_from_parts(parts, entry.disk, entry.local_header_offset, 30)
    if len(header) != 30:
        raise RuntimeError(f"Could not read local header for {entry.name}")
    values = struct.unpack("<4s5H3L2H", header)
    if values[0] != LOCAL_SIG:
        raise RuntimeError(f"Bad local header signature for {entry.name}")
    method = int(values[3])
    if method != entry.compress_type:
        raise RuntimeError(
            f"Compression method mismatch for {entry.name}: local={method}, central={entry.compress_type}"
        )
    name_len = int(values[9])
    extra_len = int(values[10])
    return _advance_position(parts, entry.disk, entry.local_header_offset, 30 + name_len + extra_len)


def _extract_entry(
    parts: list[Path],
    entry: SplitZipEntry,
    out_path: Path,
    *,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "archive_name": entry.name,
        "output": str(out_path),
        "compressed_size": entry.compress_size,
        "file_size": entry.file_size,
    }
    if out_path.exists() and not overwrite:
        status["status"] = "exists"
        return status
    if dry_run:
        status["status"] = "dry_run"
        return status

    data_disk, data_offset = _compressed_data_position(parts, entry)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    crc = 0
    written = 0
    try:
        with tmp_path.open("wb") as out_f:
            if entry.compress_type == 0:
                for chunk in _iter_part_bytes(parts, data_disk, data_offset, entry.compress_size):
                    out_f.write(chunk)
                    crc = binascii.crc32(chunk, crc)
                    written += len(chunk)
            elif entry.compress_type == 8:
                decomp = zlib.decompressobj(-15)
                for chunk in _iter_part_bytes(parts, data_disk, data_offset, entry.compress_size):
                    data = decomp.decompress(chunk)
                    if data:
                        out_f.write(data)
                        crc = binascii.crc32(data, crc)
                        written += len(data)
                tail = decomp.flush()
                if tail:
                    out_f.write(tail)
                    crc = binascii.crc32(tail, crc)
                    written += len(tail)
            else:
                raise RuntimeError(f"Unsupported compression method {entry.compress_type} for {entry.name}")

        crc &= 0xFFFFFFFF
        if written != entry.file_size:
            raise RuntimeError(f"Size mismatch for {entry.name}: wrote {written}, expected {entry.file_size}")
        if crc != entry.crc:
            raise RuntimeError(f"CRC mismatch for {entry.name}: got {crc:08x}, expected {entry.crc:08x}")
        tmp_path.replace(out_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    status["status"] = "extracted"
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract only the VideoEspresso train videos referenced by a JSON file from the split ZIP archive."
    )
    parser.add_argument(
        "--asset-root",
        default=os.getenv("REVISE_ASSET_ROOT", str(REPO_ROOT / "data" / "revise_assets")),
    )
    parser.add_argument("--json", required=True, help="VideoEspresso MC/open-ended JSON containing video_path fields.")
    parser.add_argument(
        "--archive-dir",
        default=None,
        help="Directory containing VideoEspresso_train_video.z* and .zip",
    )
    parser.add_argument("--out-root", default=None, help="Output root for extracted relative video paths.")
    parser.add_argument("--archive-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    asset_root = Path(args.asset_root).expanduser().resolve()
    archive_dir = (
        Path(args.archive_dir).expanduser().resolve()
        if args.archive_dir
        else asset_root / "VideoEspresso" / "train_video"
    )
    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else archive_dir / "all_video"
    final_zip = archive_dir / "VideoEspresso_train_video.zip"
    json_path = Path(args.json).expanduser().resolve()

    if not final_zip.exists():
        raise FileNotFoundError(final_zip)
    requested = _load_requested_rel_paths(json_path, max_rows=args.max_rows)
    entries, missing, final_disk = _wanted_entries(final_zip, requested, archive_prefix=args.archive_prefix)
    parts = _part_paths(archive_dir, final_disk)
    missing_parts = [str(path) for path in parts if not path.exists()]
    if missing_parts:
        raise FileNotFoundError(f"Missing split ZIP parts: {missing_parts[:5]}")

    extracted: list[dict[str, Any]] = []
    for rel in requested:
        entry = entries.get(rel)
        if not entry:
            continue
        extracted.append(
            {
                "video_path": rel,
                **_extract_entry(
                    parts,
                    entry,
                    out_root / rel,
                    overwrite=bool(args.overwrite),
                    dry_run=bool(args.dry_run),
                ),
            }
        )

    report = {
        "json": str(json_path),
        "archive_dir": str(archive_dir),
        "out_root": str(out_root),
        "requested": len(requested),
        "matched": len(entries),
        "missing": len(missing),
        "missing_examples": missing[:10],
        "extracted": extracted,
        "dry_run": bool(args.dry_run),
    }
    manifest_path = (
        Path(args.manifest).expanduser().resolve() if args.manifest else out_root / "extract_subset_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "requested": len(requested),
                "matched": len(entries),
                "missing": len(missing),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
