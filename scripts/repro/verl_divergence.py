#!/usr/bin/env python3
"""Classify divergence between an embedded verl tree and an upstream verl tree.

For every file present in either tree, emit one of:
- "modified": file exists in both with different content
- "added_by_revise": file exists only in embedded
- "deleted_from_upstream": file exists only in upstream

Output is a JSON document plus an optional Markdown report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _walk_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if "__pycache__" in p.parts:
            continue
        if p.name.endswith(".pyc"):
            continue
        yield p.relative_to(root)


def classify(embedded_root: Path, upstream_root: Path) -> dict:
    embedded_files = {p: _hash(embedded_root / p) for p in _walk_files(embedded_root)}
    upstream_files = {p: _hash(upstream_root / p) for p in _walk_files(upstream_root)}

    entries = []
    for path in sorted(set(embedded_files) | set(upstream_files), key=str):
        if path in embedded_files and path in upstream_files:
            if embedded_files[path] != upstream_files[path]:
                entries.append({"path": str(path), "kind": "modified"})
        elif path in embedded_files:
            entries.append({"path": str(path), "kind": "added_by_revise"})
        else:
            entries.append({"path": str(path), "kind": "deleted_from_upstream"})

    counts = {"modified": 0, "added_by_revise": 0, "deleted_from_upstream": 0}
    for entry in entries:
        counts[entry["kind"]] += 1
    return {"entries": entries, "counts": counts}


def render_markdown(
    payload: dict,
    embedded_root: Path,
    upstream_root: Path,
    upstream_sha: str,
) -> str:
    lines = [
        "# verl Embedded-vs-Upstream Divergence",
        "",
        f"- Embedded root: `{embedded_root}`",
        f"- Upstream root: `{upstream_root}`",
        f"- Upstream pinned SHA: `{upstream_sha}`",
        f"- Modified: {payload['counts']['modified']}",
        f"- Added by REVISE: {payload['counts']['added_by_revise']}",
        f"- Deleted from upstream: {payload['counts']['deleted_from_upstream']}",
        "",
        "## Modified files",
        "",
    ]
    for entry in payload["entries"]:
        if entry["kind"] == "modified":
            lines.append(f"- `{entry['path']}`")
    lines += ["", "## Added by REVISE (must re-port in Phase 0b)", ""]
    for entry in payload["entries"]:
        if entry["kind"] == "added_by_revise":
            lines.append(f"- `{entry['path']}`")
    lines += ["", "## Deleted from upstream (intentional removals)", ""]
    for entry in payload["entries"]:
        if entry["kind"] == "deleted_from_upstream":
            lines.append(f"- `{entry['path']}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedded", required=True, type=Path)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--upstream-sha", default="")
    args = parser.parse_args()

    payload = classify(args.embedded, args.upstream)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(
            render_markdown(payload, args.embedded, args.upstream, args.upstream_sha)
        )
    print(json.dumps(payload["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
