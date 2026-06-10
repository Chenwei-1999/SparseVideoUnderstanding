"""Direct regression tests for scripts/verl_divergence.py.

These tests build a tiny two-tree fixture and verify the classifier
correctly labels each path as 'modified', 'added_by_revise', or
'deleted_from_upstream'.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "verl_divergence.py"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_classifies_modified_added_deleted(tmp_path):
    embedded = tmp_path / "embedded"
    upstream = tmp_path / "upstream"

    # Upstream has a.py and b.py.
    _write(upstream / "a.py", "print('upstream a')\n")
    _write(upstream / "b.py", "print('upstream b')\n")
    _write(upstream / "removed_by_revise.py", "print('still upstream')\n")
    # Embedded modified a.py, kept b.py untouched, added c.py, and removed
    # `removed_by_revise.py`.
    _write(embedded / "a.py", "print('revise a')\n")
    _write(embedded / "b.py", "print('upstream b')\n")
    _write(embedded / "c.py", "print('revise c')\n")

    out = tmp_path / "divergence.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--embedded",
            str(embedded),
            "--upstream",
            str(upstream),
            "--json-out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    payload = json.loads(out.read_text())
    paths = {entry["path"]: entry["kind"] for entry in payload["entries"]}
    assert paths == {
        "a.py": "modified",
        "c.py": "added_by_revise",
        "removed_by_revise.py": "deleted_from_upstream",
    }, paths
    assert payload["counts"] == {
        "modified": 1,
        "added_by_revise": 1,
        "deleted_from_upstream": 1,
    }


def test_ignores_pycache_and_pyc(tmp_path):
    embedded = tmp_path / "embedded"
    upstream = tmp_path / "upstream"

    _write(upstream / "core.py", "x = 1\n")
    _write(embedded / "core.py", "x = 1\n")
    _write(embedded / "__pycache__" / "core.cpython-310.pyc", "binary noise")
    _write(embedded / "subpkg" / "__pycache__" / "m.cpython-310.pyc", "binary noise")
    _write(embedded / "subpkg" / "stale.pyc", "binary noise")

    out = tmp_path / "divergence.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--embedded",
            str(embedded),
            "--upstream",
            str(upstream),
            "--json-out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    payload = json.loads(out.read_text())
    assert payload["counts"] == {
        "modified": 0,
        "added_by_revise": 0,
        "deleted_from_upstream": 0,
    }, payload


if __name__ == "__main__":
    import tempfile

    for fn in [test_classifies_modified_added_deleted, test_ignores_pycache_and_pyc]:
        with tempfile.TemporaryDirectory() as td:
            fn(Path(td))
        print(f"PASS: {fn.__name__}")
