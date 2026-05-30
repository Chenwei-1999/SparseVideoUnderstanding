"""Tests for scripts/common.py asset-discovery helpers.

The VideoEspresso annotation files are hundreds of megabytes, so the cheap
"is this multiple-choice shaped?" probe must inspect only the first handful of
rows without loading the whole file into memory.
"""

from __future__ import annotations

import json

from scripts.common import _probe_videoespresso_mc


def _mc_row(i: int) -> dict:
    return {"options": ["a", "b", "c", "d"], "correct_answer": "a", "q": i}


def test_probe_reads_only_prefix_and_tolerates_corrupt_tail(tmp_path):
    # 40 valid multiple-choice rows, then deliberately corrupt bytes. A probe
    # that loads the whole file (json.loads(read_text())) raises and reports
    # multiple_choice=False; a bounded prefix read stops after the first rows
    # and never sees the corruption, so it must report multiple_choice=True.
    p = tmp_path / "huge.json"
    head = ",".join(json.dumps(_mc_row(i)) for i in range(40))
    p.write_text("[" + head + ", {CORRUPT NOT JSON ", encoding="utf-8")

    out = _probe_videoespresso_mc(str(p))

    assert out["multiple_choice"] is True
    assert out["reason"] == "ok"


def test_probe_detects_non_mc_rows(tmp_path):
    p = tmp_path / "f.json"
    p.write_text(json.dumps([{"foo": 1}, {"bar": 2}]), encoding="utf-8")

    out = _probe_videoespresso_mc(str(p))

    assert out["multiple_choice"] is False
    assert out["reason"] == "missing_options_or_correct_answer"


def test_probe_accepts_choices_alias(tmp_path):
    # The official schema uses "options"; some mirrors use "choices".
    p = tmp_path / "c.json"
    p.write_text(
        json.dumps([{"choices": ["a", "b"], "correct_answer": "b"}] * 3),
        encoding="utf-8",
    )

    out = _probe_videoespresso_mc(str(p))

    assert out["multiple_choice"] is True


def test_probe_empty_list_is_not_mc(tmp_path):
    p = tmp_path / "e.json"
    p.write_text("[]", encoding="utf-8")

    out = _probe_videoespresso_mc(str(p))

    assert out["multiple_choice"] is False
    assert out["reason"] == "empty_or_non_list"


def test_probe_non_list_is_not_mc(tmp_path):
    p = tmp_path / "obj.json"
    p.write_text(json.dumps({"not": "a list"}), encoding="utf-8")

    out = _probe_videoespresso_mc(str(p))

    assert out["multiple_choice"] is False
    assert out["reason"] == "empty_or_non_list"


def test_probe_missing_file_is_not_mc():
    out = _probe_videoespresso_mc("/nonexistent/path/does/not/exist.json")

    assert out["multiple_choice"] is False
    assert out["reason"] == "missing"
