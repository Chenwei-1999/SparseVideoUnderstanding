import json

from scripts.repro.collect_run_summaries import _collect_experiment


def test_collector_rejects_missing_model_call_log_rows(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    summary_path = results_dir / "toy.summary.json"
    log_path = results_dir / "toy.jsonl"

    summary_path.write_text(
        json.dumps(
            {
                "samples": 1,
                "failed": 0,
                "total_model_calls": 2,
                "accuracy": 1.0,
                "log_jsonl": str(log_path),
            }
        ),
        encoding="utf-8",
    )
    log_path.write_text(
        json.dumps({"user_text": "Question?", "raw_output": "A"}) + "\n",
        encoding="utf-8",
    )

    row = _collect_experiment({"id": "toy"}, results_dir)

    assert row["status"] == "invalid_conversation_log:missing_model_call_rows"


def test_collector_accepts_one_log_row_per_model_call(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    summary_path = results_dir / "toy.summary.json"
    log_path = results_dir / "toy.jsonl"

    summary_path.write_text(
        json.dumps(
            {
                "samples": 1,
                "failed": 0,
                "total_model_calls": 2,
                "accuracy": 1.0,
                "log_jsonl": str(log_path),
            }
        ),
        encoding="utf-8",
    )
    log_path.write_text(
        "\n".join(
            [
                json.dumps({"user_text": "Question?", "raw_output": "A"}),
                json.dumps({"user_text": "Final?", "raw_output": "B"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    row = _collect_experiment({"id": "toy"}, results_dir)

    assert row["status"] == "ok"
