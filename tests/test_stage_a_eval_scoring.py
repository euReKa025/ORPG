import os
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from orpg.stage_a_eval_scoring import score_generation_rows


def test_score_generation_rows_uses_exact_token_ids_and_shared_verifier() -> None:
    rows = [
        {
            "reward_model": {"ground_truth": "42", "style": "rule"},
            "extra_info": {
                "benchmark": "AIME-24",
                "prompt_id": "a" * 64,
            },
            "responses": ["correct", "wrong"],
            "response_token_ids": [[1, 2, 3], [4]],
            "completion_tokens_api": [3, 1],
            "finish_reasons": ["stop", "stop"],
            "request_seeds": [11, 12],
        }
    ]

    def fake_scorer(completion, ground_truth, *, response_token_count):
        assert ground_truth == "42"
        return {
            "correctness_reward": float(completion == "correct"),
            "parse_failure": float(completion == "wrong"),
            "response_token_count": response_token_count,
        }

    scored_rows, metrics = score_generation_rows(
        rows,
        max_new_tokens=32768,
        scorer=fake_scorer,
    )

    assert scored_rows[0]["response_token_counts"] == [3, 1]
    assert scored_rows[0]["correctness_rewards"] == [1.0, 0.0]
    assert scored_rows[0]["parse_failures"] == [0.0, 1.0]
    assert metrics[0].benchmark == "AIME-24"
    assert metrics[0].prompt_count == 1
    assert metrics[0].sample_count == 2
    assert metrics[0].pass_at_1 == pytest.approx(0.5)
    assert metrics[0].average_completion_tokens == pytest.approx(2.0)
    assert metrics[0].parse_failure_rate == pytest.approx(0.5)


def test_score_generation_rows_rejects_api_and_exact_token_count_drift() -> None:
    rows = [
        {
            "reward_model": {"ground_truth": "42"},
            "extra_info": {"benchmark": "AIME-24", "prompt_id": "a" * 64},
            "responses": ["answer"],
            "response_token_ids": [[1, 2]],
            "completion_tokens_api": [3],
            "finish_reasons": ["stop"],
            "request_seeds": [11],
        }
    ]

    with pytest.raises(ValueError, match="token count mismatch"):
        score_generation_rows(rows, max_new_tokens=32768)


def test_stage_a_eval_scoring_cli_uses_real_rule_verifier(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    input_path = tmp_path / "generations.parquet"
    scored_path = tmp_path / "scored.parquet"
    metrics_path = tmp_path / "metrics.json"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "reward_model": {"ground_truth": "42", "style": "rule"},
                    "extra_info": {
                        "benchmark": "AIME-24",
                        "prompt_id": "a" * 64,
                    },
                    "responses": [r"The final answer is \boxed{42}."],
                    "response_token_ids": [[1, 2, 3]],
                    "completion_tokens_api": [3],
                    "finish_reasons": ["stop"],
                    "request_seeds": [11],
                }
            ]
        ),
        input_path,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts/score_stage_a_eval.py"),
            "--input-parquet",
            str(input_path),
            "--output-scored-parquet",
            str(scored_path),
            "--output-metrics-json",
            str(metrics_path),
            "--max-new-tokens",
            "32768",
            "--max-workers",
            "2",
        ],
        cwd=project_root,
        env={**os.environ, "ORPG_RELOCATED_OUTPUT_ROOT": str(tmp_path), "ORPG_ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["benchmarks"][0]["pass_at_1"] == pytest.approx(1.0)
    assert metrics["benchmarks"][0]["average_completion_tokens"] == 3
    scored = pq.read_table(scored_path).to_pylist()
    assert scored[0]["correctness_rewards"] == [1.0]
