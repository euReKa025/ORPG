import json

import pyarrow.parquet as pq

from orpg.math_stage_a_eval_data import (
    BenchmarkSource,
    prepare_benchmark_records,
    write_benchmark_artifacts,
)


def test_prepare_benchmark_records_normalizes_prompt_answer_and_provenance() -> None:
    source = BenchmarkSource(
        benchmark="AMC-22-23",
        dataset_id="AI-MO/aimo-validation-amc",
        problem_field="problem",
        answer_field="answer",
        id_field="id",
    )

    prepared = prepare_benchmark_records(
        [
            {"id": 7, "problem": "What is 6 times 7?", "answer": 42.0},
            {"id": 8, "problem": "What is one half?", "answer": "1/2"},
        ],
        source=source,
    )

    assert prepared.manifest == {
        "benchmark": "AMC-22-23",
        "data_source": "AI-MO/aimo-validation-amc",
        "fallback_ground_truth_rows": 0,
        "rows": 2,
        "unique_prompts": 2,
    }
    first = prepared.rows[0]
    assert first["data_source"] == "AI-MO/aimo-validation-amc"
    assert first["prompt"] == [
        {
            "role": "user",
            "content": (
                "What is 6 times 7?\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            ),
        }
    ]
    assert first["reward_model"] == {"style": "rule", "ground_truth": "42"}
    assert first["extra_info"]["benchmark"] == "AMC-22-23"
    assert first["extra_info"]["benchmark_id"] == "7"
    assert len(first["extra_info"]["prompt_id"]) == 64
    assert first["extra_info"]["ground_truth_source"] == "answer"


def test_math_missing_answer_uses_solution_boxed_fallback() -> None:
    source = BenchmarkSource(
        benchmark="MATH",
        dataset_id="zwhe99/MATH",
        problem_field="problem",
        answer_field="expected_answer",
        id_field="id",
        fallback_answer_field="solution",
    )

    prepared = prepare_benchmark_records(
        [
            {
                "id": "test/number_theory/880.json",
                "problem": "What day was it?",
                "expected_answer": "",
                "solution": "It was \\boxed{\\mbox{Saturday}}.",
            }
        ],
        source=source,
    )

    row = prepared.rows[0]
    assert row["reward_model"]["ground_truth"].endswith(
        "\\boxed{\\mbox{Saturday}}."
    )
    assert row["extra_info"]["ground_truth_source"] == "solution"
    assert prepared.manifest["fallback_ground_truth_rows"] == 1


def test_olympiad_singleton_answer_sequence_is_unwrapped() -> None:
    source = BenchmarkSource(
        benchmark="OlympiadBench",
        dataset_id="zwhe99/OlympiadBench",
        problem_field="question",
        answer_field="final_answer",
        id_field="id",
        metadata_fields=("is_multiple_answer", "unit", "answer_type", "error"),
    )

    prepared = prepare_benchmark_records(
        [
            {
                "id": 1606,
                "question": "How many moves are needed?",
                "final_answer": ["2"],
                "is_multiple_answer": False,
                "unit": None,
                "answer_type": "Numerical",
                "error": None,
            }
        ],
        source=source,
    )

    assert prepared.rows[0]["reward_model"]["ground_truth"] == "2"
    assert prepared.rows[0]["extra_info"]["benchmark_metadata"] == {
        "answer_type": "Numerical",
        "error": None,
        "is_multiple_answer": False,
        "unit": None,
    }


def test_write_benchmark_artifacts_records_size_and_source_revision(
    tmp_path,
) -> None:
    source = BenchmarkSource(
        benchmark="AIME-24",
        dataset_id="HuggingFaceH4/aime_2024",
        problem_field="problem",
        answer_field="answer",
        id_field="id",
    )
    prepared = prepare_benchmark_records(
        [{"id": 1, "problem": "Compute 1+1.", "answer": "2"}],
        source=source,
    )

    manifest = write_benchmark_artifacts(
        {"aime_24.parquet": prepared},
        tmp_path,
        provenance={
            "AIME-24": {
                "source_path": "/personal/raw/aime.parquet",
                "source_revision": "fixture-revision",
            }
        },
    )

    assert pq.read_metadata(tmp_path / "aime_24.parquet").num_rows == 1
    persisted = json.loads((tmp_path / "manifest.json").read_text())
    assert persisted == manifest
    benchmark = persisted["benchmarks"]["AIME-24"]
    assert benchmark["source_revision"] == "fixture-revision"
    assert benchmark["artifact"]["rows"] == 1
    assert benchmark["artifact"]["bytes"] == (tmp_path / "aime_24.parquet").stat().st_size
