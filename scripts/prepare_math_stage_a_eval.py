#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pyarrow.parquet as pq

from orpg.math_stage_a_eval_data import (
    BenchmarkSource,
    PreparedBenchmarkRecords,
    prepare_benchmark_records,
    write_benchmark_artifacts,
)
from orpg.math_stage_a_reward import parse_stage_a_ground_truth
from orpg.stage_a_grpo import REMOTE_PROJECT_ROOT

PROJECT_ROOT = Path(str(REMOTE_PROJECT_ROOT))
RAW_ROOT = PROJECT_ROOT / "data/math/raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/math/processed/stage_a_eval_v1"


@dataclass(frozen=True, slots=True)
class SnapshotSpec:
    source: BenchmarkSource
    artifact_name: str
    parquet_path: Path
    metadata_path: Path
    expected_rows: int


SPECS = (
    SnapshotSpec(
        source=BenchmarkSource(
            benchmark="AIME-24",
            dataset_id="HuggingFaceH4/aime_2024",
            problem_field="problem",
            answer_field="answer",
            id_field="id",
            metadata_fields=("url", "year"),
        ),
        artifact_name="aime_24.parquet",
        parquet_path=RAW_ROOT / "aime_2024_hf4/data/train-00000-of-00001.parquet",
        metadata_path=RAW_ROOT
        / "aime_2024_hf4/.cache/huggingface/download/data/"
        "train-00000-of-00001.parquet.metadata",
        expected_rows=30,
    ),
    SnapshotSpec(
        source=BenchmarkSource(
            benchmark="AMC-22-23",
            dataset_id="AI-MO/aimo-validation-amc",
            problem_field="problem",
            answer_field="answer",
            id_field="id",
            metadata_fields=("url",),
        ),
        artifact_name="amc_22_23.parquet",
        parquet_path=RAW_ROOT
        / "amc_2022_2023_aimo/data/train-00000-of-00001.parquet",
        metadata_path=RAW_ROOT
        / "amc_2022_2023_aimo/.cache/huggingface/download/data/"
        "train-00000-of-00001.parquet.metadata",
        expected_rows=83,
    ),
    SnapshotSpec(
        source=BenchmarkSource(
            benchmark="MATH",
            dataset_id="zwhe99/MATH",
            problem_field="problem",
            answer_field="expected_answer",
            id_field="id",
            fallback_answer_field="solution",
            metadata_fields=("level", "type"),
        ),
        artifact_name="math.parquet",
        parquet_path=RAW_ROOT / "math_zwhe99/data/test-00000-of-00001.parquet",
        metadata_path=RAW_ROOT
        / "math_zwhe99/.cache/huggingface/download/data/"
        "test-00000-of-00001.parquet.metadata",
        expected_rows=5000,
    ),
    SnapshotSpec(
        source=BenchmarkSource(
            benchmark="Minerva-Math",
            dataset_id="zwhe99/minerva_math",
            problem_field="problem",
            answer_field="answer",
            id_field="idx",
            metadata_fields=("type",),
        ),
        artifact_name="minerva_math.parquet",
        parquet_path=RAW_ROOT
        / "minerva_math_zwhe99/data/test-00000-of-00001.parquet",
        metadata_path=RAW_ROOT
        / "minerva_math_zwhe99/.cache/huggingface/download/data/"
        "test-00000-of-00001.parquet.metadata",
        expected_rows=272,
    ),
    SnapshotSpec(
        source=BenchmarkSource(
            benchmark="OlympiadBench",
            dataset_id="zwhe99/OlympiadBench",
            problem_field="question",
            answer_field="final_answer",
            id_field="id",
            metadata_fields=(
                "answer_type",
                "error",
                "is_multiple_answer",
                "subfield",
                "unit",
            ),
        ),
        artifact_name="olympiadbench.parquet",
        parquet_path=RAW_ROOT
        / "olympiadbench_zwhe99_eval/data/test-00000-of-00001.parquet",
        metadata_path=RAW_ROOT
        / "olympiadbench_zwhe99_eval/.cache/huggingface/download/data/"
        "test-00000-of-00001.parquet.metadata",
        expected_rows=675,
    ),
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_metadata(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2 or not lines[0].strip() or not lines[1].strip():
        raise ValueError(f"invalid Hugging Face download metadata: {path}")
    result = {
        "source_revision": lines[0].strip(),
        "source_download_etag": lines[1].strip(),
    }
    if len(lines) >= 3 and lines[2].strip():
        result["source_downloaded_at_epoch"] = lines[2].strip()
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the five frozen Stage A external math benchmarks."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = _parser().parse_args()
    prepared_by_artifact: dict[str, PreparedBenchmarkRecords] = {}
    provenance: dict[str, dict[str, str | int]] = {}

    for spec in SPECS:
        table = pq.read_table(spec.parquet_path)
        if table.num_rows != spec.expected_rows:
            raise ValueError(
                f"{spec.source.benchmark} expected {spec.expected_rows} rows, "
                f"found {table.num_rows}"
            )
        prepared = prepare_benchmark_records(
            table.to_pylist(),
            source=spec.source,
        )
        unparseable_ids = [
            row["extra_info"]["benchmark_id"]
            for row in prepared.rows
            if not parse_stage_a_ground_truth(row["reward_model"]["ground_truth"])
        ]
        if unparseable_ids:
            raise ValueError(
                f"{spec.source.benchmark} has unparseable ground truths: "
                f"{unparseable_ids[:10]}"
            )
        prepared_by_artifact[spec.artifact_name] = prepared
        provenance[spec.source.benchmark] = {
            "source_path": str(spec.parquet_path),
            "expected_rows": spec.expected_rows,
            **_download_metadata(spec.metadata_path),
        }

    manifest = write_benchmark_artifacts(
        prepared_by_artifact,
        args.output_dir,
        provenance=provenance,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
