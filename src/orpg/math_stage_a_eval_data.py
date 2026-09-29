from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePath
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from orpg.math_stage_a_data import MATH_PROMPT_SUFFIX, prompt_id


@dataclass(frozen=True, slots=True)
class BenchmarkSource:
    benchmark: str
    dataset_id: str
    problem_field: str
    answer_field: str
    id_field: str
    fallback_answer_field: str | None = None
    metadata_fields: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PreparedBenchmarkRecords:
    rows: tuple[dict[str, Any], ...]
    manifest: dict[str, int | str]


def _canonical_ground_truth(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != 1:
            raise ValueError("benchmark answer sequences must contain exactly one canonical value")
        return _canonical_ground_truth(value[0])
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("benchmark answer must be finite")
        if value.is_integer():
            return str(int(value))
    return str(value).strip()


def prepare_benchmark_records(
    records: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    *,
    source: BenchmarkSource,
) -> PreparedBenchmarkRecords:
    rows: list[dict[str, Any]] = []
    fallback_ground_truth_rows = 0

    for source_index, record in enumerate(records):
        problem_value = record.get(source.problem_field)
        if not isinstance(problem_value, str) or not problem_value.strip():
            raise ValueError(
                f"{source.benchmark} row {source_index} has no non-empty problem"
            )
        problem = problem_value.strip()

        answer_value = record.get(source.answer_field)
        ground_truth = _canonical_ground_truth(answer_value)
        ground_truth_source = source.answer_field
        if not ground_truth:
            if source.fallback_answer_field is None:
                raise ValueError(
                    f"{source.benchmark} row {source_index} has no ground truth"
                )
            fallback_value = record.get(source.fallback_answer_field)
            ground_truth = _canonical_ground_truth(fallback_value)
            if not ground_truth:
                raise ValueError(
                    f"{source.benchmark} row {source_index} has no fallback ground truth"
                )
            ground_truth_source = source.fallback_answer_field
            fallback_ground_truth_rows += 1

        benchmark_id = record.get(source.id_field)
        if benchmark_id is None or not str(benchmark_id).strip():
            raise ValueError(
                f"{source.benchmark} row {source_index} has no benchmark id"
            )
        identifier = prompt_id(problem)
        extra_info: dict[str, Any] = {
            "benchmark": source.benchmark,
            "benchmark_id": str(benchmark_id),
            "source_index": source_index,
            "prompt_id": identifier,
            "ground_truth_source": ground_truth_source,
        }
        if source.metadata_fields:
            extra_info["benchmark_metadata"] = {
                field: record.get(field) for field in sorted(source.metadata_fields)
            }
        rows.append(
            {
                "data_source": source.dataset_id,
                "prompt": [
                    {
                        "role": "user",
                        "content": f"{problem}\n{MATH_PROMPT_SUFFIX}",
                    }
                ],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth,
                },
                "extra_info": extra_info,
            }
        )

    unique_prompts = len({row["extra_info"]["prompt_id"] for row in rows})
    if unique_prompts != len(rows):
        raise ValueError(f"{source.benchmark} contains duplicate normalized prompts")
    return PreparedBenchmarkRecords(
        rows=tuple(rows),
        manifest={
            "benchmark": source.benchmark,
            "data_source": source.dataset_id,
            "fallback_ground_truth_rows": fallback_ground_truth_rows,
            "rows": len(rows),
            "unique_prompts": unique_prompts,
        },
    )


def _file_metadata(path: Path, *, rows: int) -> dict[str, int | str]:
    return {
        "bytes": path.stat().st_size,
        "rows": rows,
    }


def write_benchmark_artifacts(
    prepared_by_artifact: Mapping[str, PreparedBenchmarkRecords],
    output_dir: str | Path,
    *,
    provenance: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    benchmark_entries: dict[str, Any] = {}
    prompt_ids_by_benchmark: dict[str, set[str]] = {}
    total_rows = 0

    for artifact_name, prepared in prepared_by_artifact.items():
        if PurePath(artifact_name).name != artifact_name or not artifact_name.endswith(
            ".parquet"
        ):
            raise ValueError("benchmark artifact names must be flat parquet filenames")
        benchmark = str(prepared.manifest["benchmark"])
        if benchmark in benchmark_entries:
            raise ValueError(f"duplicate benchmark: {benchmark}")
        if benchmark not in provenance:
            raise ValueError(f"missing provenance for benchmark: {benchmark}")

        artifact_path = destination / artifact_name
        pq.write_table(
            pa.Table.from_pylist(list(prepared.rows)),
            artifact_path,
            compression="zstd",
        )
        rows = len(prepared.rows)
        total_rows += rows
        prompt_ids_by_benchmark[benchmark] = {
            row["extra_info"]["prompt_id"] for row in prepared.rows
        }
        benchmark_entries[benchmark] = {
            **prepared.manifest,
            **dict(provenance[benchmark]),
            "artifact": {
                "path": artifact_name,
                **_file_metadata(artifact_path, rows=rows),
            },
        }

    benchmark_names = sorted(prompt_ids_by_benchmark)
    overlaps: list[dict[str, int | str]] = []
    for index, left in enumerate(benchmark_names):
        for right in benchmark_names[index + 1 :]:
            overlap_count = len(
                prompt_ids_by_benchmark[left] & prompt_ids_by_benchmark[right]
            )
            if overlap_count:
                overlaps.append(
                    {
                        "left": left,
                        "right": right,
                        "normalized_prompt_overlap": overlap_count,
                    }
                )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "prompt_suffix": MATH_PROMPT_SUFFIX,
        "total_rows": total_rows,
        "cross_benchmark_exact_prompt_overlaps": overlaps,
        "benchmarks": benchmark_entries,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest
