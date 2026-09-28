from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

DATA_SOURCE = "agentica-org/DeepScaleR-Preview-Dataset"
MATH_PROMPT_SUFFIX = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


@dataclass(frozen=True, slots=True)
class PreparedStageARecords:
    train_rows: tuple[dict[str, Any], ...]
    tuning_dev_rows: tuple[dict[str, Any], ...]
    manifest: dict[str, int | str]


def normalize_problem(problem: str) -> str:
    normalized = unicodedata.normalize("NFKC", problem)
    return re.sub(r"\s+", " ", normalized).strip()


def prompt_id(problem: str) -> str:
    return sha256(normalize_problem(problem).encode("utf-8")).hexdigest()


def _to_verl_row(record: Mapping[str, Any], source_index: int, split: str) -> dict[str, Any]:
    problem = str(record["problem"]).strip()
    answer = str(record["answer"]).strip()
    identifier = prompt_id(problem)
    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {
                "role": "user",
                "content": f"{problem}\n{MATH_PROMPT_SUFFIX}",
            }
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": split,
            "source_index": source_index,
            "prompt_id": identifier,
        },
    }


def prepare_stage_a_records(
    records: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    *,
    tuning_dev_size: int = 256,
    excluded_prompt_ids: Collection[str] = (),
) -> PreparedStageARecords:
    source_records = list(records)
    valid: list[tuple[int, Mapping[str, Any], str]] = []
    invalid_missing_answer_rows = 0

    for source_index, record in enumerate(source_records):
        problem = record.get("problem")
        answer = record.get("answer")
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"DeepScaleR row {source_index} has no non-empty problem")
        if answer is None or not str(answer).strip():
            invalid_missing_answer_rows += 1
            continue
        valid.append((source_index, record, prompt_id(problem)))

    excluded_prompt_id_set = frozenset(excluded_prompt_ids)
    excluded_rows = [item for item in valid if item[2] in excluded_prompt_id_set]
    eligible = [item for item in valid if item[2] not in excluded_prompt_id_set]
    unique_prompt_ids = sorted({identifier for _, _, identifier in eligible})
    if tuning_dev_size < 0:
        raise ValueError("tuning_dev_size must be non-negative")
    if tuning_dev_size > len(unique_prompt_ids):
        raise ValueError(
            f"tuning_dev_size={tuning_dev_size} exceeds "
            f"unique valid prompts={len(unique_prompt_ids)}"
        )

    tuning_dev_prompt_ids = frozenset(unique_prompt_ids[:tuning_dev_size])
    train_rows: list[dict[str, Any]] = []
    tuning_dev_rows: list[dict[str, Any]] = []
    for source_index, record, identifier in eligible:
        split = "tuning_dev" if identifier in tuning_dev_prompt_ids else "train"
        row = _to_verl_row(record, source_index, split)
        if split == "tuning_dev":
            tuning_dev_rows.append(row)
        else:
            train_rows.append(row)

    manifest: dict[str, int | str] = {
        "data_source": DATA_SOURCE,
        "source_rows": len(source_records),
        "valid_rows": len(valid),
        "eligible_rows": len(eligible),
        "invalid_missing_answer_rows": invalid_missing_answer_rows,
        "excluded_prompt_rows": len(excluded_rows),
        "excluded_unique_prompts": len({identifier for _, _, identifier in excluded_rows}),
        "unique_valid_prompts": len(unique_prompt_ids),
        "tuning_dev_unique_prompts": len(tuning_dev_prompt_ids),
        "tuning_dev_rows": len(tuning_dev_rows),
        "train_rows": len(train_rows),
        "prompt_id_normalization": "NFKC + collapse-whitespace + SHA-256",
    }
    return PreparedStageARecords(
        train_rows=tuple(train_rows),
        tuning_dev_rows=tuple(tuning_dev_rows),
        manifest=manifest,
    )


def _file_metadata(path: Path, *, rows: int | None = None) -> dict[str, int | str]:
    metadata = {"bytes": path.stat().st_size}
    if rows is not None:
        metadata["rows"] = rows
    return metadata


def write_stage_a_artifacts(
    prepared: PreparedStageARecords,
    output_dir: str | Path,
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    train_path = destination / "train.parquet"
    tuning_dev_path = destination / "tuning_dev.parquet"
    prompt_ids_path = destination / "tuning_dev_prompt_ids.json"
    manifest_path = destination / "manifest.json"

    pq.write_table(pa.Table.from_pylist(list(prepared.train_rows)), train_path, compression="zstd")
    pq.write_table(
        pa.Table.from_pylist(list(prepared.tuning_dev_rows)),
        tuning_dev_path,
        compression="zstd",
    )
    tuning_dev_prompt_ids = sorted(
        {row["extra_info"]["prompt_id"] for row in prepared.tuning_dev_rows}
    )
    prompt_ids_path.write_text(
        json.dumps(tuning_dev_prompt_ids, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        **prepared.manifest,
        "prompt_suffix": MATH_PROMPT_SUFFIX,
        "provenance": dict(provenance),
        "artifacts": {
            train_path.name: _file_metadata(train_path, rows=len(prepared.train_rows)),
            tuning_dev_path.name: _file_metadata(
                tuning_dev_path, rows=len(prepared.tuning_dev_rows)
            ),
            prompt_ids_path.name: _file_metadata(prompt_ids_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest
