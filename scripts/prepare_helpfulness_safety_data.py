"""Materialize the frozen Helpfulness-Safety data split from the official repo."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path
from typing import Any

SOURCE_FILES = {
    "alpaca_train": "safe-alignment/dataset/alpaca_prompt_only/train.parquet",
    "alpaca_eval": "safe-alignment/dataset/alpaca_prompt_only/test.parquet",
    "hh_rlhf_eval": "safe-alignment/dataset/anthropic_hh_rlhf/test.parquet",
    "pku_saferlhf_eval": "safe-alignment/dataset/pku-saferlhf/test.parquet",
}

EXPECTED_ROWS = {
    "alpaca_train": 51_490,
    "alpaca_eval": 512,
    "hh_rlhf_eval": 8_520,
    "pku_saferlhf_eval": 8_211,
}

ARTIFACT_PATHS = {
    "alpaca_source_train": "source/alpaca_train.parquet",
    "policy_train": "train/policy_train.parquet",
    "calibration": "calibration/calibration_512.parquet",
    "alpaca_eval": "eval/alpaca.parquet",
    "hh_rlhf_eval": "eval/hh_rlhf.parquet",
    "pku_saferlhf_eval": "eval/pku_saferlhf.parquet",
}


def frozen_split_indices(row_count: int, calibration_size: int, seed: int) -> tuple[list[int], list[int]]:
    """Return stable policy-train and calibration row indices."""
    if calibration_size <= 0 or calibration_size >= row_count:
        raise ValueError("calibration_size must be between zero and row_count")
    calibration = sorted(random.Random(seed).sample(range(row_count), calibration_size))
    calibration_set = set(calibration)
    policy_train = [index for index in range(row_count) if index not in calibration_set]
    return policy_train, calibration


def read_git_parquet(repo: Path, revision: str, relative_path: str) -> Any:
    """Read a Parquet file directly from a Git object without changing sparse checkout."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    completed = subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{relative_path}"],
        check=True,
        stdout=subprocess.PIPE,
    )
    if completed.stdout.startswith(b"version https://git-lfs.github.com/spec"):
        raise RuntimeError(f"Git object is an unresolved LFS pointer: {relative_path}")
    return pq.read_table(pa.BufferReader(completed.stdout))


def write_parquet(table: Any, path: Path) -> None:
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def table_record(path: Path, table: Any) -> dict[str, Any]:
    return {
        "path": str(path),
        "rows": table.num_rows,
        "columns": list(table.column_names),
    }


def validate_source_tables(tables: dict[str, Any]) -> None:
    for name, expected_rows in EXPECTED_ROWS.items():
        table = tables[name]
        if table.num_rows != expected_rows:
            raise ValueError(f"{name} rows={table.num_rows}, expected={expected_rows}")
        required_columns = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
        missing = sorted(required_columns.difference(table.column_names))
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")


def validate_existing(output_root: Path, revision: str, seed: int, calibration_size: int) -> None:
    import pyarrow.parquet as pq

    ready_path = output_root / "READY.json"
    if not ready_path.is_file():
        raise FileExistsError(f"Output exists without READY.json: {output_root}")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    expected_contract = {
        "revision": revision,
        "seed": seed,
        "calibration_size": calibration_size,
    }
    actual_contract = {
        "revision": ready["source"]["revision"],
        "seed": ready["split"]["seed"],
        "calibration_size": ready["split"]["calibration_size"],
    }
    if actual_contract != expected_contract:
        raise ValueError(f"Existing data contract differs: {actual_contract} != {expected_contract}")

    expected_artifact_rows = {
        "alpaca_source_train": 51_490,
        "policy_train": 50_978,
        "calibration": 512,
        "alpaca_eval": 512,
        "hh_rlhf_eval": 8_520,
        "pku_saferlhf_eval": 8_211,
    }
    for name, expected_rows in expected_artifact_rows.items():
        path = output_root / ARTIFACT_PATHS[name]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_rows = pq.ParquetFile(path).metadata.num_rows
        if actual_rows != expected_rows:
            raise ValueError(f"{name} rows={actual_rows}, expected={expected_rows}")
    print(json.dumps({"status": "ready", "output_root": str(output_root)}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    repo = args.official_repo.resolve()
    if output_root.exists():
        validate_existing(output_root, args.revision, args.seed, args.calibration_size)
        return

    tables = {
        name: read_git_parquet(repo, args.revision, source_path)
        for name, source_path in SOURCE_FILES.items()
    }
    validate_source_tables(tables)

    import pyarrow as pa

    policy_indices, calibration_indices = frozen_split_indices(
        tables["alpaca_train"].num_rows,
        args.calibration_size,
        args.seed,
    )
    policy_train = tables["alpaca_train"].take(pa.array(policy_indices))
    calibration = tables["alpaca_train"].take(pa.array(calibration_indices))
    if policy_train.num_rows != 50_978 or calibration.num_rows != 512:
        raise RuntimeError("Frozen split row counts violate the contract")

    output_root.mkdir(parents=True, exist_ok=False)
    artifacts = {
        "alpaca_source_train": tables["alpaca_train"],
        "policy_train": policy_train,
        "calibration": calibration,
        "alpaca_eval": tables["alpaca_eval"],
        "hh_rlhf_eval": tables["hh_rlhf_eval"],
        "pku_saferlhf_eval": tables["pku_saferlhf_eval"],
    }
    records: dict[str, Any] = {}
    for name, table in artifacts.items():
        relative_path = Path(ARTIFACT_PATHS[name])
        path = output_root / relative_path
        write_parquet(table, path)
        records[name] = table_record(relative_path, table)

    ready = {
        "schema_version": 1,
        "status": "ready",
        "source": {
            "repository": str(repo),
            "revision": args.revision,
            "files": SOURCE_FILES,
        },
        "split": {
            "seed": args.seed,
            "calibration_size": args.calibration_size,
            "policy_train_rows": policy_train.num_rows,
            "calibration_row_indices": calibration_indices,
        },
        "artifacts": records,
    }
    (output_root / "READY.json").write_text(
        json.dumps(ready, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    validate_existing(output_root, args.revision, args.seed, args.calibration_size)


if __name__ == "__main__":
    main()
