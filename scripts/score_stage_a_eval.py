#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from cw_grpo.stage_a_eval_scoring import score_generation_rows
from cw_grpo.stage_a_grpo import REMOTE_PROJECT_ROOT

PROJECT_ROOT = Path(str(REMOTE_PROJECT_ROOT)).resolve()


def _personal_path(raw_path: str | Path, *, label: str) -> Path:
    path = Path(raw_path).resolve()
    if path.is_relative_to(PROJECT_ROOT):
        return path
    relocated_output_root = os.environ.get("CW_GRPO_RELOCATED_OUTPUT_ROOT")
    if relocated_output_root:
        allowed_root = Path(relocated_output_root).expanduser().resolve()
        if path.is_relative_to(allowed_root):
            return path
    raise ValueError(f"{label} must stay under the personal project root")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_sha256() -> bool:
    return os.environ.get("CW_GRPO_RECORD_SHA256", "0").lower() not in {
        "0",
        "false",
        "no",
    }


def _artifact(path: Path) -> dict[str, str | int]:
    artifact = {
        "path": str(path),
        "bytes": path.stat().st_size,
    }
    if _record_sha256():
        artifact["sha256"] = _sha256(path)
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score Stage A generation artifacts with the shared rule verifier."
    )
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--output-scored-parquet", type=Path, required=True)
    parser.add_argument("--output-metrics-json", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    return parser


def main() -> int:
    args = _parser().parse_args()
    input_path = _personal_path(args.input_parquet, label="input parquet")
    scored_path = _personal_path(
        args.output_scored_parquet,
        label="scored output parquet",
    )
    metrics_path = _personal_path(
        args.output_metrics_json,
        label="metrics output JSON",
    )
    if not input_path.is_file():
        raise FileNotFoundError(f"generation parquet is missing: {input_path}")
    if len({input_path, scored_path, metrics_path}) != 3:
        raise ValueError("input, scored output, and metrics output paths must differ")

    rows = pq.read_table(input_path).to_pylist()
    scored_rows, metrics = score_generation_rows(
        rows,
        max_new_tokens=args.max_new_tokens,
        max_workers=args.max_workers,
    )

    scored_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_scored_path = scored_path.with_name(
        f".{scored_path.name}.{os.getpid()}.tmp"
    )
    pq.write_table(
        pa.Table.from_pylist(scored_rows),
        temporary_scored_path,
        compression="zstd",
    )
    os.replace(temporary_scored_path, scored_path)

    payload = {
        "schema_version": 1,
        "max_new_tokens": args.max_new_tokens,
        "input_artifact": _artifact(input_path),
        "scored_artifact": _artifact(scored_path),
        "benchmarks": [asdict(metric) for metric in metrics],
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_metrics_path = metrics_path.with_name(
        f".{metrics_path.name}.{os.getpid()}.tmp"
    )
    temporary_metrics_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_metrics_path, metrics_path)
    print(f"saved scored generations: {scored_path}", flush=True)
    print(f"saved evaluation metrics: {metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
