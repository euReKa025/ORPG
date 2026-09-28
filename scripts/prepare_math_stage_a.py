#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

import pyarrow.parquet as pq

from cw_grpo.math_stage_a_data import (
    DATA_SOURCE,
    prepare_stage_a_records,
    prompt_id,
    write_stage_a_artifacts,
)

DEFAULT_SOURCE_REVISION = "4500022f6cb0a1456ea5eabe2ff366d4610ee339"


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare DeepScaleR for Stage A GRPO baselines.")
    parser.add_argument("--raw-json", type=Path, required=True)
    parser.add_argument("--math-eval-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tuning-dev-size", type=int, default=256)
    parser.add_argument(
        "--benchmark-overlap-policy",
        choices=("keep", "exclude"),
        required=True,
        help="Explicitly keep or exclude exact normalized MATH benchmark overlaps.",
    )
    parser.add_argument("--source-revision", default=DEFAULT_SOURCE_REVISION)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    with args.raw_json.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise TypeError("DeepScaleR JSON root must be a list")

    valid_source_prompt_ids = {
        prompt_id(record["problem"])
        for record in records
        if isinstance(record.get("problem"), str)
        and record["problem"].strip()
        and record.get("answer") is not None
        and str(record["answer"]).strip()
    }
    math_problems = pq.read_table(args.math_eval_parquet, columns=["problem"])["problem"]
    math_prompt_ids = {prompt_id(problem.as_py()) for problem in math_problems if problem.as_py()}
    benchmark_overlap_prompt_ids = sorted(valid_source_prompt_ids & math_prompt_ids)
    excluded_prompt_ids = (
        benchmark_overlap_prompt_ids if args.benchmark_overlap_policy == "exclude" else ()
    )

    prepared = prepare_stage_a_records(
        records,
        tuning_dev_size=args.tuning_dev_size,
        excluded_prompt_ids=excluded_prompt_ids,
    )
    provenance = {
        "source_repo": DATA_SOURCE,
        "source_revision": args.source_revision,
        "source_path": str(args.raw_json.resolve()),
        "math_eval_path": str(args.math_eval_parquet.resolve()),
        "benchmark_overlap_policy": args.benchmark_overlap_policy,
        "benchmark_overlap_unique_prompts": len(benchmark_overlap_prompt_ids),
        "benchmark_overlap_prompt_ids": benchmark_overlap_prompt_ids,
    }
    manifest = write_stage_a_artifacts(
        prepared,
        args.output_dir,
        provenance=provenance,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
