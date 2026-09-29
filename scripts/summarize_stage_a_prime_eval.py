#!/usr/bin/env python3
"""Summarize the frozen three-budget Stage A-prime evaluation contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from orpg.stage_a_eval_summary import summarize_budget_metric_payloads
from orpg.stage_a_grpo import REMOTE_PROJECT_ROOT

PROJECT_ROOT = Path(str(REMOTE_PROJECT_ROOT)).resolve()
STAGE_A_PRIME_BUDGETS = (2048, 4096, 8192)
STAGE_A_PRIME_LENGTH_NORMALIZER = 8192.0


def _path(raw_path: Path, label: str) -> Path:
    path = raw_path.resolve()
    if path.is_relative_to(PROJECT_ROOT):
        return path
    relocated_output_root = os.environ.get("ORPG_RELOCATED_OUTPUT_ROOT")
    if relocated_output_root:
        allowed_root = Path(relocated_output_root).expanduser().resolve()
        if path.is_relative_to(allowed_root):
            return path
    raise ValueError(f"{label} must stay under the personal project root")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Stage A-prime three-budget metrics"
    )
    parser.add_argument("--metrics-json", action="append", type=Path, required=True)
    parser.add_argument(
        "--reward-scenario",
        default="stage_a_prime_independent_binary_length",
    )
    parser.add_argument("--length-threshold", type=int, default=4000)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    if args.length_threshold <= 0:
        raise ValueError("length threshold must be positive")

    inputs = [_path(path, "metrics JSON") for path in args.metrics_json]
    output = _path(args.output_json, "summary output JSON")
    if len(inputs) != len(set(inputs)) or len(inputs) != len(STAGE_A_PRIME_BUDGETS):
        raise ValueError(
            "Stage A-prime requires one unique metrics JSON for each of "
            "2048/4096/8192"
        )
    if output in inputs:
        raise ValueError("summary output must differ from metric inputs")

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    summary = summarize_budget_metric_payloads(
        payloads,
        budget_grid=STAGE_A_PRIME_BUDGETS,
        length_normalizer=STAGE_A_PRIME_LENGTH_NORMALIZER,
    )
    summary.update(
        {
            "reward_scenario": args.reward_scenario,
            "stage_a_prime_length_threshold": args.length_threshold,
            "standard_budget": 8192,
            "standard_metric_names": [
                "pass_at_1_at_8192",
                "average_completion_tokens_at_8192",
                "pareto_hypervolume",
            ],
            "input_paths": [str(path) for path in inputs],
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved Stage A-prime evaluation summary: {output}", flush=True)


if __name__ == "__main__":
    main()
