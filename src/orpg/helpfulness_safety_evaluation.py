from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationDatasetSpec:
    name: str
    filename: str
    rows: int


EVALUATION_DATASETS = (
    EvaluationDatasetSpec("alpaca", "alpaca.parquet", 512),
    EvaluationDatasetSpec("hh_rlhf", "hh_rlhf.parquet", 8_520),
    EvaluationDatasetSpec("pku_saferlhf", "pku_saferlhf.parquet", 8_211),
)


@dataclass(frozen=True, slots=True)
class EvaluationCalibration:
    calibration_id: str
    epsilon: float
    useful_mean: float
    useful_std: float
    harmless_mean: float
    harmless_std: float

    @classmethod
    def from_manifest(cls, path: Path) -> EvaluationCalibration:
        payload = json.loads(path.read_text(encoding="utf-8"))
        coordinates = payload.get("coordinates") or {}
        useful = coordinates.get("useful") or {}
        harmless = coordinates.get("harmless") or {}
        if coordinates.get("std_definition") != "population_std_ddof_0":
            raise ValueError("evaluation requires population calibration coordinates")
        calibration = cls(
            calibration_id=str(payload.get("calibration_id", "")),
            epsilon=float(coordinates.get("epsilon", 0.0)),
            useful_mean=float(useful.get("mean", float("nan"))),
            useful_std=float(useful.get("population_std", float("nan"))),
            harmless_mean=float(harmless.get("mean", float("nan"))),
            harmless_std=float(harmless.get("population_std", float("nan"))),
        )
        values = (
            calibration.epsilon,
            calibration.useful_mean,
            calibration.useful_std,
            calibration.harmless_mean,
            calibration.harmless_std,
        )
        if not calibration.calibration_id or not all(math.isfinite(value) for value in values):
            raise ValueError("calibration coordinates are incomplete or non-finite")
        if calibration.epsilon <= 0 or calibration.useful_std <= 0 or calibration.harmless_std <= 0:
            raise ValueError("calibration epsilon and scales must be positive")
        return calibration

    def calibrate_useful(self, value: float) -> float:
        return (value - self.useful_mean) / (self.useful_std + self.epsilon)

    def calibrate_harmless(self, value: float) -> float:
        return (value - self.harmless_mean) / (self.harmless_std + self.epsilon)


def assign_shard(index: int, *, shard_count: int) -> int:
    if index < 0:
        raise ValueError("row index cannot be negative")
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    return index % shard_count


def expected_dataset_rows() -> dict[str, int]:
    return {item.name: item.rows for item in EVALUATION_DATASETS}


def validate_complete_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_by_dataset: Mapping[str, int] | None = None,
) -> None:
    expected = dict(expected_by_dataset or expected_dataset_rows())
    observed: Counter[str] = Counter()
    identities: set[tuple[str, str]] = set()
    for row in rows:
        dataset = str(row.get("dataset", ""))
        prompt_id = str(row.get("prompt_id", ""))
        if dataset not in expected or not prompt_id:
            raise ValueError("evaluation row has an invalid dataset or prompt_id")
        identity = (dataset, prompt_id)
        if identity in identities:
            raise ValueError(f"duplicate evaluation row: {dataset}/{prompt_id}")
        identities.add(identity)
        observed[dataset] += 1
    if observed != Counter(expected):
        raise ValueError(
            f"evaluation row count mismatch: observed={dict(observed)}, expected={expected}"
        )


def _axis_summary(values: list[float]) -> dict[str, float]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("evaluation scores must be nonempty and finite")
    return {
        "mean": statistics.fmean(values),
        "population_std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def summarize_scored_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    calibration: EvaluationCalibration,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        dataset = str(row.get("dataset", ""))
        if not dataset:
            raise ValueError("scored evaluation row is missing dataset")
        grouped.setdefault(dataset, []).append(row)
    if not grouped:
        raise ValueError("no scored evaluation rows were provided")

    dataset_summaries: dict[str, dict[str, Any]] = {}
    for dataset, dataset_rows in sorted(grouped.items()):
        raw_useful = [float(row["raw_useful"]) for row in dataset_rows]
        raw_harmless = [float(row["raw_harmless"]) for row in dataset_rows]
        calibrated_useful = [calibration.calibrate_useful(value) for value in raw_useful]
        calibrated_harmless = [
            calibration.calibrate_harmless(value) for value in raw_harmless
        ]
        raw_useful_summary = _axis_summary(raw_useful)
        raw_harmless_summary = _axis_summary(raw_harmless)
        calibrated_useful_summary = _axis_summary(calibrated_useful)
        calibrated_harmless_summary = _axis_summary(calibrated_harmless)
        dataset_summaries[dataset] = {
            "prompt_count": len(dataset_rows),
            "raw_useful_mean_at_1": raw_useful_summary["mean"],
            "raw_harmless_mean_at_1": raw_harmless_summary["mean"],
            "calibrated_useful_mean_at_1": calibrated_useful_summary["mean"],
            "calibrated_harmless_mean_at_1": calibrated_harmless_summary["mean"],
            "raw_useful": raw_useful_summary,
            "raw_harmless": raw_harmless_summary,
            "calibrated_useful": calibrated_useful_summary,
            "calibrated_harmless": calibrated_harmless_summary,
        }

    macro_keys = (
        "raw_useful_mean_at_1",
        "raw_harmless_mean_at_1",
        "calibrated_useful_mean_at_1",
        "calibrated_harmless_mean_at_1",
    )
    macro = {
        key: statistics.fmean(float(summary[key]) for summary in dataset_summaries.values())
        for key in macro_keys
    }
    return {
        "schema_version": 1,
        "scenario": "helpfulness_safety",
        "protocol": "full_mean_at_1",
        "calibration_id": calibration.calibration_id,
        "datasets": dataset_summaries,
        "macro": macro,
    }
