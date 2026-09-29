"""Frozen reward coordinates shared by all Helpfulness–Safety methods."""
from __future__ import annotations
import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

@dataclass(frozen=True, slots=True)
class HelpfulnessSafetyCalibration:
    """Immutable base-policy coordinates shared by every H/S method."""

    calibration_id: str
    useful_mean: float
    useful_std: float
    harmless_mean: float
    harmless_std: float
    epsilon: float
    manifest_path: Path

    @classmethod
    def from_manifest(cls, path: str | Path) -> HelpfulnessSafetyCalibration:
        manifest_path = Path(path).resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") != "frozen":
            raise ValueError("H/S calibration manifest must have status=frozen")
        if payload.get("scenario") != "helpfulness_safety":
            raise ValueError("H/S calibration manifest has the wrong scenario")
        coordinates = payload["coordinates"]
        values = {
            "useful_mean": float(coordinates["useful"]["mean"]),
            "useful_std": float(coordinates["useful"]["population_std"]),
            "harmless_mean": float(coordinates["harmless"]["mean"]),
            "harmless_std": float(coordinates["harmless"]["population_std"]),
            "epsilon": float(coordinates["epsilon"]),
        }
        if not all(isfinite(value) for value in values.values()):
            raise ValueError("H/S calibration coordinates must be finite")
        if values["useful_std"] <= 0.0 or values["harmless_std"] <= 0.0:
            raise ValueError("H/S calibration scales must be positive")
        if values["epsilon"] < 0.0:
            raise ValueError("H/S calibration epsilon cannot be negative")
        calibration_id = str(payload.get("calibration_id", "")).strip()
        if not calibration_id:
            raise ValueError("H/S calibration_id cannot be empty")
        return cls(
            calibration_id=calibration_id,
            manifest_path=manifest_path,
            **values,
        )

    def transform(
        self,
        *,
        raw_useful: float,
        raw_harmless: float,
    ) -> tuple[float, float]:
        if not isfinite(raw_useful) or not isfinite(raw_harmless):
            raise ValueError("raw H/S rewards must be finite")
        useful = (raw_useful - self.useful_mean) / (
            self.useful_std + self.epsilon
        )
        harmless = (raw_harmless - self.harmless_mean) / (
            self.harmless_std + self.epsilon
        )
        if not isfinite(useful) or not isfinite(harmless):
            raise RuntimeError("calibrated H/S rewards are non-finite")
        return useful, harmless
