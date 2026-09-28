from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePath


class SnapshotNotReady(RuntimeError):
    """Raised when an asynchronous evaluator would observe a partial snapshot."""


@dataclass(frozen=True, slots=True)
class PolicySnapshotReport:
    step: int
    snapshot_path: Path
    weight_files: tuple[str, ...]
    total_weight_bytes: int


def _indexed_weight_files(snapshot_path: Path, index_path: Path) -> tuple[str, ...]:
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise SnapshotNotReady(f"invalid Hugging Face weight index: {index_path}") from error
    if not isinstance(weight_map, dict) or not weight_map:
        raise SnapshotNotReady(f"empty Hugging Face weight index: {index_path}")

    filenames = tuple(sorted(set(weight_map.values())))
    if not all(isinstance(name, str) and PurePath(name).name == name for name in filenames):
        raise SnapshotNotReady(f"unsafe weight filename in index: {index_path}")
    return filenames


def inspect_policy_snapshot(checkpoint_root: str | Path, *, step: int) -> PolicySnapshotReport:
    """Validate that a registered checkpoint exposes a complete HF policy snapshot.

    The checkpoint tracker is written only after the distributed save barriers.
    Requiring it before inspecting the Hugging Face files prevents an asynchronous
    evaluation job from racing a still-active checkpoint write.
    """

    if step <= 0:
        raise ValueError("step must be positive")
    root = Path(checkpoint_root)
    tracker_path = root / "latest_checkpointed_iteration.txt"
    try:
        registered_step = int(tracker_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as error:
        raise SnapshotNotReady(f"checkpoint tracker is unavailable: {tracker_path}") from error
    if registered_step < step:
        raise SnapshotNotReady(
            f"global_step_{step} is not registered; latest checkpoint is {registered_step}"
        )

    snapshot_path = root / f"global_step_{step}" / "actor" / "huggingface"
    for required_name in ("config.json", "tokenizer_config.json"):
        required_path = snapshot_path / required_name
        if not required_path.is_file() or required_path.stat().st_size == 0:
            raise SnapshotNotReady(f"snapshot metadata is missing: {required_path}")

    safetensors_index = snapshot_path / "model.safetensors.index.json"
    pytorch_index = snapshot_path / "pytorch_model.bin.index.json"
    if safetensors_index.is_file():
        weight_files = _indexed_weight_files(snapshot_path, safetensors_index)
    elif pytorch_index.is_file():
        weight_files = _indexed_weight_files(snapshot_path, pytorch_index)
    elif (snapshot_path / "model.safetensors").is_file():
        weight_files = ("model.safetensors",)
    elif (snapshot_path / "pytorch_model.bin").is_file():
        weight_files = ("pytorch_model.bin",)
    else:
        raise SnapshotNotReady(f"snapshot weights are missing: {snapshot_path}")

    total_weight_bytes = 0
    for filename in weight_files:
        weight_path = snapshot_path / filename
        if not weight_path.is_file() or weight_path.stat().st_size == 0:
            raise SnapshotNotReady(f"weight shard is missing or empty: {weight_path}")
        total_weight_bytes += weight_path.stat().st_size

    return PolicySnapshotReport(
        step=step,
        snapshot_path=snapshot_path,
        weight_files=weight_files,
        total_weight_bytes=total_weight_bytes,
    )
