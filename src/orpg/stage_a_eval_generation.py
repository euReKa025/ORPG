from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from orpg.policy_snapshot import inspect_policy_snapshot


@dataclass(frozen=True, slots=True)
class EvalRequest:
    benchmark: str
    benchmark_id: str
    prompt_id: str
    sample_id: int
    seed: int
    messages: tuple[dict[str, str], ...]


@dataclass(frozen=True, slots=True)
class GeneratedCompletion:
    token_ids: tuple[int, ...]
    finish_reason: str

    def __post_init__(self) -> None:
        if any(token_id < 0 for token_id in self.token_ids):
            raise ValueError("token ids must be non-negative")
        if not self.finish_reason:
            raise ValueError("finish_reason must be non-empty")


def _sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_sha256() -> bool:
    return os.environ.get("ORPG_RECORD_SHA256", "0").lower() not in {
        "0",
        "false",
        "no",
    }


def resolve_eval_model_path(
    model_path: str | Path,
    *,
    public_model_path: str | Path,
    personal_checkpoint_root: str | Path,
) -> Path:
    """Resolve a frozen base model or a registered, complete policy snapshot."""

    resolved = Path(model_path).resolve()
    public_model = Path(public_model_path).resolve()
    checkpoint_parent = Path(personal_checkpoint_root).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"model path is missing: {resolved}")
    if resolved == public_model:
        return resolved
    if not resolved.is_relative_to(checkpoint_parent):
        raise ValueError("model path must be the frozen public model or a personal checkpoint")

    relative_parts = resolved.relative_to(checkpoint_parent).parts
    if (
        len(relative_parts) != 4
        or relative_parts[2] != "actor"
        or relative_parts[3] != "huggingface"
        or not relative_parts[1].startswith("global_step_")
    ):
        raise ValueError(
            "checkpoint model path must end in "
            "<experiment>/global_step_<N>/actor/huggingface"
        )
    try:
        step = int(relative_parts[1].removeprefix("global_step_"))
    except ValueError as error:
        raise ValueError("checkpoint model path has an invalid global step") from error

    report = inspect_policy_snapshot(resolved.parents[2], step=step)
    if report.snapshot_path.resolve() != resolved:
        raise ValueError("checkpoint model path does not match its registered snapshot")
    return resolved


def mark_evaluation_manifest(
    manifest_path: str | Path,
    *,
    status: str,
    output_path: str | Path | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    """Atomically record evaluation lifecycle state and the completed artifact."""

    if status not in {"running", "completed", "failed"}:
        raise ValueError("evaluation status must be running, completed, or failed")
    path = Path(manifest_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid evaluation manifest: {path}") from error
    if not isinstance(manifest, dict):
        raise TypeError(f"invalid evaluation manifest object: {path}")

    manifest["status"] = status
    if status == "completed":
        if output_path is None:
            raise ValueError("completed evaluation requires an output artifact")
        artifact = Path(output_path).resolve()
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise FileNotFoundError(f"evaluation output is missing or empty: {artifact}")
        output_artifact: dict[str, str | int] = {
            "path": str(artifact),
            "bytes": artifact.stat().st_size,
        }
        if _record_sha256():
            output_artifact["sha256"] = _sha256_path(artifact)
        manifest["output_artifact"] = output_artifact
        if error_type is not None or error_message is not None:
            raise ValueError("completed evaluation cannot register an error")
    elif status == "failed":
        if not error_type or not error_message:
            raise ValueError("failed evaluation requires error type and message")
        if output_path is not None:
            raise ValueError("failed evaluation cannot register a completed artifact")
        manifest["error"] = {
            "type": error_type,
            "message": error_message[:2000],
        }
    elif output_path is not None:
        raise ValueError("running evaluation cannot register a completed artifact")
    elif error_type is not None or error_message is not None:
        raise ValueError("running evaluation cannot register an error")

    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def stable_eval_seed(base_seed: int, prompt_id: str, sample_id: int) -> int:
    if base_seed < 0 or sample_id < 0:
        raise ValueError("base_seed and sample_id must be non-negative")
    if not prompt_id:
        raise ValueError("prompt_id must be non-empty")
    material = f"{base_seed}\0{prompt_id}\0{sample_id}".encode()
    return int.from_bytes(sha256(material).digest()[:8], "big") % (2**31)


def evaluation_runtime_aliases(*, pid: int, run_hash: str) -> dict[str, str]:
    """Return short shared-root aliases that fit Ray's AF_UNIX path limit."""

    if pid <= 0:
        raise ValueError("pid must be positive")
    if len(run_hash) != 8 or not run_hash.isalnum():
        raise ValueError("run_hash must be eight alphanumeric characters")
    base = f"/proc/{pid}/cwd/.r"
    return {
        "ray": f"{base}/r/{run_hash}",
        "tmp": f"{base}/t/{run_hash}",
    }


def build_chat_completion_payload(
    request: EvalRequest,
    *,
    model_path: str | Path,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Build the deterministic vLLM request used by every evaluation method."""

    if temperature <= 0 or not 0 < top_p <= 1 or max_new_tokens <= 0:
        raise ValueError("invalid evaluation sampling parameters")
    return {
        "model": str(model_path),
        "messages": list(request.messages),
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_new_tokens,
        "seed": request.seed,
        "return_token_ids": True,
    }


def build_eval_request_plan(
    rows: Sequence[Mapping[str, Any]],
    *,
    samples_per_prompt: int | None,
    sample_ids: Sequence[int] | None = None,
    base_seed: int,
) -> tuple[EvalRequest, ...]:
    if sample_ids is None:
        if samples_per_prompt is None or samples_per_prompt <= 0:
            raise ValueError("samples_per_prompt must be positive")
        normalized_sample_ids = tuple(range(samples_per_prompt))
    else:
        if samples_per_prompt is not None:
            raise ValueError("sample_ids and samples_per_prompt are mutually exclusive")
        normalized_sample_ids = tuple(sample_ids)
        if (
            not normalized_sample_ids
            or any(
                not isinstance(sample_id, int)
                or isinstance(sample_id, bool)
                or sample_id < 0
                for sample_id in normalized_sample_ids
            )
            or len(set(normalized_sample_ids)) != len(normalized_sample_ids)
            or normalized_sample_ids != tuple(sorted(normalized_sample_ids))
        ):
            raise ValueError(
                "sample_ids must be a non-empty, unique, increasing sequence "
                "of non-negative integers"
            )

    requests: list[EvalRequest] = []
    seen_prompt_ids: set[str] = set()
    for row_index, row in enumerate(rows):
        extra_info = row.get("extra_info")
        if not isinstance(extra_info, Mapping):
            raise TypeError(f"evaluation row {row_index} has no extra_info mapping")
        benchmark = str(extra_info.get("benchmark", "")).strip()
        benchmark_id = str(extra_info.get("benchmark_id", "")).strip()
        prompt_identifier = str(extra_info.get("prompt_id", "")).strip()
        if not benchmark or not benchmark_id or not prompt_identifier:
            raise ValueError(f"evaluation row {row_index} has incomplete identity")
        if prompt_identifier in seen_prompt_ids:
            raise ValueError(f"duplicate evaluation prompt_id: {prompt_identifier}")
        seen_prompt_ids.add(prompt_identifier)

        raw_messages = row.get("prompt")
        if not isinstance(raw_messages, Sequence) or isinstance(
            raw_messages, (str, bytes, bytearray)
        ):
            raise TypeError(f"evaluation row {row_index} has no message sequence")
        messages: list[dict[str, str]] = []
        for raw_message in raw_messages:
            if not isinstance(raw_message, Mapping):
                raise TypeError(f"evaluation row {row_index} has an invalid message")
            role = str(raw_message.get("role", "")).strip()
            content = str(raw_message.get("content", ""))
            if not role or not content:
                raise ValueError(f"evaluation row {row_index} has an empty message")
            messages.append({"role": role, "content": content})

        for sample_id in normalized_sample_ids:
            requests.append(
                EvalRequest(
                    benchmark=benchmark,
                    benchmark_id=benchmark_id,
                    prompt_id=prompt_identifier,
                    sample_id=sample_id,
                    seed=stable_eval_seed(base_seed, prompt_identifier, sample_id),
                    messages=tuple(messages),
                )
            )
    return tuple(requests)


def is_prefix_equivalent(
    long_completion: GeneratedCompletion,
    short_completion: GeneratedCompletion,
    *,
    budget: int,
) -> bool:
    if budget <= 0 or len(short_completion.token_ids) > budget:
        return False
    if short_completion.finish_reason == "length":
        return (
            len(short_completion.token_ids) == budget
            and short_completion.token_ids
            == long_completion.token_ids[:budget]
        )
    return (
        short_completion.finish_reason == "stop"
        and long_completion.finish_reason == "stop"
        and short_completion.token_ids == long_completion.token_ids
    )
