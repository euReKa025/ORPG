#!/usr/bin/env python3
"""Run the frozen full-dataset Helpfulness--Safety mean@1 evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from orpg.helpfulness_safety_dataset import left_truncate_chat_messages
from orpg.helpfulness_safety_evaluation import (
    EVALUATION_DATASETS,
    EvaluationCalibration,
    assign_shard,
    expected_dataset_rows,
    summarize_scored_rows,
    validate_complete_rows,
)

SEED = 42
MAX_PROMPT_LENGTH = 512
MAX_RESPONSE_LENGTH = 1024
TEMPERATURE = 0.7
TOP_P = 1.0
SAMPLES_PER_PROMPT = 1
SHARD_COUNT = 8
RM_MAX_LENGTH = 2048


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "validate",
            "run-pipeline",
            "generate-shard",
            "merge-generation",
            "score-shard",
            "finalize",
        ),
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--policy-model", type=Path, required=True)
    parser.add_argument("--useful-model", type=Path, required=True)
    parser.add_argument("--harmless-model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--compatibility-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-training-experiment", default="base")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=SHARD_COUNT)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--samples-per-prompt", type=int, default=SAMPLES_PER_PROMPT)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--score-batch-size", type=int, default=8)
    return parser


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _check_frozen_protocol(args: argparse.Namespace) -> None:
    if args.seed < 0:
        raise ValueError("generation seed must be non-negative")
    if args.shard_count != SHARD_COUNT:
        raise ValueError(f"H/S evaluation requires exactly {SHARD_COUNT} shards")
    if not math.isclose(args.temperature, TEMPERATURE):
        raise ValueError(f"H/S evaluation requires temperature={TEMPERATURE}")
    if not math.isclose(args.top_p, TOP_P):
        raise ValueError(f"H/S evaluation requires top_p={TOP_P}")
    if args.samples_per_prompt != SAMPLES_PER_PROMPT:
        raise ValueError("H/S main evaluation requires exactly one response per prompt")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("gpu memory utilization must be in (0, 1)")
    if args.score_batch_size <= 0:
        raise ValueError("score batch size must be positive")


def _load_eval_rows(data_root: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    all_rows: list[dict[str, Any]] = []
    global_index = 0
    for dataset in EVALUATION_DATASETS:
        path = data_root / dataset.filename
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"evaluation dataset is missing: {path}")
        rows = pq.read_table(path).to_pylist()
        if len(rows) != dataset.rows:
            raise ValueError(
                f"{dataset.name} row count mismatch: {len(rows)} != {dataset.rows}"
            )
        for source_index, row in enumerate(rows):
            prompt = row.get("prompt")
            if not isinstance(prompt, list) or not prompt:
                raise ValueError(f"{dataset.name} row {source_index} has no chat prompt")
            extra_info = row.get("extra_info") or {}
            identity = extra_info.get("index", source_index)
            all_rows.append(
                {
                    "dataset": dataset.name,
                    "prompt_id": f"{dataset.name}-{identity}",
                    "source_index": source_index,
                    "global_index": global_index,
                    "prompt": prompt,
                }
            )
            global_index += 1
    validate_complete_rows(all_rows)
    return all_rows


def _validate(args: argparse.Namespace) -> None:
    _check_frozen_protocol(args)
    for model_path in (args.policy_model, args.useful_model, args.harmless_model):
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(f"model config is missing: {model_path}")
        if not list(model_path.glob("*.safetensors")) and not (
            model_path / "model.safetensors.index.json"
        ).is_file():
            raise FileNotFoundError(f"model weights are missing: {model_path}")
        if list(model_path.rglob("*.incomplete")):
            raise RuntimeError(f"model contains incomplete files: {model_path}")

    compatibility = json.loads(args.compatibility_summary.read_text(encoding="utf-8"))
    if compatibility.get("status") != "passed":
        raise RuntimeError("H/S compatibility gate has not passed")
    calibration = EvaluationCalibration.from_manifest(args.calibration_manifest)
    if calibration.calibration_id != "hs-qwen3-base-s42-p512-n4-v3":
        raise ValueError("H/S main evaluation requires frozen calibration v3")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.policy_model, trust_remote_code=False)
    rows = _load_eval_rows(args.data_root)
    truncated_count = 0
    maximum_observed = 0
    for row in rows:
        messages, truncated = left_truncate_chat_messages(
            row["prompt"], tokenizer=tokenizer, max_prompt_length=MAX_PROMPT_LENGTH
        )
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        maximum_observed = max(maximum_observed, len(prompt_ids))
        truncated_count += int(truncated)
    if maximum_observed > MAX_PROMPT_LENGTH:
        raise RuntimeError("left-truncated policy prompt still exceeds the frozen limit")

    _atomic_json(
        args.output_dir / "validation.json",
        {
            "schema_version": 1,
            "status": "ready",
            "scenario": "helpfulness_safety",
            "protocol": "full_mean_at_1",
            "method": args.method,
            "source_training_experiment": args.source_training_experiment,
            "policy_model": str(args.policy_model.resolve()),
            "useful_model": str(args.useful_model.resolve()),
            "harmless_model": str(args.harmless_model.resolve()),
            "calibration_id": calibration.calibration_id,
            "dataset_rows": expected_dataset_rows(),
            "prompt_count": len(rows),
            "prompt_left_truncated_count": truncated_count,
            "max_prompt_tokens_after_truncation": maximum_observed,
            "sampling": {
                "seed": args.seed,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "samples_per_prompt": args.samples_per_prompt,
                "max_prompt_length": MAX_PROMPT_LENGTH,
                "max_response_length": MAX_RESPONSE_LENGTH,
            },
            "shard_count": args.shard_count,
            "rm_max_length": RM_MAX_LENGTH,
        },
    )
    print(json.dumps({"status": "ready", "prompt_count": len(rows)}))


def _require_shard(args: argparse.Namespace) -> int:
    if args.shard_index is None or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("a valid shard index is required")
    return args.shard_index


def _generate_shard(args: argparse.Namespace) -> None:
    _check_frozen_protocol(args)
    shard_index = _require_shard(args)
    shard_path = args.output_dir / "generation/shards" / f"shard-{shard_index:02d}.jsonl"
    manifest_path = shard_path.with_suffix(".manifest.json")
    if shard_path.exists() or manifest_path.exists():
        raise RuntimeError(f"generation shard output already exists: {shard_path}")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.policy_model, trust_remote_code=False)
    selected = [
        row
        for row in _load_eval_rows(args.data_root)
        if assign_shard(row["global_index"], shard_count=args.shard_count) == shard_index
    ]
    formatted_prompts: list[str] = []
    prepared: list[dict[str, Any]] = []
    for row in selected:
        messages, truncated = left_truncate_chat_messages(
            row["prompt"], tokenizer=tokenizer, max_prompt_length=MAX_PROMPT_LENGTH
        )
        formatted_prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
        prepared.append({**row, "policy_prompt": messages, "prompt_left_truncated": truncated})

    llm = LLM(
        model=str(args.policy_model),
        tokenizer=str(args.policy_model),
        trust_remote_code=False,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH,
        max_num_batched_tokens=16_384,
        max_num_seqs=256,
        seed=args.seed + shard_index,
    )
    sampling = SamplingParams(
        n=SAMPLES_PER_PROMPT,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_RESPONSE_LENGTH,
    )
    outputs = llm.generate(formatted_prompts, sampling_params=sampling, use_tqdm=True)
    if len(outputs) != len(prepared):
        raise RuntimeError("vLLM returned an incomplete H/S request set")

    records: list[dict[str, Any]] = []
    for row, request in zip(prepared, outputs, strict=True):
        if len(request.outputs) != SAMPLES_PER_PROMPT:
            raise RuntimeError(f"mean@1 produced the wrong sample count for {row['prompt_id']}")
        completion = request.outputs[0]
        if not completion.text.strip():
            raise RuntimeError(f"mean@1 produced an empty response for {row['prompt_id']}")
        token_ids = getattr(completion, "token_ids", None)
        response_tokens = (
            len(token_ids)
            if token_ids is not None
            else len(tokenizer(completion.text, add_special_tokens=False)["input_ids"])
        )
        records.append(
            {
                **row,
                "response": completion.text,
                "response_token_count": int(response_tokens),
                "finish_reason": getattr(completion, "finish_reason", None),
                "generation_shard": shard_index,
                "generation_seed": args.seed + shard_index,
            }
        )
    _atomic_jsonl(shard_path, records)
    _atomic_json(
        manifest_path,
        {
            "status": "generated",
            "shard_index": shard_index,
            "shard_count": args.shard_count,
            "row_count": len(records),
            "prompt_left_truncated_count": sum(
                int(row["prompt_left_truncated"]) for row in records
            ),
        },
    )


def _merge_generation(args: argparse.Namespace) -> None:
    merged_path = args.output_dir / "generation/mean1-rollouts.jsonl"
    manifest_path = args.output_dir / "generation/generation_manifest.json"
    if merged_path.exists() or manifest_path.exists():
        raise RuntimeError("merged generation output already exists")
    rows: list[dict[str, Any]] = []
    for shard_index in range(args.shard_count):
        shard_path = args.output_dir / "generation/shards" / f"shard-{shard_index:02d}.jsonl"
        rows.extend(_load_jsonl(shard_path))
    validate_complete_rows(rows)
    rows.sort(key=lambda row: int(row["global_index"]))
    if [int(row["global_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("merged generation global indices are incomplete")
    _atomic_jsonl(merged_path, rows)
    _atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "generated",
            "protocol": "full_mean_at_1",
            "prompt_count": len(rows),
            "dataset_rows": expected_dataset_rows(),
            "prompt_left_truncated_count": sum(
                int(row["prompt_left_truncated"]) for row in rows
            ),
            "response_token_count": {
                "min": min(int(row["response_token_count"]) for row in rows),
                "max": max(int(row["response_token_count"]) for row in rows),
                "mean": statistics.fmean(
                    int(row["response_token_count"]) for row in rows
                ),
            },
            "sampling": {
                "seed": args.seed,
                "per_shard_seed": "seed+shard_index",
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "samples_per_prompt": SAMPLES_PER_PROMPT,
                "max_prompt_length": MAX_PROMPT_LENGTH,
                "max_response_length": MAX_RESPONSE_LENGTH,
            },
        },
    )


def _score_shard(args: argparse.Namespace) -> None:
    _check_frozen_protocol(args)
    shard_index = _require_shard(args)
    shard_path = args.output_dir / "scoring/shards" / f"shard-{shard_index:02d}.jsonl"
    manifest_path = shard_path.with_suffix(".manifest.json")
    if shard_path.exists() or manifest_path.exists():
        raise RuntimeError(f"scoring shard output already exists: {shard_path}")

    import torch

    from orpg.helpfulness_safety_service import (
        DualRewardScorer,
        append_assistant_response,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("H/S scoring requires CUDA")
    calibration = EvaluationCalibration.from_manifest(args.calibration_manifest)
    generation = _load_jsonl(args.output_dir / "generation/mean1-rollouts.jsonl")
    selected = [
        row
        for row in generation
        if assign_shard(int(row["global_index"]), shard_count=args.shard_count)
        == shard_index
    ]
    scorer = DualRewardScorer.from_pretrained(
        useful_model_path=args.useful_model,
        harmless_model_path=args.harmless_model,
        device=torch.device("cuda:0"),
        max_length=RM_MAX_LENGTH,
        micro_batch_size=args.score_batch_size,
    )
    conversations = [
        append_assistant_response(row["policy_prompt"], row["response"])
        for row in selected
    ]
    scores = scorer(conversations)
    records: list[dict[str, Any]] = []
    for row, (raw_useful, raw_harmless) in zip(selected, scores, strict=True):
        records.append(
            {
                **row,
                "raw_useful": raw_useful,
                "raw_harmless": raw_harmless,
                "calibrated_useful": calibration.calibrate_useful(raw_useful),
                "calibrated_harmless": calibration.calibrate_harmless(raw_harmless),
                "scoring_shard": shard_index,
            }
        )
    if not all(
        math.isfinite(float(row[key]))
        for row in records
        for key in (
            "raw_useful",
            "raw_harmless",
            "calibrated_useful",
            "calibrated_harmless",
        )
    ):
        raise RuntimeError("H/S scorer returned a non-finite value")
    _atomic_jsonl(shard_path, records)
    _atomic_json(
        manifest_path,
        {
            "status": "scored",
            "shard_index": shard_index,
            "shard_count": args.shard_count,
            "row_count": len(records),
            "calibration_id": calibration.calibration_id,
            "rm_max_length": RM_MAX_LENGTH,
        },
    )


def _finalize(args: argparse.Namespace) -> None:
    scored_path = args.output_dir / "scored_mean1.jsonl"
    summary_path = args.output_dir / "summary/mean1-summary.json"
    manifest_path = args.output_dir / "evaluation_manifest.json"
    if scored_path.exists() or summary_path.exists() or manifest_path.exists():
        raise RuntimeError("final H/S evaluation output already exists")
    rows: list[dict[str, Any]] = []
    for shard_index in range(args.shard_count):
        shard_path = args.output_dir / "scoring/shards" / f"shard-{shard_index:02d}.jsonl"
        rows.extend(_load_jsonl(shard_path))
    validate_complete_rows(rows)
    rows.sort(key=lambda row: int(row["global_index"]))
    calibration = EvaluationCalibration.from_manifest(args.calibration_manifest)
    summary = summarize_scored_rows(rows, calibration=calibration)
    summary.update(
        {
            "method": args.method,
            "seed": args.seed,
            "source_training_experiment": args.source_training_experiment,
            "policy_model": str(args.policy_model.resolve()),
        }
    )
    _atomic_jsonl(scored_path, rows)
    _atomic_json(summary_path, summary)
    _atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "succeeded",
            "scenario": "helpfulness_safety",
            "protocol": "full_mean_at_1",
            "method": args.method,
            "seed": args.seed,
            "source_training_experiment": args.source_training_experiment,
            "policy_model": str(args.policy_model.resolve()),
            "useful_model": str(args.useful_model.resolve()),
            "harmless_model": str(args.harmless_model.resolve()),
            "calibration_manifest": str(args.calibration_manifest.resolve()),
            "calibration_id": calibration.calibration_id,
            "compatibility_summary": str(args.compatibility_summary.resolve()),
            "dataset_rows": expected_dataset_rows(),
            "prompt_count": len(rows),
            "shard_count": args.shard_count,
            "sampling": {
                "seed": args.seed,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "samples_per_prompt": SAMPLES_PER_PROMPT,
                "max_prompt_length": MAX_PROMPT_LENGTH,
                "max_response_length": MAX_RESPONSE_LENGTH,
            },
            "scoring": {
                "rm_max_length": RM_MAX_LENGTH,
                "score_batch_size": args.score_batch_size,
                "formatter": "each reward model's own tokenizer chat template",
                "fix_mistral_regex": True,
            },
            "summary": str(summary_path.resolve()),
            "scored_rows": str(scored_path.resolve()),
        },
    )
    print(json.dumps({"status": "succeeded", "macro": summary["macro"]}))


def _worker_command(args: argparse.Namespace, action: str, shard_index: int) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        action,
        "--method",
        args.method,
        "--policy-model",
        str(args.policy_model),
        "--useful-model",
        str(args.useful_model),
        "--harmless-model",
        str(args.harmless_model),
        "--data-root",
        str(args.data_root),
        "--calibration-manifest",
        str(args.calibration_manifest),
        "--compatibility-summary",
        str(args.compatibility_summary),
        "--output-dir",
        str(args.output_dir),
        "--source-training-experiment",
        args.source_training_experiment,
        "--seed",
        str(args.seed),
        "--shard-index",
        str(shard_index),
        "--shard-count",
        str(args.shard_count),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--samples-per-prompt",
        str(args.samples_per_prompt),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--score-batch-size",
        str(args.score_batch_size),
    ]


def _run_worker_phase(args: argparse.Namespace, action: str) -> None:
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[int, subprocess.Popen[bytes], Any]] = []
    for shard_index in range(args.shard_count):
        log_handle = (log_dir / f"{action}-{shard_index:02d}.log").open("wb")
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(shard_index)
        environment["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        process = subprocess.Popen(
            _worker_command(args, action, shard_index),
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((shard_index, process, log_handle))
    failures: list[tuple[int, int]] = []
    for shard_index, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code != 0:
            failures.append((shard_index, return_code))
    if failures:
        raise RuntimeError(f"{action} workers failed: {failures}")


def _run_pipeline(args: argparse.Namespace) -> None:
    _check_frozen_protocol(args)
    if args.output_dir.exists():
        raise RuntimeError(f"evaluation output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    _run_worker_phase(args, "generate-shard")
    _merge_generation(args)
    _run_worker_phase(args, "score-shard")
    _finalize(args)


def main() -> None:
    args = _parser().parse_args()
    actions = {
        "validate": _validate,
        "run-pipeline": _run_pipeline,
        "generate-shard": _generate_shard,
        "merge-generation": _merge_generation,
        "score-shard": _score_shard,
        "finalize": _finalize,
    }
    actions[args.action](args)


if __name__ == "__main__":
    main()
