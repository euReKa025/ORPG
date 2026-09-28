"""Generate and score the frozen Helpfulness--Safety calibration rollout."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
from pathlib import Path
from typing import Any

PROMPT_COUNT = 512
ROLLOUTS_PER_PROMPT = 4
SEED = 42
MAX_PROMPT_LENGTH = 512
MAX_RESPONSE_LENGTH = 1024
TEMPERATURE = 0.7
TOP_P = 1.0
CALIBRATION_EPSILON = 1.0e-6


def calibration_id(output_dir: Path) -> str:
    """Use the attempt-specific output directory as the calibration identity."""
    return output_dir.name


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "generate", "score"))
    parser.add_argument("--policy-model", type=Path, required=True)
    parser.add_argument("--rm-model", type=Path, required=True)
    parser.add_argument("--cm-model", type=Path, required=True)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--compatibility-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rollouts-source", type=Path)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _load_prompts(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    if len(rows) != PROMPT_COUNT:
        raise ValueError(f"expected {PROMPT_COUNT} calibration prompts, found {len(rows)}")
    prompt_ids: set[str] = set()
    for row in rows:
        prompt = row.get("prompt")
        extra_info = row.get("extra_info") or {}
        prompt_id = f"alpaca-train-{extra_info.get('index')}"
        if not isinstance(prompt, list) or not prompt:
            raise ValueError(f"calibration row {prompt_id} has no chat prompt")
        if prompt_id in prompt_ids:
            raise ValueError(f"duplicate calibration prompt id: {prompt_id}")
        prompt_ids.add(prompt_id)
        row["_prompt_id"] = prompt_id
    return rows


def _validate_assets(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.policy_model, args.rm_model, args.cm_model):
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"model config is missing: {path}")
        if list(path.rglob("*.incomplete")):
            raise RuntimeError(f"model has incomplete files: {path}")
    if not args.input_parquet.is_file():
        raise FileNotFoundError(args.input_parquet)
    compatibility = json.loads(args.compatibility_summary.read_text(encoding="utf-8"))
    if compatibility.get("status") != "passed":
        raise RuntimeError("H/S compatibility gate has not passed")
    rows = _load_prompts(args.input_parquet)

    from transformers import AutoTokenizer

    policy_tokenizer = AutoTokenizer.from_pretrained(
        args.policy_model, trust_remote_code=False
    )
    prompt_lengths: list[int] = []
    for row in rows:
        formatted = policy_tokenizer.apply_chat_template(
            row["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        token_count = len(
            policy_tokenizer(formatted, add_special_tokens=False)["input_ids"]
        )
        prompt_lengths.append(token_count)
    if max(prompt_lengths) > MAX_PROMPT_LENGTH:
        raise RuntimeError(
            f"calibration prompt exceeds {MAX_PROMPT_LENGTH}: max={max(prompt_lengths)}"
        )
    return {
        "schema_version": 1,
        "status": "ready",
        "prompt_count": len(rows),
        "rollouts_per_prompt": ROLLOUTS_PER_PROMPT,
        "rollout_count": len(rows) * ROLLOUTS_PER_PROMPT,
        "seed": SEED,
        "max_prompt_tokens_observed": max(prompt_lengths),
        "sampling": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_response_length": MAX_RESPONSE_LENGTH,
        },
        "compatibility_summary": str(args.compatibility_summary.resolve()),
    }


def _generate(args: argparse.Namespace) -> None:
    rollout_path = args.output_dir / "rollouts.jsonl"
    manifest_path = args.output_dir / "generation_manifest.json"
    if rollout_path.exists() or manifest_path.exists():
        raise RuntimeError("calibration generation output already exists")
    validation = _validate_assets(args)
    rows = _load_prompts(args.input_parquet)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.policy_model, trust_remote_code=False)
    prompts = [
        tokenizer.apply_chat_template(
            row["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]
    llm = LLM(
        model=str(args.policy_model),
        tokenizer=str(args.policy_model),
        trust_remote_code=False,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH,
        max_num_batched_tokens=16384,
        max_num_seqs=256,
        seed=SEED,
    )
    sampling = SamplingParams(
        n=ROLLOUTS_PER_PROMPT,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_RESPONSE_LENGTH,
    )
    outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=True)
    if len(outputs) != len(rows):
        raise RuntimeError("vLLM returned an unexpected request count")

    records: list[dict[str, Any]] = []
    for row, request in zip(rows, outputs, strict=True):
        if len(request.outputs) != ROLLOUTS_PER_PROMPT:
            raise RuntimeError(f"unexpected rollout count for {row['_prompt_id']}")
        for sample_id, completion in enumerate(request.outputs):
            token_ids = getattr(completion, "token_ids", None)
            token_count = (
                len(token_ids)
                if token_ids is not None
                else len(tokenizer(completion.text, add_special_tokens=False)["input_ids"])
            )
            records.append(
                {
                    "prompt_id": row["_prompt_id"],
                    "sample_id": sample_id,
                    "prompt": row["prompt"],
                    "response": completion.text,
                    "response_token_count": int(token_count),
                    "finish_reason": getattr(completion, "finish_reason", None),
                }
            )
    if len(records) != PROMPT_COUNT * ROLLOUTS_PER_PROMPT:
        raise RuntimeError("calibration rollout count is incomplete")
    if any(not row["response"].strip() for row in records):
        raise RuntimeError("calibration contains an empty response")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(rollout_path, records)
    _atomic_json(
        manifest_path,
        {
            **validation,
            "status": "generated",
            "policy_model": str(args.policy_model.resolve()),
            "input_parquet": str(args.input_parquet.resolve()),
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "response_token_summary": {
                "min": min(row["response_token_count"] for row in records),
                "max": max(row["response_token_count"] for row in records),
                "mean": statistics.fmean(row["response_token_count"] for row in records),
            },
        },
    )
    print(json.dumps({"status": "generated", "rollout_count": len(records)}))


def summarize_scores(values: list[float]) -> dict[str, float | int]:
    """Return the frozen population-scale statistics for one reward axis."""
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("reward scores must be nonempty and finite")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "population_std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def _load_rollouts(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(records) != PROMPT_COUNT * ROLLOUTS_PER_PROMPT:
        raise RuntimeError(f"expected 2048 rollouts, found {len(records)}")
    groups: dict[str, set[int]] = {}
    for row in records:
        groups.setdefault(str(row["prompt_id"]), set()).add(int(row["sample_id"]))
    if len(groups) != PROMPT_COUNT or any(
        samples != set(range(ROLLOUTS_PER_PROMPT)) for samples in groups.values()
    ):
        raise RuntimeError("calibration rollout grouping is invalid")
    return records


def _score_axis(
    *,
    records: list[dict[str, Any]],
    model_path: Path,
    device: Any,
    batch_size: int,
) -> tuple[list[float], str]:
    import torch

    from cw_grpo.helpfulness_safety_scoring import (
        CWQwen2RewardModel,
        format_reward_conversation,
        load_reward_tokenizer,
        score_reward_texts,
    )

    tokenizer = load_reward_tokenizer(model_path)
    texts = [
        format_reward_conversation(
            tokenizer,
            [*row["prompt"], {"role": "assistant", "content": row["response"]}],
        )
        for row in records
    ]
    model = CWQwen2RewardModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    scores = score_reward_texts(
        model,
        tokenizer,
        texts,
        device=device,
        max_length=2048,
        batch_size=batch_size,
    )
    tokenizer_class = type(tokenizer).__name__
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return scores, tokenizer_class


def _score(args: argparse.Namespace) -> None:
    import torch

    rollout_path = args.output_dir / "rollouts.jsonl"
    scored_path = args.output_dir / "scored_rollouts.jsonl"
    manifest_path = args.output_dir / "calibration_manifest.json"
    if scored_path.exists() or manifest_path.exists():
        raise RuntimeError("calibration scoring output already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("calibration scoring requires CUDA")
    records = _load_rollouts(rollout_path)
    device = torch.device(args.device)
    useful, useful_tokenizer = _score_axis(
        records=records,
        model_path=args.rm_model,
        device=device,
        batch_size=args.score_batch_size,
    )
    harmless, harmless_tokenizer = _score_axis(
        records=records,
        model_path=args.cm_model,
        device=device,
        batch_size=args.score_batch_size,
    )
    for row, useful_score, harmless_score in zip(
        records, useful, harmless, strict=True
    ):
        row["raw_useful"] = useful_score
        row["raw_harmless"] = harmless_score
    useful_summary = summarize_scores(useful)
    harmless_summary = summarize_scores(harmless)
    if useful_summary["population_std"] <= 0.0 or harmless_summary["population_std"] <= 0.0:
        raise RuntimeError("a calibration reward axis has zero population std")
    mean_u = float(useful_summary["mean"])
    mean_h = float(harmless_summary["mean"])
    covariance = statistics.fmean(
        (u - mean_u) * (h - mean_h) for u, h in zip(useful, harmless, strict=True)
    )
    correlation = covariance / (
        float(useful_summary["population_std"])
        * float(harmless_summary["population_std"])
    )
    _atomic_jsonl(scored_path, records)
    _atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "frozen",
            "scenario": "helpfulness_safety",
            "calibration_id": calibration_id(args.output_dir),
            "policy_model": str(args.policy_model.resolve()),
            "useful_model": str(args.rm_model.resolve()),
            "harmless_model": str(args.cm_model.resolve()),
            "input_parquet": str(args.input_parquet.resolve()),
            "compatibility_summary": str(args.compatibility_summary.resolve()),
            "rollouts_source": str(
                (args.rollouts_source or rollout_path).resolve()
            ),
            "prompt_count": PROMPT_COUNT,
            "rollouts_per_prompt": ROLLOUTS_PER_PROMPT,
            "rollout_count": len(records),
            "seed": SEED,
            "sampling": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_prompt_length": MAX_PROMPT_LENGTH,
                "max_response_length": MAX_RESPONSE_LENGTH,
            },
            "coordinates": {
                "definition": "u_i=(r_i-mu_i0)/(sigma_i0+epsilon)",
                "epsilon": CALIBRATION_EPSILON,
                "std_definition": "population_std_ddof_0",
                "useful": useful_summary,
                "harmless": harmless_summary,
                "raw_reward_correlation": correlation,
            },
            "tokenizers": {
                "useful": useful_tokenizer,
                "harmless": harmless_tokenizer,
                "formatter": "each reward model's own tokenizer chat template",
                "fix_mistral_regex": True,
            },
        },
    )
    print(
        json.dumps(
            {
                "status": "frozen",
                "useful": useful_summary,
                "harmless": harmless_summary,
                "correlation": correlation,
            },
            sort_keys=True,
        )
    )


def main() -> None:
    args = _parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.action == "validate":
        summary = _validate_assets(args)
        _atomic_json(args.output_dir / "validation.json", summary)
        print(json.dumps({"status": "ready", "output_dir": str(args.output_dir)}))
    elif args.action == "generate":
        _generate(args)
    else:
        _score(args)


if __name__ == "__main__":
    main()
