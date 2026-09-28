#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from cw_grpo.stage_a_eval_generation import (
    EvalRequest,
    build_chat_completion_payload,
    build_eval_request_plan,
    evaluation_runtime_aliases,
    mark_evaluation_manifest,
    resolve_eval_model_path,
)
from cw_grpo.stage_a_grpo import MODEL_PATH, REMOTE_PROJECT_ROOT

PROJECT_ROOT = Path(
    os.environ.get("ORPG_ROOT", str(REMOTE_PROJECT_ROOT))
).expanduser().resolve()
VERL_ROOT = Path(os.environ.get("VERL_ROOT", PROJECT_ROOT / "third_party/verl"))
DEFAULT_BUDGET = 32768
_EXPERIMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


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


def _personal_path(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return resolved
    for env_name in (
        "CW_GRPO_RELOCATED_DATA_ROOT",
        "CW_GRPO_RELOCATED_OUTPUT_ROOT",
    ):
        relocated_root = os.environ.get(env_name)
        if not relocated_root:
            continue
        allowed_root = Path(relocated_root).expanduser().resolve()
        if resolved.is_relative_to(allowed_root):
            return resolved
    raise ValueError(f"{label} must stay under the personal project root")


def _allowed_model_path(path: Path) -> Path:
    return resolve_eval_model_path(
        path,
        public_model_path=Path(str(MODEL_PATH)),
        personal_checkpoint_root=PROJECT_ROOT / "checkpoints",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic Stage A math evaluation completions with vLLM."
    )
    parser.add_argument("--action", choices=("resolve", "generate"), required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--model-path", type=Path, default=Path(str(MODEL_PATH)))
    parser.add_argument("--input-parquet", action="append", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--samples-per-prompt", type=int)
    parser.add_argument("--sample-id", action="append", type=int)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--n-gpus-per-node", type=int, required=True)
    parser.add_argument("--rollout-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=34816)
    parser.add_argument("--max-num-batched-tokens", type=int, default=65536)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-concurrency-per-replica", type=int, default=64)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not _EXPERIMENT_NAME.fullmatch(args.experiment_name):
        raise ValueError("invalid experiment name")
    if args.samples_per_prompt is not None and args.sample_id is not None:
        raise ValueError("samples-per-prompt and sample-id are mutually exclusive")
    if args.sample_id is None:
        if args.samples_per_prompt is None:
            args.samples_per_prompt = 16
        if args.samples_per_prompt <= 0:
            raise ValueError("samples-per-prompt must be positive")
    elif (
        not args.sample_id
        or any(sample_id < 0 for sample_id in args.sample_id)
        or len(set(args.sample_id)) != len(args.sample_id)
        or args.sample_id != sorted(args.sample_id)
    ):
        raise ValueError(
            "sample-id must be repeated in unique increasing non-negative order"
        )
    if args.base_seed < 0:
        raise ValueError("base-seed must be non-negative")
    if args.max_prompts is not None and args.max_prompts <= 0:
        raise ValueError("max-prompts must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    if not 0 < args.temperature or not 0 < args.top_p <= 1:
        raise ValueError("temperature must be positive and top-p in (0, 1]")
    if args.n_gpus_per_node not in {1, 2, 4, 8}:
        raise ValueError("evaluation requires 1, 2, 4, or 8 GPUs")
    if args.rollout_tensor_parallel_size not in {1, 2}:
        raise ValueError("evaluation TP must be 1 or 2")
    if args.n_gpus_per_node % args.rollout_tensor_parallel_size:
        raise ValueError("GPU count must be divisible by evaluation TP")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("gpu-memory-utilization must be between zero and one")
    if args.max_model_len < args.max_new_tokens:
        raise ValueError("max-model-len must cover max-new-tokens")
    if min(
        args.max_num_batched_tokens,
        args.max_num_seqs,
        args.max_concurrency_per_replica,
    ) <= 0:
        raise ValueError("evaluation batching and concurrency values must be positive")


def _read_inputs(
    input_paths: Sequence[Path],
    *,
    max_prompts: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    input_provenance: list[dict[str, Any]] = []
    for raw_path in input_paths:
        path = _personal_path(raw_path, label="input parquet")
        table = pq.read_table(path)
        provenance = {
            "path": str(path),
            "rows": table.num_rows,
            "bytes": path.stat().st_size,
        }
        if _record_sha256():
            provenance["sha256"] = _sha256(path)
        input_provenance.append(provenance)
        rows.extend(table.to_pylist())

    rows.sort(key=lambda row: row["extra_info"]["prompt_id"])
    if max_prompts is not None:
        rows = rows[:max_prompts]
    if not rows:
        raise ValueError("evaluation input is empty")
    return rows, input_provenance


def _runtime_env(experiment_name: str) -> dict[str, str]:
    run_hash = sha256(experiment_name.encode()).hexdigest()[:8]
    run_root = PROJECT_ROOT / ".runtime/evaluation" / experiment_name
    physical_ray = PROJECT_ROOT / ".r/r" / run_hash
    physical_tmp = PROJECT_ROOT / ".r/t" / run_hash
    paths = {
        "HOME": run_root / "home",
        "XDG_CACHE_HOME": run_root / "cache/xdg",
        "HF_HOME": PROJECT_ROOT / ".cache/huggingface",
        "VLLM_CACHE_ROOT": run_root / "cache/vllm",
        "TORCHINDUCTOR_CACHE_DIR": run_root / "cache/torchinductor",
        "TRITON_CACHE_DIR": run_root / "cache/triton",
        "CUDA_CACHE_PATH": run_root / "cache/cuda",
        "PYTHONPYCACHEPREFIX": run_root / "cache/pycache",
    }
    for path in (*paths.values(), physical_ray, physical_tmp):
        path.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({key: str(path) for key, path in paths.items()})
    aliases = evaluation_runtime_aliases(pid=os.getpid(), run_hash=run_hash)
    env.update(
        {
            "TMPDIR": aliases["tmp"],
            "TMP": aliases["tmp"],
            "TEMP": aliases["tmp"],
            "RAY_TMPDIR": aliases["ray"],
            "TOKENIZERS_PARALLELISM": "false",
            "NCCL_DEBUG": "WARN",
            "VLLM_USE_V1": "1",
        }
    )
    env.pop("RAY_ADDRESS", None)
    return env


def _write_resolved_manifest(
    args: argparse.Namespace,
    *,
    model_path: Path,
    output_path: Path,
    rows: Sequence[Mapping[str, Any]],
    requests: Sequence[EvalRequest],
    input_provenance: Sequence[Mapping[str, Any]],
) -> Path:
    run_dir = PROJECT_ROOT / "outputs" / args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "evaluation_manifest.json"
    manifest = {
        "schema_version": 1,
        "status": "resolved",
        "experiment_name": args.experiment_name,
        "model_path": str(model_path),
        "output_parquet": str(output_path),
        "input_parquets": list(input_provenance),
        "prompt_count": len(rows),
        "request_count": len(requests),
        "samples_per_prompt": len({request.sample_id for request in requests}),
        "sample_ids": sorted({request.sample_id for request in requests}),
        "base_seed": args.base_seed,
        "max_prompts": args.max_prompts,
        "sampling": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "runtime": {
            "n_gpus_per_node": args.n_gpus_per_node,
            "rollout_tensor_parallel_size": args.rollout_tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "max_concurrency_per_replica": args.max_concurrency_per_replica,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _build_verl_config(args: argparse.Namespace, model_path: Path):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    overrides = [
        f"trainer.n_gpus_per_node={args.n_gpus_per_node}",
        "trainer.nnodes=1",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.trust_remote_code=false",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.response_length={args.max_new_tokens}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.rollout_tensor_parallel_size}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={args.gpu_memory_utilization}",
        f"actor_rollout_ref.rollout.max_model_len={args.max_model_len}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={args.max_num_batched_tokens}",
        f"actor_rollout_ref.rollout.max_num_seqs={args.max_num_seqs}",
    ]
    with initialize_config_dir(
        config_dir=str(VERL_ROOT / "verl/trainer/config"),
        version_base=None,
    ):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    OmegaConf.resolve(config)
    return config


async def _start_servers(config):
    from verl.workers.rollout.replica import get_rollout_replica_class

    tp_size = config.actor_rollout_ref.rollout.tensor_model_parallel_size
    replicas = config.trainer.n_gpus_per_node // tp_size
    server_class = get_rollout_replica_class(config.actor_rollout_ref.rollout.name)
    servers = [
        server_class(
            replica_rank=rank,
            config=config.actor_rollout_ref.rollout,
            model_config=config.actor_rollout_ref.model,
            gpus_per_node=config.trainer.n_gpus_per_node,
        )
        for rank in range(replicas)
    ]
    # Initialize each TP=1 replica serially. The Ray worker-group helper chooses
    # a short-lived free TCP port; concurrent initialization can hand the same
    # port to multiple world-size-one workers before either binds it.
    for server in servers:
        await server.init_standalone()
    server_handles = [server._server_handle for server in servers]
    server_addresses = [server._server_address for server in servers]
    if len(server_handles) != len(server_addresses):
        raise RuntimeError("rollout server handle/address count mismatch")
    return server_handles, server_addresses


async def _generate_on_server(
    server_address: str,
    indexed_requests: Sequence[tuple[int, EvalRequest]],
    *,
    model_path: Path,
    args: argparse.Namespace,
) -> list[tuple[int, dict[str, Any]]]:
    import aiohttp

    semaphore = asyncio.Semaphore(args.max_concurrency_per_replica)
    timeout = aiohttp.ClientTimeout(total=None)
    headers = {"Authorization": "Bearer token-abc123"}

    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        async def submit(index: int, request: EvalRequest):
            payload = build_chat_completion_payload(
                request,
                model_path=model_path,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
            )
            async with semaphore, session.post(
                f"http://{server_address}/v1/chat/completions",
                json=payload,
            ) as response:
                body = await response.text()
                if response.status != 200:
                    raise RuntimeError(
                        f"vLLM request failed with HTTP {response.status}: {body[:1000]}"
                    )
                data = json.loads(body)
            choice = data["choices"][0]
            usage = data.get("usage") or {}
            response_token_ids = choice.get("token_ids")
            if not isinstance(response_token_ids, list) or not all(
                isinstance(token_id, int) and token_id >= 0
                for token_id in response_token_ids
            ):
                raise RuntimeError("vLLM response did not include valid output token ids")
            return index, {
                "response": choice["message"].get("content") or "",
                "response_token_ids": response_token_ids,
                "finish_reason": choice.get("finish_reason") or "unknown",
                "completion_tokens_api": usage.get("completion_tokens"),
                "seed": request.seed,
                "sample_id": request.sample_id,
            }

        return await asyncio.gather(
            *(submit(index, request) for index, request in indexed_requests)
        )


async def _generate(
    config,
    requests: Sequence[EvalRequest],
    *,
    model_path: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    server_handles, addresses = await _start_servers(config)
    if not server_handles or len(server_handles) != len(addresses):
        raise RuntimeError("rollout server ownership was not retained")
    indexed = list(enumerate(requests))
    shard_size = (len(indexed) + len(addresses) - 1) // len(addresses)
    shards = [
        indexed[start : start + shard_size]
        for start in range(0, len(indexed), shard_size)
    ]
    shards.extend([[] for _ in range(len(addresses) - len(shards))])
    nested = await asyncio.gather(
        *(
            _generate_on_server(
                address,
                shard,
                model_path=model_path,
                args=args,
            )
            for address, shard in zip(addresses, shards, strict=True)
        )
    )
    indexed_results = sorted(
        (item for shard in nested for item in shard),
        key=lambda item: item[0],
    )
    return [result for _, result in indexed_results]


def _write_generations(
    rows: Sequence[Mapping[str, Any]],
    requests: Sequence[EvalRequest],
    results: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for request, result in zip(requests, results, strict=True):
        grouped.setdefault(request.prompt_id, []).append(result)

    output_rows = []
    for row in rows:
        prompt_identifier = row["extra_info"]["prompt_id"]
        prompt_results = sorted(
            grouped[prompt_identifier],
            key=lambda result: result["sample_id"],
        )
        output_rows.append(
            {
                **dict(row),
                "responses": [result["response"] for result in prompt_results],
                "response_token_ids": [
                    result["response_token_ids"] for result in prompt_results
                ],
                "finish_reasons": [
                    result["finish_reason"] for result in prompt_results
                ],
                "completion_tokens_api": [
                    result["completion_tokens_api"] for result in prompt_results
                ],
                "request_seeds": [result["seed"] for result in prompt_results],
                "sample_ids": [result["sample_id"] for result in prompt_results],
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(output_rows),
        output_path,
        compression="zstd",
    )


def main() -> int:
    args = _parser().parse_args()
    _validate_args(args)
    model_path = _allowed_model_path(args.model_path)
    output_path = _personal_path(args.output_parquet, label="output parquet")
    rows, input_provenance = _read_inputs(
        args.input_parquet,
        max_prompts=args.max_prompts,
    )
    requests = build_eval_request_plan(
        rows,
        samples_per_prompt=(
            args.samples_per_prompt if args.sample_id is None else None
        ),
        sample_ids=args.sample_id,
        base_seed=args.base_seed,
    )
    manifest_path = _write_resolved_manifest(
        args,
        model_path=model_path,
        output_path=output_path,
        rows=rows,
        requests=requests,
        input_provenance=input_provenance,
    )
    print(f"resolved evaluation manifest: {manifest_path}", flush=True)
    config = _build_verl_config(args, model_path)
    from omegaconf import OmegaConf

    resolved_config_path = manifest_path.with_name("resolved_eval_config.yaml")
    resolved_config_path.write_text(
        OmegaConf.to_yaml(config, resolve=True),
        encoding="utf-8",
    )
    print(f"resolved evaluation config: {resolved_config_path}", flush=True)
    if args.action == "resolve":
        return 0

    mark_evaluation_manifest(manifest_path, status="running")
    runtime_env = _runtime_env(args.experiment_name)
    os.chdir(PROJECT_ROOT)
    os.environ.update(runtime_env)
    import ray

    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "false",
                "NCCL_DEBUG": "WARN",
                "VLLM_USE_V1": "1",
            }
        }
    )
    try:
        results = asyncio.run(
            _generate(config, requests, model_path=model_path, args=args)
        )
        _write_generations(rows, requests, results, output_path)
        mark_evaluation_manifest(
            manifest_path,
            status="completed",
            output_path=output_path,
        )
    except Exception as error:
        mark_evaluation_manifest(
            manifest_path,
            status="failed",
            error_type=type(error).__name__,
            error_message=str(error) or repr(error),
        )
        raise
    finally:
        ray.shutdown()
    print(f"saved generations: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
