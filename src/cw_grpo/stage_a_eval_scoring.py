from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from cw_grpo.math_stage_a_reward import score_stage_a_response

ScoreFunction = Callable[..., Mapping[str, float | int]]


@dataclass(frozen=True, slots=True)
class BenchmarkBudgetMetrics:
    benchmark: str
    max_new_tokens: int
    prompt_count: int
    sample_count: int
    pass_at_1: float
    average_completion_tokens: float
    parse_failure_rate: float


def _response_arrays(
    row: Mapping[str, Any],
    *,
    row_number: int,
) -> tuple[Sequence[Any], Sequence[Any], Sequence[Any], Sequence[Any], Sequence[Any]]:
    fields = (
        row.get("responses"),
        row.get("response_token_ids"),
        row.get("completion_tokens_api"),
        row.get("finish_reasons"),
        row.get("request_seeds"),
    )
    if not all(
        isinstance(field, Sequence)
        and not isinstance(field, (str, bytes, bytearray))
        for field in fields
    ):
        raise TypeError(f"generation row {row_number} has invalid response arrays")
    lengths = {len(field) for field in fields}
    if len(lengths) != 1 or not next(iter(lengths)):
        raise ValueError(f"generation row {row_number} has misaligned response arrays")
    return fields


def score_generation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_new_tokens: int,
    scorer: ScoreFunction | None = None,
    max_workers: int = 4,
) -> tuple[list[dict[str, Any]], tuple[BenchmarkBudgetMetrics, ...]]:
    """Score deterministic generation rows with the shared Stage A verifier."""

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if not rows:
        raise ValueError("generation rows are empty")

    sample_inputs: list[tuple[str, str, int]] = []
    sample_locations: list[tuple[int, int, str]] = []
    sample_counts_by_row: list[int] = []
    prompts_by_benchmark: defaultdict[str, set[str]] = defaultdict(set)

    for row_number, row in enumerate(rows):
        extra_info = row.get("extra_info")
        reward_model = row.get("reward_model")
        if not isinstance(extra_info, Mapping) or not isinstance(reward_model, Mapping):
            raise TypeError(f"generation row {row_number} has invalid metadata")
        benchmark = str(extra_info.get("benchmark", "")).strip()
        prompt_id = str(extra_info.get("prompt_id", "")).strip()
        ground_truth = str(reward_model.get("ground_truth", "")).strip()
        if not benchmark or not prompt_id or not ground_truth:
            raise ValueError(f"generation row {row_number} has incomplete identity")
        prompts_by_benchmark[benchmark].add(prompt_id)

        responses, token_id_groups, api_counts, _, _ = _response_arrays(
            row,
            row_number=row_number,
        )
        sample_counts_by_row.append(len(responses))
        for sample_id, (response, raw_token_ids, api_count) in enumerate(
            zip(responses, token_id_groups, api_counts, strict=True)
        ):
            if not isinstance(raw_token_ids, Sequence) or isinstance(
                raw_token_ids, (str, bytes, bytearray)
            ):
                raise TypeError(f"generation row {row_number} has invalid token ids")
            token_ids = tuple(int(token_id) for token_id in raw_token_ids)
            if any(token_id < 0 for token_id in token_ids):
                raise ValueError(f"generation row {row_number} has negative token ids")
            if not isinstance(api_count, int) or api_count != len(token_ids):
                raise ValueError(
                    f"generation row {row_number} sample {sample_id} token count mismatch"
                )
            sample_inputs.append((str(response), ground_truth, len(token_ids)))
            sample_locations.append((row_number, sample_id, benchmark))

    score_function = scorer or score_stage_a_response

    def invoke(sample_input: tuple[str, str, int]) -> Mapping[str, float | int]:
        completion, ground_truth, token_count = sample_input
        return score_function(
            completion,
            ground_truth,
            response_token_count=token_count,
        )

    if scorer is None and len(sample_inputs) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            score_results = list(executor.map(invoke, sample_inputs))
    else:
        score_results = [invoke(sample_input) for sample_input in sample_inputs]

    correctness_by_row = [[0.0] * count for count in sample_counts_by_row]
    parse_failures_by_row = [[0.0] * count for count in sample_counts_by_row]
    token_counts_by_row = [[0] * count for count in sample_counts_by_row]
    aggregate: defaultdict[str, dict[str, float]] = defaultdict(
        lambda: {"correct": 0.0, "parse_failure": 0.0, "tokens": 0.0, "samples": 0.0}
    )
    for location, score_result in zip(sample_locations, score_results, strict=True):
        row_number, sample_id, benchmark = location
        correctness = float(score_result["correctness_reward"])
        parse_failure = float(score_result["parse_failure"])
        token_count = int(score_result["response_token_count"])
        correctness_by_row[row_number][sample_id] = correctness
        parse_failures_by_row[row_number][sample_id] = parse_failure
        token_counts_by_row[row_number][sample_id] = token_count
        aggregate[benchmark]["correct"] += correctness
        aggregate[benchmark]["parse_failure"] += parse_failure
        aggregate[benchmark]["tokens"] += token_count
        aggregate[benchmark]["samples"] += 1

    scored_rows = []
    for row_number, row in enumerate(rows):
        scored_rows.append(
            {
                **dict(row),
                "correctness_rewards": correctness_by_row[row_number],
                "parse_failures": parse_failures_by_row[row_number],
                "response_token_counts": token_counts_by_row[row_number],
            }
        )

    metrics = []
    for benchmark in sorted(aggregate):
        values = aggregate[benchmark]
        sample_count = int(values["samples"])
        metrics.append(
            BenchmarkBudgetMetrics(
                benchmark=benchmark,
                max_new_tokens=max_new_tokens,
                prompt_count=len(prompts_by_benchmark[benchmark]),
                sample_count=sample_count,
                pass_at_1=values["correct"] / sample_count,
                average_completion_tokens=values["tokens"] / sample_count,
                parse_failure_rate=values["parse_failure"] / sample_count,
            )
        )
    return scored_rows, tuple(metrics)
