from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any

from cw_grpo.stage_a_eval_metrics import (
    BudgetPoint,
    accuracy_length_hypervolume,
    pareto_front,
)

STAGE_A_BUDGETS = (2048, 4096, 8192, 16384, 32768)


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _metric_point(
    raw_metric: Mapping[str, Any],
    *,
    payload_budget: int,
) -> dict[str, Any]:
    benchmark = raw_metric.get("benchmark")
    metric_budget = raw_metric.get("max_new_tokens")
    prompt_count = raw_metric.get("prompt_count")
    sample_count = raw_metric.get("sample_count")
    pass_at_1 = raw_metric.get("pass_at_1")
    average_tokens = raw_metric.get("average_completion_tokens")
    parse_failure_rate = raw_metric.get("parse_failure_rate")
    if not isinstance(benchmark, str) or not benchmark:
        raise ValueError("benchmark metric has an invalid name")
    if metric_budget != payload_budget:
        raise ValueError(f"benchmark {benchmark} has a budget mismatch")
    if (
        not isinstance(prompt_count, int)
        or isinstance(prompt_count, bool)
        or prompt_count <= 0
        or not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or sample_count <= 0
    ):
        raise ValueError(f"benchmark {benchmark} has invalid prompt/sample counts")
    if not isinstance(pass_at_1, (int, float)) or not isinstance(
        average_tokens, (int, float)
    ):
        raise TypeError(f"benchmark {benchmark} has invalid accuracy/length metrics")
    if not isinstance(parse_failure_rate, (int, float)) or not (
        0 <= float(parse_failure_rate) <= 1
    ):
        raise ValueError(f"benchmark {benchmark} has invalid parse failure rate")
    point = BudgetPoint(
        max_new_tokens=payload_budget,
        average_completion_tokens=float(average_tokens),
        pass_at_1=float(pass_at_1),
    )
    return {
        "benchmark": benchmark,
        "max_new_tokens": point.max_new_tokens,
        "prompt_count": prompt_count,
        "sample_count": sample_count,
        "pass_at_1": point.pass_at_1,
        "average_completion_tokens": point.average_completion_tokens,
        "parse_failure_rate": float(parse_failure_rate),
    }


def summarize_budget_metric_payloads(
    payloads: Sequence[Mapping[str, Any]],
    *,
    budget_grid: Sequence[int] = STAGE_A_BUDGETS,
    length_normalizer: float = 32768.0,
) -> dict[str, Any]:
    """Create per-benchmark and macro summaries from a fixed budget grid.

    The default preserves the historical Stage A five-budget/32768 schema.
    Stage B calls this function with ``(2048, 4096, 8192)`` and a length
    normalizer of ``8192``.
    """

    budgets = tuple(sorted(int(budget) for budget in budget_grid))
    if not budgets or len(set(budgets)) != len(budgets) or any(budget <= 0 for budget in budgets):
        raise ValueError("budget_grid must contain unique positive budgets")
    if length_normalizer <= 0:
        raise ValueError("length_normalizer must be positive")

    by_budget: dict[int, dict[str, dict[str, Any]]] = {}
    for payload in payloads:
        budget = payload.get("max_new_tokens")
        if (
            not isinstance(budget, int)
            or isinstance(budget, bool)
            or budget in by_budget
        ):
            raise ValueError("metric payload has an invalid or duplicate budget")
        raw_metrics = payload.get("benchmarks")
        if not _is_sequence(raw_metrics) or not raw_metrics:
            raise ValueError(f"metric payload for budget {budget} has no benchmarks")
        benchmark_metrics: dict[str, dict[str, Any]] = {}
        for raw_metric in raw_metrics:
            if not isinstance(raw_metric, Mapping):
                raise TypeError("benchmark metric must be a mapping")
            metric = _metric_point(raw_metric, payload_budget=budget)
            benchmark = metric["benchmark"]
            if benchmark in benchmark_metrics:
                raise ValueError(
                    f"metric payload for budget {budget} repeats benchmark {benchmark}"
                )
            benchmark_metrics[benchmark] = metric
        by_budget[budget] = benchmark_metrics

    if tuple(sorted(by_budget)) != budgets:
        raise ValueError(
            f"metric payload budget grid must equal {list(budgets)}"
        )
    benchmark_names = set(by_budget[budgets[0]])
    if not benchmark_names:
        raise ValueError("metric payload benchmark set is empty")
    for budget in budgets[1:]:
        if set(by_budget[budget]) != benchmark_names:
            raise ValueError("benchmark sets differ across budget grid")

    benchmark_summaries: list[dict[str, Any]] = []
    for benchmark in sorted(benchmark_names):
        metrics = [by_budget[budget][benchmark] for budget in budgets]
        counts = {
            (metric["prompt_count"], metric["sample_count"]) for metric in metrics
        }
        if len(counts) != 1:
            raise ValueError(
                f"benchmark {benchmark} prompt/sample counts differ across budgets"
            )
        points = tuple(
            BudgetPoint(
                max_new_tokens=metric["max_new_tokens"],
                average_completion_tokens=metric["average_completion_tokens"],
                pass_at_1=metric["pass_at_1"],
            )
            for metric in metrics
        )
        front = pareto_front(points)
        metrics_by_budget = {
            metric["max_new_tokens"]: metric for metric in metrics
        }
        standard_budget = budgets[-1]
        standard = metrics_by_budget[standard_budget]
        standard_suffix = str(standard_budget)
        benchmark_summaries.append(
            {
                "benchmark": benchmark,
                "prompt_count": standard["prompt_count"],
                "sample_count": standard["sample_count"],
                f"pass_at_1_at_{standard_suffix}": standard["pass_at_1"],
                f"average_completion_tokens_at_{standard_suffix}": standard[
                    "average_completion_tokens"
                ],
                "pareto_hypervolume": accuracy_length_hypervolume(
                    points,
                    length_normalizer=length_normalizer,
                ),
                "budget_points": metrics,
                "pareto_front": [
                    metrics_by_budget[point.max_new_tokens] for point in front
                ],
            }
        )

    macro_average = {
        "benchmark_count": len(benchmark_summaries),
        f"pass_at_1_at_{budgets[-1]}": fmean(
            summary[f"pass_at_1_at_{budgets[-1]}"] for summary in benchmark_summaries
        ),
        f"average_completion_tokens_at_{budgets[-1]}": fmean(
            summary[f"average_completion_tokens_at_{budgets[-1]}"]
            for summary in benchmark_summaries
        ),
        "pareto_hypervolume": fmean(
            summary["pareto_hypervolume"] for summary in benchmark_summaries
        ),
    }
    return {
        "schema_version": 1,
        "budget_grid": list(budgets),
        "length_normalizer": length_normalizer,
        "benchmarks": benchmark_summaries,
        "macro_average": macro_average,
    }
