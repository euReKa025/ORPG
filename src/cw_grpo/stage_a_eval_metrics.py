from __future__ import annotations

from dataclasses import dataclass

DEFAULT_LENGTH_NORMALIZER = 32768.0


@dataclass(frozen=True, slots=True)
class BudgetPoint:
    max_new_tokens: int
    average_completion_tokens: float
    pass_at_1: float

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.average_completion_tokens < 0:
            raise ValueError("average completion tokens must be non-negative")
        if not 0 <= self.pass_at_1 <= 1:
            raise ValueError("pass@1 must be between zero and one")


def pareto_front(points: tuple[BudgetPoint, ...]) -> tuple[BudgetPoint, ...]:
    """Return points not dominated under lower length and higher accuracy."""

    front = []
    for index, point in enumerate(points):
        dominated = any(
            other_index != index
            and other.average_completion_tokens <= point.average_completion_tokens
            and other.pass_at_1 >= point.pass_at_1
            and (
                other.average_completion_tokens < point.average_completion_tokens
                or other.pass_at_1 > point.pass_at_1
            )
            for other_index, other in enumerate(points)
        )
        if not dominated:
            front.append(point)
    return tuple(sorted(front, key=lambda point: point.max_new_tokens))


def accuracy_length_hypervolume(
    points: tuple[BudgetPoint, ...],
    *,
    length_normalizer: float = DEFAULT_LENGTH_NORMALIZER,
) -> float:
    """Compute the 2-D maximization HV relative to (0, 0).

    Length is mapped to efficiency as ``1 - clip(avg_len / normalizer, 0, 1)``.
    Each point then contributes the rectangle from the origin to
    ``(efficiency, pass@1)``.
    """

    if length_normalizer <= 0:
        raise ValueError("length_normalizer must be positive")
    if not points:
        return 0.0

    accuracy_by_efficiency: dict[float, float] = {}
    for point in points:
        normalized_length = min(
            max(point.average_completion_tokens / length_normalizer, 0.0),
            1.0,
        )
        efficiency = 1.0 - normalized_length
        accuracy_by_efficiency[efficiency] = max(
            accuracy_by_efficiency.get(efficiency, 0.0),
            point.pass_at_1,
        )

    efficiencies = sorted(accuracy_by_efficiency)
    suffix_max_accuracy = [0.0] * len(efficiencies)
    running_max = 0.0
    for index in range(len(efficiencies) - 1, -1, -1):
        running_max = max(running_max, accuracy_by_efficiency[efficiencies[index]])
        suffix_max_accuracy[index] = running_max

    hypervolume = 0.0
    previous_efficiency = 0.0
    for efficiency, max_accuracy in zip(
        efficiencies,
        suffix_max_accuracy,
        strict=True,
    ):
        hypervolume += (efficiency - previous_efficiency) * max_accuracy
        previous_efficiency = efficiency
    return hypervolume
