import pytest

from orpg.stage_a_eval_metrics import (
    BudgetPoint,
    accuracy_length_hypervolume,
    pareto_front,
)


def test_pareto_front_uses_actual_length_and_filters_dominated_budgets() -> None:
    points = (
        BudgetPoint(max_new_tokens=2048, average_completion_tokens=1000, pass_at_1=0.30),
        BudgetPoint(max_new_tokens=4096, average_completion_tokens=2000, pass_at_1=0.50),
        BudgetPoint(max_new_tokens=8192, average_completion_tokens=2500, pass_at_1=0.45),
        BudgetPoint(max_new_tokens=16384, average_completion_tokens=5000, pass_at_1=0.70),
        BudgetPoint(max_new_tokens=32768, average_completion_tokens=6000, pass_at_1=0.75),
    )

    front = pareto_front(points)

    assert [point.max_new_tokens for point in front] == [2048, 4096, 16384, 32768]


def test_accuracy_length_hypervolume_matches_union_of_rectangles() -> None:
    points = (
        BudgetPoint(
            max_new_tokens=2048,
            average_completion_tokens=16384,
            pass_at_1=0.80,
        ),
        BudgetPoint(
            max_new_tokens=32768,
            average_completion_tokens=6553.6,
            pass_at_1=0.50,
        ),
    )

    # Efficiency points are (0.5, 0.8) and (0.8, 0.5), so the union area is
    # 0.5*0.8 + (0.8-0.5)*0.5 = 0.55.
    assert accuracy_length_hypervolume(points) == pytest.approx(0.55)


@pytest.mark.parametrize(
    "average_length,pass_at_1",
    [(-1, 0.5), (100, -0.1), (100, 1.1)],
)
def test_budget_point_rejects_invalid_metrics(
    average_length: float,
    pass_at_1: float,
) -> None:
    with pytest.raises(ValueError):
        BudgetPoint(
            max_new_tokens=2048,
            average_completion_tokens=average_length,
            pass_at_1=pass_at_1,
        )
