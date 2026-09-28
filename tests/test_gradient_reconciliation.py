from __future__ import annotations

import torch

from cw_grpo.gradient_reconciliation import (
    GradientCollection,
    cagrad_reconcile,
    correctness_priority_pcgrad_reconcile,
    modulewise_correctness_priority_pcgrad_reconcile,
    pcgrad_reconcile,
    sum_gradients,
)


def _collection(*vectors: list[float]) -> GradientCollection:
    return GradientCollection(
        objectives=[[torch.tensor(vector, dtype=torch.float32)] for vector in vectors]
    )


def test_sum_gradients_matches_scalar_sum_backward() -> None:
    parameter = torch.tensor([1.0, -2.0], requires_grad=True)
    loss_1 = (parameter * torch.tensor([2.0, 3.0])).sum()
    loss_2 = (parameter * torch.tensor([-4.0, 5.0])).sum()
    expected = torch.autograd.grad(loss_1 + loss_2, parameter)[0]

    actual, diagnostics = sum_gradients(
        _collection([2.0, 3.0], [-4.0, 5.0])
    )

    assert torch.equal(actual[0], expected)
    assert diagnostics.projection_rate == 0.0


def test_pcgrad_preserves_non_conflicting_gradient_sum() -> None:
    actual, diagnostics = pcgrad_reconcile(
        _collection([1.0, 0.0], [1.0, 1.0]),
        epsilon=1e-12,
    )

    assert torch.allclose(actual[0], torch.tensor([2.0, 1.0]))
    assert diagnostics.conflict_rate == 0.0
    assert diagnostics.projection_rate == 0.0
    assert diagnostics.cosine_similarity > 0.0


def test_pcgrad_projects_both_ordered_directions_for_two_objectives() -> None:
    actual, diagnostics = pcgrad_reconcile(
        _collection([1.0, 0.0], [-1.0, 1.0]),
        epsilon=1e-12,
    )

    # g1' = [1,0] - (-1/2)[-1,1] = [0.5,0.5]
    # g2' = [-1,1] - (-1/1)[1,0] = [0,1]
    assert torch.allclose(actual[0], torch.tensor([0.5, 1.5]))
    assert diagnostics.conflict_rate == 1.0
    assert diagnostics.projection_rate == 1.0
    assert diagnostics.cosine_similarity < 0.0
    assert diagnostics.combined_norm_after > 0.0


def test_pcgrad_handles_a_zero_objective_without_projection() -> None:
    actual, diagnostics = pcgrad_reconcile(
        _collection([0.0, 0.0], [2.0, -3.0]),
        epsilon=1e-12,
    )

    assert torch.equal(actual[0], torch.tensor([2.0, -3.0]))
    assert diagnostics.conflict_rate == 0.0
    assert diagnostics.projection_rate == 0.0


def test_all_reconcilers_bypass_solver_for_zero_secondary_objective() -> None:
    expected = torch.tensor([1.0, -2.0])
    cases = (
        lambda collection: sum_gradients(collection),
        lambda collection: pcgrad_reconcile(collection),
        lambda collection: correctness_priority_pcgrad_reconcile(collection),
        lambda collection: modulewise_correctness_priority_pcgrad_reconcile(
            collection,
            parameter_groups=("all",),
        ),
        lambda collection: cagrad_reconcile(collection),
    )

    for reconcile in cases:
        actual, diagnostics = reconcile(
            _collection(expected.tolist(), [0.0, 0.0])
        )
        assert torch.equal(actual[0], expected)
        assert diagnostics.active_objective_count == 1
        assert diagnostics.solver_bypassed
        assert diagnostics.projection_rate == 0.0


def test_correctness_priority_deviation_tracks_small_secondary_scale() -> None:
    actual, diagnostics = correctness_priority_pcgrad_reconcile(
        _collection([1.0, 0.0], [-1e-3, 1e-3])
    )

    assert torch.allclose(actual[0], torch.tensor([1.0, 1e-3]))
    assert torch.linalg.vector_norm(actual[0] - torch.tensor([1.0, 0.0])) <= 1.01e-3
    assert diagnostics.active_objective_count == 2
    assert not diagnostics.solver_bypassed


def test_correctness_priority_pcgrad_preserves_primary_gradient() -> None:
    actual, diagnostics = correctness_priority_pcgrad_reconcile(
        _collection([1.0, 0.0], [-1.0, 1.0]),
        epsilon=1e-12,
    )

    assert torch.allclose(actual[0], torch.tensor([1.0, 1.0]))
    assert diagnostics.conflict_rate == 1.0
    assert diagnostics.projection_rate == 1.0


def test_modulewise_priority_pcgrad_projects_only_conflicting_module() -> None:
    gradients = GradientCollection(
        objectives=[
            [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])],
            [torch.tensor([-1.0, 1.0]), torch.tensor([1.0, 0.0])],
        ]
    )
    actual, diagnostics = modulewise_correctness_priority_pcgrad_reconcile(
        gradients,
        parameter_groups=("attention", "mlp"),
        epsilon=1e-12,
    )

    assert torch.allclose(actual[0], torch.tensor([1.0, 1.0]))
    assert torch.allclose(actual[1], torch.tensor([1.0, 1.0]))
    assert diagnostics.conflict_rate == 1.0
    assert diagnostics.local_conflict_rate == 0.5
    assert diagnostics.projection_rate == 0.5
    assert diagnostics.local_projection_rate == 0.5


def test_cagrad_zero_conflict_aversion_matches_literal_sum_scale() -> None:
    actual, diagnostics = cagrad_reconcile(
        _collection([1.0, 2.0], [3.0, 4.0]),
        conflict_aversion=0.0,
    )

    assert torch.allclose(actual[0], torch.tensor([4.0, 6.0]))
    assert diagnostics.combined_norm_before == diagnostics.combined_norm_after


def test_cagrad_matches_official_two_objective_bounded_solution() -> None:
    actual, diagnostics = cagrad_reconcile(
        _collection([1.0, 0.0], [-1.0, 1.0]),
        conflict_aversion=0.5,
        max_iterations=96,
    )

    # Official CAGrad's mean-scale result is approximately
    # [0.166650, 0.333334]; the project multiplies by K=2 so that c=0
    # exactly matches OW-GRPO-Sum's literal gradient scale.
    assert torch.allclose(
        actual[0],
        torch.tensor([0.333300, 0.666668]),
        atol=2e-5,
        rtol=2e-5,
    )
    assert diagnostics.conflict_rate == 1.0
    assert diagnostics.projection_rate == 0.0


def test_cagrad_requires_exactly_two_objectives() -> None:
    collection = _collection([1.0], [2.0], [3.0])

    try:
        cagrad_reconcile(collection)
    except ValueError as error:
        assert "exactly two" in str(error)
    else:
        raise AssertionError("CAGrad must reject unsupported objective counts")
