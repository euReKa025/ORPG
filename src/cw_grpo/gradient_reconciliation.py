from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import torch
from torch import Tensor

ScalarReducer = Callable[[Tensor], Tensor]


@dataclass(frozen=True, slots=True)
class GradientCollection:
    """Objective gradients in a stable trainable-parameter order."""

    objectives: Sequence[Sequence[Tensor]]

    def __post_init__(self) -> None:
        if not self.objectives:
            raise ValueError("at least one objective gradient is required")
        parameter_count = len(self.objectives[0])
        if parameter_count == 0:
            raise ValueError("objective gradients cannot have an empty parameter list")
        reference_shapes = tuple(tensor.shape for tensor in self.objectives[0])
        for objective in self.objectives:
            if len(objective) != parameter_count:
                raise ValueError("all objectives must use the same parameter order")
            if tuple(tensor.shape for tensor in objective) != reference_shapes:
                raise ValueError("objective gradient shapes must match")


@dataclass(frozen=True, slots=True)
class ReconciliationDiagnostics:
    cosine_similarity: float
    conflict_rate: float
    projection_rate: float
    combined_norm_before: float
    combined_norm_after: float
    objective_norms: tuple[float, ...]
    local_conflict_rate: float = 0.0
    local_projection_rate: float = 0.0
    active_objective_count: int = 0
    solver_bypassed: bool = False


def _identity(value: Tensor) -> Tensor:
    return value


def _dot(
    left: Sequence[Tensor],
    right: Sequence[Tensor],
    *,
    reduce_scalar: ScalarReducer,
) -> Tensor:
    local = sum(
        (
            (left_tensor.float() * right_tensor.float()).sum()
            for left_tensor, right_tensor in zip(left, right, strict=True)
        ),
        start=torch.zeros((), device=left[0].device, dtype=torch.float32),
    )
    return reduce_scalar(local)


def _local_dot(left: Sequence[Tensor], right: Sequence[Tensor]) -> Tensor:
    """Return a dot product without a distributed reduction."""

    return sum(
        (
            (left_tensor.float() * right_tensor.float()).sum()
            for left_tensor, right_tensor in zip(left, right, strict=True)
        ),
        start=torch.zeros((), device=left[0].device, dtype=torch.float32),
    )


def _norm(
    gradient: Sequence[Tensor],
    *,
    reduce_scalar: ScalarReducer,
    epsilon: float,
) -> Tensor:
    del epsilon
    squared = _dot(gradient, gradient, reduce_scalar=reduce_scalar)
    return squared.clamp_min(0.0).sqrt()


def _sum(objectives: Sequence[Sequence[Tensor]]) -> list[Tensor]:
    return [
        sum(
            (objective[index] for objective in objectives[1:]),
            start=objectives[0][index].clone(),
        )
        for index in range(len(objectives[0]))
    ]


def _diagnostics(
    originals: Sequence[Sequence[Tensor]],
    combined_before: Sequence[Tensor],
    combined_after: Sequence[Tensor],
    *,
    reduce_scalar: ScalarReducer,
    epsilon: float,
    projections: int,
    projection_opportunities: int | None = None,
    local_conflicts: int = 0,
    local_pair_count: int = 0,
    local_projections: int = 0,
    local_projection_opportunities: int = 0,
) -> ReconciliationDiagnostics:
    objective_norm_tensors = [
        _norm(objective, reduce_scalar=reduce_scalar, epsilon=epsilon)
        for objective in originals
    ]
    cosines: list[float] = []
    conflicts = 0
    pair_count = 0
    for left_index in range(len(originals)):
        for right_index in range(left_index + 1, len(originals)):
            pair_count += 1
            dot = _dot(
                originals[left_index],
                originals[right_index],
                reduce_scalar=reduce_scalar,
            )
            denominator = objective_norm_tensors[left_index] * objective_norm_tensors[
                right_index
            ]
            if denominator <= epsilon:
                cosine = 0.0
            else:
                cosine = float((dot / denominator).detach().cpu())
            cosines.append(cosine)
            conflicts += int(float(dot.detach().cpu()) < 0.0)

    ordered_pair_count = len(originals) * max(len(originals) - 1, 0)
    if projection_opportunities is None:
        projection_opportunities = ordered_pair_count
    return ReconciliationDiagnostics(
        cosine_similarity=sum(cosines) / len(cosines) if cosines else 0.0,
        conflict_rate=conflicts / pair_count if pair_count else 0.0,
        projection_rate=(
            projections / projection_opportunities
            if projection_opportunities
            else 0.0
        ),
        combined_norm_before=float(
            _norm(
                combined_before,
                reduce_scalar=reduce_scalar,
                epsilon=epsilon,
            )
            .detach()
            .cpu()
        ),
        combined_norm_after=float(
            _norm(
                combined_after,
                reduce_scalar=reduce_scalar,
                epsilon=epsilon,
            )
            .detach()
            .cpu()
        ),
        objective_norms=tuple(
            float(norm.detach().cpu()) for norm in objective_norm_tensors
        ),
        local_conflict_rate=(
            local_conflicts / local_pair_count if local_pair_count else 0.0
        ),
        local_projection_rate=(
            local_projections / local_projection_opportunities
            if local_projection_opportunities
            else 0.0
        ),
        active_objective_count=sum(
            float(norm.detach().cpu()) > 0.0 for norm in objective_norm_tensors
        ),
    )


def _bypass_if_at_most_one_objective_is_active(
    originals: Sequence[Sequence[Tensor]],
    *,
    reduce_scalar: ScalarReducer,
    epsilon: float,
) -> tuple[list[Tensor], ReconciliationDiagnostics] | None:
    """Return the literal active gradient without invoking a multi-objective solver."""

    local_squared_norms = torch.stack(
        [_local_dot(objective, objective) for objective in originals]
    )
    squared_norms = reduce_scalar(local_squared_norms)
    active_objective_count = sum(
        float(value.detach().cpu()) > 0.0 for value in squared_norms
    )
    if active_objective_count > 1:
        return None
    combined = _sum(originals)
    diagnostics = _diagnostics(
        originals,
        combined,
        combined,
        reduce_scalar=reduce_scalar,
        epsilon=epsilon,
        projections=0,
    )
    return combined, replace(diagnostics, solver_bypassed=True)


def sum_gradients(
    gradients: GradientCollection,
    *,
    reduce_scalar: ScalarReducer = _identity,
    epsilon: float = 1e-12,
) -> tuple[list[Tensor], ReconciliationDiagnostics]:
    originals = [list(objective) for objective in gradients.objectives]
    bypass = _bypass_if_at_most_one_objective_is_active(
        originals,
        reduce_scalar=reduce_scalar,
        epsilon=epsilon,
    )
    if bypass is not None:
        return bypass
    combined = _sum(originals)
    diagnostics = _diagnostics(
        originals,
        combined,
        combined,
        reduce_scalar=reduce_scalar,
        epsilon=epsilon,
        projections=0,
    )
    return combined, diagnostics


def pcgrad_reconcile(
    gradients: GradientCollection,
    *,
    reduce_scalar: ScalarReducer = _identity,
    epsilon: float = 1e-12,
) -> tuple[list[Tensor], ReconciliationDiagnostics]:
    """Apply reference PCGrad projections and preserve Sum's gradient scale."""

    originals = [
        [tensor.detach().clone() for tensor in objective]
        for objective in gradients.objectives
    ]
    bypass = _bypass_if_at_most_one_objective_is_active(
        originals,
        reduce_scalar=reduce_scalar,
        epsilon=epsilon,
    )
    if bypass is not None:
        return bypass
    projected = [
        [tensor.clone() for tensor in objective] for objective in originals
    ]
    projections = 0
    for left_index, left in enumerate(projected):
        for right_index, right in enumerate(originals):
            if left_index == right_index:
                continue
            dot = _dot(left, right, reduce_scalar=reduce_scalar)
            right_squared_norm = _dot(
                right,
                right,
                reduce_scalar=reduce_scalar,
            )
            if float(dot.detach().cpu()) < 0.0 and float(
                right_squared_norm.detach().cpu()
            ) > epsilon:
                coefficient = dot / right_squared_norm.clamp_min(epsilon)
                for parameter_index in range(len(left)):
                    left[parameter_index].sub_(
                        right[parameter_index] * coefficient.to(right[parameter_index])
                    )
                projections += 1

    combined_before = _sum(originals)
    combined_after = _sum(projected)
    diagnostics = _diagnostics(
        originals,
        combined_before,
        combined_after,
        reduce_scalar=reduce_scalar,
        epsilon=epsilon,
        projections=projections,
    )
    return combined_after, diagnostics


def correctness_priority_pcgrad_reconcile(
    gradients: GradientCollection,
    *,
    reduce_scalar: ScalarReducer = _identity,
    epsilon: float = 1e-12,
) -> tuple[list[Tensor], ReconciliationDiagnostics]:
    """Preserve correctness and project only the secondary length gradient.

    Objective index zero is the correctness objective and objective index one is
    the Stage A length objective.  When their global dot product is negative,
    the length gradient is projected onto the half-space orthogonal to the
    correctness gradient.  The correctness gradient is never modified.
    """

    if len(gradients.objectives) != 2:
        raise ValueError("correctness-priority PCGrad requires exactly two objectives")
    originals = [
        [tensor.detach().clone() for tensor in objective]
        for objective in gradients.objectives
    ]
    bypass = _bypass_if_at_most_one_objective_is_active(
        originals,
        reduce_scalar=reduce_scalar,
        epsilon=epsilon,
    )
    if bypass is not None:
        return bypass
    correctness, length = originals
    projected_length = [tensor.clone() for tensor in length]
    dot = _dot(length, correctness, reduce_scalar=reduce_scalar)
    correctness_squared_norm = _dot(
        correctness,
        correctness,
        reduce_scalar=reduce_scalar,
    )
    projected = int(
        float(dot.detach().cpu()) < 0.0
        and float(correctness_squared_norm.detach().cpu()) > epsilon
    )
    if projected:
        coefficient = dot / correctness_squared_norm.clamp_min(epsilon)
        for parameter_index in range(len(projected_length)):
            projected_length[parameter_index].sub_(
                correctness[parameter_index]
                * coefficient.to(correctness[parameter_index])
            )

    combined_before = _sum(originals)
    combined_after = [
        correctness_tensor + length_tensor
        for correctness_tensor, length_tensor in zip(
            correctness,
            projected_length,
            strict=True,
        )
    ]
    diagnostics = _diagnostics(
        originals,
        combined_before,
        combined_after,
        reduce_scalar=reduce_scalar,
        epsilon=epsilon,
        projections=projected,
        projection_opportunities=1,
    )
    return combined_after, diagnostics


def modulewise_correctness_priority_pcgrad_reconcile(
    gradients: GradientCollection,
    *,
    parameter_groups: Sequence[object] | None = None,
    reduce_scalar: ScalarReducer = _identity,
    epsilon: float = 1e-12,
) -> tuple[list[Tensor], ReconciliationDiagnostics]:
    """Apply correctness-priority PCGrad independently within module groups.

    ``parameter_groups`` maps the stable parameter sequence to module IDs.  A
    missing mapping means one block per parameter tensor.  Dots and norms are
    reduced in one vector collective so FSDP data-parallel shards make the same
    projection decision without one collective per module.
    """

    if len(gradients.objectives) != 2:
        raise ValueError(
            "module-wise correctness-priority PCGrad requires exactly two objectives"
        )
    originals = [
        [tensor.detach().clone() for tensor in objective]
        for objective in gradients.objectives
    ]
    parameter_count = len(originals[0])
    if parameter_groups is None:
        normalized_groups: tuple[object, ...] = tuple(range(parameter_count))
    else:
        normalized_groups = tuple(parameter_groups)
        if len(normalized_groups) != parameter_count:
            raise ValueError("parameter_groups must align with parameter tensors")
    bypass = _bypass_if_at_most_one_objective_is_active(
        originals,
        reduce_scalar=reduce_scalar,
        epsilon=epsilon,
    )
    if bypass is not None:
        return bypass
    grouped_indices: dict[object, list[int]] = {}
    for index, group in enumerate(normalized_groups):
        try:
            grouped_indices.setdefault(group, []).append(index)
        except TypeError as error:
            raise TypeError("parameter group IDs must be hashable") from error

    correctness, length = originals
    projected_length = [tensor.clone() for tensor in length]
    groups = list(grouped_indices.values())
    local_dots = [
        _local_dot(
            [length[index] for index in indices],
            [correctness[index] for index in indices],
        )
        for indices in groups
    ]
    local_correctness_norms = [
        _local_dot(
            [correctness[index] for index in indices],
            [correctness[index] for index in indices],
        )
        for indices in groups
    ]
    reduced_values = reduce_scalar(torch.stack([*local_dots, *local_correctness_norms]))
    group_count = len(groups)
    dots = reduced_values[:group_count]
    correctness_squared_norms = reduced_values[group_count:]
    local_conflicts = 0
    projections = 0
    for indices, dot, correctness_squared_norm in zip(
        groups,
        dots,
        correctness_squared_norms,
        strict=True,
    ):
        conflict = float(dot.detach().cpu()) < 0.0
        has_primary_norm = float(correctness_squared_norm.detach().cpu()) > epsilon
        local_conflicts += int(conflict)
        if not conflict or not has_primary_norm:
            continue
        coefficient = dot / correctness_squared_norm.clamp_min(epsilon)
        for parameter_index in indices:
            projected_length[parameter_index].sub_(
                correctness[parameter_index]
                * coefficient.to(correctness[parameter_index])
            )
        projections += 1

    combined_before = _sum(originals)
    combined_after = [
        correctness_tensor + length_tensor
        for correctness_tensor, length_tensor in zip(
            correctness,
            projected_length,
            strict=True,
        )
    ]
    diagnostics = _diagnostics(
        originals,
        combined_before,
        combined_after,
        reduce_scalar=reduce_scalar,
        epsilon=epsilon,
        projections=projections,
        projection_opportunities=group_count,
        local_conflicts=local_conflicts,
        local_pair_count=group_count,
        local_projections=projections,
        local_projection_opportunities=group_count,
    )
    return combined_after, diagnostics


def _bounded_golden_section_minimize(
    objective: Callable[[float], float],
    *,
    max_iterations: int,
) -> float:
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    left = 0.0
    right = 1.0
    inverse_phi = (math.sqrt(5.0) - 1.0) / 2.0
    middle_left = right - inverse_phi * (right - left)
    middle_right = left + inverse_phi * (right - left)
    value_left = objective(middle_left)
    value_right = objective(middle_right)
    for _ in range(max_iterations):
        if value_left <= value_right:
            right = middle_right
            middle_right = middle_left
            value_right = value_left
            middle_left = right - inverse_phi * (right - left)
            value_left = objective(middle_left)
        else:
            left = middle_left
            middle_left = middle_right
            value_left = value_right
            middle_right = left + inverse_phi * (right - left)
            value_right = objective(middle_right)
    return (left + right) / 2.0


def cagrad_reconcile(
    gradients: GradientCollection,
    *,
    conflict_aversion: float = 0.5,
    reduce_scalar: ScalarReducer = _identity,
    epsilon: float = 1e-12,
    stability_epsilon: float = 1e-4,
    max_iterations: int = 64,
) -> tuple[list[Tensor], ReconciliationDiagnostics]:
    """Apply official two-objective CAGrad and preserve Sum's gradient scale."""

    if len(gradients.objectives) != 2:
        raise ValueError("Stage A CAGrad requires exactly two objectives")
    if not 0.0 <= conflict_aversion < 1.0:
        raise ValueError("conflict_aversion must be in [0, 1)")
    if stability_epsilon <= 0.0:
        raise ValueError("stability_epsilon must be positive")

    originals = [
        [tensor.detach().clone() for tensor in objective]
        for objective in gradients.objectives
    ]
    bypass = _bypass_if_at_most_one_objective_is_active(
        originals,
        reduce_scalar=reduce_scalar,
        epsilon=epsilon,
    )
    if bypass is not None:
        return bypass
    first, second = originals
    gram = torch.stack(
        [
            _dot(first, first, reduce_scalar=reduce_scalar),
            _dot(first, second, reduce_scalar=reduce_scalar),
            _dot(second, second, reduce_scalar=reduce_scalar),
        ]
    ).detach().float().cpu()
    first_squared, cross, second_squared = (
        float(value) for value in gram.unbind()
    )
    mean_squared_norm = (
        first_squared + second_squared + 2.0 * cross + stability_epsilon
    )
    mean_norm = 0.5 * math.sqrt(max(mean_squared_norm, 0.0))
    coefficient = conflict_aversion * mean_norm
    quadratic = first_squared + second_squared - 2.0 * cross
    linear = cross - second_squared

    def objective(weight: float) -> float:
        weighted_squared_norm = (
            weight * weight * quadratic
            + 2.0 * weight * linear
            + second_squared
            + stability_epsilon
        )
        return (
            coefficient * math.sqrt(max(weighted_squared_norm, 0.0))
            + 0.5 * weight * quadratic
            + (0.5 + weight) * linear
            + second_squared
        )

    weight = _bounded_golden_section_minimize(
        objective,
        max_iterations=max_iterations,
    )
    weighted_squared_norm = (
        weight * weight * first_squared
        + (1.0 - weight) * (1.0 - weight) * second_squared
        + 2.0 * weight * (1.0 - weight) * cross
        + stability_epsilon
    )
    lagrange_multiplier = coefficient / (
        math.sqrt(max(weighted_squared_norm, 0.0)) + stability_epsilon
    )

    # The official implementation returns the mean-objective scale. Multiplying
    # by K=2 makes c=0 exactly equal to the literal OW-GRPO-Sum direction.
    output_scale = 2.0 / (1.0 + conflict_aversion)
    combined_after = [
        (
            0.5 * (first_tensor + second_tensor)
            + lagrange_multiplier
            * (weight * first_tensor + (1.0 - weight) * second_tensor)
        )
        * output_scale
        for first_tensor, second_tensor in zip(first, second, strict=True)
    ]
    combined_before = _sum(originals)
    diagnostics = _diagnostics(
        originals,
        combined_before,
        combined_after,
        reduce_scalar=reduce_scalar,
        epsilon=epsilon,
        projections=0,
    )
    return combined_after, diagnostics
