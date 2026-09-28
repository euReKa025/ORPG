from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import numpy as np
import torch
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import (
    compute_gdpo_outcome_advantage,
    compute_grpo_outcome_advantage,
)

from cw_grpo.adapters.verl_gd2po import _component_reward_tensors, _query_keep_ratio

_ROLLOUT_ADVANTAGES_KEY = "objective_wise_rollout_advantages"
_FILTER_CONFLICT_KEY = "objective_wise_filter_conflict_mask"
_FILTER_PRIMARY_KEEP_KEY = "objective_wise_filter_primary_keep_mask"
_FILTER_SECONDARY_KEEP_KEY = "objective_wise_filter_secondary_keep_mask"
_FILTER_QUERY_RATIO_KEY = "objective_wise_filter_query_keep_ratio"
_RAW_GROUP_STD_KEY = "objective_wise_raw_group_std"
_RAW_GROUP_RANGE_KEY = "objective_wise_raw_group_range"
_COMPONENT_ADVANTAGE_RMS_KEY = "objective_wise_component_advantage_rms"
_COMPETENCE_GROUP_CLASS_KEY = "objective_wise_competence_group_class"
_COMPETENCE_SECONDARY_ACTIVE_KEY = "objective_wise_competence_secondary_active"
_COMPETENCE_CORRECT_FRACTION_KEY = "objective_wise_competence_correct_fraction"
_COMPETENCE_ALL_WRONG = 0
_COMPETENCE_MIXED = 1
_COMPETENCE_ALL_CORRECT = 2
_PROBE_ALPHAS: tuple[tuple[str, float], ...] = (
    ("1", 1.0),
    ("1e-1", 1e-1),
    ("1e-2", 1e-2),
    ("1e-3", 1e-3),
    ("0", 0.0),
)
_PROBE_CONSTRUCTIONS: tuple[tuple[str, str, str], ...] = (
    ("independent_z", "independent_z", "none"),
    ("shared_raw_none", "shared_raw", "none"),
    ("shared_raw_exact_zero", "shared_raw", "exact_zero"),
)
OBJECTIVE_WISE_PROBE_VARIANT_NAMES: tuple[str, ...] = tuple(
    f"{label}_alpha_{alpha_label}"
    for label, _, _ in _PROBE_CONSTRUCTIONS
    for alpha_label, _ in _PROBE_ALPHAS
) + (
    "correctness_only_shared_raw_exact_zero",
    "scalar_grpo_total_reward",
)


def _raw_group_statistics(
    component_scores: list[torch.Tensor],
    group_ids: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-objective group std/range and exact-zero rollout masks."""

    if not component_scores:
        raise ValueError("objective-wise rewards cannot be empty")
    raw_rewards = torch.stack(
        [component_score.sum(dim=-1).float() for component_score in component_scores]
    )
    if raw_rewards.ndim != 2:
        raise ValueError("objective-wise raw rewards must have shape [K, B]")
    group_ids = np.asarray(group_ids)
    if group_ids.shape != (raw_rewards.shape[1],):
        raise ValueError("group IDs must align with objective-wise rewards")

    _, inverse = np.unique(group_ids, return_inverse=True)
    group_index = torch.as_tensor(
        inverse,
        device=raw_rewards.device,
        dtype=torch.long,
    )
    objective_count, batch_size = raw_rewards.shape
    group_count = int(group_index.max().item()) + 1
    expanded_index = group_index.unsqueeze(0).expand(objective_count, batch_size)
    counts = torch.zeros(
        group_count,
        device=raw_rewards.device,
        dtype=torch.float32,
    )
    counts.scatter_add_(0, group_index, torch.ones_like(group_index, dtype=torch.float32))
    sums = torch.zeros(
        objective_count,
        group_count,
        device=raw_rewards.device,
        dtype=torch.float32,
    )
    squared_sums = torch.zeros_like(sums)
    sums.scatter_add_(1, expanded_index, raw_rewards)
    means = sums / counts.unsqueeze(0)
    centered = raw_rewards - means.gather(1, expanded_index)
    squared_sums.scatter_add_(1, expanded_index, centered.square())
    variances = squared_sums / counts.unsqueeze(0)
    group_std = variances.sqrt()

    group_min = torch.full_like(sums, torch.inf)
    group_max = torch.full_like(sums, -torch.inf)
    group_min.scatter_reduce_(
        1,
        expanded_index,
        raw_rewards,
        reduce="amin",
        include_self=True,
    )
    group_max.scatter_reduce_(
        1,
        expanded_index,
        raw_rewards,
        reduce="amax",
        include_self=True,
    )
    group_range = group_max - group_min
    rollout_group_std = group_std.gather(1, expanded_index)
    rollout_group_range = group_range.gather(1, expanded_index)
    exact_zero_rollouts = rollout_group_range == 0.0
    return rollout_group_std, rollout_group_range, exact_zero_rollouts


def _scale_group_variation(
    component_score: torch.Tensor,
    *,
    response_mask: torch.Tensor,
    group_ids: np.ndarray,
    alpha: float,
) -> torch.Tensor:
    """Scale only within-group reward variation while preserving group means."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("probe alpha must be in [0, 1]")
    raw_reward = component_score.sum(dim=-1).float()
    group_ids = np.asarray(group_ids)
    if group_ids.shape != (raw_reward.shape[0],):
        raise ValueError("group IDs must align with component scores")
    _, inverse = np.unique(group_ids, return_inverse=True)
    group_index = torch.as_tensor(
        inverse,
        device=raw_reward.device,
        dtype=torch.long,
    )
    group_count = int(group_index.max().item()) + 1
    counts = torch.zeros(group_count, device=raw_reward.device, dtype=torch.float32)
    sums = torch.zeros_like(counts)
    counts.scatter_add_(0, group_index, torch.ones_like(raw_reward))
    sums.scatter_add_(0, group_index, raw_reward)
    group_mean = (sums / counts).gather(0, group_index)
    transformed_reward = group_mean + alpha * (raw_reward - group_mean)

    response_lengths = response_mask.sum(dim=-1).long()
    if torch.any(response_lengths <= 0):
        raise ValueError("gradient probe requires at least one response token")
    terminal_index = response_lengths - 1
    transformed = torch.zeros_like(component_score)
    transformed.scatter_(
        1,
        terminal_index.unsqueeze(-1),
        transformed_reward.to(component_score.dtype).unsqueeze(-1),
    )
    return transformed


def _correct_subset_centered_secondary_advantage(
    component_scores: list[torch.Tensor],
    *,
    response_mask: torch.Tensor,
    group_ids: np.ndarray,
    non_tensor_batch: MutableMapping[str, Any],
) -> torch.Tensor:
    """Route binary length learning through the group's demonstrated competence.

    Rewards remain independent. This construction changes only the secondary
    advantage: all-wrong groups contribute no length gradient; mixed groups
    center length reward over correct rollouts and zero incorrect rollouts; and
    all-correct groups reduce to ordinary group-centered length learning.
    """

    if len(component_scores) != 2:
        raise ValueError("competence routing requires correctness and length rewards")
    correctness = component_scores[0].sum(dim=-1).float()
    length = component_scores[1].sum(dim=-1).float()
    if not bool(torch.all((correctness == 0.0) | (correctness == 1.0)).item()):
        raise ValueError("competence routing requires binary correctness rewards")

    group_ids = np.asarray(group_ids)
    if group_ids.shape != (correctness.shape[0],):
        raise ValueError("group IDs must align with competence-routed rewards")
    _, inverse = np.unique(group_ids, return_inverse=True)
    group_index = torch.as_tensor(inverse, device=correctness.device, dtype=torch.long)
    group_count = int(group_index.max().item()) + 1
    routed = torch.zeros_like(length)
    group_class = torch.empty(group_count, device=correctness.device, dtype=torch.int8)
    secondary_active = torch.zeros(
        group_count,
        device=correctness.device,
        dtype=torch.bool,
    )
    correct_fraction = torch.empty(
        group_count,
        device=correctness.device,
        dtype=torch.float32,
    )

    for current_group in range(group_count):
        rollout_indices = torch.nonzero(
            group_index == current_group,
            as_tuple=False,
        ).squeeze(-1)
        correct = correctness[rollout_indices] == 1.0
        correct_count = int(correct.sum().item())
        rollout_count = int(rollout_indices.numel())
        correct_fraction[current_group] = correct_count / rollout_count
        if correct_count == 0:
            group_class[current_group] = _COMPETENCE_ALL_WRONG
            continue
        group_class[current_group] = (
            _COMPETENCE_ALL_CORRECT
            if correct_count == rollout_count
            else _COMPETENCE_MIXED
        )
        correct_indices = rollout_indices[correct]
        correct_length = length[correct_indices]
        routed[correct_indices] = correct_length - correct_length.mean()
        secondary_active[current_group] = bool(
            correct_count > 1
            and (correct_length.max() - correct_length.min()).item() > 0.0
        )

    # DataProto non-tensor fields must align with the rollout batch.  Expand
    # group-level diagnostics back to each rollout while keeping the routed
    # advantage itself unchanged.  GRPO uses a fixed rollout count per group,
    # so means over these expanded values remain group-rate means.
    non_tensor_batch[_COMPETENCE_GROUP_CLASS_KEY] = (
        group_class.gather(0, group_index).detach().cpu().numpy()
    )
    non_tensor_batch[_COMPETENCE_SECONDARY_ACTIVE_KEY] = (
        secondary_active.gather(0, group_index).detach().cpu().numpy()
    )
    non_tensor_batch[_COMPETENCE_CORRECT_FRACTION_KEY] = (
        correct_fraction.gather(0, group_index).detach().cpu().numpy()
    )
    return routed.unsqueeze(-1) * response_mask


def _construct_matched_components(
    component_scores: list[torch.Tensor],
    *,
    weights: torch.Tensor,
    response_mask: torch.Tensor,
    group_ids: np.ndarray,
    epsilon: float,
    norm_adv_by_std_in_grpo: bool,
    config: Mapping[str, Any],
    scaling_mode: str,
    validity_mode: str,
    filtering_mode: str,
    non_tensor_batch: MutableMapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct matched token advantages under one O-layer contract."""

    if scaling_mode not in {
        "independent_z",
        "shared_raw",
        "primary_grpo_secondary_raw",
    }:
        raise ValueError(
            "objective_wise_advantage_scaling must be independent_z, shared_raw, "
            "or primary_grpo_secondary_raw"
        )
    if validity_mode not in {"none", "exact_zero"}:
        raise ValueError("objective_wise_validity_mode must be none or exact_zero")
    competence_routing = str(
        config.get("objective_wise_competence_routing", "none")
    )
    if competence_routing not in {"none", "correct_subset_centered"}:
        raise ValueError(
            "objective_wise_competence_routing must be none or "
            "correct_subset_centered"
        )
    if competence_routing != "none" and (
        scaling_mode != "primary_grpo_secondary_raw"
        or validity_mode != "exact_zero"
    ):
        raise ValueError(
            "competence routing requires primary_grpo_secondary_raw and exact_zero"
        )
    rollout_group_std, rollout_group_range, exact_zero_rollouts = (
        _raw_group_statistics(component_scores, group_ids)
    )
    component_advantages: list[torch.Tensor] = []
    for component_index, component_score in enumerate(component_scores):
        normalize_component = (
            norm_adv_by_std_in_grpo
            if scaling_mode == "independent_z"
            or (
                scaling_mode == "primary_grpo_secondary_raw"
                and component_index == 0
            )
            else False
        )
        advantage, _ = compute_grpo_outcome_advantage(
            token_level_rewards=component_score,
            response_mask=response_mask,
            index=group_ids,
            epsilon=epsilon,
            norm_adv_by_std_in_grpo=normalize_component,
            config=config,
        )
        if validity_mode == "exact_zero":
            advantage = advantage.masked_fill(
                exact_zero_rollouts[component_index].unsqueeze(-1),
                0.0,
            )
        component_advantages.append(advantage)

    if competence_routing == "correct_subset_centered":
        component_advantages[1] = _correct_subset_centered_secondary_advantage(
            component_scores,
            response_mask=response_mask,
            group_ids=group_ids,
            non_tensor_batch=non_tensor_batch,
        )

    stacked = torch.stack(component_advantages)
    weighted = stacked * weights.view(-1, 1, 1)
    if scaling_mode == "primary_grpo_secondary_raw":
        matched_components = weighted * response_mask.unsqueeze(0)
    else:
        weighted_total = weighted.sum(dim=0)
        total_variance = verl_F.masked_var(weighted_total, response_mask)
        inverse_scale = torch.rsqrt(total_variance + 1e-8)
        matched_components = torch.stack(
            [
                (component - verl_F.masked_mean(component, response_mask))
                * inverse_scale
                * response_mask
                for component in weighted
            ]
        )
    if validity_mode == "exact_zero":
        matched_components = matched_components.masked_fill(
            exact_zero_rollouts.unsqueeze(-1),
            0.0,
        )
    matched_components = apply_objective_wise_filtering(
        matched_components,
        response_mask=response_mask,
        group_ids=group_ids,
        mode=filtering_mode,
        sign_epsilon=float(config.get("objective_wise_filtering_sign_epsilon", 1e-8)),
        non_tensor_batch=non_tensor_batch,
    )
    return (
        matched_components,
        rollout_group_std,
        rollout_group_range,
        exact_zero_rollouts,
    )


def _rollout_component_advantages(
    matched_components: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    response_lengths = response_mask.sum(dim=-1).clamp_min(1.0)
    return (
        (matched_components * response_mask).sum(dim=-1) / response_lengths
    ).transpose(0, 1)


def _build_objective_probe_rollout_advantages(
    component_scores: list[torch.Tensor],
    *,
    weights: torch.Tensor,
    response_mask: torch.Tensor,
    group_ids: np.ndarray,
    epsilon: float,
    norm_adv_by_std_in_grpo: bool,
    config: Mapping[str, Any],
) -> torch.Tensor:
    """Build compact fixed-rollout O-layer variants with shape [B, V, K]."""

    if len(component_scores) != 2:
        raise ValueError("Stage B gradient probe requires exactly two objectives")
    variants: list[torch.Tensor] = []
    for _, scaling_mode, validity_mode in _PROBE_CONSTRUCTIONS:
        for _, alpha in _PROBE_ALPHAS:
            scaled_length = _scale_group_variation(
                component_scores[1],
                response_mask=response_mask,
                group_ids=group_ids,
                alpha=alpha,
            )
            matched, _, _, _ = _construct_matched_components(
                [component_scores[0], scaled_length],
                weights=weights,
                response_mask=response_mask,
                group_ids=group_ids,
                epsilon=epsilon,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=config,
                scaling_mode=scaling_mode,
                validity_mode=validity_mode,
                filtering_mode="none",
                non_tensor_batch={},
            )
            variants.append(_rollout_component_advantages(matched, response_mask))

    correctness_matched, _, _, _ = _construct_matched_components(
        component_scores,
        weights=weights,
        response_mask=response_mask,
        group_ids=group_ids,
        epsilon=epsilon,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
        scaling_mode="shared_raw",
        validity_mode="exact_zero",
        filtering_mode="none",
        non_tensor_batch={},
    )
    correctness_only = _rollout_component_advantages(
        correctness_matched,
        response_mask,
    )
    correctness_only[:, 1] = 0.0
    variants.append(correctness_only)

    total_reward = sum(
        (
            component_score * weight
            for component_score, weight in zip(
                component_scores,
                weights,
                strict=True,
            )
        ),
        start=torch.zeros_like(component_scores[0]),
    )
    scalar_advantage, _ = compute_grpo_outcome_advantage(
        token_level_rewards=total_reward,
        response_mask=response_mask,
        index=group_ids,
        epsilon=epsilon,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )
    scalar_rollout = _rollout_component_advantages(
        scalar_advantage.unsqueeze(0),
        response_mask,
    )
    variants.append(
        torch.stack(
            (scalar_rollout[:, 0], torch.zeros_like(scalar_rollout[:, 0])),
            dim=-1,
        )
    )
    if len(variants) != len(OBJECTIVE_WISE_PROBE_VARIANT_NAMES):
        raise RuntimeError("objective-wise probe registry differs from built variants")
    return torch.stack(variants, dim=1)


def objective_wise_metrics(
    non_tensor_batch: Mapping[str, Any],
) -> dict[str, float]:
    if _ROLLOUT_ADVANTAGES_KEY not in non_tensor_batch:
        return {}
    advantages = np.asarray(
        non_tensor_batch[_ROLLOUT_ADVANTAGES_KEY],
        dtype=np.float32,
    )
    if advantages.ndim != 2 or advantages.shape[1] == 0:
        raise ValueError("objective-wise rollout advantages must have shape [B, K]")
    sign_epsilon = 1e-8
    mixed_sign = (advantages > sign_epsilon).any(axis=1) & (
        advantages < -sign_epsilon
    ).any(axis=1)
    nonzero = np.abs(advantages) > sign_epsilon
    metrics = {
        "objective_wise/mixed_sign_advantage_rate": float(mixed_sign.mean()),
        "objective_wise/nonzero_advantage_rate": float(nonzero.mean()),
        "objective_wise/objective_count": float(advantages.shape[1]),
    }
    validity_keys = (
        _RAW_GROUP_STD_KEY,
        _RAW_GROUP_RANGE_KEY,
        _COMPONENT_ADVANTAGE_RMS_KEY,
    )
    if all(key in non_tensor_batch for key in validity_keys):
        group_std = np.asarray(non_tensor_batch[_RAW_GROUP_STD_KEY], dtype=np.float32)
        group_range = np.asarray(
            non_tensor_batch[_RAW_GROUP_RANGE_KEY],
            dtype=np.float32,
        )
        component_rms = np.asarray(
            non_tensor_batch[_COMPONENT_ADVANTAGE_RMS_KEY],
            dtype=np.float32,
        )
        objective_count = advantages.shape[1]
        if not (
            group_std.ndim == 2
            and group_std.shape == advantages.shape
            and group_range.shape == group_std.shape
            and component_rms.shape == advantages.shape
        ):
            raise ValueError("objective-wise validity diagnostics must align")
        for objective_index in range(objective_count):
            prefix = f"objective_wise/objective_{objective_index}"
            objective_std = group_std[:, objective_index]
            objective_range = group_range[:, objective_index]
            zero_variance = objective_range == 0.0
            metrics.update(
                {
                    f"{prefix}/raw_group_std_mean": float(objective_std.mean()),
                    f"{prefix}/raw_group_std_max": float(objective_std.max()),
                    f"{prefix}/raw_group_range_mean": float(objective_range.mean()),
                    f"{prefix}/raw_group_range_max": float(objective_range.max()),
                    f"{prefix}/zero_variance_group_rate": float(
                        zero_variance.mean()
                    ),
                    f"{prefix}/active_group_rate": float((~zero_variance).mean()),
                    f"{prefix}/component_advantage_rms": float(
                        component_rms[:, objective_index].mean()
                    ),
                }
            )
    competence_keys = (
        _COMPETENCE_GROUP_CLASS_KEY,
        _COMPETENCE_SECONDARY_ACTIVE_KEY,
        _COMPETENCE_CORRECT_FRACTION_KEY,
    )
    if all(key in non_tensor_batch for key in competence_keys):
        group_class = np.asarray(
            non_tensor_batch[_COMPETENCE_GROUP_CLASS_KEY],
            dtype=np.int8,
        )
        secondary_active = np.asarray(
            non_tensor_batch[_COMPETENCE_SECONDARY_ACTIVE_KEY],
            dtype=np.bool_,
        )
        correct_fraction = np.asarray(
            non_tensor_batch[_COMPETENCE_CORRECT_FRACTION_KEY],
            dtype=np.float32,
        )
        if not (
            group_class.ndim == 1
            and secondary_active.shape == group_class.shape
            and correct_fraction.shape == group_class.shape
            and group_class.size > 0
            and np.isin(
                group_class,
                (
                    _COMPETENCE_ALL_WRONG,
                    _COMPETENCE_MIXED,
                    _COMPETENCE_ALL_CORRECT,
                ),
            ).all()
        ):
            raise ValueError("competence-routing diagnostics must align by rollout")
        mixed_groups = group_class == _COMPETENCE_MIXED
        all_correct_groups = group_class == _COMPETENCE_ALL_CORRECT
        mixed_secondary_active_rate = (
            float(secondary_active[mixed_groups].mean())
            if mixed_groups.any()
            else 0.0
        )
        all_correct_secondary_active_rate = (
            float(secondary_active[all_correct_groups].mean())
            if all_correct_groups.any()
            else 0.0
        )
        metrics.update(
            {
                "objective_wise/competence_routing/all_wrong_group_rate": float(
                    (group_class == _COMPETENCE_ALL_WRONG).mean()
                ),
                "objective_wise/competence_routing/mixed_group_rate": float(
                    (group_class == _COMPETENCE_MIXED).mean()
                ),
                "objective_wise/competence_routing/all_correct_group_rate": float(
                    (group_class == _COMPETENCE_ALL_CORRECT).mean()
                ),
                "objective_wise/competence_routing/secondary_active_group_rate": float(
                    secondary_active.mean()
                ),
                "objective_wise/competence_routing/mixed_secondary_active_rate": (
                    mixed_secondary_active_rate
                ),
                "objective_wise/competence_routing/all_correct_secondary_active_rate": (
                    all_correct_secondary_active_rate
                ),
                "objective_wise/competence_routing/correct_fraction_mean": float(
                    correct_fraction.mean()
                ),
            }
        )
    filter_keys = (
        _FILTER_CONFLICT_KEY,
        _FILTER_PRIMARY_KEEP_KEY,
        _FILTER_SECONDARY_KEEP_KEY,
        _FILTER_QUERY_RATIO_KEY,
    )
    if all(key in non_tensor_batch for key in filter_keys):
        conflict = np.asarray(non_tensor_batch[_FILTER_CONFLICT_KEY], dtype=np.bool_)
        primary_keep = np.asarray(
            non_tensor_batch[_FILTER_PRIMARY_KEEP_KEY], dtype=np.bool_
        )
        secondary_keep = np.asarray(
            non_tensor_batch[_FILTER_SECONDARY_KEEP_KEY], dtype=np.bool_
        )
        query_ratio = np.asarray(
            non_tensor_batch[_FILTER_QUERY_RATIO_KEY], dtype=np.float32
        )
        if not (
            conflict.ndim == 1
            and primary_keep.shape == conflict.shape
            and secondary_keep.shape == conflict.shape
            and query_ratio.shape == conflict.shape
        ):
            raise ValueError("objective-wise filtering diagnostics must align")
        metrics.update(
            {
                "objective_wise/filter_conflict_rate": float(conflict.mean()),
                "objective_wise/filter_primary_keep_rate": float(
                    primary_keep.mean()
                ),
                "objective_wise/filter_secondary_keep_rate": float(
                    secondary_keep.mean()
                ),
                "objective_wise/filter_query_keep_ratio_max": float(
                    query_ratio.max()
                ),
                "objective_wise/filter_query_keep_ratio_mean": float(
                    query_ratio.mean()
                ),
                "objective_wise/filter_query_keep_ratio_min": float(
                    query_ratio.min()
                ),
                "objective_wise/filter_query_keep_ratio_std": float(
                    query_ratio.std()
                ),
            }
        )
    return metrics


def apply_objective_wise_filtering(
    matched_components: torch.Tensor,
    *,
    response_mask: torch.Tensor,
    group_ids: np.ndarray,
    mode: str,
    non_tensor_batch: MutableMapping[str, Any],
    sign_epsilon: float = 1e-8,
) -> torch.Tensor:
    """Filter conflicting responses while preserving explicit objective semantics.

    ``symmetric`` applies the GD2PO-style response keep mask and query keep
    ratio to every objective. ``primary_preserving`` leaves objective 0
    (correctness) untouched and applies the same mask only to later objectives.
    """

    if mode not in {"none", "symmetric", "primary_preserving"}:
        raise ValueError(f"unsupported objective-wise filtering mode: {mode}")
    if matched_components.ndim != 3:
        raise ValueError("matched_components must have shape [K, B, R]")
    if matched_components.shape[0] != 2:
        raise ValueError("Stage A filtering requires exactly two objectives")
    if response_mask.shape != matched_components.shape[1:]:
        raise ValueError("response_mask must align with matched_components")
    if sign_epsilon < 0.0:
        raise ValueError("filtering sign epsilon must be non-negative")

    response_lengths = response_mask.sum(dim=-1).clamp_min(1.0)
    rollout_advantages = (
        matched_components * response_mask.unsqueeze(0)
    ).sum(dim=-1) / response_lengths.unsqueeze(0)
    positive = rollout_advantages > sign_epsilon
    negative = rollout_advantages < -sign_epsilon
    conflict = positive.any(dim=0) & negative.any(dim=0)
    secondary_keep = ~conflict if mode != "none" else torch.ones_like(conflict)
    primary_keep = (
        secondary_keep
        if mode == "symmetric"
        else torch.ones_like(secondary_keep)
    )
    query_ratio = (
        _query_keep_ratio(secondary_keep, np.asarray(group_ids))
        if mode != "none"
        else torch.ones_like(secondary_keep, dtype=torch.float32)
    )

    if mode == "none":
        filtered = matched_components
    else:
        secondary_scale = (
            secondary_keep.to(matched_components.dtype)
            * query_ratio.to(matched_components.dtype)
        )
        primary_scale = (
            secondary_scale
            if mode == "symmetric"
            else torch.ones_like(secondary_scale)
        )
        scales = torch.stack((primary_scale, secondary_scale)).unsqueeze(-1)
        filtered = matched_components * scales

    non_tensor_batch[_FILTER_CONFLICT_KEY] = conflict.detach().cpu().numpy()
    non_tensor_batch[_FILTER_PRIMARY_KEEP_KEY] = (
        primary_keep.detach().cpu().numpy()
    )
    non_tensor_batch[_FILTER_SECONDARY_KEEP_KEY] = (
        secondary_keep.detach().cpu().numpy()
    )
    non_tensor_batch[_FILTER_QUERY_RATIO_KEY] = query_ratio.detach().cpu().numpy()
    return filtered


def install_objective_wise_overlay() -> None:
    """Install the project advantage estimator and diagnostics in the driver."""

    from verl.trainer.ppo import core_algos, ray_trainer

    current_estimator = core_algos.ADV_ESTIMATOR_REGISTRY.get("gdpo")
    current_metrics = ray_trainer.compute_data_metrics
    if current_estimator is compute_objective_wise_outcome_advantage and getattr(
        current_metrics,
        "_cw_grpo_objective_wise",
        False,
    ):
        return
    if current_estimator is not compute_gdpo_outcome_advantage:
        raise RuntimeError(
            "refusing to replace an unexpected process-local GDPO estimator"
        )

    def compute_data_metrics_with_objective_wise(
        *,
        batch: Any,
        use_critic: bool = True,
    ) -> dict[str, Any]:
        metrics = current_metrics(batch=batch, use_critic=use_critic)
        metrics.update(objective_wise_metrics(batch.non_tensor_batch))
        return metrics

    compute_data_metrics_with_objective_wise._cw_grpo_objective_wise = True  # type: ignore[attr-defined]
    core_algos.ADV_ESTIMATOR_REGISTRY["gdpo"] = (
        compute_objective_wise_outcome_advantage
    )
    ray_trainer.compute_data_metrics = compute_data_metrics_with_objective_wise


@torch.no_grad()
def compute_objective_wise_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Mapping[str, Any] | None = None,
    non_tensor_batch: MutableMapping[str, Any] | None = None,
    batch: MutableMapping[str, torch.Tensor] | None = None,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build matched reward-specific GRPO advantages for independent losses."""

    if config is None or non_tensor_batch is None or batch is None:
        raise ValueError(
            "objective-wise GRPO requires config, non_tensor_batch, and batch"
        )
    component_scores, weights = _component_reward_tensors(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        config=config,
        non_tensor_batch=non_tensor_batch,
        batch=batch,
    )
    scaling_mode = str(
        config.get("objective_wise_advantage_scaling", "independent_z")
    )
    validity_mode = str(config.get("objective_wise_validity_mode", "none"))
    group_ids = np.asarray(index)
    (
        matched_components,
        rollout_group_std,
        rollout_group_range,
        _,
    ) = _construct_matched_components(
        component_scores,
        weights=weights,
        response_mask=response_mask,
        group_ids=group_ids,
        epsilon=epsilon,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
        scaling_mode=scaling_mode,
        validity_mode=validity_mode,
        filtering_mode=str(config.get("objective_wise_filtering_mode", "none")),
        non_tensor_batch=non_tensor_batch,
    )
    component_rms = torch.stack(
        [
            verl_F.masked_mean(component.square(), response_mask).sqrt()
            for component in matched_components
        ]
    )
    advantages = matched_components.sum(dim=0) * response_mask

    batch["objective_advantages"] = matched_components.permute(1, 0, 2)
    rollout_advantages = _rollout_component_advantages(
        matched_components,
        response_mask,
    )
    if bool(config.get("objective_wise_gradient_probe", False)):
        if str(config.get("objective_wise_filtering_mode", "none")) != "none":
            raise ValueError("objective-wise gradient probe requires filtering=none")
        batch["objective_probe_rollout_advantages"] = (
            _build_objective_probe_rollout_advantages(
                component_scores,
                weights=weights,
                response_mask=response_mask,
                group_ids=group_ids,
                epsilon=epsilon,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=config,
            )
        )
    non_tensor_batch[_ROLLOUT_ADVANTAGES_KEY] = (
        rollout_advantages.detach().cpu().numpy()
    )
    non_tensor_batch[_RAW_GROUP_STD_KEY] = (
        rollout_group_std.transpose(0, 1).detach().cpu().numpy()
    )
    non_tensor_batch[_RAW_GROUP_RANGE_KEY] = (
        rollout_group_range.transpose(0, 1).detach().cpu().numpy()
    )
    non_tensor_batch[_COMPONENT_ADVANTAGE_RMS_KEY] = (
        component_rms.unsqueeze(0)
        .expand(response_mask.shape[0], -1)
        .detach()
        .cpu()
        .numpy()
    )
    return advantages, advantages
