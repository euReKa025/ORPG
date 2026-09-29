from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class AdvantageBundle:
    """Matched component advantages and their exact summed total."""

    component_advantages: Tensor
    total_advantage: Tensor
    valid_component_mask: Tensor
    total_scale: Tensor


def _require_shape(tensor: Tensor, ndim: int, name: str) -> None:
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got shape {tuple(tensor.shape)}")


def masked_mean(values: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    """Mean over all valid entries; mask must be broadcastable to values."""
    mask = mask.to(dtype=values.dtype)
    weighted = values * mask
    denominator = mask.expand_as(values).sum().clamp_min(eps)
    return weighted.sum() / denominator


def compute_grpo_advantage(
    component_rewards: Tensor,
    reward_weights: Tensor,
    eps: float = 1e-8,
    std_threshold: float = 1e-6,
) -> Tensor:
    """Sum raw rewards, then normalize the scalar reward inside each prompt group."""
    _require_shape(component_rewards, 3, "component_rewards")
    _require_shape(reward_weights, 1, "reward_weights")
    if component_rewards.shape[-1] != reward_weights.shape[0]:
        raise ValueError("reward_weights length must match reward component count")

    scalar_reward = (component_rewards * reward_weights.view(1, 1, -1)).sum(dim=-1)
    mean = scalar_reward.mean(dim=1, keepdim=True)
    std = scalar_reward.std(dim=1, keepdim=True, unbiased=False)
    valid = std > std_threshold
    return torch.where(valid, (scalar_reward - mean) / (std + eps), torch.zeros_like(scalar_reward))


def compute_component_group_advantages(
    component_rewards: Tensor,
    eps: float = 1e-8,
    std_threshold: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Normalize each reward component independently inside each prompt group."""
    _require_shape(component_rewards, 3, "component_rewards")

    mean = component_rewards.mean(dim=1, keepdim=True)
    std = component_rewards.std(dim=1, keepdim=True, unbiased=False)
    valid = std > std_threshold
    normalized = torch.where(
        valid,
        (component_rewards - mean) / (std + eps),
        torch.zeros_like(component_rewards),
    )
    return normalized, valid


def match_component_scale(
    component_advantages: Tensor,
    reward_weights: Tensor,
    eps: float = 1e-8,
) -> AdvantageBundle:
    """Apply preference weights and match the exact total-advantage scale.

    The sum of the returned component advantages equals a batch-standardized
    weighted total advantage, up to floating-point error.
    """
    _require_shape(component_advantages, 3, "component_advantages")
    _require_shape(reward_weights, 1, "reward_weights")
    if component_advantages.shape[-1] != reward_weights.shape[0]:
        raise ValueError("reward_weights length must match reward component count")

    weighted = component_advantages * reward_weights.view(1, 1, -1)
    component_mean = weighted.mean(dim=(0, 1), keepdim=True)
    centered = weighted - component_mean

    total_pre_whiten = weighted.sum(dim=-1)
    total_scale = total_pre_whiten.std(unbiased=False).clamp_min(eps)

    matched = centered / total_scale
    total = matched.sum(dim=-1)
    valid = torch.ones(
        (component_advantages.shape[0], 1, component_advantages.shape[-1]),
        dtype=torch.bool,
        device=component_advantages.device,
    )
    return AdvantageBundle(
        component_advantages=matched,
        total_advantage=total,
        valid_component_mask=valid,
        total_scale=total_scale,
    )


def build_advantage_bundle(
    component_rewards: Tensor,
    reward_weights: Tensor,
    eps: float = 1e-8,
    std_threshold: float = 1e-6,
) -> AdvantageBundle:
    """Component group normalization followed by matched scaling."""
    component_adv, valid = compute_component_group_advantages(
        component_rewards,
        eps=eps,
        std_threshold=std_threshold,
    )
    bundle = match_component_scale(component_adv, reward_weights, eps=eps)
    bundle.valid_component_mask = valid
    return bundle


def _ratio_and_clipped_ratio(
    current_log_probs: Tensor,
    old_log_probs: Tensor,
    clip_low: float,
    clip_high: float,
) -> tuple[Tensor, Tensor]:
    _require_shape(current_log_probs, 3, "current_log_probs")
    _require_shape(old_log_probs, 3, "old_log_probs")
    if current_log_probs.shape != old_log_probs.shape:
        raise ValueError("current_log_probs and old_log_probs must have the same shape")

    ratio = torch.exp(current_log_probs - old_log_probs)
    clipped = torch.clamp(ratio, min=1.0 - clip_low, max=1.0 + clip_high)
    return ratio, clipped


def clipped_surrogate(ratio: Tensor, clipped_ratio: Tensor, advantage: Tensor) -> Tensor:
    return torch.minimum(ratio * advantage, clipped_ratio * advantage)


def grpo_policy_loss(
    current_log_probs: Tensor,
    old_log_probs: Tensor,
    response_mask: Tensor,
    total_advantage: Tensor,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
    eps: float = 1e-8,
) -> Tensor:
    """One scalar-advantage GRPO clipped policy loss."""
    _require_shape(total_advantage, 2, "total_advantage")
    ratio, clipped = _ratio_and_clipped_ratio(
        current_log_probs, old_log_probs, clip_low, clip_high
    )
    advantage_token = total_advantage.unsqueeze(-1)
    objective = clipped_surrogate(ratio, clipped, advantage_token)
    return -masked_mean(objective, response_mask, eps=eps)


def gdpo_policy_loss(
    current_log_probs: Tensor,
    old_log_probs: Tensor,
    response_mask: Tensor,
    component_advantages: Tensor,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
    eps: float = 1e-8,
) -> Tensor:
    """Sum matched component advantages, then apply one clipped surrogate."""
    _require_shape(component_advantages, 3, "component_advantages")
    total_advantage = component_advantages.sum(dim=-1)
    return grpo_policy_loss(
        current_log_probs=current_log_probs,
        old_log_probs=old_log_probs,
        response_mask=response_mask,
        total_advantage=total_advantage,
        clip_low=clip_low,
        clip_high=clip_high,
        eps=eps,
    )


def orpg_policy_loss(
    current_log_probs: Tensor,
    old_log_probs: Tensor,
    response_mask: Tensor,
    component_advantages: Tensor,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
    eps: float = 1e-8,
) -> Tensor:
    """Clip each component surrogate before summing components."""
    _require_shape(component_advantages, 3, "component_advantages")
    ratio, clipped = _ratio_and_clipped_ratio(
        current_log_probs, old_log_probs, clip_low, clip_high
    )

    ratio = ratio.unsqueeze(-1)
    clipped = clipped.unsqueeze(-1)
    advantage = component_advantages.unsqueeze(-2)

    component_objective = clipped_surrogate(ratio, clipped, advantage)
    objective = component_objective.sum(dim=-1)
    return -masked_mean(objective, response_mask, eps=eps)
