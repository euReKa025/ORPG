from __future__ import annotations

import torch
from torch import Tensor


def compute_mechanism_metrics(
    ratio: Tensor,
    component_advantages: Tensor,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
    eps: float = 1e-8,
) -> dict[str, Tensor]:
    """Compute diagnostics for the OW-GRPO-Sum surrogate mechanism."""
    if ratio.ndim != 3:
        raise ValueError("ratio must have shape [B, G, T]")
    if component_advantages.ndim != 3:
        raise ValueError("component_advantages must have shape [B, G, K]")
    if ratio.shape[:2] != component_advantages.shape[:2]:
        raise ValueError("ratio and component_advantages must agree on [B, G]")

    positive = component_advantages > 0
    negative = component_advantages < 0
    mixed_response = positive.any(dim=-1) & negative.any(dim=-1)

    abs_sum = component_advantages.abs().sum(dim=-1)
    signed_abs = component_advantages.sum(dim=-1).abs()
    cancellation = 1.0 - signed_abs / (abs_sum + eps)

    clip_low_mask = ratio < (1.0 - clip_low)
    clip_high_mask = ratio > (1.0 + clip_high)
    clipped_token = clip_low_mask | clip_high_mask

    mixed_token = mixed_response.unsqueeze(-1).expand_as(clipped_token)
    active = clipped_token & mixed_token

    positive_mass = component_advantages.clamp_min(0).sum(dim=-1)
    negative_mass = (-component_advantages.clamp_max(0)).sum(dim=-1)
    conflict_mass = torch.minimum(positive_mass, negative_mass)

    distance = torch.where(
        ratio < (1.0 - clip_low),
        (1.0 - clip_low) - ratio,
        torch.where(ratio > (1.0 + clip_high), ratio - (1.0 + clip_high), 0.0),
    )
    predicted_gap_magnitude = distance * conflict_mass.unsqueeze(-1)

    return {
        "mixed_sign_response_rate": mixed_response.float().mean(),
        "cancellation_index_mean": cancellation.mean(),
        "clip_low_fraction": clip_low_mask.float().mean(),
        "clip_high_fraction": clip_high_mask.float().mean(),
        "clip_fraction": clipped_token.float().mean(),
        "active_mechanism_rate": active.float().mean(),
        "predicted_gap_magnitude": predicted_gap_magnitude.mean(),
        "component_adv_positive_rate": positive.float().mean(),
        "component_adv_negative_rate": negative.float().mean(),
        "component_adv_zero_rate": (component_advantages == 0).float().mean(),
    }
