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

_DIAGNOSTIC_KEYS = (
    "gd2po_hard_conflict_mask",
    "gd2po_hard_keep_mask",
    "gd2po_hard_query_keep_ratio",
)


def _component_reward_tensors(
    *,
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    config: Mapping[str, Any],
    non_tensor_batch: Mapping[str, Any],
    batch: Mapping[str, torch.Tensor],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    reward_keys = config.get("gdpo_reward_keys")
    if not reward_keys:
        raise ValueError("GD2PO-Hard requires algorithm.gdpo_reward_keys")

    prompt_length = batch["prompts"].size(1)
    valid_response_end = batch["attention_mask"][:, prompt_length:].sum(dim=1) - 1
    if torch.any(valid_response_end < 0):
        raise ValueError("GD2PO-Hard cannot score an empty response")

    batch_size = response_mask.size(0)
    device = token_level_rewards.device
    component_scores: list[torch.Tensor] = []
    for key in reward_keys:
        if key not in non_tensor_batch:
            raise KeyError(
                f"GD2PO-Hard reward key {key!r} is missing; "
                f"available keys: {list(non_tensor_batch)}"
            )
        values = np.asarray(non_tensor_batch[key], dtype=np.float32)
        if values.shape != (batch_size,):
            raise ValueError(
                f"GD2PO-Hard reward key {key!r} must have shape "
                f"({batch_size},), got {values.shape}"
            )
        terminal_rewards = torch.zeros_like(response_mask, dtype=torch.float32)
        terminal_rewards[
            torch.arange(batch_size, device=device),
            valid_response_end.to(device=device),
        ] = torch.as_tensor(values, device=device)
        component_scores.append(terminal_rewards)

    configured_weights = config.get("gdpo_reward_weights")
    if configured_weights is None:
        weights = torch.ones(len(component_scores), dtype=torch.float32, device=device)
    else:
        if len(configured_weights) != len(component_scores):
            raise ValueError("gdpo_reward_weights must align with gdpo_reward_keys")
        weights = torch.as_tensor(
            list(configured_weights),
            dtype=torch.float32,
            device=device,
        )
    return component_scores, weights


def _query_keep_ratio(
    keep_mask: torch.Tensor,
    group_ids: np.ndarray,
) -> torch.Tensor:
    if len(group_ids) != keep_mask.numel():
        raise ValueError("group IDs must align with the rollout batch")
    ratios = torch.zeros_like(keep_mask, dtype=torch.float32)
    for group_id in np.unique(group_ids):
        members = torch.as_tensor(
            group_ids == group_id,
            device=keep_mask.device,
            dtype=torch.bool,
        )
        ratios[members] = keep_mask[members].to(torch.float32).mean()
    return ratios


def gd2po_hard_metrics(non_tensor_batch: Mapping[str, Any]) -> dict[str, float]:
    """Aggregate paper-facing filtering diagnostics without changing updates."""

    if not all(key in non_tensor_batch for key in _DIAGNOSTIC_KEYS):
        return {}
    conflict = np.asarray(
        non_tensor_batch["gd2po_hard_conflict_mask"],
        dtype=np.bool_,
    )
    keep = np.asarray(
        non_tensor_batch["gd2po_hard_keep_mask"],
        dtype=np.bool_,
    )
    query_ratio = np.asarray(
        non_tensor_batch["gd2po_hard_query_keep_ratio"],
        dtype=np.float32,
    )
    if conflict.ndim != 1 or keep.shape != conflict.shape or query_ratio.shape != conflict.shape:
        raise ValueError("GD2PO-Hard diagnostics must be aligned one-dimensional arrays")
    if conflict.size == 0:
        raise ValueError("GD2PO-Hard diagnostics cannot be empty")
    return {
        "gd2po/hard_conflict_rate": float(conflict.mean()),
        "gd2po/hard_keep_rate": float(keep.mean()),
        "gd2po/query_keep_ratio_max": float(query_ratio.max()),
        "gd2po/query_keep_ratio_mean": float(query_ratio.mean()),
        "gd2po/query_keep_ratio_min": float(query_ratio.min()),
        "gd2po/query_keep_ratio_std": float(query_ratio.std()),
    }


def install_gd2po_hard_overlay() -> None:
    """Install the project estimator and metrics in this driver process only."""

    from verl.trainer.ppo import core_algos, ray_trainer

    current_estimator = core_algos.ADV_ESTIMATOR_REGISTRY.get("gdpo")
    current_metrics = ray_trainer.compute_data_metrics
    if current_estimator is compute_gd2po_hard_outcome_advantage and getattr(
        current_metrics,
        "_orpg_gd2po_hard",
        False,
    ):
        return
    if current_estimator is not compute_gdpo_outcome_advantage:
        raise RuntimeError(
            "refusing to replace an unexpected process-local GDPO estimator"
        )

    def compute_data_metrics_with_gd2po_hard(
        *,
        batch: Any,
        use_critic: bool = True,
    ) -> dict[str, Any]:
        metrics = current_metrics(batch=batch, use_critic=use_critic)
        metrics.update(gd2po_hard_metrics(batch.non_tensor_batch))
        return metrics

    compute_data_metrics_with_gd2po_hard._orpg_gd2po_hard = True  # type: ignore[attr-defined]
    core_algos.ADV_ESTIMATOR_REGISTRY["gdpo"] = (
        compute_gd2po_hard_outcome_advantage
    )
    ray_trainer.compute_data_metrics = compute_data_metrics_with_gd2po_hard


@torch.no_grad()
def compute_gd2po_hard_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Mapping[str, Any] | None = None,
    non_tensor_batch: MutableMapping[str, Any] | None = None,
    batch: Mapping[str, torch.Tensor] | None = None,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Port the official GD2PO-Hard filter and query reweighting to pinned Verl."""

    if config is None or non_tensor_batch is None or batch is None:
        raise ValueError("GD2PO-Hard requires config, non_tensor_batch, and batch")

    component_scores, weights = _component_reward_tensors(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        config=config,
        non_tensor_batch=non_tensor_batch,
        batch=batch,
    )
    component_advantages: list[torch.Tensor] = []
    for component_score in component_scores:
        advantage, _ = compute_grpo_outcome_advantage(
            token_level_rewards=component_score,
            response_mask=response_mask,
            index=index,
            epsilon=epsilon,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )
        component_advantages.append(advantage)

    stacked = torch.stack(component_advantages)
    rollout_advantages = (stacked * response_mask).sum(dim=-1)
    sign_epsilon = float(config.get("gd2po_hard_sign_epsilon", 1e-8))
    positive = rollout_advantages > sign_epsilon
    negative = rollout_advantages < -sign_epsilon
    conflict_mask = positive.any(dim=0) & negative.any(dim=0)
    keep_mask = ~conflict_mask
    query_keep_ratio = _query_keep_ratio(keep_mask, np.asarray(index))

    weighted_advantage = torch.einsum("k,kbt->bt", weights, stacked)
    retain_mask = keep_mask.unsqueeze(-1).to(response_mask.dtype) * response_mask
    filtered_advantage = (
        weighted_advantage * query_keep_ratio.unsqueeze(-1) * retain_mask
    )
    if retain_mask.sum() > 0:
        advantages = verl_F.masked_whiten(filtered_advantage, retain_mask) * retain_mask
    else:
        advantages = torch.zeros_like(filtered_advantage)

    non_tensor_batch["gd2po_hard_conflict_mask"] = (
        conflict_mask.detach().cpu().numpy()
    )
    non_tensor_batch["gd2po_hard_keep_mask"] = keep_mask.detach().cpu().numpy()
    non_tensor_batch["gd2po_hard_query_keep_ratio"] = (
        query_keep_ratio.detach().cpu().numpy()
    )
    return advantages, advantages
