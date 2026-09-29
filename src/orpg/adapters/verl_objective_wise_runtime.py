from __future__ import annotations

import json
import os
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import is_dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from omegaconf import open_dict
from verl.single_controller.base.decorator import Dispatch, register
from verl.trainer.main_ppo import TaskRunner
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import need_reference_policy
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.utils.losses import ppo_loss

from orpg.adapters.verl_objective_wise import install_objective_wise_overlay
from orpg.adapters.verl_objective_wise_engine import (
    ObjectiveWiseEngineSettings,
    ObjectiveWiseFSDPEngineWithLMHead,
)

_OBJECTIVE_WISE_POLICY_KEYS = (
    "objective_wise_aggregator",
    "objective_wise_reward_count",
    "objective_wise_cagrad_conflict_aversion",
    "objective_wise_gradient_probe_only",
    "objective_wise_positive_rule",
    "objective_wise_positive_strength",
    "objective_wise_positive_q",
    "objective_wise_positive_preserve_norm",
)
_OBJECTIVE_WISE_TRANSPORT_KEY = "_orpg_objective_wise"


def make_policy_only_actor_config(actor_config: Any) -> Any:
    """Copy actor loss config and remove regularizers shared by all rewards."""

    if is_dataclass(actor_config):
        return replace(
            actor_config,
            use_kl_loss=False,
            entropy_coeff=0.0,
        )
    policy_only = deepcopy(actor_config)
    policy_only.use_kl_loss = False
    policy_only.entropy_coeff = 0.0
    return policy_only


def build_objective_wise_engine_settings(
    actor_config: Any,
    *,
    optimizer_config: Any,
    objective_wise_config: Mapping[str, Any],
) -> ObjectiveWiseEngineSettings:
    aggregator = str(objective_wise_config.get("objective_wise_aggregator", ""))
    objective_count = int(
        objective_wise_config.get("objective_wise_reward_count", 0)
    )
    conflict_aversion = float(
        objective_wise_config.get(
            "objective_wise_cagrad_conflict_aversion",
            0.5,
        )
    )
    return ObjectiveWiseEngineSettings(
        aggregator=aggregator,
        objective_count=objective_count,
        shared_regularizer=bool(
            actor_config.use_kl_loss or float(actor_config.entropy_coeff) != 0.0
        ),
        cagrad_conflict_aversion=conflict_aversion,
        final_clip_norm=float(optimizer_config.clip_grad),
        gradient_probe_only=bool(
            objective_wise_config.get("objective_wise_gradient_probe_only", False)
        ),
        positive_rule=str(objective_wise_config.get("objective_wise_positive_rule", "off")),
        positive_strength=float(objective_wise_config.get("objective_wise_positive_strength", 0.0)),
        positive_q=float(objective_wise_config.get("objective_wise_positive_q", 0.5)),
        positive_preserve_norm=objective_wise_config.get("objective_wise_positive_preserve_norm", True),
    )


_PROBE_TENSOR_KEYS = (
    "input_ids",
    "attention_mask",
    "position_ids",
    "responses",
    "response_mask",
    "old_log_probs",
    "ref_log_prob",
    "token_level_scores",
    "token_level_rewards",
    "advantages",
    "objective_advantages",
    "objective_probe_rollout_advantages",
)
_PROBE_NON_TENSOR_KEYS = (
    "uid",
    "correctness_reward",
    "length_reward",
    "response_token_count",
    "calibration_id",
    "length_threshold_low",
    "length_threshold_high",
)


def write_objective_probe_capture(
    batch: Any,
    *,
    capture_dir: str | Path,
    global_step: int,
    config_metadata: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Persist the exact pre-update driver batch needed for replay."""

    if global_step <= 0:
        raise ValueError("gradient probe capture requires a positive global step")
    directory = Path(capture_dir)
    directory.mkdir(parents=True, exist_ok=True)
    tensor_path = directory / f"step_{global_step:06d}.pt"
    metadata_path = directory / f"step_{global_step:06d}.json"
    if tensor_path.exists() or metadata_path.exists():
        raise FileExistsError(f"gradient probe capture already exists for step {global_step}")

    tensors = {
        key: batch.batch[key].detach().cpu().contiguous()
        for key in _PROBE_TENSOR_KEYS
        if key in batch.batch
    }
    required = {
        "input_ids",
        "attention_mask",
        "position_ids",
        "responses",
        "response_mask",
        "old_log_probs",
        "objective_probe_rollout_advantages",
    }
    missing = required.difference(tensors)
    if missing:
        raise KeyError(f"gradient probe capture missing tensors: {sorted(missing)}")
    non_tensors = {
        key: np.asarray(batch.non_tensor_batch[key]).copy()
        for key in _PROBE_NON_TENSOR_KEYS
        if key in batch.non_tensor_batch
    }
    if "uid" not in non_tensors:
        raise KeyError("gradient probe capture requires repeated rollout uid values")

    payload = {
        "schema_version": 1,
        "global_step": global_step,
        "tensors": tensors,
        "non_tensors": non_tensors,
    }
    metadata = {
        "schema_version": 1,
        "global_step": global_step,
        "tensor_file": tensor_path.name,
        "rollout_count": int(next(iter(tensors.values())).shape[0]),
        "tensor_shapes": {key: list(value.shape) for key, value in tensors.items()},
        "tensor_dtypes": {key: str(value.dtype) for key, value in tensors.items()},
        "non_tensor_keys": sorted(non_tensors),
        "config": dict(config_metadata),
    }
    tensor_partial = tensor_path.with_suffix(".pt.partial")
    metadata_partial = metadata_path.with_suffix(".json.partial")
    for partial_path in (tensor_partial, metadata_partial):
        if partial_path.exists():
            partial_path.unlink()
    torch.save(payload, tensor_partial)
    metadata_partial.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tensor_partial, tensor_path)
    os.replace(metadata_partial, metadata_path)
    return tensor_path, metadata_path


def pop_objective_wise_policy_settings(actor_config: Any) -> dict[str, Any]:
    """Extract project fields before Verl instantiates PolicyLossConfig."""

    extracted: dict[str, Any] = {}
    with open_dict(actor_config.policy_loss):
        for key in _OBJECTIVE_WISE_POLICY_KEYS:
            if key in actor_config.policy_loss:
                extracted[key] = actor_config.policy_loss.pop(key)
    missing = {
        "objective_wise_aggregator",
        "objective_wise_reward_count",
    }.difference(extracted)
    if missing:
        raise ValueError(
            f"objective-wise policy config is missing fields: {sorted(missing)}"
        )
    return extracted


def make_native_validation_config(config: Any) -> Any:
    """Copy an objective-wise config and remove project-only transport fields."""

    validation_config = deepcopy(config)
    policy_loss = validation_config.actor_rollout_ref.actor.policy_loss
    if any(key in policy_loss for key in _OBJECTIVE_WISE_POLICY_KEYS):
        pop_objective_wise_policy_settings(validation_config.actor_rollout_ref.actor)
    return validation_config


def stage_objective_wise_settings_for_worker(config: Any) -> dict[str, Any]:
    """Move project settings through a native dict until the worker consumes them."""

    actor_config = config.actor_rollout_ref.actor
    extracted = pop_objective_wise_policy_settings(actor_config)
    with open_dict(actor_config):
        if "global_batch_info" not in actor_config:
            actor_config.global_batch_info = {}
    with open_dict(actor_config.global_batch_info):
        if _OBJECTIVE_WISE_TRANSPORT_KEY in actor_config.global_batch_info:
            raise ValueError("objective-wise worker settings are already staged")
        actor_config.global_batch_info[_OBJECTIVE_WISE_TRANSPORT_KEY] = extracted
    return extracted


def pop_staged_objective_wise_settings(actor_config: Any) -> dict[str, Any]:
    """Consume and remove project settings before native ActorConfig construction."""

    with open_dict(actor_config.global_batch_info):
        if _OBJECTIVE_WISE_TRANSPORT_KEY not in actor_config.global_batch_info:
            raise ValueError("objective-wise worker settings were not staged")
        staged = actor_config.global_batch_info.pop(_OBJECTIVE_WISE_TRANSPORT_KEY)
    return dict(staged)


def install_objective_wise_engine_registry() -> None:
    """Select the project FSDP engine inside objective-wise worker processes."""

    from verl.workers.engine import EngineRegistry
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

    language_model_engines = EngineRegistry._engines["language_model"]
    for backend in ("fsdp", "fsdp2"):
        for device, current in language_model_engines[backend].items():
            if current not in {
                FSDPEngineWithLMHead,
                ObjectiveWiseFSDPEngineWithLMHead,
            }:
                raise RuntimeError(
                    "refusing to replace an unexpected language-model FSDP engine"
                )
            language_model_engines[backend][device] = (
                ObjectiveWiseFSDPEngineWithLMHead
            )


class ObjectiveWiseActorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self) -> None:
        install_objective_wise_engine_registry()
        objective_wise_config = pop_staged_objective_wise_settings(self.config.actor)
        super().init_model()
        if not self._is_actor:
            return
        engine = self.actor.engine
        if not isinstance(engine, ObjectiveWiseFSDPEngineWithLMHead):
            raise TypeError("objective-wise actor did not receive the project FSDP engine")
        full_loss_function = self.actor.loss_fn
        actor_config = full_loss_function.keywords["config"]
        policy_only_config = make_policy_only_actor_config(actor_config)
        engine.configure_objective_wise(
            settings=build_objective_wise_engine_settings(
                actor_config,
                optimizer_config=self.actor.optimizer_config,
                objective_wise_config=objective_wise_config,
            ),
            policy_loss_function=partial(ppo_loss, config=policy_only_config),
        )


class ObjectiveWiseRayPPOTrainer(RayPPOTrainer):
    """Capture a fixed pre-update batch before the probe reaches actor workers."""

    def fit(self):
        from orpg.exploration_budget import exploration_stop_at_step

        # TaskRunner has already initialized workers with the unchanged 100-step
        # config. Pinned Verl uses this driver field only for progress/last-step
        # save/termination in fit. Never shorten the optimizer's configuration.
        stop = exploration_stop_at_step(self)
        if stop is None:
            return super().fit()
        scheduled_steps = self.total_training_steps
        self._exploration_stop = stop
        self.total_training_steps = stop
        try:
            return super().fit()
        finally:
            self.total_training_steps = scheduled_steps
            del self._exploration_stop

    def _load_checkpoint(self):
        from orpg.exploration_budget import validate_exploration_resume_step

        result = super()._load_checkpoint()
        validate_exploration_resume_step(
            self.global_steps, getattr(self, '_exploration_stop', None)
        )
        return result

    def _update_actor(self, batch):
        if bool(self.config.algorithm.get("objective_wise_gradient_probe", False)):
            capture_dir = self.config.trainer.get(
                "objective_wise_probe_capture_dir",
                None,
            )
            if not capture_dir:
                raise ValueError(
                    "objective-wise gradient probe requires a capture directory"
                )
            write_objective_probe_capture(
                batch,
                capture_dir=str(capture_dir),
                global_step=int(self.global_steps),
                config_metadata={
                    "experiment_name": str(self.config.trainer.experiment_name),
                    "model_path": str(self.config.actor_rollout_ref.model.path),
                    "rollout_n": int(self.config.actor_rollout_ref.rollout.n),
                    "max_response_length": int(self.config.data.max_response_length),
                    "ppo_mini_batch_size": int(
                        self.config.actor_rollout_ref.actor.ppo_mini_batch_size
                    ),
                    "loss_agg_mode": str(
                        self.config.actor_rollout_ref.actor.loss_agg_mode
                    ),
                    "advantage_scaling": str(
                        self.config.algorithm.objective_wise_advantage_scaling
                    ),
                    "validity_mode": str(
                        self.config.algorithm.objective_wise_validity_mode
                    ),
                    "filtering_mode": str(
                        self.config.algorithm.objective_wise_filtering_mode
                    ),
                    "reward_keys": list(self.config.algorithm.gdpo_reward_keys),
                    "reward_weights": list(
                        self.config.algorithm.gdpo_reward_weights
                    ),
                },
            )
        return super()._update_actor(batch)


def install_objective_wise_trainer() -> None:
    """Select the project trainer without modifying the pinned Verl checkout."""

    from verl.trainer import main_ppo

    current = main_ppo.RayPPOTrainer
    if current not in {RayPPOTrainer, ObjectiveWiseRayPPOTrainer}:
        raise RuntimeError("refusing to replace an unexpected Verl RayPPOTrainer")
    main_ppo.RayPPOTrainer = ObjectiveWiseRayPPOTrainer


class ObjectiveWiseTaskRunner(TaskRunner):
    def add_actor_rollout_worker(self, config):
        if config.trainer.get("use_legacy_worker_impl", "auto") != "disable":
            raise ValueError("objective-wise training requires the new Verl worker")

        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = (
            lora_rank > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )
        if need_reference_policy(config) and not ref_in_actor:
            role = Role.ActorRolloutRef
        else:
            role = Role.ActorRollout
        self.role_worker_mapping[role] = ray.remote(
            ObjectiveWiseActorRolloutRefWorker
        )
        self.mapping[role] = "global_pool"
        return ObjectiveWiseActorRolloutRefWorker, RayWorkerGroup

    def run(self, config):
        install_objective_wise_overlay()
        install_objective_wise_trainer()
        stage_objective_wise_settings_for_worker(config)
        return super().run(config)
