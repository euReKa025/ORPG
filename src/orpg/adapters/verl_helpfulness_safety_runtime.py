from __future__ import annotations

import re
from typing import Any

import ray
from verl.experimental.reward_loop.reward_loop import (
    RewardLoopManager as VerlRewardLoopManager,
)
from verl.trainer.main_ppo import TaskRunner

from orpg.adapters.verl_objective_wise_runtime import ObjectiveWiseTaskRunner


def _actor_name(experiment_name: str, index: int) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_-]", "-", experiment_name)[:80]
    return f"hs-reward-{safe_name}-{index}"


class HelpfulnessSafetyRewardLoopManager(VerlRewardLoopManager):
    """Reserve GPU Ray actors for persistent, dynamically batched dual-RM scoring."""

    def __init__(self, config: Any, rm_resource_pool: Any = None) -> None:
        if bool(config.reward.reward_model.enable):
            raise ValueError("H/S uses the project dual-RM manager, not native single RM")
        if rm_resource_pool is not None:
            raise ValueError("H/S dual-RM actors must not receive a native RM pool")
        expected_workers = int(config.reward.helpfulness_safety_reward_gpus)
        if int(config.reward.num_workers) != expected_workers:
            raise ValueError("H/S reward.num_workers must equal reserved RM GPUs")
        super().__init__(config=config, rm_resource_pool=None)

    def _init_reward_loop_workers(self) -> None:
        self.reward_loop_workers = []
        num_workers = int(self.config.reward.num_workers)
        nodes = [
            node
            for node in ray.nodes()
            if node["Alive"] and node["Resources"].get("GPU", 0) > 0
        ]
        if not nodes:
            raise RuntimeError("H/S reward loop found no live GPU Ray node")
        experiment_name = str(self.config.trainer.experiment_name)
        for index in range(num_workers):
            node_id = nodes[index % len(nodes)]["NodeID"]
            strategy = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            )
            worker = self.reward_loop_workers_class.options(
                name=_actor_name(experiment_name, index),
                num_cpus=2,
                num_gpus=1,
                scheduling_strategy=strategy,
            ).remote(self.config, self.reward_router_address)
            self.reward_loop_workers.append(worker)


def install_helpfulness_safety_reward_loop() -> None:
    """Patch only the TaskRunner process before RayPPOTrainer initializes."""
    import verl.experimental.reward_loop as reward_loop_package

    current = reward_loop_package.RewardLoopManager
    if current is HelpfulnessSafetyRewardLoopManager:
        return
    if current is not VerlRewardLoopManager:
        raise RuntimeError("refusing to replace an unexpected RewardLoopManager")
    reward_loop_package.RewardLoopManager = HelpfulnessSafetyRewardLoopManager


class HelpfulnessSafetyTaskRunner(TaskRunner):
    def run(self, config):
        install_helpfulness_safety_reward_loop()
        return super().run(config)


class HelpfulnessSafetyGD2POTaskRunner(TaskRunner):
    def run(self, config):
        from orpg.adapters.verl_gd2po import install_gd2po_hard_overlay

        install_helpfulness_safety_reward_loop()
        install_gd2po_hard_overlay()
        return super().run(config)


class HelpfulnessSafetyObjectiveWiseTaskRunner(ObjectiveWiseTaskRunner):
    def run(self, config):
        install_helpfulness_safety_reward_loop()
        return super().run(config)
