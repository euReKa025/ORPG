from __future__ import annotations

import asyncio
from typing import Any

import torch
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

from cw_grpo.helpfulness_safety_service import (
    AsyncRewardBatcher,
    DualRewardScorer,
    append_assistant_response,
)
from cw_grpo.helpfulness_safety_training import (
    HelpfulnessSafetyCalibration,
    build_helpfulness_safety_reward,
)


class HelpfulnessSafetyRewardManager(RewardManagerBase):
    """Persistent two-axis RM manager with dynamic tensor batching."""

    def __init__(
        self,
        config: Any,
        tokenizer: Any,
        compute_score: Any,
        reward_router_address: str | None = None,
        reward_model_tokenizer: Any = None,
    ) -> None:
        super().__init__(config=config, tokenizer=tokenizer, compute_score=compute_score)
        if not torch.cuda.is_available():
            raise RuntimeError("H/S reward manager requires its reserved CUDA GPU")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "H/S reward worker must see exactly one reserved CUDA device"
            )
        torch.cuda.set_device(0)
        reward_config = config.reward
        self.calibration = HelpfulnessSafetyCalibration.from_manifest(
            reward_config.helpfulness_safety_calibration_manifest
        )
        self.scorer = DualRewardScorer.from_pretrained(
            useful_model_path=reward_config.helpfulness_safety_useful_model,
            harmless_model_path=reward_config.helpfulness_safety_harmless_model,
            device=torch.device("cuda", 0),
            max_length=int(reward_config.helpfulness_safety_max_length),
            micro_batch_size=int(
                reward_config.helpfulness_safety_score_batch_size
            ),
        )
        self.batcher: AsyncRewardBatcher[
            list[dict[str, str]], tuple[float, float]
        ] = AsyncRewardBatcher(
            score_batch=self.scorer,
            max_batch_size=int(
                reward_config.helpfulness_safety_request_batch_size
            ),
            max_wait_ms=int(reward_config.helpfulness_safety_batch_wait_ms),
        )

    async def run_single(self, data: DataProto) -> dict[str, Any]:
        if len(data) != 1:
            raise ValueError(
                f"HelpfulnessSafetyRewardManager expects one item, got {len(data)}"
            )
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_width = response_ids.shape[-1]
        response_mask = data_item.batch["attention_mask"][-response_width:]
        valid_response_length = int(response_mask.sum().item())
        if valid_response_length <= 0:
            raise ValueError("H/S reward cannot score an empty response mask")
        valid_response_ids = response_ids[:valid_response_length]
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(
                valid_response_ids,
                skip_special_tokens=True,
            ),
        )
        raw_prompt = data_item.non_tensor_batch.get("raw_prompt")
        if raw_prompt is None:
            raise KeyError("H/S reward requires raw_prompt in non_tensor_batch")
        conversation = append_assistant_response(list(raw_prompt), response)
        raw_useful, raw_harmless = await self.batcher.score(conversation)
        reward = build_helpfulness_safety_reward(
            raw_useful=raw_useful,
            raw_harmless=raw_harmless,
            calibration=self.calibration,
        )
        return {
            "reward_score": reward["score"],
            "reward_extra_info": reward,
        }
