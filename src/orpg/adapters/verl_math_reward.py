from __future__ import annotations

import asyncio
from math import isfinite
from typing import Any

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

from orpg.math_stage_a_reward import DEFAULT_LENGTH_THRESHOLD, score_stage_a_response


class MathStageARewardManager(RewardManagerBase):
    """Scalar GRPO reward plus reward-component diagnostics for Stage A math."""

    def __init__(
        self,
        config: Any,
        tokenizer: Any,
        compute_score: Any,
        reward_router_address: str | None = None,
        reward_model_tokenizer: Any = None,
    ) -> None:
        super().__init__(config=config, tokenizer=tokenizer, compute_score=compute_score)
        self.length_threshold = int(
            config.reward.get("stage_a_length_threshold", DEFAULT_LENGTH_THRESHOLD)
        )
        self.scalar_length_weight = float(
            config.reward.get("stage_a_scalar_length_weight", 1.0)
        )
        if not isfinite(self.scalar_length_weight) or self.scalar_length_weight < 0.0:
            raise ValueError(
                "stage_a_scalar_length_weight must be non-negative and finite"
            )

    async def run_single(self, data: DataProto) -> dict[str, Any]:
        if len(data) != 1:
            raise ValueError(f"MathStageARewardManager expects one item, got {len(data)}")

        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_width = response_ids.shape[-1]
        response_mask = data_item.batch["attention_mask"][-response_width:]
        valid_response_length = int(response_mask.sum().item())
        if valid_response_length <= 0:
            raise ValueError("Stage A reward cannot score an empty response mask")

        valid_response_ids = response_ids[:valid_response_length]
        loop = asyncio.get_running_loop()
        response_str = await loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True),
        )
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        result = score_stage_a_response(
            response_str,
            ground_truth,
            response_token_count=valid_response_length,
            length_threshold=self.length_threshold,
        )
        result["score"] = (
            float(result["correctness_reward"])
            + self.scalar_length_weight * float(result["length_reward"])
        )
        # Async Verl stores reward extras in NumPy arrays and later passes each
        # scalar directly to json.dumps(). NumPy float64 is JSON-compatible,
        # while NumPy int64 is not, so keep this diagnostic numeric but floating.
        result["response_token_count"] = float(result["response_token_count"])
        return {"reward_score": result["score"], "reward_extra_info": result}
