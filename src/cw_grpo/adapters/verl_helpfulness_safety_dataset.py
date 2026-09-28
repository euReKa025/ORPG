from __future__ import annotations

from typing import Any

from verl.utils.dataset.rl_dataset import RLHFDataset

from cw_grpo.helpfulness_safety_dataset import left_truncate_chat_messages


class LeftTruncatingHelpfulnessSafetyDataset(RLHFDataset):
    """Preserve every frozen H/S row while enforcing async prompt width."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not self.return_raw_chat:
            raise ValueError("H/S custom dataset requires data.return_raw_chat=true")
        if self.truncation != "left":
            raise ValueError("H/S custom dataset requires data.truncation=left")

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = super().__getitem__(item)
        raw_prompt, was_truncated = left_truncate_chat_messages(
            row["raw_prompt"],
            tokenizer=self.tokenizer,
            max_prompt_length=int(self.max_prompt_length),
        )
        row["raw_prompt"] = raw_prompt
        row["hs_prompt_left_truncated"] = was_truncated
        return row
