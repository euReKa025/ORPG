"""Reward-model formatting and scoring primitives for Helpfulness--Safety."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer, Qwen2Model, Qwen2PreTrainedModel
from transformers.utils.generic import ModelOutput


@dataclass
class ScoreModelOutput(ModelOutput):
    """Subset of the Align-Anything score-model output used by CW-GRPO."""

    scores: torch.FloatTensor | None = None
    end_scores: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    end_last_hidden_state: torch.FloatTensor | None = None
    end_index: torch.LongTensor | None = None


class CWQwen2RewardModel(Qwen2PreTrainedModel):
    """Qwen2 backbone plus the score head used by the official RM and CM."""

    supports_gradient_checkpointing = True

    def __init__(self, config: AutoConfig):
        super().__init__(config)
        setattr(self, self.base_model_prefix, Qwen2Model(config))
        self.score_head = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ScoreModelOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        last_hidden_state = outputs.hidden_states[-1]
        scores = self.score_head(last_hidden_state).float()
        batch_size, sequence_length, _ = last_hidden_state.size()
        if attention_mask is None:
            if batch_size > 1:
                raise ValueError("attention_mask is required when batch size > 1")
            attention_mask = last_hidden_state.new_ones(
                batch_size, sequence_length, dtype=torch.bool
            )
        end_index = torch.cat([mask.nonzero()[-1] for mask in attention_mask])
        end_last_hidden_state = torch.gather(
            last_hidden_state,
            dim=1,
            index=(
                end_index.to(last_hidden_state.device)
                .unsqueeze(dim=1)
                .unsqueeze(dim=2)
                .expand(-1, -1, last_hidden_state.size(-1))
            ),
        ).squeeze(dim=1)
        end_scores = torch.gather(
            scores,
            dim=1,
            index=(
                end_index.to(scores.device)
                .unsqueeze(dim=1)
                .unsqueeze(dim=2)
                .expand(-1, -1, scores.size(-1))
            ),
        ).squeeze(dim=1)
        return ScoreModelOutput(
            scores=scores,
            end_scores=end_scores,
            last_hidden_state=last_hidden_state,
            end_last_hidden_state=end_last_hidden_state,
            end_index=end_index,
        )


def load_reward_tokenizer(model_path: str | Path) -> Any:
    """Load every RM/CM tokenizer with the same corrected regex contract."""
    resolved_path = Path(model_path).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_path,
        trust_remote_code=False,
        fix_mistral_regex=True,
    )
    tokenizer.padding_side = "right"
    if not tokenizer.chat_template:
        raise RuntimeError(f"reward tokenizer has no chat template: {resolved_path}")
    return tokenizer


def format_reward_conversation(
    tokenizer: Any,
    messages: list[dict[str, str]],
) -> str:
    """Format with the RM/CM tokenizer, never with the policy tokenizer."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=True,
    )


@torch.inference_mode()
def score_reward_texts(
    model: CWQwen2RewardModel,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    device: torch.device,
    max_length: int = 2048,
    batch_size: int = 4,
    fail_on_truncation: bool = False,
) -> list[float]:
    """Return one finite end score per formatted conversation."""
    if not texts:
        return []
    scores: list[float] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        tokenizer_kwargs: dict[str, Any] = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": not fail_on_truncation,
        }
        if not fail_on_truncation:
            tokenizer_kwargs["max_length"] = max_length
        inputs = tokenizer(batch, **tokenizer_kwargs)
        if fail_on_truncation:
            token_lengths = inputs["attention_mask"].sum(dim=1)
            if bool((token_lengths > max_length).any()):
                longest = int(token_lengths.max().item())
                raise ValueError(
                    "formatted reward-model input exceeds reward-model "
                    f"max_length={max_length}: observed {longest}"
                )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        output = model(**inputs, use_cache=False)
        scores.extend(output.end_scores.squeeze(-1).detach().float().cpu().tolist())
    if not all(torch.isfinite(torch.tensor(scores))):
        raise RuntimeError("reward model returned a non-finite score")
    return scores
