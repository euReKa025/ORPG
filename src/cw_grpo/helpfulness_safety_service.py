"""Persistent, dynamically batched scoring primitives for the two H/S RMs."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


class DualRewardScorer:
    """Keep both reward models resident and score aligned conversation batches."""

    def __init__(
        self,
        *,
        useful_model: Any,
        useful_tokenizer: Any,
        harmless_model: Any,
        harmless_tokenizer: Any,
        device: Any,
        max_length: int,
        micro_batch_size: int,
        formatter: Callable[[Any, list[dict[str, str]]], str],
        score_texts: Callable[..., list[float]],
    ) -> None:
        if max_length <= 0 or micro_batch_size <= 0:
            raise ValueError("reward-model length and micro batch must be positive")
        self.useful_model = useful_model
        self.useful_tokenizer = useful_tokenizer
        self.harmless_model = harmless_model
        self.harmless_tokenizer = harmless_tokenizer
        self.device = device
        self.max_length = max_length
        self.micro_batch_size = micro_batch_size
        self.formatter = formatter
        self.score_texts = score_texts

    @classmethod
    def from_pretrained(
        cls,
        *,
        useful_model_path: str | Path,
        harmless_model_path: str | Path,
        device: Any,
        max_length: int,
        micro_batch_size: int,
    ) -> DualRewardScorer:
        import torch

        from cw_grpo.helpfulness_safety_scoring import (
            CWQwen2RewardModel,
            format_reward_conversation,
            load_reward_tokenizer,
            score_reward_texts,
        )

        def load_axis(model_path: str | Path) -> tuple[Any, Any]:
            resolved_path = Path(model_path).resolve()
            if not resolved_path.is_dir():
                raise FileNotFoundError(f"reward model directory not found: {resolved_path}")
            tokenizer = load_reward_tokenizer(resolved_path)
            model = CWQwen2RewardModel.from_pretrained(
                resolved_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            ).to(device)
            model.eval()
            return model, tokenizer

        useful_model, useful_tokenizer = load_axis(useful_model_path)
        harmless_model, harmless_tokenizer = load_axis(harmless_model_path)
        return cls(
            useful_model=useful_model,
            useful_tokenizer=useful_tokenizer,
            harmless_model=harmless_model,
            harmless_tokenizer=harmless_tokenizer,
            device=device,
            max_length=max_length,
            micro_batch_size=micro_batch_size,
            formatter=format_reward_conversation,
            score_texts=score_reward_texts,
        )

    def __call__(
        self,
        conversations: list[list[dict[str, str]]],
    ) -> list[tuple[float, float]]:
        useful_texts = [
            self.formatter(self.useful_tokenizer, conversation)
            for conversation in conversations
        ]
        harmless_texts = [
            self.formatter(self.harmless_tokenizer, conversation)
            for conversation in conversations
        ]
        useful_scores = self.score_texts(
            self.useful_model,
            self.useful_tokenizer,
            useful_texts,
            device=self.device,
            max_length=self.max_length,
            batch_size=self.micro_batch_size,
            fail_on_truncation=True,
        )
        harmless_scores = self.score_texts(
            self.harmless_model,
            self.harmless_tokenizer,
            harmless_texts,
            device=self.device,
            max_length=self.max_length,
            batch_size=self.micro_batch_size,
            fail_on_truncation=True,
        )
        if len(useful_scores) != len(conversations) or len(harmless_scores) != len(
            conversations
        ):
            raise RuntimeError("dual reward scorer returned a misaligned batch")
        paired = list(zip(useful_scores, harmless_scores, strict=True))
        if not all(math.isfinite(value) for pair in paired for value in pair):
            raise RuntimeError("dual reward scorer returned a non-finite score")
        return paired


def append_assistant_response(
    raw_prompt: Sequence[dict[str, str]],
    response: str,
) -> list[dict[str, str]]:
    """Copy a policy prompt and append exactly one assistant response."""
    if not isinstance(response, str):
        raise TypeError("assistant response must be a string")
    conversation: list[dict[str, str]] = []
    for message in raw_prompt:
        if not isinstance(message, dict):
            raise TypeError("raw_prompt messages must be dictionaries")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise TypeError("raw_prompt messages require string role and content")
        conversation.append({"role": role, "content": content})
    conversation.append({"role": "assistant", "content": response})
    return conversation


@dataclass(slots=True)
class _Pending(Generic[RequestT, ResultT]):
    request: RequestT
    future: asyncio.Future[ResultT]


class AsyncRewardBatcher(Generic[RequestT, ResultT]):
    """Coalesce concurrent ``run_single`` calls into serialized tensor batches."""

    def __init__(
        self,
        *,
        score_batch: Callable[[list[RequestT]], list[ResultT]],
        max_batch_size: int,
        max_wait_ms: int,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms cannot be negative")
        self._score_batch = score_batch
        self._max_batch_size = max_batch_size
        self._max_wait_seconds = max_wait_ms / 1000.0
        self._pending: list[_Pending[RequestT, ResultT]] = []
        self._pending_lock: asyncio.Lock | None = None
        self._score_lock: asyncio.Lock | None = None
        self._delayed_flush: asyncio.Task[None] | None = None

    def _locks(self) -> tuple[asyncio.Lock, asyncio.Lock]:
        if self._pending_lock is None:
            self._pending_lock = asyncio.Lock()
            self._score_lock = asyncio.Lock()
        assert self._score_lock is not None
        return self._pending_lock, self._score_lock

    async def score(self, request: RequestT) -> ResultT:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ResultT] = loop.create_future()
        pending_lock, _ = self._locks()
        batch: list[_Pending[RequestT, ResultT]] | None = None
        async with pending_lock:
            self._pending.append(_Pending(request=request, future=future))
            if len(self._pending) >= self._max_batch_size:
                batch = self._take_pending_locked()
            elif self._delayed_flush is None:
                self._delayed_flush = asyncio.create_task(self._flush_after_wait())
        if batch:
            asyncio.create_task(self._run_batch(batch))
        return await future

    def _take_pending_locked(self) -> list[_Pending[RequestT, ResultT]]:
        batch = self._pending[: self._max_batch_size]
        del self._pending[: self._max_batch_size]
        if self._delayed_flush is not None:
            self._delayed_flush.cancel()
            self._delayed_flush = None
        if self._pending:
            self._delayed_flush = asyncio.create_task(self._flush_after_wait())
        return batch

    async def _flush_after_wait(self) -> None:
        try:
            await asyncio.sleep(self._max_wait_seconds)
            pending_lock, _ = self._locks()
            async with pending_lock:
                self._delayed_flush = None
                if not self._pending:
                    return
                batch = self._take_pending_locked()
            await self._run_batch(batch)
        except asyncio.CancelledError:
            return

    async def _run_batch(
        self,
        batch: list[_Pending[RequestT, ResultT]],
    ) -> None:
        _, score_lock = self._locks()
        try:
            async with score_lock:
                results = await asyncio.to_thread(
                    self._score_batch,
                    [item.request for item in batch],
                )
            if len(results) != len(batch):
                raise RuntimeError(
                    "reward scorer returned a different number of results"
                )
        except Exception as exc:  # noqa: BLE001 - fan out worker failure to callers.
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)
            return
        for item, result in zip(batch, results, strict=True):
            if not item.future.done():
                item.future.set_result(result)
