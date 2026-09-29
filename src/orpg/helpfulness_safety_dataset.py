from __future__ import annotations

from copy import deepcopy
from typing import Any


def _normalize_token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("expected one tokenized sequence")
        value = value[0]
    return [int(token_id) for token_id in value]


def chat_prompt_token_ids(tokenizer: Any, messages: list[dict[str, Any]]) -> list[int]:
    return _normalize_token_ids(
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )
    )


def left_truncate_chat_messages(
    messages: list[dict[str, Any]],
    *,
    tokenizer: Any,
    max_prompt_length: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Apply the configured left-truncation contract to a text-only raw chat.

    verl's asynchronous raw-chat rollout path formats the chat after the dataset
    item is returned.  Its tokenizer padding call does not truncate an already
    overlong sequence, so the dataset adapter must shorten the raw message first.
    """

    if max_prompt_length <= 0:
        raise ValueError("max_prompt_length must be positive")
    copied = deepcopy(messages)
    if len(chat_prompt_token_ids(tokenizer, copied)) <= max_prompt_length:
        return copied, False
    if not copied or copied[0].get("role") != "user" or copied[-1].get("role") != "user":
        raise ValueError("H/S left truncation requires a user-started, user-ended chat")
    for message in copied:
        if message.get("role") not in {"user", "assistant"}:
            raise ValueError("H/S left truncation only supports user/assistant chat roles")
        if not isinstance(message.get("content"), str):
            raise TypeError("H/S left truncation requires string message content")

    # Match left truncation without beginning the retained prompt in the middle
    # of an assistant turn.  Drop the oldest complete history until only the
    # newest user-led suffix remains or it fits as-is.
    while len(copied) > 1:
        next_user = next(
            (index for index, message in enumerate(copied[1:], start=1) if message["role"] == "user"),
            None,
        )
        if next_user is None:
            break
        copied = copied[next_user:]
        if len(chat_prompt_token_ids(tokenizer, copied)) <= max_prompt_length:
            return copied, True

    if len(copied) != 1 or copied[0].get("role") != "user":
        raise ValueError("unable to retain a valid final user turn during H/S truncation")
    content = copied[0].get("content")
    assert isinstance(content, str)

    empty = deepcopy(copied)
    empty[0]["content"] = ""
    template_overhead = len(chat_prompt_token_ids(tokenizer, empty))
    content_budget = max_prompt_length - template_overhead
    if content_budget <= 0:
        raise ValueError("chat template alone exceeds max_prompt_length")

    encoded = tokenizer(
        content,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    content_ids = _normalize_token_ids(encoded)
    suffix_ids = content_ids[-content_budget:]
    while suffix_ids:
        copied[0]["content"] = tokenizer.decode(
            suffix_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if len(chat_prompt_token_ids(tokenizer, copied)) <= max_prompt_length:
            return copied, True
        suffix_ids = suffix_ids[1:]
    raise ValueError("unable to left-truncate H/S prompt within max_prompt_length")
