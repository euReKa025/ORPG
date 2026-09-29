from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any

CANONICAL_EVAL_BUDGET = 32768
_SCORED_FIELDS = frozenset(
    {"correctness_rewards", "parse_failures", "response_token_counts"}
)
Decoder = Callable[[Sequence[int]], str]


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def materialize_budget_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    decoder: Decoder,
) -> list[dict[str, Any]]:
    """Derive one exact token-prefix budget view from canonical 32k generations."""

    if budget <= 0 or budget >= CANONICAL_EVAL_BUDGET:
        raise ValueError(
            f"budget must be positive and smaller than {CANONICAL_EVAL_BUDGET}"
        )
    if not rows:
        raise ValueError("generation rows are empty")

    output_rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows):
        stale_fields = sorted(_SCORED_FIELDS.intersection(row))
        if stale_fields:
            raise ValueError(
                "budget views require unscored generation rows; "
                f"row {row_number} contains {stale_fields}"
            )

        fields = (
            row.get("responses"),
            row.get("response_token_ids"),
            row.get("completion_tokens_api"),
            row.get("finish_reasons"),
            row.get("request_seeds"),
        )
        if not all(_is_sequence(field) for field in fields):
            raise TypeError(f"generation row {row_number} has invalid response arrays")
        lengths = {len(field) for field in fields}
        if len(lengths) != 1 or not next(iter(lengths)):
            raise ValueError(f"generation row {row_number} has misaligned response arrays")

        responses: list[str] = []
        token_id_groups: list[list[int]] = []
        api_counts: list[int] = []
        finish_reasons: list[str] = []
        request_seeds: list[int] = []
        for sample_id, values in enumerate(zip(*fields, strict=True)):
            response, raw_token_ids, api_count, finish_reason, request_seed = values
            if not isinstance(response, str):
                raise TypeError(
                    f"generation row {row_number} sample {sample_id} has invalid text"
                )
            if not _is_sequence(raw_token_ids) or not all(
                isinstance(token_id, int) and not isinstance(token_id, bool)
                for token_id in raw_token_ids
            ):
                raise TypeError(
                    f"generation row {row_number} sample {sample_id} has invalid token ids"
                )
            token_ids = [int(token_id) for token_id in raw_token_ids]
            if any(token_id < 0 for token_id in token_ids):
                raise ValueError(
                    f"generation row {row_number} sample {sample_id} has negative token ids"
                )
            if (
                not isinstance(api_count, int)
                or isinstance(api_count, bool)
                or api_count != len(token_ids)
            ):
                raise ValueError(
                    f"generation row {row_number} sample {sample_id} token count mismatch"
                )
            if not isinstance(finish_reason, str) or not finish_reason:
                raise ValueError(
                    f"generation row {row_number} sample {sample_id} has invalid finish reason"
                )
            if not isinstance(request_seed, int) or isinstance(request_seed, bool):
                raise TypeError(
                    f"generation row {row_number} sample {sample_id} has invalid seed"
                )

            if len(token_ids) > budget:
                view_token_ids = token_ids[:budget]
                view_response = decoder(view_token_ids)
                if not isinstance(view_response, str):
                    raise TypeError("tokenizer decoder must return text")
                view_finish_reason = "length"
            else:
                view_token_ids = token_ids
                view_response = response
                view_finish_reason = finish_reason

            responses.append(view_response)
            token_id_groups.append(view_token_ids)
            api_counts.append(len(view_token_ids))
            finish_reasons.append(view_finish_reason)
            request_seeds.append(request_seed)

        output_row = copy.deepcopy(dict(row))
        output_row.update(
            {
                "responses": responses,
                "response_token_ids": token_id_groups,
                "completion_tokens_api": api_counts,
                "finish_reasons": finish_reasons,
                "request_seeds": request_seeds,
            }
        )
        output_rows.append(output_row)
    return output_rows
