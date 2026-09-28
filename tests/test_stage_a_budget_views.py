import pytest
from cw_grpo.stage_a_budget_views import materialize_budget_rows

def _generation_rows() -> list[dict[str, object]]:
    return [
        {
            "reward_model": {"ground_truth": "42", "style": "rule"},
            "extra_info": {
                "benchmark": "AIME-24",
                "prompt_id": "a" * 64,
            },
            "responses": ["long original", "short original"],
            "response_token_ids": [[1, 2, 3, 4], [9, 8]],
            "completion_tokens_api": [4, 2],
            "finish_reasons": ["stop", "stop"],
            "request_seeds": [11, 12],
        }
    ]


def test_materialize_budget_rows_truncates_ids_and_preserves_stopped_samples() -> None:
    source_rows = _generation_rows()

    output_rows = materialize_budget_rows(
        source_rows,
        budget=2,
        decoder=lambda token_ids: f"decoded:{','.join(map(str, token_ids))}",
    )

    assert source_rows == _generation_rows()
    assert output_rows[0]["reward_model"] == source_rows[0]["reward_model"]
    assert output_rows[0]["extra_info"] == source_rows[0]["extra_info"]
    assert output_rows[0]["responses"] == ["decoded:1,2", "short original"]
    assert output_rows[0]["response_token_ids"] == [[1, 2], [9, 8]]
    assert output_rows[0]["completion_tokens_api"] == [2, 2]
    assert output_rows[0]["finish_reasons"] == ["length", "stop"]
    assert output_rows[0]["request_seeds"] == [11, 12]


def test_materialize_budget_rows_rejects_drift_and_stale_scores() -> None:
    count_drift = _generation_rows()
    count_drift[0]["completion_tokens_api"] = [3, 2]
    with pytest.raises(ValueError, match="token count mismatch"):
        materialize_budget_rows(count_drift, budget=2, decoder=lambda _: "")

    stale_scores = _generation_rows()
    stale_scores[0]["correctness_rewards"] = [1.0, 0.0]
    with pytest.raises(ValueError, match="unscored generation rows"):
        materialize_budget_rows(stale_scores, budget=2, decoder=lambda _: "")
