from __future__ import annotations

import json
from pathlib import Path

import pytest

from orpg.helpfulness_safety_dataset import (
    chat_prompt_token_ids,
    left_truncate_chat_messages,
)
from orpg.helpfulness_safety_evaluation import (
    EVALUATION_DATASETS,
    EvaluationCalibration,
    assign_shard,
    summarize_scored_rows,
    validate_complete_rows,
)


class _MultiTurnCharacterTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt: bool, tokenize: bool):
        assert add_generation_prompt
        assert tokenize
        token_ids = [1]
        for message in messages:
            token_ids.append(3 if message["role"] == "user" else 4)
            token_ids.extend(ord(character) for character in message["content"])
        token_ids.append(2)
        return token_ids

    def __call__(self, text, *, add_special_tokens: bool, return_attention_mask: bool):
        assert not add_special_tokens
        assert not return_attention_mask
        return {"input_ids": [ord(character) for character in text]}

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ):
        assert skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(chr(token_id) for token_id in token_ids)


def test_hs_mean1_contract_uses_all_three_official_eval_sets() -> None:
    assert [(item.name, item.rows) for item in EVALUATION_DATASETS] == [
        ("alpaca", 512),
        ("hh_rlhf", 8_520),
        ("pku_saferlhf", 8_211),
    ]
    assert sum(item.rows for item in EVALUATION_DATASETS) == 17_243


def test_sharding_is_deterministic_and_exhaustive() -> None:
    assignments = [assign_shard(index, shard_count=8) for index in range(100)]
    assert assignments[:10] == [0, 1, 2, 3, 4, 5, 6, 7, 0, 1]
    assert set(assignments) == set(range(8))
    with pytest.raises(ValueError, match="shard_count"):
        assign_shard(0, shard_count=0)


def test_multiturn_left_truncation_drops_oldest_complete_turns() -> None:
    tokenizer = _MultiTurnCharacterTokenizer()
    messages = [
        {"role": "user", "content": "old-question"},
        {"role": "assistant", "content": "old-answer"},
        {"role": "user", "content": "latest-question"},
    ]

    truncated, changed = left_truncate_chat_messages(
        messages,
        tokenizer=tokenizer,
        max_prompt_length=20,
    )

    assert changed is True
    assert truncated == [{"role": "user", "content": "latest-question"}]
    assert len(chat_prompt_token_ids(tokenizer, truncated)) <= 20
    assert messages[0]["content"] == "old-question"


def test_complete_row_validation_rejects_duplicates_and_missing_rows() -> None:
    rows = [
        {"dataset": "alpaca", "prompt_id": "alpaca-0"},
        {"dataset": "alpaca", "prompt_id": "alpaca-1"},
    ]
    validate_complete_rows(rows, expected_by_dataset={"alpaca": 2})

    with pytest.raises(ValueError, match="duplicate"):
        validate_complete_rows([rows[0], rows[0]], expected_by_dataset={"alpaca": 2})
    with pytest.raises(ValueError, match="row count"):
        validate_complete_rows(rows[:1], expected_by_dataset={"alpaca": 2})


def test_summary_reports_raw_and_frozen_calibrated_mean1_with_equal_macro() -> None:
    calibration = EvaluationCalibration(
        calibration_id="cal-v3",
        epsilon=1.0e-6,
        useful_mean=2.0,
        useful_std=2.0,
        harmless_mean=4.0,
        harmless_std=4.0,
    )
    rows = [
        {"dataset": "alpaca", "raw_useful": 2.0, "raw_harmless": 4.0},
        {"dataset": "alpaca", "raw_useful": 4.0, "raw_harmless": 8.0},
        {"dataset": "hh_rlhf", "raw_useful": 6.0, "raw_harmless": 0.0},
        {"dataset": "pku_saferlhf", "raw_useful": 0.0, "raw_harmless": 12.0},
    ]

    summary = summarize_scored_rows(rows, calibration=calibration)

    assert summary["protocol"] == "full_mean_at_1"
    assert summary["calibration_id"] == "cal-v3"
    assert summary["datasets"]["alpaca"]["raw_useful_mean_at_1"] == pytest.approx(3.0)
    assert summary["datasets"]["alpaca"]["calibrated_useful_mean_at_1"] == pytest.approx(
        0.5 / (1.0 + 0.5e-6)
    )
    assert summary["macro"]["raw_useful_mean_at_1"] == pytest.approx((3.0 + 6.0 + 0.0) / 3)
    assert summary["macro"]["raw_harmless_mean_at_1"] == pytest.approx(
        (6.0 + 0.0 + 12.0) / 3
    )






def test_calibration_manifest_parser_uses_frozen_population_coordinates(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "calibration_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "calibration_id": "hs-v3",
                "coordinates": {
                    "epsilon": 1.0e-6,
                    "std_definition": "population_std_ddof_0",
                    "useful": {"mean": 2.5, "population_std": 2.25},
                    "harmless": {"mean": 3.0, "population_std": 2.0},
                },
            }
        ),
        encoding="utf-8",
    )

    calibration = EvaluationCalibration.from_manifest(manifest)

    assert calibration.calibration_id == "hs-v3"
    assert calibration.useful_mean == 2.5
    assert calibration.harmless_std == 2.0
