from orpg.math_stage_a_data import prepare_stage_a_records, prompt_id


def test_prepare_stage_a_records_excludes_invalid_and_all_dev_duplicates() -> None:
    records = [
        {"problem": "Repeated problem", "answer": "1", "solution": "first"},
        {"problem": "Repeated   problem", "answer": "1", "solution": "duplicate"},
        {"problem": "Another problem", "answer": "2", "solution": "second"},
        {"problem": "Third problem", "answer": "3", "solution": "third"},
        {"problem": "Missing answer", "answer": "", "solution": "invalid"},
    ]

    prepared = prepare_stage_a_records(records, tuning_dev_size=1)

    dev_prompt_ids = {row["extra_info"]["prompt_id"] for row in prepared.tuning_dev_rows}
    train_prompt_ids = {row["extra_info"]["prompt_id"] for row in prepared.train_rows}

    assert len(dev_prompt_ids) == 1
    assert dev_prompt_ids.isdisjoint(train_prompt_ids)
    assert prepared.manifest["source_rows"] == 5
    assert prepared.manifest["invalid_missing_answer_rows"] == 1
    assert prepared.manifest["unique_valid_prompts"] == 3
    assert prepared.manifest["tuning_dev_unique_prompts"] == 1
    assert prepared.manifest["train_rows"] + prepared.manifest["tuning_dev_rows"] == 4

    for row in (*prepared.train_rows, *prepared.tuning_dev_rows):
        assert row["data_source"] == "agentica-org/DeepScaleR-Preview-Dataset"
        assert row["ability"] == "math"
        assert row["prompt"][0]["role"] == "user"
        assert row["prompt"][0]["content"].endswith(
            "Please reason step by step, and put your final answer within \\boxed{}."
        )
        assert row["reward_model"]["style"] == "rule"
        assert row["reward_model"]["ground_truth"]


def test_prepare_stage_a_records_can_exclude_frozen_benchmark_overlap() -> None:
    records = [
        {"problem": "Keep one", "answer": "1", "solution": ""},
        {"problem": "External benchmark problem", "answer": "2", "solution": ""},
        {"problem": "Keep two", "answer": "3", "solution": ""},
    ]
    overlap_id = prompt_id("External benchmark problem")

    prepared = prepare_stage_a_records(
        records,
        tuning_dev_size=1,
        excluded_prompt_ids={overlap_id},
    )

    output_ids = {
        row["extra_info"]["prompt_id"]
        for row in (*prepared.train_rows, *prepared.tuning_dev_rows)
    }
    assert overlap_id not in output_ids
    assert prepared.manifest["excluded_prompt_rows"] == 1
    assert prepared.manifest["excluded_unique_prompts"] == 1
