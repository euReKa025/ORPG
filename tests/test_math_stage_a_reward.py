import sys
from concurrent.futures import ThreadPoolExecutor

import orpg.math_stage_a_reward as reward_module
from orpg.math_stage_a_reward import score_stage_a_response


def test_score_stage_a_response_combines_correctness_and_binary_length() -> None:
    result = score_stage_a_response(
        "The final answer is \\boxed{\\frac{1}{2}}.",
        "0.5",
        response_token_count=4000,
    )

    assert result == {
        "score": 2.0,
        "correctness_reward": 1.0,
        "length_reward": 1.0,
        "response_token_count": 4000,
        "parse_failure": 0.0,
    }


def test_score_stage_a_response_parses_unboxed_latex_ground_truth() -> None:
    result = score_stage_a_response(
        "The final answer is \\boxed{\\frac{\\sqrt{2}}{2}}.",
        "\\frac{\\sqrt{2}}{2}",
        response_token_count=100,
    )

    assert result["correctness_reward"] == 1.0
    assert result["parse_failure"] == 0.0


def test_score_stage_a_response_is_safe_inside_reward_worker_thread() -> None:
    """Verl's async reward loop invokes custom scoring from a worker thread."""

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            score_stage_a_response,
            "The final answer is \\boxed{42}.",
            "42",
            response_token_count=64,
        )

    assert future.result()["correctness_reward"] == 1.0


def test_score_math_subprocess_suppresses_third_party_parser_output(
    monkeypatch,
    capsys,
) -> None:
    def noisy_parse(text: str):
        print(f"parser output: {text}")
        return [text]

    def noisy_verify(ground_truth, completion) -> bool:
        print("verifier output", file=sys.stderr)
        return ground_truth == ["42"] and completion == ["answer"]

    monkeypatch.setattr(reward_module, "parse", noisy_parse)
    monkeypatch.setattr(reward_module, "verify", noisy_verify)

    result = reward_module._score_math_in_subprocess("answer", "42")

    assert result == (1.0, 0.0)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
