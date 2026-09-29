from __future__ import annotations

import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

from math_verify import parse, verify

DEFAULT_LENGTH_THRESHOLD = 4000
DEFAULT_SCORING_TIMEOUT_SECONDS = 30.0

_score_pool: ProcessPoolExecutor | None = None
_score_pool_lock = threading.Lock()


def parse_stage_a_ground_truth(ground_truth: str):
    """Parse DeepScaleR's bare final-answer field with a boxed fallback."""

    text = str(ground_truth).strip()
    parsed = parse(text)
    if parsed:
        return parsed
    return parse(f"\\boxed{{{text}}}")


def _get_score_pool() -> ProcessPoolExecutor:
    global _score_pool
    if _score_pool is None:
        with _score_pool_lock:
            if _score_pool is None:
                _score_pool = ProcessPoolExecutor(
                    max_workers=4,
                    mp_context=multiprocessing.get_context("spawn"),
                )
    return _score_pool


def _score_math_in_subprocess(
    completion: str, ground_truth: str
) -> tuple[float, float]:
    """Run signal-based math-verify timeouts in a subprocess main thread."""

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        parsed_ground_truth = parse_stage_a_ground_truth(ground_truth)
        if not parsed_ground_truth:
            raise ValueError(f"ground truth is not parseable: {ground_truth!r}")

        parsed_completion = parse(completion)
        parse_failure = float(not parsed_completion)
        correctness_reward = float(
            bool(parsed_completion) and verify(parsed_ground_truth, parsed_completion)
        )
    return correctness_reward, parse_failure


def _score_math_with_timeout(
    completion: str,
    ground_truth: str,
    *,
    timeout_seconds: float = DEFAULT_SCORING_TIMEOUT_SECONDS,
) -> tuple[float, float]:
    future = _get_score_pool().submit(
        _score_math_in_subprocess,
        completion,
        ground_truth,
    )
    try:
        return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError:
        future.cancel()
        return 0.0, 1.0


def score_stage_a_response(
    completion: str,
    ground_truth: str,
    *,
    response_token_count: int,
    length_threshold: int = DEFAULT_LENGTH_THRESHOLD,
) -> dict[str, float | int]:
    if response_token_count < 0:
        raise ValueError("response_token_count must be non-negative")
    if length_threshold < 0:
        raise ValueError("length_threshold must be non-negative")

    correctness_reward, parse_failure = _score_math_with_timeout(
        completion,
        ground_truth,
    )
    length_reward = float(response_token_count <= length_threshold)

    return {
        "score": correctness_reward + length_reward,
        "correctness_reward": correctness_reward,
        "length_reward": length_reward,
        "response_token_count": response_token_count,
        "parse_failure": parse_failure,
    }
