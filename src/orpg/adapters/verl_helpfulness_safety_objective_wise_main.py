from __future__ import annotations

from verl.trainer import main_ppo
from verl.trainer.main_ppo import TaskRunner as VerlTaskRunner

from orpg.adapters.verl_helpfulness_safety_runtime import (
    HelpfulnessSafetyObjectiveWiseTaskRunner,
)


def install_task_runner() -> None:
    current = main_ppo.TaskRunner
    if current not in {VerlTaskRunner, HelpfulnessSafetyObjectiveWiseTaskRunner}:
        raise RuntimeError("refusing to replace an unexpected Verl TaskRunner")
    main_ppo.TaskRunner = HelpfulnessSafetyObjectiveWiseTaskRunner


def main() -> None:
    install_task_runner()
    main_ppo.main()


if __name__ == "__main__":
    main()
