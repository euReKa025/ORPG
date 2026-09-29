from __future__ import annotations

from verl.trainer import main_ppo
from verl.trainer.main_ppo import TaskRunner as VerlTaskRunner

from orpg.adapters.verl_objective_wise_runtime import ObjectiveWiseTaskRunner


def install_objective_wise_task_runner() -> None:
    current = main_ppo.TaskRunner
    if current not in {VerlTaskRunner, ObjectiveWiseTaskRunner}:
        raise RuntimeError("refusing to replace an unexpected Verl TaskRunner")
    main_ppo.TaskRunner = ObjectiveWiseTaskRunner


def main() -> None:
    install_objective_wise_task_runner()
    main_ppo.main()


if __name__ == "__main__":
    main()
