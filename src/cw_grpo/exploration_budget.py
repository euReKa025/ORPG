"""Bound an isolated exploration prefix without shortening worker LR schedules."""
from __future__ import annotations


def exploration_stop_at_step(trainer):
    config = trainer.config
    stop = config.trainer.get('exploration_stop_at_step', None)
    if stop is None:
        return None
    if type(stop) is not int or not 1 <= stop <= 100:
        raise ValueError('exploration_stop_at_step must be an integer in [1, 100]')
    if not str(config.trainer.experiment_name).startswith('positive-'):
        raise ValueError('exploration requires its own positive- experiment namespace')
    if config.trainer.total_training_steps != 100 or trainer.total_training_steps != 100:
        raise ValueError('exploration must initialize a 100-step training schedule')
    if config.actor_rollout_ref.actor.optim.total_training_steps != 100:
        raise ValueError('actor worker optimizer schedule must remain 100 steps')
    if config.trainer.save_freq <= 0:
        raise ValueError('exploration must save the complete prefix checkpoint')
    if config.trainer.test_freq != -1 or config.trainer.val_before_train:
        raise ValueError('exploration evaluation must use its separately frozen dev entry')
    return stop


def validate_exploration_resume_step(global_steps, stop):
    if stop is not None and not 0 <= global_steps < stop:
        raise ValueError('restored checkpoint already reaches or exceeds exploration stop; refuse extra update')
