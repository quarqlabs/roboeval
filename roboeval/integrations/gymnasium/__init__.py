"""Gymnasium integration for roboeval.

Wraps any ``gymnasium.Env`` into roboeval's ``EnvironmentAdapter`` so policies
can be evaluated against Gymnasium environments using ``EvalRunner``.
"""

from .adapter import (
    GymnasiumEnvironmentAdapter,
    default_action_from_decision,
    default_events_from_step,
    default_observation_to_state,
    default_options_from_scenario,
    default_outcome_from_step,
    default_seed_from_scenario,
)
from .batched_adapter import BatchedGymnasiumEnvironmentAdapter

__all__ = [
    "BatchedGymnasiumEnvironmentAdapter",
    "GymnasiumEnvironmentAdapter",
    "default_action_from_decision",
    "default_events_from_step",
    "default_observation_to_state",
    "default_options_from_scenario",
    "default_outcome_from_step",
    "default_seed_from_scenario",
]
