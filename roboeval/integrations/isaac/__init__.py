"""Isaac Lab integration for roboeval.

Wraps any single-env Isaac Lab environment (``gym.make("Isaac-...-v0", num_envs=1)``)
into roboeval's ``EnvironmentAdapter`` so policies can be evaluated against Isaac
through ``EvalRunner``.

Quick start::

    import gymnasium as gym
    from roboeval import EvalRunner, Ruleset, Scenario, require_metric
    from roboeval.integrations.isaac import IsaacEnvironmentAdapter

    env = gym.make("Isaac-Cartpole-Direct-v0", num_envs=1)
    adapter = IsaacEnvironmentAdapter(env=env, name="isaac_cartpole")

    report = EvalRunner(
        policies=[my_policy],
        scenarios=[Scenario("smoke", {"seed": 0}, max_steps=200)],
        ruleset=Ruleset([require_metric("episode_return", ">=", 50.0)]),
        baseline_policy="my_policy",
        environment=adapter,
    ).run()
    report.save("runs/isaac_smoke")

Isaac Lab installation is non-trivial (NVIDIA GPU + Linux/Windows + Isaac Sim
+ Isaac Lab build). See ``README.md`` for the workflow.
"""

from .adapter import (
    IsaacEnvironmentAdapter,
    default_action_from_decision,
    default_events_from_step,
    default_observation_to_state,
    default_options_from_scenario,
    default_outcome_from_step,
    default_seed_from_scenario,
    tensor_to_numpy,
)

__all__ = [
    "IsaacEnvironmentAdapter",
    "default_action_from_decision",
    "default_events_from_step",
    "default_observation_to_state",
    "default_options_from_scenario",
    "default_outcome_from_step",
    "default_seed_from_scenario",
    "tensor_to_numpy",
]
