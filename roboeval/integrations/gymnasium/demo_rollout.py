"""Manual rollout demo for the Gymnasium integration spike.

Runs CartPole-v1 through ``GymnasiumEnvironmentAdapter`` step-by-step and
prints the resulting ``StepOutcome`` for each step. Demonstrates:

- Wrapping a vanilla ``gymnasium.Env`` with zero modifications.
- ``Action = Any`` means a plain ``int`` action travels straight through.
- ``next_state`` exposes the raw observation; ``metrics`` carries ``reward``
  and ``episode_return``; ``info["gymnasium"]`` carries the raw terminated /
  truncated / info structure.
- ``terminal`` becomes ``True`` either when the pole falls (``terminated``)
  or the time limit is hit (``truncated``).

Run::

    python -m roboeval.integrations.gymnasium.demo_rollout
"""

from __future__ import annotations

import gymnasium as gym

from roboeval.core import Scenario
from roboeval.integrations.gymnasium import GymnasiumEnvironmentAdapter


def naive_balance_policy(state: dict) -> int:
    """If the pole is leaning left (negative angle), push left (action 0).
    Otherwise push right (action 1). Trivial heuristic — not great, but enough
    to drive the loop past the first step."""
    pole_angle = float(state["observation"][2])
    return 0 if pole_angle < 0 else 1


def main() -> None:
    env = gym.make("CartPole-v1")
    adapter = GymnasiumEnvironmentAdapter(env=env, name="cartpole_v1")

    scenario = Scenario(
        name="cartpole_smoke",
        initial_state={"seed": 0},
        max_steps=200,
    )

    state = adapter.reset(scenario)
    print(f"[reset]   state={state}")

    for step in range(scenario.max_steps):
        action = naive_balance_policy(state)
        outcome = adapter.step(action, scenario)
        print(
            f"[step {step:2d}] action={action} "
            f"outcome={outcome.outcome:<22} terminal={outcome.terminal} "
            f"reward={outcome.metrics['reward']:.1f} "
            f"return={outcome.metrics['episode_return']:.1f} "
            f"events={outcome.events}"
        )
        state = outcome.next_state
        if outcome.terminal:
            break

    adapter.close()


if __name__ == "__main__":
    main()
