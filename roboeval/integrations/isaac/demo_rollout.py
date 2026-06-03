"""Manual rollout demo for the Isaac Lab integration.

Runs Isaac-Cartpole-Direct-v0 through ``IsaacEnvironmentAdapter`` and prints
the resulting ``StepOutcome`` for each step. Demonstrates:

- Wrapping a single-env Isaac Lab environment (``num_envs=1``)
- GPU tensor → numpy state coercion happening automatically
- Action passthrough from a numpy/scalar policy to a batched torch tensor
- ``info["isaac"]`` namespace with terminated / truncated / raw_info

Requires Isaac Sim + Isaac Lab installed (NVIDIA GPU + Linux/Windows). Will
not run on Mac. See ``README.md`` for the recommended cloud-GPU workflow.

Run::

    python -m roboeval.integrations.isaac.demo_rollout
"""

from __future__ import annotations

import sys


def naive_policy(state: dict) -> int:
    """Push the cart in the direction the pole is leaning.

    Isaac Cartpole observation typically exposes
    ``state["policy"]`` or ``state["observation"]`` as a 4-vector
    ``[cart_position, cart_velocity, pole_angle, pole_velocity]``.
    """
    obs = state.get("policy")
    if obs is None:
        obs = state.get("observation")
    if obs is None:
        # Unknown observation shape; default to 0
        return 0
    pole_angle = float(obs[2]) if len(obs) > 2 else 0.0
    return 0 if pole_angle < 0 else 1


def main() -> None:
    try:
        import gymnasium as gym
    except ImportError:
        print(
            "ERROR: gymnasium is required. Install via `pip install gymnasium>=0.29`.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        import isaaclab  # noqa: F401  - just to verify Isaac Lab is installed
        # NOTE: Some Isaac Lab installs require additional setup (running
        # ``./isaaclab.sh -p ...``); see the README for details.
    except ImportError:
        print(
            "ERROR: Isaac Lab is not installed. This demo requires a working "
            "Isaac Sim + Isaac Lab install. See README.md for setup.",
            file=sys.stderr,
        )
        sys.exit(1)

    from roboeval.core import Scenario
    from roboeval.integrations.isaac import IsaacEnvironmentAdapter

    env_id = "Isaac-Cartpole-Direct-v0"
    print(f"Creating env: {env_id} (num_envs=1)")
    env = gym.make(env_id, num_envs=1)

    adapter = IsaacEnvironmentAdapter(env=env, name="isaac_cartpole")

    scenario = Scenario(
        name="isaac_cartpole_smoke",
        initial_state={"seed": 0},
        max_steps=200,
    )

    state = adapter.reset(scenario)
    print(f"[reset]   state keys={list(state.keys())}")

    for step in range(scenario.max_steps):
        action = naive_policy(state)
        outcome = adapter.step(action, scenario)
        print(
            f"[step {step:3d}] action={action} "
            f"outcome={outcome.outcome:<22} terminal={outcome.terminal} "
            f"reward={outcome.metrics['reward']:+.2f} "
            f"return={outcome.metrics['episode_return']:.2f} "
            f"events={outcome.events}"
        )
        state = outcome.next_state
        if outcome.terminal:
            break

    print(f"\n✓ Demo complete. Final return: {adapter._episode_return:.2f}")
    adapter.close()


if __name__ == "__main__":
    main()
