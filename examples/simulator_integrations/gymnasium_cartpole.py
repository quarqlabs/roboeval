"""Run RoboEval against Gymnasium CartPole using example scenario data."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roboeval import EvalRunner, Ruleset, require_metric
from roboeval.loaders import load_scenarios_json


SCENARIO_PATH = Path(__file__).with_name("data") / "gymnasium_cartpole_scenarios.json"
OUTPUT_DIR = REPO_ROOT / "runs" / "simulator_integrations" / "gymnasium_cartpole"


def cartpole_balance_policy(state: dict[str, Any]) -> dict[str, Any]:
    """Small deterministic CartPole heuristic."""
    observation = state["observation"]
    cart_position = float(observation[0])
    pole_angle = float(observation[2])
    action = 0 if pole_angle + 0.1 * cart_position < 0 else 1
    return {
        "action": action,
        "debug_info": {
            "policy": "angle_position_heuristic",
            "pole_angle": pole_angle,
        },
    }


def main() -> None:
    try:
        import gymnasium as gym
        from roboeval.integrations.gymnasium import GymnasiumEnvironmentAdapter
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Gymnasium is not installed. Run: pip install -e \".[gymnasium]\""
        ) from exc

    scenarios = load_scenarios_json(SCENARIO_PATH)
    env = gym.make("CartPole-v1")
    adapter = GymnasiumEnvironmentAdapter(
        env=env,
        name="gymnasium_cartpole_v1",
        coerce_observations=True,
    )

    ruleset = Ruleset([
        require_metric("episode_return", ">=", 10.0, aggregate="last"),
    ])

    try:
        report = EvalRunner(
            policies=[cartpole_balance_policy],
            scenarios=scenarios,
            ruleset=ruleset,
            baseline_policy="cartpole_balance_policy",
            environment=adapter,
        ).run()
        report.save(OUTPUT_DIR)
    finally:
        adapter.close()

    print(f"Gymnasium CartPole report: {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
