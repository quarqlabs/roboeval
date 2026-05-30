"""Minimal raw MuJoCo rollout through RoboEval's adapter shape.

Run with:

    python -m roboeval.integrations.mujoco.demo_rollout
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from roboeval import Scenario
from roboeval.integrations.mujoco import MuJoCoEnvironmentAdapter, default_metrics_from_step


ASSET_PATH = Path(__file__).with_name("assets") / "point_mass.xml"


def point_mass_policy(state: dict) -> list[float]:
    """Small proportional controller that moves qpos[0] toward 0.5."""
    qpos = _first_float(state["qpos"])
    qvel = _first_float(state["qvel"])
    target = 0.5
    ctrl = np.clip(6.0 * (target - qpos) - 1.2 * qvel, -1.0, 1.0)
    return [float(ctrl)]


def point_mass_outcome(model, data, scenario, step_index):
    target = float(scenario.metadata.get("target_qpos", 0.5))
    tolerance = float(scenario.metadata.get("goal_tolerance", 0.035))
    distance = abs(float(data.qpos[0]) - target)

    if distance <= tolerance:
        return ("goal_reached", "", True)
    if abs(float(data.qpos[0])) > 2.0:
        return ("out_of_bounds", "out_of_bounds", True)
    return ("progress", "", False)


def point_mass_metrics(model, data, scenario, step_index):
    metrics = default_metrics_from_step(model, data, scenario, step_index)
    target = float(scenario.metadata.get("target_qpos", 0.5))
    metrics["distance_to_target"] = abs(float(data.qpos[0]) - target)
    return metrics


def main() -> None:
    scenario = Scenario(
        name="point_mass_reaches_target",
        initial_state={"qpos": [0.0], "qvel": [0.0], "ctrl": [0.0]},
        max_steps=120,
        metadata={"target_qpos": 0.5, "goal_tolerance": 0.035},
    )
    adapter = MuJoCoEnvironmentAdapter.from_xml_path(
        ASSET_PATH,
        name="mujoco_point_mass",
        outcome_from_step=point_mass_outcome,
        metrics_from_step=point_mass_metrics,
        substeps=2,
        coerce_observations=True,
    )

    state = adapter.reset(scenario)
    print(f"reset state={state}")
    for step in range(scenario.max_steps):
        action = point_mass_policy(state)
        outcome = adapter.step(action, scenario)
        distance = outcome.metrics["distance_to_target"]
        qpos = _first_float(outcome.next_state["qpos"])
        print(
            f"step={step:02d} action={action} qpos={qpos:.3f} "
            f"distance={distance:.3f} outcome={outcome.outcome}"
        )
        state = outcome.next_state
        if outcome.terminal:
            print(f"terminal outcome={outcome.outcome} failure={outcome.failure_label!r}")
            break


def _first_float(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        return float(value[0])
    if hasattr(value, "shape") and getattr(value, "shape", None):
        return float(value[0])
    return float(value)


if __name__ == "__main__":
    main()
