"""Run RoboEval against a raw MuJoCo point-mass XML model."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roboeval import EvalRunner, Ruleset, require_metric, require_outcome
from roboeval.loaders import load_scenarios_json


SCENARIO_PATH = Path(__file__).with_name("data") / "mujoco_point_mass_scenarios.json"
XML_PATH = REPO_ROOT / "roboeval" / "integrations" / "mujoco" / "assets" / "point_mass.xml"
OUTPUT_DIR = REPO_ROOT / "runs" / "simulator_integrations" / "mujoco_point_mass"


def point_mass_controller(state: dict[str, Any], target: float = 0.5) -> dict[str, Any]:
    qpos = _first_float(state["qpos"])
    qvel = _first_float(state["qvel"])
    ctrl = _clip(6.0 * (target - qpos) - 1.2 * qvel, -1.0, 1.0)
    return {
        "action": [ctrl],
        "debug_info": {
            "policy": "proportional_derivative",
            "target_qpos": target,
            "qpos": qpos,
            "qvel": qvel,
        },
    }


class ScenarioTargetPolicy:
    version = "point_mass_pd_controller"

    def decide(self, state: dict[str, Any]) -> dict[str, Any]:
        target = float(state.get("target_qpos", 0.5))
        return point_mass_controller(state, target=target)


def point_mass_observation(model, data, scenario) -> dict[str, Any]:
    return {
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "ctrl": data.ctrl.copy(),
        "time": float(data.time),
        "target_qpos": float(scenario.metadata.get("target_qpos", 0.5)),
    }


def point_mass_outcome(model, data, scenario, step_index) -> tuple[str, str, bool]:
    target = float(scenario.metadata.get("target_qpos", 0.5))
    tolerance = float(scenario.metadata.get("goal_tolerance", 0.035))
    distance = abs(float(data.qpos[0]) - target)

    if distance <= tolerance:
        return ("goal_reached", "", True)
    if abs(float(data.qpos[0])) > 2.0:
        return ("out_of_bounds", "out_of_bounds", True)
    return ("progress", "", False)


def point_mass_metrics(model, data, scenario, step_index) -> dict[str, float | int]:
    from roboeval.integrations.mujoco import default_metrics_from_step

    metrics = default_metrics_from_step(model, data, scenario, step_index)
    target = float(scenario.metadata.get("target_qpos", 0.5))
    metrics["distance_to_target"] = abs(float(data.qpos[0]) - target)
    return metrics


def main() -> None:
    try:
        from roboeval.integrations.mujoco import MuJoCoEnvironmentAdapter
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "MuJoCo is not installed. Run: pip install -e \".[mujoco]\""
        ) from exc

    scenarios = load_scenarios_json(SCENARIO_PATH)
    adapter = MuJoCoEnvironmentAdapter.from_xml_path(
        XML_PATH,
        name="mujoco_point_mass",
        observation_from_data=point_mass_observation,
        outcome_from_step=point_mass_outcome,
        metrics_from_step=point_mass_metrics,
        substeps=2,
        coerce_observations=True,
    )

    ruleset = Ruleset([
        require_outcome("goal_reached"),
        require_metric("distance_to_target", "<=", 0.05, aggregate="last"),
    ])

    report = EvalRunner(
        policies=[ScenarioTargetPolicy()],
        scenarios=scenarios,
        ruleset=ruleset,
        baseline_policy="point_mass_pd_controller",
        environment=adapter,
    ).run()
    report.save(OUTPUT_DIR)

    print(f"MuJoCo point-mass report: {OUTPUT_DIR / 'report.md'}")


def _first_float(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        return float(value[0])
    if hasattr(value, "shape") and getattr(value, "shape", None):
        return float(value[0])
    return float(value)


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


if __name__ == "__main__":
    main()
