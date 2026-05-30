"""Tests for the raw MuJoCo integration spike.

Run with:

    python -m unittest discover -s roboeval/integrations/mujoco/tests
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from roboeval import EvalRunner, Ruleset, Scenario, require_metric, require_outcome
from roboeval.core import to_serializable
from roboeval.environment import StepOutcome
from roboeval.integrations.mujoco import MuJoCoEnvironmentAdapter


POINT_MASS_XML = """
<mujoco model="test_point_mass">
  <option timestep="0.02" gravity="0 0 0"/>
  <worldbody>
    <body name="point" pos="0 0 0">
      <joint name="slide_x" type="slide" axis="1 0 0" damping="1"/>
      <geom name="point_geom" type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor_x" joint="slide_x" gear="1" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


class MuJoCoAdapterConstructionTest(unittest.TestCase):
    def test_loads_model_from_xml_string(self) -> None:
        adapter = MuJoCoEnvironmentAdapter.from_xml_string(POINT_MASS_XML)
        self.assertEqual(adapter.model.nq, 1)
        self.assertEqual(adapter.model.nv, 1)
        self.assertEqual(adapter.model.nu, 1)

    def test_loads_model_from_xml_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir) / "point_mass.xml"
            xml_path.write_text(POINT_MASS_XML, encoding="utf-8")
            adapter = MuJoCoEnvironmentAdapter.from_xml_path(xml_path)
        self.assertEqual(adapter.model.nu, 1)

    def test_invalid_substeps_raise_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "substeps"):
            MuJoCoEnvironmentAdapter.from_xml_string(POINT_MASS_XML, substeps=0)


class MuJoCoAdapterResetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = MuJoCoEnvironmentAdapter.from_xml_string(POINT_MASS_XML)

    def test_reset_applies_qpos_qvel_ctrl_from_scenario(self) -> None:
        scenario = Scenario(
            "reset_values",
            {"qpos": [0.25], "qvel": [0.1], "ctrl": [0.5]},
            max_steps=3,
        )
        state = self.adapter.reset(scenario)
        self.assertAlmostEqual(float(state["qpos"][0]), 0.25)
        self.assertAlmostEqual(float(state["qvel"][0]), 0.1)
        self.assertAlmostEqual(float(state["ctrl"][0]), 0.5)

    def test_wrong_qpos_shape_raises_clear_error(self) -> None:
        scenario = Scenario("bad_qpos", {"qpos": [0.1, 0.2]}, max_steps=1)
        with self.assertRaisesRegex(ValueError, "qpos expected length 1"):
            self.adapter.reset(scenario)

    def test_wrong_qvel_shape_raises_clear_error(self) -> None:
        scenario = Scenario("bad_qvel", {"qvel": [0.1, 0.2]}, max_steps=1)
        with self.assertRaisesRegex(ValueError, "qvel expected length 1"):
            self.adapter.reset(scenario)

    def test_wrong_ctrl_shape_raises_clear_error(self) -> None:
        scenario = Scenario("bad_ctrl", {"ctrl": [0.1, 0.2]}, max_steps=1)
        with self.assertRaisesRegex(ValueError, "ctrl expected length 1"):
            self.adapter.reset(scenario)


class MuJoCoAdapterStepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = MuJoCoEnvironmentAdapter.from_xml_string(POINT_MASS_XML)
        self.scenario = Scenario("step", {"qpos": [0.0], "qvel": [0.0], "ctrl": [0.0]}, max_steps=5)
        self.adapter.reset(self.scenario)

    def test_action_list_writes_to_ctrl(self) -> None:
        self.adapter.step([0.7], self.scenario)
        self.assertAlmostEqual(float(self.adapter.data.ctrl[0]), 0.7)

    def test_action_tuple_writes_to_ctrl(self) -> None:
        self.adapter.step((0.3,), self.scenario)
        self.assertAlmostEqual(float(self.adapter.data.ctrl[0]), 0.3)

    def test_action_array_writes_to_ctrl(self) -> None:
        self.adapter.step(np.array([0.2]), self.scenario)
        self.assertAlmostEqual(float(self.adapter.data.ctrl[0]), 0.2)

    def test_action_dict_ctrl_writes_to_ctrl(self) -> None:
        self.adapter.step({"ctrl": [0.4]}, self.scenario)
        self.assertAlmostEqual(float(self.adapter.data.ctrl[0]), 0.4)

    def test_wrong_action_shape_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "action/ctrl expected length 1"):
            self.adapter.step([0.1, 0.2], self.scenario)

    def test_step_returns_step_outcome_with_metrics_and_info(self) -> None:
        outcome = self.adapter.step([0.5], self.scenario)
        self.assertIsInstance(outcome, StepOutcome)
        self.assertIn("qpos", outcome.next_state)
        self.assertEqual(outcome.outcome, "progress")
        self.assertFalse(outcome.terminal)
        self.assertIn("qpos_norm", outcome.metrics)
        self.assertIn("mujoco", outcome.info)
        self.assertEqual(outcome.info["mujoco"]["model_nu"], 1)

    def test_metrics_are_json_serializable_through_sdk_serialization(self) -> None:
        outcome = self.adapter.step([0.5], self.scenario)
        payload = to_serializable({"state": outcome.next_state, "metrics": outcome.metrics, "info": outcome.info})
        json.dumps(payload)


class MuJoCoAdapterHooksAndRunnerTest(unittest.TestCase):
    def test_custom_outcome_hook_can_emit_goal_reached_and_terminal(self) -> None:
        def outcome_from_step(model, data, scenario, step_index):
            if float(data.qpos[0]) >= 0.0:
                return ("goal_reached", "", True)
            return ("progress", "", False)

        adapter = MuJoCoEnvironmentAdapter.from_xml_string(
            POINT_MASS_XML,
            outcome_from_step=outcome_from_step,
        )
        scenario = Scenario("goal", {"qpos": [0.0], "qvel": [0.0], "ctrl": [0.0]}, max_steps=2)
        adapter.reset(scenario)
        outcome = adapter.step([0.0], scenario)
        self.assertEqual(outcome.outcome, "goal_reached")
        self.assertTrue(outcome.terminal)

    def test_eval_runner_can_run_policy_with_mujoco_adapter(self) -> None:
        def outcome_from_step(model, data, scenario, step_index):
            if step_index >= 2:
                return ("goal_reached", "", True)
            return ("progress", "", False)

        adapter = MuJoCoEnvironmentAdapter.from_xml_string(
            POINT_MASS_XML,
            outcome_from_step=outcome_from_step,
            substeps=1,
        )

        def push_right_policy(state):
            return {"action": [1.0], "debug_info": {"controller": "constant_push"}}

        scenario = Scenario("runner_goal", {"qpos": [0.0], "qvel": [0.0], "ctrl": [0.0]}, max_steps=5)
        report = EvalRunner(
            policies=[push_right_policy],
            scenarios=[scenario],
            ruleset=Ruleset([
                require_outcome("goal_reached"),
                require_metric("time", ">=", 0.0),
            ]),
            baseline_policy="push_right_policy",
            environment=adapter,
        ).run()

        self.assertEqual(report.metadata["environment_name"], "mujoco_env")
        self.assertEqual(len(report.episodes), 1)
        self.assertTrue(report.episodes[0].success)
        self.assertEqual(report.episodes[0].terminal_outcome, "goal_reached")


if __name__ == "__main__":
    unittest.main()
