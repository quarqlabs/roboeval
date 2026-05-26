from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from roboeval import (
    CallableEnvironmentAdapter,
    EpisodeContext,
    EvalRunner,
    EnvironmentAdapter,
    RuleResult,
    Ruleset,
    Scenario,
    StepOutcome,
    SuccessCriteria,
    forbid_failure,
    load_eval_config,
    max_steps,
    require_metric,
    require_outcome,
)
from roboeval.loaders import load_environment, load_scenarios_csv, load_scenarios_json
from tests.fixtures.policies import baseline_policy, probabilistic_policy, regressing_policy
from tests.fixtures.scenarios import fixture_scenarios


ROOT = Path(__file__).resolve().parents[1]


class RobotEvalsSdkTest(unittest.TestCase):
    def test_import_based_sdk_usage_runs_without_cli(self) -> None:
        report = EvalRunner(
            policies=[baseline_policy, regressing_policy],
            scenarios=fixture_scenarios(),
            success_criteria=SuccessCriteria(),
            baseline_policy="baseline_policy",
        ).run()

        self.assertIn("baseline_policy", report.policy_summary)
        self.assertIn("regressing_policy", report.policy_summary)
        self.assertGreater(len(report.episodes), 0)

    def test_json_and_csv_scenarios_load_to_common_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            json_path, csv_path = _write_scenario_files(tmp_path)
            json_scenarios = load_scenarios_json(json_path)
            csv_scenarios = load_scenarios_csv(csv_path)

        self.assertEqual(json_scenarios[0].name, "json_low_distance_noise_goal_forward")
        self.assertEqual(csv_scenarios[0].name, "csv_narrow_gap_right")
        self.assertIn("front_distance", json_scenarios[0].initial_state)
        self.assertIn("front_distance", csv_scenarios[0].initial_state)
        self.assertEqual(json_scenarios[0].metadata["scenario_type"], "low_distance_noise")
        self.assertEqual(csv_scenarios[0].metadata["tags"], ["safety", "narrow_gap"])

    def test_config_loads_all_demo_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = _write_config_tree(Path(tmpdir))
            config = load_eval_config(config_path)

        self.assertEqual(len(config.policies), 2)
        self.assertEqual(len(config.scenarios), 3)
        self.assertEqual(config.baseline_policy, "baseline_policy")

    def test_report_detects_regressions_and_failure_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = _write_config_tree(Path(tmpdir))
            config = load_eval_config(config_path)
            report = EvalRunner(
                policies=config.policies,
                scenarios=config.scenarios,
                success_criteria=config.success_criteria,
                baseline_policy=config.baseline_policy,
            ).run()

        self.assertGreater(len(report.failure_cases), 0)
        self.assertTrue(
            any(regression["policy_version"] == "regressing_policy" for regression in report.regressions),
            "Expected regressing policy to regress against baseline.",
        )

    def test_report_files_are_generated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config = load_eval_config(_write_config_tree(tmp_path))
            report = EvalRunner(
                policies=config.policies,
                scenarios=config.scenarios,
                success_criteria=config.success_criteria,
                baseline_policy=config.baseline_policy,
            ).run()
            self.assertIn("metadata", report.to_dict())
            self.assertIn("highlights", report.to_dict())
            output_dir = tmp_path / "out"
            report.save(output_dir)
            self.assertTrue((output_dir / "decision_logs.jsonl").exists())
            self.assertTrue((output_dir / "episode_results.json").exists())
            self.assertTrue((output_dir / "comparison_report.json").exists())
            self.assertTrue((output_dir / "report.md").exists())
            comparison = json.loads((output_dir / "comparison_report.json").read_text(encoding="utf-8"))
            episode_results = json.loads((output_dir / "episode_results.json").read_text(encoding="utf-8"))
            markdown = (output_dir / "report.md").read_text(encoding="utf-8")

        self.assertIn("policy_summary", comparison)
        self.assertIn("metadata", comparison)
        self.assertIn("sdk_version", comparison["metadata"])
        self.assertIn("environment_name", comparison["metadata"])
        self.assertIn("highlights", comparison)
        self.assertIn("regressions", comparison)
        self.assertIn("action_divergences", comparison)
        self.assertIn("grouped_metrics", comparison)
        self.assertIn("rule_results", episode_results[0])
        self.assertIn("## Run Metadata", markdown)
        self.assertIn("## Highlights", markdown)
        self.assertIn("## Action Divergences", markdown)
        self.assertIn("## Scenario Groups", markdown)
        self.assertNotIn("Unsafe Actions", markdown)

    def test_cli_runs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = _write_config_tree(tmp_path)
            output_dir = tmp_path / "out"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "roboeval",
                    "run",
                    str(config_path),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("Eval complete", result.stdout)
            self.assertTrue((output_dir / "report.md").exists())

    def test_rule_results_next_state_and_first_failure_step_are_recorded(self) -> None:
        report = EvalRunner(
            policies=[regressing_policy],
            scenarios=[
                Scenario(
                    name="unsafe_forward",
                    initial_state={
                        "front_distance": 16,
                        "left_distance": 28,
                        "right_distance": 62,
                        "goal_direction": "forward",
                        "previous_action": "none",
                        "step_count": 0,
                    },
                    max_steps=4,
                    metadata={"scenario_type": "low_distance_noise"},
                )
            ],
            success_criteria=SuccessCriteria(),
            baseline_policy="regressing_policy",
        ).run()

        episode = report.episodes[0]
        self.assertFalse(episode.success)
        self.assertEqual(episode.failure_label, "unsafe_forward_action")
        self.assertEqual(episode.first_failure_step, 0)
        self.assertIn("unsafe_forward_action", [result.name for result in episode.rule_results])
        self.assertIn("next_state", episode.logs[0].to_dict())
        self.assertTrue(episode.logs[0].is_terminal)

    def test_custom_rule_can_fail_episode(self) -> None:
        def max_one_step_rule(logs, terminal_outcome):
            return RuleResult(
                name="max_one_step",
                passed=len(logs) <= 1,
                reason="episode took more than one step",
                step=1 if len(logs) > 1 else None,
            )

        report = EvalRunner(
            policies=[baseline_policy],
            scenarios=fixture_scenarios(),
            success_criteria=SuccessCriteria(custom_rules=[max_one_step_rule]),
            baseline_policy="baseline_policy",
        ).run()

        self.assertFalse(report.episodes[0].success)
        self.assertEqual(report.episodes[0].failure_label, "max_one_step")

    def test_action_divergence_and_grouped_metrics_are_reported(self) -> None:
        report = EvalRunner(
            policies=[baseline_policy, regressing_policy],
            scenarios=[
                Scenario(
                    name="divergent_low_distance",
                    initial_state={
                        "front_distance": 16,
                        "left_distance": 28,
                        "right_distance": 62,
                        "goal_direction": "forward",
                        "previous_action": "none",
                        "step_count": 0,
                    },
                    max_steps=6,
                    metadata={"scenario_type": "low_distance_noise", "tags": ["safety"]},
                )
            ],
            success_criteria=SuccessCriteria(),
            baseline_policy="baseline_policy",
        ).run()

        self.assertEqual(report.action_divergences[0]["step"], 0)
        self.assertEqual(report.action_divergences[0]["baseline_action"], "turn_right")
        self.assertEqual(report.action_divergences[0]["candidate_action"], "move_forward")
        self.assertIn("low_distance_noise", report.grouped_metrics)
        self.assertIn("safety", report.grouped_metrics)

    def test_probabilities_and_logits_are_preserved_in_failure_case_debug_info(self) -> None:
        report = EvalRunner(
            policies=[probabilistic_policy],
            scenarios=[
                Scenario(
                    name="probabilistic_failure",
                    initial_state={
                        "front_distance": 16,
                        "left_distance": 28,
                        "right_distance": 62,
                        "goal_direction": "forward",
                        "previous_action": "none",
                        "step_count": 0,
                    },
                    max_steps=4,
                )
            ],
            success_criteria=SuccessCriteria(),
            baseline_policy="probabilistic_policy",
        ).run()

        debug_info = report.failure_cases[0]["debug_info"]
        self.assertIn("probabilities", debug_info)
        self.assertIn("logits", debug_info)
        self.assertEqual(debug_info["model_version"], "probabilistic_policy")

    def test_custom_environment_adapter_and_step_outcome_are_public(self) -> None:
        class OneStepGoalEnvironment:
            name = "one_step_goal_environment"

            def reset(self, scenario: Scenario) -> dict[str, object]:
                return dict(scenario.initial_state)

            def step(self, action: str, scenario: Scenario) -> StepOutcome:
                next_state = {**scenario.initial_state, "previous_action": action, "step_count": 1}
                return StepOutcome(next_state=next_state, outcome="goal_reached", failure_label="", terminal=True)

        self.assertIsNotNone(EnvironmentAdapter)
        report = EvalRunner(
            policies=[baseline_policy],
            scenarios=[
                Scenario(
                    name="custom_env_goal",
                    initial_state={
                        "front_distance": 80,
                        "left_distance": 40,
                        "right_distance": 40,
                        "goal_direction": "forward",
                        "previous_action": "none",
                        "step_count": 0,
                    },
                    max_steps=2,
                )
            ],
            success_criteria=SuccessCriteria(),
            baseline_policy="baseline_policy",
            environment=OneStepGoalEnvironment(),
        ).run()

        self.assertTrue(report.episodes[0].success)
        self.assertEqual(report.metadata["environment_name"], "one_step_goal_environment")
        self.assertEqual(report.metadata["scenario_count"], 1)

    def test_highlights_explain_improvement_and_trained_policy_story(self) -> None:
        def policy_v4_trained(state):
            if (
                float(state["front_distance"]) < 20
                and float(state["left_distance"]) < 20
                and float(state["right_distance"]) < 20
            ):
                return {"action": "reverse", "model_version": "policy_v4_trained"}
            return {"action": "move_forward", "model_version": "policy_v4_trained"}

        report = EvalRunner(
            policies=[regressing_policy, policy_v4_trained],
            scenarios=[
                Scenario(
                    name="dead_end_reverse_needed",
                    initial_state={
                        "front_distance": 8,
                        "left_distance": 10,
                        "right_distance": 11,
                        "goal_direction": "forward",
                        "previous_action": "none",
                        "step_count": 0,
                    },
                    max_steps=5,
                    metadata={"required_forward_steps": 2, "scenario_type": "dead_end"},
                )
            ],
            success_criteria=SuccessCriteria(),
            baseline_policy="regressing_policy",
        ).run()
        markdown = report.to_markdown()

        self.assertTrue(any("policy_v4_trained improved dead_end_reverse_needed" in item for item in report.highlights))
        self.assertTrue(any("regressing_policy failed dead_end_reverse_needed on rule unsafe_forward_action" in item for item in report.highlights))
        self.assertTrue(any("policy_v4_trained outcome trace: escape_reverse -> progress -> goal_reached" in item for item in report.highlights))
        self.assertIn("## Highlights", markdown)
        self.assertIn("policy_v4_trained improved dead_end_reverse_needed", markdown)

    def test_ruleset_supports_robot_arm_without_navigation_fields(self) -> None:
        class ArmEnvironment:
            name = "arm_environment"

            def __init__(self) -> None:
                self.state = {}

            def reset(self, scenario: Scenario) -> dict[str, object]:
                self.state = dict(scenario.initial_state)
                return dict(self.state)

            def step(self, action: str, scenario: Scenario) -> StepOutcome:
                if action == "close_gripper":
                    self.state = {**self.state, "gripper_closed": True, "has_object": True}
                    return StepOutcome(
                        next_state=dict(self.state),
                        outcome="object_grasped",
                        failure_label="",
                        terminal=True,
                        metrics={"grip_force": 0.7},
                        events=["gripper_closed"],
                    )
                return StepOutcome(dict(self.state), "no_progress", "wrong_action", True, metrics={"grip_force": 0.1})

        def close_gripper_policy(state):
            return {"action": "close_gripper", "model_version": "arm_policy_v1"}

        report = EvalRunner(
            policies=[close_gripper_policy],
            scenarios=[
                Scenario(
                    name="grasp_cube",
                    initial_state={"object_pose": "center", "gripper_closed": False},
                    max_steps=3,
                    metadata={"scenario_type": "robot_arm", "tags": ["grasp"]},
                )
            ],
            ruleset=Ruleset([
                require_outcome("object_grasped"),
                forbid_failure("dropped_object"),
                require_metric("grip_force", "<=", 0.9),
                max_steps(3),
            ]),
            baseline_policy="close_gripper_policy",
            environment=ArmEnvironment(),
            allowed_actions=["close_gripper"],
            required_state_keys=["object_pose", "gripper_closed"],
        ).run()

        self.assertTrue(report.episodes[0].success)
        self.assertEqual(report.episodes[0].logs[0].metrics["grip_force"], 0.7)
        self.assertIn("grip_force", report.metric_summary["close_gripper_policy"])
        self.assertIn("object_grasped", report.outcome_counts["close_gripper_policy"])

    def test_custom_ruleset_rule_can_use_episode_context(self) -> None:
        def require_event(context: EpisodeContext) -> RuleResult:
            passed = any("scan_complete" in log.events for log in context.logs)
            return RuleResult(name="scan_complete_event", passed=passed, reason="" if passed else "scan event missing")

        class DroneEnvironment:
            name = "drone_environment"

            def reset(self, scenario: Scenario) -> dict[str, object]:
                return dict(scenario.initial_state)

            def step(self, action: str, scenario: Scenario) -> StepOutcome:
                return StepOutcome(
                    next_state={"battery_used": 12, "target_seen": True},
                    outcome="waypoint_inspected",
                    failure_label="",
                    terminal=True,
                    metrics={"battery_used": 12},
                    events=["scan_complete"],
                )

        def scan_policy(state):
            return "scan_target"

        report = EvalRunner(
            policies=[scan_policy],
            scenarios=[Scenario("inspect_waypoint", {"battery_used": 0}, max_steps=2)],
            ruleset=Ruleset([require_outcome("waypoint_inspected"), require_event]),
            baseline_policy="scan_policy",
            environment=DroneEnvironment(),
        ).run()

        self.assertTrue(report.episodes[0].success)

    def test_config_loads_generic_ruleset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            scenario_path = tmp_path / "scenario.json"
            scenario_path.write_text(
                json.dumps({"scenarios": [{"name": "factory_weld", "initial_state": {"temperature": 20}, "max_steps": 2}]}),
                encoding="utf-8",
            )
            ruleset_path = tmp_path / "ruleset.json"
            ruleset_path.write_text(
                json.dumps(
                    {
                        "rules": [
                            {"type": "require_outcome", "outcome": "weld_completed"},
                            {"type": "forbid_failure", "failure_label": "overheat"},
                            {"type": "require_metric", "metric": "temperature", "operator": "<=", "value": 80},
                            {"type": "max_steps", "max_steps": 2},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            config_path = tmp_path / "eval_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "baseline_policy": "regressing_policy",
                        "policies": [{"name": "regressing_policy", "path": "tests.fixtures.policies:regressing_policy"}],
                        "scenario_sources": [{"type": "json", "path": str(scenario_path)}],
                        "ruleset": str(ruleset_path),
                    }
                ),
                encoding="utf-8",
            )
            config = load_eval_config(config_path)

        self.assertIsNone(config.success_criteria)
        self.assertIsNotNone(config.ruleset)

    def test_non_string_actions_flow_to_environment_and_reports(self) -> None:
        received_actions = []

        def vector_policy(state):
            return {
                "action": [0.25, -0.5],
                "debug_info": {"kind": "continuous_vector"},
            }

        def reset_fn(scenario: Scenario) -> dict[str, object]:
            return dict(scenario.initial_state)

        def step_fn(action, scenario: Scenario) -> StepOutcome:
            received_actions.append(action)
            return StepOutcome(
                next_state={"pose": [1.0, 2.0], "last_action": action},
                outcome="pose_reached",
                failure_label="",
                terminal=True,
                metrics={"distance_error": 0.01},
            )

        report = EvalRunner(
            policies=[vector_policy],
            scenarios=[Scenario("continuous_action", {"pose": [0.0, 0.0]}, max_steps=2)],
            ruleset=Ruleset([require_outcome("pose_reached")]),
            baseline_policy="vector_policy",
            environment=CallableEnvironmentAdapter(reset_fn, step_fn, name="vector_env"),
            allowed_actions=[[0.25, -0.5]],
        ).run()

        self.assertEqual(received_actions, [[0.25, -0.5]])
        self.assertEqual(report.episodes[0].logs[0].action, [0.25, -0.5])
        report_dict = report.to_dict()
        self.assertEqual(report_dict["episodes"][0]["logs"][0]["action"], [0.25, -0.5])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "out"
            report.save(output_dir)
            decision_log = (output_dir / "decision_logs.jsonl").read_text(encoding="utf-8")
            comparison = json.loads((output_dir / "comparison_report.json").read_text(encoding="utf-8"))

        self.assertIn('"action": [0.25, -0.5]', decision_log)
        self.assertEqual(comparison["metadata"]["environment_name"], "vector_env")

    def test_action_divergence_uses_serialized_keys_for_non_string_actions(self) -> None:
        def baseline_vector_policy(state):
            return [0.1, 0.2]

        def candidate_vector_policy(state):
            return [0.2, 0.2]

        def reset_fn(scenario: Scenario) -> dict[str, object]:
            return dict(scenario.initial_state)

        def step_fn(action, scenario: Scenario) -> StepOutcome:
            return StepOutcome(
                next_state={"last_action": action},
                outcome="done",
                failure_label="",
                terminal=True,
            )

        report = EvalRunner(
            policies=[baseline_vector_policy, candidate_vector_policy],
            scenarios=[Scenario("vector_divergence", {"position": [0, 0]}, max_steps=1)],
            ruleset=Ruleset([require_outcome("done")]),
            baseline_policy="baseline_vector_policy",
            environment=CallableEnvironmentAdapter(reset_fn, step_fn, name="vector_divergence_env"),
        ).run()

        divergence = report.action_divergences[0]
        self.assertEqual(divergence["baseline_action"], [0.1, 0.2])
        self.assertEqual(divergence["candidate_action"], [0.2, 0.2])
        self.assertEqual(divergence["baseline_action_key"], "[0.1,0.2]")
        self.assertEqual(divergence["candidate_action_key"], "[0.2,0.2]")

    def test_cli_loads_custom_environment_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            scenario_path = tmp_path / "scenario.json"
            scenario_path.write_text(
                json.dumps({"scenarios": [{"name": "cli_custom_env", "initial_state": {"x": 1}, "max_steps": 1}]}),
                encoding="utf-8",
            )
            config_path = tmp_path / "eval_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "baseline_policy": "regressing_policy",
                        "policies": [{"name": "regressing_policy", "path": "tests.fixtures.policies:regressing_policy"}],
                        "scenario_sources": [{"type": "json", "path": str(scenario_path)}],
                        "ruleset": {"rules": [{"type": "require_outcome", "outcome": "factory_goal"}]},
                        "environment": {
                            "path": "tests.fixtures.environments:make_cli_environment",
                            "kwargs": {"outcome": "factory_goal"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            output_dir = tmp_path / "out"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "roboeval",
                    "run",
                    str(config_path),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            comparison = json.loads((output_dir / "comparison_report.json").read_text(encoding="utf-8"))

        self.assertIn("Eval complete", result.stdout)
        self.assertEqual(comparison["metadata"]["environment_name"], "cli_custom_environment")

    def test_environment_loader_supports_class_factory_and_instance(self) -> None:
        class_env = load_environment(
            {
                "path": "tests.fixtures.environments:CliCustomEnvironment",
                "kwargs": {"outcome": "class_goal"},
            }
        )
        factory_env = load_environment(
            {
                "path": "tests.fixtures.environments:make_cli_environment",
                "kwargs": {"outcome": "factory_goal"},
            }
        )
        instance_env = load_environment({"path": "tests.fixtures.environments:INSTANCE_ENVIRONMENT"})

        self.assertEqual(class_env.outcome, "class_goal")
        self.assertEqual(factory_env.outcome, "factory_goal")
        self.assertEqual(instance_env.outcome, "instance_goal")

    def test_root_demo_runs_end_to_end(self) -> None:
        result = subprocess.run(
            [sys.executable, "demo2.py"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn("Demo complete", result.stdout)
        self.assertTrue((ROOT / "runs" / "demo" / "robot_arm" / "report.md").exists())
        self.assertTrue((ROOT / "runs" / "demo" / "drone" / "report.md").exists())
        self.assertTrue((ROOT / "runs" / "demo" / "factory" / "report.md").exists())


def _write_config_tree(tmp_path: Path) -> Path:
    json_path, csv_path = _write_scenario_files(tmp_path)
    criteria_path = tmp_path / "criteria.json"
    criteria_path.write_text(
        json.dumps(
            {
                "must_reach_goal": True,
                "collision_is_failure": True,
                "stuck_is_failure": True,
                "unsafe_forward_min_distance": 20,
                "unsafe_forward_action": "move_forward",
                "goal_outcome": "goal_reached",
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "eval_config.json"
    config_path.write_text(
        json.dumps(
            {
                "baseline_policy": "baseline_policy",
                "policies": [
                    {"name": "baseline_policy", "path": "tests.fixtures.policies:baseline_policy"},
                    {"name": "regressing_policy", "path": "tests.fixtures.policies:regressing_policy"},
                ],
                "scenario_sources": [
                    {"type": "python", "path": "tests.fixtures.scenarios:fixture_scenarios"},
                    {"type": "json", "path": str(json_path)},
                    {"type": "csv", "path": str(csv_path)},
                ],
                "success_criteria": str(criteria_path),
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _write_scenario_files(tmp_path: Path) -> tuple[Path, Path]:
    json_path = tmp_path / "scenarios.json"
    csv_path = tmp_path / "scenarios.csv"
    json_path.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "name": "json_low_distance_noise_goal_forward",
                        "initial_state": {
                            "front_distance": 16,
                            "left_distance": 28,
                            "right_distance": 62,
                            "goal_direction": "forward",
                            "previous_action": "none",
                            "step_count": 0,
                        },
                        "max_steps": 6,
                        "required_forward_steps": 2,
                        "scenario_type": "low_distance_noise",
                        "tags": ["safety"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    csv_path.write_text(
        "\n".join(
            [
                "name,front_distance,left_distance,right_distance,goal_direction,previous_action,step_count,max_steps,required_forward_steps,scenario_type,tags",
                "csv_narrow_gap_right,18,14,55,right,none,0,6,2,narrow_gap,safety|narrow_gap",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return json_path, csv_path


if __name__ == "__main__":
    unittest.main()
