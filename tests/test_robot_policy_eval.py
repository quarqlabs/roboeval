from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from robot_policy_eval import EvalRunner, SuccessCriteria, load_eval_config
from robot_policy_eval.loaders import load_scenarios_csv, load_scenarios_json
from tests.fixtures.policies import baseline_policy, regressing_policy
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
            output_dir = tmp_path / "out"
            report.save(output_dir)
            self.assertTrue((output_dir / "decision_logs.jsonl").exists())
            self.assertTrue((output_dir / "episode_results.json").exists())
            self.assertTrue((output_dir / "comparison_report.json").exists())
            self.assertTrue((output_dir / "report.md").exists())
            comparison = json.loads((output_dir / "comparison_report.json").read_text(encoding="utf-8"))

        self.assertIn("policy_summary", comparison)
        self.assertIn("regressions", comparison)

    def test_cli_runs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = _write_config_tree(tmp_path)
            output_dir = tmp_path / "out"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robot_policy_eval",
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
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    csv_path.write_text(
        "\n".join(
            [
                "name,front_distance,left_distance,right_distance,goal_direction,previous_action,step_count,max_steps,required_forward_steps",
                "csv_narrow_gap_right,18,14,55,right,none,0,6,2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return json_path, csv_path


if __name__ == "__main__":
    unittest.main()
