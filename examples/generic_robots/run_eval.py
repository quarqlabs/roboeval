from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roboeval import EvalRunner, Ruleset, forbid_failure, max_steps, require_metric, require_outcome

from examples.generic_robots.environments import DroneInspectionEnvironment, FactoryWeldEnvironment, RobotArmEnvironment
from examples.generic_robots.policies import (
    ARM_ACTIONS,
    DRONE_ACTIONS,
    FACTORY_ACTIONS,
    arm_policy_v0_rules,
    arm_policy_v1_linear,
    arm_policy_v2_mlp,
    arm_policy_v3_regression,
    drone_policy_v0_rules,
    drone_policy_v1_linear,
    drone_policy_v2_mlp,
    drone_policy_v3_risky,
    factory_policy_v0_rules,
    factory_policy_v1_linear,
    factory_policy_v2_mlp,
    factory_policy_v3_hot,
)
from examples.generic_robots.scenarios import arm_scenarios, drone_scenarios, factory_scenarios


def build_runs() -> list[tuple[str, object, list[object], list[object], Ruleset, str, list[str]]]:
    return [
        (
            "robot_arm",
            RobotArmEnvironment(),
            [arm_policy_v0_rules, arm_policy_v1_linear, arm_policy_v2_mlp, arm_policy_v3_regression],
            arm_scenarios(),
            Ruleset([
                require_outcome("object_lifted"),
                forbid_failure("dropped_object"),
                forbid_failure("missed_grasp"),
                forbid_failure("object_missing"),
                forbid_failure("over_force"),
                require_metric("grip_force", "<=", 0.98, aggregate="max"),
                max_steps(8),
            ]),
            "arm_policy_v0_rules",
            ARM_ACTIONS,
        ),
        (
            "drone",
            DroneInspectionEnvironment(),
            [drone_policy_v0_rules, drone_policy_v1_linear, drone_policy_v2_mlp, drone_policy_v3_risky],
            drone_scenarios(),
            Ruleset([
                require_outcome("waypoint_inspected"),
                forbid_failure("no_fly_zone_violation"),
                forbid_failure("lost_target"),
                forbid_failure("battery_exceeded"),
                forbid_failure("unsafe_low_altitude"),
                require_metric("battery_used", "<=", 30, aggregate="last"),
                max_steps(7),
            ]),
            "drone_policy_v0_rules",
            DRONE_ACTIONS,
        ),
        (
            "factory",
            FactoryWeldEnvironment(),
            [factory_policy_v0_rules, factory_policy_v1_linear, factory_policy_v2_mlp, factory_policy_v3_hot],
            factory_scenarios(),
            Ruleset([
                require_outcome("weld_completed"),
                forbid_failure("overheat"),
                forbid_failure("underheated_weld"),
                forbid_failure("inspection_missing"),
                forbid_failure("no_work_completed"),
                require_metric("temperature", "<=", 95, aggregate="max"),
                max_steps(5),
            ]),
            "factory_policy_v0_rules",
            FACTORY_ACTIONS,
        ),
    ]


def run_all(output_root: str = "runs", name_prefix: str = "generic_") -> None:
    for name, environment, policies, scenarios, ruleset, baseline_policy, allowed_actions in build_runs():
        report = EvalRunner(
            policies=policies,
            scenarios=scenarios,
            ruleset=ruleset,
            baseline_policy=baseline_policy,
            environment=environment,
            allowed_actions=allowed_actions,
        ).run()
        output_dir = str(Path(output_root) / f"{name_prefix}{name}")
        report.save(output_dir)
        print(f"\n{name}: report={output_dir}/report.md")
        for policy, summary in report.policy_summary.items():
            print(
                "  {policy}: success_rate={success_rate} failures={failure_count} avg_steps={average_steps}".format(
                    policy=policy,
                    **summary,
                )
            )
        print(f"  regressions={len(report.regressions)} improvements={len(report.improvements)} failures={len(report.failure_cases)}")


def main() -> None:
    run_all()


if __name__ == "__main__":
    main()
