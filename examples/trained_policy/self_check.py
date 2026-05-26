from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roboeval import EvalRunner, SuccessCriteria

from examples.demo_robot.scenarios import get_python_scenarios
from examples.trained_policy.actions import ACTIONS
from examples.trained_policy.data import clean_rows, generate_synthetic_rows
from examples.trained_policy.policy import policy_v4_trained


def main() -> None:
    rows = generate_synthetic_rows(count=300, seed=123)
    clean, rejected = clean_rows(rows)
    assert len(clean) == 300
    assert len(rejected) == 0
    assert set(ACTIONS).issubset({row["expert_action"] for row in clean})

    decision = policy_v4_trained(
        {
            "front_distance": 16,
            "left_distance": 28,
            "right_distance": 62,
            "goal_direction": "forward",
            "previous_action": "none",
            "step_count": 0,
        }
    )
    assert decision["action"] in ACTIONS
    assert "probabilities" in decision["debug_info"]
    assert "logits" in decision["debug_info"]
    assert decision["debug_info"]["model_version"] == "policy_v4_trained"

    report = EvalRunner(
        policies=[policy_v4_trained],
        scenarios=get_python_scenarios(),
        success_criteria=SuccessCriteria(),
        baseline_policy="policy_v4_trained",
    ).run()
    assert "policy_v4_trained" in report.policy_summary
    assert report.policy_summary["policy_v4_trained"]["success_rate"] >= 0.5
    assert report.metadata["environment_name"] == "DemoRobotEnvironment"
    print("trained policy self-check passed")


if __name__ == "__main__":
    main()
