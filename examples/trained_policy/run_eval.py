from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roboeval import EvalRunner, SuccessCriteria

from examples.demo_robot.policies import policy_v1_cautious, policy_v2_aggressive, policy_v3_balanced
from examples.demo_robot.scenarios import get_python_scenarios
from examples.trained_policy.policy import policy_v4_trained


def main() -> None:
    report = EvalRunner(
        policies=[policy_v1_cautious, policy_v2_aggressive, policy_v3_balanced, policy_v4_trained],
        scenarios=get_python_scenarios(),
        success_criteria=SuccessCriteria(),
        baseline_policy="policy_v1_cautious",
    ).run()
    report.save("runs/trained_policy_v4")
    print("trained policy eval complete")
    print("report: runs/trained_policy_v4/report.md")
    print(f"environment: {report.metadata['environment_name']}")
    print(f"policies: {', '.join(report.metadata['policy_versions'])}")
    print(f"regressions: {len(report.regressions)}")
    print(f"failure cases: {len(report.failure_cases)}")
    if report.highlights:
        print("highlights:")
        for highlight in report.highlights[:5]:
            print(f"- {highlight}")


if __name__ == "__main__":
    main()
