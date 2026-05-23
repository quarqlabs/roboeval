# Robot Policy Eval

`robot-policy-eval` is a local Python SDK for running repeatable evaluations on robot policy versions.

The SDK is designed for teams that already have their own policies, scenarios, simulators, and robot repositories. Version 1 focuses on the local workflow: plug in policies, define scenarios and success criteria, run evals, and get structured results back.

This repository currently contains the SDK core only. Demo examples are being kept local for now and will be published later.

## Why This Exists

Robot policy development is hard to debug because failures are often spread across observations, actions, simulator state, logs, and policy versions.

This SDK aims to make the evaluation loop simple:

```text
policies + scenarios + success criteria
        |
        v
run local evals
        |
        v
state/action/outcome logs
        |
        v
metrics + failures + regressions
        |
        v
structured report
```

## Current Scope

Version 1 is local-only.

Included:

- importable Python SDK
- local CLI
- policy adapter interface
- scenario and success-criteria models
- deterministic eval runner
- state/action/outcome decision logs
- policy-version comparison
- regression and failure-case detection
- JSONL, JSON, and Markdown report outputs

Not included yet:

- GitHub Actions integration
- hosted dashboard
- Isaac/Gazebo/MuJoCo/ROS2 adapters
- real robot capture integrations
- public demo example package

## Install Locally

From the repository root:

```bash
python3 -m pip install -e .
```

## SDK Usage

```python
from robot_policy_eval import EvalRunner, Scenario, SuccessCriteria


def policy_v1(state):
    return {"action": "move_forward", "debug_info": {"version": "policy_v1"}}


def policy_v2(state):
    return {"action": "turn_right", "debug_info": {"version": "policy_v2"}}


scenarios = [
    Scenario(
        name="open_path",
        initial_state={
            "front_distance": 80,
            "left_distance": 45,
            "right_distance": 45,
            "goal_direction": "forward",
            "previous_action": "none",
            "step_count": 0,
        },
        max_steps=6,
        metadata={"required_forward_steps": 2},
    )
]

report = EvalRunner(
    policies=[policy_v1, policy_v2],
    scenarios=scenarios,
    success_criteria=SuccessCriteria(),
    baseline_policy="policy_v1",
).run()

report.save("runs/latest")
```

## CLI Usage

The CLI is available as a module:

```bash
python3 -m robot_policy_eval run path/to/eval_config.json
```

If installed locally, it is also available as:

```bash
robot-policy-eval run path/to/eval_config.json
```

The CLI writes:

```text
runs/latest/decision_logs.jsonl
runs/latest/episode_results.json
runs/latest/comparison_report.json
runs/latest/report.md
```

## Report Outputs

`decision_logs.jsonl` contains step-level records:

- episode id
- scenario name
- policy version
- step
- state
- action
- outcome
- failure label
- policy debug info

`comparison_report.json` contains:

- policy summaries
- success rates
- failure counts
- collision/stuck/unsafe-action counts
- average steps
- improvements against baseline
- regressions against baseline
- failure cases

`report.md` is the human-readable summary.

## Development

Run tests:

```bash
python3 -m unittest discover -s tests
```

Run the package CLI:

```bash
python3 -m robot_policy_eval --help
```

## Roadmap

Near-term:

- publish demo examples
- improve config ergonomics
- add richer failure explanations
- add more report formats

Later:

- GitHub Actions integration
- hosted eval history and dashboards
- simulator adapters for Isaac, Gazebo, MuJoCo, and ROS2 workflows
