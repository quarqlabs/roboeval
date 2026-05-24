# Robot Policy Eval

`robot-policy-eval` is a local Python SDK for running repeatable evaluations on robot policy versions.

The SDK is designed for teams that already have their own policies, scenarios, simulators, and robot repositories. Version 1 focuses on the local workflow: plug in policies, define scenarios and success criteria, run evals, and get structured results back.

This repository currently contains the SDK core and v1 docs. Demo examples and expanded tests are being kept local for now.

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
- environment adapter interface
- scenario and success-criteria models
- deterministic eval runner
- state/action/outcome decision logs
- transition logs with next state and terminal markers
- structured rule-level pass/fail results
- baseline action divergence detection
- scenario grouped metrics
- policy-version comparison
- regression and failure-case detection
- JSONL, JSON, and Markdown report outputs

Not included yet:

- GitHub Actions integration
- hosted dashboard
- Isaac/Gazebo/MuJoCo/ROS2 adapters
- real robot capture integrations
- public demo example package

## SDK v1 Docs

- [Overview](docs/v1/overview.md)
- [Environment Adapter](docs/v1/environment-adapter.md)
- [Policy Adapter](docs/v1/policy-adapter.md)
- [Reports](docs/v1/reports.md)
- [Adapter Guide In 20 Lines](docs/v1/adapter-guide-20-lines.md)
- [Clean Demo Flow](docs/v1/demo-flow.md)

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
    return {
        "action": "turn_right",
        "probabilities": {"turn_right": 0.8, "move_forward": 0.2},
        "model_version": "policy_v2",
    }


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
        metadata={"required_forward_steps": 2, "scenario_type": "navigation", "tags": ["smoke"]},
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

## Custom Rules

`SuccessCriteria` includes built-in checks for goal reached, collision, stuck, and unsafe forward actions. You can add custom rules when your policy has domain-specific constraints.

```python
from robot_policy_eval import RuleResult, SuccessCriteria


def max_three_steps(logs, terminal_outcome):
    passed = len(logs) <= 3
    return RuleResult(
        name="max_three_steps",
        passed=passed,
        reason="" if passed else "episode took more than 3 steps",
        step=3 if not passed else None,
    )


criteria = SuccessCriteria(custom_rules=[max_three_steps])
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
- next state
- outcome
- failure label
- terminal marker
- policy debug info

`comparison_report.json` contains:

- run metadata: SDK version, timestamp, baseline, policies, scenario count, and environment name
- policy summaries
- success rates
- failure counts
- collision/stuck/unsafe-action counts
- average steps
- rule results per episode
- first failure step
- improvements against baseline
- regressions against baseline
- failure cases
- first action divergences against baseline
- grouped metrics by `scenario_type` and `tags`
- human-readable highlights

`report.md` is the human-readable summary. It includes run metadata, highlights, policy summaries, regressions, improvements, failure explanations, action divergences, and scenario groups.

Policy debug info can include standard ML fields such as:

- `scores`
- `probabilities`
- `logits`
- `confidence`
- `model_version`

These are preserved in decision logs and failure cases so trained-model behavior is easier to inspect.

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
