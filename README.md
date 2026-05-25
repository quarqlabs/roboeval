# Robot Policy Eval

`robot-policy-eval` is a local Python SDK for running repeatable evaluations on robot policy versions.

The SDK is designed for teams that already have their own policies, scenarios, simulators, and robot repositories. Version 1 focuses on the local workflow: plug in policies, define scenarios and success criteria, run evals, and get structured results back.

This repository contains the SDK core, v1 docs, tests, and examples.

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
- scenario, ruleset, and success-criteria models
- deterministic eval runner
- state/action/outcome decision logs
- transition logs with next state and terminal markers
- structured rule-level pass/fail results
- baseline action divergence detection
- scenario grouped metrics
- policy-version comparison
- regression and failure-case detection
- JSONL, JSON, and Markdown report outputs
- generic rule helpers for arbitrary robot domains

Not included yet:

- GitHub Actions integration
- hosted dashboard
- Isaac/Gazebo/MuJoCo/ROS2 adapters
- real robot capture integrations
- packaged example distribution

## SDK v1 Docs

- [Overview](docs/v1/overview.md)
- [Generic Rules](docs/v1/generic-rules.md)
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

## Quickstart Demo

Clone the repo and run the local demo:

```bash
python3 demo.py
```

This runs three dependency-free eval suites under `examples/generic_robots/`:

- robot arm/gripper
- drone inspection
- factory welding process

Reports are written to:

```text
runs/demo/robot_arm/report.md
runs/demo/drone/report.md
runs/demo/factory/report.md
```

## SDK Usage

```python
from robot_policy_eval import EvalRunner, Ruleset, Scenario, forbid_failure, max_steps, require_outcome


def policy_v1(state):
    return {"action": "move_arm_down", "debug_info": {"version": "policy_v1"}}


def policy_v2(state):
    return {
        "action": "close_gripper",
        "probabilities": {"close_gripper": 0.8, "move_arm_down": 0.2},
        "model_version": "policy_v2",
    }


scenarios = [
    Scenario(
        name="grasp_cube",
        initial_state={
            "object_pose": "center",
            "gripper_closed": False,
            "has_object": False,
        },
        max_steps=6,
        metadata={"scenario_type": "robot_arm", "tags": ["grasp"]},
    )
]

# `my_robot_env` implements reset(scenario) and step(action, scenario).
report = EvalRunner(
    policies=[policy_v1, policy_v2],
    scenarios=scenarios,
    ruleset=Ruleset([
        require_outcome("object_grasped"),
        forbid_failure("dropped_object"),
        max_steps(6),
    ]),
    baseline_policy="policy_v1",
    environment=my_robot_env,
).run()

report.save("runs/latest")
```

## Generic Rules

`Ruleset` is the recommended generic API. It works for mobile robots, robot arms, drones, factory robots, and policies with arbitrary action/state names.

```python
from robot_policy_eval import Ruleset, forbid_failure, require_metric, require_outcome


arm_rules = Ruleset([
    require_outcome("object_grasped"),
    forbid_failure("dropped_object"),
    require_metric("grip_force", "<=", 0.9),
])
```

`SuccessCriteria()` still exists as a backward-compatible mobile-navigation preset.

## CLI Usage

The CLI is available as a module:

```bash
python3 -m robot_policy_eval run path/to/eval_config.json
```

If installed locally, it is also available as:

```bash
robot-policy-eval run path/to/eval_config.json
```

Config files can optionally provide a custom environment:

```json
{
  "environment": {
    "path": "my_robot.envs:make_eval_environment",
    "kwargs": {"seed": 7}
  }
}
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
- average steps
- rule results per episode
- first failure step
- improvements against baseline
- regressions against baseline
- failure cases
- first action divergences against baseline
- grouped metrics by `scenario_type` and `tags`
- human-readable highlights
- outcome counts and metric summaries

`report.md` is the human-readable summary. It includes run metadata, highlights, policy summaries, regressions, improvements, failure explanations, action divergences, and scenario groups.

Policy debug info can include standard ML fields such as:

- `scores`
- `probabilities`
- `logits`
- `confidence`
- `model_version`

These are preserved in decision logs and failure cases so trained-model behavior is easier to inspect.

Policy actions are generic. String actions like `move_forward` work, and so do continuous/vector actions such as lists, tuples, arrays, or tensor-like objects. The SDK passes the raw action to the environment and serializes a JSON-safe copy in reports.

## Examples

Run the one-command demo:

```bash
python3 demo.py
```

Run generic any-robot examples:

```bash
python3 examples/generic_robots/run_eval.py
```

Run the config-based mobile demo:

```bash
python3 -m robot_policy_eval run examples/configs/eval_config.json --output-dir runs/demo_robot
```

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
