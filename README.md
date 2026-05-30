# roboeval

`roboeval` is a local Python SDK for running repeatable evaluations on robot policy versions.

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
- Isaac/Gazebo/ROS2 adapters
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

This runs three policies against three scenarios and prints a structured report to the terminal.

The report is also saved to:

```text
runs/demo/report.md
runs/demo/comparison_report.json
runs/demo/decision_logs.jsonl
runs/demo/episode_results.json
```

## SDK Usage

```python
from roboeval import EvalRunner, Ruleset, Scenario, forbid_failure, max_steps, require_outcome


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
from roboeval import Ruleset, forbid_failure, require_metric, require_outcome


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
python3 -m roboeval run path/to/eval_config.json
```

If installed locally, it is also available as:

```bash
roboeval run path/to/eval_config.json
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

## Optional Simulator Integrations

The core SDK stays dependency-free. Simulator adapters are optional extras:

```bash
pip install "roboeval[gymnasium]"
pip install "roboeval[mujoco]"
```

- `roboeval.integrations.gymnasium` wraps any single `gymnasium.Env`.
- `roboeval.integrations.mujoco` wraps raw official MuJoCo `MjModel` / `MjData` XML worlds.

MuJoCo integration uses the official `mujoco` package, not `mujoco-py`. Gymnasium MuJoCo environments should use the Gymnasium adapter.

## Examples

This repo includes demo workloads across four robot domains:

- mobile navigation robot
- robot arm / gripper
- drone inspection robot
- factory welding/process robot

The examples are designed to show the same SDK flow across different state,
action, outcome, metric, and rule shapes.

### Quick Demo

Run the small mobile navigation demo:

```bash
python3 demo.py
```

This runs:

```text
3 policies x 3 scenarios = 9 eval episodes
```

### Generic Robot Demo

Run the larger any-robot demo:

```bash
python3 demo2.py
```

This runs three robot domains:

| Robot domain | Policies | Scenarios | Rules | Eval episodes |
| --- | ---: | ---: | ---: | ---: |
| robot arm / gripper | 4 | 5 | 7 | 20 |
| drone inspection | 4 | 5 | 7 | 20 |
| factory welding/process | 4 | 5 | 7 | 20 |

Generic robot total:

```text
3 robot domains
12 policies
15 scenarios
21 rules
60 eval episodes
```

The generic demo writes reports under:

```text
runs/demo/robot_arm/report.md
runs/demo/drone/report.md
runs/demo/factory/report.md
```

### Config-Based Mobile Demo

Run evals via the CLI with a config file:

```bash
python3 -m roboeval run examples/configs/eval_config.json --output-dir runs/demo_robot
```

This mobile robot demo loads scenarios from Python, JSON, and CSV:

```text
5 policies x 7 scenarios = 35 eval episodes
```

It uses `SuccessCriteria` with checks for:

- goal reached
- collision failure
- stuck failure
- unsafe forward action below the safe distance

### Trained Policy Demo

The trained-policy example adds a real saved PyTorch policy wrapper:

```bash
python3 examples/trained_policy/run_eval.py
```

This runs:

```text
4 policies x 2 scenarios = 8 eval episodes
```

The trained model uses:

```text
1500 synthetic training rows
1200 train rows
300 validation rows
13 input features
5 output actions
~94% validation accuracy
```

### Demo Coverage Summary

Main runnable demo workload:

| Area | Robot domain | Eval episodes |
| --- | --- | ---: |
| `demo.py` | mobile navigation | 9 |
| `examples/demo_robot` | mobile navigation | 35 |
| `examples/trained_policy` | mobile navigation + trained model | 8 |
| `demo2.py` / `examples/generic_robots` | arm, drone, factory | 60 |

Total main demo coverage:

```text
4 robot domains
24 policy entries across demos
27 scenario entries across demos
112 eval episodes
```

## Development

Run tests:

```bash
python3 -m unittest discover -s tests
```

Run the package CLI:

```bash
python3 -m roboeval --help
```

## Roadmap

Near-term:

- improve config ergonomics
- add richer failure explanations
- add more report formats

Later:

- GitHub Actions integration
- hosted eval history and dashboards
- simulator adapters for Isaac, Gazebo, and ROS2 workflows
