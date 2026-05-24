# Robot Policy Eval SDK v1

`robot-policy-eval` is a local Python SDK for repeatable robot policy evaluation.

SDK v1 focuses on the workflow a robotics team can run inside its own repo:

```text
policies + scenarios + success criteria + environment
        |
        v
local eval runner
        |
        v
decision logs + failures + regressions + report
```

## What v1 Does

- Runs multiple policy versions on the same scenarios.
- Captures state, action, next state, outcome, and policy debug info.
- Evaluates built-in and custom success rules.
- Compares every candidate policy against a baseline.
- Flags regressions, improvements, failure cases, and action divergences.
- Generates JSONL, JSON, and Markdown reports.
- Lets users provide their own policy and environment adapters.

## What v1 Does Not Do Yet

- No hosted dashboard.
- No GitHub Actions integration.
- No Isaac, Gazebo, MuJoCo, ROS2, or real robot capture adapter yet.
- No automatic scenario generation yet.
- No public demo example package yet.

The goal of v1 is to prove the local eval loop first. Simulator integrations and hosted tracking can sit on top once the local contract is stable.

## Main SDK Objects

- `EvalRunner`: runs policies against scenarios in an environment.
- `Scenario`: describes an eval case and its initial state.
- `SuccessCriteria`: evaluates pass/fail rules.
- `PolicyAdapter`: normalizes user policies into `decide(state) -> Decision`.
- `EnvironmentAdapter`: normalizes user environments into `reset()` and `step()`.
- `EvalReport`: stores metrics, logs, comparisons, failures, and report output.

