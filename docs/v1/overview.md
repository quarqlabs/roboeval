# RoboEval SDK v1

`roboeval` is a local Python SDK for repeatable robot policy evaluation.

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

- Provides a one-command local demo with `python3 demo.py`.
- Runs multiple policy versions on the same scenarios.
- Captures state, action, next state, outcome, and policy debug info.
- Supports generic action objects, including strings, vectors, dicts, arrays, and tensor-like values.
- Evaluates generic `Ruleset` rules and custom success rules.
- Compares every candidate policy against a baseline.
- Flags regressions, improvements, failure cases, and action divergences.
- Generates JSONL, JSON, and Markdown reports.
- Lets users provide their own policy and environment adapters through Python or CLI config.

## What v1 Does Not Do Yet

- No hosted dashboard.
- No GitHub Actions integration.
- No Isaac, Gazebo, MuJoCo, ROS2, or real robot capture adapter yet.
- No automatic scenario generation yet.
- No packaged example distribution yet.

The goal of v1 is to prove the local eval loop first. Simulator integrations and hosted tracking can sit on top once the local contract is stable.

## Main SDK Objects

- `EvalRunner`: runs policies against scenarios in an environment.
- `Scenario`: describes an eval case and its initial state.
- `Ruleset`: recommended generic pass/fail API for any robot domain.
- `SuccessCriteria`: backward-compatible mobile-navigation preset.
- `PolicyAdapter`: normalizes user policies into `decide(state) -> Decision`.
- `EnvironmentAdapter`: normalizes user environments into `reset()` and `step()`.
- `EvalReport`: stores metrics, logs, comparisons, failures, and report output.
