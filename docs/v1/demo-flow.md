# Clean Demo Flow

The v1 demo flow is intentionally small and local.

It shows the product loop without requiring Isaac, Gazebo, MuJoCo, ROS2, or real robot hardware.

Run it from the repo root:

```bash
python3 demo.py
```

Reports are written under `runs/demo/`.

## Flow

```text
baseline policies + policy_v4_trained
        |
        v
same scenarios
        |
        v
same success criteria
        |
        v
EvalRunner
        |
        v
structured report
```

## Demo Policies

- Baseline rules policy: simple and explainable.
- Cautious policy: safer behavior.
- Aggressive policy: faster but may regress.
- Balanced policy: safer improvement.
- `policy_v4_trained`: trained model wrapper that emits actions and model debug info.

## Demo Scenarios

The demo scenarios should include:

- open path
- front obstacle
- narrow gap
- dead end
- noisy low-distance obstacle
- goal misalignment
- stuck-loop case

## What The Report Should Prove

The report should make the eval/debug workflow obvious:

- which policy improved
- which policy regressed
- which scenario failed
- which step caused the failure
- what action the policy took
- what the baseline did differently
- what the trained model's probabilities/logits looked like

Example story:

```text
arm_policy_v2 improved grasp_cube.
arm_policy_v1 failed rule object_grasped.
arm_policy_v2 outcome trace: object_aligned -> object_grasped.
```

That is the product feeling we want: not just pass/fail, but an explanation of what changed.

## Generic Demos

The SDK also includes generic examples for non-navigation robots:

- robot arm/gripper
- drone inspection
- factory welding process

These use `Ruleset` and prove that states/actions/outcomes do not need navigation names.
