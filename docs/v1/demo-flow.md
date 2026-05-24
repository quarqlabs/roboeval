# Clean Demo Flow

The v1 demo flow is intentionally small and local.

It shows the product loop without requiring Isaac, Gazebo, MuJoCo, ROS2, or real robot hardware.

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
policy_v4_trained improved dead_end_reverse_needed.
policy_v1_cautious moved forward unsafely.
policy_v4_trained reversed, escaped, then reached goal.
```

That is the product feeling we want: not just pass/fail, but an explanation of what changed.

