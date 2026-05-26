# Plug In Your Own Env In 20 Lines

This is the smallest shape of a custom environment and policy.

```python
from roboeval import EvalRunner, Ruleset, Scenario, StepOutcome, forbid_failure, require_outcome


class MyEnv:
    name = "my_env"

    def reset(self, scenario):
        return dict(scenario.initial_state)

    def step(self, action, scenario):
        next_state = {**scenario.initial_state, "previous_action": action, "step_count": 1}
        return StepOutcome(next_state, "object_grasped", "", True)


def my_policy(state):
    return {"action": [0.0, 0.2, 0.7], "model_version": "my_policy"}


scenario = Scenario("grasp_object", {"object_pose": "center"}, max_steps=1)
ruleset = Ruleset([require_outcome("object_grasped"), forbid_failure("dropped_object")])
report = EvalRunner([my_policy], [scenario], ruleset=ruleset, baseline_policy="my_policy", environment=MyEnv()).run()
report.save("runs/latest")
```

In a real robot repo, `MyEnv.step()` would call the simulator or robot wrapper and translate the result into `StepOutcome`. The action can be a string, class id, vector, dict, or tensor-like object; the SDK passes it through unchanged.
