# Plug In Your Own Env In 20 Lines

This is the smallest shape of a custom environment and policy.

```python
from robot_policy_eval import EvalRunner, Scenario, StepOutcome, SuccessCriteria


class MyEnv:
    name = "my_env"

    def reset(self, scenario):
        return dict(scenario.initial_state)

    def step(self, action, scenario):
        next_state = {**scenario.initial_state, "previous_action": action, "step_count": 1}
        return StepOutcome(next_state, "goal_reached", "", True)


def my_policy(state):
    return {"action": "move_forward", "model_version": "my_policy"}


scenario = Scenario("smoke_goal", {"front_distance": 80}, max_steps=1)
report = EvalRunner([my_policy], [scenario], SuccessCriteria(), "my_policy", MyEnv()).run()
report.save("runs/latest")
```

In a real robot repo, `MyEnv.step()` would call the simulator or robot wrapper and translate the result into `StepOutcome`.

