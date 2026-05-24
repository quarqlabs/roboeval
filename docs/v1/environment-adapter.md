# Environment Adapter

The environment adapter is the bridge between the SDK and a user's robot world.

That world can be our demo environment, a Python simulator, a wrapped robotics stack, or later a real Isaac/Gazebo/MuJoCo/ROS2 integration.

## Interface

An environment must implement:

```python
from robot_policy_eval import Scenario, StepOutcome


class MyEnvironment:
    name = "my_environment"

    def reset(self, scenario: Scenario) -> dict:
        ...

    def step(self, action: str, scenario: Scenario) -> StepOutcome:
        ...
```

## `reset(scenario)`

`reset()` prepares the environment for one scenario and returns the first state.

The state is a plain dictionary because robot teams have different observation formats:

```python
{
    "front_distance": 80,
    "left_distance": 45,
    "right_distance": 45,
    "goal_direction": "forward",
    "previous_action": "none",
    "step_count": 0,
}
```

## `step(action, scenario)`

`step()` applies one policy action and returns:

```python
StepOutcome(
    next_state={...},
    outcome="progress",
    failure_label="",
    terminal=False,
)
```

Fields:

- `next_state`: the observation after the action.
- `outcome`: what happened at this step, such as `progress`, `goal_reached`, `collision`, or `stuck`.
- `failure_label`: empty when there is no failure, otherwise a machine-readable failure label.
- `terminal`: `True` when the episode should stop.

## Why This Exists

The SDK should not force users into our demo robot. The adapter contract lets them keep their own environment and only translate their world into the SDK's small step format.

