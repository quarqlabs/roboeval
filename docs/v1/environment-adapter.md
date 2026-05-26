# Environment Adapter

The environment adapter is the bridge between the SDK and a user's robot world.

That world can be our demo environment, a Python simulator, a wrapped robotics stack, or later a real Isaac/Gazebo/MuJoCo/ROS2 integration.

## Interface

An environment must implement:

```python
from typing import Any

from roboeval import Scenario, StepOutcome


class MyEnvironment:
    name = "my_environment"

    def reset(self, scenario: Scenario) -> dict:
        ...

    def step(self, action: Any, scenario: Scenario) -> StepOutcome:
        ...
```

## `reset(scenario)`

`reset()` prepares the environment for one scenario and returns the first state.

The state is a plain dictionary because robot teams have different observation formats:

```python
{
    "object_pose": "center",
    "gripper_closed": False,
    "has_object": False,
}
```

## `step(action, scenario)`

`step()` applies one policy action and returns:

```python
StepOutcome(
    next_state={...},
    outcome="object_grasped",
    failure_label="",
    terminal=True,
    metrics={"grip_force": 0.72},
    events=["gripper_closed"],
)
```

The `action` can be any policy output shape: a string command, integer class,
list of joint targets, numpy-like array, tensor-like object, or dict. The SDK
does not convert it before calling your environment.

Fields:

- `next_state`: the observation after the action.
- `outcome`: what happened at this step, such as `object_grasped`, `waypoint_inspected`, or `weld_completed`.
- `failure_label`: empty when there is no failure, otherwise a machine-readable failure label.
- `terminal`: `True` when the episode should stop.
- `metrics`: optional numeric values for report summaries and metric rules.
- `events`: optional event names for custom rules and debugging.

## Callable Adapter

If a team already has reset and step functions, they can wrap them without
creating a class:

```python
from roboeval import CallableEnvironmentAdapter


env = CallableEnvironmentAdapter(
    reset_fn=my_reset,
    step_fn=my_step,
    name="my_existing_sim",
)
```

## CLI Environment Config

The CLI can import a custom environment from config:

```json
{
  "environment": {
    "path": "my_robot.envs:make_eval_environment",
    "kwargs": {"seed": 7}
  }
}
```

The imported object can be an environment instance, an environment class, or a
factory function. The final object must provide `reset()` and `step()`.

## Why This Exists

The SDK should not force users into our demo robot. The adapter contract lets them keep their own environment and only translate their world into the SDK's small step format.
