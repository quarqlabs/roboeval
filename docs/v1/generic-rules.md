# Generic Rules

`Ruleset` is the recommended SDK v1 API for success and failure logic.

It avoids assumptions like `move_forward`, `front_distance`, `collision`, or `goal_reached`.

## Mobile Robot

```python
from roboeval import Ruleset, forbid_failure, max_steps, require_outcome


ruleset = Ruleset([
    require_outcome("goal_reached"),
    forbid_failure("collision"),
    max_steps(50),
])
```

## Robot Arm

```python
from roboeval import Ruleset, forbid_failure, require_metric, require_outcome


ruleset = Ruleset([
    require_outcome("object_grasped"),
    forbid_failure("dropped_object"),
    require_metric("grip_force", "<=", 0.9),
])
```

## Drone Inspection

```python
ruleset = Ruleset([
    require_outcome("waypoint_inspected"),
    forbid_failure("no_fly_zone_violation"),
    require_metric("battery_used", "<=", 30),
])
```

## Factory Robot

```python
ruleset = Ruleset([
    require_outcome("weld_completed"),
    forbid_failure("overheat"),
    require_metric("temperature", "<=", 80),
])
```

`SuccessCriteria()` still exists, but it is a convenience preset for the mobile-navigation demo.
