# MuJoCo Integration

This spike wraps raw MuJoCo Python bindings into RoboEval's `EnvironmentAdapter` shape.

It targets the official `mujoco` package:

```bash
pip install "roboeval[mujoco]"
```

This is not `mujoco-py`, and it does not require a Gymnasium wrapper. If a team already uses Gymnasium MuJoCo environments, they can use `roboeval.integrations.gymnasium` instead.

## What It Wraps

```python
import mujoco

from roboeval.integrations.mujoco import MuJoCoEnvironmentAdapter

model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)

env = MuJoCoEnvironmentAdapter(model=model, data=data)
```

or:

```python
env = MuJoCoEnvironmentAdapter.from_xml_path("robot.xml")
env = MuJoCoEnvironmentAdapter.from_xml_string(xml_string)
```

## Mapping

| Raw MuJoCo | RoboEval |
| --- | --- |
| `mj_resetData(model, data)` | `reset(scenario)` |
| `scenario.initial_state["qpos"]` | `data.qpos` |
| `scenario.initial_state["qvel"]` | `data.qvel` |
| `scenario.initial_state["ctrl"]` | `data.ctrl` |
| policy action | `data.ctrl` |
| `mj_step(model, data)` | `step(action, scenario)` |
| `data.qpos`, `data.qvel`, `data.ctrl`, `data.time` | `StepOutcome.next_state` |
| task hook | `StepOutcome.outcome`, `failure_label`, `terminal` |
| norms/time hook | `StepOutcome.metrics` |
| MuJoCo metadata | `StepOutcome.info["mujoco"]` |

## Custom Task Semantics

Raw MuJoCo does not know what "success" means for your robot. The adapter exposes hooks so teams can define their own task semantics without changing RoboEval core files:

```python
def outcome_from_step(model, data, scenario, step_index):
    target = scenario.metadata["target_qpos"]
    if abs(float(data.qpos[0]) - target) < 0.03:
        return ("goal_reached", "", True)
    return ("progress", "", False)


env = MuJoCoEnvironmentAdapter.from_xml_path(
    "point_mass.xml",
    outcome_from_step=outcome_from_step,
)
```

Available hooks:

- `observation_from_data(model, data, scenario) -> dict`
- `action_to_ctrl(action, model, data, scenario) -> None`
- `outcome_from_step(model, data, scenario, step_index) -> tuple[str, str, bool]`
- `metrics_from_step(model, data, scenario, step_index) -> dict`
- `events_from_step(model, data, scenario, step_index) -> list[str]`
- `info_from_step(model, data, scenario, step_index) -> dict`
- `reset_from_scenario(scenario, model, data) -> None`

## Demo

```bash
python -m roboeval.integrations.mujoco.demo_rollout
```

The demo loads `assets/point_mass.xml`, applies a tiny controller, and stops when the point mass reaches the target.

## Scope

Included:

- raw `mujoco.MjModel` / `mujoco.MjData` adapter
- XML path and XML string constructors
- qpos/qvel/ctrl reset from `Scenario`
- action-to-control mapping
- hook-based outcomes, metrics, events, and info
- a tiny point-mass XML demo

Not included yet:

- video/render artifact capture
- batched/vector simulation
- named joint/sensor helper APIs
- contact-specific rule helpers
- Gymnasium MuJoCo environments, because those use the Gymnasium adapter
