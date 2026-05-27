# Gymnasium Integration

Wrap any [Gymnasium](https://gymnasium.farama.org/) environment as a roboeval `EnvironmentAdapter` so existing policies can be evaluated against Gymnasium envs through `EvalRunner` — no changes to the core SDK.

## Install

This integration is shipped inside the `roboeval` package but Gymnasium itself is an optional dependency.

```bash
pip install gymnasium>=0.29
```

(See `requirements.txt` in this folder. A future SDK release may expose this as `pip install roboeval[gymnasium]`.)

## Quick start

```python
import gymnasium as gym

from roboeval import EvalRunner, Ruleset, Scenario, require_metric, require_outcome
from roboeval.integrations.gymnasium import GymnasiumEnvironmentAdapter


def naive_policy(state):
    pole_angle = float(state["observation"][2])
    return {"action": 0 if pole_angle < 0 else 1, "debug_info": {"version": "naive"}}


env = gym.make("CartPole-v1")
adapter = GymnasiumEnvironmentAdapter(env=env, name="cartpole_v1")

ruleset = Ruleset([
    require_outcome("terminated_success"),
    require_metric("episode_return", ">=", 50.0),
])

report = EvalRunner(
    policies=[naive_policy],
    scenarios=[Scenario("cartpole_smoke", {"seed": 0}, max_steps=200)],
    ruleset=ruleset,
    baseline_policy="naive_policy",
    environment=adapter,
).run()

report.save("runs/gym_smoke")
```

For a runnable end-to-end demo:

```bash
python -m roboeval.integrations.gymnasium.demo_rollout
```

## How Gymnasium maps to the SDK

Gymnasium has 5-tuple `step()` returns; roboeval has the 8-field `StepOutcome`. The adapter performs six translations:

| Gymnasium | roboeval `StepOutcome` | Default behavior |
|-----------|------------------------|------------------|
| `obs` from `env.reset()` / `env.step()` | `next_state: dict` | If `obs` is a dict, pass through; otherwise wrap as `{"observation": obs}`. Values pass through raw (the SDK's `to_serializable` handles JSON coercion at report time). Override via `observation_to_state` hook. |
| `decision.action` (`Action = Any`) | passed straight into `env.step(action)` | Identity — the SDK already accepts any action type. Override via `action_from_decision` hook (e.g. for string action vocabularies). |
| `reward: float` | `metrics["reward"]` + `metrics["episode_return"]` (running sum) | Reward is reported per-step; episode return is reset on each `adapter.reset()`. |
| `terminated: bool` + `truncated: bool` | `terminal = terminated or truncated`, plus `info["gymnasium"]["terminated"]` and `info["gymnasium"]["truncated"]` | Distinguishes the two terminations in `info` so rules / reports can tell timeouts from natural episode ends. |
| `info: dict` | `info["gymnasium"]["raw_info"]` | Pass-through. Use the `info_keys` allowlist to filter heavy keys out. |
| Derived `(outcome, failure_label)` | `outcome` and `failure_label` fields | Default mapping: `terminated and reward > 0` → `terminated_success`; `terminated` → `terminated_failure`; `truncated` → `truncated`; else `progress`. Override via `outcome_from_step` hook for env-specific semantics. |
| `Scenario` | `env.reset(seed=..., options=...)` | `seed` from `scenario.initial_state["seed"]` or `scenario.metadata["seed"]`. `options` from `scenario.metadata["reset_options"]`. Override via `seed_from_scenario` / `options_from_scenario` hooks. |

Additionally, `events` ride along on each step with default tags: `episode_terminated`, `episode_truncated`, `reward_negative`. Customize via `events_from_step`.

## The `StepOutcome` shape produced

```python
StepOutcome(
    next_state={"observation": ...},                # or dict obs passthrough
    outcome="progress",                             # or terminated_success / terminated_failure / truncated
    failure_label="",                               # populated on failure
    terminal=False,                                 # True when terminated or truncated
    metrics={
        "reward": 1.0,                              # this step's reward
        "episode_return": 17.0,                     # running sum since reset
    },
    events=["episode_terminated"],                  # tags for rule filtering
    info={
        "gymnasium": {
            "terminated": False,
            "truncated": False,
            "raw_info": {...},                      # whatever the env's info dict carries
        }
    },
)
```

## Customizing the boundary

All six translations are public callables — override any subset:

```python
from roboeval.integrations.gymnasium import (
    GymnasiumEnvironmentAdapter,
    default_outcome_from_step,
)


def my_outcome(reward, terminated, truncated, info):
    # Manipulation envs often expose info["is_success"]
    if info.get("is_success"):
        return ("goal_reached", "")
    return default_outcome_from_step(reward, terminated, truncated, info)


adapter = GymnasiumEnvironmentAdapter(
    env=gym.make("FetchPickAndPlace-v3"),
    name="fetch_pick_place",
    outcome_from_step=my_outcome,
    info_keys=["is_success", "TimeLimit.truncated"],   # allowlist heavy keys
)
```

A string-action vocabulary on a Discrete env:

```python
ACTION_TABLE = {"left": 0, "right": 1}

adapter = GymnasiumEnvironmentAdapter(
    env=gym.make("CartPole-v1"),
    action_from_decision=lambda a: ACTION_TABLE[a] if isinstance(a, str) else a,
)
```

Continuous-action envs (Box action space) work without any hook override because `Action = Any` already accepts `np.ndarray`:

```python
import numpy as np

env = gym.make("Pendulum-v1")
adapter = GymnasiumEnvironmentAdapter(env=env)

def my_policy(state):
    # state["observation"] is the (cos, sin, vel) tuple
    return {"action": np.array([0.0], dtype=np.float32)}
```

## Configuration reference

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `env` | `gymnasium.Env` | required | The wrapped env. `VectorEnv` is explicitly refused. |
| `name` | `str` | `"gymnasium_env"` | Display name in reports (read by the runner). |
| `observation_to_state` | `Callable` \| `None` | `default_observation_to_state` | Obs → `State` dict. |
| `action_from_decision` | `Callable` \| `None` | `default_action_from_decision` | Decision action → Gymnasium action. |
| `outcome_from_step` | `Callable` \| `None` | `default_outcome_from_step` | Step result → `(outcome, failure_label)`. |
| `events_from_step` | `Callable` \| `None` | `default_events_from_step` | Step result → event tag list. |
| `seed_from_scenario` | `Callable` \| `None` | `default_seed_from_scenario` | Scenario → `env.reset(seed=...)`. |
| `options_from_scenario` | `Callable` \| `None` | `default_options_from_scenario` | Scenario → `env.reset(options=...)`. |
| `info_keys` | `list[str]` \| `None` | `None` (pass everything) | Allowlist for keys passed through from gym's `info`. Useful to drop heavy tensors. |
| `coerce_observations` | `bool` | `False` | When `True`, pre-coerce observations to JSON-safe Python types via `to_serializable`. Off by default because the runner already coerces at report-write time. |

## What this spike does not cover

- **`gym.vector.VectorEnv`** — refused at construction. The single-episode `EvalRunner` cannot consume batched step returns. When a batched runner ships, drop the `__post_init__` check.
- **Render frames as artifacts** — `render_mode="rgb_array"` frames could populate `StepOutcome.artifacts`, but this spike does not. A follow-up can add an `artifacts_from_step` hook.
- **Legacy `gym.Env`** — only `gymnasium>=0.29` is supported (5-tuple `step()` return). The pre-`gymnasium` `gym` package returns a 4-tuple (`done`) and is not handled.

See `notes.md` for the full design rationale and follow-up items.
