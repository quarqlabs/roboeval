# Isaac Lab Integration

Wrap any single-env [Isaac Lab](https://github.com/isaac-sim/IsaacLab) environment as a roboeval `EnvironmentAdapter` so existing policies can be evaluated against Isaac Lab tasks through `EvalRunner` — no changes to the SDK core.

## Important — what runs where

Isaac Sim + Isaac Lab require **NVIDIA GPU + Linux or Windows**. They will not run on macOS.

The recommended workflow is:

1. **Write adapter / policy code locally** (Mac is fine; the adapter file imports torch but doesn't import Isaac Lab at the module top)
2. **Run mocked unit tests locally** (this directory's tests use mock Isaac envs, no Isaac install needed)
3. **Validate against real Isaac on a cloud GPU** (RunPod, Lambda Labs, AWS p3, GCP T4, internal Linux workstation) — see "Cloud GPU workflow" below

## Quick start (once Isaac Lab is installed)

```python
import gymnasium as gym

from roboeval import EvalRunner, Ruleset, Scenario, require_metric, require_outcome
from roboeval.integrations.isaac import IsaacEnvironmentAdapter


def naive_policy(state):
    obs = state.get("policy") or state.get("observation")
    pole_angle = float(obs[2])
    return {"action": 0 if pole_angle < 0 else 1}


env = gym.make("Isaac-Cartpole-Direct-v0", num_envs=1)
adapter = IsaacEnvironmentAdapter(env=env, name="isaac_cartpole")

report = EvalRunner(
    policies=[naive_policy],
    scenarios=[Scenario("smoke", {"seed": 0}, max_steps=200)],
    ruleset=Ruleset([
        require_outcome("terminated_success"),
        require_metric("episode_return", ">=", 50.0),
    ]),
    baseline_policy="naive_policy",
    environment=adapter,
).run()

report.save("runs/isaac_smoke")
```

For a manual rollout (prints step-by-step output):

```bash
python -m roboeval.integrations.isaac.demo_rollout
```

## How Isaac Lab maps to the SDK

Isaac Lab envs are gymnasium-compatible but always vectorized. The adapter handles three differences from vanilla gym envs:

| Concern | What Isaac Lab does | What the adapter does |
|---------|---------------------|------------------------|
| **Batch dimension** | Even `num_envs=1` envs return tensors with shape `(1, ...)` | Slices `batch_index=0` (configurable) to expose single-episode semantics |
| **GPU tensors** | Observations are `torch.Tensor` on `cuda:0` | Coerces to CPU numpy via `tensor_to_numpy()` before they hit the runner's JSON writer |
| **Tensor actions** | `env.step()` expects `torch.Tensor` on the env's device, shape `(num_envs, action_dim)` | Wraps user actions (numpy arrays, scalars, lists) in a batched torch tensor on the right device |

Plus the standard six translation hooks (matching `GymnasiumEnvironmentAdapter`):

| Hook | Default behavior | Override when... |
|------|------------------|-------------------|
| `observation_to_state` | If dict → pass through; else wrap as `{"observation": obs}` | User wants to rename/restructure keys |
| `action_from_decision` | Identity | User wants a custom action vocabulary (`"left"` → `0`) |
| `outcome_from_step` | `terminated && reward > 0` → success; else failure / truncated / progress | Env uses `info["is_success"]` or task-specific signal |
| `events_from_step` | Emit `episode_terminated`, `episode_truncated`, `reward_negative` | Add domain-specific tags |
| `seed_from_scenario` | Read `scenario.initial_state["seed"]`, then `metadata["seed"]` | Custom seed routing |
| `options_from_scenario` | Read `scenario.metadata["reset_options"]` | Env supports task-specific reset options |

## The `StepOutcome` shape produced

```python
StepOutcome(
    next_state={"policy": ndarray([cart_pos, cart_vel, pole_angle, pole_vel])},
    outcome="progress",                       # or terminated_success / failure / truncated
    failure_label="",                         # populated on failure
    terminal=False,                           # True when terminated or truncated
    metrics={
        "reward": 1.0,                        # this step's reward
        "episode_return": 17.0,               # running sum since reset
    },
    events=["episode_terminated"],            # tags for rule filtering
    info={
        "isaac": {
            "terminated": False,
            "truncated": False,
            "raw_info": {...},                # whatever info dict the env emits
            "batch_index": 0,                 # which env slice we're reading
        }
    },
)
```

## What this integration does NOT support (yet)

- **`num_envs > 1` parallelism.** The SDK runner is single-episode. The adapter accepts `num_envs > 1` but warns and reads only `batch_index=0`. Other envs run but are ignored. For true vector eval, pass `num_envs=1` to `gym.make()`; vector-eval throughput will come when the SDK runner gains batched execution.
- **Render-mode frame capture.** `StepOutcome.artifacts` is empty. A follow-up can add an `artifacts_from_step` hook for `render_mode="rgb_array"` frames.
- **Task-specific success defaults.** Each Isaac task has different success conventions (`info["is_success"]`, terminal reward shaping, etc.). The default `outcome_from_step` is a starting point; override per task.

## Cloud GPU workflow

When you don't have a Linux+NVIDIA workstation handy:

### Option A — RunPod (recommended for spikes)

1. Create a RunPod account, billing set up.
2. Spin up a pod with an A40 / RTX 4090 / A100 (whatever's cheap; A40 around $0.40/hr).
3. Choose the "Isaac Sim" template if available, or start from `runpod/pytorch:2.x-py3.10-cuda12.1`.
4. SSH in or open the in-browser terminal.
5. Install Isaac Sim:
   ```bash
   pip install isaacsim==4.5.* --extra-index-url https://pypi.nvidia.com
   ```
6. Install Isaac Lab:
   ```bash
   git clone https://github.com/isaac-sim/IsaacLab.git
   cd IsaacLab && ./isaaclab.sh --install
   ```
7. Install roboeval from your branch:
   ```bash
   git clone https://github.com/quarqlabs/roboeval.git
   cd roboeval && git checkout spike/isaac-integration
   pip install -e .
   ```
8. Run the smoke test:
   ```bash
   python -m roboeval.integrations.isaac.demo_rollout
   ```

Expect ~5–20 min for the install, then sub-second per-step in sim. Total spike validation cost: a few dollars.

### Option B — Lambda Labs / AWS / GCP

Same shape; pick the lowest-cost GPU instance with Linux + CUDA. Persistence of the disk matters if you'll iterate over multiple sessions.

### Option C — Internal Linux workstation

If the team has a Linux box with an NVIDIA GPU, that's the most cost-effective dev experience. Just install Isaac Sim + Isaac Lab + roboeval and iterate.

## Known sharp edges

- **CartPole `outcome_from_step` defaults are imperfect.** Isaac-Cartpole-Direct gives `reward = +1` every step including the terminal step where the pole falls, so the default classifier reports `terminated_success` even when the pole fell. Score with `require_metric("episode_return", ">=", threshold)` rather than `require_outcome("terminated_success")` for these envs, OR override `outcome_from_step` with the env-specific success detector. (Same sharp edge applies to vanilla Gymnasium CartPole; documented in the Gymnasium integration too.)
- **`env.reset(options=...)` not universally accepted.** Some Isaac envs don't accept the `options` kwarg. The adapter catches the `TypeError` and falls back to `env.reset(seed=...)` only.
- **`env.spec.max_episode_steps` may conflict with `scenario.max_steps`.** Whichever is smaller wins. Treat `scenario.max_steps` as authoritative; if the env's TimeLimit fires first, episodes are reported as `truncated`.
- **GPU memory.** Isaac Sim is heavy. A40 / RTX 4090 / A100 all have enough memory for single-env CartPole; manipulation tasks may need bigger GPUs or `num_envs=1` discipline.

See `notes.md` for the full design rationale and follow-up items.
