# Design Notes — Gymnasium Integration Spike

Engineering notes captured while implementing this spike. Audience: anyone extending this adapter, or templating the next integration (MuJoCo, PyBullet, Isaac).

---

## 1. Why this folder layout

The integrations subpackage exists so the SDK core can stay zero-dependency. Gymnasium is heavy; not every roboeval user wants it pulled into their base install. By isolating each framework adapter to its own folder under `roboeval/integrations/<name>/` with its own `requirements.txt`, we keep the core install path clean and make the future extras-based distribution (`pip install roboeval[gymnasium]`) a straight folder→extras mapping.

Each integration folder owns six files: `__init__.py`, `adapter.py`, `demo_rollout.py`, `README.md`, `requirements.txt`, `notes.md`. The shape is meant to be copy-pasteable when starting the next adapter.

## 2. The Gymnasium ↔ `StepOutcome` mapping

The SDK's `StepOutcome` already has the fields needed to absorb Gymnasium's 5-tuple step return cleanly. No subclassing, no namespace gymnastics in `next_state`. The mapping baked into `adapter.py`:

| From Gymnasium | To `StepOutcome` |
|----------------|------------------|
| `obs` | `next_state` (wrapped as dict if not already) |
| `reward` | `metrics["reward"]` and `metrics["episode_return"]` (running sum) |
| `terminated`, `truncated` | `terminal = terminated or truncated`, plus structured under `info["gymnasium"]["terminated" / "truncated"]` so rules can distinguish |
| `info` | `info["gymnasium"]["raw_info"]` (allowlistable via `info_keys`) |

This puts numeric values in `metrics` (where `_metric_summary` and `require_metric` rules find them) and raw / debug data in `info` (where it round-trips through reports without polluting the domain-level state).

## 3. Why not subclass `StepOutcome`

Because the existing 8 fields cover everything Gymnasium produces. A subclass would add a type the runner does not understand and create migration risk if `StepOutcome` evolves in core. The duck-typed Protocol means whatever we return passes validation as long as the field shapes are right, so direct construction is both safer and more readable.

If patterns emerge across multiple integrations (e.g. every framework wanting `raw_info` and `terminated/truncated`), that's the trigger to standardize those fields in core. For now, the namespace under `info["gymnasium"]` is the right granularity.

## 4. Why hooks instead of inheritance

Override hooks (six of them) let users customize the SDK ↔ framework boundary without subclassing. They mirror the composition style already established by `CallableEnvironmentAdapter`. The defaults handle the common path (Discrete action, Box observation, sparse-reward end-of-episode envs); overrides cover the long tail. Every default is a public function so user overrides can compose on top rather than reimplementing the whole path.

## 5. `Action = Any` does the heavy lifting for free

The SDK's typing decision to make `Action = Any` means:

- Discrete envs work with `int` actions (CartPole, FrozenLake, Atari-discrete).
- Box envs work with `np.ndarray` actions (Pendulum, MuJoCo continuous control).
- Dict envs work with dict actions.

No coercion required in the adapter. `default_action_from_decision` is literally identity. Users who want a string action vocabulary (e.g. `"left" -> 0`) override that one hook in three lines.

This is the single biggest reason the spike stays small.

## 6. JSON-safety is handled by the SDK

The runner serializes via `roboeval.core.to_serializable`, which already handles `np.ndarray` (`.tolist()`), `np` scalars (`.item()`), dataclasses (`asdict()`), and nested containers. Reusing it means the adapter does not need its own coercion helper.

What we do need: when we ourselves call `json` on values (e.g. when nesting Gymnasium's raw `info` dict under `info["gymnasium"]["raw_info"]`), we still pass through `to_serializable` to make sure heavy tensors do not break the report writer. That is the one explicit `to_serializable` call inside the adapter.

`coerce_observations` is exposed as an opt-in for users who want JSON-safe state passed to their policies too — handy for debug prints, but off by default to keep raw observations available for ML policies that expect arrays.

## 7. Vector envs are explicitly refused

`gym.vector.VectorEnv` returns batched arrays from a single `step()` call. The SDK's runner is single-episode; passing a vector env would cause confusing slicing bugs. `__post_init__` raises `NotImplementedError` with a clear next-step message ("pass `env.envs[0]`, or use `gym.make_vec(..., num_envs=1)` and unwrap").

When the SDK gains a batched runner, the only change needed here is dropping that check. The rest of the adapter is single-env by construction and would need no other modification.

## 8. `TimeLimit` wrapper vs `scenario.max_steps`

Two truncation sources coexist when wrapping a Gymnasium env:

- The env's own `TimeLimit` wrapper (default for most registered envs) flips `truncated=True` at its configured limit.
- The roboeval `Scenario.max_steps` ends the outer eval loop in the runner.

Whichever comes first wins. Both produce sane behavior:

- If `TimeLimit` fires first, the adapter reports `terminal=True` with `events=["episode_truncated"]`.
- If `scenario.max_steps` fires first, the runner just stops the loop; the last step is non-terminal.

Recommendation: treat `scenario.max_steps` as authoritative for the eval budget. If the env's `TimeLimit` is lower than `scenario.max_steps`, the eval will simply end early via the env. If you want longer episodes than the env's default, wrap with `gym.wrappers.TimeLimit(env, max_episode_steps=...)` or use `env.spec.max_episode_steps = ...`.

## 9. The default outcome mapping has a known sharp edge

`default_outcome_from_step` classifies a terminal step as `terminated_success` when `reward > 0`. For CartPole-v1 this misfires: every step (including the terminal one where the pole falls) gives `reward = 1.0`, so falling registers as `terminated_success`. The total `episode_return` is what tells you whether the policy actually did well.

Users with envs where the terminal reward does not signal success should override `outcome_from_step`. The default is meant as a starting point, not a universal classifier. The docstring on the default function says as much.

The alternative — using the running episode return as the success signal — couples the default to a threshold that varies per env, which is worse. The right way to score CartPole is via `require_metric("episode_return", ">=", 195.0)` in a `Ruleset`, which is exactly what the rule API is for.

## 10. `gym` vs `gymnasium`

The adapter targets `gymnasium>=0.29` (the version that stabilized the 5-tuple `step()` return: `obs, reward, terminated, truncated, info`). The legacy `gym` package returns a 4-tuple (`obs, reward, done, info`) and is not supported. If a user passes a legacy `gym.Env`, the call will fail at the first `step()` unpack — there is no explicit refusal because checking the version string is brittle and `gym` has been unmaintained for years.

If we ever need to support `gym` envs, the right path is a separate adapter (`LegacyGymEnvironmentAdapter`) that explicitly handles the 4-tuple and translates `done` into `(terminated, truncated)`. Cleaner than dispatching inside the same class.

## 11. Future integrations follow this template

`integrations/mujoco/`, `integrations/pybullet/`, `integrations/isaac/` should mirror this folder's shape. The big translations to think about per framework:

- MuJoCo (raw, not through Gymnasium): step is `mj_step(model, data)` with no reward/done abstractions. The adapter has to define what success means per task.
- PyBullet: similar to MuJoCo — manual stepping, manual success criteria. Wraps cleanly via the `CallableEnvironmentAdapter` pattern if there's no central API.
- Isaac Sim / Isaac Lab: provides `gym.Env`-compatible envs via Isaac Lab. The Gymnasium adapter may work directly with minor hook overrides; needs validation.

In all cases, namespace framework-specific extras under `info["<framework>"]` to keep downstream consumers framework-agnostic.

## 12. Follow-up items for the team

1. **`pyproject.toml` extras line.** Add to `[project.optional-dependencies]`:
   ```toml
   gymnasium = ["gymnasium>=0.29"]
   ```
   So `pip install roboeval[gymnasium]` installs both the core and the underlying dependency. The integration folder is already in the SDK's package-find glob (`include = ["roboeval*"]`), so the import path works without further changes. Held out of this spike per the "don't modify pyproject.toml" constraint.

2. **`artifacts` field.** The adapter does not currently populate `StepOutcome.artifacts`. A natural extension is capturing rendered frames when `env.render_mode == "rgb_array"`, stored either as numpy arrays or PNG paths. Best surfaced via an `artifacts_from_step` hook.

3. **`VectorEnv` support.** Refused at construction today. When the SDK runner gains batched execution, drop the `__post_init__` check and add per-env state tracking for the running `episode_return`.

4. **Multiple integrations sharing a base.** When the next adapter ships (MuJoCo or PyBullet), evaluate whether common patterns warrant a shared `_BaseIntegrationAdapter` or a shared `info_keys` / `events_from_step` helper module. Premature today.

5. **Seed re-seeding semantics.** Gymnasium re-seeds the underlying RNG only when `seed` is non-None on `reset()`. The adapter forwards whatever `seed_from_scenario` returns, so a scenario with no `seed` lets the env continue its current RNG state across resets — which can produce non-reproducible episodes. Document explicitly that scenarios should always specify a seed for reproducibility.
