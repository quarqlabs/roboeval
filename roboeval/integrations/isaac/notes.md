# Design Notes — Isaac Lab Integration Spike

Engineering notes captured while implementing this spike. Audience: anyone extending this adapter or templating the next integration (MuJoCo direct, PyBullet, ROS 2, real robots).

---

## 1. Why a separate adapter from `GymnasiumEnvironmentAdapter`?

Isaac Lab envs ARE gymnasium-compatible at the API level (they subclass `gym.vector.VectorEnv`). It would be possible to extend `GymnasiumEnvironmentAdapter` and override hooks. We chose a separate class for three reasons:

1. **Different fundamental assumptions.** Gymnasium envs are usually single-env; Isaac envs are always vectorized. Embedding the batch-slicing logic in `GymnasiumEnvironmentAdapter` would either pollute it (`if isinstance(env, VectorEnv): ...` branches) or force a complicated inheritance hierarchy.

2. **Different tensor lifecycle.** The Gymnasium adapter assumes observations are `numpy.ndarray` or JSON-safe types. The Isaac adapter handles PyTorch tensors on GPU explicitly, including device routing for actions.

3. **Different failure modes for the user.** The Gymnasium adapter refuses `VectorEnv` outright. The Isaac adapter must accept it (because all Isaac envs are vector envs). A user error like "wrong tensor dtype" produces different debugging guidance.

A future refactor could share helpers between the two via a base class once a third integration (MuJoCo via dm_control, say) reveals which patterns actually generalize. Premature today.

## 2. The batch-index trick

Isaac Lab envs return tensors with a batch dimension even at `num_envs=1`:

```python
obs.shape       # (1, obs_dim)
reward.shape    # (1,)
terminated.shape  # (1,)
```

The adapter takes `batch_index=0` slices throughout. The runner sees scalar reward, single-vector observations, single boolean done — matching the single-episode runner's expectations.

Why expose `batch_index` as a constructor parameter rather than always using 0? Two reasons:

- **Defensive programming.** If a user passes `num_envs > 1` by accident, they can at least pick *which* env they care about rather than getting a silent crash.
- **Forward compatibility.** When the SDK runner gains batched execution, we can ship a `BatchIsaacEnvironmentAdapter` that emits N rollouts in parallel. The `batch_index` parameter becomes a list of indices to track.

For v1, the standard pattern is `num_envs=1` + `batch_index=0` + a warning if `num_envs > 1`.

## 3. Tensor-to-numpy coercion happens AT THE ADAPTER BOUNDARY

A design principle: the runner should never see tensors. The adapter does the coercion so:

- Reports (`decision_logs.jsonl`, `episode_results.json`) are JSON-safe without depending on `to_serializable` knowing about torch
- The policy's `decide(state)` sees plain numpy / Python types, not GPU tensors
- The runner's `dict(state)` copy works without surprise

The cost is one GPU→CPU sync per step. For Isaac Sim at single-env throughput (~30–60 Hz for Cartpole), this is negligible. For real-time control loops at 1000+ Hz this would matter — but real-time control is out of scope for the runner.

There's a public `tensor_to_numpy()` function in `adapter.py` so users overriding `observation_to_state` can call it consistently.

## 4. Action conversion handles three input shapes

Policies might return:
- Python scalars (`0`, `1.5`)
- 1-D numpy arrays / lists (`[0.5, -0.3]`)
- 2-D arrays already batched (`[[0.5, -0.3]]`)

The adapter's `_to_batched_torch_action` normalizes all three to the expected `(num_envs, action_dim)` torch tensor on the env's device. Device routing is conservative: read `env.device`, fall back to `env.sim_device`, fall back to the action's existing device, fall back to `cuda` if available, else `cpu`.

We expose a hook (`action_from_decision`) for users to customize the user-action → policy-action translation (e.g. discrete vocabulary `"left"` → `0`). The torch conversion happens after the hook.

## 5. The `info["isaac"]` namespace mirrors the Gymnasium adapter's `info["gymnasium"]` pattern

Same shape:

```python
"info": {
    "isaac": {
        "terminated": bool,
        "truncated": bool,
        "raw_info": {...},      # filtered via info_keys allowlist
        "batch_index": int,
    }
}
```

This consistency means downstream consumers (the dashboard, future agents) can find sim-specific raw data at a predictable location. The `batch_index` field is Isaac-specific; it's useful when debugging multi-env scenarios.

## 6. What we deliberately did NOT do

- **Vector eval support.** Out of scope for v1. The SDK runner is single-episode. When that changes, a `BatchIsaacEnvironmentAdapter` ships as a separate class.
- **`info["is_success"]` auto-detection.** Many Isaac tasks expose success this way (Franka Cabinet, Lift Cube, etc.). Tempting to default to it, but every task has different keys (`success`, `is_success`, `task_success`). We'd ship false positives for tasks that use those keys with different semantics. Instead, document the override pattern.
- **Per-task hook presets.** A `presets/franka_cabinet.py` module with the right `outcome_from_step` for that task would be useful. Out of scope until we know which tasks users actually want.
- **Render-mode frame capture.** `StepOutcome.artifacts` left empty. Should be an `artifacts_from_step` hook in v2.
- **Determinism guarantees.** Isaac Sim has its own determinism story (depends on physics version, seed, GPU model). We pass through the seed; we don't promise byte-exact replay.

## 7. Why `num_envs > 1` warns instead of refusing

Earlier draft refused `num_envs > 1` outright. We softened to a warning because:

- A user debugging a vector-env-only Isaac task ("Why is my parallel eval broken?") wants the adapter to do *something*, not crash
- The Gymnasium adapter already refuses `gym.vector.VectorEnv` outright; the Isaac case needs different behavior (since Isaac envs are always VectorEnvs)
- Reading `batch_index=0` of an N-env batch is correct behavior, just throughput-wasteful

The warning is loud enough that users notice ("Other 99 envs run but are ignored") and the docs explain the right fix (`num_envs=1`).

## 8. Why `env.reset(options=...)` is wrapped in try/except

Some Isaac envs accept the `options` kwarg per the standard gymnasium API; some pre-date that API change. To support both:

```python
try:
    obs, info = self.env.reset(seed=seed, options=options)
except TypeError:
    obs, info = self.env.reset(seed=seed)
```

This is mildly defensive. The alternative (sniffing the signature) is brittle. The cost (one extra try/except per reset) is irrelevant since reset is once per scenario.

## 9. Why mocked tests instead of real-Isaac tests

Two reasons:

1. **No Isaac on Mac.** Running real Isaac requires GPU + Linux. Tests should be runnable on the dev machine (which is a Mac).
2. **CI cost.** Running real Isaac in CI is impractical for an open-source SDK — requires GPU runners, which are paid.

The mock-based tests validate:
- Tensor coercion (using real torch tensors, not Isaac)
- Batch slicing
- Action shape normalization
- Hook wiring
- `StepOutcome` field population

They DON'T validate:
- Whether the action gets sent to Isaac's physics engine correctly
- Whether observations from Isaac actually arrive in the expected shape
- Sim-to-real behavior

For those, the demo_rollout.py on real Isaac is the authoritative test.

## 10. CUDA version pinning is the user's problem

Isaac Sim 4.5 wants CUDA 11.8 or 12.x. Our adapter doesn't care which CUDA — we just use whatever torch is installed. If the user mismatches torch's CUDA with Isaac's expected CUDA, the env construction will fail before our adapter even runs.

If you hit "RuntimeError: CUDA error: invalid device function" — that's not us; it's a CUDA/torch version mismatch. Use the install commands from `requirements.txt` exactly.

## 11. Follow-up items for the team

1. **Add `[isaac]` extras to `pyproject.toml`.** Won't be a one-liner (Isaac Sim isn't pip-installable in a single line), but `gymnasium>=0.29` and `torch>=2.0` can be added so `pip install roboeval[isaac]` at least gets the Python dependencies right.

2. **Ship a `IsaacBatchEnvironmentAdapter` when the runner gains batched execution.** Reuse most of this code; emit a list of `StepOutcome` per step instead of one.

3. **Per-task success preset modules.** `roboeval/integrations/isaac/presets/franka_cabinet.py` with pre-configured `outcome_from_step` for common Isaac Lab tasks. Document the convention so users contribute their own.

4. **Render-mode frame capture as `artifacts`.** An `artifacts_from_step` hook that calls `env.render()` if `render_mode="rgb_array"` and stuffs the frame under `artifacts["frame"]`. Useful for failure debugging.

5. **`env.spec.id` propagation.** Capture `env.spec.id` in the StepOutcome metadata so reports show "ran against Isaac-Cartpole-Direct-v0" without the user doing it manually.

6. **Determinism docs.** Write a short note about Isaac's determinism story (where it's reliable, where it's not) so users know when replay-from-seed is trustworthy.

## 12. Validation steps before merging

This spike was written on Mac without real Isaac. Before this lands in main:

1. **Real-Isaac smoke test on a cloud GPU box.** Run `python -m roboeval.integrations.isaac.demo_rollout`. Expect 30–200 step logs ending in `terminal=True`.

2. **Real-Isaac EvalRunner integration test.** Wrap the adapter in `EvalRunner` with a `Ruleset` and confirm a report is produced.

3. **Real-Isaac edge cases.** Verify `env.reset(options=...)` works on at least Isaac-Cartpole-Direct-v0 and Isaac-Lift-Cube-Franka-v0. Try `num_envs=4` to confirm the warning fires and batch_index=0 is used.

4. **Existing SDK tests still pass on the dev machine.** This adapter must not touch core SDK files. `cd roboeval && python -m unittest discover -s tests` after `touch tests/__init__.py`.

Document each step's results in this notes file so future contributors know what's been validated and what hasn't.
