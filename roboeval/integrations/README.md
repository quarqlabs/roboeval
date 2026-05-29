# `roboeval.integrations`

Optional simulator-side adapters that wrap third-party robotics frameworks into the SDK's `EnvironmentAdapter` Protocol.

## Why this exists

The `roboeval` core stays lightweight and dependency-free (see the root `pyproject.toml` — `dependencies = []`). Each integration in this subpackage carries its own framework dependency, kept out of the core install path so a base `pip install roboeval` never pulls in heavy simulators users may not need.

The longer-term distribution model is optional extras:

```bash
pip install roboeval                # core only
pip install roboeval[gymnasium]     # adds Gymnasium adapter
pip install roboeval[mujoco]        # adds MuJoCo adapter (planned)
pip install roboeval[pybullet]      # adds PyBullet adapter (planned)
```

Each integration folder also ships its own `requirements.txt` as a local reference, but the package extras in `pyproject.toml` are the authoritative install path.

## Current integrations

| Folder | Wraps | Status |
|--------|-------|--------|
| `gymnasium/` | Any `gymnasium.Env` | Spike (single-env, no vector support, no render-frame capture) |

## Template for new integrations

Every integration folder should provide:

```
integrations/<name>/
  __init__.py            # re-exports the adapter class
  adapter.py             # the adapter implementation
  demo_rollout.py        # minimal end-to-end smoke test
  README.md              # usage + mapping table
  requirements.txt       # third-party deps for this integration
  notes.md               # design rationale, gotchas, follow-ups
```

Each adapter should:

- Match the style of `CallableEnvironmentAdapter` in `roboeval/environment.py` — `@dataclass`, `name: str` field, no inheritance, two duck-typed methods.
- Construct `StepOutcome` directly using the existing 8 fields (no subclassing).
- Expose override hooks for the framework-specific translations so users can customize the boundary without subclassing.
- Reuse `roboeval.core.to_serializable` for JSON safety rather than reinventing it.
- Namespace framework-specific raw data under `info["<framework>"]` (e.g. `info["gymnasium"]`, `info["mujoco"]`) so the SDK's `_metric_summary`, rules, and reports remain framework-agnostic.

The Gymnasium adapter follows this template — use it as the reference.
