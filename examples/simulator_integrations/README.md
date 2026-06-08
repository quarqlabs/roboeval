# Simulator Integration Examples

These examples exercise the optional Gymnasium and raw MuJoCo adapters with
small scenario JSON files.

The base SDK stays dependency-free, so install the extra you want before
running the corresponding example:

```bash
pip install -e ".[gymnasium]"
pip install -e ".[mujoco]"
```

## Gymnasium CartPole

```bash
python3 examples/simulator_integrations/gymnasium_cartpole.py
```

Uses:

- `data/gymnasium_cartpole_scenarios.json`
- `roboeval.integrations.gymnasium.GymnasiumEnvironmentAdapter`
- a tiny CartPole heuristic policy

The report is written to `runs/simulator_integrations/gymnasium_cartpole/`.

## Raw MuJoCo Point Mass

```bash
python3 examples/simulator_integrations/mujoco_point_mass.py
```

Uses:

- `data/mujoco_point_mass_scenarios.json`
- `roboeval/integrations/mujoco/assets/point_mass.xml`
- `roboeval.integrations.mujoco.MuJoCoEnvironmentAdapter`
- a small proportional controller policy

The report is written to `runs/simulator_integrations/mujoco_point_mass/`.
