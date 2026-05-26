# Examples

These examples show the SDK v1 local eval flow with both hand-written policies and a small trained PyTorch policy.

## One-Command Demo

From the repo root:

```bash
python3 demo.py
```

This runs the dependency-free generic robot suites and writes reports under
`runs/demo/`.

## Hand-Written Policy Eval

Run the config-based demo:

```bash
python3 -m roboeval run examples/configs/eval_config.json --output-dir runs/demo_robot
```

This uses:

- policies from `examples/demo_robot/policies.py`
- Python scenarios from `examples/demo_robot/scenarios.py`
- JSON scenarios from `examples/scenarios/scenarios.json`
- CSV scenarios from `examples/scenarios/scenarios.csv`
- criteria from `examples/criteria/safety_criteria.json`

## Trained Policy Eval

Run the trained policy demo:

```bash
python3 examples/trained_policy/self_check.py
python3 examples/trained_policy/run_eval.py
```

To regenerate the synthetic data and weights:

```bash
python3 examples/trained_policy/train.py
```

The trained demo plugs `policy_v4_trained` into the same SDK runner as the baseline policies. The report includes run metadata, failure cases, action divergences, grouped metrics, and human-readable highlights.

## Generic Any-Robot Eval

Run the generic examples:

```bash
python3 examples/generic_robots/run_eval.py
```

This runs three non-navigation eval suites. Each suite has multiple scenarios,
multiple policy versions, ML-style policy scoring, generic metrics, and failure
cases:

- robot arm: grasping scenarios with alignment, fragile objects, heavy parts,
  `grip_force`, `alignment_error`, and actions like `align_gripper`,
  `soft_close_gripper`, and `lift_object`
- drone inspection: waypoint scenarios with wind, battery budget, no-fly-zone
  detours, `battery_used`, `distance_to_waypoint`, and actions like
  `fly_to_waypoint`, `detour_to_waypoint`, and `scan_target`
- factory process: welding scenarios with preheat, thermal safety, inspection
  requirements, `temperature`, and actions like `micro_preheat`, `weld`,
  `inspect`, and `cool_down`

These examples use `Ruleset` instead of the navigation-specific
`SuccessCriteria` preset, so they show how the same SDK can evaluate robots that
do not share the mobile robot state/action names.
