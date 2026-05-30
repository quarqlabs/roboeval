# MuJoCo Integration Notes

## Design Rationale

This adapter is intentionally raw-MuJoCo first. Many robotics teams already own XML worlds, `MjModel`, `MjData`, task reset logic, and low-level controllers. The adapter should let them keep that stack and only translate the eval boundary into RoboEval's `reset` / `step` / `StepOutcome` shape.

## Why Not Subclass StepOutcome

RoboEval reports and rules already understand `StepOutcome`. MuJoCo-specific data belongs in `info["mujoco"]` so the core SDK stays framework-agnostic and other integrations can follow the same pattern.

## Main Semantic Gap

Raw MuJoCo provides physics, not task meaning. It does not know if a rollout succeeded, failed, violated safety, or reached a goal. That is why `outcome_from_step`, `metrics_from_step`, and `events_from_step` are hooks.

## Action Mapping

The default action mapping writes the policy action into `data.ctrl`. This works for simple actuator control. Real robots may need richer mapping, for example:

- high-level action names to target joint controls
- torque vectors with clipping
- position-control commands
- dict actions like `{"ctrl": [...], "mode": "hold"}`

Those users should override `action_to_ctrl`.

## Follow-Ups

- render/video artifact capture
- contact and collision event helpers
- named joint/sensor observation helpers
- multi-model benchmark suites
- optional package metadata for `roboeval[mujoco]` release
