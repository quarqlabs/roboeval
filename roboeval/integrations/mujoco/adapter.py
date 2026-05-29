"""Raw MuJoCo ↔ RoboEval environment adapter.

This adapter targets the official ``mujoco`` Python bindings. It wraps a
``mujoco.MjModel`` and ``mujoco.MjData`` pair into RoboEval's lightweight
``EnvironmentAdapter`` shape without changing SDK core files.

Gymnasium-style MuJoCo environments should use the Gymnasium adapter. This
module is for teams that work directly with MuJoCo XML worlds and raw
``MjModel`` / ``MjData`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

from roboeval.core import Action, Scenario, State, to_serializable
from roboeval.environment import StepOutcome


ObservationHook = Callable[[mujoco.MjModel, mujoco.MjData, Scenario], dict[str, Any]]
ActionHook = Callable[[Action, mujoco.MjModel, mujoco.MjData, Scenario], None]
OutcomeHook = Callable[[mujoco.MjModel, mujoco.MjData, Scenario, int], tuple[str, str, bool]]
MetricsHook = Callable[[mujoco.MjModel, mujoco.MjData, Scenario, int], dict[str, float | int]]
EventsHook = Callable[[mujoco.MjModel, mujoco.MjData, Scenario, int], list[str]]
InfoHook = Callable[[mujoco.MjModel, mujoco.MjData, Scenario, int], dict[str, Any]]
ResetHook = Callable[[Scenario, mujoco.MjModel, mujoco.MjData], None]


def default_observation_from_data(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scenario: Scenario,
) -> dict[str, Any]:
    """Return the default MuJoCo state snapshot.

    Arrays are copied so policy code and logs see the state at this exact
    transition, not a live view that mutates after the next simulator step.
    """
    return {
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "ctrl": data.ctrl.copy(),
        "time": float(data.time),
    }


def default_reset_from_scenario(
    scenario: Scenario,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    """Reset MuJoCo data and apply optional qpos/qvel/ctrl from the scenario."""
    mujoco.mj_resetData(model, data)
    initial_state = scenario.initial_state

    if "qpos" in initial_state:
        data.qpos[:] = _as_vector(initial_state["qpos"], model.nq, "qpos")
    if "qvel" in initial_state:
        data.qvel[:] = _as_vector(initial_state["qvel"], model.nv, "qvel")
    if "ctrl" in initial_state:
        data.ctrl[:] = _as_vector(initial_state["ctrl"], model.nu, "ctrl")
    if "time" in initial_state:
        data.time = float(initial_state["time"])

    mujoco.mj_forward(model, data)


def default_action_to_ctrl(
    action: Action,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scenario: Scenario,
) -> None:
    """Map a policy action to ``data.ctrl``.

    Accepted default shapes:
    - scalar action for one actuator
    - list / tuple / numpy array with length ``model.nu``
    - ``{"ctrl": ...}`` mapping when users want action metadata around ctrl
    """
    if isinstance(action, dict) and "ctrl" in action:
        action = action["ctrl"]

    if model.nu == 0:
        if action in (None, (), [], {}):
            return
        raise ValueError("MuJoCo model has no actuators, but policy returned a non-empty action.")

    data.ctrl[:] = _as_vector(action, model.nu, "action/ctrl")


def default_outcome_from_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scenario: Scenario,
    step_index: int,
) -> tuple[str, str, bool]:
    """Default MuJoCo outcome mapping.

    Raw MuJoCo has physics state but no built-in task semantics, so the default
    reports ongoing progress. Users should override this hook for goals,
    crashes, contact conditions, or timeouts specific to their robot.
    """
    return ("progress", "", False)


def default_metrics_from_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scenario: Scenario,
    step_index: int,
) -> dict[str, float | int]:
    """Default numeric metrics that are useful across most MuJoCo models."""
    return {
        "time": float(data.time),
        "qpos_norm": float(np.linalg.norm(data.qpos)),
        "qvel_norm": float(np.linalg.norm(data.qvel)),
        "ctrl_norm": float(np.linalg.norm(data.ctrl)),
    }


def default_events_from_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scenario: Scenario,
    step_index: int,
) -> list[str]:
    """Default event hook for raw MuJoCo steps."""
    return []


def default_info_from_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scenario: Scenario,
    step_index: int,
) -> dict[str, Any]:
    """Default MuJoCo namespaced debug info."""
    return {
        "model_nq": int(model.nq),
        "model_nv": int(model.nv),
        "model_nu": int(model.nu),
        "step_index": int(step_index),
        "time": float(data.time),
    }


@dataclass
class MuJoCoEnvironmentAdapter:
    """Wrap raw MuJoCo ``MjModel`` / ``MjData`` into RoboEval's env contract.

    The adapter is intentionally small and hook-based. Users can keep their
    existing MuJoCo model and task logic, then override only the boundary
    translations that are domain-specific: observation, action, outcome,
    metrics, events, info, and reset behavior.
    """

    model: mujoco.MjModel
    data: mujoco.MjData | None = None
    name: str = "mujoco_env"
    observation_from_data: ObservationHook | None = None
    action_to_ctrl: ActionHook | None = None
    outcome_from_step: OutcomeHook | None = None
    metrics_from_step: MetricsHook | None = None
    events_from_step: EventsHook | None = None
    info_from_step: InfoHook | None = None
    reset_from_scenario: ResetHook | None = None
    substeps: int = 1
    coerce_observations: bool = False

    def __post_init__(self) -> None:
        if self.substeps < 1:
            raise ValueError("substeps must be >= 1.")

        if self.data is None:
            self.data = mujoco.MjData(self.model)

        if self.observation_from_data is None:
            self.observation_from_data = default_observation_from_data
        if self.action_to_ctrl is None:
            self.action_to_ctrl = default_action_to_ctrl
        if self.outcome_from_step is None:
            self.outcome_from_step = default_outcome_from_step
        if self.metrics_from_step is None:
            self.metrics_from_step = default_metrics_from_step
        if self.events_from_step is None:
            self.events_from_step = default_events_from_step
        if self.info_from_step is None:
            self.info_from_step = default_info_from_step
        if self.reset_from_scenario is None:
            self.reset_from_scenario = default_reset_from_scenario

        self._step_index = 0

    @classmethod
    def from_xml_path(cls, xml_path: str | Path, **kwargs: Any) -> "MuJoCoEnvironmentAdapter":
        """Build an adapter from a MuJoCo XML file path."""
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        return cls(model=model, **kwargs)

    @classmethod
    def from_xml_string(cls, xml_string: str, **kwargs: Any) -> "MuJoCoEnvironmentAdapter":
        """Build an adapter from an in-memory MuJoCo XML string."""
        model = mujoco.MjModel.from_xml_string(xml_string)
        return cls(model=model, **kwargs)

    def reset(self, scenario: Scenario) -> State:
        self.reset_from_scenario(scenario, self.model, self.data)
        self._step_index = 0
        return self._state_from_data(scenario)

    def step(self, action: Action, scenario: Scenario) -> StepOutcome:
        self.action_to_ctrl(action, self.model, self.data, scenario)
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
            self._step_index += 1

        next_state = self._state_from_data(scenario)
        outcome, failure_label, terminal = self.outcome_from_step(
            self.model,
            self.data,
            scenario,
            self._step_index,
        )
        metrics = self.metrics_from_step(self.model, self.data, scenario, self._step_index)
        events = self.events_from_step(self.model, self.data, scenario, self._step_index)
        mujoco_info = self.info_from_step(self.model, self.data, scenario, self._step_index)

        return StepOutcome(
            next_state=next_state,
            outcome=str(outcome),
            failure_label=str(failure_label),
            terminal=bool(terminal),
            metrics=dict(metrics or {}),
            events=list(events or []),
            info={"mujoco": to_serializable(dict(mujoco_info or {}))},
        )

    def _state_from_data(self, scenario: Scenario) -> State:
        state = self.observation_from_data(self.model, self.data, scenario)
        if self.coerce_observations:
            return to_serializable(state)
        return state


def _as_vector(value: Any, expected_len: int, field_name: str) -> np.ndarray:
    if expected_len == 0:
        array = np.asarray(value, dtype=float).reshape(-1)
        if array.size:
            raise ValueError(f"{field_name} expected length 0, got {array.size}.")
        return array

    if np.isscalar(value):
        array = np.asarray([value], dtype=float)
    else:
        array = np.asarray(value, dtype=float).reshape(-1)

    if array.size != expected_len:
        raise ValueError(f"{field_name} expected length {expected_len}, got {array.size}.")
    return array
