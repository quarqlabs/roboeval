from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .core import Scenario, State


@dataclass
class StepOutcome:
    next_state: State
    outcome: str
    failure_label: str
    terminal: bool


class EnvironmentAdapter(Protocol):
    """Protocol every user-provided robot environment should implement."""

    def reset(self, scenario: Scenario) -> State:
        """Reset the environment for one scenario and return the first state."""
        ...

    def step(self, action: str, scenario: Scenario) -> StepOutcome:
        """Apply one policy action and return the resulting transition."""
        ...


class DemoRobotEnvironment:
    """Small deterministic robot environment used for SDK v1 demos."""

    def __init__(self, safe_distance: float = 20.0) -> None:
        self.safe_distance = safe_distance
        self._state: State = {}
        self._forward_progress = 0
        self._no_progress_steps = 0

    def reset(self, scenario: Scenario) -> State:
        self._state = dict(scenario.initial_state)
        self._forward_progress = 0
        self._no_progress_steps = 0
        return dict(self._state)

    def step(self, action: str, scenario: Scenario) -> StepOutcome:
        state = dict(self._state)
        required_forward_steps = int(scenario.metadata.get("required_forward_steps", 2))
        step_count = int(state.get("step_count", 0)) + 1

        if action == "move_forward":
            if float(state["front_distance"]) < self.safe_distance:
                return StepOutcome(state, "collision", "collision", True)
            if state.get("goal_direction") == "forward":
                self._forward_progress += 1
                self._no_progress_steps = 0
                self._state = _next_state(state, action, front_delta=-8, goal_direction="forward", step_count=step_count)
                if self._forward_progress >= required_forward_steps:
                    return StepOutcome(dict(self._state), "goal_reached", "", True)
                return StepOutcome(dict(self._state), "progress", "", False)
            self._no_progress_steps += 1
            self._state = _next_state(state, action, front_delta=-6, goal_direction=state["goal_direction"], step_count=step_count)
            return self._no_progress_outcome()

        if action in {"turn_left", "turn_right"}:
            side_key = "left_distance" if action == "turn_left" else "right_distance"
            side_distance = float(state[side_key])
            if side_distance < self.safe_distance:
                return StepOutcome(state, "collision", "collision", True)
            if (action == "turn_left" and state.get("goal_direction") == "left") or (
                action == "turn_right" and state.get("goal_direction") == "right"
            ):
                self._no_progress_steps = 0
                self._state = {
                    **state,
                    "front_distance": side_distance,
                    "left_distance": 35,
                    "right_distance": 35,
                    "goal_direction": "forward",
                    "previous_action": action,
                    "step_count": step_count,
                }
                return StepOutcome(dict(self._state), "aligned_turn", "", False)
            self._no_progress_steps += 1
            self._state = {
                **state,
                "front_distance": side_distance,
                "left_distance": 35,
                "right_distance": 35,
                "goal_direction": "right" if action == "turn_left" else "left",
                "previous_action": action,
                "step_count": step_count,
            }
            return self._no_progress_outcome()

        if action == "stop":
            self._no_progress_steps += 1
            self._state = {**state, "previous_action": action, "step_count": step_count}
            if self._no_progress_steps >= 2:
                return StepOutcome(dict(self._state), "stuck_after_stop", "stuck", True)
            return StepOutcome(dict(self._state), "safe_stop", "", False)

        if action == "reverse":
            if (
                float(state["front_distance"]) < self.safe_distance
                and float(state["left_distance"]) < self.safe_distance
                and float(state["right_distance"]) < self.safe_distance
            ):
                self._no_progress_steps = 0
                self._state = {
                    **state,
                    "front_distance": 35,
                    "left_distance": 25,
                    "right_distance": 25,
                    "goal_direction": "forward",
                    "previous_action": action,
                    "step_count": step_count,
                }
                return StepOutcome(dict(self._state), "escape_reverse", "", False)
            self._no_progress_steps += 1
            self._state = _next_state(state, action, front_delta=10, goal_direction=state["goal_direction"], step_count=step_count)
            return self._no_progress_outcome()

        return StepOutcome(state, "invalid_action", "invalid_action", True)

    def _no_progress_outcome(self) -> StepOutcome:
        if self._no_progress_steps >= 3:
            return StepOutcome(dict(self._state), "stuck_no_progress", "stuck", True)
        return StepOutcome(dict(self._state), "no_progress", "", False)


def _next_state(state: State, action: str, front_delta: float, goal_direction: str, step_count: int) -> State:
    return {
        **state,
        "front_distance": max(float(state["front_distance"]) + front_delta, 0),
        "goal_direction": goal_direction,
        "previous_action": action,
        "step_count": step_count,
    }
