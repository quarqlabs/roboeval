from __future__ import annotations

from typing import Any

from examples.trained_policy.actions import GOAL_DIRECTIONS, PREVIOUS_ACTIONS


FEATURE_NAMES = [
    "front_distance_norm",
    "left_distance_norm",
    "right_distance_norm",
    "step_count_norm",
    "goal_forward",
    "goal_left",
    "goal_right",
    "previous_none",
    "previous_move_forward",
    "previous_turn_left",
    "previous_turn_right",
    "previous_stop",
    "previous_reverse",
]


def encode_state(state: dict[str, Any]) -> list[float]:
    goal_direction = str(state.get("goal_direction", "forward"))
    previous_action = str(state.get("previous_action", "none"))
    return [
        float(state["front_distance"]) / 100.0,
        float(state["left_distance"]) / 100.0,
        float(state["right_distance"]) / 100.0,
        min(float(state.get("step_count", 0)), 10.0) / 10.0,
        *[1.0 if goal_direction == direction else 0.0 for direction in GOAL_DIRECTIONS],
        *[1.0 if previous_action == action else 0.0 for action in PREVIOUS_ACTIONS],
    ]
