from __future__ import annotations

from typing import Any


def policy_v0_rules(state: dict[str, Any]) -> dict[str, Any]:
    front = float(state["front_distance"])
    left = float(state["left_distance"])
    right = float(state["right_distance"])
    goal = state["goal_direction"]

    if front < 15 and left < 15 and right < 15:
        action = "reverse"
    elif front < 20:
        action = "turn_left" if left > right and left >= 20 else "turn_right" if right >= 20 else "stop"
    elif goal == "left" and left >= 20:
        action = "turn_left"
    elif goal == "right" and right >= 20:
        action = "turn_right"
    else:
        action = "move_forward"
    return {"action": action, "debug_info": {"policy_type": "rules"}}


def policy_v1_cautious(state: dict[str, Any]) -> dict[str, Any]:
    return _linear_policy(
        state,
        version="policy_v1_cautious",
        weights={
            "move_forward": (-0.8, 3.2, -0.2, -0.2, 1.2, -0.6, -0.6),
            "turn_left": (-0.2, -1.2, 2.4, -0.3, -0.2, 1.7, -0.6),
            "turn_right": (-0.2, -1.2, -0.3, 2.4, -0.2, -0.6, 1.7),
            "stop": (0.4, -2.2, -0.4, -0.4, -0.1, -0.1, -0.1),
            "reverse": (-0.1, -2.0, -1.0, -1.0, 0.5, -0.1, -0.1),
        },
    )


def policy_v2_aggressive(state: dict[str, Any]) -> dict[str, Any]:
    return _linear_policy(
        state,
        version="policy_v2_aggressive",
        weights={
            "move_forward": (0.2, 2.7, -0.2, -0.2, 2.5, -0.4, -0.4),
            "turn_left": (-0.4, -0.8, 2.0, -0.2, -0.2, 1.3, -0.4),
            "turn_right": (-0.4, -0.8, -0.2, 2.0, -0.2, -0.4, 1.3),
            "stop": (-0.2, -1.5, -0.5, -0.5, -0.2, -0.2, -0.2),
            "reverse": (-0.7, -1.4, -0.9, -0.9, 0.2, -0.2, -0.2),
        },
    )


def policy_v3_balanced(state: dict[str, Any]) -> dict[str, Any]:
    front = float(state["front_distance"])
    left = float(state["left_distance"])
    right = float(state["right_distance"])
    goal = state["goal_direction"]
    if front < 20 and left < 20 and right < 20:
        action = "reverse"
    elif front < 20 and right >= 20:
        action = "turn_right"
    elif front < 20 and left >= 20:
        action = "turn_left"
    elif goal == "left":
        action = "turn_left"
    elif goal == "right":
        action = "turn_right"
    else:
        action = "move_forward"
    return {"action": action, "debug_info": {"policy_type": "balanced"}}


def policy_v4_noisy(state: dict[str, Any]) -> dict[str, Any]:
    step = int(state.get("step_count", 0))
    action = "stop" if step % 2 == 0 else "turn_left"
    return {"action": action, "debug_info": {"policy_type": "unstable", "step_parity": step % 2}}


def _linear_policy(state: dict[str, Any], version: str, weights: dict[str, tuple[float, ...]]) -> dict[str, Any]:
    features = _features(state)
    scores = {
        action: round(sum(weight * feature for weight, feature in zip(action_weights, features, strict=True)), 4)
        for action, action_weights in weights.items()
    }
    action = max(scores, key=scores.get)
    return {"action": action, "debug_info": {"policy_type": "linear_model", "version": version, "scores": scores}}


def _features(state: dict[str, Any]) -> tuple[float, ...]:
    return (
        1.0,
        float(state["front_distance"]) / 100.0,
        float(state["left_distance"]) / 100.0,
        float(state["right_distance"]) / 100.0,
        1.0 if state["goal_direction"] == "forward" else 0.0,
        1.0 if state["goal_direction"] == "left" else 0.0,
        1.0 if state["goal_direction"] == "right" else 0.0,
    )
