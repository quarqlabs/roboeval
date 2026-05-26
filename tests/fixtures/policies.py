from __future__ import annotations

from typing import Any


def baseline_policy(state: dict[str, Any]) -> dict[str, Any]:
    front = float(state["front_distance"])
    right = float(state["right_distance"])
    if front < 20 and right >= 20:
        action = "turn_right"
    else:
        action = "move_forward"
    return {"action": action, "debug_info": {"version": "baseline_policy"}}


def regressing_policy(state: dict[str, Any]) -> dict[str, Any]:
    return {"action": "move_forward", "debug_info": {"version": "regressing_policy"}}


def probabilistic_policy(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": "move_forward",
        "probabilities": {"move_forward": 0.7, "turn_right": 0.3},
        "logits": {"move_forward": 1.2, "turn_right": 0.4},
        "confidence": 0.7,
        "model_version": "probabilistic_policy",
    }
