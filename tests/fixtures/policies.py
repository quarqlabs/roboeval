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
