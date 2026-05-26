from __future__ import annotations

import math
from typing import Any


Number = float | int | bool


def arm_policy_v0_rules(state: dict[str, Any]) -> dict[str, Any]:
    """Brittle baseline: tries to go straight down, grip, then lift."""
    if state.get("has_object"):
        action = "lift_object"
    elif float(state.get("end_effector_height", 10)) > 1.2:
        action = "move_arm_down"
    else:
        action = "close_gripper"
    return {
        "action": action,
        "model_version": "arm_policy_v0_rules",
        "confidence": 0.62,
        "scores": _one_hot_scores(action, ARM_ACTIONS),
    }


def arm_policy_v1_linear(state: dict[str, Any]) -> dict[str, Any]:
    return _linear_policy(
        state,
        version="arm_policy_v1_linear",
        actions=ARM_ACTIONS,
        features=_arm_features(state),
        weights={
            "move_arm_down": {
                "bias": 0.1,
                "height_norm": 2.8,
                "alignment_error": -1.0,
                "has_object": -3.0,
                "gripper_closed": -1.5,
            },
            "align_gripper": {
                "bias": -0.2,
                "height_norm": 0.4,
                "alignment_error": 3.2,
                "near_object": 0.8,
                "has_object": -2.5,
            },
            "close_gripper": {
                "bias": -0.4,
                "near_object": 2.6,
                "alignment_error": -2.3,
                "object_present": 1.0,
                "fragile": -0.2,
                "has_object": -3.0,
            },
            "soft_close_gripper": {
                "bias": -1.0,
                "near_object": 1.8,
                "alignment_error": -1.8,
                "fragile": 0.8,
                "required_force": -0.5,
                "has_object": -3.0,
            },
            "lift_object": {
                "bias": -1.2,
                "has_object": 4.0,
                "gripper_closed": 1.2,
            },
        },
    )


def arm_policy_v2_mlp(state: dict[str, Any]) -> dict[str, Any]:
    features = _arm_features(state)
    hidden = _hidden_layer(
        features,
        {
            "approach_ready": {"bias": -0.3, "height_norm": 2.4, "alignment_error": -0.7, "has_object": -3.0},
            "needs_alignment": {"bias": -0.5, "alignment_error": 3.2, "has_object": -2.0},
            "fragile_grip": {"bias": -1.8, "near_object": 3.0, "fragile": 1.8, "alignment_error": -2.4},
            "firm_grip": {
                "bias": -1.2,
                "near_object": 2.0,
                "fragile": -1.4,
                "required_force": 1.7,
                "alignment_error": -2.0,
            },
            "ready_to_lift": {"bias": -1.0, "has_object": 3.5, "gripper_closed": 1.0},
        },
    )
    return _mlp_policy(
        state,
        version="arm_policy_v2_mlp",
        actions=ARM_ACTIONS,
        features=features,
        hidden=hidden,
        output_weights={
            "move_arm_down": {"bias": 0.1, "approach_ready": 2.5, "needs_alignment": -0.5},
            "align_gripper": {"bias": -0.1, "needs_alignment": 2.8, "approach_ready": 0.2},
            "close_gripper": {"bias": -0.4, "firm_grip": 2.7, "fragile_grip": -1.2},
            "soft_close_gripper": {"bias": -0.5, "fragile_grip": 3.0, "firm_grip": -0.8},
            "lift_object": {"bias": -0.3, "ready_to_lift": 3.2},
        },
    )


def arm_policy_v3_regression(state: dict[str, Any]) -> dict[str, Any]:
    """Regression-prone arm policy: fast, but closes before it is aligned."""
    if state.get("has_object"):
        action = "lift_object"
    elif float(state.get("end_effector_height", 10)) <= 3.5:
        action = "close_gripper"
    else:
        action = "move_arm_down"
    return {
        "action": action,
        "model_version": "arm_policy_v3_regression",
        "confidence": 0.78,
        "scores": _one_hot_scores(action, ARM_ACTIONS),
    }


def drone_policy_v0_rules(state: dict[str, Any]) -> dict[str, Any]:
    """Brittle baseline: reach altitude, then scan immediately."""
    action = "scan_target" if float(state.get("altitude", 0)) >= 10 else "ascend"
    return {
        "action": action,
        "model_version": "drone_policy_v0_rules",
        "confidence": 0.6,
        "scores": _one_hot_scores(action, DRONE_ACTIONS),
    }


def drone_policy_v1_linear(state: dict[str, Any]) -> dict[str, Any]:
    return _linear_policy(
        state,
        version="drone_policy_v1_linear",
        actions=DRONE_ACTIONS,
        features=_drone_features(state),
        weights={
            "ascend": {
                "bias": 0.2,
                "altitude_low": 3.0,
                "target_far": 0.2,
                "battery_ratio": -0.9,
            },
            "fly_to_waypoint": {
                "bias": -0.1,
                "altitude_safe": 1.4,
                "distance_norm": 2.5,
                "near_no_fly_zone": -0.8,
                "battery_ratio": -1.1,
            },
            "detour_to_waypoint": {
                "bias": -0.7,
                "altitude_safe": 1.0,
                "distance_norm": 2.0,
                "near_no_fly_zone": 1.5,
                "battery_ratio": -1.4,
            },
            "scan_target": {
                "bias": -1.0,
                "altitude_safe": 1.6,
                "target_close": 3.0,
                "battery_ratio": -0.4,
            },
            "return_home": {
                "bias": -1.5,
                "battery_ratio": 2.7,
                "near_no_fly_zone": 0.8,
            },
        },
    )


def drone_policy_v2_mlp(state: dict[str, Any]) -> dict[str, Any]:
    features = _drone_features(state)
    hidden = _hidden_layer(
        features,
        {
            "safe_altitude_needed": {"bias": -0.2, "altitude_low": 3.2, "battery_ratio": -0.7},
            "direct_path_good": {
                "bias": -0.4,
                "altitude_safe": 1.8,
                "distance_norm": 2.2,
                "near_no_fly_zone": -2.0,
                "altitude_low": -2.5,
            },
            "detour_needed": {
                "bias": -0.5,
                "altitude_safe": 1.5,
                "distance_norm": 1.7,
                "near_no_fly_zone": 2.8,
                "battery_ratio": -1.2,
                "altitude_low": -3.0,
            },
            "scan_ready": {"bias": -0.7, "target_close": 3.5, "altitude_safe": 1.2},
            "abort_needed": {"bias": -1.2, "battery_ratio": 3.1, "near_no_fly_zone": 1.0},
        },
    )
    return _mlp_policy(
        state,
        version="drone_policy_v2_mlp",
        actions=DRONE_ACTIONS,
        features=features,
        hidden=hidden,
        output_weights={
            "ascend": {"bias": 0.0, "safe_altitude_needed": 2.6},
            "fly_to_waypoint": {"bias": -0.2, "direct_path_good": 2.8, "scan_ready": -0.8},
            "detour_to_waypoint": {"bias": -0.3, "detour_needed": 3.0, "scan_ready": -0.5},
            "scan_target": {"bias": -0.2, "scan_ready": 3.2},
            "return_home": {"bias": -0.5, "abort_needed": 2.7},
        },
    )


def drone_policy_v3_risky(state: dict[str, Any]) -> dict[str, Any]:
    """Regression-prone drone policy: saves time by flying straight through risky zones."""
    if float(state.get("altitude", 0)) < 8:
        action = "ascend"
    elif float(state.get("distance_to_waypoint", 0)) > 4:
        action = "fly_to_waypoint"
    else:
        action = "scan_target"
    return {
        "action": action,
        "model_version": "drone_policy_v3_risky",
        "confidence": 0.81,
        "scores": _one_hot_scores(action, DRONE_ACTIONS),
    }


def factory_policy_v0_rules(state: dict[str, Any]) -> dict[str, Any]:
    """Brittle baseline: preheat until 60 C, weld, and never inspect."""
    if state.get("weld_complete"):
        action = "stop"
    elif float(state.get("temperature", 20)) < 60:
        action = "preheat"
    else:
        action = "weld"
    return {
        "action": action,
        "model_version": "factory_policy_v0_rules",
        "confidence": 0.58,
        "scores": _one_hot_scores(action, FACTORY_ACTIONS),
    }


def factory_policy_v1_linear(state: dict[str, Any]) -> dict[str, Any]:
    return _linear_policy(
        state,
        version="factory_policy_v1_linear",
        actions=FACTORY_ACTIONS,
        features=_factory_features(state),
        weights={
            "preheat": {
                "bias": 0.0,
                "below_target": 2.8,
                "thin_material": -1.2,
                "near_max_temp": -2.5,
                "weld_complete": -3.0,
            },
            "micro_preheat": {
                "bias": -0.6,
                "below_target": 1.9,
                "thin_material": 1.9,
                "near_max_temp": -2.0,
                "weld_complete": -3.0,
            },
            "weld": {
                "bias": -0.5,
                "at_target": 3.0,
                "near_max_temp": -1.6,
                "weld_complete": -3.0,
            },
            "inspect": {
                "bias": -1.1,
                "weld_complete": 3.3,
                "inspection_required": 1.8,
                "inspected": -3.0,
            },
            "cool_down": {
                "bias": -1.0,
                "near_max_temp": 3.0,
                "below_target": -1.0,
            },
            "stop": {
                "bias": -1.5,
                "weld_complete": 1.7,
                "inspection_required": -1.2,
            },
        },
    )


def factory_policy_v2_mlp(state: dict[str, Any]) -> dict[str, Any]:
    features = _factory_features(state)
    hidden = _hidden_layer(
        features,
        {
            "needs_heat": {"bias": -0.2, "below_target": 3.0, "slightly_below_target": -2.0, "near_max_temp": -2.8},
            "needs_gentle_heat": {
                "bias": -1.0,
                "slightly_below_target": 1.6,
                "thin_material": 1.8,
                "near_max_temp": -2.0,
            },
            "ready_to_weld": {"bias": -0.4, "at_target": 3.2, "near_max_temp": -1.0, "weld_complete": -3.0},
            "needs_inspection": {
                "bias": -1.3,
                "weld_complete": 4.0,
                "inspection_required": 1.0,
                "inspected": -3.2,
            },
            "thermal_risk": {"bias": -0.8, "near_max_temp": 3.3, "below_target": -0.8},
        },
    )
    return _mlp_policy(
        state,
        version="factory_policy_v2_mlp",
        actions=FACTORY_ACTIONS,
        features=features,
        hidden=hidden,
        output_weights={
            "preheat": {"bias": -0.1, "needs_heat": 2.5, "needs_gentle_heat": -1.4, "thermal_risk": -1.6},
            "micro_preheat": {"bias": -0.2, "needs_gentle_heat": 2.8, "thermal_risk": -1.2},
            "weld": {"bias": -0.3, "ready_to_weld": 3.0},
            "inspect": {"bias": -0.1, "needs_inspection": 3.2},
            "cool_down": {"bias": -0.4, "thermal_risk": 2.8},
            "stop": {"bias": -1.3, "needs_inspection": -1.0, "ready_to_weld": -0.7},
        },
    )


def factory_policy_v3_hot(state: dict[str, Any]) -> dict[str, Any]:
    """Regression-prone factory policy: high throughput, poor thermal safety."""
    if state.get("weld_complete") and state.get("inspection_required") and not state.get("inspected"):
        action = "inspect"
    elif not state.get("weld_complete") and float(state.get("temperature", 20)) < 75:
        action = "preheat"
    elif not state.get("weld_complete"):
        action = "weld"
    else:
        action = "stop"
    return {
        "action": action,
        "model_version": "factory_policy_v3_hot",
        "confidence": 0.83,
        "scores": _one_hot_scores(action, FACTORY_ACTIONS),
    }


# Backward-compatible aliases used by the earlier simple generic demo.
arm_policy_v1 = arm_policy_v1_linear
drone_policy_v1 = drone_policy_v1_linear
factory_policy_v1 = factory_policy_v1_linear


ARM_ACTIONS = [
    "move_arm_down",
    "align_gripper",
    "close_gripper",
    "soft_close_gripper",
    "lift_object",
]
DRONE_ACTIONS = [
    "ascend",
    "fly_to_waypoint",
    "detour_to_waypoint",
    "scan_target",
    "return_home",
]
FACTORY_ACTIONS = [
    "preheat",
    "micro_preheat",
    "weld",
    "inspect",
    "cool_down",
    "stop",
]


def _linear_policy(
    state: dict[str, Any],
    *,
    version: str,
    actions: list[str],
    features: dict[str, float],
    weights: dict[str, dict[str, float]],
) -> dict[str, Any]:
    scores = {
        action: sum(features.get(feature, 0.0) * weight for feature, weight in action_weights.items())
        for action, action_weights in weights.items()
    }
    return _decision(state, version, actions, features, scores, model_family="linear_classifier")


def _mlp_policy(
    state: dict[str, Any],
    *,
    version: str,
    actions: list[str],
    features: dict[str, float],
    hidden: dict[str, float],
    output_weights: dict[str, dict[str, float]],
) -> dict[str, Any]:
    scores = {
        action: sum(
            (hidden.get(feature, 0.0) if feature != "bias" else 1.0) * weight
            for feature, weight in action_weights.items()
        )
        for action, action_weights in output_weights.items()
    }
    return _decision(state, version, actions, {**features, **hidden}, scores, model_family="two_layer_mlp_style")


def _decision(
    state: dict[str, Any],
    version: str,
    actions: list[str],
    features: dict[str, float],
    scores: dict[str, float],
    *,
    model_family: str,
) -> dict[str, Any]:
    scores = {action: scores.get(action, -10.0) for action in actions}
    probabilities = _softmax(scores)
    action = max(scores, key=scores.get)
    return {
        "action": action,
        "model_version": version,
        "scores": _rounded(scores),
        "probabilities": _rounded(probabilities),
        "confidence": round(probabilities[action], 4),
        "debug_info": {
            "model_family": model_family,
            "features": _rounded(features),
            "state_snapshot": _compact_state(state),
        },
    }


def _hidden_layer(features: dict[str, float], weights: dict[str, dict[str, float]]) -> dict[str, float]:
    hidden: dict[str, float] = {}
    for unit, unit_weights in weights.items():
        raw = sum(
            (1.0 if feature == "bias" else features.get(feature, 0.0)) * weight
            for feature, weight in unit_weights.items()
        )
        hidden[unit] = max(raw, 0.0)
    return hidden


def _arm_features(state: dict[str, Any]) -> dict[str, float]:
    height = float(state.get("end_effector_height", 10.0))
    alignment_error = abs(float(state.get("horizontal_error", 0.0)))
    return {
        "bias": 1.0,
        "height_norm": min(height / 10.0, 1.5),
        "near_object": 1.0 if height <= 1.2 else 0.0,
        "alignment_error": min(alignment_error / 3.0, 1.5),
        "object_present": _flag(state.get("object_present", True)),
        "fragile": _flag(state.get("object_fragile", False)),
        "required_force": float(state.get("required_grip_force", 0.72)),
        "max_safe_force": float(state.get("max_safe_force", 0.9)),
        "gripper_closed": _flag(state.get("gripper_closed", False)),
        "has_object": _flag(state.get("has_object", False)),
    }


def _drone_features(state: dict[str, Any]) -> dict[str, float]:
    altitude = float(state.get("altitude", 0.0))
    distance = float(state.get("distance_to_waypoint", 0.0))
    battery = float(state.get("battery_used", 0.0))
    no_fly_distance = float(state.get("no_fly_zone_distance", 10.0))
    wind = float(state.get("wind_speed", 0.0))
    return {
        "bias": 1.0,
        "altitude_low": 1.0 if altitude < 10.0 else 0.0,
        "altitude_safe": 1.0 if altitude >= 10.0 else 0.0,
        "distance_norm": min(distance / 80.0, 1.5),
        "target_far": 1.0 if distance > 8.0 else 0.0,
        "target_close": 1.0 if distance <= 12.0 else 0.0,
        "battery_ratio": min(battery / 30.0, 2.0),
        "near_no_fly_zone": 1.0 if no_fly_distance < 2.0 else 0.0,
        "wind_norm": min(wind / 12.0, 1.5),
    }


def _factory_features(state: dict[str, Any]) -> dict[str, float]:
    temperature = float(state.get("temperature", 20.0))
    target_temperature = float(state.get("target_temperature", 60.0))
    max_temperature = float(state.get("max_temperature", 90.0))
    material_thickness = float(state.get("material_thickness", 1.0))
    return {
        "bias": 1.0,
        "temp_norm": temperature / 100.0,
        "below_target": 1.0 if temperature < target_temperature else 0.0,
        "slightly_below_target": 1.0 if 0 < target_temperature - temperature <= 15 else 0.0,
        "at_target": 1.0 if temperature >= target_temperature else 0.0,
        "near_max_temp": 1.0 if temperature >= max_temperature - 10 else 0.0,
        "thin_material": 1.0 if material_thickness < 0.6 else 0.0,
        "inspection_required": _flag(state.get("inspection_required", False)),
        "weld_complete": _flag(state.get("weld_complete", False)),
        "inspected": _flag(state.get("inspected", False)),
    }


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    top = max(scores.values())
    exps = {key: math.exp(value - top) for key, value in scores.items()}
    total = sum(exps.values()) or 1.0
    return {key: value / total for key, value in exps.items()}


def _one_hot_scores(action: str, actions: list[str]) -> dict[str, float]:
    return {candidate: (1.0 if candidate == action else 0.0) for candidate in actions}


def _rounded(values: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 4) for key, value in values.items()}


def _flag(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _compact_state(state: dict[str, Any]) -> dict[str, Number | str]:
    return {
        key: value
        for key, value in state.items()
        if isinstance(value, (int, float, bool, str))
    }
