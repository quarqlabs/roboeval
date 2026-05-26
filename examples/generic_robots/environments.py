from __future__ import annotations

from typing import Any

from roboeval import Scenario, StepOutcome


class RobotArmEnvironment:
    name = "robot_arm_environment"

    def __init__(self) -> None:
        self.state: dict[str, Any] = {}

    def reset(self, scenario: Scenario) -> dict[str, Any]:
        self.state = {**scenario.initial_state, "step_count": 0, "last_grip_force": 0.0}
        return dict(self.state)

    def step(self, action: str, scenario: Scenario) -> StepOutcome:
        state = {**self.state, "step_count": int(self.state.get("step_count", 0)) + 1}
        if action == "move_arm_down":
            height = max(float(state.get("end_effector_height", 10.0)) - 4.0, 0.8)
            self.state = {**state, "end_effector_height": height}
            outcome = "approached_object" if height <= 1.2 else "descending"
            return self._outcome(outcome, "", False, events=["arm_down"])

        if action == "align_gripper":
            error = max(abs(float(state.get("horizontal_error", 0.0))) - 1.3, 0.0)
            self.state = {**state, "horizontal_error": error}
            return self._outcome("object_aligned", "", False, events=["alignment_adjusted"])

        if action in {"close_gripper", "soft_close_gripper"}:
            grip_force = 0.62 if action == "soft_close_gripper" else 0.92
            self.state = {**state, "last_grip_force": grip_force}
            if not bool(state.get("object_present", True)):
                return self._outcome("object_missing", "object_missing", True, events=["empty_grasp"])
            if float(state.get("end_effector_height", 10.0)) > 1.3 or abs(float(state.get("horizontal_error", 0.0))) > 0.7:
                return self._outcome("missed_grasp", "missed_grasp", True, events=["bad_pose"])
            if grip_force > float(state.get("max_safe_force", 0.9)):
                return self._outcome("over_force", "over_force", True, events=["force_limit_exceeded"])
            if grip_force < float(state.get("required_grip_force", 0.7)):
                return self._outcome("dropped_object", "dropped_object", True, events=["grip_too_weak"])
            self.state = {**self.state, "gripper_closed": True, "has_object": True}
            return self._outcome("object_grasped", "", False, events=["gripper_closed"])

        if action == "lift_object":
            if bool(state.get("has_object", False)):
                self.state = {**state, "end_effector_height": 6.0}
                return self._outcome("object_lifted", "", True, events=["object_lifted"])
            return self._outcome("dropped_object", "dropped_object", True, events=["lift_without_object"])

        return StepOutcome(dict(self.state), "invalid_action", "invalid_action", True)

    def _outcome(self, outcome: str, failure_label: str, terminal: bool, *, events: list[str]) -> StepOutcome:
        return StepOutcome(
            dict(self.state),
            outcome,
            failure_label,
            terminal,
            metrics={
                "grip_force": float(self.state.get("last_grip_force", 0.0)),
                "alignment_error": abs(float(self.state.get("horizontal_error", 0.0))),
                "end_effector_height": float(self.state.get("end_effector_height", 0.0)),
            },
            events=events,
            info={"domain": "robot_arm"},
        )


class DroneInspectionEnvironment:
    name = "drone_inspection_environment"

    def __init__(self) -> None:
        self.state: dict[str, Any] = {}

    def reset(self, scenario: Scenario) -> dict[str, Any]:
        self.state = {**scenario.initial_state, "step_count": 0}
        return dict(self.state)

    def step(self, action: str, scenario: Scenario) -> StepOutcome:
        state = {**self.state, "step_count": int(self.state.get("step_count", 0)) + 1}
        wind = float(state.get("wind_speed", 0.0))

        if action == "ascend":
            self.state = {
                **state,
                "altitude": min(float(state.get("altitude", 0.0)) + 10.0, 24.0),
                "battery_used": float(state.get("battery_used", 0.0)) + 5.0 + wind * 0.2,
            }
            return self._battery_checked_outcome("safe_altitude", "", False, events=["ascended"])

        if action in {"fly_to_waypoint", "detour_to_waypoint"}:
            if float(state.get("altitude", 0.0)) < 10.0:
                self.state = state
                return self._outcome("unsafe_low_altitude", "unsafe_low_altitude", True, events=["low_altitude_move"])
            if action == "fly_to_waypoint" and float(state.get("no_fly_zone_distance", 10.0)) < 1.6:
                self.state = state
                return self._outcome("no_fly_zone_violation", "no_fly_zone_violation", True, events=["unsafe_shortcut"])
            distance_drop = max((24.0 if action == "fly_to_waypoint" else 18.0) - wind * 0.8, 8.0)
            battery_delta = (7.0 if action == "fly_to_waypoint" else 9.0) + wind * 0.25
            self.state = {
                **state,
                "distance_to_waypoint": max(float(state.get("distance_to_waypoint", 0.0)) - distance_drop, 0.0),
                "battery_used": float(state.get("battery_used", 0.0)) + battery_delta,
                "no_fly_zone_distance": float(state.get("no_fly_zone_distance", 10.0)) + (0.8 if action == "detour_to_waypoint" else 0.0),
            }
            outcome = "target_in_range" if float(self.state.get("distance_to_waypoint", 0.0)) <= 12.0 else "approached_waypoint"
            return self._battery_checked_outcome(outcome, "", False, events=[action])

        if action == "scan_target":
            if float(state.get("altitude", 0.0)) < 10.0:
                self.state = state
                return self._outcome("unsafe_low_altitude", "unsafe_low_altitude", True, events=["scan_too_low"])
            if float(state.get("distance_to_waypoint", 0.0)) > 12.0:
                self.state = state
                return self._outcome("lost_target", "lost_target", True, events=["scan_out_of_range"])
            self.state = {
                **state,
                "battery_used": float(state.get("battery_used", 0.0)) + 4.0,
                "target_scanned": True,
            }
            return self._battery_checked_outcome("waypoint_inspected", "", True, events=["scan_complete"])

        if action == "return_home":
            self.state = {**state, "battery_used": float(state.get("battery_used", 0.0)) + 3.0}
            return self._outcome("returned_home", "", True, events=["returned_home"])

        return StepOutcome(dict(self.state), "invalid_action", "invalid_action", True)

    def _battery_checked_outcome(self, outcome: str, failure_label: str, terminal: bool, *, events: list[str]) -> StepOutcome:
        if float(self.state.get("battery_used", 0.0)) > 30.0:
            return self._outcome("battery_exceeded", "battery_exceeded", True, events=[*events, "battery_limit"])
        return self._outcome(outcome, failure_label, terminal, events=events)

    def _outcome(self, outcome: str, failure_label: str, terminal: bool, *, events: list[str]) -> StepOutcome:
        return StepOutcome(
            dict(self.state),
            outcome,
            failure_label,
            terminal,
            metrics={
                "battery_used": float(self.state.get("battery_used", 0.0)),
                "altitude": float(self.state.get("altitude", 0.0)),
                "distance_to_waypoint": float(self.state.get("distance_to_waypoint", 0.0)),
            },
            events=events,
            info={"domain": "drone_inspection"},
        )


class FactoryWeldEnvironment:
    name = "factory_weld_environment"

    def __init__(self) -> None:
        self.state: dict[str, Any] = {}

    def reset(self, scenario: Scenario) -> dict[str, Any]:
        self.state = {**scenario.initial_state, "step_count": 0}
        return dict(self.state)

    def step(self, action: str, scenario: Scenario) -> StepOutcome:
        state = {**self.state, "step_count": int(self.state.get("step_count", 0)) + 1}
        temperature = float(state.get("temperature", 20.0))
        max_temperature = float(state.get("max_temperature", 90.0))
        target_temperature = float(state.get("target_temperature", 60.0))

        if action in {"preheat", "micro_preheat"}:
            delta = 28.0 if action == "preheat" else 10.0
            self.state = {**state, "temperature": temperature + delta}
            if float(self.state["temperature"]) > max_temperature:
                return self._outcome("overheat", "overheat", True, events=["thermal_limit_exceeded"])
            return self._outcome("preheated", "", False, events=[action])

        if action == "cool_down":
            self.state = {**state, "temperature": max(temperature - 18.0, 20.0)}
            return self._outcome("cooled_down", "", False, events=["cool_down"])

        if action == "weld":
            if temperature < target_temperature:
                self.state = state
                return self._outcome("underheated_weld", "underheated_weld", True, events=["weld_too_cold"])
            self.state = {**state, "temperature": temperature + 12.0, "weld_complete": True}
            if float(self.state["temperature"]) > max_temperature:
                return self._outcome("overheat", "overheat", True, events=["thermal_limit_exceeded"])
            terminal = not bool(state.get("inspection_required", False))
            return self._outcome("weld_completed", "", terminal, events=["weld_finished"])

        if action == "inspect":
            if not bool(state.get("weld_complete", False)):
                self.state = state
                return self._outcome("inspection_missing", "inspection_missing", True, events=["inspect_before_weld"])
            self.state = {**state, "inspected": True}
            return self._outcome("inspection_passed", "", True, events=["inspection_passed"])

        if action == "stop":
            self.state = state
            if bool(state.get("inspection_required", False)) and bool(state.get("weld_complete", False)) and not bool(state.get("inspected", False)):
                return self._outcome("inspection_missing", "inspection_missing", True, events=["stopped_before_inspection"])
            if bool(state.get("weld_complete", False)):
                return self._outcome("process_stopped", "", True, events=["stopped"])
            return self._outcome("idle_stop", "no_work_completed", True, events=["idle_stop"])

        return StepOutcome(dict(self.state), "invalid_action", "invalid_action", True)

    def _outcome(self, outcome: str, failure_label: str, terminal: bool, *, events: list[str]) -> StepOutcome:
        return StepOutcome(
            dict(self.state),
            outcome,
            failure_label,
            terminal,
            metrics={
                "temperature": float(self.state.get("temperature", 0.0)),
                "material_thickness": float(self.state.get("material_thickness", 0.0)),
            },
            events=events,
            info={"domain": "factory_weld"},
        )
