from __future__ import annotations

from roboeval import Scenario


def arm_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="arm_center_cube_easy",
            initial_state={
                "end_effector_height": 8.0,
                "horizontal_error": 0.2,
                "gripper_closed": False,
                "has_object": False,
                "object_present": True,
                "object_fragile": False,
                "required_grip_force": 0.72,
                "max_safe_force": 0.95,
            },
            max_steps=5,
            metadata={"scenario_type": "robot_arm", "tags": ["grasp", "easy"]},
        ),
        Scenario(
            name="arm_offset_cube_needs_alignment",
            initial_state={
                "end_effector_height": 8.0,
                "horizontal_error": 2.8,
                "gripper_closed": False,
                "has_object": False,
                "object_present": True,
                "object_fragile": False,
                "required_grip_force": 0.72,
                "max_safe_force": 0.95,
            },
            max_steps=7,
            metadata={"scenario_type": "robot_arm", "tags": ["grasp", "alignment"]},
        ),
        Scenario(
            name="arm_fragile_vial_low_force",
            initial_state={
                "end_effector_height": 6.0,
                "horizontal_error": 0.3,
                "gripper_closed": False,
                "has_object": False,
                "object_present": True,
                "object_fragile": True,
                "required_grip_force": 0.48,
                "max_safe_force": 0.75,
            },
            max_steps=5,
            metadata={"scenario_type": "robot_arm", "tags": ["grasp", "fragile"]},
        ),
        Scenario(
            name="arm_heavy_part_firm_grip",
            initial_state={
                "end_effector_height": 6.0,
                "horizontal_error": 0.4,
                "gripper_closed": False,
                "has_object": False,
                "object_present": True,
                "object_fragile": False,
                "required_grip_force": 0.84,
                "max_safe_force": 0.98,
            },
            max_steps=5,
            metadata={"scenario_type": "robot_arm", "tags": ["grasp", "heavy"]},
        ),
        Scenario(
            name="arm_precise_tall_start",
            initial_state={
                "end_effector_height": 12.0,
                "horizontal_error": 1.4,
                "gripper_closed": False,
                "has_object": False,
                "object_present": True,
                "object_fragile": False,
                "required_grip_force": 0.7,
                "max_safe_force": 0.92,
            },
            max_steps=8,
            metadata={"scenario_type": "robot_arm", "tags": ["grasp", "precision"]},
        ),
    ]


def drone_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="drone_near_waypoint",
            initial_state={
                "altitude": 0.0,
                "distance_to_waypoint": 20.0,
                "battery_used": 0.0,
                "target_scanned": False,
                "no_fly_zone_distance": 8.0,
                "wind_speed": 0.0,
            },
            max_steps=5,
            metadata={"scenario_type": "drone", "tags": ["inspection", "easy"]},
        ),
        Scenario(
            name="drone_far_waypoint",
            initial_state={
                "altitude": 0.0,
                "distance_to_waypoint": 62.0,
                "battery_used": 0.0,
                "target_scanned": False,
                "no_fly_zone_distance": 8.0,
                "wind_speed": 1.0,
            },
            max_steps=6,
            metadata={"scenario_type": "drone", "tags": ["inspection", "far"]},
        ),
        Scenario(
            name="drone_no_fly_zone_detour",
            initial_state={
                "altitude": 0.0,
                "distance_to_waypoint": 34.0,
                "battery_used": 0.0,
                "target_scanned": False,
                "no_fly_zone_distance": 1.4,
                "wind_speed": 1.0,
            },
            max_steps=6,
            metadata={"scenario_type": "drone", "tags": ["inspection", "safety"]},
        ),
        Scenario(
            name="drone_low_battery_close_target",
            initial_state={
                "altitude": 12.0,
                "distance_to_waypoint": 14.0,
                "battery_used": 17.0,
                "target_scanned": False,
                "no_fly_zone_distance": 8.0,
                "wind_speed": 0.0,
            },
            max_steps=4,
            metadata={"scenario_type": "drone", "tags": ["inspection", "battery"]},
        ),
        Scenario(
            name="drone_windy_target",
            initial_state={
                "altitude": 0.0,
                "distance_to_waypoint": 44.0,
                "battery_used": 0.0,
                "target_scanned": False,
                "no_fly_zone_distance": 8.0,
                "wind_speed": 9.0,
            },
            max_steps=7,
            metadata={"scenario_type": "drone", "tags": ["inspection", "wind"]},
        ),
    ]


def factory_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="factory_standard_weld",
            initial_state={
                "temperature": 20.0,
                "target_temperature": 60.0,
                "max_temperature": 90.0,
                "material_thickness": 1.0,
                "weld_complete": False,
                "inspection_required": False,
                "inspected": False,
            },
            max_steps=4,
            metadata={"scenario_type": "factory_process", "tags": ["welding", "standard"]},
        ),
        Scenario(
            name="factory_cold_start_heavy_part",
            initial_state={
                "temperature": 5.0,
                "target_temperature": 70.0,
                "max_temperature": 95.0,
                "material_thickness": 1.4,
                "weld_complete": False,
                "inspection_required": False,
                "inspected": False,
            },
            max_steps=5,
            metadata={"scenario_type": "factory_process", "tags": ["welding", "cold_start"]},
        ),
        Scenario(
            name="factory_hot_surface_needs_cooling",
            initial_state={
                "temperature": 78.0,
                "target_temperature": 60.0,
                "max_temperature": 82.0,
                "material_thickness": 1.0,
                "weld_complete": False,
                "inspection_required": False,
                "inspected": False,
            },
            max_steps=5,
            metadata={"scenario_type": "factory_process", "tags": ["welding", "thermal_safety"]},
        ),
        Scenario(
            name="factory_inspection_required",
            initial_state={
                "temperature": 58.0,
                "target_temperature": 60.0,
                "max_temperature": 90.0,
                "material_thickness": 1.0,
                "weld_complete": False,
                "inspection_required": True,
                "inspected": False,
            },
            max_steps=4,
            metadata={"scenario_type": "factory_process", "tags": ["welding", "inspection"]},
        ),
        Scenario(
            name="factory_thin_material_gentle_heat",
            initial_state={
                "temperature": 35.0,
                "target_temperature": 50.0,
                "max_temperature": 70.0,
                "material_thickness": 0.35,
                "weld_complete": False,
                "inspection_required": False,
                "inspected": False,
            },
            max_steps=5,
            metadata={"scenario_type": "factory_process", "tags": ["welding", "thin_material"]},
        ),
    ]
