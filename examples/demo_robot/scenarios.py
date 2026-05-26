from __future__ import annotations

from roboeval import Scenario


def get_python_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="python_open_path_goal_forward",
            initial_state={
                "front_distance": 80,
                "left_distance": 45,
                "right_distance": 45,
                "goal_direction": "forward",
                "previous_action": "none",
                "step_count": 0,
            },
            max_steps=6,
            metadata={"required_forward_steps": 2, "scenario_type": "open_path", "tags": ["smoke", "navigation"]},
        ),
        Scenario(
            name="python_dead_end_reverse_needed",
            initial_state={
                "front_distance": 8,
                "left_distance": 10,
                "right_distance": 11,
                "goal_direction": "forward",
                "previous_action": "none",
                "step_count": 0,
            },
            max_steps=6,
            metadata={"required_forward_steps": 2, "scenario_type": "dead_end", "tags": ["safety", "regression"]},
        ),
    ]
