from __future__ import annotations

from roboeval import Scenario


def fixture_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="fixture_open_path",
            initial_state={
                "front_distance": 80,
                "left_distance": 45,
                "right_distance": 45,
                "goal_direction": "forward",
                "previous_action": "none",
                "step_count": 0,
            },
            max_steps=6,
            metadata={"required_forward_steps": 2, "scenario_type": "open_path", "tags": ["smoke"]},
        )
    ]
