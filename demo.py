"""
robot-policy-eval — quick demo

Three policies, three scenarios, one structured report.

Run:
    python3.11 demo.py
"""
from pathlib import Path

from roboeval import EvalRunner, Ruleset, Scenario, forbid_failure, require_outcome


# ── Policies ─────────────────────────────────────────────────────────────────

def baseline_v1(state):
    """Simple rules policy: move forward if clear, reverse if all sides blocked."""
    front = float(state["front_distance"])
    right = float(state["right_distance"])
    if front >= 20:
        action = "move_forward"
    elif right >= 20:
        action = "turn_right"
    else:
        action = "reverse"
    return {"action": action, "debug_info": {"version": "baseline_v1", "front_dist": front}}


def aggressive_v2(state):
    """Always moves forward — fast, but blind to obstacles."""
    return {"action": "move_forward", "debug_info": {"version": "aggressive_v2"}}


def cautious_v3(state):
    """Goal-direction aware: aligns toward goal first, avoids all collisions."""
    goal = state["goal_direction"]
    front = float(state["front_distance"])
    left = float(state["left_distance"])
    right = float(state["right_distance"])
    if goal == "left" and left >= 20:
        action = "turn_left"
    elif goal == "right" and right >= 20:
        action = "turn_right"
    elif front >= 20:
        action = "move_forward"
    elif right >= 20:
        action = "turn_right"
    elif left >= 20:
        action = "turn_left"
    else:
        action = "reverse"
    return {"action": action, "debug_info": {"version": "cautious_v3", "goal": goal, "front_dist": front}}


# ── Scenarios ─────────────────────────────────────────────────────────────────

scenarios = [
    Scenario(
        name="open_path",
        initial_state={
            "front_distance": 80,
            "left_distance": 45,
            "right_distance": 45,
            "goal_direction": "forward",
            "previous_action": "none",
            "step_count": 0,
        },
        max_steps=6,
        metadata={"required_forward_steps": 2, "scenario_type": "navigation"},
    ),
    Scenario(
        name="blocked_all_sides",
        initial_state={
            "front_distance": 12,
            "left_distance": 10,
            "right_distance": 10,
            "goal_direction": "forward",
            "previous_action": "none",
            "step_count": 0,
        },
        max_steps=8,
        metadata={"required_forward_steps": 2, "scenario_type": "obstacle"},
    ),
    Scenario(
        name="goal_misalignment",
        initial_state={
            "front_distance": 50,
            "left_distance": 45,
            "right_distance": 45,
            "goal_direction": "right",
            "previous_action": "none",
            "step_count": 0,
        },
        max_steps=6,
        metadata={"required_forward_steps": 2, "scenario_type": "navigation"},
    ),
]


# ── Run ───────────────────────────────────────────────────────────────────────

OUTPUT_DIR = "runs/demo"

print("roboeval demo")
print(f"  policies : baseline_v1  |  aggressive_v2  |  cautious_v3")
print(f"  scenarios: {', '.join(s.name for s in scenarios)}")
print()

report = EvalRunner(
    policies=[baseline_v1, aggressive_v2, cautious_v3],
    scenarios=scenarios,
    ruleset=Ruleset([
        require_outcome("goal_reached"),
        forbid_failure("collision"),
    ]),
    baseline_policy="baseline_v1",
).run()

report.save(OUTPUT_DIR)

print(Path(OUTPUT_DIR, "report.md").read_text())
print(f"Artifacts saved to: {OUTPUT_DIR}/")
