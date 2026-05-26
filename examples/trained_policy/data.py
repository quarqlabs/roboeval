from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from examples.trained_policy.actions import ACTIONS, GOAL_DIRECTIONS, PREVIOUS_ACTIONS


REQUIRED_COLUMNS = [
    "front_distance",
    "left_distance",
    "right_distance",
    "goal_direction",
    "previous_action",
    "step_count",
    "expert_action",
    "scenario_type",
]


def expert_action(state: dict[str, Any], safe_distance: float = 20.0) -> str:
    front = float(state["front_distance"])
    left = float(state["left_distance"])
    right = float(state["right_distance"])
    goal = str(state["goal_direction"])

    if front < safe_distance and left < 15 and right < 15:
        return "reverse"
    if front < safe_distance:
        if left >= safe_distance and left >= right:
            return "turn_left"
        if right >= safe_distance:
            return "turn_right"
        return "stop"
    if goal == "left" and left >= safe_distance:
        return "turn_left"
    if goal == "right" and right >= safe_distance:
        return "turn_right"
    return "move_forward"


def generate_synthetic_rows(count: int = 1200, seed: int = 7) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    scenario_types = [
        "open_path",
        "front_obstacle",
        "narrow_gap",
        "dead_end",
        "goal_misalignment",
        "low_distance_noise",
    ]
    for index in range(count):
        scenario_type = scenario_types[index % len(scenario_types)]
        state = _sample_state(rng, scenario_type)
        rows.append({**state, "expert_action": expert_action(state), "scenario_type": scenario_type})
    return rows


def write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def load_rows_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def clean_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clean = []
    rejected = []
    for row in rows:
        reason = _invalid_reason(row)
        if reason:
            rejected.append({**row, "reject_reason": reason})
        else:
            clean.append(_normalize_row(row))
    return clean, rejected


def split_rows(rows: list[dict[str, Any]], train_ratio: float = 0.8, seed: int = 13) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    split_index = int(len(shuffled) * train_ratio)
    return shuffled[:split_index], shuffled[split_index:]


def _sample_state(rng: random.Random, scenario_type: str) -> dict[str, Any]:
    if scenario_type == "open_path":
        front = rng.randint(45, 95)
        left = rng.randint(25, 90)
        right = rng.randint(25, 90)
        goal = rng.choice(GOAL_DIRECTIONS)
    elif scenario_type == "front_obstacle":
        front = rng.randint(5, 19)
        left = rng.randint(15, 90)
        right = rng.randint(15, 90)
        goal = rng.choice(GOAL_DIRECTIONS)
    elif scenario_type == "narrow_gap":
        front = rng.randint(10, 35)
        left = rng.randint(5, 22)
        right = rng.randint(35, 90)
        goal = "right"
    elif scenario_type == "dead_end":
        front = rng.randint(3, 18)
        left = rng.randint(3, 18)
        right = rng.randint(3, 18)
        goal = rng.choice(GOAL_DIRECTIONS)
    elif scenario_type == "goal_misalignment":
        front = rng.randint(35, 90)
        left = rng.randint(25, 90)
        right = rng.randint(25, 90)
        goal = rng.choice(["left", "right"])
    else:
        front = rng.randint(12, 24)
        left = rng.randint(20, 45)
        right = rng.randint(35, 90)
        goal = "forward"

    return {
        "front_distance": front,
        "left_distance": left,
        "right_distance": right,
        "goal_direction": goal,
        "previous_action": rng.choice(PREVIOUS_ACTIONS),
        "step_count": rng.randint(0, 6),
    }


def _invalid_reason(row: dict[str, Any]) -> str:
    missing = [column for column in REQUIRED_COLUMNS if column not in row or row[column] in ("", None)]
    if missing:
        return f"missing:{','.join(missing)}"
    try:
        distances = [float(row["front_distance"]), float(row["left_distance"]), float(row["right_distance"])]
        step_count = int(row["step_count"])
    except (TypeError, ValueError):
        return "non_numeric_distance_or_step"
    if any(distance < 0 or distance > 150 for distance in distances):
        return "distance_out_of_range"
    if step_count < 0:
        return "negative_step_count"
    if row["goal_direction"] not in GOAL_DIRECTIONS:
        return "unknown_goal_direction"
    if row["previous_action"] not in PREVIOUS_ACTIONS:
        return "unknown_previous_action"
    if row["expert_action"] not in ACTIONS:
        return "unknown_expert_action"
    return ""


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "front_distance": float(row["front_distance"]),
        "left_distance": float(row["left_distance"]),
        "right_distance": float(row["right_distance"]),
        "goal_direction": str(row["goal_direction"]),
        "previous_action": str(row["previous_action"]),
        "step_count": int(row["step_count"]),
        "expert_action": str(row["expert_action"]),
        "scenario_type": str(row["scenario_type"]),
    }
