from __future__ import annotations


ACTIONS = ["move_forward", "turn_left", "turn_right", "stop", "reverse"]
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTIONS)}
INDEX_TO_ACTION = {index: action for action, index in ACTION_TO_INDEX.items()}

GOAL_DIRECTIONS = ["forward", "left", "right"]
PREVIOUS_ACTIONS = ["none", *ACTIONS]
