"""Batched analogs of the single-env State and StepOutcome.

Vectorized envs (Isaac Lab, gym.vector, Brax, MJX) run N envs in parallel and
return batched results. This module mirrors the single-env types from
roboeval.core and roboeval.environment, but every payload is a per-slot list of
length num_envs.

Design notes:
  * BatchedState is a list[State] alias rather than a dataclass so callers can
    slice, append, and reorder slots with plain list operations.
  * BatchedStepOutcome is a dataclass with strict length checks; mismatched
    list lengths are the most common source of bugs at the SDK/sim boundary.
  * The .slot(i) helper produces a single-env StepOutcome for that slot. The
    runner uses it to fan the batch back into per-slot episode buffers, which
    keeps the report format identical between single-env and batched runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from roboeval.core import State
from roboeval.environment import StepOutcome


BatchedState = list[State]


@dataclass
class BatchedStepOutcome:
    next_states: list[State]
    outcomes: list[str]
    failure_labels: list[str]
    terminals: list[bool]
    metrics: list[dict[str, float | int] | None] | None = None
    events: list[list[str] | None] | None = None
    artifacts: list[dict[str, object] | None] | None = None
    info: list[dict[str, object] | None] | None = None

    def __post_init__(self) -> None:
        n = len(self.next_states)
        _check_length("outcomes", self.outcomes, n)
        _check_length("failure_labels", self.failure_labels, n)
        _check_length("terminals", self.terminals, n)
        if self.metrics is not None:
            _check_length("metrics", self.metrics, n)
        if self.events is not None:
            _check_length("events", self.events, n)
        if self.artifacts is not None:
            _check_length("artifacts", self.artifacts, n)
        if self.info is not None:
            _check_length("info", self.info, n)

    @property
    def num_envs(self) -> int:
        return len(self.next_states)

    def slot(self, i: int) -> StepOutcome:
        """Return slot ``i`` as a single-env StepOutcome."""
        return StepOutcome(
            next_state=self.next_states[i],
            outcome=self.outcomes[i],
            failure_label=self.failure_labels[i],
            terminal=self.terminals[i],
            metrics=self.metrics[i] if self.metrics is not None else None,
            events=self.events[i] if self.events is not None else None,
            artifacts=self.artifacts[i] if self.artifacts is not None else None,
            info=self.info[i] if self.info is not None else None,
        )


def _check_length(field_name: str, value: list, expected: int) -> None:
    if len(value) != expected:
        raise ValueError(
            f"BatchedStepOutcome.{field_name} has length {len(value)}, "
            f"expected {expected} (num_envs)."
        )
