"""BatchedEnvironmentAdapter Protocol — the vectorized counterpart of
EnvironmentAdapter.

A batched adapter wraps a vectorized simulator (Isaac Lab ManagerBasedRLEnv,
gym.vector.VectorEnv, Brax, MJX, etc.) and exposes a uniform N-env interface
to the runner.

Method semantics:
  * reset(scenarios)        — bulk reset all num_envs slots. ``len(scenarios)``
                              must equal ``num_envs``. Returns one State per
                              slot.
  * step(actions)           — apply one action per slot, return a
                              BatchedStepOutcome with num_envs entries. The
                              runner detects terminal slots from
                              ``outcome.terminals[i]``.
  * reset_slots(slots,
                scenarios)  — selective reset for slots that just terminated.
                              Slots are 0-indexed; ``len(scenarios)`` must
                              match ``len(slots)``. Returns the new initial
                              state for each reset slot. Implementations whose
                              underlying sim auto-resets (Gymnasium vector)
                              should still implement this — at minimum to
                              re-seed and to read off the post-reset obs.

The Protocol intentionally does NOT take a scenario on step(): scenarios are
fixed for the lifetime of the slot's current episode and the runner tracks
the mapping. Reset is where scenario-dependent state (seeds, options) flows
into the env.

A vectorized adapter should expose ``num_envs`` as either an attribute or a
read-only property.
"""

from __future__ import annotations

from typing import Protocol

from roboeval.core import Action, Scenario, State

from .types import BatchedState, BatchedStepOutcome


class BatchedEnvironmentAdapter(Protocol):
    """Protocol every vectorized environment adapter should implement."""

    num_envs: int

    def reset(self, scenarios: list[Scenario]) -> BatchedState:
        """Bulk-reset all slots. ``len(scenarios) == num_envs`` required."""
        ...

    def step(self, actions: list[Action]) -> BatchedStepOutcome:
        """Apply one action per slot and return the batched transition."""
        ...

    def reset_slots(
        self, slots: list[int], scenarios: list[Scenario]
    ) -> list[State]:
        """Reset a subset of slots, returning the new initial state for each.

        Called by the runner when slots terminate and get new scenario
        assignments. ``len(slots) == len(scenarios)`` required.
        """
        ...
