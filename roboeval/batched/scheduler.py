"""SlotScheduler — maps a queue of (scenario, replica) tasks to N env slots.

Owns the bookkeeping for vectorized rollouts:
  * which slot is running which scenario, and which replica index of it
  * which slots are active (have a task) vs idle (queue drained)
  * when a slot terminates, whether the runner should refill it from the
    pending queue or leave it idle

The scheduler is intentionally environment- and policy-agnostic so it can be
unit-tested in isolation. The runner calls into it; the scheduler never calls
back into the runner.

Task identity:
  Each (Scenario, replica_idx) pair is one task. When ``replicas > 1`` the
  replica_idx differentiates rollouts of the same Scenario; downstream
  reporting suffixes ``#r{i}`` to ``scenario_name`` so each replica produces
  a distinct EpisodeResult row.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from roboeval.core import Scenario


@dataclass(frozen=True)
class SlotTask:
    """One pending or in-flight task: which scenario, which replica."""
    scenario: Scenario
    replica_idx: int


class SlotScheduler:
    """Assigns (Scenario, replica) tasks to a fixed bank of ``num_envs`` slots.

    Lifecycle:
        sched = SlotScheduler(num_envs=8, scenarios=[...], replicas=4)
        initial_scenarios = sched.initialize()    # len == num_envs, may include None entries
        # ... runner resets the env with initial_scenarios as a batch ...
        while not sched.is_done():
            active = sched.active_slots()
            # ... runner calls policy + env.step ...
            for slot in active:
                if terminal[slot] or max_steps_hit[slot]:
                    refilled = sched.complete_slot(slot)   # returns Optional[SlotTask]
                    if refilled:
                        # ... runner calls env.reset_slots([slot], [refilled.scenario]) ...

    Args:
        num_envs: width of the env batch — number of parallel slots
        scenarios: ordered list of scenarios to evaluate
        replicas: how many times each scenario is rolled out (D1: replication mode)
    """

    def __init__(
        self,
        num_envs: int,
        scenarios: list[Scenario],
        replicas: int = 1,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}.")
        if replicas <= 0:
            raise ValueError(f"replicas must be positive, got {replicas}.")
        self.num_envs = num_envs
        self.replicas = replicas
        self._pending: deque[SlotTask] = deque(
            SlotTask(scenario=sc, replica_idx=r)
            for sc in scenarios
            for r in range(replicas)
        )
        self._slot_tasks: list[SlotTask | None] = [None] * num_envs

    # ----- queries -----

    @property
    def total_tasks(self) -> int:
        """Total tasks scheduled across the whole run (used for sizing reports)."""
        return len(self._slot_tasks) + len(self._pending) - sum(
            1 for t in self._slot_tasks if t is None
        )

    def active_slots(self) -> list[int]:
        return [i for i, t in enumerate(self._slot_tasks) if t is not None]

    def current_task(self, slot: int) -> SlotTask:
        """Task currently assigned to ``slot``. Raises if slot is idle."""
        task = self._slot_tasks[slot]
        if task is None:
            raise ValueError(f"Slot {slot} is idle — no task currently assigned.")
        return task

    def is_done(self) -> bool:
        return not self._pending and all(t is None for t in self._slot_tasks)

    # ----- mutations -----

    def initialize(self) -> list[Scenario | None]:
        """Pull tasks off the queue and assign to every slot.

        Returns one entry per slot:
          * a Scenario for slots that received a task
          * None for slots left idle (queue smaller than num_envs)

        The runner uses this list to construct the env.reset batch. Idle
        slots can be padded with any placeholder; the runner ignores their
        outcomes.
        """
        initial: list[Scenario | None] = []
        for i in range(self.num_envs):
            if self._pending:
                task = self._pending.popleft()
                self._slot_tasks[i] = task
                initial.append(task.scenario)
            else:
                self._slot_tasks[i] = None
                initial.append(None)
        return initial

    def complete_slot(self, slot: int) -> SlotTask | None:
        """Finish the current task on ``slot``, pull the next from the queue.

        Returns the new SlotTask if the queue had pending work, or None if
        the queue is now empty (slot becomes idle). The runner uses the
        returned task to call env.reset_slots([slot], [task.scenario]).
        """
        if self._slot_tasks[slot] is None:
            raise ValueError(f"Slot {slot} has no task to complete.")
        if self._pending:
            next_task = self._pending.popleft()
            self._slot_tasks[slot] = next_task
            return next_task
        self._slot_tasks[slot] = None
        return None
