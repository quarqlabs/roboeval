"""Tests for SlotScheduler — task assignment, refill, replication, idle handling."""

from __future__ import annotations

import unittest

from roboeval.batched.scheduler import SlotScheduler, SlotTask
from roboeval.core import Scenario


def _scenarios(*names: str) -> list[Scenario]:
    return [Scenario(name=n, initial_state={"seed": i}, max_steps=5) for i, n in enumerate(names)]


class TestSchedulerInitialization(unittest.TestCase):
    def test_rejects_zero_or_negative_num_envs(self) -> None:
        with self.assertRaisesRegex(ValueError, r"num_envs must be positive"):
            SlotScheduler(num_envs=0, scenarios=_scenarios("a"))
        with self.assertRaisesRegex(ValueError, r"num_envs must be positive"):
            SlotScheduler(num_envs=-2, scenarios=_scenarios("a"))

    def test_rejects_zero_replicas(self) -> None:
        with self.assertRaisesRegex(ValueError, r"replicas must be positive"):
            SlotScheduler(num_envs=4, scenarios=_scenarios("a"), replicas=0)

    def test_initialize_fills_slots_when_tasks_ge_slots(self) -> None:
        sched = SlotScheduler(num_envs=2, scenarios=_scenarios("a", "b", "c"))
        initial = sched.initialize()
        self.assertEqual([sc.name for sc in initial], ["a", "b"])
        self.assertEqual(sched.active_slots(), [0, 1])
        self.assertFalse(sched.is_done())

    def test_initialize_pads_with_none_when_tasks_lt_slots(self) -> None:
        sched = SlotScheduler(num_envs=5, scenarios=_scenarios("a", "b"))
        initial = sched.initialize()
        self.assertEqual([sc.name if sc else None for sc in initial], ["a", "b", None, None, None])
        self.assertEqual(sched.active_slots(), [0, 1])

    def test_initialize_with_zero_scenarios_is_legal(self) -> None:
        sched = SlotScheduler(num_envs=4, scenarios=[])
        initial = sched.initialize()
        self.assertEqual(initial, [None, None, None, None])
        self.assertTrue(sched.is_done())
        self.assertEqual(sched.active_slots(), [])


class TestSchedulerReplication(unittest.TestCase):
    def test_three_scenarios_with_two_replicas_yields_six_tasks(self) -> None:
        sched = SlotScheduler(num_envs=2, scenarios=_scenarios("a", "b", "c"), replicas=2)
        initial = sched.initialize()
        # First 2 tasks: (a, r=0), (a, r=1)
        self.assertEqual(sched.current_task(0), SlotTask(scenario=initial[0], replica_idx=0))
        self.assertEqual(sched.current_task(1), SlotTask(scenario=initial[1], replica_idx=1))
        self.assertEqual(initial[0].name, "a")
        self.assertEqual(initial[1].name, "a")  # second replica of "a"
        # 4 more tasks pending
        sched.complete_slot(0)  # -> (b, r=0)
        sched.complete_slot(1)  # -> (b, r=1)
        sched.complete_slot(0)  # -> (c, r=0)
        sched.complete_slot(1)  # -> (c, r=1)
        # next two complete drain the queue
        next_task = sched.complete_slot(0)
        self.assertIsNone(next_task)
        next_task = sched.complete_slot(1)
        self.assertIsNone(next_task)
        self.assertTrue(sched.is_done())

    def test_replica_indices_increment_per_scenario(self) -> None:
        sched = SlotScheduler(num_envs=1, scenarios=_scenarios("solo"), replicas=4)
        sched.initialize()
        seen_indices = [sched.current_task(0).replica_idx]
        for _ in range(3):
            task = sched.complete_slot(0)
            self.assertIsNotNone(task)
            seen_indices.append(task.replica_idx)
        self.assertEqual(seen_indices, [0, 1, 2, 3])
        self.assertIsNone(sched.complete_slot(0))


class TestSchedulerRefill(unittest.TestCase):
    def test_complete_slot_pulls_next_pending(self) -> None:
        sched = SlotScheduler(num_envs=2, scenarios=_scenarios("a", "b", "c", "d"))
        sched.initialize()
        refilled = sched.complete_slot(0)
        self.assertIsNotNone(refilled)
        self.assertEqual(refilled.scenario.name, "c")
        self.assertEqual(sched.current_task(0).scenario.name, "c")

    def test_complete_slot_returns_none_when_queue_empty(self) -> None:
        sched = SlotScheduler(num_envs=4, scenarios=_scenarios("a", "b"))
        sched.initialize()
        # Only 2 tasks, 4 slots — slots 2,3 are already idle
        self.assertEqual(sched.active_slots(), [0, 1])
        self.assertIsNone(sched.complete_slot(0))
        self.assertEqual(sched.active_slots(), [1])
        self.assertIsNone(sched.complete_slot(1))
        self.assertEqual(sched.active_slots(), [])
        self.assertTrue(sched.is_done())

    def test_complete_slot_on_idle_slot_raises(self) -> None:
        sched = SlotScheduler(num_envs=4, scenarios=_scenarios("a"))
        sched.initialize()
        with self.assertRaisesRegex(ValueError, r"Slot 3 has no task"):
            sched.complete_slot(3)

    def test_current_task_on_idle_slot_raises(self) -> None:
        sched = SlotScheduler(num_envs=4, scenarios=_scenarios("a"))
        sched.initialize()
        with self.assertRaisesRegex(ValueError, r"Slot 2 is idle"):
            sched.current_task(2)


class TestSchedulerEndToEnd(unittest.TestCase):
    def test_drains_queue_correctly_with_mixed_completion_order(self) -> None:
        """Slots terminate out of order — scheduler must still drain the queue
        without losing or duplicating tasks."""
        sched = SlotScheduler(num_envs=3, scenarios=_scenarios("a", "b", "c", "d", "e", "f", "g"))
        sched.initialize()
        # Started: slot0=a, slot1=b, slot2=c. Pending: d,e,f,g.

        seen = ["a", "b", "c"]  # initial assignments

        # slot 2 finishes first
        t = sched.complete_slot(2)
        seen.append(t.scenario.name)  # d
        # slot 0 finishes
        t = sched.complete_slot(0)
        seen.append(t.scenario.name)  # e
        # slot 2 again
        t = sched.complete_slot(2)
        seen.append(t.scenario.name)  # f
        # slot 1 finishes
        t = sched.complete_slot(1)
        seen.append(t.scenario.name)  # g

        # Now queue empty. Remaining slot completions return None.
        self.assertIsNone(sched.complete_slot(2))
        self.assertIsNone(sched.complete_slot(0))
        self.assertIsNone(sched.complete_slot(1))
        self.assertTrue(sched.is_done())

        # Every scenario ran exactly once
        self.assertEqual(sorted(seen), ["a", "b", "c", "d", "e", "f", "g"])


if __name__ == "__main__":
    unittest.main()
