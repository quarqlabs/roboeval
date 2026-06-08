"""Tests for BatchedStepOutcome construction, validation, and slot extraction."""

from __future__ import annotations

import unittest

from roboeval.batched.types import BatchedStepOutcome
from roboeval.environment import StepOutcome


class TestBatchedStepOutcomeMinimalConstruction(unittest.TestCase):
    def test_constructs_with_required_fields_only(self) -> None:
        outcome = BatchedStepOutcome(
            next_states=[{"obs": 1}, {"obs": 2}, {"obs": 3}],
            outcomes=["progress", "progress", "terminated_success"],
            failure_labels=["", "", ""],
            terminals=[False, False, True],
        )
        self.assertEqual(outcome.num_envs, 3)
        self.assertIsNone(outcome.metrics)
        self.assertIsNone(outcome.events)
        self.assertIsNone(outcome.artifacts)
        self.assertIsNone(outcome.info)

    def test_num_envs_zero_is_legal(self) -> None:
        outcome = BatchedStepOutcome(
            next_states=[], outcomes=[], failure_labels=[], terminals=[]
        )
        self.assertEqual(outcome.num_envs, 0)


class TestBatchedStepOutcomeFullConstruction(unittest.TestCase):
    def test_constructs_with_all_fields(self) -> None:
        outcome = BatchedStepOutcome(
            next_states=[{"obs": 1}, {"obs": 2}],
            outcomes=["progress", "terminated_failure"],
            failure_labels=["", "collision"],
            terminals=[False, True],
            metrics=[{"reward": 1.0}, {"reward": -1.0}],
            events=[["step_logged"], ["episode_terminated"]],
            artifacts=[{}, {"frame": b"\x00\x01"}],
            info=[{"env": "a"}, {"env": "b", "error": "collision_detected"}],
        )
        self.assertEqual(outcome.num_envs, 2)
        self.assertEqual(outcome.metrics[1]["reward"], -1.0)
        self.assertEqual(outcome.events[1], ["episode_terminated"])
        self.assertEqual(outcome.info[1]["error"], "collision_detected")


class TestBatchedStepOutcomeValidation(unittest.TestCase):
    def test_outcomes_length_mismatch_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, r"outcomes.*length 2.*expected 3"):
            BatchedStepOutcome(
                next_states=[{"a": 1}, {"a": 2}, {"a": 3}],
                outcomes=["progress", "progress"],
                failure_labels=["", "", ""],
                terminals=[False, False, False],
            )

    def test_terminals_length_mismatch_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, r"terminals.*length 1.*expected 2"):
            BatchedStepOutcome(
                next_states=[{"a": 1}, {"a": 2}],
                outcomes=["progress", "progress"],
                failure_labels=["", ""],
                terminals=[False],
            )

    def test_metrics_length_mismatch_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, r"metrics.*length 1.*expected 2"):
            BatchedStepOutcome(
                next_states=[{"a": 1}, {"a": 2}],
                outcomes=["progress", "progress"],
                failure_labels=["", ""],
                terminals=[False, False],
                metrics=[{"reward": 1.0}],
            )

    def test_failure_labels_length_mismatch_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, r"failure_labels.*length 0.*expected 1"):
            BatchedStepOutcome(
                next_states=[{"a": 1}],
                outcomes=["progress"],
                failure_labels=[],
                terminals=[False],
            )

    def test_info_length_mismatch_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, r"info.*length 3.*expected 2"):
            BatchedStepOutcome(
                next_states=[{"a": 1}, {"a": 2}],
                outcomes=["progress", "progress"],
                failure_labels=["", ""],
                terminals=[False, False],
                info=[{}, {}, {}],
            )


class TestBatchedStepOutcomeSlotExtraction(unittest.TestCase):
    def test_slot_returns_single_env_step_outcome(self) -> None:
        outcome = BatchedStepOutcome(
            next_states=[{"obs": 1}, {"obs": 2}, {"obs": 3}],
            outcomes=["progress", "progress", "terminated_success"],
            failure_labels=["", "", ""],
            terminals=[False, False, True],
            metrics=[{"reward": 1.0}, {"reward": 0.5}, {"reward": 10.0}],
            events=[["a"], ["b"], ["episode_terminated"]],
            info=[{"x": 1}, {"x": 2}, {"x": 3}],
        )
        slot2 = outcome.slot(2)
        self.assertIsInstance(slot2, StepOutcome)
        self.assertEqual(slot2.next_state, {"obs": 3})
        self.assertEqual(slot2.outcome, "terminated_success")
        self.assertEqual(slot2.terminal, True)
        self.assertEqual(slot2.metrics, {"reward": 10.0})
        self.assertEqual(slot2.events, ["episode_terminated"])
        self.assertEqual(slot2.info, {"x": 3})

    def test_slot_with_none_optional_fields(self) -> None:
        outcome = BatchedStepOutcome(
            next_states=[{"obs": 1}],
            outcomes=["progress"],
            failure_labels=[""],
            terminals=[False],
        )
        slot0 = outcome.slot(0)
        self.assertIsNone(slot0.metrics)
        self.assertIsNone(slot0.events)
        self.assertIsNone(slot0.artifacts)
        self.assertIsNone(slot0.info)

    def test_slot_out_of_range_raises_index_error(self) -> None:
        outcome = BatchedStepOutcome(
            next_states=[{"obs": 1}],
            outcomes=["progress"],
            failure_labels=[""],
            terminals=[False],
        )
        with self.assertRaises(IndexError):
            outcome.slot(1)


if __name__ == "__main__":
    unittest.main()
