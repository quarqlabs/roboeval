"""Tests for BatchedPolicyAdapter normalization and the from_single shim."""

from __future__ import annotations

import unittest

from roboeval.batched.policy import (
    BatchedPolicyAdapter,
    from_single,
    normalize_batched_policy,
)
from roboeval.core import Decision


class TestBatchedPolicyAdapterNormalizesItems(unittest.TestCase):
    def test_normalizes_dict_returns(self) -> None:
        def policy(states):
            return [{"action": i, "debug_info": {"slot": i}} for i in range(len(states))]

        adapter = BatchedPolicyAdapter(name="dict_policy", policy=policy)
        decisions = adapter.decide([{}, {}, {}])
        self.assertEqual(len(decisions), 3)
        self.assertTrue(all(isinstance(d, Decision) for d in decisions))
        self.assertEqual(decisions[0].action, 0)
        self.assertEqual(decisions[2].debug_info["slot"], 2)

    def test_normalizes_tuple_returns(self) -> None:
        def policy(states):
            return [(i, {"version": "v1"}) for i in range(len(states))]

        adapter = BatchedPolicyAdapter(name="tuple_policy", policy=policy)
        decisions = adapter.decide([{}, {}])
        self.assertEqual(decisions[0].action, 0)
        self.assertEqual(decisions[1].debug_info, {"version": "v1"})

    def test_normalizes_raw_action_returns(self) -> None:
        def policy(states):
            return [0 for _ in states]

        adapter = BatchedPolicyAdapter(name="raw_policy", policy=policy)
        decisions = adapter.decide([{}, {}])
        self.assertEqual(decisions[0].action, 0)
        self.assertEqual(decisions[0].debug_info, {})

    def test_normalizes_decision_returns_unchanged(self) -> None:
        canned = [Decision(action=42, debug_info={"k": "v"})]

        def policy(states):
            return canned

        adapter = BatchedPolicyAdapter(name="decision_policy", policy=policy)
        decisions = adapter.decide([{}])
        self.assertEqual(decisions[0].action, 42)
        self.assertEqual(decisions[0].debug_info, {"k": "v"})

    def test_mixed_return_shapes_in_one_batch(self) -> None:
        def policy(states):
            return [{"action": 1}, (2, {"version": "v2"}), 3]

        adapter = BatchedPolicyAdapter(name="mixed", policy=policy)
        decisions = adapter.decide([{}, {}, {}])
        self.assertEqual([d.action for d in decisions], [1, 2, 3])
        self.assertEqual(decisions[1].debug_info, {"version": "v2"})


class TestBatchedPolicyAdapterValidation(unittest.TestCase):
    def test_non_list_return_raises(self) -> None:
        def bad_policy(states):
            return {"oops": "dict not list"}

        adapter = BatchedPolicyAdapter(name="bad", policy=bad_policy)
        with self.assertRaisesRegex(TypeError, r"must return a list"):
            adapter.decide([{}, {}])

    def test_length_mismatch_raises(self) -> None:
        def short_policy(states):
            return [0]

        adapter = BatchedPolicyAdapter(name="short", policy=short_policy)
        with self.assertRaisesRegex(ValueError, r"returned 1 decisions for 3 input"):
            adapter.decide([{}, {}, {}])

    def test_non_callable_no_decide_raises(self) -> None:
        adapter = BatchedPolicyAdapter(name="bad", policy=object())
        with self.assertRaisesRegex(TypeError, r"is not callable and has no decide"):
            adapter.decide([{}])

    def test_object_with_decide_method_works(self) -> None:
        class CtrlPolicy:
            def decide(self, states):
                return [{"action": "noop"} for _ in states]

        adapter = BatchedPolicyAdapter(name="ctrl", policy=CtrlPolicy())
        decisions = adapter.decide([{}, {}])
        self.assertEqual(decisions[0].action, "noop")


class TestNormalizeBatchedPolicy(unittest.TestCase):
    def test_passes_through_existing_adapter(self) -> None:
        def f(states):
            return [0] * len(states)

        original = BatchedPolicyAdapter(name="orig", policy=f)
        self.assertIs(normalize_batched_policy(original), original)

    def test_uses_function_name_when_no_name_given(self) -> None:
        def my_named_policy(states):
            return [0] * len(states)

        adapter = normalize_batched_policy(my_named_policy)
        self.assertEqual(adapter.name, "my_named_policy")

    def test_uses_version_attribute_if_present(self) -> None:
        class VersionedPolicy:
            version = "v3.1.4"

            def __call__(self, states):
                return [0] * len(states)

        adapter = normalize_batched_policy(VersionedPolicy())
        self.assertEqual(adapter.name, "v3.1.4")


class TestFromSingleShim(unittest.TestCase):
    def test_wraps_single_state_policy(self) -> None:
        def single(state):
            return {"action": state["x"] * 2}

        batched = from_single(single)
        decisions = batched.decide([{"x": 1}, {"x": 2}, {"x": 5}])
        self.assertEqual([d.action for d in decisions], [2, 4, 10])

    def test_inherits_function_name(self) -> None:
        def baseline_policy(state):
            return 0

        batched = from_single(baseline_policy)
        self.assertEqual(batched.name, "baseline_policy")

    def test_explicit_name_wins(self) -> None:
        def fn(state):
            return 0

        batched = from_single(fn, name="renamed")
        self.assertEqual(batched.name, "renamed")

    def test_returns_normalized_decisions(self) -> None:
        def returns_tuple(state):
            return (state["v"], {"x": "y"})

        batched = from_single(returns_tuple)
        decisions = batched.decide([{"v": 7}])
        self.assertEqual(decisions[0].action, 7)
        self.assertEqual(decisions[0].debug_info, {"x": "y"})


if __name__ == "__main__":
    unittest.main()
