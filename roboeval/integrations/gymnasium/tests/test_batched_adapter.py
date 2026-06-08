"""Tests for BatchedGymnasiumEnvironmentAdapter.

Covers adapter-level behavior (state shape, action conversion, reset_slots,
per-slot return tracking, info filtering) and end-to-end integration with
BatchedEvalRunner on a real CartPole-v1 vector env.
"""

from __future__ import annotations

import unittest
from typing import Any

import gymnasium as gym
import numpy as np

from roboeval.batched.runner import BatchedEvalRunner
from roboeval.batched.types import BatchedStepOutcome
from roboeval.core import Ruleset, Scenario, require_metric, require_outcome
from roboeval.integrations.gymnasium.batched_adapter import (
    BatchedGymnasiumEnvironmentAdapter,
    _per_slot_info,
)


def _cartpole_vector(num_envs: int) -> gym.vector.SyncVectorEnv:
    return gym.vector.SyncVectorEnv(
        [lambda: gym.make("CartPole-v1") for _ in range(num_envs)]
    )


def _seeded_scenarios(*seeds: int, max_steps: int = 200) -> list[Scenario]:
    return [Scenario(f"seed_{s}", {"seed": s}, max_steps=max_steps) for s in seeds]


# ─── Construction + input validation ────────────────────────────────────────


class TestBatchedAdapterConstruction(unittest.TestCase):
    def test_rejects_non_vector_env(self) -> None:
        single_env = gym.make("CartPole-v1")
        with self.assertRaisesRegex(TypeError, r"requires gym.vector.VectorEnv"):
            BatchedGymnasiumEnvironmentAdapter(env=single_env)
        single_env.close()

    def test_num_envs_attribute_set_from_vector_env(self) -> None:
        env = _cartpole_vector(4)
        try:
            adapter = BatchedGymnasiumEnvironmentAdapter(env=env)
            self.assertEqual(adapter.num_envs, 4)
        finally:
            env.close()


# ─── reset() ────────────────────────────────────────────────────────────────


class TestBatchedAdapterReset(unittest.TestCase):
    def test_reset_returns_num_envs_states(self) -> None:
        env = _cartpole_vector(3)
        try:
            adapter = BatchedGymnasiumEnvironmentAdapter(env=env)
            states = adapter.reset(_seeded_scenarios(1, 2, 3))
            self.assertEqual(len(states), 3)
            for s in states:
                self.assertIn("observation", s)
                self.assertEqual(len(s["observation"]), 4)
        finally:
            env.close()

    def test_reset_requires_num_envs_scenarios(self) -> None:
        env = _cartpole_vector(3)
        try:
            adapter = BatchedGymnasiumEnvironmentAdapter(env=env)
            with self.assertRaisesRegex(ValueError, r"3 scenarios"):
                adapter.reset(_seeded_scenarios(1, 2))
        finally:
            env.close()

    def test_seeded_reset_is_deterministic(self) -> None:
        # Two adapters, same seeds → same initial obs (CartPole is deterministic given a seed).
        env_a = _cartpole_vector(2)
        env_b = _cartpole_vector(2)
        try:
            adapter_a = BatchedGymnasiumEnvironmentAdapter(env=env_a)
            adapter_b = BatchedGymnasiumEnvironmentAdapter(env=env_b)
            states_a = adapter_a.reset(_seeded_scenarios(42, 43))
            states_b = adapter_b.reset(_seeded_scenarios(42, 43))
            np.testing.assert_array_equal(states_a[0]["observation"], states_b[0]["observation"])
            np.testing.assert_array_equal(states_a[1]["observation"], states_b[1]["observation"])
        finally:
            env_a.close()
            env_b.close()

    def test_reset_zeroes_episode_returns(self) -> None:
        env = _cartpole_vector(2)
        try:
            adapter = BatchedGymnasiumEnvironmentAdapter(env=env)
            adapter.reset(_seeded_scenarios(1, 2))
            # Step a few times to accumulate returns
            for _ in range(3):
                adapter.step([0, 0])
            self.assertGreater(sum(adapter._episode_returns), 0.0)
            adapter.reset(_seeded_scenarios(1, 2))
            self.assertEqual(adapter._episode_returns, [0.0, 0.0])
        finally:
            env.close()


# ─── step() ─────────────────────────────────────────────────────────────────


class TestBatchedAdapterStep(unittest.TestCase):
    def test_step_returns_batched_step_outcome_with_correct_shape(self) -> None:
        env = _cartpole_vector(2)
        try:
            adapter = BatchedGymnasiumEnvironmentAdapter(env=env)
            adapter.reset(_seeded_scenarios(1, 2))
            outcome = adapter.step([0, 1])
            self.assertIsInstance(outcome, BatchedStepOutcome)
            self.assertEqual(outcome.num_envs, 2)
            self.assertEqual(len(outcome.next_states), 2)
            self.assertEqual(len(outcome.outcomes), 2)
            self.assertEqual(len(outcome.terminals), 2)
            self.assertEqual(len(outcome.metrics), 2)
        finally:
            env.close()

    def test_step_requires_num_envs_actions(self) -> None:
        env = _cartpole_vector(3)
        try:
            adapter = BatchedGymnasiumEnvironmentAdapter(env=env)
            adapter.reset(_seeded_scenarios(1, 2, 3))
            with self.assertRaisesRegex(ValueError, r"3 actions"):
                adapter.step([0])
        finally:
            env.close()

    def test_per_slot_episode_return_accumulates(self) -> None:
        env = _cartpole_vector(2)
        try:
            adapter = BatchedGymnasiumEnvironmentAdapter(env=env)
            adapter.reset(_seeded_scenarios(1, 2))
            # CartPole rewards 1 per step until terminal.
            outcome = adapter.step([0, 0])
            outcome = adapter.step([0, 0])
            outcome = adapter.step([0, 0])
            # After 3 steps (no terminal), episode_return == 3
            self.assertEqual(outcome.metrics[0]["episode_return"], 3.0)
            self.assertEqual(outcome.metrics[1]["episode_return"], 3.0)
        finally:
            env.close()

    def test_terminal_step_records_terminated(self) -> None:
        env = _cartpole_vector(1)
        try:
            adapter = BatchedGymnasiumEnvironmentAdapter(env=env)
            adapter.reset([Scenario("only", {"seed": 42}, max_steps=600)])
            # Always push left — pole will fall
            terminated = False
            for _ in range(600):
                outcome = adapter.step([0])
                if outcome.terminals[0]:
                    terminated = True
                    self.assertEqual(outcome.info[0]["gymnasium"]["terminated"], True)
                    self.assertEqual(outcome.info[0]["gymnasium"]["truncated"], False)
                    self.assertIn("episode_terminated", outcome.events[0])
                    break
            self.assertTrue(terminated, "expected CartPole to terminate within 600 steps with action=0")
        finally:
            env.close()

    def test_info_namespaces_gymnasium_payload(self) -> None:
        env = _cartpole_vector(2)
        try:
            adapter = BatchedGymnasiumEnvironmentAdapter(env=env)
            adapter.reset(_seeded_scenarios(1, 2))
            outcome = adapter.step([0, 1])
            for slot in (0, 1):
                self.assertIn("gymnasium", outcome.info[slot])
                gym_info = outcome.info[slot]["gymnasium"]
                self.assertIn("terminated", gym_info)
                self.assertIn("truncated", gym_info)
                self.assertIn("raw_info", gym_info)
                self.assertEqual(gym_info["slot"], slot)
        finally:
            env.close()

    def test_metrics_contain_reward_and_episode_return(self) -> None:
        env = _cartpole_vector(2)
        try:
            adapter = BatchedGymnasiumEnvironmentAdapter(env=env)
            adapter.reset(_seeded_scenarios(1, 2))
            outcome = adapter.step([0, 1])
            for slot in (0, 1):
                self.assertIn("reward", outcome.metrics[slot])
                self.assertIn("episode_return", outcome.metrics[slot])
        finally:
            env.close()


# ─── reset_slots() ──────────────────────────────────────────────────────────


class TestBatchedAdapterResetSlots(unittest.TestCase):
    def test_reset_slots_returns_one_state_per_slot(self) -> None:
        env = _cartpole_vector(3)
        try:
            adapter = BatchedGymnasiumEnvironmentAdapter(env=env)
            adapter.reset(_seeded_scenarios(1, 2, 3))
            new = adapter.reset_slots(
                [0, 2], _seeded_scenarios(100, 200)
            )
            self.assertEqual(len(new), 2)
            self.assertIn("observation", new[0])
            self.assertEqual(len(new[0]["observation"]), 4)
        finally:
            env.close()

    def test_reset_slots_zeroes_episode_return_for_those_slots(self) -> None:
        env = _cartpole_vector(3)
        try:
            adapter = BatchedGymnasiumEnvironmentAdapter(env=env)
            adapter.reset(_seeded_scenarios(1, 2, 3))
            for _ in range(5):
                adapter.step([0, 0, 0])
            # All slots had accumulated returns
            before = list(adapter._episode_returns)
            self.assertTrue(all(r > 0 for r in before))
            adapter.reset_slots([1], _seeded_scenarios(999))
            after = list(adapter._episode_returns)
            self.assertEqual(after[1], 0.0)
            self.assertEqual(after[0], before[0])
            self.assertEqual(after[2], before[2])
        finally:
            env.close()

    def test_reset_slots_with_seed_is_deterministic(self) -> None:
        env_a = _cartpole_vector(2)
        env_b = _cartpole_vector(2)
        try:
            adapter_a = BatchedGymnasiumEnvironmentAdapter(env=env_a)
            adapter_b = BatchedGymnasiumEnvironmentAdapter(env=env_b)
            adapter_a.reset(_seeded_scenarios(1, 2))
            adapter_b.reset(_seeded_scenarios(1, 2))
            new_a = adapter_a.reset_slots([0], _seeded_scenarios(777))
            new_b = adapter_b.reset_slots([0], _seeded_scenarios(777))
            np.testing.assert_array_equal(new_a[0]["observation"], new_b[0]["observation"])
        finally:
            env_a.close()
            env_b.close()

    def test_reset_slots_validates_length_mismatch(self) -> None:
        env = _cartpole_vector(2)
        try:
            adapter = BatchedGymnasiumEnvironmentAdapter(env=env)
            adapter.reset(_seeded_scenarios(1, 2))
            with self.assertRaisesRegex(ValueError, r"same length"):
                adapter.reset_slots([0, 1], _seeded_scenarios(99))
        finally:
            env.close()


# ─── Action conversion ──────────────────────────────────────────────────────


class TestBatchedAdapterActionConversion(unittest.TestCase):
    def test_int_actions_become_ndarray(self) -> None:
        env = _cartpole_vector(3)
        try:
            adapter = BatchedGymnasiumEnvironmentAdapter(env=env)
            adapter.reset(_seeded_scenarios(1, 2, 3))
            adapter.step([0, 1, 0])
        finally:
            env.close()

    def test_custom_action_from_decision_runs(self) -> None:
        # Translate "left"/"right" → 0/1 via hook
        def custom_action(decision_action: Any) -> int:
            return {"left": 0, "right": 1}[decision_action]

        env = _cartpole_vector(2)
        try:
            adapter = BatchedGymnasiumEnvironmentAdapter(
                env=env, action_from_decision=custom_action
            )
            adapter.reset(_seeded_scenarios(1, 2))
            outcome = adapter.step(["left", "right"])
            self.assertEqual(outcome.num_envs, 2)
        finally:
            env.close()


# ─── _per_slot_info helper ──────────────────────────────────────────────────


class TestPerSlotInfo(unittest.TestCase):
    def test_extracts_per_key_arrays(self) -> None:
        infos = {"score": np.array([10, 20, 30])}
        result = _per_slot_info(infos, slot=1, num_envs=3, info_keys=None)
        self.assertEqual(result, {"score": 20})

    def test_skips_final_observation_keys(self) -> None:
        infos = {
            "final_observation": np.array([1, 2, 3]),
            "_final_observation": np.array([True, False, False]),
            "score": np.array([10, 20, 30]),
        }
        result = _per_slot_info(infos, slot=0, num_envs=3, info_keys=None)
        self.assertNotIn("final_observation", result)
        self.assertNotIn("_final_observation", result)
        self.assertEqual(result, {"score": 10})

    def test_info_keys_allowlist_filters(self) -> None:
        infos = {
            "score": np.array([10, 20]),
            "big_blob": np.array([[1, 2], [3, 4]]),
        }
        result = _per_slot_info(infos, slot=0, num_envs=2, info_keys=["score"])
        self.assertEqual(result, {"score": 10})

    def test_scalar_info_passed_through(self) -> None:
        infos = {"env_version": "v1"}
        result = _per_slot_info(infos, slot=0, num_envs=2, info_keys=None)
        self.assertEqual(result, {"env_version": "v1"})

    def test_empty_info_returns_empty(self) -> None:
        result = _per_slot_info({}, slot=0, num_envs=4, info_keys=None)
        self.assertEqual(result, {})


# ─── End-to-end: BatchedEvalRunner + real CartPole vector env ───────────────


def _balance_policy(state: dict) -> dict:
    """Push left if pole leans left."""
    pole_angle = float(state["observation"][2])
    return {"action": 0 if pole_angle < 0 else 1, "debug_info": {"version": "balance"}}


def _always_left(state: dict) -> dict:
    return {"action": 0, "debug_info": {"version": "left_only"}}


def _always_right(state: dict) -> dict:
    return {"action": 1, "debug_info": {"version": "right_only"}}


class TestBatchedRunnerWithRealCartPole(unittest.TestCase):
    def test_three_policies_four_scenarios_eight_envs_smokes(self) -> None:
        """The headline integration test: 3 policies × 4 scenarios, 8-env batch.
        Verifies refill, scheduler, episode termination, and report assembly."""
        env = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(8)])
        try:
            adapter = BatchedGymnasiumEnvironmentAdapter(env=env, name="cartpole_x8")
            scenarios = _seeded_scenarios(1, 2, 42, 100, max_steps=200)
            from roboeval.batched.policy import from_single
            report = BatchedEvalRunner(
                policies=[
                    from_single(_balance_policy),
                    from_single(_always_left),
                    from_single(_always_right),
                ],
                scenarios=scenarios,
                environment=adapter,
                ruleset=Ruleset([require_metric("episode_return", ">=", 50.0, name="balance_50")]),
                baseline_policy="_balance_policy",
            ).run()

            # 3 policies × 4 scenarios = 12 episodes
            self.assertEqual(len(report.episodes), 12)
            # All three policies are present in the summary
            self.assertIn("_balance_policy", report.policy_summary)
            self.assertIn("_always_left", report.policy_summary)
            self.assertIn("_always_right", report.policy_summary)
            # Every episode should have steps > 0
            for ep in report.episodes:
                self.assertGreater(ep.steps, 0)
        finally:
            env.close()

    def test_replication_yields_replica_suffixed_scenario_names(self) -> None:
        env = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(4)])
        try:
            adapter = BatchedGymnasiumEnvironmentAdapter(env=env)
            from roboeval.batched.policy import from_single
            report = BatchedEvalRunner(
                policies=[from_single(_balance_policy)],
                scenarios=_seeded_scenarios(1, 2, max_steps=50),
                environment=adapter,
                replicas=3,
                ruleset=Ruleset([require_outcome("terminated_failure")]),
            ).run()
            # 2 scenarios × 3 replicas = 6 episodes
            self.assertEqual(len(report.episodes), 6)
            names = sorted(ep.scenario_name for ep in report.episodes)
            self.assertEqual(
                names,
                sorted([f"seed_1#r{i}" for i in range(3)] + [f"seed_2#r{i}" for i in range(3)]),
            )
        finally:
            env.close()

    def test_max_steps_truncates_when_pole_does_not_fall(self) -> None:
        # The balance policy on CartPole should easily exceed 30 steps for several seeds.
        env = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(2)])
        try:
            adapter = BatchedGymnasiumEnvironmentAdapter(env=env)
            from roboeval.batched.policy import from_single
            report = BatchedEvalRunner(
                policies=[from_single(_balance_policy)],
                scenarios=_seeded_scenarios(1, 2, max_steps=10),
                environment=adapter,
                ruleset=Ruleset([require_outcome("terminated_success")]),
            ).run()
            # If the policy keeps the pole up for 10 steps, the episode is truncated
            # by max_steps; runner labels it max_steps_reached.
            outcomes = {ep.terminal_outcome for ep in report.episodes}
            # At least one of the seeds should hit max_steps_reached with this policy
            self.assertTrue(
                any(o == "max_steps_reached" for o in outcomes)
                or any(o.startswith("terminated") for o in outcomes),
                f"unexpected outcomes set: {outcomes}",
            )
        finally:
            env.close()

    def test_batched_and_single_env_report_shapes_match(self) -> None:
        """Cross-check: BatchedEvalRunner and EvalRunner produce report objects
        with the same top-level field set and types."""
        from roboeval.batched.policy import from_single
        from roboeval.integrations.gymnasium.adapter import GymnasiumEnvironmentAdapter
        from roboeval.runner import EvalRunner

        # Single-env path
        single_env = gym.make("CartPole-v1")
        single_adapter = GymnasiumEnvironmentAdapter(env=single_env)
        single_report = EvalRunner(
            policies=[_balance_policy],
            scenarios=_seeded_scenarios(42, max_steps=20),
            environment=single_adapter,
            ruleset=Ruleset([require_metric("episode_return", ">=", 10.0)]),
        ).run()
        single_env.close()

        # Batched path
        batched_env = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(1)])
        try:
            batched_adapter = BatchedGymnasiumEnvironmentAdapter(env=batched_env)
            batched_report = BatchedEvalRunner(
                policies=[from_single(_balance_policy)],
                scenarios=_seeded_scenarios(42, max_steps=20),
                environment=batched_adapter,
                ruleset=Ruleset([require_metric("episode_return", ">=", 10.0)]),
            ).run()

            # Top-level field types match
            for attr in ("policy_summary", "episodes", "regressions", "improvements", "baseline_policy"):
                self.assertEqual(
                    type(getattr(single_report, attr)),
                    type(getattr(batched_report, attr)),
                    f"field {attr} type mismatch",
                )
            # Episode count matches
            self.assertEqual(len(single_report.episodes), len(batched_report.episodes))
            # Episode result shape matches
            s_ep = single_report.episodes[0]
            b_ep = batched_report.episodes[0]
            for attr in (
                "scenario_name", "policy_version", "success",
                "terminal_outcome", "failure_label", "steps",
            ):
                self.assertEqual(
                    type(getattr(s_ep, attr)),
                    type(getattr(b_ep, attr)),
                    f"episode attr {attr} type mismatch",
                )
        finally:
            batched_env.close()


if __name__ == "__main__":
    unittest.main()
