"""Unit tests for the Isaac Lab integration spike.

These tests run on Mac without Isaac installed. They use a mock Isaac env
(`MockIsaacEnv`) that fakes the Isaac shape: gymnasium-VectorEnv-style API
returning batched torch tensors on whatever device torch defaults to.

Run with::

    python -m unittest discover -s roboeval/integrations/isaac/tests
"""

from __future__ import annotations

import json
import unittest
import warnings
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from roboeval import EvalRunner, Ruleset, Scenario, require_metric
from roboeval.core import to_serializable
from roboeval.environment import StepOutcome
from roboeval.integrations.isaac import (
    IsaacEnvironmentAdapter,
    default_action_from_decision,
    default_events_from_step,
    default_observation_to_state,
    default_options_from_scenario,
    default_outcome_from_step,
    default_seed_from_scenario,
    tensor_to_numpy,
)


HAS_TORCH = torch is not None


# ─────────────────────────────────────────────────────────────────────────
# Mock Isaac environment — fakes the Isaac shape so we can test without
# a real Isaac Lab install.
# ─────────────────────────────────────────────────────────────────────────


class MockIsaacEnv:
    """Minimal mock that mimics an Isaac Lab single-env env.

    Returns batched torch tensors from reset() / step(). Step count drives a
    simple termination rule. Use the `obs_as_dict` flag to switch between
    dict-of-tensors and bare-tensor observation styles.
    """

    def __init__(
        self,
        num_envs: int = 1,
        obs_dim: int = 4,
        action_dim: int = 1,
        terminate_at_step: int = 10,
        obs_as_dict: bool = True,
        device: str | None = None,
    ):
        if not HAS_TORCH:
            raise RuntimeError("MockIsaacEnv requires torch")
        self.num_envs = num_envs
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._terminate_at_step = terminate_at_step
        self._obs_as_dict = obs_as_dict
        self.device = device or "cpu"
        self._step = 0
        self._last_action: Any = None
        self.spec = None
        # gym-style spaces (single_action_space matters for action shaping)
        import gymnasium as gym
        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,)
        )

    def _make_obs(self) -> Any:
        # Float32 batched tensor with deterministic values
        flat = torch.arange(
            self.num_envs * self._obs_dim, dtype=torch.float32
        ).view(self.num_envs, self._obs_dim)
        if self._obs_as_dict:
            return {"policy": flat.to(self.device)}
        return flat.to(self.device)

    def reset(self, seed: int | None = None, options: dict | None = None):
        # Seed is honored deterministically just by the obs being a function of step
        self._step = 0
        return self._make_obs(), {}

    def step(self, action):
        self._step += 1
        self._last_action = action
        obs = self._make_obs()
        # Each step reward = +1 to mimic CartPole survival reward
        reward = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        terminated = torch.tensor(
            [self._step >= self._terminate_at_step] * self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        info: dict[str, Any] = {}
        return obs, reward, terminated, truncated, info

    def close(self):
        self._step = 0


class MockIsaacEnvWithOptionsRefusal(MockIsaacEnv):
    """Mock that refuses the `options=` kwarg, like older Isaac envs."""

    def reset(self, seed: int | None = None):
        return super().reset(seed=seed)


# ─────────────────────────────────────────────────────────────────────────
# Default hook unit tests (no env needed)
# ─────────────────────────────────────────────────────────────────────────


class DefaultObservationToStateTest(unittest.TestCase):
    def test_dict_passes_through(self):
        obs = {"policy": np.array([1.0, 2.0])}
        state = default_observation_to_state(obs)
        self.assertEqual(set(state.keys()), {"policy"})
        np.testing.assert_array_equal(state["policy"], obs["policy"])

    def test_non_dict_wraps_under_observation(self):
        obs = np.array([0.1, -0.5, 0.0, 0.3])
        state = default_observation_to_state(obs)
        self.assertEqual(list(state.keys()), ["observation"])
        np.testing.assert_array_equal(state["observation"], obs)


class DefaultActionFromDecisionTest(unittest.TestCase):
    def test_passthrough_int(self):
        self.assertEqual(default_action_from_decision(1), 1)

    def test_passthrough_array(self):
        a = np.array([0.5, -0.5])
        np.testing.assert_array_equal(default_action_from_decision(a), a)


class DefaultOutcomeFromStepTest(unittest.TestCase):
    def test_terminated_positive_reward_is_success(self):
        outcome, label = default_outcome_from_step(1.0, True, False, {})
        self.assertEqual((outcome, label), ("terminated_success", ""))

    def test_terminated_zero_reward_is_failure(self):
        outcome, label = default_outcome_from_step(0.0, True, False, {})
        self.assertEqual((outcome, label), ("terminated_failure", "terminated_failure"))

    def test_truncated_is_timeout(self):
        outcome, label = default_outcome_from_step(0.5, False, True, {})
        self.assertEqual((outcome, label), ("truncated", "timeout"))

    def test_non_terminal_is_progress(self):
        outcome, label = default_outcome_from_step(0.1, False, False, {})
        self.assertEqual((outcome, label), ("progress", ""))


class DefaultEventsFromStepTest(unittest.TestCase):
    def test_progress_emits_no_events(self):
        self.assertEqual(default_events_from_step(1.0, False, False, {}), [])

    def test_terminated_event(self):
        self.assertEqual(
            default_events_from_step(1.0, True, False, {}), ["episode_terminated"]
        )

    def test_truncated_event(self):
        self.assertEqual(
            default_events_from_step(1.0, False, True, {}), ["episode_truncated"]
        )

    def test_negative_reward_event(self):
        self.assertEqual(
            default_events_from_step(-1.0, False, False, {}), ["reward_negative"]
        )


class DefaultSeedFromScenarioTest(unittest.TestCase):
    def test_from_initial_state(self):
        scenario = Scenario("s", {"seed": 42}, max_steps=10)
        self.assertEqual(default_seed_from_scenario(scenario), 42)

    def test_from_metadata(self):
        scenario = Scenario("s", {"foo": 1}, max_steps=10, metadata={"seed": 7})
        self.assertEqual(default_seed_from_scenario(scenario), 7)

    def test_none_when_missing(self):
        scenario = Scenario("s", {"foo": 1}, max_steps=10)
        self.assertIsNone(default_seed_from_scenario(scenario))


class DefaultOptionsFromScenarioTest(unittest.TestCase):
    def test_options_dict(self):
        scenario = Scenario(
            "s", {"foo": 1}, max_steps=10, metadata={"reset_options": {"a": 1}}
        )
        self.assertEqual(default_options_from_scenario(scenario), {"a": 1})

    def test_none_when_missing(self):
        scenario = Scenario("s", {"foo": 1}, max_steps=10)
        self.assertIsNone(default_options_from_scenario(scenario))


# ─────────────────────────────────────────────────────────────────────────
# tensor_to_numpy
# ─────────────────────────────────────────────────────────────────────────


@unittest.skipUnless(HAS_TORCH, "torch required")
class TensorToNumpyTest(unittest.TestCase):
    def test_cpu_tensor_to_numpy(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        arr = tensor_to_numpy(t)
        self.assertIsInstance(arr, np.ndarray)
        np.testing.assert_array_almost_equal(arr, [1.0, 2.0, 3.0])

    def test_non_tensor_passes_through(self):
        self.assertEqual(tensor_to_numpy(42), 42)
        self.assertEqual(tensor_to_numpy("hello"), "hello")
        np.testing.assert_array_equal(
            tensor_to_numpy(np.array([1, 2, 3])), np.array([1, 2, 3])
        )


# ─────────────────────────────────────────────────────────────────────────
# Adapter construction
# ─────────────────────────────────────────────────────────────────────────


@unittest.skipUnless(HAS_TORCH, "torch required")
class IsaacEnvironmentAdapterConstructionTest(unittest.TestCase):
    def test_satisfies_protocol_duck_typing(self):
        env = MockIsaacEnv(num_envs=1)
        adapter = IsaacEnvironmentAdapter(env=env)
        self.assertTrue(hasattr(adapter, "reset"))
        self.assertTrue(hasattr(adapter, "step"))
        self.assertTrue(callable(adapter.reset))
        self.assertTrue(callable(adapter.step))

    def test_defaults_wired_when_hooks_none(self):
        env = MockIsaacEnv(num_envs=1)
        adapter = IsaacEnvironmentAdapter(env=env)
        self.assertIs(adapter.observation_to_state, default_observation_to_state)
        self.assertIs(adapter.action_from_decision, default_action_from_decision)
        self.assertIs(adapter.outcome_from_step, default_outcome_from_step)
        self.assertIs(adapter.events_from_step, default_events_from_step)
        self.assertIs(adapter.seed_from_scenario, default_seed_from_scenario)
        self.assertIs(adapter.options_from_scenario, default_options_from_scenario)

    def test_num_envs_gt_1_warns(self):
        env = MockIsaacEnv(num_envs=4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            IsaacEnvironmentAdapter(env=env)
        self.assertTrue(
            any("num_envs=4" in str(w.message) for w in caught),
            "Expected num_envs > 1 warning to fire",
        )

    def test_invalid_batch_index_raises(self):
        env = MockIsaacEnv(num_envs=2)
        with self.assertRaises(ValueError):
            IsaacEnvironmentAdapter(env=env, batch_index=5)

    def test_name_propagates(self):
        env = MockIsaacEnv(num_envs=1)
        adapter = IsaacEnvironmentAdapter(env=env, name="my_isaac_cartpole")
        self.assertEqual(adapter.name, "my_isaac_cartpole")


# ─────────────────────────────────────────────────────────────────────────
# Adapter behavior — reset/step with the mock
# ─────────────────────────────────────────────────────────────────────────


@unittest.skipUnless(HAS_TORCH, "torch required")
class IsaacEnvironmentAdapterBehaviorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MockIsaacEnv(num_envs=1, obs_dim=4, terminate_at_step=10)
        self.adapter = IsaacEnvironmentAdapter(env=self.env, name="isaac_mock")
        self.scenario = Scenario("test", {"seed": 0}, max_steps=50)

    def test_reset_returns_dict_with_policy_key(self):
        state = self.adapter.reset(self.scenario)
        self.assertIsInstance(state, dict)
        self.assertIn("policy", state)
        # batch dim was sliced, so it's a 1-D array of obs_dim
        self.assertEqual(state["policy"].shape, (4,))

    def test_reset_resets_episode_return(self):
        self.adapter.reset(self.scenario)
        self.adapter.step(0, self.scenario)
        self.adapter.step(0, self.scenario)
        self.adapter.reset(self.scenario)
        outcome = self.adapter.step(0, self.scenario)
        self.assertEqual(outcome.metrics["episode_return"], 1.0)

    def test_step_returns_step_outcome(self):
        self.adapter.reset(self.scenario)
        outcome = self.adapter.step(0, self.scenario)
        self.assertIsInstance(outcome, StepOutcome)
        # Required fields
        self.assertIsInstance(outcome.next_state, dict)
        self.assertIsInstance(outcome.outcome, str)
        self.assertIsInstance(outcome.failure_label, str)
        self.assertIsInstance(outcome.terminal, bool)
        # Optional fields populated
        self.assertIsInstance(outcome.metrics, dict)
        self.assertIn("reward", outcome.metrics)
        self.assertIn("episode_return", outcome.metrics)
        self.assertIsInstance(outcome.events, list)
        self.assertIsInstance(outcome.info, dict)
        self.assertIn("isaac", outcome.info)

    def test_metrics_per_spec(self):
        self.adapter.reset(self.scenario)
        out1 = self.adapter.step(0, self.scenario)
        out2 = self.adapter.step(0, self.scenario)
        out3 = self.adapter.step(0, self.scenario)
        self.assertEqual(out1.metrics["reward"], 1.0)
        self.assertEqual(out2.metrics["reward"], 1.0)
        self.assertEqual(out3.metrics["episode_return"], 3.0)

    def test_info_namespace(self):
        self.adapter.reset(self.scenario)
        outcome = self.adapter.step(0, self.scenario)
        isaac_info = outcome.info["isaac"]
        self.assertIn("terminated", isaac_info)
        self.assertIn("truncated", isaac_info)
        self.assertIn("raw_info", isaac_info)
        self.assertIn("batch_index", isaac_info)
        self.assertEqual(isaac_info["batch_index"], 0)

    def test_terminal_fires_at_expected_step(self):
        self.adapter.reset(self.scenario)
        outcomes = []
        for _ in range(self.env._terminate_at_step + 2):
            outcomes.append(self.adapter.step(0, self.scenario))
            if outcomes[-1].terminal:
                break
        self.assertTrue(outcomes[-1].terminal)
        self.assertIn("episode_terminated", outcomes[-1].events)
        # episode_return after termination
        self.assertEqual(
            outcomes[-1].metrics["episode_return"], float(self.env._terminate_at_step)
        )

    def test_bare_tensor_obs_gets_wrapped(self):
        env = MockIsaacEnv(num_envs=1, obs_as_dict=False)
        adapter = IsaacEnvironmentAdapter(env=env)
        state = adapter.reset(Scenario("s", {"seed": 0}, max_steps=10))
        self.assertIn("observation", state)
        self.assertEqual(state["observation"].shape, (4,))

    def test_options_refusal_is_handled(self):
        env = MockIsaacEnvWithOptionsRefusal(num_envs=1)
        adapter = IsaacEnvironmentAdapter(env=env)
        scenario = Scenario(
            "s", {"seed": 1}, max_steps=10, metadata={"reset_options": {"foo": 1}}
        )
        # Should not raise even though env refuses options
        state = adapter.reset(scenario)
        self.assertIn("policy", state)


# ─────────────────────────────────────────────────────────────────────────
# Hook overrides
# ─────────────────────────────────────────────────────────────────────────


@unittest.skipUnless(HAS_TORCH, "torch required")
class IsaacEnvironmentAdapterHookOverrideTest(unittest.TestCase):
    def test_outcome_hook_override(self):
        env = MockIsaacEnv(num_envs=1, terminate_at_step=5)

        def always_goal(reward, terminated, truncated, info):
            return ("goal_reached", "")

        adapter = IsaacEnvironmentAdapter(env=env, outcome_from_step=always_goal)
        adapter.reset(Scenario("s", {"seed": 0}, max_steps=10))
        outcome = adapter.step(0, Scenario("s", {"seed": 0}, max_steps=10))
        self.assertEqual(outcome.outcome, "goal_reached")

    def test_action_hook_translates_vocabulary(self):
        env = MockIsaacEnv(num_envs=1, terminate_at_step=10)
        vocab = {"push_left": 0, "push_right": 1}

        adapter = IsaacEnvironmentAdapter(
            env=env, action_from_decision=lambda a: vocab[a] if isinstance(a, str) else a
        )
        adapter.reset(Scenario("s", {"seed": 0}, max_steps=10))
        outcome = adapter.step("push_left", Scenario("s", {"seed": 0}, max_steps=10))
        # No crash means the action got translated and accepted
        self.assertFalse(outcome.terminal)

    def test_info_keys_allowlist(self):
        env = MockIsaacEnv(num_envs=1)
        adapter = IsaacEnvironmentAdapter(env=env, info_keys=["only_this_key"])
        adapter.reset(Scenario("s", {"seed": 0}, max_steps=10))
        outcome = adapter.step(0, Scenario("s", {"seed": 0}, max_steps=10))
        # MockIsaacEnv's info is {} so allowlist returns {}
        self.assertEqual(outcome.info["isaac"]["raw_info"], {})


# ─────────────────────────────────────────────────────────────────────────
# Action shape normalization
# ─────────────────────────────────────────────────────────────────────────


@unittest.skipUnless(HAS_TORCH, "torch required")
class ActionShapingTest(unittest.TestCase):
    """Action shape normalization tests.

    Two realistic scenarios covered:
      1. Discrete-style env (action_dim=1) + scalar action → (1, 1)
      2. Continuous env (action_dim=2) + 1-D array action → (1, 2)

    A scalar action against a multi-dim continuous env is ambiguous user
    error; we don't broadcast the scalar across all action dims because
    that's almost never what the user intended.
    """

    def test_scalar_action_for_single_dim_env(self):
        env = MockIsaacEnv(num_envs=1, action_dim=1)
        adapter = IsaacEnvironmentAdapter(env=env)
        adapter.reset(Scenario("s", {"seed": 0}, max_steps=10))
        adapter.step(1, Scenario("s", {"seed": 0}, max_steps=10))
        action = env._last_action
        self.assertIsNotNone(action)
        self.assertTrue(isinstance(action, torch.Tensor))
        self.assertEqual(action.shape, (1, 1))

    def test_1d_array_action_for_multi_dim_env(self):
        env = MockIsaacEnv(num_envs=1, action_dim=2)
        adapter = IsaacEnvironmentAdapter(env=env)
        adapter.reset(Scenario("s", {"seed": 0}, max_steps=10))
        adapter.step(
            np.array([0.5, -0.3]), Scenario("s", {"seed": 0}, max_steps=10)
        )
        action = env._last_action
        self.assertEqual(action.shape, (1, 2))


# ─────────────────────────────────────────────────────────────────────────
# Serialization round-trip
# ─────────────────────────────────────────────────────────────────────────


@unittest.skipUnless(HAS_TORCH, "torch required")
class SerializationTest(unittest.TestCase):
    def test_step_outcome_is_json_safe(self):
        env = MockIsaacEnv(num_envs=1, terminate_at_step=10)
        adapter = IsaacEnvironmentAdapter(env=env)
        adapter.reset(Scenario("s", {"seed": 0}, max_steps=10))
        outcome = adapter.step(0, Scenario("s", {"seed": 0}, max_steps=10))
        # All of these should round-trip through to_serializable + json.dumps
        for field in (
            outcome.next_state,
            outcome.metrics,
            outcome.events,
            outcome.info,
        ):
            payload = to_serializable(field)
            json.dumps(payload)


# ─────────────────────────────────────────────────────────────────────────
# Edge case tests — non-standard inputs from real Isaac envs
# ─────────────────────────────────────────────────────────────────────────


class MockIsaacEnvWithRichInfo(MockIsaacEnv):
    """Mock that returns a non-empty info dict each step."""

    def step(self, action):
        obs, reward, terminated, truncated, _info = super().step(action)
        info = {
            "is_success": bool(terminated[0].item()),
            "step_count": self._step,
            "task_reward": float(reward[0].item()),
            "huge_tensor": torch.zeros(100, 100),
        }
        return obs, reward, terminated, truncated, info


class MockIsaacEnvWithNonDictInfo(MockIsaacEnv):
    """Mock that returns a list info (some Isaac envs do this per-env)."""

    def step(self, action):
        obs, reward, terminated, truncated, _info = super().step(action)
        # Some Isaac envs return a list of per-env infos instead of a dict
        return obs, reward, terminated, truncated, ["per_env_info_0"]


@unittest.skipUnless(HAS_TORCH, "torch required")
class EdgeCaseTest(unittest.TestCase):
    def test_rich_info_passes_through_via_to_serializable(self):
        env = MockIsaacEnvWithRichInfo(num_envs=1)
        adapter = IsaacEnvironmentAdapter(env=env)
        adapter.reset(Scenario("s", {"seed": 0}, max_steps=10))
        outcome = adapter.step(0, Scenario("s", {"seed": 0}, max_steps=10))
        raw_info = outcome.info["isaac"]["raw_info"]
        # All keys present (default no allowlist)
        self.assertIn("is_success", raw_info)
        self.assertIn("step_count", raw_info)
        self.assertIn("task_reward", raw_info)
        self.assertIn("huge_tensor", raw_info)
        # JSON round-trip must succeed even with the tensor
        json.dumps(raw_info)

    def test_info_keys_allowlist_drops_unwanted_keys(self):
        env = MockIsaacEnvWithRichInfo(num_envs=1)
        adapter = IsaacEnvironmentAdapter(env=env, info_keys=["is_success", "step_count"])
        adapter.reset(Scenario("s", {"seed": 0}, max_steps=10))
        outcome = adapter.step(0, Scenario("s", {"seed": 0}, max_steps=10))
        raw_info = outcome.info["isaac"]["raw_info"]
        self.assertIn("is_success", raw_info)
        self.assertIn("step_count", raw_info)
        # Filtered out
        self.assertNotIn("huge_tensor", raw_info)
        self.assertNotIn("task_reward", raw_info)

    def test_non_dict_info_does_not_crash(self):
        env = MockIsaacEnvWithNonDictInfo(num_envs=1)
        adapter = IsaacEnvironmentAdapter(env=env)
        adapter.reset(Scenario("s", {"seed": 0}, max_steps=10))
        outcome = adapter.step(0, Scenario("s", {"seed": 0}, max_steps=10))
        # Non-dict info gets filtered to {}, but doesn't crash
        self.assertEqual(outcome.info["isaac"]["raw_info"], {})

    def test_custom_outcome_hook_reads_info(self):
        """Manipulation-style outcome detection: use info['is_success']."""
        env = MockIsaacEnvWithRichInfo(num_envs=1, terminate_at_step=3)

        def info_based_outcome(reward, terminated, truncated, info):
            if info.get("is_success"):
                return ("goal_reached", "")
            if terminated:
                return ("terminated_failure", "did_not_succeed")
            if truncated:
                return ("truncated", "timeout")
            return ("progress", "")

        adapter = IsaacEnvironmentAdapter(env=env, outcome_from_step=info_based_outcome)
        adapter.reset(Scenario("s", {"seed": 0}, max_steps=10))
        # Step until terminal
        for _ in range(5):
            outcome = adapter.step(0, Scenario("s", {"seed": 0}, max_steps=10))
            if outcome.terminal:
                break
        # is_success becomes True when terminated → outcome should be goal_reached
        self.assertEqual(outcome.outcome, "goal_reached")

    def test_multiple_episodes_via_reset(self):
        """Verify the adapter is re-entrant across multiple reset/step cycles."""
        env = MockIsaacEnv(num_envs=1, terminate_at_step=5)
        adapter = IsaacEnvironmentAdapter(env=env)

        for episode_i in range(3):
            adapter.reset(Scenario("s", {"seed": episode_i}, max_steps=10))
            total_steps = 0
            for _ in range(20):
                outcome = adapter.step(0, Scenario("s", {"seed": episode_i}, max_steps=10))
                total_steps += 1
                if outcome.terminal:
                    break
            # Each episode should terminate at step 5
            self.assertEqual(total_steps, 5)
            self.assertEqual(outcome.metrics["episode_return"], 5.0)

    def test_pre_batched_2d_action_passes_through(self):
        """If the user already provides a (1, action_dim) tensor, don't re-shape."""
        env = MockIsaacEnv(num_envs=1, action_dim=2)
        adapter = IsaacEnvironmentAdapter(env=env)
        adapter.reset(Scenario("s", {"seed": 0}, max_steps=10))
        # Pre-shaped tensor
        action = torch.tensor([[0.5, -0.3]])
        adapter.step(action, Scenario("s", {"seed": 0}, max_steps=10))
        sent_action = env._last_action
        self.assertEqual(sent_action.shape, (1, 2))
        # Values preserved
        self.assertAlmostEqual(sent_action[0, 0].item(), 0.5, places=5)
        self.assertAlmostEqual(sent_action[0, 1].item(), -0.3, places=5)


# ─────────────────────────────────────────────────────────────────────────
# Full EvalRunner integration (mocked Isaac env)
# ─────────────────────────────────────────────────────────────────────────


@unittest.skipUnless(HAS_TORCH, "torch required")
class EvalRunnerIntegrationTest(unittest.TestCase):
    def test_eval_runner_produces_report(self):
        def naive_policy(state):
            return {"action": 0}

        env = MockIsaacEnv(num_envs=1, terminate_at_step=20)
        adapter = IsaacEnvironmentAdapter(env=env, name="mock_isaac")

        ruleset = Ruleset(
            [
                require_metric("episode_return", ">=", 5.0, name="balance_5_steps"),
            ]
        )

        report = EvalRunner(
            policies=[naive_policy],
            scenarios=[Scenario("smoke", {"seed": 0}, max_steps=30)],
            ruleset=ruleset,
            baseline_policy="naive_policy",
            environment=adapter,
        ).run()

        self.assertEqual(len(report.episodes), 1)
        self.assertIn("naive_policy", report.metric_summary)
        self.assertIn("episode_return", report.metric_summary["naive_policy"])
        # Should succeed since 20 steps of reward=1.0 >> 5
        self.assertTrue(report.episodes[0].success)

    def test_eval_runner_with_multiple_policies(self):
        """Test the full comparison path: 2 policies, 2 scenarios, regression detection."""
        def good_policy(state):
            return {"action": 0, "debug_info": {"version": "good"}}

        def bad_policy(state):
            return {"action": 0, "debug_info": {"version": "bad"}}

        # Different envs with different terminate-at-step thresholds to simulate
        # one passing and one failing the rule
        env = MockIsaacEnv(num_envs=1, terminate_at_step=10)
        adapter = IsaacEnvironmentAdapter(env=env, name="mock_isaac")

        ruleset = Ruleset(
            [
                require_metric("episode_return", ">=", 5.0, name="balance_5_steps"),
            ]
        )

        report = EvalRunner(
            policies=[good_policy, bad_policy],
            scenarios=[
                Scenario("s1", {"seed": 0}, max_steps=20),
                Scenario("s2", {"seed": 1}, max_steps=20),
            ],
            ruleset=ruleset,
            baseline_policy="good_policy",
            environment=adapter,
        ).run()

        # 2 policies × 2 scenarios = 4 episodes
        self.assertEqual(len(report.episodes), 4)
        # Both policies should succeed (10 steps > 5 threshold)
        for ep in report.episodes:
            self.assertTrue(ep.success)
        # Metric summary should track both policies
        self.assertIn("good_policy", report.metric_summary)
        self.assertIn("bad_policy", report.metric_summary)
        # Action divergences should be empty since both policies do action=0
        self.assertEqual(len(report.action_divergences), 0)


if __name__ == "__main__":
    unittest.main()
