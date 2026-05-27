"""Unit tests for the Gymnasium integration spike.

These tests live inside the spike folder, not in the SDK's root ``tests/``.
Run with::

    python -m unittest discover -s roboeval/integrations/gymnasium/tests
"""

from __future__ import annotations

import json
import unittest
from typing import Any

import gymnasium as gym
import numpy as np

from roboeval import EvalRunner, Ruleset, Scenario, require_metric, require_outcome
from roboeval.core import to_serializable
from roboeval.environment import EnvironmentAdapter, StepOutcome
from roboeval.integrations.gymnasium import (
    GymnasiumEnvironmentAdapter,
    default_action_from_decision,
    default_events_from_step,
    default_observation_to_state,
    default_options_from_scenario,
    default_outcome_from_step,
    default_seed_from_scenario,
)


# --- Default hook tests (no env, no adapter, just the pure functions) -------


class DefaultObservationToStateTest(unittest.TestCase):
    def test_dict_observation_is_passed_through(self) -> None:
        obs = {"image": np.zeros((2, 2)), "joints": np.array([1.0, 2.0])}
        state = default_observation_to_state(obs)
        self.assertEqual(set(state.keys()), {"image", "joints"})
        self.assertIs(state["image"], obs["image"])

    def test_non_dict_observation_is_wrapped(self) -> None:
        obs = np.array([0.1, -0.5, 0.0, 0.3])
        state = default_observation_to_state(obs)
        self.assertEqual(list(state.keys()), ["observation"])
        self.assertIs(state["observation"], obs)

    def test_scalar_observation_is_wrapped(self) -> None:
        state = default_observation_to_state(7)
        self.assertEqual(state, {"observation": 7})


class DefaultActionFromDecisionTest(unittest.TestCase):
    def test_int_action_passes_through(self) -> None:
        self.assertEqual(default_action_from_decision(1), 1)

    def test_ndarray_action_passes_through(self) -> None:
        action = np.array([0.1, -0.3], dtype=np.float32)
        result = default_action_from_decision(action)
        self.assertIs(result, action)

    def test_string_action_passes_through(self) -> None:
        self.assertEqual(default_action_from_decision("move_forward"), "move_forward")


class DefaultOutcomeFromStepTest(unittest.TestCase):
    def test_terminated_with_positive_reward_is_success(self) -> None:
        outcome, label = default_outcome_from_step(1.0, terminated=True, truncated=False, info={})
        self.assertEqual(outcome, "terminated_success")
        self.assertEqual(label, "")

    def test_terminated_with_zero_reward_is_failure(self) -> None:
        outcome, label = default_outcome_from_step(0.0, terminated=True, truncated=False, info={})
        self.assertEqual(outcome, "terminated_failure")
        self.assertEqual(label, "terminated_failure")

    def test_truncated_yields_timeout(self) -> None:
        outcome, label = default_outcome_from_step(0.5, terminated=False, truncated=True, info={})
        self.assertEqual(outcome, "truncated")
        self.assertEqual(label, "timeout")

    def test_non_terminal_yields_progress(self) -> None:
        outcome, label = default_outcome_from_step(0.1, terminated=False, truncated=False, info={})
        self.assertEqual(outcome, "progress")
        self.assertEqual(label, "")


class DefaultEventsFromStepTest(unittest.TestCase):
    def test_no_events_on_plain_progress(self) -> None:
        events = default_events_from_step(1.0, terminated=False, truncated=False, info={})
        self.assertEqual(events, [])

    def test_terminated_event(self) -> None:
        events = default_events_from_step(1.0, terminated=True, truncated=False, info={})
        self.assertEqual(events, ["episode_terminated"])

    def test_truncated_event(self) -> None:
        events = default_events_from_step(1.0, terminated=False, truncated=True, info={})
        self.assertEqual(events, ["episode_truncated"])

    def test_reward_negative_event(self) -> None:
        events = default_events_from_step(-1.0, terminated=False, truncated=False, info={})
        self.assertEqual(events, ["reward_negative"])

    def test_terminated_with_negative_reward_emits_two_events(self) -> None:
        events = default_events_from_step(-1.0, terminated=True, truncated=False, info={})
        self.assertEqual(events, ["episode_terminated", "reward_negative"])


class DefaultSeedFromScenarioTest(unittest.TestCase):
    def test_seed_from_initial_state(self) -> None:
        scenario = Scenario("s", {"seed": 42}, max_steps=10)
        self.assertEqual(default_seed_from_scenario(scenario), 42)

    def test_seed_from_metadata_when_initial_state_lacks_it(self) -> None:
        scenario = Scenario("s", {"foo": 1}, max_steps=10, metadata={"seed": 7})
        self.assertEqual(default_seed_from_scenario(scenario), 7)

    def test_no_seed_returns_none(self) -> None:
        scenario = Scenario("s", {"foo": 1}, max_steps=10)
        self.assertIsNone(default_seed_from_scenario(scenario))


class DefaultOptionsFromScenarioTest(unittest.TestCase):
    def test_options_from_metadata(self) -> None:
        scenario = Scenario("s", {"foo": 1}, max_steps=10, metadata={"reset_options": {"a": 1}})
        self.assertEqual(default_options_from_scenario(scenario), {"a": 1})

    def test_no_options_returns_none(self) -> None:
        scenario = Scenario("s", {"foo": 1}, max_steps=10)
        self.assertIsNone(default_options_from_scenario(scenario))

    def test_non_dict_options_returns_none(self) -> None:
        scenario = Scenario("s", {"foo": 1}, max_steps=10, metadata={"reset_options": "not_a_dict"})
        self.assertIsNone(default_options_from_scenario(scenario))


# --- Adapter construction tests (no env step yet) ---------------------------


class GymnasiumEnvironmentAdapterConstructionTest(unittest.TestCase):
    def test_satisfies_environment_adapter_protocol(self) -> None:
        """Duck-check: adapter has the methods the EnvironmentAdapter Protocol requires."""
        adapter = GymnasiumEnvironmentAdapter(env=gym.make("CartPole-v1"))
        self.assertTrue(hasattr(adapter, "reset"))
        self.assertTrue(hasattr(adapter, "step"))
        self.assertTrue(callable(adapter.reset))
        self.assertTrue(callable(adapter.step))
        # Treated as EnvironmentAdapter by the runner (duck typing)
        env: EnvironmentAdapter = adapter  # noqa: F841 - type-check is the point

    def test_refuses_vector_env(self) -> None:
        vec_env = gym.make_vec("CartPole-v1", num_envs=2)
        with self.assertRaises(NotImplementedError) as ctx:
            GymnasiumEnvironmentAdapter(env=vec_env)
        self.assertIn("VectorEnv", str(ctx.exception))
        vec_env.close()

    def test_defaults_wired_when_hooks_are_none(self) -> None:
        adapter = GymnasiumEnvironmentAdapter(env=gym.make("CartPole-v1"))
        self.assertIs(adapter.observation_to_state, default_observation_to_state)
        self.assertIs(adapter.action_from_decision, default_action_from_decision)
        self.assertIs(adapter.outcome_from_step, default_outcome_from_step)
        self.assertIs(adapter.events_from_step, default_events_from_step)
        self.assertIs(adapter.seed_from_scenario, default_seed_from_scenario)
        self.assertIs(adapter.options_from_scenario, default_options_from_scenario)

    def test_name_propagates_for_report_metadata(self) -> None:
        adapter = GymnasiumEnvironmentAdapter(env=gym.make("CartPole-v1"), name="my_env")
        self.assertEqual(adapter.name, "my_env")


# --- Adapter behavior tests (real env, full reset/step cycle) ---------------


class GymnasiumEnvironmentAdapterBehaviorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = gym.make("CartPole-v1")
        self.adapter = GymnasiumEnvironmentAdapter(env=self.env, name="cartpole_v1")
        self.scenario = Scenario("test", {"seed": 0}, max_steps=200)

    def tearDown(self) -> None:
        self.adapter.close()

    def test_reset_returns_dict_with_observation_key(self) -> None:
        state = self.adapter.reset(self.scenario)
        self.assertIsInstance(state, dict)
        self.assertIn("observation", state)
        self.assertEqual(state["observation"].shape, (4,))  # CartPole obs is 4-dim

    def test_step_returns_step_outcome_with_required_fields(self) -> None:
        self.adapter.reset(self.scenario)
        outcome = self.adapter.step(0, self.scenario)
        self.assertIsInstance(outcome, StepOutcome)
        # Four mandatory fields
        self.assertIsInstance(outcome.next_state, dict)
        self.assertIsInstance(outcome.outcome, str)
        self.assertIsInstance(outcome.failure_label, str)
        self.assertIsInstance(outcome.terminal, bool)

    def test_step_populates_metrics_per_spec(self) -> None:
        self.adapter.reset(self.scenario)
        outcome = self.adapter.step(0, self.scenario)
        self.assertIsInstance(outcome.metrics, dict)
        self.assertIn("reward", outcome.metrics)
        self.assertIn("episode_return", outcome.metrics)
        self.assertEqual(outcome.metrics["reward"], outcome.metrics["episode_return"])

    def test_episode_return_accumulates_across_steps(self) -> None:
        self.adapter.reset(self.scenario)
        # Take 3 steps; in CartPole each non-terminal step returns reward=1.0
        out1 = self.adapter.step(0, self.scenario)
        out2 = self.adapter.step(1, self.scenario)
        out3 = self.adapter.step(0, self.scenario)
        self.assertEqual(out1.metrics["episode_return"], 1.0)
        self.assertEqual(out2.metrics["episode_return"], 2.0)
        self.assertEqual(out3.metrics["episode_return"], 3.0)

    def test_episode_return_resets_on_reset(self) -> None:
        self.adapter.reset(self.scenario)
        self.adapter.step(0, self.scenario)
        self.adapter.step(1, self.scenario)
        self.adapter.reset(self.scenario)
        outcome = self.adapter.step(0, self.scenario)
        self.assertEqual(outcome.metrics["episode_return"], 1.0)

    def test_step_populates_gymnasium_info_namespace(self) -> None:
        self.adapter.reset(self.scenario)
        outcome = self.adapter.step(0, self.scenario)
        self.assertIsInstance(outcome.info, dict)
        self.assertIn("gymnasium", outcome.info)
        gym_info = outcome.info["gymnasium"]
        self.assertIn("terminated", gym_info)
        self.assertIn("truncated", gym_info)
        self.assertIn("raw_info", gym_info)
        self.assertIsInstance(gym_info["terminated"], bool)
        self.assertIsInstance(gym_info["truncated"], bool)

    def test_seed_makes_episodes_reproducible(self) -> None:
        """Same seed + same actions should produce same observations across resets."""
        state_a = self.adapter.reset(self.scenario)
        out_a = self.adapter.step(0, self.scenario)
        state_b = self.adapter.reset(self.scenario)
        out_b = self.adapter.step(0, self.scenario)
        np.testing.assert_allclose(state_a["observation"], state_b["observation"])
        np.testing.assert_allclose(out_a.next_state["observation"], out_b.next_state["observation"])

    def test_full_episode_eventually_terminates(self) -> None:
        """Running long enough with a naive policy should hit terminal=True."""
        self.adapter.reset(self.scenario)
        terminal_seen = False
        for _ in range(200):
            outcome = self.adapter.step(0, self.scenario)  # always push left
            if outcome.terminal:
                terminal_seen = True
                self.assertIn("episode_terminated", outcome.events)
                break
        self.assertTrue(terminal_seen, "Expected naive policy to hit terminal within 200 steps")


# --- Hook override tests ----------------------------------------------------


class GymnasiumEnvironmentAdapterHookOverrideTest(unittest.TestCase):
    def test_outcome_hook_override(self) -> None:
        def always_goal(reward, terminated, truncated, info):
            return ("goal_reached", "")

        env = gym.make("CartPole-v1")
        adapter = GymnasiumEnvironmentAdapter(env=env, outcome_from_step=always_goal)
        adapter.reset(Scenario("s", {"seed": 0}, max_steps=10))
        outcome = adapter.step(0, Scenario("s", {"seed": 0}, max_steps=10))
        self.assertEqual(outcome.outcome, "goal_reached")
        adapter.close()

    def test_action_hook_can_translate_string_vocabulary(self) -> None:
        env = gym.make("CartPole-v1")
        adapter = GymnasiumEnvironmentAdapter(
            env=env,
            action_from_decision=lambda a: {"left": 0, "right": 1}[a],
        )
        adapter.reset(Scenario("s", {"seed": 0}, max_steps=10))
        # Passes "left" through hook; env step expects int 0
        outcome = adapter.step("left", Scenario("s", {"seed": 0}, max_steps=10))
        self.assertFalse(outcome.terminal)
        adapter.close()

    def test_info_keys_allowlist_filters_raw_info(self) -> None:
        # CartPole's info dict is usually empty; inject a custom outcome hook
        # that puts a fake value into info before allowlist filtering.
        env = gym.make("CartPole-v1")
        adapter = GymnasiumEnvironmentAdapter(env=env, info_keys=["only_this_key"])
        adapter.reset(Scenario("s", {"seed": 0}, max_steps=10))
        outcome = adapter.step(0, Scenario("s", {"seed": 0}, max_steps=10))
        # CartPole's info is {} so allowlist returns {}; no extra keys leak
        self.assertEqual(outcome.info["gymnasium"]["raw_info"], {})
        adapter.close()

    def test_coerce_observations_flag_converts_ndarray_to_list(self) -> None:
        env = gym.make("CartPole-v1")
        adapter = GymnasiumEnvironmentAdapter(env=env, coerce_observations=True)
        state = adapter.reset(Scenario("s", {"seed": 0}, max_steps=10))
        # With coercion, observation should be a Python list, not ndarray
        self.assertIsInstance(state["observation"], list)
        adapter.close()


# --- Serialization / runner integration tests -------------------------------


class GymnasiumEnvironmentAdapterSerializationTest(unittest.TestCase):
    def test_step_outcome_is_json_safe_via_to_serializable(self) -> None:
        env = gym.make("CartPole-v1")
        adapter = GymnasiumEnvironmentAdapter(env=env)
        adapter.reset(Scenario("s", {"seed": 0}, max_steps=10))
        outcome = adapter.step(0, Scenario("s", {"seed": 0}, max_steps=10))
        # to_serializable should produce a JSON-encodable structure
        for field in (outcome.next_state, outcome.metrics, outcome.events, outcome.info):
            payload = to_serializable(field)
            json.dumps(payload)
        adapter.close()


class GymnasiumEnvironmentAdapterRunnerIntegrationTest(unittest.TestCase):
    """End-to-end through EvalRunner + Ruleset (the canonical integration)."""

    def test_eval_runner_produces_report_with_metrics_and_rules(self) -> None:
        def naive_policy(state: dict) -> dict[str, Any]:
            angle = float(state["observation"][2])
            return {"action": 0 if angle < 0 else 1}

        adapter = GymnasiumEnvironmentAdapter(env=gym.make("CartPole-v1"), name="cartpole_v1")
        ruleset = Ruleset([
            require_outcome("terminated_success"),
            require_metric("episode_return", ">=", 5.0),  # easy threshold for the test
        ])

        report = EvalRunner(
            policies=[naive_policy],
            scenarios=[Scenario("eval_smoke", {"seed": 0}, max_steps=200)],
            ruleset=ruleset,
            baseline_policy="naive_policy",
            environment=adapter,
        ).run()

        self.assertEqual(len(report.episodes), 1)
        # Metric summary picked up our `metrics` dict automatically
        self.assertIn("naive_policy", report.metric_summary)
        self.assertIn("episode_return", report.metric_summary["naive_policy"])
        self.assertIn("reward", report.metric_summary["naive_policy"])
        # At least one rule result was recorded
        self.assertGreaterEqual(len(report.episodes[0].rule_results), 2)


if __name__ == "__main__":
    unittest.main()
