"""Unit + integration tests for BatchedIsaacEnvironmentAdapter.

Runs on Mac without Isaac installed. Uses a MockBatchedIsaacEnv that fakes
the Isaac Lab vectorized shape — real torch tensors, batched obs/reward/
terminal flags, and configurable per-slot termination steps.
"""

from __future__ import annotations

import unittest
import warnings
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from roboeval.batched.runner import BatchedEvalRunner
from roboeval.batched.types import BatchedStepOutcome
from roboeval.core import Ruleset, Scenario, require_metric, require_outcome
from roboeval.integrations.isaac.batched_adapter import (
    BatchedIsaacEnvironmentAdapter,
    _per_slot_isaac_info,
    _scalar_at,
)


HAS_TORCH = torch is not None


# ─── Mock Isaac vectorized env ──────────────────────────────────────────────


class MockBatchedIsaacEnv:
    """Fakes a vectorized Isaac Lab env with real torch tensors.

    Each env in the batch has its own step counter and terminates when it
    hits its assigned ``terminate_at_step``. Supports ``_reset_idx`` for
    selective reset (the recommended Isaac path) and ``_get_observations``
    so the adapter can read post-reset obs without stepping.
    """

    def __init__(
        self,
        num_envs: int = 4,
        obs_dim: int = 4,
        action_dim: int = 2,
        terminate_at_steps: list[int] | int | None = None,
        obs_as_dict: bool = True,
        device: str = "cpu",
        rich_info: bool = False,
    ) -> None:
        if not HAS_TORCH:
            raise RuntimeError("MockBatchedIsaacEnv requires torch")
        self.num_envs = num_envs
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        if terminate_at_steps is None:
            terminate_at_steps = [10] * num_envs
        elif isinstance(terminate_at_steps, int):
            terminate_at_steps = [terminate_at_steps] * num_envs
        self._terminate_at = list(terminate_at_steps)
        self._obs_as_dict = obs_as_dict
        self.device = device
        self.rich_info = rich_info
        self._step_counts = [0] * num_envs
        self._last_action: Any = None
        self.reset_calls = 0
        self.reset_idx_calls = 0
        self.step_calls = 0
        self._last_obs: Any = None

    def _make_obs(self) -> Any:
        # Per-slot obs is the step count, broadcast across obs_dim
        flat = torch.tensor(
            [[float(c)] * self._obs_dim for c in self._step_counts],
            dtype=torch.float32, device=self.device,
        )
        if self._obs_as_dict:
            return {"policy": flat}
        return flat

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.reset_calls += 1
        self._step_counts = [0] * self.num_envs
        self._last_obs = self._make_obs()
        return self._last_obs, {}

    def step(self, action):
        if hasattr(action, "shape") and action.shape[0] != self.num_envs:
            raise ValueError(
                f"action batch dim {action.shape[0]} != num_envs {self.num_envs}"
            )
        self.step_calls += 1
        self._last_action = action
        for i in range(self.num_envs):
            self._step_counts[i] += 1
        obs = self._make_obs()
        self._last_obs = obs
        reward = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        terminated = torch.tensor(
            [self._step_counts[i] >= self._terminate_at[i] for i in range(self.num_envs)],
            dtype=torch.bool, device=self.device,
        )
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        info: dict[str, Any] = {}
        if self.rich_info:
            info["task_reward"] = torch.tensor(
                [float(c) * 2 for c in self._step_counts],
                dtype=torch.float32, device=self.device,
            )
            info["step_count"] = torch.tensor(
                self._step_counts, dtype=torch.long, device=self.device
            )
            info["sim_version"] = "mock_v1"  # scalar info
        return obs, reward, terminated, truncated, info

    # ── Isaac Lab Manager API surface (for selective reset) ─────────────
    @property
    def unwrapped(self):
        return self

    def _reset_idx(self, env_ids) -> None:
        self.reset_idx_calls += 1
        if hasattr(env_ids, "tolist"):
            env_ids = env_ids.tolist()
        for i in env_ids:
            self._step_counts[i] = 0
        # Refresh obs buffer so _get_observations returns post-reset state
        self._last_obs = self._make_obs()

    def _get_observations(self):
        return self._last_obs

    def close(self):
        pass


class MockBatchedIsaacEnvNoSelectiveReset:
    """Mock that doesn't expose _reset_idx — exercises the full-reset fallback."""

    def __init__(self, num_envs: int = 2):
        if not HAS_TORCH:
            raise RuntimeError("torch required")
        self.num_envs = num_envs
        self.device = "cpu"
        self._step_counts = [0] * num_envs
        self.reset_calls = 0

    @property
    def unwrapped(self):
        # No _reset_idx, no _get_observations
        return object()

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.reset_calls += 1
        self._step_counts = [0] * self.num_envs
        return {"policy": torch.zeros(self.num_envs, 4)}, {}

    def step(self, action):
        for i in range(self.num_envs):
            self._step_counts[i] += 1
        obs = {"policy": torch.tensor([[float(c)] * 4 for c in self._step_counts])}
        reward = torch.ones(self.num_envs)
        terminated = torch.tensor([c >= 3 for c in self._step_counts])
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        return obs, reward, terminated, truncated, {}


# ─── Construction + validation ──────────────────────────────────────────────


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestBatchedIsaacConstruction(unittest.TestCase):
    def test_rejects_env_without_num_envs(self) -> None:
        class NoNumEnvs:
            def reset(self): return None, {}
            def step(self, a): return None, None, None, None, {}

        with self.assertRaisesRegex(TypeError, r"num_envs"):
            BatchedIsaacEnvironmentAdapter(env=NoNumEnvs())

    def test_rejects_zero_num_envs(self) -> None:
        class ZeroNumEnvs:
            num_envs = 0
            def reset(self, **k): return None, {}
            def step(self, a): return None, None, None, None, {}

        with self.assertRaisesRegex(ValueError, r"num_envs must be positive"):
            BatchedIsaacEnvironmentAdapter(env=ZeroNumEnvs())

    def test_num_envs_set_from_env(self) -> None:
        env = MockBatchedIsaacEnv(num_envs=8)
        adapter = BatchedIsaacEnvironmentAdapter(env=env)
        self.assertEqual(adapter.num_envs, 8)

    def test_num_envs_falls_back_to_unwrapped(self) -> None:
        """gym.make() wraps Isaac envs and the outer wrapper doesn't proxy
        num_envs. Adapter must read it from env.unwrapped — regression check
        for the bug discovered while validating against real Isaac-Cartpole."""
        inner = MockBatchedIsaacEnv(num_envs=8)

        class GymStyleWrapper:
            """Simulates the gym.OrderEnforcing/TimeLimit wrapper layer that
            sits between gym.make()'s return value and the underlying env."""
            def __init__(self, inner_env):
                self.unwrapped = inner_env

            def reset(self, **kwargs):
                return self.unwrapped.reset(**kwargs)

            def step(self, action):
                return self.unwrapped.step(action)

        wrapped = GymStyleWrapper(inner)
        # The wrapped env has NO .num_envs attribute, only .unwrapped.num_envs
        self.assertFalse(hasattr(wrapped, "num_envs"))
        adapter = BatchedIsaacEnvironmentAdapter(env=wrapped)
        self.assertEqual(adapter.num_envs, 8)


# ─── reset() ─────────────────────────────────────────────────────────────────


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestBatchedIsaacReset(unittest.TestCase):
    def test_reset_returns_num_envs_states(self) -> None:
        env = MockBatchedIsaacEnv(num_envs=4)
        adapter = BatchedIsaacEnvironmentAdapter(env=env)
        scenarios = [Scenario(f"s{i}", {"seed": i}, max_steps=20) for i in range(4)]
        states = adapter.reset(scenarios)
        self.assertEqual(len(states), 4)
        for s in states:
            self.assertIn("policy", s)
            # Each state's policy obs is numpy (coerced from torch)
            self.assertIsInstance(s["policy"], np.ndarray)
            self.assertEqual(s["policy"].shape, (4,))

    def test_reset_requires_num_envs_scenarios(self) -> None:
        env = MockBatchedIsaacEnv(num_envs=3)
        adapter = BatchedIsaacEnvironmentAdapter(env=env)
        with self.assertRaisesRegex(ValueError, r"3 scenarios"):
            adapter.reset([Scenario("only", {}, max_steps=10)])

    def test_reset_zeroes_episode_returns(self) -> None:
        env = MockBatchedIsaacEnv(num_envs=2, terminate_at_steps=100)
        adapter = BatchedIsaacEnvironmentAdapter(env=env)
        adapter.reset([Scenario("a", {}, max_steps=50), Scenario("b", {}, max_steps=50)])
        # Step a few times to accumulate
        for _ in range(3):
            adapter.step([0, 0])
        self.assertGreater(sum(adapter._episode_returns), 0.0)
        adapter.reset([Scenario("a", {}, max_steps=50), Scenario("b", {}, max_steps=50)])
        self.assertEqual(adapter._episode_returns, [0.0, 0.0])

    def test_reset_tolerates_env_without_options_kwarg(self) -> None:
        class NoOptionsEnv(MockBatchedIsaacEnv):
            def reset(self, seed=None):
                return super().reset(seed=seed)

        env = NoOptionsEnv(num_envs=2)
        adapter = BatchedIsaacEnvironmentAdapter(env=env)
        states = adapter.reset([Scenario("a", {}, max_steps=10), Scenario("b", {}, max_steps=10)])
        self.assertEqual(len(states), 2)


# ─── step() ─────────────────────────────────────────────────────────────────


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestBatchedIsaacStep(unittest.TestCase):
    def test_step_returns_batched_step_outcome(self) -> None:
        env = MockBatchedIsaacEnv(num_envs=3, terminate_at_steps=[2, 4, 6])
        adapter = BatchedIsaacEnvironmentAdapter(env=env)
        adapter.reset([Scenario(f"s{i}", {}, max_steps=10) for i in range(3)])
        outcome = adapter.step([0, 1, 0])
        self.assertIsInstance(outcome, BatchedStepOutcome)
        self.assertEqual(outcome.num_envs, 3)

    def test_step_requires_num_envs_actions(self) -> None:
        env = MockBatchedIsaacEnv(num_envs=2)
        adapter = BatchedIsaacEnvironmentAdapter(env=env)
        adapter.reset([Scenario("a", {}, max_steps=5), Scenario("b", {}, max_steps=5)])
        with self.assertRaisesRegex(ValueError, r"2 actions"):
            adapter.step([0])

    def test_per_slot_terminals_independent(self) -> None:
        """Slot 0 terminates at step 2, slot 1 at step 4. Both must surface correctly."""
        env = MockBatchedIsaacEnv(num_envs=2, terminate_at_steps=[2, 4])
        adapter = BatchedIsaacEnvironmentAdapter(env=env)
        adapter.reset([Scenario("a", {}, max_steps=10), Scenario("b", {}, max_steps=10)])
        # Step 1: nobody terminates
        out = adapter.step([0, 0])
        self.assertEqual(out.terminals, [False, False])
        # Step 2: slot 0 terminates
        out = adapter.step([0, 0])
        self.assertEqual(out.terminals, [True, False])
        # Step 3: slot 0 just got auto-stepped (mock keeps counting); only slot 1 matters
        out = adapter.step([0, 0])
        self.assertEqual(out.terminals[1], False)
        # Step 4: slot 1 terminates
        out = adapter.step([0, 0])
        self.assertTrue(out.terminals[1])

    def test_per_slot_episode_return_accumulates(self) -> None:
        env = MockBatchedIsaacEnv(num_envs=2, terminate_at_steps=100)
        adapter = BatchedIsaacEnvironmentAdapter(env=env)
        adapter.reset([Scenario("a", {}, max_steps=10), Scenario("b", {}, max_steps=10)])
        for _ in range(3):
            out = adapter.step([0, 0])
        # Each slot got reward=1 per step for 3 steps
        self.assertEqual(out.metrics[0]["episode_return"], 3.0)
        self.assertEqual(out.metrics[1]["episode_return"], 3.0)

    def test_terminal_zeroes_episode_return_for_that_slot(self) -> None:
        env = MockBatchedIsaacEnv(num_envs=2, terminate_at_steps=[2, 10])
        adapter = BatchedIsaacEnvironmentAdapter(env=env)
        adapter.reset([Scenario("a", {}, max_steps=20), Scenario("b", {}, max_steps=20)])
        adapter.step([0, 0])  # step 1
        adapter.step([0, 0])  # step 2 — slot 0 terminates; episode_return zeroed afterward
        self.assertEqual(adapter._episode_returns[0], 0.0)
        # Slot 1 keeps accumulating
        self.assertGreater(adapter._episode_returns[1], 0.0)

    def test_info_namespace_isaac(self) -> None:
        env = MockBatchedIsaacEnv(num_envs=2, terminate_at_steps=10, rich_info=True)
        adapter = BatchedIsaacEnvironmentAdapter(env=env)
        adapter.reset([Scenario("a", {}, max_steps=10), Scenario("b", {}, max_steps=10)])
        out = adapter.step([0, 0])
        for slot in (0, 1):
            self.assertIn("isaac", out.info[slot])
            isaac_info = out.info[slot]["isaac"]
            self.assertIn("terminated", isaac_info)
            self.assertIn("truncated", isaac_info)
            self.assertIn("raw_info", isaac_info)
            self.assertEqual(isaac_info["slot"], slot)
            # rich_info batched values land per-slot
            self.assertIn("task_reward", isaac_info["raw_info"])
            self.assertIn("step_count", isaac_info["raw_info"])
            # scalar info is forwarded
            self.assertEqual(isaac_info["raw_info"]["sim_version"], "mock_v1")

    def test_metrics_carry_reward_and_episode_return(self) -> None:
        env = MockBatchedIsaacEnv(num_envs=2)
        adapter = BatchedIsaacEnvironmentAdapter(env=env)
        adapter.reset([Scenario("a", {}, max_steps=10), Scenario("b", {}, max_steps=10)])
        out = adapter.step([0, 0])
        for slot in (0, 1):
            self.assertIn("reward", out.metrics[slot])
            self.assertIn("episode_return", out.metrics[slot])

    def test_obs_with_bare_tensor_handled(self) -> None:
        env = MockBatchedIsaacEnv(num_envs=2, obs_as_dict=False)
        adapter = BatchedIsaacEnvironmentAdapter(env=env)
        states = adapter.reset([Scenario("a", {}, max_steps=10), Scenario("b", {}, max_steps=10)])
        # Bare tensor → state has "observation" key
        for s in states:
            self.assertIn("observation", s)


# ─── Action stacking ────────────────────────────────────────────────────────


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestActionStacking(unittest.TestCase):
    def test_int_actions_become_batched_tensor(self) -> None:
        env = MockBatchedIsaacEnv(num_envs=3, action_dim=1)
        adapter = BatchedIsaacEnvironmentAdapter(env=env)
        adapter.reset([Scenario(f"s{i}", {}, max_steps=10) for i in range(3)])
        adapter.step([0, 1, 0])
        # Mock stores last action — verify shape
        last = env._last_action
        self.assertEqual(tuple(last.shape), (3, 1))

    def test_list_actions_become_batched_tensor(self) -> None:
        env = MockBatchedIsaacEnv(num_envs=2, action_dim=3)
        adapter = BatchedIsaacEnvironmentAdapter(env=env)
        adapter.reset([Scenario(f"s{i}", {}, max_steps=10) for i in range(2)])
        adapter.step([[0.1, 0.2, 0.3], [-0.1, 0.0, 0.5]])
        last = env._last_action
        self.assertEqual(tuple(last.shape), (2, 3))

    def test_tensor_actions_stack(self) -> None:
        env = MockBatchedIsaacEnv(num_envs=2, action_dim=2)
        adapter = BatchedIsaacEnvironmentAdapter(env=env)
        adapter.reset([Scenario(f"s{i}", {}, max_steps=10) for i in range(2)])
        adapter.step([torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4])])
        last = env._last_action
        self.assertEqual(tuple(last.shape), (2, 2))


# ─── reset_slots() ──────────────────────────────────────────────────────────


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestBatchedIsaacResetSlots(unittest.TestCase):
    def test_selective_reset_via_reset_idx(self) -> None:
        """When env exposes ``_reset_idx``, we hit that path (not full reset)."""
        env = MockBatchedIsaacEnv(num_envs=4, terminate_at_steps=[2, 100, 100, 100])
        adapter = BatchedIsaacEnvironmentAdapter(env=env)
        adapter.reset([Scenario(f"s{i}", {}, max_steps=20) for i in range(4)])
        # Step until slot 0 has accumulated some step count
        for _ in range(3):
            adapter.step([0, 0, 0, 0])
        before_full_resets = env.reset_calls
        # Refill slot 0 with a new scenario
        new_states = adapter.reset_slots([0], [Scenario("refilled", {"seed": 99}, max_steps=20)])
        self.assertEqual(len(new_states), 1)
        # The selective path was used
        self.assertGreaterEqual(env.reset_idx_calls, 1)
        # No additional full reset was needed
        self.assertEqual(env.reset_calls, before_full_resets)
        # Other slots' step counts are NOT zeroed (proves selectivity)
        self.assertEqual(env._step_counts[0], 0)
        self.assertGreater(env._step_counts[1], 0)
        self.assertGreater(env._step_counts[2], 0)
        self.assertGreater(env._step_counts[3], 0)

    def test_fallback_full_reset_when_no_selective_path(self) -> None:
        """When env has no ``_reset_idx``, the adapter warns and full-resets."""
        env = MockBatchedIsaacEnvNoSelectiveReset(num_envs=2)
        adapter = BatchedIsaacEnvironmentAdapter(env=env)
        adapter.reset([Scenario("a", {}, max_steps=10), Scenario("b", {}, max_steps=10)])
        before = env.reset_calls
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            adapter.reset_slots([0], [Scenario("refill", {"seed": 1}, max_steps=10)])
        # Full reset was called
        self.assertGreater(env.reset_calls, before)
        # And a warning was emitted
        messages = [str(w.message) for w in caught]
        self.assertTrue(any("selective reset" in m for m in messages))

    def test_reset_slots_zeroes_per_slot_episode_return(self) -> None:
        env = MockBatchedIsaacEnv(num_envs=2, terminate_at_steps=100)
        adapter = BatchedIsaacEnvironmentAdapter(env=env)
        adapter.reset([Scenario("a", {}, max_steps=20), Scenario("b", {}, max_steps=20)])
        for _ in range(3):
            adapter.step([0, 0])
        before = list(adapter._episode_returns)
        self.assertTrue(all(r > 0 for r in before))
        adapter.reset_slots([0], [Scenario("refilled", {}, max_steps=20)])
        self.assertEqual(adapter._episode_returns[0], 0.0)
        self.assertEqual(adapter._episode_returns[1], before[1])

    def test_reset_slots_length_mismatch_raises(self) -> None:
        env = MockBatchedIsaacEnv(num_envs=2)
        adapter = BatchedIsaacEnvironmentAdapter(env=env)
        adapter.reset([Scenario("a", {}, max_steps=10), Scenario("b", {}, max_steps=10)])
        with self.assertRaisesRegex(ValueError, r"same length"):
            adapter.reset_slots([0, 1], [Scenario("only", {}, max_steps=10)])

    def test_reset_slots_empty_input_is_noop(self) -> None:
        env = MockBatchedIsaacEnv(num_envs=2)
        adapter = BatchedIsaacEnvironmentAdapter(env=env)
        adapter.reset([Scenario("a", {}, max_steps=10), Scenario("b", {}, max_steps=10)])
        out = adapter.reset_slots([], [])
        self.assertEqual(out, [])


# ─── Helper functions ───────────────────────────────────────────────────────


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestScalarAt(unittest.TestCase):
    def test_extracts_from_tensor(self) -> None:
        t = torch.tensor([1.0, 2.0, 3.0])
        self.assertEqual(_scalar_at(t, 1, default=0.0), 2.0)

    def test_extracts_from_numpy(self) -> None:
        arr = np.array([True, False, True])
        self.assertEqual(_scalar_at(arr, 0, default=False), True)

    def test_fallback_on_bad_index(self) -> None:
        # Scalar value gets passed through
        self.assertEqual(_scalar_at(1.5, 0, default=0.0), 1.5)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestPerSlotIsaacInfo(unittest.TestCase):
    def test_extracts_per_slot_tensor_values(self) -> None:
        info = {"task_reward": torch.tensor([10.0, 20.0, 30.0])}
        result = _per_slot_isaac_info(info, slot=1, num_envs=3, info_keys=None)
        self.assertEqual(result["task_reward"], 20.0)

    def test_passes_through_scalar_info(self) -> None:
        info = {"sim_version": "v1"}
        result = _per_slot_isaac_info(info, slot=0, num_envs=4, info_keys=None)
        self.assertEqual(result, {"sim_version": "v1"})

    def test_info_keys_allowlist_filters(self) -> None:
        info = {
            "task_reward": torch.tensor([1.0, 2.0]),
            "big_tensor": torch.tensor([[1, 2], [3, 4]]),
        }
        result = _per_slot_isaac_info(info, slot=0, num_envs=2, info_keys=["task_reward"])
        self.assertEqual(set(result.keys()), {"task_reward"})

    def test_non_dict_info_returns_empty(self) -> None:
        result = _per_slot_isaac_info(None, slot=0, num_envs=2, info_keys=None)
        self.assertEqual(result, {})


# ─── End-to-end: BatchedEvalRunner + MockBatchedIsaacEnv ────────────────────


def _balance_policy(state):
    """Use slot's policy obs first value to pick an action."""
    if "policy" in state:
        v = float(state["policy"][0])
    else:
        v = float(state["observation"][0])
    return {"action": [0.5 if v < 5 else -0.5], "debug_info": {"v": v}}


def _zero_policy(state):
    return {"action": [0.0], "debug_info": {"version": "zero"}}


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestEndToEndBatchedIsaacWithRunner(unittest.TestCase):
    def test_three_policies_four_scenarios_eight_envs(self) -> None:
        env = MockBatchedIsaacEnv(num_envs=8, action_dim=1, terminate_at_steps=15)
        adapter = BatchedIsaacEnvironmentAdapter(env=env, name="mock_isaac_x8")
        scenarios = [
            Scenario(f"seed_{s}", {"seed": s}, max_steps=30,
                     metadata={"tags": ["mock_isaac"]})
            for s in (1, 2, 42, 100)
        ]
        from roboeval.batched.policy import from_single
        report = BatchedEvalRunner(
            policies=[
                from_single(_balance_policy),
                from_single(_zero_policy),
                from_single(_balance_policy, name="balance_v2"),
            ],
            scenarios=scenarios,
            environment=adapter,
            ruleset=Ruleset([require_metric("episode_return", ">=", 10.0)]),
            baseline_policy="_balance_policy",
        ).run()
        # 3 policies × 4 scenarios = 12 episodes
        self.assertEqual(len(report.episodes), 12)
        for ep in report.episodes:
            self.assertGreater(ep.steps, 0)
            # Each step record carries the isaac namespace
            for log in ep.logs:
                self.assertIn("isaac", log.info)

    def test_replication_with_isaac_mock(self) -> None:
        env = MockBatchedIsaacEnv(num_envs=4, action_dim=1, terminate_at_steps=10)
        adapter = BatchedIsaacEnvironmentAdapter(env=env)
        from roboeval.batched.policy import from_single
        report = BatchedEvalRunner(
            policies=[from_single(_zero_policy)],
            scenarios=[Scenario("only", {"seed": 0}, max_steps=20)],
            environment=adapter,
            replicas=4,
            ruleset=Ruleset([require_outcome("terminated_success")]),
        ).run()
        self.assertEqual(len(report.episodes), 4)
        names = sorted(ep.scenario_name for ep in report.episodes)
        self.assertEqual(names, ["only#r0", "only#r1", "only#r2", "only#r3"])

    def test_ruleset_flow_through(self) -> None:
        env = MockBatchedIsaacEnv(num_envs=2, action_dim=1, terminate_at_steps=5)
        adapter = BatchedIsaacEnvironmentAdapter(env=env)
        from roboeval.batched.policy import from_single
        # episode_return = 5 (5 steps * reward=1); ruleset requires >= 100
        report = BatchedEvalRunner(
            policies=[from_single(_zero_policy)],
            scenarios=[Scenario("a", {}, max_steps=20), Scenario("b", {}, max_steps=20)],
            environment=adapter,
            ruleset=Ruleset([require_metric("episode_return", ">=", 100.0, name="hard_target")]),
        ).run()
        self.assertEqual(len(report.episodes), 2)
        for ep in report.episodes:
            self.assertFalse(ep.success)
            self.assertEqual(ep.failure_label, "hard_target")

    def test_reset_slots_called_during_refill(self) -> None:
        """num_envs=2 with 4 scenarios → refill kicks in twice, exercising reset_slots."""
        env = MockBatchedIsaacEnv(num_envs=2, action_dim=1, terminate_at_steps=3)
        adapter = BatchedIsaacEnvironmentAdapter(env=env)
        from roboeval.batched.policy import from_single
        report = BatchedEvalRunner(
            policies=[from_single(_zero_policy)],
            scenarios=[Scenario(f"s{i}", {}, max_steps=10) for i in range(4)],
            environment=adapter,
            ruleset=Ruleset([require_metric("episode_return", ">=", 1.0)]),
        ).run()
        self.assertEqual(len(report.episodes), 4)
        # Each refill triggers a _reset_idx call on the underlying mock
        self.assertGreaterEqual(env.reset_idx_calls, 1)


if __name__ == "__main__":
    unittest.main()
