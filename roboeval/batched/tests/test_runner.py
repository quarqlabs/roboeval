"""Tests for BatchedEvalRunner — uses a deterministic in-memory mock adapter.

Coverage:
  * single-policy / single-scenario smoke
  * num_envs == num_tasks (no refill)
  * num_envs < num_tasks (refill required, scheduler exercises queue)
  * num_envs > num_tasks (idle slots — placeholder action handling)
  * multi-policy iteration
  * replication mode (replicas > 1) and scenario_name #r suffix
  * max_steps enforcement (force-terminate without env terminal)
  * Ruleset pass/fail flows through to EpisodeResult
  * EvalReport shape matches single-env EvalRunner expectations
"""

from __future__ import annotations

import unittest
from typing import Any

from roboeval.batched.runner import BatchedEvalRunner
from roboeval.batched.types import BatchedStepOutcome
from roboeval.core import Ruleset, Scenario, require_metric, require_outcome


# ─── Mock vectorized environment ────────────────────────────────────────────


class MockBatchedEnv:
    """Deterministic mock that terminates per-scenario at a configurable step.

    Args:
        num_envs: batch width
        terminate_at: dict mapping scenario.name -> step count after which the
            slot returns ``terminal=True``. Missing entries never terminate
            (force-truncate via scenario.max_steps).
        outcome_on_terminate: which ``outcome`` string to emit on terminal step
    """

    def __init__(
        self,
        num_envs: int,
        terminate_at: dict[str, int] | None = None,
        outcome_on_terminate: str = "terminated_success",
    ) -> None:
        self.num_envs = num_envs
        self._terminate_at = terminate_at or {}
        self._outcome_on_terminate = outcome_on_terminate
        self._slot_step_counts: list[int] = [0] * num_envs
        self._slot_scenarios: list[Scenario | None] = [None] * num_envs
        self.step_calls = 0
        self.reset_calls = 0
        self.reset_slots_calls = 0

    def reset(self, scenarios: list[Scenario]) -> list[dict[str, Any]]:
        assert len(scenarios) == self.num_envs
        self.reset_calls += 1
        self._slot_step_counts = [0] * self.num_envs
        self._slot_scenarios = list(scenarios)
        return [{"obs": [0.0], "slot_scenario": s.name} for s in scenarios]

    def step(self, actions: list[Any]) -> BatchedStepOutcome:
        assert len(actions) == self.num_envs
        self.step_calls += 1
        next_states = []
        outcomes = []
        failure_labels = []
        terminals = []
        metrics = []
        for i in range(self.num_envs):
            self._slot_step_counts[i] += 1
            sc = self._slot_scenarios[i]
            term_at = self._terminate_at.get(sc.name) if sc is not None else None
            is_term = term_at is not None and self._slot_step_counts[i] >= term_at
            next_states.append(
                {"obs": [float(self._slot_step_counts[i])], "slot_scenario": sc.name}
            )
            outcomes.append(self._outcome_on_terminate if is_term else "progress")
            failure_labels.append("")
            terminals.append(is_term)
            metrics.append({"reward": 1.0, "episode_return": float(self._slot_step_counts[i])})
        return BatchedStepOutcome(
            next_states=next_states,
            outcomes=outcomes,
            failure_labels=failure_labels,
            terminals=terminals,
            metrics=metrics,
        )

    def reset_slots(
        self, slots: list[int], scenarios: list[Scenario]
    ) -> list[dict[str, Any]]:
        assert len(slots) == len(scenarios)
        self.reset_slots_calls += 1
        new_states = []
        for slot, sc in zip(slots, scenarios):
            self._slot_step_counts[slot] = 0
            self._slot_scenarios[slot] = sc
            new_states.append({"obs": [0.0], "slot_scenario": sc.name})
        return new_states


# ─── Helpers ─────────────────────────────────────────────────────────────────


def constant_policy(action: int):
    """Return a single-state policy that always emits the same action."""

    def _policy(state):
        return {"action": action, "debug_info": {}}

    _policy.__name__ = f"const_{action}"
    return _policy


def batched_constant_policy(action: int, name: str | None = None):
    """A natively-batched constant policy."""

    def _batched(states):
        return [{"action": action} for _ in states]

    _batched.__name__ = name or f"batched_const_{action}"
    return _batched


def scenarios_named(*names: str, max_steps: int = 50) -> list[Scenario]:
    return [Scenario(name=n, initial_state={"seed": i}, max_steps=max_steps) for i, n in enumerate(names)]


def from_single(fn):
    from roboeval.batched.policy import from_single as _fs

    return _fs(fn)


# ─── Tests ───────────────────────────────────────────────────────────────────


class TestBatchedRunnerSmoke(unittest.TestCase):
    def test_single_policy_single_scenario_single_slot(self) -> None:
        env = MockBatchedEnv(num_envs=1, terminate_at={"only": 3})
        report = BatchedEvalRunner(
            policies=[from_single(constant_policy(0))],
            scenarios=scenarios_named("only", max_steps=10),
            environment=env,
        ).run()
        self.assertEqual(len(report.episodes), 1)
        ep = report.episodes[0]
        self.assertEqual(ep.scenario_name, "only")
        self.assertEqual(ep.steps, 3)
        self.assertEqual(ep.terminal_outcome, "terminated_success")

    def test_num_envs_equals_num_scenarios_no_refill(self) -> None:
        env = MockBatchedEnv(
            num_envs=3, terminate_at={"a": 2, "b": 3, "c": 4}
        )
        report = BatchedEvalRunner(
            policies=[from_single(constant_policy(0))],
            scenarios=scenarios_named("a", "b", "c", max_steps=10),
            environment=env,
        ).run()
        self.assertEqual(len(report.episodes), 3)
        # No slot refill — only the bulk reset happens
        self.assertEqual(env.reset_calls, 1)
        self.assertEqual(env.reset_slots_calls, 0)


class TestBatchedRunnerRefill(unittest.TestCase):
    def test_num_envs_less_than_scenarios_triggers_refill(self) -> None:
        env = MockBatchedEnv(
            num_envs=2,
            terminate_at={n: 2 for n in ("a", "b", "c", "d", "e")},
        )
        report = BatchedEvalRunner(
            policies=[from_single(constant_policy(0))],
            scenarios=scenarios_named("a", "b", "c", "d", "e", max_steps=10),
            environment=env,
        ).run()
        self.assertEqual(len(report.episodes), 5)
        # Started with 2 in flight; 3 refills needed to drain queue
        self.assertGreaterEqual(env.reset_slots_calls, 1)
        # Every scenario name appears exactly once across episodes
        seen = sorted(ep.scenario_name for ep in report.episodes)
        self.assertEqual(seen, ["a", "b", "c", "d", "e"])

    def test_idle_slots_when_more_envs_than_tasks(self) -> None:
        env = MockBatchedEnv(num_envs=5, terminate_at={"a": 2, "b": 3})
        report = BatchedEvalRunner(
            policies=[from_single(constant_policy(0))],
            scenarios=scenarios_named("a", "b", max_steps=10),
            environment=env,
        ).run()
        self.assertEqual(len(report.episodes), 2)
        # 3 slots were idle the whole time; their outcomes should be ignored.
        seen = sorted(ep.scenario_name for ep in report.episodes)
        self.assertEqual(seen, ["a", "b"])


class TestBatchedRunnerMultiPolicy(unittest.TestCase):
    def test_two_policies_two_scenarios_produces_four_episodes(self) -> None:
        env = MockBatchedEnv(num_envs=2, terminate_at={"a": 3, "b": 3})
        report = BatchedEvalRunner(
            policies=[
                from_single(constant_policy(0)),
                from_single(constant_policy(1)),
            ],
            scenarios=scenarios_named("a", "b", max_steps=10),
            environment=env,
            baseline_policy="const_0",
        ).run()
        self.assertEqual(len(report.episodes), 4)
        self.assertIn("const_0", report.policy_summary)
        self.assertIn("const_1", report.policy_summary)
        # Each policy gets its own bulk reset
        self.assertEqual(env.reset_calls, 2)


class TestBatchedRunnerReplication(unittest.TestCase):
    def test_replicas_produce_suffixed_scenario_names(self) -> None:
        env = MockBatchedEnv(num_envs=2, terminate_at={"a": 2, "b": 2})
        report = BatchedEvalRunner(
            policies=[from_single(constant_policy(0))],
            scenarios=scenarios_named("a", "b", max_steps=10),
            environment=env,
            replicas=3,
        ).run()
        # 2 scenarios * 3 replicas = 6 episodes
        self.assertEqual(len(report.episodes), 6)
        seen = sorted(ep.scenario_name for ep in report.episodes)
        expected = sorted([f"a#r{i}" for i in range(3)] + [f"b#r{i}" for i in range(3)])
        self.assertEqual(seen, expected)

    def test_replicas_one_keeps_plain_scenario_names(self) -> None:
        env = MockBatchedEnv(num_envs=2, terminate_at={"a": 2, "b": 2})
        report = BatchedEvalRunner(
            policies=[from_single(constant_policy(0))],
            scenarios=scenarios_named("a", "b"),
            environment=env,
            replicas=1,
        ).run()
        seen = sorted(ep.scenario_name for ep in report.episodes)
        self.assertEqual(seen, ["a", "b"])


class TestBatchedRunnerMaxSteps(unittest.TestCase):
    def test_max_steps_force_terminates_without_env_terminal(self) -> None:
        # terminate_at empty -> env never returns terminal=True
        env = MockBatchedEnv(num_envs=1, terminate_at={})
        report = BatchedEvalRunner(
            policies=[from_single(constant_policy(0))],
            scenarios=scenarios_named("never_terminates", max_steps=4),
            environment=env,
        ).run()
        self.assertEqual(len(report.episodes), 1)
        ep = report.episodes[0]
        self.assertEqual(ep.steps, 4)
        self.assertEqual(ep.terminal_outcome, "max_steps_reached")

    def test_env_terminal_wins_over_max_steps(self) -> None:
        env = MockBatchedEnv(num_envs=1, terminate_at={"a": 3})
        report = BatchedEvalRunner(
            policies=[from_single(constant_policy(0))],
            scenarios=scenarios_named("a", max_steps=10),
            environment=env,
        ).run()
        ep = report.episodes[0]
        self.assertEqual(ep.steps, 3)
        self.assertEqual(ep.terminal_outcome, "terminated_success")


class TestBatchedRunnerRulesetFlow(unittest.TestCase):
    def test_ruleset_failures_appear_in_episode_result(self) -> None:
        env = MockBatchedEnv(num_envs=1, terminate_at={"x": 5})
        # Ruleset requires episode_return >= 100 — way above what 5 steps gives
        report = BatchedEvalRunner(
            policies=[from_single(constant_policy(0))],
            scenarios=scenarios_named("x", max_steps=10),
            environment=env,
            ruleset=Ruleset([
                require_metric("episode_return", ">=", 100.0, name="needs_100"),
            ]),
        ).run()
        ep = report.episodes[0]
        self.assertFalse(ep.success)
        self.assertEqual(ep.failure_label, "needs_100")

    def test_ruleset_success_flows_through(self) -> None:
        env = MockBatchedEnv(num_envs=1, terminate_at={"x": 3})
        report = BatchedEvalRunner(
            policies=[from_single(constant_policy(0))],
            scenarios=scenarios_named("x", max_steps=10),
            environment=env,
            ruleset=Ruleset([require_outcome("terminated_success")]),
        ).run()
        ep = report.episodes[0]
        self.assertTrue(ep.success)


class TestBatchedRunnerReportShape(unittest.TestCase):
    def test_report_contains_expected_fields(self) -> None:
        env = MockBatchedEnv(
            num_envs=2, terminate_at={"a": 3, "b": 3}
        )
        report = BatchedEvalRunner(
            policies=[
                from_single(constant_policy(0)),
                from_single(constant_policy(1)),
            ],
            scenarios=scenarios_named("a", "b"),
            environment=env,
            baseline_policy="const_0",
        ).run()
        # _build_report fills these in
        self.assertEqual(report.baseline_policy, "const_0")
        self.assertEqual(len(report.episodes), 4)
        self.assertIsInstance(report.policy_summary, dict)
        self.assertIsInstance(report.regressions, list)
        self.assertIsInstance(report.improvements, list)

    def test_steprecord_logs_carry_per_slot_info(self) -> None:
        env = MockBatchedEnv(num_envs=2, terminate_at={"a": 3, "b": 5})
        report = BatchedEvalRunner(
            policies=[from_single(constant_policy(0))],
            scenarios=scenarios_named("a", "b"),
            environment=env,
        ).run()
        for ep in report.episodes:
            for log in ep.logs:
                self.assertIn("slot_scenario", log.next_state)
                self.assertEqual(log.next_state["slot_scenario"], ep.scenario_name)


class TestBatchedRunnerNativeBatchedPolicy(unittest.TestCase):
    def test_natively_batched_policy_runs(self) -> None:
        env = MockBatchedEnv(num_envs=2, terminate_at={"a": 2, "b": 2})
        report = BatchedEvalRunner(
            policies=[batched_constant_policy(0, name="native_v0")],
            scenarios=scenarios_named("a", "b"),
            environment=env,
        ).run()
        self.assertEqual(len(report.episodes), 2)
        self.assertIn("native_v0", report.policy_summary)


class TestBatchedRunnerInputValidation(unittest.TestCase):
    def test_no_policies_raises(self) -> None:
        env = MockBatchedEnv(num_envs=1)
        with self.assertRaisesRegex(ValueError, r"at least one policy"):
            BatchedEvalRunner(policies=[], scenarios=scenarios_named("a"), environment=env)

    def test_no_scenarios_raises(self) -> None:
        env = MockBatchedEnv(num_envs=1)
        with self.assertRaisesRegex(ValueError, r"at least one scenario"):
            BatchedEvalRunner(
                policies=[from_single(constant_policy(0))],
                scenarios=[],
                environment=env,
            )

    def test_environment_without_num_envs_raises(self) -> None:
        class Bad:
            def reset(self, scenarios): return []
            def step(self, actions): return None
            def reset_slots(self, slots, scenarios): return []

        with self.assertRaisesRegex(TypeError, r"num_envs"):
            BatchedEvalRunner(
                policies=[from_single(constant_policy(0))],
                scenarios=scenarios_named("a"),
                environment=Bad(),
            )

    def test_zero_replicas_raises(self) -> None:
        env = MockBatchedEnv(num_envs=1)
        with self.assertRaisesRegex(ValueError, r"replicas must be positive"):
            BatchedEvalRunner(
                policies=[from_single(constant_policy(0))],
                scenarios=scenarios_named("a"),
                environment=env,
                replicas=0,
            )


if __name__ == "__main__":
    unittest.main()
