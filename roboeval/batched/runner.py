"""BatchedEvalRunner — the vectorized counterpart of EvalRunner.

Drives a BatchedEnvironmentAdapter through (policies × scenarios × replicas)
rollouts using a SlotScheduler to keep all N env slots busy. Per-slot
episodes are buffered into normal single-env StepRecord lists, then fed into
the same _build_report() that the single-env runner uses — so reports come
out byte-identical between the two paths (decision D5).

Execution model:
  * One policy at a time. The runner does all of policy P's rollouts before
    moving to policy P+1. This keeps things simple and is how lerobot_eval
    works. Mixing policies across slots is a future optimization.
  * Idle slot handling: when the queue drains but other slots are still
    rolling, idle slots receive a placeholder action (copied from the first
    active slot) and their outcomes are discarded.
  * max_steps enforcement: per-slot step counter. If a slot reaches
    scenario.max_steps without env-terminating, the runner force-terminates
    that episode with terminal_outcome="max_steps_reached".

Replication (D1): pass ``replicas=N`` to run each scenario N times. With
replicas > 1, EpisodeResult.scenario_name is suffixed with ``#r{i}`` so each
replica produces a distinct report row.
"""

from __future__ import annotations

from typing import Any, Iterable

from roboeval.core import (
    EpisodeContext,
    EpisodeResult,
    EvalReport,
    Ruleset,
    Scenario,
    StepRecord,
    SuccessCriteria,
)
from roboeval.runner import _build_report, _environment_name, _validate_step_outcome

from .environment import BatchedEnvironmentAdapter
from .policy import BatchedPolicyAdapter, normalize_batched_policy
from .scheduler import SlotScheduler, SlotTask
from .types import BatchedStepOutcome


class BatchedEvalRunner:
    """Vectorized eval runner. Produces the same EvalReport as EvalRunner."""

    def __init__(
        self,
        policies: Iterable[Any],
        scenarios: Iterable[Scenario],
        environment: BatchedEnvironmentAdapter,
        ruleset: Ruleset | None = None,
        baseline_policy: str | None = None,
        replicas: int = 1,
    ) -> None:
        self.policies: list[BatchedPolicyAdapter] = [
            normalize_batched_policy(policy) for policy in policies
        ]
        self.scenarios: list[Scenario] = list(scenarios)
        self.environment = environment
        self.environment_name = _environment_name(environment)
        self.ruleset = ruleset or SuccessCriteria().to_ruleset()
        self.baseline_policy = baseline_policy or (
            self.policies[0].name if self.policies else ""
        )
        if replicas <= 0:
            raise ValueError(f"replicas must be positive, got {replicas}.")
        self.replicas = replicas
        if not self.policies:
            raise ValueError("BatchedEvalRunner needs at least one policy.")
        if not self.scenarios:
            raise ValueError("BatchedEvalRunner needs at least one scenario.")
        if not hasattr(environment, "num_envs"):
            raise TypeError(
                "BatchedEvalRunner.environment must expose num_envs "
                "(BatchedEnvironmentAdapter Protocol)."
            )

    # ----- public -----

    def run(self) -> EvalReport:
        episodes: list[EpisodeResult] = []
        for policy in self.policies:
            episodes.extend(self._run_policy(policy))
        return _build_report(episodes, self.baseline_policy, self.environment_name)

    # ----- per-policy rollout -----

    def _run_policy(self, policy: BatchedPolicyAdapter) -> list[EpisodeResult]:
        n = self.environment.num_envs
        sched = SlotScheduler(
            num_envs=n, scenarios=self.scenarios, replicas=self.replicas
        )
        initial_scenarios = sched.initialize()

        if sched.is_done():
            return []

        # Build the reset batch: every slot needs a Scenario for env.reset.
        # Idle slots get a placeholder (any active scenario), their outcomes
        # are ignored throughout.
        placeholder_scenario = next(s for s in initial_scenarios if s is not None)
        reset_batch = [s if s is not None else placeholder_scenario for s in initial_scenarios]
        slot_states = list(self.environment.reset(reset_batch))
        if len(slot_states) != n:
            raise ValueError(
                f"environment.reset returned {len(slot_states)} states, expected {n}."
            )

        # Per-slot rollout buffers
        slot_step_counts: list[int] = [0] * n
        slot_logs: list[list[StepRecord]] = [[] for _ in range(n)]
        completed: list[EpisodeResult] = []

        while not sched.is_done():
            active = sched.active_slots()
            active_states = [slot_states[i] for i in active]
            decisions = policy.decide(active_states)
            slot_decisions = [None] * n
            for j, slot in enumerate(active):
                slot_decisions[slot] = decisions[j]

            # Build full action batch (idle slots get a placeholder action).
            placeholder_action = slot_decisions[active[0]].action
            actions = [
                slot_decisions[i].action if slot_decisions[i] is not None else placeholder_action
                for i in range(n)
            ]

            outcome_batch = self.environment.step(actions)
            _validate_batched_step_outcome(outcome_batch, n)

            slots_to_refill_idx: list[int] = []
            scenarios_to_refill: list[Scenario] = []

            for slot in active:
                task = sched.current_task(slot)
                step_outcome = outcome_batch.slot(slot)
                # Per-slot validation reuses the single-env validator
                _validate_step_outcome(step_outcome)
                decision = slot_decisions[slot]
                step_idx = slot_step_counts[slot]

                episode_id, scenario_name = _episode_identity(
                    task, policy.name, self.replicas
                )

                slot_logs[slot].append(
                    StepRecord(
                        episode_id=episode_id,
                        scenario_name=scenario_name,
                        policy_version=policy.name,
                        step=step_idx,
                        state=dict(slot_states[slot]),
                        action=decision.action,
                        outcome=step_outcome.outcome,
                        failure_label=step_outcome.failure_label,
                        next_state=dict(step_outcome.next_state),
                        is_terminal=bool(step_outcome.terminal),
                        debug_info=dict(decision.debug_info),
                        metrics=dict(step_outcome.metrics or {}),
                        events=list(step_outcome.events or []),
                        artifacts=dict(step_outcome.artifacts or {}),
                        info=dict(step_outcome.info or {}),
                    )
                )

                slot_states[slot] = step_outcome.next_state
                slot_step_counts[slot] = step_idx + 1

                env_term = bool(step_outcome.terminal)
                max_steps_hit = slot_step_counts[slot] >= task.scenario.max_steps
                if env_term or max_steps_hit:
                    terminal_outcome = (
                        step_outcome.outcome if env_term else "max_steps_reached"
                    )
                    completed.append(
                        _finalize_episode(
                            task=task,
                            policy_name=policy.name,
                            replicas=self.replicas,
                            logs=slot_logs[slot],
                            terminal_outcome=terminal_outcome,
                            ruleset=self.ruleset,
                        )
                    )
                    refill = sched.complete_slot(slot)
                    if refill is not None:
                        slots_to_refill_idx.append(slot)
                        scenarios_to_refill.append(refill.scenario)
                    slot_step_counts[slot] = 0
                    slot_logs[slot] = []

            if slots_to_refill_idx:
                new_states = self.environment.reset_slots(
                    slots_to_refill_idx, scenarios_to_refill
                )
                if len(new_states) != len(slots_to_refill_idx):
                    raise ValueError(
                        f"reset_slots returned {len(new_states)} states for "
                        f"{len(slots_to_refill_idx)} requested slots."
                    )
                for j, slot in enumerate(slots_to_refill_idx):
                    slot_states[slot] = new_states[j]

        return completed


# ----- helpers -----


def _episode_identity(
    task: SlotTask, policy_name: str, replicas: int
) -> tuple[str, str]:
    """Return (episode_id, scenario_name) for the task.

    With replicas > 1, scenario_name carries a #r{idx} suffix so each replica
    produces a distinct EpisodeResult row in the report.
    """
    if replicas > 1:
        scenario_name = f"{task.scenario.name}#r{task.replica_idx}"
    else:
        scenario_name = task.scenario.name
    episode_id = f"{scenario_name}:{policy_name}"
    return episode_id, scenario_name


def _finalize_episode(
    *,
    task: SlotTask,
    policy_name: str,
    replicas: int,
    logs: list[StepRecord],
    terminal_outcome: str,
    ruleset: Ruleset,
) -> EpisodeResult:
    episode_id, scenario_name = _episode_identity(task, policy_name, replicas)
    context = EpisodeContext(
        episode_id=episode_id,
        scenario=task.scenario,
        policy_version=policy_name,
        logs=logs,
        terminal_outcome=terminal_outcome,
    )
    rule_results = ruleset.evaluate(context)
    first_failure = next((r for r in rule_results if not r.passed), None)
    success = first_failure is None
    failure_label = "" if success else first_failure.name
    return EpisodeResult(
        episode_id=episode_id,
        scenario_name=scenario_name,
        policy_version=policy_name,
        success=success,
        terminal_outcome=terminal_outcome,
        failure_label=failure_label,
        steps=len(logs),
        logs=logs,
        rule_results=rule_results,
        first_failure_step=first_failure.step if first_failure else None,
        scenario_metadata=dict(task.scenario.metadata),
    )


def _validate_batched_step_outcome(outcome: Any, expected_num_envs: int) -> None:
    if not isinstance(outcome, BatchedStepOutcome):
        raise TypeError(
            f"BatchedEnvironmentAdapter.step must return BatchedStepOutcome, "
            f"got {type(outcome).__name__}."
        )
    if outcome.num_envs != expected_num_envs:
        raise ValueError(
            f"BatchedStepOutcome.num_envs is {outcome.num_envs}, expected "
            f"{expected_num_envs} (matches environment.num_envs)."
        )
