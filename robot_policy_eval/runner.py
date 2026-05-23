from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .adapters import PolicyLike, normalize_policy
from .core import EpisodeResult, EvalReport, Scenario, StepRecord, SuccessCriteria
from .environment import DemoRobotEnvironment


class EvalRunner:
    def __init__(
        self,
        policies: Iterable[PolicyLike],
        scenarios: Iterable[Scenario],
        success_criteria: SuccessCriteria | None = None,
        baseline_policy: str | None = None,
        environment: DemoRobotEnvironment | None = None,
    ) -> None:
        self.policies = [normalize_policy(policy) for policy in policies]
        self.scenarios = list(scenarios)
        self.success_criteria = success_criteria or SuccessCriteria()
        self.baseline_policy = baseline_policy or self.policies[0].name
        self.environment = environment or DemoRobotEnvironment()

    def run(self) -> EvalReport:
        episodes: list[EpisodeResult] = []
        for policy in self.policies:
            for scenario in self.scenarios:
                episodes.append(self._run_episode(policy, scenario))
        return _build_report(episodes, self.baseline_policy)

    def _run_episode(self, policy, scenario: Scenario) -> EpisodeResult:
        episode_id = f"{scenario.name}:{policy.name}"
        state = self.environment.reset(scenario)
        logs: list[StepRecord] = []
        terminal_outcome = "max_steps_reached"

        for step in range(scenario.max_steps):
            decision = policy.decide(state)
            step_outcome = self.environment.step(decision.action, scenario)
            logs.append(
                StepRecord(
                    episode_id=episode_id,
                    scenario_name=scenario.name,
                    policy_version=policy.name,
                    step=step,
                    state=dict(state),
                    action=decision.action,
                    outcome=step_outcome.outcome,
                    failure_label=step_outcome.failure_label,
                    debug_info=decision.debug_info,
                )
            )
            state = step_outcome.next_state
            if step_outcome.terminal:
                terminal_outcome = step_outcome.outcome
                break

        success, failure_label = self.success_criteria.evaluate(logs, terminal_outcome)
        return EpisodeResult(
            episode_id=episode_id,
            scenario_name=scenario.name,
            policy_version=policy.name,
            success=success,
            terminal_outcome=terminal_outcome,
            failure_label=failure_label,
            steps=len(logs),
            logs=logs,
        )


def _build_report(episodes: list[EpisodeResult], baseline_policy: str) -> EvalReport:
    by_policy: dict[str, list[EpisodeResult]] = defaultdict(list)
    by_scenario: dict[str, dict[str, EpisodeResult]] = defaultdict(dict)

    for episode in episodes:
        by_policy[episode.policy_version].append(episode)
        by_scenario[episode.scenario_name][episode.policy_version] = episode

    policy_summary = {
        policy: _summarize(policy_episodes)
        for policy, policy_episodes in by_policy.items()
    }
    regressions = []
    improvements = []
    failure_cases = []

    for episode in episodes:
        if not episode.success:
            failure_cases.append(_failure_case(episode))

    for scenario_name, scenario_results in by_scenario.items():
        baseline = scenario_results.get(baseline_policy)
        if baseline is None:
            continue
        for policy, result in scenario_results.items():
            if policy == baseline_policy:
                continue
            if baseline.success and not result.success:
                regressions.append(
                    {
                        "scenario_name": scenario_name,
                        "baseline_policy": baseline_policy,
                        "policy_version": policy,
                        "failure_label": result.failure_label,
                        "terminal_outcome": result.terminal_outcome,
                    }
                )
            if not baseline.success and result.success:
                improvements.append(
                    {
                        "scenario_name": scenario_name,
                        "baseline_policy": baseline_policy,
                        "policy_version": policy,
                        "baseline_failure_label": baseline.failure_label,
                    }
                )

    return EvalReport(
        baseline_policy=baseline_policy,
        policy_summary=policy_summary,
        episodes=episodes,
        regressions=regressions,
        improvements=improvements,
        failure_cases=failure_cases,
    )


def _summarize(episodes: list[EpisodeResult]) -> dict[str, float | int]:
    total = len(episodes)
    success_count = sum(1 for episode in episodes if episode.success)
    return {
        "success_rate": round(success_count / total, 3) if total else 0.0,
        "failure_count": total - success_count,
        "collision_count": sum(episode.collision_count for episode in episodes),
        "stuck_count": sum(episode.stuck_count for episode in episodes),
        "unsafe_action_count": sum(episode.unsafe_action_count for episode in episodes),
        "average_steps": round(sum(episode.steps for episode in episodes) / total, 2) if total else 0.0,
    }


def _failure_case(episode: EpisodeResult) -> dict[str, object]:
    last_log = episode.logs[-1] if episode.logs else None
    return {
        "episode_id": episode.episode_id,
        "scenario_name": episode.scenario_name,
        "policy_version": episode.policy_version,
        "failure_label": episode.failure_label,
        "terminal_outcome": episode.terminal_outcome,
        "steps": episode.steps,
        "last_state": last_log.state if last_log else {},
        "last_action": last_log.action if last_log else "",
        "debug_info": last_log.debug_info if last_log else {},
    }
