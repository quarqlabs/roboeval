from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

from .adapters import PolicyAdapter, PolicyLike, normalize_policy
from .core import SDK_NAME, SDK_VERSION, EpisodeResult, EvalReport, Scenario, StepRecord, SuccessCriteria
from .environment import DemoRobotEnvironment, EnvironmentAdapter


class EvalRunner:
    def __init__(
        self,
        policies: Iterable[PolicyLike],
        scenarios: Iterable[Scenario],
        success_criteria: SuccessCriteria | None = None,
        baseline_policy: str | None = None,
        environment: EnvironmentAdapter | None = None,
    ) -> None:
        self.policies = [normalize_policy(policy) for policy in policies]
        self.scenarios = list(scenarios)
        self.success_criteria = success_criteria or SuccessCriteria()
        self.baseline_policy = baseline_policy or self.policies[0].name
        self.environment = environment or DemoRobotEnvironment()
        self.environment_name = _environment_name(self.environment)
        _validate_runner_inputs(self.policies, self.scenarios, self.baseline_policy)

    def run(self) -> EvalReport:
        episodes: list[EpisodeResult] = []
        for policy in self.policies:
            for scenario in self.scenarios:
                episodes.append(self._run_episode(policy, scenario))
        return _build_report(episodes, self.baseline_policy, self.environment_name)

    def _run_episode(self, policy: PolicyAdapter, scenario: Scenario) -> EpisodeResult:
        episode_id = f"{scenario.name}:{policy.name}"
        state = self.environment.reset(scenario)
        if not isinstance(state, dict):
            raise TypeError("Environment reset() must return a state dict.")
        logs: list[StepRecord] = []
        terminal_outcome = "max_steps_reached"

        for step in range(scenario.max_steps):
            decision = policy.decide(state)
            step_outcome = self.environment.step(decision.action, scenario)
            _validate_step_outcome(step_outcome)
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
                    next_state=dict(step_outcome.next_state),
                    is_terminal=bool(step_outcome.terminal),
                    debug_info=decision.debug_info,
                )
            )
            state = step_outcome.next_state
            if step_outcome.terminal:
                terminal_outcome = step_outcome.outcome
                break

        rule_results = self.success_criteria.evaluate_rules(logs, terminal_outcome)
        first_failure = next((result for result in rule_results if not result.passed), None)
        success = first_failure is None
        failure_label = "" if success else first_failure.name
        return EpisodeResult(
            episode_id=episode_id,
            scenario_name=scenario.name,
            policy_version=policy.name,
            success=success,
            terminal_outcome=terminal_outcome,
            failure_label=failure_label,
            steps=len(logs),
            logs=logs,
            rule_results=rule_results,
            first_failure_step=first_failure.step if first_failure else None,
            scenario_metadata=dict(scenario.metadata),
        )


def _build_report(episodes: list[EpisodeResult], baseline_policy: str, environment_name: str) -> EvalReport:
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
                        "first_failure_step": result.first_failure_step,
                    }
                )
            if not baseline.success and result.success:
                improvements.append(
                    {
                        "scenario_name": scenario_name,
                        "baseline_policy": baseline_policy,
                        "policy_version": policy,
                        "baseline_failure_label": baseline.failure_label,
                        "baseline_first_failure_step": baseline.first_failure_step,
                    }
                )

    action_divergences = _action_divergences(by_scenario, baseline_policy)
    return EvalReport(
        baseline_policy=baseline_policy,
        policy_summary=policy_summary,
        episodes=episodes,
        regressions=regressions,
        improvements=improvements,
        failure_cases=failure_cases,
        grouped_metrics=_grouped_metrics(episodes),
        action_divergences=action_divergences,
        metadata=_run_metadata(episodes, baseline_policy, environment_name),
        highlights=_build_highlights(improvements, regressions, failure_cases, action_divergences, by_scenario),
    )


def _run_metadata(episodes: list[EpisodeResult], baseline_policy: str, environment_name: str) -> dict[str, Any]:
    policy_versions = list(dict.fromkeys(episode.policy_version for episode in episodes))
    scenario_names = list(dict.fromkeys(episode.scenario_name for episode in episodes))
    return {
        "sdk_name": SDK_NAME,
        "sdk_version": SDK_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "baseline_policy": baseline_policy,
        "policy_versions": policy_versions,
        "scenario_count": len(scenario_names),
        "episode_count": len(episodes),
        "environment_name": environment_name,
    }


def _build_highlights(
    improvements: list[dict[str, Any]],
    regressions: list[dict[str, Any]],
    failure_cases: list[dict[str, Any]],
    action_divergences: list[dict[str, Any]],
    by_scenario: dict[str, dict[str, EpisodeResult]],
) -> list[str]:
    highlights: list[str] = []

    for improvement in improvements:
        scenario_name = str(improvement["scenario_name"])
        policy_version = str(improvement["policy_version"])
        baseline_policy = str(improvement["baseline_policy"])
        baseline_failure = str(improvement["baseline_failure_label"])
        highlights.append(
            f"{policy_version} improved {scenario_name}; baseline {baseline_policy} failed with {baseline_failure}."
        )
        candidate = by_scenario.get(scenario_name, {}).get(policy_version)
        story = _episode_story(candidate) if candidate else ""
        if story:
            highlights.append(f"{policy_version} {story} on {scenario_name}.")

    for regression in regressions:
        highlights.append(
            "{policy_version} regressed on {scenario_name}; baseline {baseline_policy} passed but candidate failed with {failure_label}.".format(
                **regression
            )
        )

    for failure in failure_cases:
        step = failure.get("first_failure_step")
        policy_version = str(failure["policy_version"])
        scenario_name = str(failure["scenario_name"])
        failure_label = str(failure["failure_label"])
        if failure_label == "unsafe_forward_action":
            highlights.append(f"{policy_version} moved forward unsafely on {scenario_name} at step {step}.")
        else:
            action = str(failure.get("failure_action", ""))
            if action:
                highlights.append(
                    f"{policy_version} {_past_action(action)} on {scenario_name} and failed with {failure_label} at step {step}."
                )
            else:
                highlights.append(f"{policy_version} failed {scenario_name} with {failure_label} at step {step}.")

    for divergence in action_divergences[:10]:
        highlights.append(
            "{policy_version} chose {candidate_action} while baseline {baseline_policy} chose {baseline_action} on {scenario_name} at step {step}.".format(
                **divergence
            )
        )

    return highlights


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


def _episode_story(episode: EpisodeResult) -> str:
    phrases: list[str] = []
    for log in episode.logs:
        if log.action == "reverse":
            phrases.append("reversed")
        if log.outcome == "escape_reverse":
            phrases.append("escaped")
        if log.outcome == "aligned_turn":
            phrases.append("aligned with the goal")
        if log.failure_label == "collision":
            phrases.append("collided")
        if log.failure_label == "stuck":
            phrases.append("got stuck")
    if episode.terminal_outcome == "goal_reached":
        phrases.append("reached goal")
    elif not phrases:
        phrases.append(f"ended with {episode.terminal_outcome}")
    return _join_story(_dedupe(phrases))


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def _join_story(phrases: list[str]) -> str:
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]}, then {phrases[1]}"
    return f"{', '.join(phrases[:-1])}, then {phrases[-1]}"


def _past_action(action: str) -> str:
    return {
        "move_forward": "moved forward",
        "turn_left": "turned left",
        "turn_right": "turned right",
        "stop": "stopped",
        "reverse": "reversed",
    }.get(action, f"took action {action}")


def _failure_case(episode: EpisodeResult) -> dict[str, object]:
    last_log = episode.logs[-1] if episode.logs else None
    failure_log = _log_for_step(episode.logs, episode.first_failure_step) or last_log
    failed_rules = [result.to_dict() for result in episode.rule_results if not result.passed]
    return {
        "episode_id": episode.episode_id,
        "scenario_name": episode.scenario_name,
        "policy_version": episode.policy_version,
        "failure_label": episode.failure_label,
        "terminal_outcome": episode.terminal_outcome,
        "steps": episode.steps,
        "first_failure_step": episode.first_failure_step,
        "failed_rules": failed_rules,
        "failure_state": failure_log.state if failure_log else {},
        "failure_next_state": failure_log.next_state if failure_log else {},
        "failure_action": failure_log.action if failure_log else "",
        "last_state": last_log.state if last_log else {},
        "last_next_state": last_log.next_state if last_log else {},
        "last_action": last_log.action if last_log else "",
        "debug_info": failure_log.debug_info if failure_log else {},
    }


def _action_divergences(
    by_scenario: dict[str, dict[str, EpisodeResult]],
    baseline_policy: str,
) -> list[dict[str, Any]]:
    divergences: list[dict[str, Any]] = []
    for scenario_name, scenario_results in by_scenario.items():
        baseline = scenario_results.get(baseline_policy)
        if baseline is None:
            continue
        for policy, candidate in scenario_results.items():
            if policy == baseline_policy:
                continue
            divergence = _first_action_divergence(baseline, candidate)
            if divergence:
                divergences.append({"scenario_name": scenario_name, **divergence})
    return divergences


def _first_action_divergence(baseline: EpisodeResult, candidate: EpisodeResult) -> dict[str, Any] | None:
    max_steps = max(len(baseline.logs), len(candidate.logs))
    for index in range(max_steps):
        baseline_log = baseline.logs[index] if index < len(baseline.logs) else None
        candidate_log = candidate.logs[index] if index < len(candidate.logs) else None
        baseline_action = baseline_log.action if baseline_log else "<ended>"
        candidate_action = candidate_log.action if candidate_log else "<ended>"
        if baseline_action != candidate_action:
            reference_log = candidate_log or baseline_log
            return {
                "baseline_policy": baseline.policy_version,
                "policy_version": candidate.policy_version,
                "step": index,
                "baseline_action": baseline_action,
                "candidate_action": candidate_action,
                "state": reference_log.state if reference_log else {},
                "candidate_outcome": candidate_log.outcome if candidate_log else "<ended>",
                "baseline_outcome": baseline_log.outcome if baseline_log else "<ended>",
                "candidate_debug_info": candidate_log.debug_info if candidate_log else {},
                "baseline_debug_info": baseline_log.debug_info if baseline_log else {},
            }
    return None


def _grouped_metrics(episodes: list[EpisodeResult]) -> dict[str, dict[str, dict[str, float | int]]]:
    grouped: dict[str, dict[str, list[EpisodeResult]]] = defaultdict(lambda: defaultdict(list))
    for episode in episodes:
        for group_name in _scenario_groups(episode):
            grouped[group_name][episode.policy_version].append(episode)
    return {
        group_name: {
            policy: _summarize(group_episodes)
            for policy, group_episodes in policy_groups.items()
        }
        for group_name, policy_groups in grouped.items()
    }


def _scenario_groups(episode: EpisodeResult) -> list[str]:
    groups = []
    scenario_type = episode.scenario_metadata.get("scenario_type")
    if scenario_type:
        groups.append(str(scenario_type))
    tags = episode.scenario_metadata.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    groups.extend(str(tag) for tag in tags)
    return groups


def _log_for_step(logs: list[StepRecord], step: int | None) -> StepRecord | None:
    if step is None:
        return None
    return next((log for log in logs if log.step == step), None)


def _validate_runner_inputs(policies: list[PolicyAdapter], scenarios: list[Scenario], baseline_policy: str) -> None:
    if not policies:
        raise ValueError("EvalRunner requires at least one policy.")
    if not scenarios:
        raise ValueError("EvalRunner requires at least one scenario.")
    policy_names = {policy.name for policy in policies}
    if baseline_policy not in policy_names:
        raise ValueError(f"Baseline policy {baseline_policy!r} was not found in policies: {sorted(policy_names)}")
    for scenario in scenarios:
        if not scenario.name:
            raise ValueError("Scenario name cannot be empty.")
        if not isinstance(scenario.initial_state, dict) or not scenario.initial_state:
            raise ValueError(f"Scenario {scenario.name!r} must provide a non-empty initial_state.")
        if scenario.max_steps <= 0:
            raise ValueError(f"Scenario {scenario.name!r} must have max_steps > 0.")


def _validate_step_outcome(step_outcome: Any) -> None:
    for attr in ("next_state", "outcome", "failure_label", "terminal"):
        if not hasattr(step_outcome, attr):
            raise TypeError(f"Environment step() result is missing required attribute {attr!r}.")
    if not isinstance(step_outcome.next_state, dict):
        raise TypeError("Environment step() result next_state must be a dict.")


def _environment_name(environment: EnvironmentAdapter) -> str:
    name = getattr(environment, "name", None)
    return str(name) if name else environment.__class__.__name__
