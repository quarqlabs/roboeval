from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

from .adapters import PolicyAdapter, PolicyLike, normalize_policy
from .core import (
    SDK_NAME,
    SDK_VERSION,
    Action,
    ActionValidator,
    EpisodeContext,
    EpisodeResult,
    EvalReport,
    Ruleset,
    Scenario,
    StateValidator,
    StepRecord,
    SuccessCriteria,
    action_key,
    display_value,
    to_serializable,
)
from .environment import DemoRobotEnvironment, EnvironmentAdapter


class EvalRunner:
    def __init__(
        self,
        policies: Iterable[PolicyLike],
        scenarios: Iterable[Scenario],
        success_criteria: SuccessCriteria | None = None,
        ruleset: Ruleset | None = None,
        baseline_policy: str | None = None,
        environment: EnvironmentAdapter | None = None,
        allowed_actions: Iterable[Action] | None = None,
        required_state_keys: Iterable[str] | None = None,
        state_validator: StateValidator | None = None,
        action_validator: ActionValidator | None = None,
    ) -> None:
        if ruleset is not None and success_criteria is not None:
            raise ValueError("Provide either ruleset or success_criteria, not both.")
        self.policies = [normalize_policy(policy) for policy in policies]
        self.scenarios = list(scenarios)
        self.success_criteria = success_criteria
        self.ruleset = ruleset or (success_criteria or SuccessCriteria()).to_ruleset()
        self.baseline_policy = baseline_policy or self.policies[0].name
        self.environment = environment or DemoRobotEnvironment()
        self.environment_name = _environment_name(self.environment)
        self.allowed_actions = list(allowed_actions or [])
        self.allowed_action_keys = {action_key(action) for action in self.allowed_actions}
        self.required_state_keys = set(required_state_keys or [])
        self.state_validator = state_validator
        self.action_validator = action_validator
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
        self._validate_state(state, scenario.name)
        logs: list[StepRecord] = []
        terminal_outcome = "max_steps_reached"

        for step in range(scenario.max_steps):
            decision = policy.decide(state)
            self._validate_action(decision.action, policy.name)
            step_outcome = self.environment.step(decision.action, scenario)
            _validate_step_outcome(step_outcome)
            self._validate_state(step_outcome.next_state, scenario.name)
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
                    metrics=dict(step_outcome.metrics or {}),
                    events=list(step_outcome.events or []),
                    artifacts=dict(step_outcome.artifacts or {}),
                    info=dict(step_outcome.info or {}),
                )
            )
            state = step_outcome.next_state
            if step_outcome.terminal:
                terminal_outcome = step_outcome.outcome
                break

        context = EpisodeContext(
            episode_id=episode_id,
            scenario=scenario,
            policy_version=policy.name,
            logs=logs,
            terminal_outcome=terminal_outcome,
        )
        rule_results = self.ruleset.evaluate(context)
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

    def _validate_state(self, state: dict[str, Any], scenario_name: str) -> None:
        missing_keys = sorted(key for key in self.required_state_keys if key not in state)
        if missing_keys:
            raise ValueError(f"Scenario {scenario_name!r} state is missing required keys: {missing_keys}")
        if self.state_validator:
            _validate_callback_result(self.state_validator(state), "state_validator")

    def _validate_action(self, action: Action, policy_name: str) -> None:
        if self.allowed_action_keys and action_key(action) not in self.allowed_action_keys:
            raise ValueError(
                f"Policy {policy_name!r} returned action {display_value(action)!r}, not in allowed actions."
            )
        if self.action_validator:
            _validate_callback_result(self.action_validator(action), "action_validator")


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
        failure_counts=_failure_counts(episodes),
        outcome_counts=_outcome_counts(episodes),
        metric_summary=_metric_summary(episodes),
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
        action = display_value(failure.get("failure_action", ""))
        outcome = str(failure.get("failure_outcome", ""))
        if action:
            highlights.append(
                f"{policy_version} failed {scenario_name} on rule {failure_label} at step {step}; action={action}, outcome={outcome}."
            )
        else:
            highlights.append(f"{policy_version} failed {scenario_name} on rule {failure_label} at step {step}.")

    for divergence in action_divergences[:10]:
        highlights.append(
            f"{divergence['policy_version']} chose {display_value(divergence['candidate_action'])} while baseline "
            f"{divergence['baseline_policy']} chose {display_value(divergence['baseline_action'])} on "
            f"{divergence['scenario_name']} at step {divergence['step']}."
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


def _failure_counts(episodes: list[EpisodeResult]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for episode in episodes:
        for log in episode.logs:
            if log.failure_label:
                counts[episode.policy_version][log.failure_label] += 1
        if not episode.success and episode.failure_label:
            counts[episode.policy_version][episode.failure_label] += 0
    return {policy: dict(policy_counts) for policy, policy_counts in counts.items()}


def _outcome_counts(episodes: list[EpisodeResult]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for episode in episodes:
        for log in episode.logs:
            if log.outcome:
                counts[episode.policy_version][log.outcome] += 1
    return {policy: dict(policy_counts) for policy, policy_counts in counts.items()}


def _metric_summary(episodes: list[EpisodeResult]) -> dict[str, dict[str, dict[str, float | int]]]:
    values_by_policy: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    last_by_policy: dict[str, dict[str, float]] = defaultdict(dict)
    for episode in episodes:
        for log in episode.logs:
            for metric_name, raw_value in log.metrics.items():
                if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                    continue
                value = float(raw_value)
                values_by_policy[episode.policy_version][metric_name].append(value)
                last_by_policy[episode.policy_version][metric_name] = value
    return {
        policy: {
            metric_name: {
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "avg": round(sum(values) / len(values), 4),
                "last": round(last_by_policy[policy][metric_name], 4),
            }
            for metric_name, values in metric_values.items()
        }
        for policy, metric_values in values_by_policy.items()
    }


def _episode_story(episode: EpisodeResult) -> str:
    outcomes = _dedupe_consecutive(episode.outcome_trace)
    if outcomes:
        return f"outcome trace: {_join_trace(outcomes)}"
    return f"ended with {episode.terminal_outcome}"


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def _dedupe_consecutive(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if not deduped or deduped[-1] != value:
            deduped.append(value)
    return deduped


def _join_trace(values: list[str]) -> str:
    if not values:
        return ""
    return " -> ".join(values)


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
        "failure_state": to_serializable(failure_log.state) if failure_log else {},
        "failure_next_state": to_serializable(failure_log.next_state) if failure_log else {},
        "failure_action": failure_log.action if failure_log else "",
        "failure_outcome": failure_log.outcome if failure_log else "",
        "failure_metrics": to_serializable(failure_log.metrics) if failure_log else {},
        "failure_events": to_serializable(failure_log.events) if failure_log else [],
        "outcome_trace": to_serializable(episode.outcome_trace),
        "action_trace": to_serializable(episode.action_trace),
        "last_state": to_serializable(last_log.state) if last_log else {},
        "last_next_state": to_serializable(last_log.next_state) if last_log else {},
        "last_action": last_log.action if last_log else "",
        "debug_info": to_serializable(failure_log.debug_info) if failure_log else {},
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
        if action_key(baseline_action) != action_key(candidate_action):
            reference_log = candidate_log or baseline_log
            return {
                "baseline_policy": baseline.policy_version,
                "policy_version": candidate.policy_version,
                "step": index,
                "baseline_action": to_serializable(baseline_action),
                "candidate_action": to_serializable(candidate_action),
                "baseline_action_key": action_key(baseline_action),
                "candidate_action_key": action_key(candidate_action),
                "state": to_serializable(reference_log.state) if reference_log else {},
                "candidate_outcome": candidate_log.outcome if candidate_log else "<ended>",
                "baseline_outcome": baseline_log.outcome if baseline_log else "<ended>",
                "candidate_debug_info": to_serializable(candidate_log.debug_info) if candidate_log else {},
                "baseline_debug_info": to_serializable(baseline_log.debug_info) if baseline_log else {},
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
    for attr in ("metrics", "artifacts", "info"):
        value = getattr(step_outcome, attr, None)
        if value is not None and not isinstance(value, dict):
            raise TypeError(f"Environment step() result {attr} must be a dict when provided.")
    if getattr(step_outcome, "events", None) is not None and not isinstance(step_outcome.events, list):
        raise TypeError("Environment step() result events must be a list when provided.")


def _environment_name(environment: EnvironmentAdapter) -> str:
    name = getattr(environment, "name", None)
    return str(name) if name else environment.__class__.__name__


def _validate_callback_result(result: bool | str | None, callback_name: str) -> None:
    if result is False:
        raise ValueError(f"{callback_name} rejected the value.")
    if isinstance(result, str) and result:
        raise ValueError(result)
