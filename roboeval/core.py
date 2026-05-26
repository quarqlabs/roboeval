from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable


Action = Any
State = dict[str, Any]
SDK_NAME = "RoboEval"
SDK_VERSION = "0.1.0"


@dataclass(frozen=True)
class Decision:
    action: Action
    debug_info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    name: str
    initial_state: State
    max_steps: int = 10
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Scenario":
        metadata = dict(data.get("metadata", {}))
        if "required_forward_steps" in data:
            metadata["required_forward_steps"] = data["required_forward_steps"]
        if "scenario_type" in data:
            metadata["scenario_type"] = data["scenario_type"]
        if "tags" in data:
            metadata["tags"] = data["tags"]
        return cls(
            name=str(data["name"]),
            initial_state=dict(data["initial_state"]),
            max_steps=int(data.get("max_steps", 10)),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return to_serializable(asdict(self))


@dataclass(frozen=True)
class RuleResult:
    name: str
    passed: bool
    reason: str = ""
    step: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_serializable(asdict(self))


EvalRule = Callable[[list["StepRecord"], str], RuleResult]
Rule = Callable[["EpisodeContext"], RuleResult]
StateValidator = Callable[[State], bool | str | None]
ActionValidator = Callable[[Action], bool | str | None]


@dataclass(frozen=True)
class EpisodeContext:
    episode_id: str
    scenario: Scenario
    policy_version: str
    logs: list["StepRecord"]
    terminal_outcome: str

    @property
    def action_trace(self) -> list[Action]:
        return [log.action for log in self.logs]

    @property
    def outcome_trace(self) -> list[str]:
        return [log.outcome for log in self.logs if log.outcome]

    @property
    def failure_trace(self) -> list[str]:
        return [log.failure_label for log in self.logs if log.failure_label]

    @property
    def terminal_log(self) -> "StepRecord | None":
        return self.logs[-1] if self.logs else None


@dataclass(frozen=True)
class Ruleset:
    rules: list[Rule] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Ruleset":
        return cls([_rule_from_mapping(rule_data) for rule_data in data.get("rules", [])])

    def evaluate(self, context: EpisodeContext) -> list[RuleResult]:
        return [rule(context) for rule in self.rules]


def require_outcome(outcome: str, name: str | None = None) -> Rule:
    rule_name = name or f"require_outcome:{outcome}"

    def _rule(context: EpisodeContext) -> RuleResult:
        matching_log = next((log for log in context.logs if log.outcome == outcome), None)
        passed = matching_log is not None or context.terminal_outcome == outcome
        return RuleResult(
            name=rule_name,
            passed=passed,
            reason="" if passed else f"required outcome {outcome!r} was not observed",
            step=matching_log.step if matching_log else None,
            details={"required_outcome": outcome, "terminal_outcome": context.terminal_outcome},
        )

    return _rule


def forbid_failure(failure_label: str, name: str | None = None) -> Rule:
    rule_name = name or f"forbid_failure:{failure_label}"

    def _rule(context: EpisodeContext) -> RuleResult:
        failure = next((log for log in context.logs if log.failure_label == failure_label), None)
        return RuleResult(
            name=rule_name,
            passed=failure is None,
            reason="" if failure is None else f"forbidden failure {failure_label!r} occurred",
            step=failure.step if failure else None,
            details={"failure_label": failure_label},
        )

    return _rule


def max_steps(limit: int, name: str | None = None) -> Rule:
    rule_name = name or f"max_steps:{limit}"

    def _rule(context: EpisodeContext) -> RuleResult:
        passed = len(context.logs) <= limit
        return RuleResult(
            name=rule_name,
            passed=passed,
            reason="" if passed else f"episode used {len(context.logs)} steps, limit is {limit}",
            step=limit if not passed else None,
            details={"limit": limit, "steps": len(context.logs)},
        )

    return _rule


def require_metric(
    metric_name: str,
    operator: str,
    value: float,
    name: str | None = None,
    aggregate: str = "last",
) -> Rule:
    rule_name = name or f"require_metric:{metric_name}:{operator}:{value}"

    def _rule(context: EpisodeContext) -> RuleResult:
        values = [
            float(log.metrics[metric_name])
            for log in context.logs
            if metric_name in log.metrics and _is_number(log.metrics[metric_name])
        ]
        if not values:
            return RuleResult(
                name=rule_name,
                passed=False,
                reason=f"metric {metric_name!r} was not recorded",
                details={"metric": metric_name, "operator": operator, "value": value, "aggregate": aggregate},
            )
        metric_value = _aggregate_metric(values, aggregate)
        passed = _compare_metric(metric_value, operator, float(value))
        return RuleResult(
            name=rule_name,
            passed=passed,
            reason="" if passed else f"{metric_name} {aggregate} value {metric_value} did not satisfy {operator} {value}",
            details={
                "metric": metric_name,
                "operator": operator,
                "value": value,
                "observed": metric_value,
                "aggregate": aggregate,
            },
        )

    return _rule


def custom_rule(rule: Rule, name: str | None = None) -> Rule:
    if name is None:
        return rule

    def _rule(context: EpisodeContext) -> RuleResult:
        result = rule(context)
        return RuleResult(name=name, passed=result.passed, reason=result.reason, step=result.step, details=result.details)

    return _rule


@dataclass(frozen=True)
class SuccessCriteria:
    must_reach_goal: bool = True
    collision_is_failure: bool = True
    stuck_is_failure: bool = True
    unsafe_forward_min_distance: float | None = 20.0
    unsafe_forward_action: Action = "move_forward"
    goal_outcome: str = "goal_reached"
    custom_rules: list[EvalRule] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "SuccessCriteria":
        return cls(
            must_reach_goal=bool(data.get("must_reach_goal", True)),
            collision_is_failure=bool(data.get("collision_is_failure", True)),
            stuck_is_failure=bool(data.get("stuck_is_failure", True)),
            unsafe_forward_min_distance=data.get("unsafe_forward_min_distance", 20.0),
            unsafe_forward_action=data.get("unsafe_forward_action", "move_forward"),
            goal_outcome=str(data.get("goal_outcome", "goal_reached")),
        )

    def evaluate(self, logs: list["StepRecord"], terminal_outcome: str) -> tuple[bool, str]:
        rule_results = self.evaluate_rules(logs, terminal_outcome)
        first_failure = next((result for result in rule_results if not result.passed), None)
        return first_failure is None, first_failure.name if first_failure else ""

    def evaluate_rules(self, logs: list["StepRecord"], terminal_outcome: str) -> list[RuleResult]:
        context = EpisodeContext(
            episode_id="legacy",
            scenario=Scenario(name="legacy", initial_state={}),
            policy_version="legacy",
            logs=logs,
            terminal_outcome=terminal_outcome,
        )
        return self.to_ruleset().evaluate(context)

    def to_ruleset(self) -> Ruleset:
        rules: list[Rule] = []

        if self.unsafe_forward_min_distance is not None:
            rules.append(_unsafe_forward_rule(self.unsafe_forward_action, self.unsafe_forward_min_distance))

        if self.collision_is_failure:
            rules.append(forbid_failure("collision", name="collision"))

        if self.stuck_is_failure:
            rules.append(forbid_failure("stuck", name="stuck"))

        if self.must_reach_goal:
            rules.append(_terminal_outcome_rule(self.goal_outcome, name="goal_reached"))

        for custom_rule in self.custom_rules:
            rules.append(_legacy_rule(custom_rule))

        return Ruleset(rules)


@dataclass(frozen=True)
class StepRecord:
    episode_id: str
    scenario_name: str
    policy_version: str
    step: int
    state: State
    action: Action
    outcome: str
    failure_label: str
    next_state: State = field(default_factory=dict)
    is_terminal: bool = False
    debug_info: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float | int] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "scenario_name": self.scenario_name,
            "policy_version": self.policy_version,
            "step": self.step,
            "state": to_serializable(self.state),
            "action": to_serializable(self.action),
            "outcome": self.outcome,
            "failure_label": self.failure_label,
            "next_state": to_serializable(self.next_state),
            "is_terminal": self.is_terminal,
            "debug_info": to_serializable(self.debug_info),
            "metrics": to_serializable(self.metrics),
            "events": to_serializable(self.events),
            "artifacts": to_serializable(self.artifacts),
            "info": to_serializable(self.info),
        }


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: str
    scenario_name: str
    policy_version: str
    success: bool
    terminal_outcome: str
    failure_label: str
    steps: int
    logs: list[StepRecord]
    rule_results: list[RuleResult] = field(default_factory=list)
    first_failure_step: int | None = None
    scenario_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def collision_count(self) -> int:
        return sum(1 for log in self.logs if log.failure_label == "collision")

    @property
    def stuck_count(self) -> int:
        return sum(1 for log in self.logs if log.failure_label == "stuck")

    @property
    def unsafe_action_count(self) -> int:
        return 1 if self.failure_label == "unsafe_forward_action" else 0

    @property
    def outcome_trace(self) -> list[str]:
        return [log.outcome for log in self.logs if log.outcome]

    @property
    def action_trace(self) -> list[Action]:
        return [log.action for log in self.logs]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "scenario_name": self.scenario_name,
            "policy_version": self.policy_version,
            "success": self.success,
            "terminal_outcome": self.terminal_outcome,
            "failure_label": self.failure_label,
            "steps": self.steps,
            "logs": [log.to_dict() for log in self.logs],
            "rule_results": [result.to_dict() for result in self.rule_results],
            "first_failure_step": self.first_failure_step,
            "scenario_metadata": to_serializable(self.scenario_metadata),
        }


@dataclass(frozen=True)
class EvalReport:
    baseline_policy: str
    policy_summary: dict[str, dict[str, float | int]]
    episodes: list[EpisodeResult]
    regressions: list[dict[str, Any]]
    improvements: list[dict[str, Any]]
    failure_cases: list[dict[str, Any]]
    grouped_metrics: dict[str, dict[str, dict[str, float | int]]] = field(default_factory=dict)
    action_divergences: list[dict[str, Any]] = field(default_factory=list)
    failure_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    outcome_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    metric_summary: dict[str, dict[str, dict[str, float | int]]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    highlights: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return to_serializable({
            "metadata": self.metadata,
            "baseline_policy": self.baseline_policy,
            "policy_summary": self.policy_summary,
            "episodes": [episode.to_dict() for episode in self.episodes],
            "regressions": self.regressions,
            "improvements": self.improvements,
            "failure_cases": self.failure_cases,
            "grouped_metrics": self.grouped_metrics,
            "action_divergences": self.action_divergences,
            "failure_counts": self.failure_counts,
            "outcome_counts": self.outcome_counts,
            "metric_summary": self.metric_summary,
            "highlights": self.highlights,
        })

    def save(self, output_dir: str | Path) -> None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self._write_decision_logs(destination / "decision_logs.jsonl")
        self._write_json(destination / "episode_results.json", [episode.to_dict() for episode in self.episodes])
        self._write_json(
            destination / "comparison_report.json",
            {
                "metadata": self.metadata,
                "baseline_policy": self.baseline_policy,
                "policy_summary": self.policy_summary,
                "regressions": self.regressions,
                "improvements": self.improvements,
                "failure_cases": self.failure_cases,
                "grouped_metrics": self.grouped_metrics,
                "action_divergences": self.action_divergences,
                "failure_counts": self.failure_counts,
                "outcome_counts": self.outcome_counts,
                "metric_summary": self.metric_summary,
                "highlights": self.highlights,
            },
        )
        (destination / "report.md").write_text(self.to_markdown(), encoding="utf-8")

    def to_markdown(self) -> str:
        lines = [
            "# RoboEval Report",
            "",
            "## Run Metadata",
            "",
            "| Field | Value |",
            "| --- | --- |",
        ]
        for key, value in self.metadata.items():
            lines.append(f"| `{key}` | {_markdown_value(value)} |")

        lines.extend(
            [
                "",
                "## Highlights",
                "",
            ]
        )
        if self.highlights:
            for highlight in self.highlights:
                lines.append(f"- {highlight}")
        else:
            lines.append("- No regressions, improvements, or failure highlights detected.")

        lines.extend([
            "",
            "## Policy Summary",
            "",
            "| Policy | Success Rate | Failures | Avg Steps |",
            "| --- | ---: | ---: | ---: |",
        ])
        for policy, summary in self.policy_summary.items():
            lines.append(
                "| {policy} | {success_rate} | {failure_count} | {average_steps} |".format(
                    policy=policy,
                    **summary,
                )
            )

        lines.extend(["", "## Failure Counts", ""])
        if self.failure_counts:
            for policy, counts in self.failure_counts.items():
                lines.append(f"- `{policy}`: {_counts_text(counts)}")
        else:
            lines.append("- No failures recorded.")

        lines.extend(["", "## Outcome Counts", ""])
        if self.outcome_counts:
            for policy, counts in self.outcome_counts.items():
                lines.append(f"- `{policy}`: {_counts_text(counts)}")
        else:
            lines.append("- No outcomes recorded.")

        lines.extend(["", "## Metric Summary", ""])
        if self.metric_summary:
            for policy, metrics in self.metric_summary.items():
                lines.extend(["", f"### {policy}", ""])
                lines.append("| Metric | Min | Max | Avg | Last |")
                lines.append("| --- | ---: | ---: | ---: | ---: |")
                for metric, summary in metrics.items():
                    lines.append(
                        f"| {metric} | {summary['min']} | {summary['max']} | {summary['avg']} | {summary['last']} |"
                    )
        else:
            lines.append("- No numeric metrics recorded.")

        lines.extend(["", "## Regressions", ""])
        if self.regressions:
            for regression in self.regressions:
                lines.append(
                    f"- `{regression['policy_version']}` regressed on `{regression['scenario_name']}`: {regression['failure_label']}"
                )
        else:
            lines.append("- No regressions detected.")

        lines.extend(["", "## Improvements", ""])
        if self.improvements:
            for improvement in self.improvements:
                lines.append(
                    f"- `{improvement['policy_version']}` improved `{improvement['scenario_name']}`; baseline failed with `{improvement['baseline_failure_label']}`."
                )
        else:
            lines.append("- No improvements detected against baseline.")

        lines.extend(["", "## Failure Cases", ""])
        if self.failure_cases:
            for failure in self.failure_cases:
                lines.append(
                    f"- `{failure['policy_version']}` failed `{failure['scenario_name']}` with `{failure['failure_label']}` after {failure['steps']} steps. First failure step: `{failure['first_failure_step']}`."
                )
        else:
            lines.append("- No failure cases.")

        lines.extend(["", "## Action Divergences", ""])
        if self.action_divergences:
            for divergence in self.action_divergences[:10]:
                lines.append(
                    f"- `{divergence['policy_version']}` diverged from `{divergence['baseline_policy']}` on `{divergence['scenario_name']}` at step {divergence['step']}: baseline `{display_value(divergence['baseline_action'])}`, candidate `{display_value(divergence['candidate_action'])}`."
                )
        else:
            lines.append("- No action divergences detected.")

        lines.extend(["", "## Scenario Groups", ""])
        if self.grouped_metrics:
            for group_name, metrics_by_policy in self.grouped_metrics.items():
                lines.extend(["", f"### {group_name}", ""])
                lines.append("| Policy | Success Rate | Failures | Avg Steps |")
                lines.append("| --- | ---: | ---: | ---: |")
                for policy, metrics in metrics_by_policy.items():
                    lines.append(
                        f"| {policy} | {metrics['success_rate']} | {metrics['failure_count']} | {metrics['average_steps']} |"
                    )
        else:
            lines.append("- No scenario group metadata found.")

        return "\n".join(lines) + "\n"

    def _write_decision_logs(self, output_path: Path) -> None:
        with output_path.open("w", encoding="utf-8") as handle:
            for episode in self.episodes:
                for log in episode.logs:
                    handle.write(json.dumps(log.to_dict()) + "\n")

    @staticmethod
    def _write_json(output_path: Path, data: Any) -> None:
        output_path.write_text(json.dumps(to_serializable(data), indent=2) + "\n", encoding="utf-8")


def to_serializable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(to_serializable(key)): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, set):
        return [to_serializable(item) for item in sorted(value, key=repr)]
    if is_dataclass(value):
        return to_serializable(asdict(value))
    item = _call_no_arg(value, "item")
    if item is not _SERIALIZE_SENTINEL:
        return to_serializable(item)
    listed = _call_no_arg(value, "tolist")
    if listed is not _SERIALIZE_SENTINEL:
        return to_serializable(listed)
    return repr(value)


def action_key(action: Action) -> str:
    return json.dumps(to_serializable(action), sort_keys=True, separators=(",", ":"))


def display_value(value: Any) -> str:
    serialized = to_serializable(value)
    if isinstance(serialized, str):
        return serialized
    return json.dumps(serialized, sort_keys=True)


_SERIALIZE_SENTINEL = object()


def _call_no_arg(value: Any, method_name: str) -> Any:
    method = getattr(value, method_name, None)
    if not callable(method):
        return _SERIALIZE_SENTINEL
    try:
        return method()
    except Exception:
        return _SERIALIZE_SENTINEL


def _markdown_value(value: Any) -> str:
    value = to_serializable(value)
    if isinstance(value, list):
        return ", ".join(f"`{item}`" for item in value)
    if isinstance(value, dict):
        return f"`{json.dumps(value, sort_keys=True)}`"
    if isinstance(value, str):
        return f"`{value}`"
    return f"`{value}`"


def _rule_from_mapping(data: dict[str, Any]) -> Rule:
    rule_type = data["type"]
    name = data.get("name")
    if rule_type == "require_outcome":
        return require_outcome(str(data["outcome"]), name=name)
    if rule_type == "forbid_failure":
        return forbid_failure(str(data["failure_label"]), name=name)
    if rule_type == "max_steps":
        return max_steps(int(data["max_steps"]), name=name)
    if rule_type == "require_metric":
        return require_metric(
            metric_name=str(data["metric"]),
            operator=str(data["operator"]),
            value=float(data["value"]),
            name=name,
            aggregate=str(data.get("aggregate", "last")),
        )
    raise ValueError(f"Unsupported ruleset rule type: {rule_type!r}")


def _unsafe_forward_rule(action: Action, min_distance: float) -> Rule:
    expected_action_key = action_key(action)

    def _rule(context: EpisodeContext) -> RuleResult:
        failure = next(
            (
                log
                for log in context.logs
                if action_key(log.action) == expected_action_key
                and float(log.state.get("front_distance", 9999)) < min_distance
            ),
            None,
        )
        return RuleResult(
            name="unsafe_forward_action",
            passed=failure is None,
            reason=(
                ""
                if failure is None
                else f"{display_value(failure.action)} with front_distance={failure.state.get('front_distance')} below {min_distance}"
            ),
            step=failure.step if failure else None,
            details={"action": to_serializable(action), "min_distance": min_distance},
        )

    return _rule


def _terminal_outcome_rule(outcome: str, name: str) -> Rule:
    def _rule(context: EpisodeContext) -> RuleResult:
        passed = context.terminal_outcome == outcome
        return RuleResult(
            name=name,
            passed=passed,
            reason="" if passed else f"terminal outcome was {context.terminal_outcome!r}, expected {outcome!r}",
            step=None if passed or not context.logs else context.logs[-1].step,
            details={"terminal_outcome": context.terminal_outcome, "goal_outcome": outcome},
        )

    return _rule


def _legacy_rule(rule: EvalRule) -> Rule:
    def _rule(context: EpisodeContext) -> RuleResult:
        return rule(context.logs, context.terminal_outcome)

    return _rule


def _aggregate_metric(values: list[float], aggregate: str) -> float:
    if aggregate == "last":
        return values[-1]
    if aggregate == "min":
        return min(values)
    if aggregate == "max":
        return max(values)
    if aggregate == "mean":
        return sum(values) / len(values)
    raise ValueError(f"Unsupported metric aggregate: {aggregate!r}")


def _compare_metric(observed: float, operator: str, expected: float) -> bool:
    if operator == "<":
        return observed < expected
    if operator == "<=":
        return observed <= expected
    if operator == ">":
        return observed > expected
    if operator == ">=":
        return observed >= expected
    if operator == "==":
        return observed == expected
    if operator == "!=":
        return observed != expected
    raise ValueError(f"Unsupported metric operator: {operator!r}")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _counts_text(counts: dict[str, int]) -> str:
    return ", ".join(f"`{name}`={count}" for name, count in sorted(counts.items())) or "none"
