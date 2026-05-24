from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


State = dict[str, Any]
SDK_NAME = "robot-policy-eval"
SDK_VERSION = "0.1.0"


@dataclass(frozen=True)
class Decision:
    action: str
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
        return asdict(self)


@dataclass(frozen=True)
class RuleResult:
    name: str
    passed: bool
    reason: str = ""
    step: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


EvalRule = Callable[[list["StepRecord"], str], RuleResult]


@dataclass(frozen=True)
class SuccessCriteria:
    must_reach_goal: bool = True
    collision_is_failure: bool = True
    stuck_is_failure: bool = True
    unsafe_forward_min_distance: float | None = 20.0
    unsafe_forward_action: str = "move_forward"
    goal_outcome: str = "goal_reached"
    custom_rules: list[EvalRule] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "SuccessCriteria":
        return cls(
            must_reach_goal=bool(data.get("must_reach_goal", True)),
            collision_is_failure=bool(data.get("collision_is_failure", True)),
            stuck_is_failure=bool(data.get("stuck_is_failure", True)),
            unsafe_forward_min_distance=data.get("unsafe_forward_min_distance", 20.0),
            unsafe_forward_action=str(data.get("unsafe_forward_action", "move_forward")),
            goal_outcome=str(data.get("goal_outcome", "goal_reached")),
        )

    def evaluate(self, logs: list["StepRecord"], terminal_outcome: str) -> tuple[bool, str]:
        rule_results = self.evaluate_rules(logs, terminal_outcome)
        first_failure = next((result for result in rule_results if not result.passed), None)
        return first_failure is None, first_failure.name if first_failure else ""

    def evaluate_rules(self, logs: list["StepRecord"], terminal_outcome: str) -> list[RuleResult]:
        rule_results: list[RuleResult] = []

        if self.unsafe_forward_min_distance is not None:
            failure = next(
                (
                    log
                    for log in logs
                    if log.action == self.unsafe_forward_action
                    and float(log.state.get("front_distance", 9999)) < self.unsafe_forward_min_distance
                ),
                None,
            )
            rule_results.append(
                RuleResult(
                    name="unsafe_forward_action",
                    passed=failure is None,
                    reason=(
                        ""
                        if failure is None
                        else f"{failure.action} with front_distance={failure.state.get('front_distance')} below {self.unsafe_forward_min_distance}"
                    ),
                    step=failure.step if failure else None,
                    details={"action": self.unsafe_forward_action, "min_distance": self.unsafe_forward_min_distance},
                )
            )

        if self.collision_is_failure:
            failure = next((log for log in logs if log.failure_label == "collision"), None)
            rule_results.append(
                RuleResult(
                    name="collision",
                    passed=failure is None,
                    reason="" if failure is None else f"collision at step {failure.step}",
                    step=failure.step if failure else None,
                )
            )

        if self.stuck_is_failure:
            failure = next((log for log in logs if log.failure_label == "stuck"), None)
            rule_results.append(
                RuleResult(
                    name="stuck",
                    passed=failure is None,
                    reason="" if failure is None else f"stuck at step {failure.step}",
                    step=failure.step if failure else None,
                )
            )

        if self.must_reach_goal:
            passed = terminal_outcome == self.goal_outcome
            rule_results.append(
                RuleResult(
                    name="goal_reached",
                    passed=passed,
                    reason="" if passed else f"terminal outcome was {terminal_outcome!r}, expected {self.goal_outcome!r}",
                    step=None if passed or not logs else logs[-1].step,
                    details={"terminal_outcome": terminal_outcome, "goal_outcome": self.goal_outcome},
                )
            )

        for custom_rule in self.custom_rules:
            rule_results.append(custom_rule(logs, terminal_outcome))

        return rule_results


@dataclass(frozen=True)
class StepRecord:
    episode_id: str
    scenario_name: str
    policy_version: str
    step: int
    state: State
    action: str
    outcome: str
    failure_label: str
    next_state: State = field(default_factory=dict)
    is_terminal: bool = False
    debug_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["logs"] = [log.to_dict() for log in self.logs]
        data["rule_results"] = [result.to_dict() for result in self.rule_results]
        return data


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
    metadata: dict[str, Any] = field(default_factory=dict)
    highlights: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "baseline_policy": self.baseline_policy,
            "policy_summary": self.policy_summary,
            "episodes": [episode.to_dict() for episode in self.episodes],
            "regressions": self.regressions,
            "improvements": self.improvements,
            "failure_cases": self.failure_cases,
            "grouped_metrics": self.grouped_metrics,
            "action_divergences": self.action_divergences,
            "highlights": self.highlights,
        }

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
                "highlights": self.highlights,
            },
        )
        (destination / "report.md").write_text(self.to_markdown(), encoding="utf-8")

    def to_markdown(self) -> str:
        lines = [
            "# Robot Policy Eval Report",
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
            "| Policy | Success Rate | Failures | Collisions | Stuck | Unsafe Actions | Avg Steps |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for policy, summary in self.policy_summary.items():
            lines.append(
                "| {policy} | {success_rate} | {failure_count} | {collision_count} | {stuck_count} | {unsafe_action_count} | {average_steps} |".format(
                    policy=policy,
                    **summary,
                )
            )

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
                    f"- `{divergence['policy_version']}` diverged from `{divergence['baseline_policy']}` on `{divergence['scenario_name']}` at step {divergence['step']}: baseline `{divergence['baseline_action']}`, candidate `{divergence['candidate_action']}`."
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
        output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _markdown_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(f"`{item}`" for item in value)
    if isinstance(value, str):
        return f"`{value}`"
    return f"`{value}`"
