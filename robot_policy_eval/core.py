from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


State = dict[str, Any]


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
        return cls(
            name=str(data["name"]),
            initial_state=dict(data["initial_state"]),
            max_steps=int(data.get("max_steps", 10)),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SuccessCriteria:
    must_reach_goal: bool = True
    collision_is_failure: bool = True
    stuck_is_failure: bool = True
    unsafe_forward_min_distance: float | None = 20.0
    unsafe_forward_action: str = "move_forward"
    goal_outcome: str = "goal_reached"

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
        for log in logs:
            if (
                self.unsafe_forward_min_distance is not None
                and log.action == self.unsafe_forward_action
                and float(log.state.get("front_distance", 9999)) < self.unsafe_forward_min_distance
            ):
                return False, "unsafe_forward_action"
            if self.collision_is_failure and log.failure_label == "collision":
                return False, "collision"
            if self.stuck_is_failure and log.failure_label == "stuck":
                return False, "stuck"

        if self.must_reach_goal and terminal_outcome != self.goal_outcome:
            return False, "goal_not_reached"
        return True, ""


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
        return data


@dataclass(frozen=True)
class EvalReport:
    baseline_policy: str
    policy_summary: dict[str, dict[str, float | int]]
    episodes: list[EpisodeResult]
    regressions: list[dict[str, Any]]
    improvements: list[dict[str, Any]]
    failure_cases: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_policy": self.baseline_policy,
            "policy_summary": self.policy_summary,
            "episodes": [episode.to_dict() for episode in self.episodes],
            "regressions": self.regressions,
            "improvements": self.improvements,
            "failure_cases": self.failure_cases,
        }

    def save(self, output_dir: str | Path) -> None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self._write_decision_logs(destination / "decision_logs.jsonl")
        self._write_json(destination / "episode_results.json", [episode.to_dict() for episode in self.episodes])
        self._write_json(
            destination / "comparison_report.json",
            {
                "baseline_policy": self.baseline_policy,
                "policy_summary": self.policy_summary,
                "regressions": self.regressions,
                "improvements": self.improvements,
                "failure_cases": self.failure_cases,
            },
        )
        (destination / "report.md").write_text(self.to_markdown(), encoding="utf-8")

    def to_markdown(self) -> str:
        lines = [
            "# Robot Policy Eval Report",
            "",
            f"Baseline policy: `{self.baseline_policy}`",
            "",
            "## Policy Summary",
            "",
            "| Policy | Success Rate | Failures | Collisions | Stuck | Unsafe Actions | Avg Steps |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
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

        lines.extend(["", "## Failure Cases", ""])
        if self.failure_cases:
            for failure in self.failure_cases:
                lines.append(
                    f"- `{failure['policy_version']}` failed `{failure['scenario_name']}` with `{failure['failure_label']}` after {failure['steps']} steps."
                )
        else:
            lines.append("- No failure cases.")

        return "\n".join(lines) + "\n"

    def _write_decision_logs(self, output_path: Path) -> None:
        with output_path.open("w", encoding="utf-8") as handle:
            for episode in self.episodes:
                for log in episode.logs:
                    handle.write(json.dumps(log.to_dict()) + "\n")

    @staticmethod
    def _write_json(output_path: Path, data: Any) -> None:
        output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
