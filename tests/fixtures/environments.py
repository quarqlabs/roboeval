from __future__ import annotations

from typing import Any

from roboeval import Scenario, StepOutcome


class CliCustomEnvironment:
    name = "cli_custom_environment"

    def __init__(self, outcome: str = "custom_goal") -> None:
        self.outcome = outcome

    def reset(self, scenario: Scenario) -> dict[str, Any]:
        return dict(scenario.initial_state)

    def step(self, action: Any, scenario: Scenario) -> StepOutcome:
        return StepOutcome(
            next_state={**scenario.initial_state, "received_action": action},
            outcome=self.outcome,
            failure_label="",
            terminal=True,
            info={"received_action_type": type(action).__name__},
        )


def make_cli_environment(outcome: str = "factory_goal") -> CliCustomEnvironment:
    return CliCustomEnvironment(outcome=outcome)


INSTANCE_ENVIRONMENT = CliCustomEnvironment(outcome="instance_goal")
