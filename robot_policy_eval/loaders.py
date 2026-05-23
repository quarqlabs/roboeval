from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters import PolicyAdapter, load_object, load_policy
from .core import Scenario, SuccessCriteria


@dataclass(frozen=True)
class EvalConfig:
    policies: list[PolicyAdapter]
    scenarios: list[Scenario]
    success_criteria: SuccessCriteria
    baseline_policy: str


def load_eval_config(config_path: str | Path) -> EvalConfig:
    path = Path(config_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    base_dir = path.parent

    policies = [
        load_policy(policy_data["path"], name=policy_data.get("name"))
        for policy_data in data["policies"]
    ]

    scenarios: list[Scenario] = []
    for source in data["scenario_sources"]:
        source_type = source["type"]
        if source_type == "python":
            scenarios.extend(load_scenarios_python(source["path"]))
        elif source_type == "json":
            scenarios.extend(load_scenarios_json(_resolve(base_dir, source["path"])))
        elif source_type == "csv":
            scenarios.extend(load_scenarios_csv(_resolve(base_dir, source["path"])))
        else:
            raise ValueError(f"Unsupported scenario source type: {source_type}")

    criteria_path = data.get("success_criteria")
    if criteria_path:
        success_criteria = load_success_criteria_json(_resolve(base_dir, criteria_path))
    else:
        success_criteria = SuccessCriteria()

    return EvalConfig(
        policies=policies,
        scenarios=scenarios,
        success_criteria=success_criteria,
        baseline_policy=data.get("baseline_policy", policies[0].name),
    )


def load_scenarios_python(import_path: str) -> list[Scenario]:
    raw = load_object(import_path)
    if callable(raw):
        raw = raw()
    return [_normalize_scenario(item) for item in raw]


def load_scenarios_json(path: str | Path) -> list[Scenario]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [_normalize_scenario(item) for item in data["scenarios"]]


def load_scenarios_csv(path: str | Path) -> list[Scenario]:
    scenarios = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row.pop("name")
            max_steps = int(row.pop("max_steps", "10"))
            required_forward_steps = int(row.pop("required_forward_steps", "2"))
            initial_state = {key: _parse_value(value) for key, value in row.items() if value != ""}
            scenarios.append(
                Scenario(
                    name=name,
                    initial_state=initial_state,
                    max_steps=max_steps,
                    metadata={"required_forward_steps": required_forward_steps},
                )
            )
    return scenarios


def load_success_criteria_json(path: str | Path) -> SuccessCriteria:
    return SuccessCriteria.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


def _normalize_scenario(raw: Any) -> Scenario:
    if isinstance(raw, Scenario):
        return raw
    if isinstance(raw, dict):
        return Scenario.from_mapping(raw)
    raise TypeError(f"Unsupported scenario shape: {type(raw).__name__}")


def _resolve(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else base_dir / path


def _parse_value(value: str) -> Any:
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
