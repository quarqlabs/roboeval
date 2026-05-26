from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters import PolicyAdapter, load_object, load_policy
from .core import Rule, Ruleset, Scenario, SuccessCriteria, custom_rule
from .environment import EnvironmentAdapter


@dataclass(frozen=True)
class EvalConfig:
    policies: list[PolicyAdapter]
    scenarios: list[Scenario]
    success_criteria: SuccessCriteria | None
    ruleset: Ruleset | None
    baseline_policy: str
    environment: EnvironmentAdapter | None = None


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

    ruleset_source = data.get("ruleset")
    criteria_path = data.get("success_criteria")
    if ruleset_source:
        ruleset = load_ruleset(_resolve_config_value(base_dir, ruleset_source))
        success_criteria = None
    elif criteria_path:
        ruleset = None
        success_criteria = load_success_criteria_json(_resolve(base_dir, criteria_path))
    else:
        ruleset = None
        success_criteria = SuccessCriteria()

    return EvalConfig(
        policies=policies,
        scenarios=scenarios,
        success_criteria=success_criteria,
        ruleset=ruleset,
        baseline_policy=data.get("baseline_policy", policies[0].name),
        environment=load_environment(data["environment"]) if "environment" in data else None,
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
            scenario_type = row.pop("scenario_type", "")
            raw_tags = row.pop("tags", "")
            metadata: dict[str, Any] = {"required_forward_steps": required_forward_steps}
            if scenario_type:
                metadata["scenario_type"] = scenario_type
            if raw_tags:
                metadata["tags"] = [tag.strip() for tag in raw_tags.split("|") if tag.strip()]
            initial_state = {key: _parse_value(value) for key, value in row.items() if value != ""}
            scenarios.append(
                Scenario(
                    name=name,
                    initial_state=initial_state,
                    max_steps=max_steps,
                    metadata=metadata,
                )
            )
    return scenarios


def load_success_criteria_json(path: str | Path) -> SuccessCriteria:
    return SuccessCriteria.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


def load_ruleset(source: str | Path | dict[str, Any]) -> Ruleset:
    if isinstance(source, dict):
        return _load_ruleset_mapping(source)
    data = json.loads(Path(source).read_text(encoding="utf-8"))
    return _load_ruleset_mapping(data)


def load_environment(data: dict[str, Any]) -> EnvironmentAdapter:
    env_obj = load_object(data["path"])
    kwargs = dict(data.get("kwargs", {}))
    if isinstance(env_obj, type):
        environment = env_obj(**kwargs)
    elif _looks_like_environment(env_obj):
        if kwargs:
            raise ValueError("Environment kwargs can only be used with a class or factory function.")
        environment = env_obj
    elif callable(env_obj):
        environment = env_obj(**kwargs)
    else:
        raise TypeError(f"Environment {data['path']!r} is not an instance, class, or factory function.")
    if not _looks_like_environment(environment):
        raise TypeError(f"Environment {data['path']!r} must provide reset() and step() methods.")
    return environment


def _load_ruleset_mapping(data: dict[str, Any]) -> Ruleset:
    rules: list[Rule] = []
    for rule_data in data.get("rules", []):
        if rule_data.get("type") == "custom":
            rule = load_object(rule_data["path"])
            if not callable(rule):
                raise TypeError(f"Custom rule {rule_data['path']!r} is not callable.")
            rules.append(custom_rule(rule, name=rule_data.get("name")))
        else:
            rules.extend(Ruleset.from_mapping({"rules": [rule_data]}).rules)
    return Ruleset(rules)


def _normalize_scenario(raw: Any) -> Scenario:
    if isinstance(raw, Scenario):
        return raw
    if isinstance(raw, dict):
        return Scenario.from_mapping(raw)
    raise TypeError(f"Unsupported scenario shape: {type(raw).__name__}")


def _resolve(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else base_dir / path


def _resolve_config_value(base_dir: Path, value: Any) -> Any:
    if isinstance(value, str):
        return _resolve(base_dir, value)
    return value


def _looks_like_environment(value: Any) -> bool:
    return callable(getattr(value, "reset", None)) and callable(getattr(value, "step", None))


def _parse_value(value: str) -> Any:
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
