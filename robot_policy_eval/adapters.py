from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

from .core import Decision, State


@dataclass(frozen=True)
class PolicyAdapter:
    name: str
    policy: Any

    def decide(self, state: State) -> Decision:
        if hasattr(self.policy, "decide"):
            raw = self.policy.decide(state)
        elif callable(self.policy):
            raw = self.policy(state)
        else:
            raise TypeError(f"Policy {self.name!r} is not callable and has no decide() method.")
        return _normalize_decision(raw)


def normalize_policy(policy: Any, name: str | None = None) -> PolicyAdapter:
    if isinstance(policy, PolicyAdapter):
        return policy
    policy_name = name or getattr(policy, "version", None) or getattr(policy, "__name__", None)
    if not policy_name:
        policy_name = policy.__class__.__name__
    return PolicyAdapter(name=str(policy_name), policy=policy)


def load_object(import_path: str) -> Any:
    if ":" not in import_path:
        raise ValueError(f"Import path must look like 'module.path:object_name', got {import_path!r}.")
    module_name, object_name = import_path.split(":", 1)
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in object_name.split("."):
        obj = getattr(obj, part)
    return obj


def load_policy(import_path: str, name: str | None = None) -> PolicyAdapter:
    return normalize_policy(load_object(import_path), name=name)


def _normalize_decision(raw: Any) -> Decision:
    if isinstance(raw, Decision):
        return raw
    if isinstance(raw, str):
        return Decision(action=raw)
    if isinstance(raw, tuple) and len(raw) == 2:
        action, debug_info = raw
        return Decision(action=str(action), debug_info=dict(debug_info or {}))
    if isinstance(raw, dict):
        if "action" not in raw:
            raise ValueError("Policy decision dict must include an 'action' key.")
        debug_info = dict(raw.get("debug_info", {}))
        if "scores" in raw:
            debug_info["scores"] = raw["scores"]
        return Decision(action=str(raw["action"]), debug_info=debug_info)
    raise TypeError(f"Unsupported policy decision shape: {type(raw).__name__}")


PolicyLike = Callable[[State], Any] | PolicyAdapter
