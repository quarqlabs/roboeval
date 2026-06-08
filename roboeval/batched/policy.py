"""BatchedPolicy Protocol + adapter + single-state shim.

A batched policy receives a list of per-slot states and returns one Decision
per slot. This is the right interface for GPU-batched inference (VLAs,
diffusion policies, transformer-based controllers) where calling the model
N times sequentially defeats the point of vectorization.

For legacy single-state policies (plain functions taking one State and
returning a dict/tuple/Decision), use ``from_single(fn)`` to get a
BatchedPolicy that iterates internally. It is a strict throughput regression
relative to a natively-batched policy, but it preserves backward compat for
the existing single-env policy zoo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from roboeval.adapters import _normalize_decision
from roboeval.core import Decision, State

from .types import BatchedState


class BatchedPolicy(Protocol):
    """Protocol every batched policy should implement."""

    name: str

    def decide(self, states: BatchedState) -> list[Decision]:
        """Return one Decision per slot. ``len(states) == len(returned)``."""
        ...


@dataclass(frozen=True)
class BatchedPolicyAdapter:
    """Normalize a user-provided batched policy to the BatchedPolicy Protocol.

    Accepts either:
      * an object with ``.decide(states)`` returning list[Decision | dict |
        tuple | raw_action]
      * a callable ``fn(states) -> list[...]`` (same item shapes as above)

    Each per-slot return value is normalized through ``_normalize_decision``
    (the same helper used in single-env adapters), so dict/tuple/raw returns
    are accepted uniformly.
    """

    name: str
    policy: Any

    def decide(self, states: BatchedState) -> list[Decision]:
        if hasattr(self.policy, "decide"):
            raw = self.policy.decide(states)
        elif callable(self.policy):
            raw = self.policy(states)
        else:
            raise TypeError(
                f"Batched policy {self.name!r} is not callable and has no decide() method."
            )
        if not isinstance(raw, list):
            raise TypeError(
                f"Batched policy {self.name!r} must return a list, got {type(raw).__name__}."
            )
        if len(raw) != len(states):
            raise ValueError(
                f"Batched policy {self.name!r} returned {len(raw)} decisions "
                f"for {len(states)} input states."
            )
        return [_normalize_decision(item) for item in raw]


def normalize_batched_policy(
    policy: Any, name: str | None = None
) -> BatchedPolicyAdapter:
    if isinstance(policy, BatchedPolicyAdapter):
        return policy
    policy_name = name or getattr(policy, "version", None) or getattr(policy, "__name__", None)
    if not policy_name:
        policy_name = policy.__class__.__name__
    return BatchedPolicyAdapter(name=str(policy_name), policy=policy)


def from_single(
    policy: Callable[[State], Any], name: str | None = None
) -> BatchedPolicyAdapter:
    """Wrap a single-state policy as a BatchedPolicy by looping over slots.

    Throughput note: each call sequentially evaluates the inner policy N
    times. Use for backward compat with single-env policy zoos; for
    production GPU-batched models, write a natively-batched policy instead.
    """
    inner_name = name or getattr(policy, "version", None) or getattr(policy, "__name__", None)
    if not inner_name:
        inner_name = policy.__class__.__name__

    def _batched(states: BatchedState) -> list[Any]:
        return [policy(state) for state in states]

    _batched.__name__ = f"batched_{inner_name}"
    return BatchedPolicyAdapter(name=str(inner_name), policy=_batched)
