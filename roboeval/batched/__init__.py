"""Vectorized-environment namespace for roboeval.

Single-env users should keep importing from ``roboeval`` directly. This
subpackage adds parallel types/Protocols for N-env vectorized rollouts
(Isaac Lab, gym.vector, Brax, MJX), strictly isolated from the single-env
public API so existing callers see no behavior change.
"""

from .environment import BatchedEnvironmentAdapter
from .policy import (
    BatchedPolicy,
    BatchedPolicyAdapter,
    from_single,
    normalize_batched_policy,
)
from .types import BatchedState, BatchedStepOutcome


__all__ = [
    "BatchedEnvironmentAdapter",
    "BatchedPolicy",
    "BatchedPolicyAdapter",
    "BatchedState",
    "BatchedStepOutcome",
    "from_single",
    "normalize_batched_policy",
]
