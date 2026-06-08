"""Batched Isaac Lab ↔ roboeval adapter.

Wraps a vectorized Isaac Lab env (``ManagerBasedRLEnv`` or the gymnasium-style
wrapper around it) as a ``BatchedEnvironmentAdapter`` so ``BatchedEvalRunner``
can read all N parallel slots — the way Isaac actually wants to be used.

Difference from ``IsaacEnvironmentAdapter`` (single-env)
--------------------------------------------------------
The single-env adapter takes ``batch_index=0`` and slices, discarding the
other 99% of the GPU's work. This batched adapter reads every slot:

  * ``reset(scenarios)``         — bulk reset all num_envs slots
  * ``step(actions)``             — stack per-slot actions into a (N, *) tensor,
                                    step the env, fan the (N,) reward/terminal
                                    tensors back into per-slot lists
  * ``reset_slots(slots, scs)``   — selective reset using ``env_ids`` if the
                                    underlying env supports it, falling back to
                                    a full reset with a warning

The translation hooks (``observation_to_state``, ``outcome_from_step``, etc.)
match the single-env Isaac adapter so user customizations port over unchanged.

Per-slot seeding limitation
---------------------------
Isaac Lab's reset accepts a single ``seed`` — there's no public per-env seed
API. ``reset(scenarios)`` uses the first scenario's seed as the global seed
and proceeds. For deterministic per-replica variance, run the same scenario
with different reset seeds across separate ``run()`` calls or use Isaac's
domain randomization config.

Selective reset
---------------
Isaac Lab's ``ManagerBasedRLEnv`` exposes ``_reset_idx(env_ids)`` for in-place
per-env reset. The gym wrapper sometimes does too. We try, in order:

  1. ``env.reset(env_ids=...)``           (gym wrapper, if implemented)
  2. ``env.unwrapped._reset_idx(env_ids)`` (ManagerBasedRLEnv direct)
  3. Full ``env.reset()``                  (fallback — warns once)

In all cases we re-read the obs to capture the post-reset state for the
just-reset slots.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable

from roboeval.batched.types import BatchedStepOutcome
from roboeval.core import Action, Scenario, State, to_serializable

from .adapter import (
    default_action_from_decision,
    default_events_from_step,
    default_observation_to_state,
    default_options_from_scenario,
    default_outcome_from_step,
    default_seed_from_scenario,
    tensor_to_numpy,
)


@dataclass
class BatchedIsaacEnvironmentAdapter:
    """Wraps a vectorized Isaac Lab env as a ``BatchedEnvironmentAdapter``.

    Parameters
    ----------
    env :
        The Isaac Lab vectorized env. Must expose ``num_envs``.
    name :
        Display name for reports.
    observation_to_state, action_from_decision, outcome_from_step,
    events_from_step, seed_from_scenario, options_from_scenario :
        Override hooks. ``None`` selects the single-env Isaac adapter's
        defaults.
    info_keys :
        Optional allowlist for keys passed through to
        ``info["isaac"]["raw_info"]`` per slot.
    """

    env: Any
    name: str = "batched_isaac_env"

    observation_to_state: Callable[[Any], dict[str, Any]] | None = None
    action_from_decision: Callable[[Action], Any] | None = None
    outcome_from_step: (
        Callable[[float, bool, bool, dict], tuple[str, str]] | None
    ) = None
    events_from_step: (
        Callable[[float, bool, bool, dict], list[str]] | None
    ) = None
    seed_from_scenario: Callable[[Scenario], int | None] | None = None
    options_from_scenario: Callable[[Scenario], dict | None] | None = None

    info_keys: list[str] | None = None

    def __post_init__(self) -> None:
        # Gymnasium wrappers (OrderEnforcing, TimeLimit, etc.) wrap Isaac Lab
        # envs after gym.make(...) and don't proxy num_envs. Fall back to
        # env.unwrapped — discovered while validating against real Isaac.
        num_envs = getattr(self.env, "num_envs", None)
        if num_envs is None:
            unwrapped = getattr(self.env, "unwrapped", None)
            num_envs = getattr(unwrapped, "num_envs", None) if unwrapped is not None else None
        if num_envs is None:
            raise TypeError(
                "BatchedIsaacEnvironmentAdapter requires the wrapped env to "
                "expose .num_envs (checked self.env and self.env.unwrapped). "
                "Got an env with no num_envs attribute."
            )
        if num_envs <= 0:
            raise ValueError(f"env.num_envs must be positive, got {num_envs}.")
        self.num_envs = int(num_envs)

        if self.observation_to_state is None:
            self.observation_to_state = default_observation_to_state
        if self.action_from_decision is None:
            self.action_from_decision = default_action_from_decision
        if self.outcome_from_step is None:
            self.outcome_from_step = default_outcome_from_step
        if self.events_from_step is None:
            self.events_from_step = default_events_from_step
        if self.seed_from_scenario is None:
            self.seed_from_scenario = default_seed_from_scenario
        if self.options_from_scenario is None:
            self.options_from_scenario = default_options_from_scenario

        self._episode_returns: list[float] = [0.0] * self.num_envs
        self._device: str | None = None
        self._warned_full_reset = False

    # ── BatchedEnvironmentAdapter Protocol ──────────────────────────────────

    def reset(self, scenarios: list[Scenario]) -> list[State]:
        if len(scenarios) != self.num_envs:
            raise ValueError(
                f"reset() requires {self.num_envs} scenarios (num_envs), "
                f"got {len(scenarios)}."
            )
        seeds = [self.seed_from_scenario(sc) for sc in scenarios]
        seed = next((s for s in seeds if s is not None), None)
        options = self.options_from_scenario(scenarios[0])

        try:
            obs, _info = self.env.reset(seed=seed, options=options)
        except TypeError:
            obs, _info = self.env.reset(seed=seed)
        self._episode_returns = [0.0] * self.num_envs
        return [self._state_from_obs_slot(obs, i) for i in range(self.num_envs)]

    def step(self, actions: list[Action]) -> BatchedStepOutcome:
        if len(actions) != self.num_envs:
            raise ValueError(
                f"step() requires {self.num_envs} actions (num_envs), "
                f"got {len(actions)}."
            )
        decision_actions = [self.action_from_decision(a) for a in actions]
        batched_action = self._stack_actions(decision_actions)
        obs, reward, terminated, truncated, info = self.env.step(batched_action)

        rewards_np = tensor_to_numpy(reward)
        terminateds_np = tensor_to_numpy(terminated)
        truncateds_np = tensor_to_numpy(truncated)

        next_states: list[State] = []
        outcomes: list[str] = []
        failure_labels: list[str] = []
        terminals: list[bool] = []
        metrics: list[dict[str, float | int] | None] = []
        events: list[list[str] | None] = []
        info_list: list[dict[str, object] | None] = []

        for i in range(self.num_envs):
            reward_i = float(_scalar_at(rewards_np, i, default=0.0))
            term_i = bool(_scalar_at(terminateds_np, i, default=False))
            trunc_i = bool(_scalar_at(truncateds_np, i, default=False))
            slot_info = _per_slot_isaac_info(info, i, self.num_envs, self.info_keys)
            self._episode_returns[i] += reward_i

            next_states.append(self._state_from_obs_slot(obs, i))
            outcome, failure_label = self.outcome_from_step(reward_i, term_i, trunc_i, slot_info)
            outcomes.append(outcome)
            failure_labels.append(failure_label)
            terminals.append(term_i or trunc_i)
            metrics.append({
                "reward": reward_i,
                "episode_return": float(self._episode_returns[i]),
            })
            events.append(self.events_from_step(reward_i, term_i, trunc_i, slot_info))
            info_list.append({
                "isaac": {
                    "terminated": term_i,
                    "truncated": trunc_i,
                    "raw_info": to_serializable(slot_info),
                    "slot": i,
                }
            })

            if term_i or trunc_i:
                self._episode_returns[i] = 0.0

        return BatchedStepOutcome(
            next_states=next_states,
            outcomes=outcomes,
            failure_labels=failure_labels,
            terminals=terminals,
            metrics=metrics,
            events=events,
            info=info_list,
        )

    def reset_slots(
        self, slots: list[int], scenarios: list[Scenario]
    ) -> list[State]:
        if len(slots) != len(scenarios):
            raise ValueError(
                f"reset_slots: slots and scenarios must be same length, "
                f"got {len(slots)} and {len(scenarios)}."
            )
        if not slots:
            return []

        seed = next(
            (self.seed_from_scenario(sc) for sc in scenarios
             if self.seed_from_scenario(sc) is not None),
            None,
        )

        env_ids = self._env_ids_from_slots(slots)
        post_obs = self._try_selective_reset(env_ids, seed)
        if post_obs is None:
            post_obs = self._fallback_full_reset(seed)

        new_states: list[State] = []
        for slot in slots:
            self._episode_returns[slot] = 0.0
            new_states.append(self._state_from_obs_slot(post_obs, slot))
        return new_states

    def close(self) -> None:
        """Forward close to the wrapped env."""
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    # ── Internals ──────────────────────────────────────────────────────────

    def _state_from_obs_slot(self, obs: Any, slot: int) -> State:
        sliced = self._slice_obs_at_slot(obs, slot)
        return self.observation_to_state(sliced)

    def _slice_obs_at_slot(self, obs: Any, slot: int) -> Any:
        """Index batched obs at ``slot`` and coerce tensors to numpy.

        Handles two common Isaac obs shapes:
          * dict like ``{"policy": tensor(N, *)}``  → slice each value at slot
          * tensor ``(N, *)`` directly              → slice at slot
        """
        if isinstance(obs, dict):
            return {k: tensor_to_numpy(_index_safe(v, slot)) for k, v in obs.items()}
        return tensor_to_numpy(_index_safe(obs, slot))

    def _stack_actions(self, actions: list[Any]) -> Any:
        """Stack per-slot actions into a single (N, *action_dim) tensor.

        Accepts ints, floats, lists, numpy arrays, or torch tensors per slot.
        Output device matches the env's device when available.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for the Isaac adapter. Install via "
                "`pip install torch` or as part of your Isaac Lab install."
            ) from exc

        device = self._resolve_device(actions)
        as_tensors: list[Any] = []
        for a in actions:
            if isinstance(a, torch.Tensor):
                t = a
            else:
                t = torch.as_tensor(a)
            if t.device != torch.device(device):
                t = t.to(device)
            # Ensure each slot's action is 1-D (vector); a scalar becomes (1,)
            if t.ndim == 0:
                t = t.view(1)
            as_tensors.append(t)
        stacked = torch.stack(as_tensors, dim=0)
        return stacked

    def _resolve_device(self, sample_actions: list[Any]) -> str:
        if self._device is not None:
            return self._device

        # Same wrapper-proxy issue as num_envs — check env.unwrapped too.
        unwrapped = getattr(self.env, "unwrapped", self.env)
        env_device = (
            getattr(self.env, "device", None)
            or getattr(self.env, "sim_device", None)
            or getattr(unwrapped, "device", None)
            or getattr(unwrapped, "sim_device", None)
        )
        if env_device is not None:
            self._device = str(env_device)
            return self._device

        try:
            import torch
        except ImportError:
            self._device = "cpu"
            return self._device

        for a in sample_actions:
            if hasattr(a, "device"):
                self._device = str(a.device)
                return self._device

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def _env_ids_from_slots(self, slots: list[int]) -> Any:
        """Return slots as a torch tensor on the env's device when possible."""
        try:
            import torch
            device = self._resolve_device([])
            return torch.tensor(slots, dtype=torch.long, device=device)
        except ImportError:
            return slots

    def _try_selective_reset(self, env_ids: Any, seed: int | None) -> Any:
        """Attempt selective reset via two known Isaac code paths.

        Returns the post-reset obs (batched, with all envs included) if a
        selective reset succeeded, or None if neither path was available.
        """
        # Path 1: gym wrapper with env_ids kwarg
        try:
            obs, _info = self.env.reset(seed=seed, env_ids=env_ids)
            return obs
        except TypeError:
            pass

        # Path 2: ManagerBasedRLEnv._reset_idx
        unwrapped = getattr(self.env, "unwrapped", self.env)
        reset_idx = getattr(unwrapped, "_reset_idx", None)
        if callable(reset_idx):
            try:
                reset_idx(env_ids)
                # Read post-reset obs without stepping
                obs = self._observe_unwrapped(unwrapped)
                if obs is not None:
                    return obs
            except Exception:  # noqa: BLE001 — any error → fall through
                pass

        return None

    def _observe_unwrapped(self, unwrapped: Any) -> Any:
        """Pull current obs without stepping. Tries common Isaac entry points."""
        get_obs = getattr(unwrapped, "_get_observations", None)
        if callable(get_obs):
            try:
                return get_obs()
            except Exception:  # noqa: BLE001
                return None
        obs_buf = getattr(unwrapped, "obs_buf", None)
        if obs_buf is not None:
            return obs_buf
        return None

    def _fallback_full_reset(self, seed: int | None) -> Any:
        if not self._warned_full_reset:
            warnings.warn(
                "BatchedIsaacEnvironmentAdapter could not perform a selective "
                "reset (no env_ids kwarg and no _reset_idx). Falling back to "
                "full env.reset() — this resets ALL slots including ones still "
                "rolling. For deterministic per-slot refills, wire the env to "
                "support env_ids or expose _reset_idx.",
                stacklevel=3,
            )
            self._warned_full_reset = True
        try:
            obs, _info = self.env.reset(seed=seed)
        except TypeError:
            obs, _info = self.env.reset()
        return obs


# ── module helpers ──────────────────────────────────────────────────────────


def _index_safe(value: Any, slot: int) -> Any:
    try:
        return value[slot]
    except (TypeError, IndexError, KeyError):
        return value


def _scalar_at(value: Any, slot: int, default: Any) -> Any:
    """Read a scalar from a 1-D numpy/tensor at ``slot``, falling back gracefully."""
    sliced = _index_safe(value, slot)
    if hasattr(sliced, "item"):
        try:
            return sliced.item()
        except (ValueError, RuntimeError):
            pass
    try:
        return float(sliced) if not isinstance(sliced, bool) else bool(sliced)
    except (TypeError, ValueError):
        return default


def _per_slot_isaac_info(
    info: Any,
    slot: int,
    num_envs: int,
    info_keys: list[str] | None,
) -> dict[str, Any]:
    """Pluck slot-specific entries from a batched Isaac info dict.

    Isaac's info is a dict where per-env entries are tensors of shape
    ``(num_envs, *)``; scalar entries apply to the whole batch.
    """
    if not isinstance(info, dict):
        return {}
    slot_info: dict[str, Any] = {}
    keys = info_keys if info_keys is not None else list(info.keys())
    for key in keys:
        if key not in info:
            continue
        value = info[key]
        if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
            try:
                if len(value) == num_envs:
                    slot_info[key] = tensor_to_numpy(value[slot])
                    continue
            except TypeError:
                pass
        slot_info[key] = tensor_to_numpy(value) if hasattr(value, "is_cuda") else value
    return slot_info
