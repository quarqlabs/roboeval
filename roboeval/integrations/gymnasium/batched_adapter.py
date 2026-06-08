"""Batched Gymnasium ↔ roboeval adapter.

Wraps ``gymnasium.vector.SyncVectorEnv`` (and any subclass exposing
``env.envs``) as a ``BatchedEnvironmentAdapter`` so ``BatchedEvalRunner`` can
drive N parallel Gymnasium environments in one step call.

Translation parity with the single-env adapter
----------------------------------------------
The six per-axis translation hooks are imported directly from the single-env
``adapter`` module so behavior matches byte-for-byte at the slot level:

  * ``default_observation_to_state``
  * ``default_action_from_decision``
  * ``default_outcome_from_step``
  * ``default_events_from_step``
  * ``default_seed_from_scenario``
  * ``default_options_from_scenario``

A user who customized hooks on the single-env adapter drops them in here
unchanged.

Gymnasium 1.x autoreset
-----------------------
Default ``AutoresetMode.NEXT_STEP``: when slot i terminates on step N, the
returned ``obs[i]`` is the *terminal* observation. On step N+1, slot i has
been auto-reset internally to a fresh episode. Our runner intercepts the
terminal between those steps and calls ``reset_slots([i], [new_scenario])``,
which explicitly re-seeds slot i via ``env.envs[i].reset(seed=...)`` — that
overrides the implicit auto-reset with the deterministic, scenario-driven
one. ``SAME_STEP`` mode is not currently supported.

AsyncVectorEnv is refused because ``env.envs`` is not directly addressable
across the subprocess boundary; supporting it requires ``env.call("reset", ...)``
which has different return-value semantics. Future enhancement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from roboeval.batched.types import BatchedStepOutcome
from roboeval.core import Action, Scenario, State, to_serializable

from .adapter import (
    default_action_from_decision,
    default_events_from_step,
    default_observation_to_state,
    default_options_from_scenario,
    default_outcome_from_step,
    default_seed_from_scenario,
)


@dataclass
class BatchedGymnasiumEnvironmentAdapter:
    """Wraps ``gym.vector.SyncVectorEnv`` as a ``BatchedEnvironmentAdapter``.

    The same six hooks as the single-env adapter are exposed; they're applied
    per slot. Per-slot ``episode_return`` is tracked internally and reset on
    each slot's terminal step.

    Parameters
    ----------
    env :
        ``gym.vector.SyncVectorEnv`` (or any class exposing ``env.envs`` as
        the addressable list of per-slot envs).
    name :
        Display name for reports.
    info_keys :
        Optional allowlist for keys passed through to
        ``info["gymnasium"]["raw_info"]`` per slot.
    coerce_observations :
        When ``True``, observations are coerced to JSON-safe Python types
        before being returned (off by default; the runner coerces at write time).
    """

    env: gym.vector.VectorEnv
    name: str = "batched_gymnasium_env"

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
    coerce_observations: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.env, gym.vector.VectorEnv):
            raise TypeError(
                "BatchedGymnasiumEnvironmentAdapter requires gym.vector.VectorEnv; "
                f"got {type(self.env).__name__}."
            )
        if not hasattr(self.env, "envs"):
            raise TypeError(
                "Vector env must expose an addressable .envs list "
                "(SyncVectorEnv supports this; AsyncVectorEnv does not yet)."
            )
        autoreset_mode = self.env.metadata.get("autoreset_mode") if hasattr(self.env, "metadata") else None
        if autoreset_mode is not None and getattr(autoreset_mode, "value", str(autoreset_mode)) not in (
            "NextStep", "next-step", "NEXT_STEP"
        ):
            # We tolerate envs that don't declare autoreset_mode (older or custom
            # vector wrappers). We refuse only when we KNOW it's SAME_STEP.
            if "Same" in str(autoreset_mode):
                raise NotImplementedError(
                    f"BatchedGymnasiumEnvironmentAdapter currently supports "
                    f"NEXT_STEP autoreset only; got {autoreset_mode}."
                )

        self.num_envs = self.env.num_envs

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

    # --- BatchedEnvironmentAdapter Protocol ---------------------------------

    def reset(self, scenarios: list[Scenario]) -> list[State]:
        if len(scenarios) != self.num_envs:
            raise ValueError(
                f"reset() requires {self.num_envs} scenarios (num_envs), "
                f"got {len(scenarios)}."
            )
        # Build a per-slot seed list. Gymnasium VectorEnv.reset() accepts
        # ``seed: int | list[int] | None``. If any slot has no seed we pass
        # None (relinquishing determinism) — otherwise we forward the full list.
        seeds = [self.seed_from_scenario(sc) for sc in scenarios]
        seed_arg: int | list[int] | None
        if all(s is None for s in seeds):
            seed_arg = None
        else:
            seed_arg = [int(s) if s is not None else 0 for s in seeds]

        # gym.vector.reset() does not accept per-slot options; if any scenario
        # specifies options we apply them per-slot by calling env.envs[i].reset.
        per_slot_options = [self.options_from_scenario(sc) for sc in scenarios]

        obs, _info = self.env.reset(seed=seed_arg)
        self._episode_returns = [0.0] * self.num_envs

        states = [self._state_from_obs(obs[i]) for i in range(self.num_envs)]
        # Re-reset any slot that requested non-None options
        for i, opts in enumerate(per_slot_options):
            if opts is not None:
                slot_seed = seeds[i] if seeds[i] is not None else None
                obs_i, _info_i = self.env.envs[i].reset(seed=slot_seed, options=opts)
                states[i] = self._state_from_obs(obs_i)
        return states

    def step(self, actions: list[Action]) -> BatchedStepOutcome:
        if len(actions) != self.num_envs:
            raise ValueError(
                f"step() requires {self.num_envs} actions (num_envs), "
                f"got {len(actions)}."
            )
        gym_actions = [self.action_from_decision(a) for a in actions]
        action_array = self._actions_to_array(gym_actions)

        obs, rewards, terminateds, truncateds, infos = self.env.step(action_array)

        next_states: list[State] = []
        outcomes: list[str] = []
        failure_labels: list[str] = []
        terminals: list[bool] = []
        metrics: list[dict[str, float | int] | None] = []
        events: list[list[str] | None] = []
        info_list: list[dict[str, object] | None] = []

        for i in range(self.num_envs):
            reward_i = float(rewards[i])
            term_i = bool(terminateds[i])
            trunc_i = bool(truncateds[i])
            slot_info = _per_slot_info(infos, i, self.num_envs, self.info_keys)
            self._episode_returns[i] += reward_i

            next_states.append(self._state_from_obs(obs[i]))
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
                "gymnasium": {
                    "terminated": term_i,
                    "truncated": trunc_i,
                    "raw_info": to_serializable(slot_info),
                    "slot": i,
                }
            })

            # Reset return so the next episode for this slot starts at 0.
            # The runner's reset_slots() call comes between steps; this keeps
            # state consistent if the runner force-truncates via max_steps too.
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
        new_states: list[State] = []
        for slot, sc in zip(slots, scenarios):
            seed = self.seed_from_scenario(sc)
            options = self.options_from_scenario(sc)
            obs, _info = self.env.envs[slot].reset(seed=seed, options=options)
            self._episode_returns[slot] = 0.0
            new_states.append(self._state_from_obs(obs))
        return new_states

    def close(self) -> None:
        """Forward close to the wrapped env. Optional — not part of the Protocol."""
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    # --- Internals ----------------------------------------------------------

    def _state_from_obs(self, obs: Any) -> State:
        state = self.observation_to_state(obs)
        if self.coerce_observations:
            state = to_serializable(state)
        return state

    def _actions_to_array(self, actions: list[Any]) -> Any:
        """Stack per-slot actions into the shape gym.vector expects.

        For Discrete actions (int) -> ndarray shape (num_envs,).
        For Box actions (np.ndarray) -> ndarray shape (num_envs, *action_dim).
        For Dict/Tuple action spaces, the array form differs; the safest path
        is to let numpy figure it out via np.array(actions), falling back to
        a Python list for spaces that error on that conversion.
        """
        try:
            return np.array(actions)
        except (TypeError, ValueError):
            return actions


def _per_slot_info(
    infos: dict[str, Any],
    slot: int,
    num_envs: int,
    info_keys: list[str] | None,
) -> dict[str, Any]:
    """Extract the slot-specific entries from a batched info dict.

    Gymnasium's batched info has per-key arrays of length ``num_envs`` for
    most entries. ``_final_observation`` / ``final_observation`` (when
    present) are skipped — they're handled separately by the autoreset
    contract. Scalar entries (rare) are forwarded unchanged.
    """
    slot_info: dict[str, Any] = {}
    keys = info_keys if info_keys is not None else list(infos.keys())
    for key in keys:
        if key in ("final_observation", "_final_observation"):
            continue
        if key not in infos:
            continue
        value = infos[key]
        # Strings/bytes have __len__ too but are scalar info; don't index into them.
        if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
            try:
                if len(value) == num_envs:
                    slot_info[key] = value[slot]
                    continue
            except TypeError:
                pass
        slot_info[key] = value
    return slot_info
