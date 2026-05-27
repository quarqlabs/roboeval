"""Gymnasium ↔ roboeval environment adapter.

Wraps any ``gymnasium.Env`` into roboeval's ``EnvironmentAdapter`` Protocol so
the existing ``EvalRunner`` can drive Gymnasium environments without any change
to the core SDK.

Translation layer
-----------------
The adapter performs six translations between Gymnasium and the SDK. Each is
exposed as an overridable hook so users can customize the boundary without
subclassing this adapter or modifying the SDK:

1. ``observation_to_state``    Gymnasium observation -> ``State`` dict
2. ``action_from_decision``    Decision action -> Gymnasium-native action
3. ``outcome_from_step``       (reward, terminated, truncated, info) -> outcome label
4. ``events_from_step``        (reward, terminated, truncated, info) -> event tags
5. ``seed_from_scenario``      ``Scenario`` -> ``env.reset(seed=...)``
6. ``options_from_scenario``   ``Scenario`` -> ``env.reset(options=...)``

Each hook has a sensible default that works for the common cases (Discrete
action spaces, Box observations, sparse-reward end-of-episode envs).

StepOutcome shape
-----------------
The adapter populates ``StepOutcome`` directly using the SDK's existing 8
fields. Gymnasium-specific raw data lives under ``info["gymnasium"]`` so it
travels through the eval pipeline as structured data without polluting the
domain-level ``next_state`` dict::

    StepOutcome(
        next_state={"observation": ...},
        outcome="progress",
        failure_label="",
        terminal=False,
        metrics={"reward": float, "episode_return": float},
        events=["episode_terminated", ...],
        info={"gymnasium": {"terminated": ..., "truncated": ..., "raw_info": ...}},
    )

Numeric values land in ``metrics`` so the SDK's metric summaries and
``require_metric`` rules pick them up automatically. Raw / debug data lands in
``info`` so it round-trips through the eval reports without semantic coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import gymnasium as gym

from roboeval.core import Action, Scenario, State, to_serializable
from roboeval.environment import StepOutcome


# --- Default hook implementations -------------------------------------------
#
# All defaults are public so user overrides can compose on top of them rather
# than reimplementing the common path. Example::
#
#     def my_outcome(reward, terminated, truncated, info):
#         outcome, label = default_outcome_from_step(reward, terminated, truncated, info)
#         if info.get("is_success"):
#             return ("goal_reached", "")
#         return outcome, label
#
#     adapter = GymnasiumEnvironmentAdapter(env=..., outcome_from_step=my_outcome)


def default_observation_to_state(obs: Any) -> dict[str, Any]:
    """Wrap a Gymnasium observation into a ``State`` dict.

    - ``dict`` observations are passed through (shallow-copied).
    - Non-dict observations (Box, Discrete, etc.) are wrapped as
      ``{"observation": obs}`` so the SDK's dict invariant holds.

    Values are not coerced here — the SDK's ``to_serializable`` handles JSON
    safety at report-write time. If a user wants pre-coerced observations
    visible to their policies, set ``coerce_observations=True`` on the adapter.
    """
    if isinstance(obs, dict):
        return dict(obs)
    return {"observation": obs}


def default_action_from_decision(action: Action) -> Any:
    """Pass through the decision action unchanged.

    The SDK's ``Action = Any`` type means Gymnasium-native actions (int for
    Discrete, ``np.ndarray`` for Box, dict for Dict spaces) round-trip without
    coercion. Override this hook to translate from a user-defined action
    vocabulary (e.g. ``"left" -> 0``).
    """
    return action


def default_outcome_from_step(
    reward: float, terminated: bool, truncated: bool, info: dict
) -> tuple[str, str]:
    """Default ``(outcome, failure_label)`` mapping for generic Gymnasium envs.

    Override for envs with richer semantics (e.g. ``info["is_success"]`` in
    manipulation benchmarks).
    """
    if terminated:
        if reward > 0:
            return ("terminated_success", "")
        return ("terminated_failure", "terminated_failure")
    if truncated:
        return ("truncated", "timeout")
    return ("progress", "")


def default_events_from_step(
    reward: float, terminated: bool, truncated: bool, info: dict
) -> list[str]:
    """Default event tags emitted from each step.

    Events surface in reports and feed custom ``Rule`` definitions. Add more
    domain-specific tags by overriding this hook.
    """
    events: list[str] = []
    if terminated:
        events.append("episode_terminated")
    if truncated:
        events.append("episode_truncated")
    if reward < 0:
        events.append("reward_negative")
    return events


def default_seed_from_scenario(scenario: Scenario) -> int | None:
    """Pull a Gymnasium ``seed`` from the scenario, preferring ``initial_state``."""
    seed = scenario.initial_state.get("seed")
    if seed is None:
        seed = scenario.metadata.get("seed")
    return int(seed) if seed is not None else None


def default_options_from_scenario(scenario: Scenario) -> dict | None:
    """Pull Gymnasium reset ``options`` from ``scenario.metadata['reset_options']``.

    Returns ``None`` when no options are provided so the env's own defaults run.
    """
    options = scenario.metadata.get("reset_options")
    if isinstance(options, dict):
        return dict(options)
    return None


# --- The adapter ------------------------------------------------------------


@dataclass
class GymnasiumEnvironmentAdapter:
    """Wraps any ``gymnasium.Env`` into roboeval's ``EnvironmentAdapter``.

    The style mirrors ``CallableEnvironmentAdapter`` in ``roboeval.environment``:
    a small dataclass with a ``name`` field, no inheritance, and explicit hook
    callables for the per-axis translations.

    Parameters
    ----------
    env :
        The Gymnasium environment to wrap. Must be a single-env
        ``gymnasium.Env`` (vector envs are explicitly refused — see
        ``__post_init__``).
    name :
        Display name for reports (read by the runner's ``_environment_name``).
    observation_to_state, action_from_decision, outcome_from_step,
    events_from_step, seed_from_scenario, options_from_scenario :
        Override hooks. ``None`` selects the corresponding ``default_*``
        implementation in this module.
    info_keys :
        Optional allowlist for keys passed through from Gymnasium's ``info``
        dict into ``StepOutcome.info["gymnasium"]["raw_info"]``. Useful when the
        env emits large tensors in ``info`` that should not balloon the eval
        logs. ``None`` passes everything through.
    coerce_observations :
        When ``True``, observations are pre-coerced to JSON-safe Python types
        via ``to_serializable`` before being returned. Off by default because
        the runner already coerces at report-write time.
    """

    env: gym.Env
    name: str = "gymnasium_env"

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
        if isinstance(self.env, gym.vector.VectorEnv):
            raise NotImplementedError(
                "GymnasiumEnvironmentAdapter does not support gym.vector.VectorEnv. "
                "The roboeval runner is single-episode; pass a single env "
                "(e.g. env.envs[0], or gym.make(env_id) without vectorization)."
            )

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

        self._episode_return: float = 0.0

    # --- EnvironmentAdapter Protocol ----------------------------------------

    def reset(self, scenario: Scenario) -> State:
        seed = self.seed_from_scenario(scenario)
        options = self.options_from_scenario(scenario)
        obs, _info = self.env.reset(seed=seed, options=options)
        self._episode_return = 0.0
        return self._state_from_obs(obs)

    def step(self, action: Action, scenario: Scenario) -> StepOutcome:
        gym_action = self.action_from_decision(action)
        obs, reward, terminated, truncated, info = self.env.step(gym_action)
        reward_value = float(reward)
        self._episode_return += reward_value
        return self._build_outcome(obs, reward_value, bool(terminated), bool(truncated), info)

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

    def _filter_info(self, info: dict) -> dict:
        if self.info_keys is None:
            return dict(info)
        return {key: info[key] for key in self.info_keys if key in info}

    def _build_outcome(
        self,
        obs: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> StepOutcome:
        next_state = self._state_from_obs(obs)
        outcome, failure_label = self.outcome_from_step(reward, terminated, truncated, info)
        events = self.events_from_step(reward, terminated, truncated, info)
        filtered_info = self._filter_info(info)

        return StepOutcome(
            next_state=next_state,
            outcome=outcome,
            failure_label=failure_label,
            terminal=bool(terminated or truncated),
            metrics={
                "reward": reward,
                "episode_return": float(self._episode_return),
            },
            events=events,
            info={
                "gymnasium": {
                    "terminated": terminated,
                    "truncated": truncated,
                    "raw_info": to_serializable(filtered_info),
                }
            },
        )
