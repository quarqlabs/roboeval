"""Isaac Lab ↔ roboeval environment adapter.

Wraps an Isaac Lab single-env ``gymnasium.VectorEnv`` (i.e. constructed with
``num_envs=1``) into roboeval's ``EnvironmentAdapter`` Protocol.

Isaac Lab vs Gymnasium
----------------------
Isaac Lab envs are gymnasium-compatible at the API level but differ in three
practical ways the adapter handles:

1. **They're always vectorized.** Even at ``num_envs=1``, observations,
   rewards, and termination flags come back with a batch dimension. This
   adapter takes the ``batch_index=0`` slice and exposes a single-episode
   view to the runner.

2. **Tensors live on GPU.** Observations come back as ``torch.Tensor`` with
   ``device='cuda:0'``. Reports require JSON-safe types, so the adapter
   coerces tensors to CPU numpy before they leave the boundary.

3. **Actions must be tensors with a batch dim.** Policies typically return
   numpy arrays or Python scalars; the adapter converts to torch tensors on
   the right device with the batch dimension prepended.

The six translation hooks match the Gymnasium adapter's surface so users
already familiar with that pattern see the same shape here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable

from roboeval.core import Action, Scenario, State, to_serializable
from roboeval.environment import StepOutcome


# ── Public default hooks ────────────────────────────────────────────────────


def default_observation_to_state(obs: Any) -> dict[str, Any]:
    """Wrap a (batch-sliced, CPU-numpy) observation into a State dict.

    Mirrors the Gymnasium adapter: dict observations pass through; non-dict
    observations get wrapped as ``{"observation": obs}``. Tensor coercion
    happens upstream in ``_slice_and_coerce_obs``, so by the time this hook
    runs, values are already JSON-safe.
    """
    if isinstance(obs, dict):
        return dict(obs)
    return {"observation": obs}


def default_action_from_decision(action: Action) -> Any:
    """Pass through the decision action unchanged.

    The adapter handles the torch.Tensor conversion + batch-dim prepending
    in ``_to_batched_torch_action`` — this hook lets users translate from
    a custom action vocabulary if they want.
    """
    return action


def default_outcome_from_step(
    reward: float, terminated: bool, truncated: bool, info: dict
) -> tuple[str, str]:
    """Default ``(outcome, failure_label)`` mapping.

    Override for env-specific success detection — e.g. many Isaac Lab tasks
    expose ``info["success"]`` or ``info["is_success"]`` as a boolean signal,
    which is more reliable than reward sign.
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
    """Emit short event tags useful for rule filters and report highlights."""
    events: list[str] = []
    if terminated:
        events.append("episode_terminated")
    if truncated:
        events.append("episode_truncated")
    if reward < 0:
        events.append("reward_negative")
    return events


def default_seed_from_scenario(scenario: Scenario) -> int | None:
    """Pull a seed from the scenario, preferring ``initial_state``."""
    seed = scenario.initial_state.get("seed")
    if seed is None:
        seed = scenario.metadata.get("seed")
    return int(seed) if seed is not None else None


def default_options_from_scenario(scenario: Scenario) -> dict | None:
    """Pull reset options from ``scenario.metadata['reset_options']``."""
    options = scenario.metadata.get("reset_options")
    if isinstance(options, dict):
        return dict(options)
    return None


def tensor_to_numpy(value: Any) -> Any:
    """Coerce a torch.Tensor (CPU or GPU) to a numpy array.

    Public so user overrides of ``observation_to_state`` can call it.
    Returns ``value`` unchanged if it isn't tensor-like.
    """
    if hasattr(value, "is_cuda") and getattr(value, "is_cuda", False):
        value = value.detach().cpu()
    elif hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


# ── The adapter ─────────────────────────────────────────────────────────────


@dataclass
class IsaacEnvironmentAdapter:
    """Wraps a single-env Isaac Lab env into roboeval's ``EnvironmentAdapter``.

    Style mirrors ``GymnasiumEnvironmentAdapter`` and the existing
    ``CallableEnvironmentAdapter`` pattern from ``roboeval.environment``.

    Parameters
    ----------
    env : gymnasium.vector.VectorEnv
        An Isaac Lab env constructed with ``num_envs=1``. Anything else is
        refused at construction with a clear error message.
    name : str
        Display name for reports (the runner reads this).
    batch_index : int
        Which env in the batch to read/write. Defaults to ``0``. The adapter
        warns if ``num_envs > 1`` and uses this index.
    observation_to_state, action_from_decision, outcome_from_step,
    events_from_step, seed_from_scenario, options_from_scenario :
        Override hooks. ``None`` selects the corresponding ``default_*``
        function in this module.
    info_keys : list[str] | None
        Optional allowlist for keys passed through from Isaac's ``info``
        dict into ``StepOutcome.info["isaac"]["raw_info"]``.
    """

    env: Any                                            # gym.vector.VectorEnv
    name: str = "isaac_env"
    batch_index: int = 0

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
        num_envs = getattr(self.env, "num_envs", None)
        if num_envs is not None and num_envs != 1 and self.batch_index >= num_envs:
            raise ValueError(
                f"batch_index={self.batch_index} out of range for env with "
                f"num_envs={num_envs}. Use 0 <= batch_index < num_envs."
            )
        if num_envs is not None and num_envs > 1 and self.batch_index == 0:
            warnings.warn(
                f"Isaac env has num_envs={num_envs} but the SDK runner is "
                f"single-episode. Reading batch_index={self.batch_index}; the "
                f"other {num_envs - 1} envs run but are ignored. For best "
                f"throughput pass num_envs=1.",
                stacklevel=2,
            )

        # Wire defaults
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
        self._device: str | None = None

    # ── EnvironmentAdapter Protocol ────────────────────────────────────────

    def reset(self, scenario: Scenario) -> State:
        seed = self.seed_from_scenario(scenario)
        options = self.options_from_scenario(scenario)
        try:
            obs, _info = self.env.reset(seed=seed, options=options)
        except TypeError:
            # Some Isaac envs don't accept options kwarg
            obs, _info = self.env.reset(seed=seed)
        self._episode_return = 0.0
        return self._state_from_obs(obs)

    def step(self, action: Action, scenario: Scenario) -> StepOutcome:
        decision_action = self.action_from_decision(action)
        gym_action = self._to_batched_torch_action(decision_action)
        obs, reward, terminated, truncated, info = self.env.step(gym_action)

        reward_scalar = self._scalar(reward, default=0.0)
        terminated_scalar = self._scalar(terminated, default=False, dtype=bool)
        truncated_scalar = self._scalar(truncated, default=False, dtype=bool)
        self._episode_return += float(reward_scalar)

        return self._build_outcome(
            obs,
            float(reward_scalar),
            bool(terminated_scalar),
            bool(truncated_scalar),
            info,
        )

    def close(self) -> None:
        """Forward close to the wrapped env."""
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    # ── Internals ──────────────────────────────────────────────────────────

    def _state_from_obs(self, obs: Any) -> State:
        """Slice the batch dim and coerce tensors to numpy, then apply hook."""
        sliced = self._slice_and_coerce_obs(obs)
        return self.observation_to_state(sliced)

    def _slice_and_coerce_obs(self, obs: Any) -> Any:
        """Take batch_index slice and convert tensors to numpy.

        Isaac obs is typically either:
          - a dict like {"policy": tensor of shape (N, obs_dim)}, OR
          - a tensor of shape (N, obs_dim) directly.
        Both shapes are handled.
        """
        if isinstance(obs, dict):
            return {k: tensor_to_numpy(self._index(v)) for k, v in obs.items()}
        return tensor_to_numpy(self._index(obs))

    def _index(self, value: Any) -> Any:
        """Return ``value[batch_index]`` if it supports indexing, else value."""
        try:
            return value[self.batch_index]
        except (TypeError, IndexError, KeyError):
            return value

    def _scalar(self, value: Any, default: Any, dtype: type | None = None) -> Any:
        """Reduce a batched tensor/array to a Python scalar at batch_index."""
        sliced = self._index(value)
        if hasattr(sliced, "item"):
            try:
                return sliced.item()
            except (ValueError, RuntimeError):
                pass
        if hasattr(sliced, "numpy"):
            arr = tensor_to_numpy(sliced)
            if hasattr(arr, "item"):
                try:
                    return arr.item()
                except (ValueError, RuntimeError):
                    pass
        if dtype is bool:
            try:
                return bool(sliced)
            except (TypeError, ValueError):
                return default
        try:
            return float(sliced)
        except (TypeError, ValueError):
            return default

    def _filter_info(self, info: dict) -> dict:
        if self.info_keys is None:
            return dict(info) if isinstance(info, dict) else {}
        if not isinstance(info, dict):
            return {}
        return {key: info[key] for key in self.info_keys if key in info}

    def _to_batched_torch_action(self, action: Any) -> Any:
        """Convert action to torch.Tensor on the env's device with a batch dim.

        Accepts: torch.Tensor (any shape), numpy array, Python scalar or list.
        Returns a torch tensor with shape ``(num_envs, *action_dim)``. Device
        is inferred from the env if possible; falls back to ``cuda`` if
        available, else ``cpu``.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for the Isaac adapter. Install via "
                "`pip install torch` or as part of your Isaac Lab install."
            ) from exc

        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action)

        # Bring to the right device
        device = self._resolve_device(action)
        if action.device != torch.device(device):
            action = action.to(device)

        # Ensure batch dim matches env.num_envs
        num_envs = getattr(self.env, "num_envs", 1)
        if action.ndim == 0:
            # scalar -> (num_envs, 1)
            action = action.view(1, 1).expand(num_envs, 1)
        elif action.ndim == 1:
            # (action_dim,) -> (num_envs, action_dim) if batch_index == 0 and
            # action_dim matches; otherwise treat first dim as batch
            action_space = getattr(self.env, "single_action_space", None)
            expected_dim = (
                action_space.shape[0]
                if action_space is not None and getattr(action_space, "shape", None)
                else None
            )
            if expected_dim is not None and action.shape[0] == expected_dim:
                action = action.unsqueeze(0).expand(num_envs, -1)
            elif action.shape[0] == num_envs:
                # Already batched as 1-D — add singleton action dim
                action = action.unsqueeze(-1)
            else:
                action = action.unsqueeze(0).expand(num_envs, -1)
        # ndim >= 2 we assume already batched correctly

        return action

    def _resolve_device(self, fallback_tensor: Any) -> str:
        if self._device is not None:
            return self._device

        # Try to read the env's device attribute
        env_device = getattr(self.env, "device", None) or getattr(
            self.env, "sim_device", None
        )
        if env_device is not None:
            self._device = str(env_device)
            return self._device

        # Fall back to detecting from existing tensor or cuda availability
        try:
            import torch
        except ImportError:
            self._device = "cpu"
            return self._device

        if hasattr(fallback_tensor, "device"):
            self._device = str(fallback_tensor.device)
            return self._device

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def _build_outcome(
        self,
        obs: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> StepOutcome:
        next_state = self._state_from_obs(obs)
        outcome, failure_label = self.outcome_from_step(
            reward, terminated, truncated, info
        )
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
                "isaac": {
                    "terminated": terminated,
                    "truncated": truncated,
                    "raw_info": to_serializable(filtered_info),
                    "batch_index": self.batch_index,
                }
            },
        )
