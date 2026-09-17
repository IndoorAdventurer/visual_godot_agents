"""
GodotVectorEnv — gymnasium.vector.VectorEnv wrapper around MultiIPCClient.

Plug-and-play with gymnasium-compatible training loops that honour
metadata["autoreset_mode"]. Loops written against SAME_STEP autoreset — CleanRL's
PPO reference implementations among them — need their GAE masking adjusted: a done
flag here marks the TERMINAL observation, so the new episode begins on the next one.
All action encoding/decoding happens internally; callers work with native
gymnasium spaces throughout.

Supported action space types
-----------------------------
Box(dtype=*)      — raw dtype copy; action_size must equal flat_dim * itemsize
MultiDiscrete     — one uint8 per dimension; action_size must equal len(nvec)
Discrete          — single uint8; action_size must equal 1
"""

import numpy as np
import gymnasium
import gymnasium.spaces.utils as gym_utils
from gymnasium.vector import AutoresetMode
from gymnasium.vector.utils import batch_space
from typing import Any

from .multi_ipc_client import MultiIPCClient


class GodotVectorEnv(gymnasium.vector.VectorEnv):
    """
    Wraps MultiIPCClient as a gymnasium VectorEnv.

    Observation space is per-environment. Pass a Box for visual-only obs, or a
    Dict{"visual": Box, "scalar": Box} when include_scalar_obs=True.

    If godot_binary is None, Godot is not launched — the caller is responsible
    for starting it manually before calling reset() (only valid for num_instances=1).

    Autoreset is NEXT_STEP: a done flag arrives with the terminal observation, and
    the first observation of the new episode follows on the next step() with reward
    0.0 and both flags false.
    """

    # Subclasses extend this with {**GodotVectorEnv.metadata, ...}; __init__ gives
    # each instance its own copy, so in-place edits stay local to that instance.
    metadata = {"autoreset_mode": AutoresetMode.NEXT_STEP}
    autoreset_mode = AutoresetMode.NEXT_STEP

    def __init__(
        self,
        name: str,
        num_envs: int,
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        include_scalar_obs: bool = False,
        num_instances: int = 1,
        godot_binary: str | None = None,
        project_path: str | None = None,
        obs_width: int | None = None,
        obs_height: int | None = None,
        obs_channels: int | None = None,
        step_rate_hz: int | None = None,
        extra_args: dict[str, str] | None = None,
    ):
        # gymnasium 1.x VectorEnv.__init__ takes no args; set required attributes directly.
        super().__init__()
        self.metadata = dict(self.metadata)
        self.num_envs = num_envs
        self.single_observation_space = observation_space
        self.single_action_space = action_space
        self.observation_space = batch_space(observation_space, num_envs)
        self.action_space = batch_space(action_space, num_envs)

        self._include_scalar_obs = include_scalar_obs
        self._action_dtype, self._action_flat_dim = _parse_action_space(action_space)
        self._connected = False

        self._client = MultiIPCClient(
            name,
            num_envs=num_envs,
            num_instances=num_instances,
            godot_binary=godot_binary,
            project_path=project_path,
            obs_width=obs_width,
            obs_height=obs_height,
            obs_channels=obs_channels,
            step_rate_hz=step_rate_hz,
            extra_args=extra_args,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict]:
        # TODO: seed is ignored — forwarding it needs its own IPC channel and a
        # GDScript-side API for consuming it.
        if self._connected:
            self._state = self._client.reset()
            return self._get_obs(), {}

        self._state = self._client.connect()
        self._connected = True
        self._validate_layout()
        return self._get_obs(), {}

    def step(
        self, actions: np.ndarray
    ) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict]:
        encoded = self._encode_actions(actions)
        self._state = self._client.step(encoded)
        return (
            self._get_obs(),
            self._state.rewards,
            self._state.terminated.astype(bool),
            self._state.truncated.astype(bool),
            {},
        )

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> Any:
        if self._include_scalar_obs:
            return {
                "visual": self._state.visual_obs,
                "scalar": self._state.scalar_obs,
            }
        return self._state.visual_obs

    def _encode_actions(self, actions: np.ndarray) -> np.ndarray:
        """Convert actions from the gymnasium action space dtype to uint8 bytes."""
        # View the flat per-env byte representation as uint8 for the IPC buffer.
        flat = actions.reshape(self.num_envs, self._action_flat_dim)
        if flat.dtype != self._action_dtype:
            flat = flat.astype(self._action_dtype)
        return flat.view(np.uint8).reshape(self.num_envs, -1)

    def _validate_layout(self) -> None:
        """
        Cross-check that the observation/action spaces match what Godot's header
        reports. Raises ValueError on any mismatch so stale space definitions are
        caught immediately rather than causing silent wrong-shape training runs.
        """
        c = self._client

        if c.num_envs != self.num_envs:
            raise ValueError(
                f"num_envs mismatch: GodotVectorEnv was told {self.num_envs} "
                f"but Godot reports {c.num_envs}"
            )

        # Visual obs
        visual_space = (
            self.single_observation_space["visual"]
            if self._include_scalar_obs
            else self.single_observation_space
        )
        expected_visual = (c.visual_height, c.visual_width, c.visual_channels)
        if visual_space.shape != expected_visual:
            raise ValueError(
                f"visual obs shape mismatch: space has {visual_space.shape}, "
                f"Godot reports {expected_visual}"
            )

        # Scalar obs (optional)
        if self._include_scalar_obs:
            scalar_space = self.single_observation_space["scalar"]
            if scalar_space.shape[0] != c.scalar_obs_size:
                raise ValueError(
                    f"scalar obs size mismatch: space has {scalar_space.shape[0]}, "
                    f"Godot reports {c.scalar_obs_size}"
                )

        # Actions: flat_dim * itemsize must equal the raw byte count per env
        expected_bytes = self._action_flat_dim * np.dtype(self._action_dtype).itemsize
        if expected_bytes != c.action_size:
            raise ValueError(
                f"action size mismatch: space implies {expected_bytes} bytes per env "
                f"({self._action_flat_dim} elements × "
                f"{np.dtype(self._action_dtype).itemsize} bytes), "
                f"Godot reports {c.action_size}"
            )


def _parse_action_space(space: gymnasium.Space) -> tuple[np.dtype, int]:
    """
    Return (dtype, flat_element_count) for the given action space.

    The IPC buffer stores action_size bytes per env. For dtype d and flat_dim
    elements: action_size must equal flat_dim * d.itemsize.
    """
    if isinstance(space, gymnasium.spaces.Box):
        return np.dtype(space.dtype), gym_utils.flatdim(space)
    if isinstance(space, gymnasium.spaces.MultiDiscrete):
        return np.dtype(np.uint8), len(space.nvec)
    if isinstance(space, gymnasium.spaces.Discrete):
        return np.dtype(np.uint8), 1
    raise TypeError(
        f"Unsupported action space type {type(space).__name__}. "
        "Use Box, MultiDiscrete, or Discrete."
    )