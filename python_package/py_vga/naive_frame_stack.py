"""
Portable frame stacking for vector environments.

NaiveFrameStack returns real stacked observations through the standard gymnasium
VectorEnv API, as numpy arrays with a leading stack axis, so it composes with any
consumer of that API. Every observation the caller stores is `stack` frames, so a
rollout buffer built from it grows with stack depth; the wrapper itself holds only
(stack - 1) * stride + 1 frames per env.

See py_vga.indexed_frame_stack.IndexedFrameStack for a variant whose memory does
not grow with stack depth, at the cost of returning handles rather than
observations.

A stack never reads across an episode boundary: the observation following a done
flag is treated as the first of a new episode.

Observations are returned in their native dtype — scale them in the policy.

This module needs only gymnasium and numpy. gymnasium is an optional dependency of
py_vga, so import directly:

    from py_vga.naive_frame_stack import NaiveFrameStack
"""

import numpy as np
from gymnasium.vector import VectorWrapper
from gymnasium.vector.utils import batch_space

from .frame_stack_common import SINGLE_KEY, leaf_spaces, validate_stacking


class NaiveFrameStack(VectorWrapper):
    """
    Vector env wrapper returning stacked observations as numpy arrays.

    Args:
        env: vector env whose single observation space is a Box, or a Dict of Boxes;
            every leaf is stacked independently.
        stack: frames per observation; 1 disables stacking.
        stride: gap between stacked frames; 1 is consecutive, n takes every nth.
        padding: fill for a stack reaching past an episode start, "repeat" or "zero".

    Stacked leaves gain a leading stack axis, (stack, *leaf_shape), newest last,
    matching gymnasium.wrappers.FrameStackObservation.
    """

    def __init__(self, env, *, stack: int = 1, stride: int = 1, padding: str = "repeat"):
        super().__init__(env)
        validate_stacking(stack, stride, padding)

        self.stack = stack
        self.stride = stride
        self.padding = padding
        self.hist = (stack - 1) * stride
        self._window = self.hist + 1

        self.frame_observation_space = env.single_observation_space
        n = self.num_envs
        self._buffers = {
            key: np.zeros((self._window, n) + space.shape, dtype=space.dtype)
            for key, space in leaf_spaces(self.frame_observation_space).items()
        }

        # Oldest -> newest, so the newest frame lands last in the stack.
        self._offsets = np.arange(stack - 1, -1, -1) * stride
        self._env_ids = np.arange(n)
        self._ep_start = np.zeros(n, dtype=np.int64)
        self._pending_new_ep = np.ones(n, dtype=bool)
        self._t = -1

        self.single_observation_space = batch_space(self.frame_observation_space, stack)
        self.observation_space = batch_space(self.single_observation_space, n)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._pending_new_ep = np.ones(self.num_envs, dtype=bool)
        stacked = self._record(obs)
        self._pending_new_ep = np.zeros(self.num_envs, dtype=bool)
        return stacked, info

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        stacked = self._record(obs)
        # Done marks this observation as terminal; the next one starts an episode.
        self._pending_new_ep = np.logical_or(terminations, truncations)
        return stacked, rewards, terminations, truncations, infos

    def _record(self, obs):
        """Store one batched observation and return the stack ending at it."""
        self._t += 1
        t = self._t
        self._ep_start = np.where(self._pending_new_ep, t, self._ep_start)

        slot = t % self._window
        for key, buf in self._buffers.items():
            buf[slot] = obs if key is SINGLE_KEY else obs[key]

        raw = (t - self._offsets)[None, :]
        idx = np.maximum(raw, self._ep_start[:, None])  # never cross an episode start
        invalid = raw < self._ep_start[:, None] if self.padding == "zero" else None

        out = {}
        for key, buf in self._buffers.items():
            gathered = buf[idx % self._window, self._env_ids[:, None]]
            if invalid is not None:
                gathered[np.broadcast_to(invalid, gathered.shape[:2])] = 0
            out[key] = gathered
        return out[SINGLE_KEY] if SINGLE_KEY in out else out
