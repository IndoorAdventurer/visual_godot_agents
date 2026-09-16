"""
Frame stacking for vector environments, via observation handles.

step() and reset() return an integer handle per environment instead of an
observation; get_obs(handles) resolves handles into stacked observations, whether
those are the handles just returned while acting or a shuffled minibatch of them
while training. Frames are stored once, so memory does not grow with stack depth.

observation_space describes a handle, not an observation, so this does not compose
with tooling that expects step() to return real observations — use it when you
control the training loop. py_vga.naive_frame_stack.NaiveFrameStack is the portable
alternative, returning real observations at a rollout buffer `stack` times larger.

A handle encodes (timestep, env) and is never reused, including across clear().
Frames are retained for `capacity` timesteps; a handle whose stack has been evicted
raises IndexError.

A stack never reads across an episode boundary: the observation following a done
flag is treated as the first of a new episode.

Observations are returned in their native dtype — scale them in the policy.

torch and gymnasium are optional dependencies of py_vga, so import directly:

    from py_vga.indexed_frame_stack import IndexedFrameStack
"""

# TODO: SAC support — a next_obs lookup (handle + 1, stopping at episode boundaries)
#       and a ring sized for replay rather than a single rollout.
# TODO: helper to derive `capacity` from num_steps, stack and stride.

import gymnasium
import numpy as np
import torch
from gymnasium.vector import VectorWrapper
from gymnasium.vector.utils import batch_space

from .frame_stack_common import SINGLE_KEY, leaf_spaces, validate_stacking


class IndexedFrameStack(VectorWrapper):
    """
    Vector env wrapper returning observation handles instead of observations.

    Args:
        env: vector env whose single observation space is a Box, or a Dict of Boxes;
            every leaf is stacked independently.
        capacity: timesteps of frames to retain. Must exceed (stack - 1) * stride.
            For PPO: num_steps + (stack - 1) * stride.
        stack: frames per observation; 1 disables stacking.
        stride: gap between stacked frames; 1 is consecutive, n takes every nth.
        padding: fill for a stack reaching past an episode start, "repeat" or "zero".
        device: where frames are stored.
        compute_device: where get_obs returns tensors. Defaults to `device`.

    Stacked leaves are (batch, stack * C, H, W) for (H, W, C) leaves and
    (batch, stack * D) for (D,) leaves, newest last. See `stacked_observation_space`.
    """

    def __init__(
        self,
        env,
        capacity: int,
        *,
        stack: int = 1,
        stride: int = 1,
        padding: str = "repeat",
        device: torch.device | str = "cpu",
        compute_device: torch.device | str | None = None,
    ):
        super().__init__(env)
        validate_stacking(stack, stride, padding)

        self.stack = stack
        self.stride = stride
        self.padding = padding
        self.hist = (stack - 1) * stride
        if capacity <= self.hist:
            raise ValueError(
                f"capacity must exceed (stack - 1) * stride = {self.hist}, got {capacity}. "
                f"For PPO use num_steps + {self.hist}."
            )
        self.capacity = capacity

        self.device = torch.device(device)
        self.compute_device = (
            torch.device(compute_device) if compute_device is not None else self.device
        )

        self.frame_observation_space = env.single_observation_space
        leaves = leaf_spaces(self.frame_observation_space)

        n = self.num_envs
        # Pinned host memory lets the gather's copy to an accelerator overlap compute.
        pin = self.device.type == "cpu" and self.compute_device.type != "cpu"
        self._stores = {}
        for key, space in leaves.items():
            self._stores[key] = torch.zeros(
                (capacity, n) + space.shape,
                dtype=_torch_dtype(space.dtype),
                device=self.device,
                pin_memory=pin,
            )

        # Episode start per stored row, so any resident handle can be resolved.
        self._ep_start = torch.zeros((capacity, n), dtype=torch.int64, device=self.device)
        self._cur_ep_start = torch.zeros(n, dtype=torch.int64, device=self.device)
        self._pending_new_ep = np.ones(n, dtype=bool)
        self._newest_t = -1
        self._floor_t = 0

        # Oldest -> newest, so the newest frame lands last in the stack.
        self._offsets = torch.arange(stack - 1, -1, -1, device=self.device) * stride

        handle_space = gymnasium.spaces.Box(
            low=0, high=np.iinfo(np.int64).max, shape=(), dtype=np.int64
        )
        self.single_observation_space = handle_space
        self.observation_space = batch_space(handle_space, n)
        self.stacked_observation_space = _stacked_space(self.frame_observation_space, stack)

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._pending_new_ep = np.ones(self.num_envs, dtype=bool)
        handles = self._record(obs)
        self._pending_new_ep = np.zeros(self.num_envs, dtype=bool)
        return handles, info

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        handles = self._record(obs)
        # Done marks this observation as terminal; the next one starts an episode.
        self._pending_new_ep = np.logical_or(terminations, truncations)
        return handles, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Observation lookup
    # ------------------------------------------------------------------

    def get_obs(self, handles):
        """
        Stacked observations for `handles`, in the wrapped env's structure: a tensor
        for a Box space, a dict of tensors for a Dict space, on `compute_device`.

        `handles` may be a sequence, numpy array or torch tensor on any device; it is
        flattened, so the batch dimension is the number given. Raises IndexError for a
        handle not yet produced, or whose stack has been evicted.
        """
        if isinstance(handles, torch.Tensor):
            h = handles.detach().reshape(-1).to(self.device, torch.int64)
        else:
            # ascontiguousarray: torch cannot view numpy arrays with negative strides.
            h = torch.as_tensor(
                np.ascontiguousarray(handles, dtype=np.int64).reshape(-1), device=self.device
            )
        t = torch.div(h, self.num_envs, rounding_mode="floor")
        env_ids = h - t * self.num_envs

        if h.numel() and int(t.max()) > self._newest_t:
            bad = int(h[int(torch.argmax(t))])
            raise IndexError(
                f"handle {bad} has not been produced yet "
                f"(newest timestep is {self._newest_t})"
            )

        raw = t.unsqueeze(1) - self._offsets  # pre-clamp, may precede the episode
        ep_start = self._ep_start[t % self.capacity, env_ids]
        idx = torch.maximum(raw, ep_start.unsqueeze(1))

        oldest = self._oldest_t()
        if h.numel():
            needed = idx.min(dim=1).values
            if int(needed.min()) < oldest:
                bad = int(h[int(torch.argmin(needed))])
                raise IndexError(
                    f"handle {bad} needs timestep {int(needed.min())}, which has been "
                    f"evicted (oldest retained is {oldest})"
                )

        slots = (idx % self.capacity) * self.num_envs + env_ids.unsqueeze(1)
        invalid = raw < ep_start.unsqueeze(1) if self.padding == "zero" else None

        out = {}
        for key, store in self._stores.items():
            leaf = store.view(self.capacity * self.num_envs, *store.shape[2:])[slots]
            leaf = leaf.to(self.compute_device, non_blocking=True)
            if invalid is not None:
                mask = invalid.to(self.compute_device)
                leaf = leaf.masked_fill(mask.view(*mask.shape, *([1] * (leaf.ndim - 2))), 0)
            if leaf.ndim == 5:  # (batch, stack, H, W, C) -> (batch, stack * C, H, W)
                leaf = leaf.permute(0, 1, 4, 2, 3)
            out[key] = leaf.flatten(1, 2)
        return out[SINGLE_KEY] if SINGLE_KEY in out else out

    def clear(self) -> None:
        """
        Drop the retained rollout, keeping the frames needed to keep stacking across
        the boundary into the next one. Call after the update epochs.

        Handles are not reset, so anything from the discarded rollout raises after.
        """
        self._floor_t = max(0, self._newest_t + 1 - self.hist)

    @property
    def nbytes(self) -> int:
        """Bytes held by the frame stores."""
        return sum(s.numel() * s.element_size() for s in self._stores.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _oldest_t(self) -> int:
        return max(self._floor_t, self._newest_t - self.capacity + 1, 0)

    def _record(self, obs) -> np.ndarray:
        """Store one batched observation and return its handles."""
        t = self._newest_t + 1
        slot = t % self.capacity

        starts = torch.as_tensor(self._pending_new_ep, device=self.device).bool()
        self._cur_ep_start = torch.where(
            starts, torch.full_like(self._cur_ep_start, t), self._cur_ep_start
        )
        self._ep_start[slot] = self._cur_ep_start

        for key, store in self._stores.items():
            leaf = obs if key is SINGLE_KEY else obs[key]
            if isinstance(leaf, np.ndarray):
                leaf = torch.from_numpy(leaf)
            store[slot] = leaf.to(self.device, non_blocking=True)

        self._newest_t = t
        return t * self.num_envs + np.arange(self.num_envs, dtype=np.int64)


def _stacked_leaf(space, stack: int):
    if len(space.shape) == 3:  # (H, W, C) image
        h, w, c = space.shape
        shape = (stack * c, h, w)
    elif len(space.shape) == 1:  # (D,) vector
        shape = (stack * space.shape[0],)
    else:
        raise NotImplementedError(
            f"stacking a leaf of shape {space.shape} is not supported; "
            "expected (H, W, C) or (D,)"
        )
    return gymnasium.spaces.Box(
        low=space.low.min(), high=space.high.max(), shape=shape, dtype=space.dtype
    )


def _stacked_space(space, stack: int):
    """Space describing what get_obs returns for a single env."""
    if isinstance(space, gymnasium.spaces.Box):
        return _stacked_leaf(space, stack)
    return gymnasium.spaces.Dict(
        {key: _stacked_leaf(leaf, stack) for key, leaf in space.spaces.items()}
    )


def _torch_dtype(np_dtype) -> torch.dtype:
    return torch.from_numpy(np.zeros(1, dtype=np_dtype)).dtype
