"""
MultiIPCClient — manages one or more IPCClient shards across N Godot processes.

For num_instances=1 the shard IPC name equals the base name; for N>1 shards are
named "<base_name>_0" … "<base_name>_{N-1}". Multi-instance (N>1) requires a
godot_binary since launching N Godot processes manually is not supported.

connect() and step() fan out to all shards in parallel when N>1 (semaphore waits
release the GIL so threading is sufficient). _merge() allocates a fresh EnvState
each call so callers own the returned arrays and can store them safely without an
extra copy.
"""

import atexit
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from .ipc_client import IPCClient, EnvState


class MultiIPCClient:
    """
    Unified client for one or more Godot instances.

    Presents the same connect() / step() / close() interface as IPCClient.
    Layout attributes (num_envs, visual_width, visual_height, visual_channels,
    scalar_obs_size, action_size) are populated after connect().
    """

    def __init__(
        self,
        name: str,
        num_envs: int,
        num_instances: int = 1,
        godot_binary: str | None = None,
        project_path: str | None = None,
        obs_width: int | None = None,
        obs_height: int | None = None,
        obs_channels: int | None = None,
        step_rate_hz: int | None = None,
        extra_args: dict[str, str] | None = None,
    ):
        if num_envs % num_instances != 0:
            raise ValueError(
                f"num_envs ({num_envs}) must be divisible by num_instances ({num_instances})"
            )
        if num_instances > 1 and godot_binary is None:
            raise ValueError(
                "godot_binary is required when num_instances > 1; "
                "launching multiple Godot instances manually is not supported"
            )

        self._envs_per_shard = num_envs // num_instances

        # Single-instance keeps the bare name so it matches existing Godot scenes.
        if num_instances == 1:
            self._shards = [IPCClient(name)]
        else:
            self._shards = [IPCClient(f"{name}_{i}") for i in range(num_instances)]

        self._executor: ThreadPoolExecutor | None = None
        atexit.register(self.close)

        # Layout — populated on connect().
        self.num_envs        = num_envs
        self.visual_obs_size = 0
        self.visual_width    = 0
        self.visual_height   = 0
        self.visual_channels = 0
        self.scalar_obs_size = 0
        self.action_size     = 0

        if godot_binary is not None:
            for shard in self._shards:
                shard.launch_godot(
                    godot_binary,
                    project_path=project_path,
                    num_envs=self._envs_per_shard,
                    obs_width=obs_width,
                    obs_height=obs_height,
                    obs_channels=obs_channels,
                    step_rate_hz=step_rate_hz,
                    extra_args=extra_args,
                )

    def connect(self) -> EnvState:
        """Connect to all shards (in parallel if N>1) and return the first combined state."""
        shard_states = self._run_parallel(lambda shard: shard.connect())

        s0 = self._shards[0]
        self.visual_obs_size = s0.visual_obs_size
        self.visual_width    = s0.visual_width
        self.visual_height   = s0.visual_height
        self.visual_channels = s0.visual_channels
        self.scalar_obs_size = s0.scalar_obs_size
        self.action_size     = s0.action_size

        return self._merge(shard_states)

    def step(self, actions: np.ndarray) -> EnvState:
        """Fan out actions to all shards, wait for results, return a fresh combined state."""
        n = self._envs_per_shard
        shard_states = self._run_parallel(
            lambda shard, i=None: shard.step(actions[i * n:(i + 1) * n]),
            pass_index=True,
        )
        return self._merge(shard_states)

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        for shard in self._shards:
            shard.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_parallel(self, fn, pass_index: bool = False) -> list[EnvState]:
        """Call fn(shard) — or fn(shard, i) when pass_index=True — for every shard.
        Uses a ThreadPoolExecutor for N>1; calls directly for N=1."""
        if len(self._shards) == 1:
            shard = self._shards[0]
            return [fn(shard, 0) if pass_index else fn(shard)]

        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=len(self._shards))

        if pass_index:
            futures = [self._executor.submit(fn, shard, i) for i, shard in enumerate(self._shards)]
        else:
            futures = [self._executor.submit(fn, shard) for shard in self._shards]
        return [f.result() for f in futures]

    def _merge(self, shard_states: list[EnvState]) -> EnvState:
        """Allocate a fresh EnvState and copy each shard's slice into it."""
        n = self._envs_per_shard
        state = EnvState(
            visual_obs= np.empty((self.num_envs, self.visual_height, self.visual_width, self.visual_channels), dtype=np.uint8),
            scalar_obs= np.empty((self.num_envs, self.scalar_obs_size), dtype=np.uint8),
            rewards=    np.empty((self.num_envs,), dtype=np.float32),
            terminated= np.empty((self.num_envs,), dtype=np.uint8),
            truncated=  np.empty((self.num_envs,), dtype=np.uint8),
        )
        for i, s in enumerate(shard_states):
            lo, hi = i * n, (i + 1) * n
            state.visual_obs[lo:hi] = s.visual_obs
            state.scalar_obs[lo:hi] = s.scalar_obs
            state.rewards[lo:hi]    = s.rewards
            state.terminated[lo:hi] = s.terminated
            state.truncated[lo:hi]  = s.truncated
        return state