import ctypes
import mmap
import posix_ipc
import numpy as np
from dataclasses import dataclass


@dataclass
class EnvState:
    """
    Numpy views into shared memory for all N environments.
    Arrays are updated in-place by Godot each step — copy explicitly if you
    need to retain values across a step() call.
    """
    visual_obs: np.ndarray   # (N, H, W, C)  uint8
    scalar_obs: np.ndarray   # (N, scalar_obs_size)  uint8
    rewards:    np.ndarray   # (N,)                  float32
    dones:      np.ndarray   # (N,)                  uint8


class _Header(ctypes.Structure):
    """Mirrors IPCController::Header in ipc_controller.h."""
    _fields_ = [
        ("num_envs",         ctypes.c_uint32),
        ("visual_obs_size",  ctypes.c_uint32),
        ("scalar_obs_size",  ctypes.c_uint32),
        ("action_size",      ctypes.c_uint32),
        ("visual_width",     ctypes.c_uint32),
        ("visual_height",    ctypes.c_uint32),
        ("visual_channels",  ctypes.c_uint32),
    ]

_HEADER_SIZE = ctypes.sizeof(_Header)


class IPCClient:
    """
    Low-level Python client for the HPA IPC protocol.

    Python creates and owns the semaphores; Godot creates and owns the shared
    memory. On connect(), Python blocks on env_ready until Godot has written
    the first state, then opens the shared memory that is guaranteed to exist
    by that point.

    The EnvState returned by connect() and step() holds numpy views directly
    into the shared memory buffer — no per-step copies. The same EnvState
    instance is returned every call; its arrays are updated in-place by Godot
    between steps.

    Usage:
    ```
    with IPCClient("my_env") as client:
        state = client.connect()
        while True:
            actions = np.zeros((client.num_envs, client.action_size), dtype=np.uint8)
            state = client.step(actions)
    ```
    """

    def __init__(self, name: str):
        self._name = name
        self._closed = False

        # Pre-unlink any semaphores left by a previous crash, then create fresh.
        for sem_name in [f"/{name}_env_ready", f"/{name}_act_ready"]:
            try:
                posix_ipc.unlink_semaphore(sem_name)
            except posix_ipc.ExistentialError:
                pass

        self._env_ready = posix_ipc.Semaphore(
            f"/{name}_env_ready", posix_ipc.O_CREAT, initial_value=0
        )
        self._act_ready = posix_ipc.Semaphore(
            f"/{name}_act_ready", posix_ipc.O_CREAT, initial_value=0
        )
        self._mem: mmap.mmap | None = None

        # Layout dimensions — populated by _init_layout() on connect():
        self.num_envs:        int = 0
        self.visual_obs_size: int = 0
        self.visual_width:    int = 0
        self.visual_height:   int = 0
        self.visual_channels: int = 0
        self.scalar_obs_size: int = 0
        self.action_size:     int = 0

        self._state:       EnvState | None    = None
        self._actions_buf: np.ndarray | None  = None

    def connect(self) -> EnvState:
        """
        Block until Godot signals env_ready, open shared memory, and return
        the first environment state. Call this once before stepping.
        """
        self._env_ready.acquire()
        shm = posix_ipc.SharedMemory(f"/{self._name}")
        self._mem = mmap.mmap(shm.fd, shm.size)
        shm.close_fd()
        self._init_layout()
        return self._state

    def step(self, actions: np.ndarray) -> EnvState:
        """Write actions, signal Godot, wait for the next state, return it."""
        self._write_actions(actions)
        self._act_ready.release()
        self._env_ready.acquire()
        return self._state

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._state = None
        self._actions_buf = None
        if self._mem is not None:
            self._mem.close()
            self._mem = None
        self._env_ready.unlink()
        self._act_ready.unlink()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _init_layout(self) -> None:
        """Parse header, compute offsets, and create shared-memory-backed numpy views."""
        header = _Header.from_buffer_copy(self._mem[:_HEADER_SIZE])
        self.num_envs        = header.num_envs
        self.visual_obs_size = header.visual_obs_size
        self.visual_width    = header.visual_width
        self.visual_height   = header.visual_height
        self.visual_channels = header.visual_channels
        self.scalar_obs_size = header.scalar_obs_size
        self.action_size     = header.action_size

        visual_obs_off  = _HEADER_SIZE
        scalar_obs_off  = visual_obs_off + self.num_envs * self.visual_obs_size
        rewards_off     = scalar_obs_off + self.num_envs * self.scalar_obs_size
        done_flags_off  = rewards_off    + self.num_envs * 4  # float32
        actions_off     = done_flags_off + self.num_envs      # uint8

        view = lambda offset, dtype, shape: np.ndarray(shape=shape, dtype=dtype, buffer=self._mem, offset=offset)

        self._state = EnvState(
            visual_obs=view(visual_obs_off, np.uint8,   (self.num_envs, self.visual_height, self.visual_width, self.visual_channels)),
            scalar_obs=view(scalar_obs_off, np.uint8,   (self.num_envs, self.scalar_obs_size)),
            rewards=   view(rewards_off,    np.float32, (self.num_envs,)),
            dones=     view(done_flags_off, np.uint8,   (self.num_envs,)),
        )
        self._actions_buf = view(actions_off, np.uint8, (self.num_envs, self.action_size))

    def _write_actions(self, actions: np.ndarray) -> None:
        expected_shape = (self.num_envs, self.action_size)
        if actions.shape != expected_shape:
            raise ValueError(f"actions shape {actions.shape} != expected {expected_shape}")
        if actions.dtype != np.uint8:
            raise TypeError(f"actions dtype {actions.dtype} != expected uint8")
        self._actions_buf[:] = actions