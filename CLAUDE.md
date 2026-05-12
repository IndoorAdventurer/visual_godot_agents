# High Performance Godot Agents

GDExtension plugin for vision-based RL environments: runs N parallel environment instances
in Godot, communicates observations/actions to Python via POSIX shared memory and semaphores.

## Build

```bash
uv tool run scons [scons arguments...]
```

All `.cpp` files in `src/` are compiled automatically. Generated files go to `src/gen/`.

## Architecture (current)

**Simulation time is fully decoupled from real time.** Godot runs as fast as the CPU/GPU
allows with a fixed time delta — there is no vsync, no wall-clock pacing, no
assumption that one tick takes any particular amount of real time. How to implement
that is still not decided, so for the time being we should make as little assumptions
about the event loop as possible.

**C++ (GDExtension)**
- `HPAMasterNode` — root node; owns N SubViewports (one per simulated environment) and is
  responsible for the IPC exchanges with Python. IPC functionality is delegated to:
  - `IPCInterface` — manages the low-level POSIX shared memory + semaphores
  - `SharedMemoryLayout` — manages the data in shared memory: serialises env state and dispatches actions
- `HPAAgentNode` — GDScript-overridable data gateway for an individual environment: collects observations,
  rewards and done flags; receives actions

Visual observations are read back from each SubViewport's GPU texture and written directly
into the shared memory visual block (`SharedMemoryLayout::visual_obs_block_ptr`). The
readback mechanism is not yet decided.

Only classes exposed as Godot nodes need `GDREGISTER_CLASS` in `src/register_types.cpp` and
XML documentation in `doc_classes/`. Internal C++ components need neither.

**Python** (`python/` — install with `uv sync` from that directory)
- `godot_hpa/` — Python package; `IPCClient` is the low-level IPC primitive
- `scripts/` — utility and test scripts

**Startup order**: Python must start first — it creates the semaphores and blocks on `env_ready`.
Godot then opens the semaphores, creates shared memory, and posts `env_ready`. Python opens the
shared memory after that post. No further handshake is needed.
