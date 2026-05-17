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
allows with a fixed time delta — there is no vsync, no wall-clock pacing, no assumption
that one tick takes any particular amount of real time. `HPAMasterNode::_ready()` applies
the necessary engine settings (`physics_ticks_per_second`, `time_scale`, `max_fps`, etc.)
and disables the automatic render loop; `_physics_process` drives rendering manually via
`RenderingServer::force_draw()` before each IPC exchange.

**Environment logic must use `_physics_process`** (or physics-mode equivalents such as
physics-mode `Timer` nodes). `_process` receives a wall-clock delta scaled by `time_scale`
which is not a meaningful simulation time — `AnimationPlayer`, `Tween`, and `Timer` nodes
in default (idle) process mode will all expire immediately.

**C++ (GDExtension)**
- `HPAMasterNode` — root node; owns N SubViewports (one per simulated environment) and is
  responsible for the IPC exchanges with Python. IPC functionality is delegated to:
  - `IPCController` — orchestrates all IPC: owns `IPCPosix` and `IPCVisuals`, drives the
    exchange loop, and manages shared memory layout
    - `IPCPosix` — low-level POSIX shared memory + semaphores
    - `IPCVisuals` — GPU readback pipeline (compute shader → staging buffer → shared memory)
- `HPAAgentNode` — GDScript-overridable data gateway for an individual environment: collects observations,
  rewards and done flags; receives actions

Visual observations are read back from each SubViewport's GPU texture and written directly
into the shared memory visual block by `IPCVisuals::fetch_frame()`.

Only classes exposed as Godot nodes need `GDREGISTER_CLASS` in `src/register_types.cpp` and
XML documentation in `doc_classes/`. Internal C++ components need neither.

**Python** (`python/` — install with `uv sync` from that directory)
- `godot_hpa/` — Python package; `IPCClient` is the low-level IPC primitive
- `scripts/` — utility and test scripts

**Startup order**: Python must start first — it creates the semaphores and blocks on `env_ready`.
Godot then opens the semaphores, creates shared memory, and posts `env_ready`. Python opens the
shared memory after that post. No further handshake is needed.
