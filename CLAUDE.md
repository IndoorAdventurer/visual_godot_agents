# Visual Godot Agents

GDExtension plugin for vision-based RL environments: runs N parallel environment instances
in Godot, communicates observations/actions to Python via POSIX shared memory and semaphores.

## Build

```bash
cd godot_plugin
uv tool run scons [scons arguments...]
```

All `.cpp` files in `godot_plugin/src/` are compiled automatically. Generated files go to `godot_plugin/src/gen/`.

## Architecture (current)

**Simulation time is fully decoupled from real time.** Godot runs as fast as the CPU/GPU
allows with a fixed time delta — there is no vsync, no wall-clock pacing, no assumption
that one tick takes any particular amount of real time. `VGAMasterNode::_ready()` applies
the necessary engine settings (`physics_ticks_per_second`, `time_scale`, `max_fps`, etc.)
and disables the automatic render loop; `_physics_process` drives rendering manually via
`RenderingServer::force_draw()` before each IPC exchange.

**Environment logic must use `_physics_process`** (or physics-mode equivalents such as
physics-mode `Timer` nodes). `_process` receives a wall-clock delta scaled by `time_scale`
which is not a meaningful simulation time — `AnimationPlayer`, `Tween`, and `Timer` nodes
in default (idle) process mode will all expire immediately.

**C++ (GDExtension)**
- `VGAMasterNode` — root node; owns N SubViewports (one per simulated environment) and is
  responsible for the IPC exchanges with Python. IPC functionality is delegated to:
  - `IPCController` — orchestrates all IPC: owns `IPCPosix` and `IPCVisuals`, drives the
    exchange loop, and manages shared memory layout
    - `IPCPosix` — low-level POSIX shared memory + semaphores
    - `IPCVisuals` — GPU readback pipeline (compute shader → staging buffer → shared memory)
- `VGAAgentNode` — GDScript-overridable data gateway for an individual environment: collects observations,
  rewards and done flags; receives actions

Visual observations are read back from each SubViewport's GPU texture and written directly
into the shared memory visual block by `IPCVisuals::fetch_frame()`.

Only classes exposed as Godot nodes need `GDREGISTER_CLASS` in `godot_plugin/src/register_types.cpp` and
XML documentation in `godot_plugin/doc_classes/`. Internal C++ components need neither.

**Python** (`python_package/` — install with `uv sync` from that directory)
- `py_vga/` — Python package; `IPCClient` is the low-level IPC primitive
- `benchmarks/` — IPC-layer benchmarks (latency, scaling, warmup profiling)

**Example projects** (`example_projects/`)
- `roomba_demo/` — Roomba cleaning demo; `godot_project/` is the Godot project, `scripts/` holds PPO training (`clean_rl_ppo_test.py`, `train.sh`), interactive testing, and episode recording. The symlink `godot_project/addons/visual_godot_agents` points to `godot_plugin/addons/visual_godot_agents`.
- `ipc_test_env/` — Minimal environment for validating the IPC layer; `godot_project/` is gitignored (work in progress), `scripts/` holds `launch_smoke_test.py`, `inspect_autoreset.py`, and `inspect_startup.py`.

**Startup order**: Python must start first — it creates the semaphores and blocks on `env_ready`.
Godot then opens the semaphores, creates shared memory, and posts `env_ready`. Python opens the
shared memory after that post. No further handshake is needed.
