# RL Compute Shader Demo

A demo project to test the core functionality of our GDExtension library —
specifically compute shaders and GPU readback in an RL context.

## Environment

A roomba-like robot (a cylinder) that navigates a 5m x 5m plane and cleans up
dirt particles that live on the GPU.

The robot spawns in the centre of the plane. Dirt particles are initialised with
the following distribution (decided per episode):
- 50% chance: one concentrated Gaussian cluster (narrow variance)
- 50% chance: two Gaussian clusters, so the robot must navigate to one and then
  the other

The robot gets a reward for each particle it collects and a penalty for falling
off the plane. The episode truncates after 1500 steps; it terminates early when all
particles are collected or the robot falls off.

Observations: three visual channels — R: depth, G: dirt-particle mask, B: robot
trail (path traveled this episode). The camera is a 3rd-person follow cam at a
fixed offset behind and above the robot.

Action space: forward/backward, strafe left/right, turn left/right (3 continuous values, 12 bytes).

## GPU Particle System

Particle positions live entirely on the GPU in an SSBO after initialisation:
- **Init**: Gaussian positions are generated on the CPU and uploaded once to the
  SSBO at episode start.
- **Step**: a compute shader runs each physics tick. It moves particles toward
  the robot when within suction range, and marks collected particles dead
  (alive flag w = 0) so they are skipped cheaply.
- **Render**: the compute shader writes transforms directly into the MultiMesh
  GPU buffer — no CPU round-trip.
- **Reward readback**: an atomic counter in the SSBO tracks particles collected
  this step. It is read back via `buffer_get_data` in the `post_draw` hook on
  VGAAgentNode, after the draw commands have flushed.

## Scene Structure

Three scenes:

```
robot.tscn
  CharacterBody3D        (RoombaRobot.gd — movement, reset)
  ├── CollisionShape3D
  ├── MeshInstance3D     (cylinder)
  ├── Camera3D           (fixed local offset: behind + above)
  └── VGAAgentNode       (RoombaAgent.gd — obs/reward/action overrides)

world.tscn
  Node3D                 (World.gd)
  ├── DirectionalLight3D
  ├── StaticBody3D       (floor)
  │   ├── CollisionShape3D
  │   └── MeshInstance3D
  └── DirtSystem         (DirtSystem.gd — SSBO, compute shader, render shader)

env.tscn
  Node3D                 (Env.gd — orchestration, reset coordination)
  ├── World              (instance of world.tscn)
  └── Robot              (instance of robot.tscn)
```

`Env.gd` passes `robot.global_position` to `DirtSystem.step()` each physics
tick. Reset flows downward: `Env.gd` calls `Robot.reset()` and `World.reset()`
directly. Termination events (robot fell off, all particles collected) flow
upward via signals from robot/world to `Env.gd`.

## Human vs Agent Mode

`env.tscn` works in both modes:
- **Human mode**: run `env.tscn` directly (no VGAMasterNode in the tree).
  `RoombaRobot.gd` checks at `_ready` whether an VGAMasterNode exists at the
  tree root; if not, it reads keyboard input each `_physics_process`.
- **Agent mode**: `env.tscn` is instantiated inside an VGAMasterNode SubViewport
  as usual. Actions arrive via `RoombaAgent._set_action()`.
