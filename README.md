# Visual Godot Agents
**A high-performance interface between Godot and Python for Reinforcement Learning (RL) using POSIX shared memory.**

## Overview
This library provides a low-latency bridge for RL applications. While inspired by the [Godot RL Agents Package](https://github.com/edbeeching/godot_rl_agents), this implementation is optimized for high-throughput use cases where standard networking protocols become a bottleneck.

### Key Use Case: Vision-Based RL
This library is specifically designed for environments with **large observation spaces**, such as high-resolution camera feeds or depth maps. By leveraging POSIX shared memory, it eliminates the serialization overhead typically found in socket-based communication.

> **Note:** As this relies on POSIX shared memory, this library is currently optimized for Linux/Unix-based environments.

## Repository structure

```
godot_plugin/       GDExtension C++ source and build system
python_package/     Python client package (py_vga) + benchmarks
example_projects/
  roomba_demo/      Roomba cleaning demo: Godot project + training scripts
  ipc_test_env/     Minimal IPC test environment
```

[//]: # (TODO: expand this into a proper user manual covering:)
[//]: # (- installation: copy addons/visual_godot_agents/ into your Godot project)
[//]: # (- usage: VGAMasterNode setup, SubViewport structure, Python startup order)
[//]: # (- VGAAgentNode virtual method reference with guaranteed call order per step:)
[//]: # (    1. _reset — called at start of exchange after a terminated/truncated episode)
[//]: # (    2. render + visual obs readback)
[//]: # (    3. _collect_scalar_obs)
[//]: # (    4. _get_reward)
[//]: # (    5. _get_episode_state)
[//]: # (    6. wait for Python actions)
[//]: # (    7. _apply_action)
[//]: # (- reward design notes: _get_reward is always called before _get_episode_state in the same step)
