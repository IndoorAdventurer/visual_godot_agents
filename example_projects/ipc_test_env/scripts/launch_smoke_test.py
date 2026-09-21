"""
Smoke test for IPCClient.launch_godot(): launches Godot from Python, connects,
runs a fixed number of steps, then exits cleanly.

Usage:
    python scripts/launch_smoke_test.py --godot-binary <path> [options]

Options:
    --godot-binary   Path to Godot editor binary or standalone export (required)
    --project-path   Path to the Godot project directory (omit for standalone export)
    --num-envs       Override num_envs on VGAMasterNode (default: not set)
    --step-rate-hz   Override step_rate_hz on VGAMasterNode (default: not set)
    --steps          Number of steps to run before exiting (default: 20)
    --scenario       Custom scenario arg forwarded to Godot user_args (default: test)
    --real-time      Enable real_time_mode on VGAMasterNode (default: off)
"""

import argparse
import time

import numpy as np
from py_vga.ipc_client import IPCClient

parser = argparse.ArgumentParser()
parser.add_argument("--godot-binary", required=True)
parser.add_argument("--project-path", default=None)
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--obs-width", type=int, default=None)
parser.add_argument("--obs-height", type=int, default=None)
parser.add_argument("--obs-channels", type=int, default=None)
parser.add_argument("--step-rate-hz", type=int, default=None)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--scenario", default="test")
parser.add_argument("--real-time", action="store_true")
args = parser.parse_args()

with IPCClient("vga") as client:
    print("Launching Godot...")
    client.launch_godot(
        godot_binary=args.godot_binary,
        project_path=args.project_path,
        num_envs=args.num_envs,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
        obs_channels=args.obs_channels,
        step_rate_hz=args.step_rate_hz,
        extra_args={"scenario": args.scenario,
                    "real_time_mode": "1" if args.real_time else "0"},
    )

    print("Waiting for Godot to connect...")
    state = client.connect()

    print(f"Connected — {client.num_envs} envs | "
          f"obs: {client.visual_width}x{client.visual_height}x{client.visual_channels} | "
          f"scalar_obs: {client.scalar_obs_size}B | "
          f"action: {client.action_size}B")

    actions = np.zeros((client.num_envs, client.action_size), dtype=np.uint8)

    start = time.perf_counter()
    for step in range(args.steps):
        print(f"[step {step:>4}]  rewards: {state.rewards}  terminated: {state.terminated}  truncated: {state.truncated}")
        state = client.step(actions)
    elapsed = time.perf_counter() - start

    print(f"Done — {args.steps} steps in {elapsed:.2f}s "
          f"({args.steps / elapsed:.1f} steps/s), shutting down Godot.")
