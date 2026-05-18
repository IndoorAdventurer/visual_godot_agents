"""
Smoke test for IPCClient.launch_godot(): launches Godot from Python, connects,
runs a fixed number of steps, then exits cleanly.

Usage:
    python scripts/launch_smoke_test.py --godot-binary <path> [options]

Options:
    --godot-binary   Path to Godot editor binary or standalone export (required)
    --project-path   Path to the Godot project directory (omit for standalone export)
    --num-envs       Override num_envs on HPAMasterNode (default: not set)
    --step-rate-hz   Override step_rate_hz on HPAMasterNode (default: not set)
    --steps          Number of steps to run before exiting (default: 20)
    --scenario       Custom scenario arg forwarded to Godot user_args (default: test)
"""

import argparse
import numpy as np
from godot_hpa.ipc_client import IPCClient

parser = argparse.ArgumentParser()
parser.add_argument("--godot-binary", required=True)
parser.add_argument("--project-path", default=None)
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--obs-width", type=int, default=None)
parser.add_argument("--obs-height", type=int, default=None)
parser.add_argument("--step-rate-hz", type=int, default=None)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--scenario", default="test")
args = parser.parse_args()

with IPCClient("hpa") as client:
    print("Launching Godot...")
    client.launch_godot(
        godot_binary=args.godot_binary,
        project_path=args.project_path,
        num_envs=args.num_envs,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
        step_rate_hz=args.step_rate_hz,
        extra_args={"scenario": args.scenario},
    )

    print("Waiting for Godot to connect...")
    state = client.connect()

    print(f"Connected — {client.num_envs} envs | "
          f"obs: {client.visual_width}x{client.visual_height}x{client.visual_channels} | "
          f"scalar_obs: {client.scalar_obs_size}B | "
          f"action: {client.action_size}B")

    actions = np.zeros((client.num_envs, client.action_size), dtype=np.uint8)

    for step in range(args.steps):
        print(f"[step {step:>4}]  rewards: {state.rewards}  dones: {state.dones}")
        state = client.step(actions)

    print(f"Done — {args.steps} steps completed, shutting down Godot.")
