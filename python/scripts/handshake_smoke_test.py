"""
Minimal test script: connects to a running Godot HPA environment, sends zero
actions every step, and prints rewards and done flags.

Usage:
    python scripts/handshake_smoke_test.py [name]

The name must match the one configured in the Godot HPAMasterNode (default: hpa).
"""

import sys
import numpy as np
from godot_hpa.ipc_client import IPCClient

name = sys.argv[1] if len(sys.argv) > 1 else "hpa"

print(f"Waiting for Godot environment '{name}'...")

with IPCClient(name) as client:
    state = client.connect()

    print(f"Connected — {client.num_envs} envs | "
          f"visual_obs: {client.visual_obs_size}B | "
          f"scalar_obs: {client.scalar_obs_size}B | "
          f"action: {client.action_size}B")

    actions = np.zeros((client.num_envs, client.action_size), dtype=np.uint8)
    step = 0

    try:
        while True:
            print(f"[step {step:>6}]  rewards: {state.rewards}  terminated: {state.terminated}  truncated: {state.truncated}")
            state = client.step(actions)
            step += 1
    except KeyboardInterrupt:
        print(f"\nStopped after {step} steps.")
