"""
Capture the first N_FRAMES frames from a running Godot HPA environment and plot
them in a grid: rows = frames (0 = frame returned by connect()), columns = envs.

This lets you verify that visual data is valid from frame zero — i.e. that the
very first state returned by client.connect() already contains a real rendered
image rather than zeros or garbage.

Usage:
    python scripts/inspect_startup.py [name]

The name must match the one configured in the Godot HPAMasterNode (default: hpa).

--- Findings ---
Visual data is valid from frame 0: the GPU readback is working correctly and
client.connect() returns a real rendered image immediately.

However, randomisation applied in _ready() (e.g. random agent position/rotation)
does not take effect until around frame 9–10. Frames 0–8 show the environment in
its default (non-randomised) initialisation state. This is a Godot timing issue:
_ready() runs before the physics/scene tree has fully settled, so the randomised
state is overwritten or not yet reflected in the rendered output.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from godot_hpa.ipc_client import IPCClient

# How many frames to capture.
N_FRAMES = 15
# How many frames to skip (step through without capturing) before capturing.
SKIP_FRAMES = 0


def extract_display(frame: np.ndarray, channels: int) -> np.ndarray:
    """Convert (H, W, C) uint8 → shape suitable for imshow."""
    if channels == 4:
        return frame[:, :, :3]
    if channels == 1:
        return frame[:, :, 0]
    return frame


name = sys.argv[1] if len(sys.argv) > 1 else "hpa"

print(f"Waiting for Godot environment '{name}'...")

with IPCClient(name) as client:
    state = client.connect()

    W = client.visual_width
    H = client.visual_height
    C = client.visual_channels
    N = client.num_envs

    print(f"Connected — {N} envs | {W}×{H}×{C} visual obs")

    actions = np.zeros((N, client.action_size), dtype=np.uint8)

    if SKIP_FRAMES:
        print(f"Skipping {SKIP_FRAMES} frames...")
        for _ in range(SKIP_FRAMES):
            state = client.step(actions)

    print(f"Capturing {N_FRAMES} frames (starting at frame {SKIP_FRAMES})...")

    frames: list[list[np.ndarray]] = []  # frames[frame_idx][env_idx]

    for frame_idx in range(N_FRAMES):
        row = [extract_display(state.visual_obs[env_idx].copy(), C) for env_idx in range(N)]
        frames.append(row)
        if frame_idx < N_FRAMES - 1:
            state = client.step(actions)

print("Done capturing. Data sanity check:")
for fi, row in enumerate(frames):
    means = [f"{r.mean():.1f}" for r in row]
    identical = [np.array_equal(row[0], row[ei]) for ei in range(1, N)]
    print(f"  frame {fi}: means={means}  env0==envN: {identical}")

print("Rendering plot.")

fig, axes = plt.subplots(
    N_FRAMES, N,
    figsize=(N * 3, N_FRAMES * 3),
    squeeze=False,
)
fig.suptitle(
    f"First {N_FRAMES} frames — {N} envs ({W}×{H}, {C}ch)\n"
    "Row 0 = frame returned by connect()",
    fontsize=10,
)

cmap = "gray" if C == 1 else None

for frame_idx, row in enumerate(frames):
    for env_idx, img in enumerate(row):
        ax = axes[frame_idx][env_idx]
        ax.imshow(img, vmin=0, vmax=255, cmap=cmap, interpolation="nearest")
        ax.set_title(f"env {env_idx}  frame {frame_idx}", fontsize=7)
        ax.axis("off")

plt.tight_layout()
plt.show()
