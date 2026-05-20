"""
Capture the first N_FRAMES frames from a running Godot HPA environment and plot
them in a grid: rows = frames (0 = frame returned by connect()), columns = envs.

Each environment receives a distinct cycling action sequence (env e gets action
(step + e) % N_COLORS) so all envs show different colours at the same frame —
this verifies the per-environment IPC path independently.

After capture, scalar observations are decoded as float32 and compared against
the expected value (sent_action + 10.0). Only mismatches are printed.

Usage:
    python scripts/inspect_startup.py [name]

The name must match the one configured in the Godot HPAMasterNode (default: hpa).
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from godot_hpa.ipc_client import IPCClient

# How many frames to capture (frame 0 = returned by connect()).
N_FRAMES = 15
# How many frames to step through without capturing before the main capture.
SKIP_FRAMES = 0

# Must match world.gd's COLORS array order.
COLOR_NAMES = ["RED", "GREEN", "BLUE", "YELLOW", "CYAN", "MAGENTA", "ORANGE", "WHITE"]
N_COLORS = len(COLOR_NAMES)


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

    if SKIP_FRAMES:
        print(f"Skipping {SKIP_FRAMES} frames...")
        for _ in range(SKIP_FRAMES):
            state = client.step(np.zeros((N, client.action_size), dtype=np.uint8))

    print(f"Capturing {N_FRAMES} frames...")

    frames: list[list[np.ndarray]] = []          # frames[frame_idx][env_idx]
    scalar_snapshots: list[np.ndarray] = []      # (N, scalar_obs_size) uint8 per frame
    # Action sent before each frame; None for frame 0 (connect, no action sent yet).
    action_log: list[np.ndarray | None] = []

    # Frame 0: state returned by connect() — last_action_byte is 0 in all envs.
    frames.append([extract_display(state.visual_obs[e].copy(), C) for e in range(N)])
    scalar_snapshots.append(state.scalar_obs.copy())
    action_log.append(None)

    for step in range(N_FRAMES - 1):
        # Give each env a different action by offsetting with env index.
        actions = np.array([[(step + e) % N_COLORS] for e in range(N)], dtype=np.uint8)
        action_log.append(actions[:, 0].copy())
        state = client.step(actions)
        frames.append([extract_display(state.visual_obs[e].copy(), C) for e in range(N)])
        scalar_snapshots.append(state.scalar_obs.copy())

# Scalar obs verification — only print on mismatch.
print("Scalar obs check (silent = all correct):")
all_ok = True
for fi in range(1, N_FRAMES):
    for e in range(N):
        sent = int(action_log[fi][e])
        expected = float(sent) + 10.0
        actual = np.frombuffer(scalar_snapshots[fi][e].tobytes(), dtype=np.float32)[0]
        if not np.isclose(actual, expected):
            print(f"  FAIL frame={fi} env={e}: sent={sent} expected={expected:.1f} got={actual:.6f}")
            all_ok = False
if all_ok:
    print("  All OK.")

print("Rendering plot.")

fig, axes = plt.subplots(
    N_FRAMES, N,
    figsize=(N * 3, N_FRAMES * 3),
    squeeze=False,
)
fig.suptitle(
    f"First {N_FRAMES} frames — {N} envs ({W}×{H}, {C}ch)\n"
    "Row 0 = connect() (no action). Subsequent rows: env e → action (step+e) % 8.",
    fontsize=10,
)

cmap = "gray" if C == 1 else None

for fi, row in enumerate(frames):
    for e, img in enumerate(row):
        ax = axes[fi][e]
        ax.imshow(img, vmin=0, vmax=255, cmap=cmap, interpolation="nearest")
        if action_log[fi] is not None:
            a = int(action_log[fi][e])
            label = f"env {e}  frame {fi}\naction={a} ({COLOR_NAMES[a]})"
        else:
            label = f"env {e}  frame {fi}\n(initial)"
        ax.text(0.5, 0.97, label, transform=ax.transAxes,
                fontsize=7, ha="center", va="top",
                color="white", bbox=dict(facecolor="black", alpha=0.55, pad=1))
        ax.axis("off")

plt.tight_layout()
plt.show()
