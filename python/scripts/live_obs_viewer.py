"""
Live visualization of visual observations coming from a running Godot VGA environment.

Opens a single matplotlib window showing all N environments side-by-side,
updated in real time as Godot produces frames.

Usage:
    python scripts/live_obs_viewer.py [name]

The name must match the one configured in the Godot VGAMasterNode (default: vga).
"""

import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from py_vga.ipc_client import IPCClient


def _blank_frame(h: int, w: int, c: int) -> np.ndarray:
    """Return a black frame in the shape imshow expects."""
    if c == 1:
        return np.zeros((h, w), dtype=np.uint8)
    # RGB or RGBA → show as RGB
    return np.zeros((h, w, min(c, 3)), dtype=np.uint8)


name = sys.argv[1] if len(sys.argv) > 1 else "vga"

print(f"Waiting for Godot environment '{name}'...")

with IPCClient(name) as client:
    state = client.connect()

    W  = client.visual_width
    H  = client.visual_height
    C  = client.visual_channels
    N  = client.num_envs

    print(f"Connected — {N} envs | {W}×{H}×{C} visual obs | "
          f"scalar_obs: {client.scalar_obs_size}B | action: {client.action_size}B")

    # Arrange envs in a grid with at most 4 columns.
    cols = min(N, 4)
    rows = math.ceil(N / cols)

    plt.ion()
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), squeeze=False)
    fig.suptitle(f"VGA visual obs — {N} envs ({W}×{H}, {C}ch)")

    # Pre-create one imshow per env; hide any unused subplot slots.
    im_handles = []
    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        if idx < N:
            frame = _blank_frame(H, W, C)
            im = ax.imshow(frame, vmin=0, vmax=255,
                           cmap="gray" if C == 1 else None,
                           interpolation="nearest")
            ax.set_title(f"env {idx}", fontsize=8)
            ax.axis("off")
            im_handles.append(im)
        else:
            ax.set_visible(False)

    plt.tight_layout()

    actions = np.zeros((N, client.action_size), dtype=np.uint8)
    step = 0

    try:
        while True:
            for idx, im in enumerate(im_handles):
                frame = state.visual_obs[idx]
                # Drop alpha for display if RGBA; matplotlib doesn't need it.
                if C == 4:
                    frame = frame[:, :, :3]
                elif C == 1:
                    frame = frame[:, :, 0]
                im.set_data(frame)

            fig.canvas.flush_events()
            plt.pause(0.001)

            state = client.step(actions)
            step += 1

    except KeyboardInterrupt:
        print(f"\nStopped after {step} steps.")