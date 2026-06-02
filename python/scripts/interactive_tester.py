"""
Interactive step-by-step tester for the Roomba demo environment.

Displays visual observations via OpenCV and advances the simulation one step
at a time via keyboard input. The same action is broadcast to all environments.

Controls:
    W / S     — forward / backward
    A / D     — strafe left / right
    Q / E     — turn left / right
    Space     — no-op (zero action)
    ESC       — quit

Usage:
    python scripts/interactive_tester.py [name]

The name must match the one configured in the Godot HPAMasterNode (default: hpa).
"""

import math
import struct
import sys

import cv2
import numpy as np

from godot_hpa.ipc_client import IPCClient

MAX_COLS = 4

KEY_ACTIONS: dict[int, tuple[float, float, float]] = {
    ord("w"): ( 1.0,  0.0,  0.0),
    ord("s"): (-1.0,  0.0,  0.0),
    ord("a"): ( 0.0, -1.0,  0.0),
    ord("d"): ( 0.0,  1.0,  0.0),
    ord("q"): ( 0.0,  0.0, -1.0),
    ord("e"): ( 0.0,  0.0,  1.0),
}


def _make_actions(n: int, action_size: int, forward: float, strafe: float, turn: float) -> np.ndarray:
    raw = np.frombuffer(struct.pack("<fff", forward, strafe, turn), dtype=np.uint8)
    assert action_size == len(raw), f"action_size {action_size} != expected {len(raw)} bytes (3 floats)"
    actions = np.empty((n, action_size), dtype=np.uint8)
    actions[:] = raw
    return actions


def _to_bgr(frames: np.ndarray) -> np.ndarray:
    """Convert (N, H, W, C) uint8 obs to (N, H, W, 3) BGR for OpenCV."""
    c = frames.shape[-1]
    if c >= 3:
        bgr = np.ascontiguousarray(frames[:, :, :, [2, 1, 0]])  # RGBA/RGB → BGR
    else:
        bgr = np.ascontiguousarray(np.repeat(frames, 3, axis=-1))
    return bgr


def _tile(frames: np.ndarray, max_cols: int = MAX_COLS) -> np.ndarray:
    """Tile (N, H, W, 3) into a single display image."""
    n, h, w, _ = frames.shape
    cols = min(n, max_cols)
    rows = math.ceil(n / cols)
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = frame
    return canvas


name = sys.argv[1] if len(sys.argv) > 1 else "hpa"
print(f"Waiting for Godot environment '{name}'...")

with IPCClient(name) as client:
    state = client.connect()

    N = client.num_envs
    W = client.visual_width
    H = client.visual_height
    C = client.visual_channels
    print(f"Connected — {N} envs | {W}×{H}×{C} visual obs | action: {client.action_size}B")
    print("W/S=fwd/back  A/D=strafe  Q/E=turn  Space=no-op  ESC=quit")

    step = 0
    cols = min(N, MAX_COLS)

    cv2.namedWindow("HPA Demo — press key to step", cv2.WINDOW_NORMAL)

    while True:
        raw = state.visual_obs.copy()
        bgr = _to_bgr(raw)
        canvas = _tile(bgr)

        # Overlay reward + episode state on each env tile.
        for i in range(N):
            r, c = divmod(i, cols)
            x, y = c * W + 4, r * H + 16
            t, tr = int(state.terminated[i]), int(state.truncated[i])
            label = "TERM" if t else ("TRUNC" if tr else "RUN")
            text = f"r={state.rewards[i]:.3f}  {label}"
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("HPA Demo — press key to step", canvas)

        key = cv2.waitKey(0) & 0xFF

        if key == 27 or cv2.getWindowProperty("HPA Demo — press key to step", cv2.WND_PROP_VISIBLE) < 1:
            break

        forward, strafe, turn = KEY_ACTIONS.get(key, (0.0, 0.0, 0.0))
        actions = _make_actions(N, client.action_size, forward, strafe, turn)
        state = client.step(actions)
        step += 1

        print(
            f"step {step:4d} | "
            f"rewards: {np.array2string(state.rewards, precision=3, floatmode='fixed')} | "
            f"terminated: {state.terminated.tolist()} | "
            f"truncated: {state.truncated.tolist()}"
        )

cv2.destroyAllWindows()
print(f"\nStopped after {step} steps.")
