"""
Benchmark script for the HPA event loop.

Measures per-step wall-clock latency with a no-op policy and reports a full
distribution summary. Run before and after the event loop decoupling
implementation to quantify the speedup.

Also validates that visual observations are actually changing between steps
(i.e. the render pipeline is live), and optionally injects a Python-side stall
to verify Godot resumes cleanly with no catch-up burst.

Usage:
    python benchmarks/benchmark_ipc_loop.py [--name NAME] [--steps N] [--warmup N]
                                            [--stall-after N --stall-seconds S]

Example — baseline before implementation:
    python benchmarks/benchmark_ipc_loop.py --steps 500

Example — stall robustness test:
    python benchmarks/benchmark_ipc_loop.py --steps 200 --stall-after 50 --stall-seconds 5
"""

import argparse
import time
import sys
import numpy as np
from godot_hpa.ipc_client import IPCClient


# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="HPA event loop benchmark")
parser.add_argument("--name",          default="hpa",  help="IPC name (default: hpa)")
parser.add_argument("--steps",         type=int, default=500,
                    help="Number of steps to measure (default: 500)")
parser.add_argument("--warmup",        type=int, default=10,
                    help="Warmup steps excluded from statistics (default: 10)")
parser.add_argument("--stall-after",   type=int, default=None,
                    metavar="N", help="Inject a stall after this many steps")
parser.add_argument("--stall-seconds", type=float, default=5.0,
                    metavar="S", help="Duration of the injected stall in seconds (default: 5)")
args = parser.parse_args()

PERCENTILES = [1, 5, 25, 50, 75, 95, 99]
HISTOGRAM_BINS = 16
HISTOGRAM_WIDTH = 50  # characters


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_ms(seconds: float) -> str:
    return f"{seconds * 1000:.3f} ms"


def print_histogram(latencies_ms: np.ndarray) -> None:
    lo, hi = latencies_ms.min(), latencies_ms.max()
    edges = np.linspace(lo, hi, HISTOGRAM_BINS + 1)
    counts, _ = np.histogram(latencies_ms, bins=edges)
    max_count = counts.max()

    print()
    for i, count in enumerate(counts):
        bar_len = int(HISTOGRAM_WIDTH * count / max_count) if max_count > 0 else 0
        bar = "█" * bar_len
        label = f"  {edges[i]:8.3f} – {edges[i+1]:8.3f} ms"
        print(f"{label}  {bar}  ({count})")
    print()


def print_summary(latencies_s: np.ndarray, stall_step: int | None,
                  stall_duration: float | None, first_obs: np.ndarray,
                  last_obs: np.ndarray) -> None:
    lat_ms = latencies_s * 1000
    steps_per_sec = 1.0 / latencies_s.mean()
    pct = np.percentile(lat_ms, PERCENTILES)

    print("\n" + "═" * 60)
    print(f"  Steps measured : {len(lat_ms)}")
    print(f"  Steps / second : {steps_per_sec:.1f}")
    print()
    print(f"  Mean   : {lat_ms.mean():.3f} ms")
    print(f"  Std    : {lat_ms.std():.3f} ms")
    print(f"  Min    : {lat_ms.min():.3f} ms")
    for p, v in zip(PERCENTILES, pct):
        print(f"  p{p:<3}  : {v:.3f} ms")
    print(f"  Max    : {lat_ms.max():.3f} ms")

    print()
    print("  Latency distribution (ms):")
    print_histogram(lat_ms)

    # Pixel data sanity check: first and last frame should differ.
    pixel_change = np.abs(first_obs.astype(np.int16) - last_obs.astype(np.int16)).mean()
    status = "OK" if pixel_change > 0.5 else "WARN — frames may be identical"
    print(f"  Visual obs change (first→last) : mean |Δpx| = {pixel_change:.2f}  [{status}]")

    if stall_step is not None:
        # The step after the stall should not have a dramatically larger latency
        # than the surrounding window (catch-up burst would show up here).
        stall_idx = stall_step - 1  # 0-based index into measured latencies
        if 0 <= stall_idx < len(lat_ms):
            post_stall_lat = lat_ms[stall_idx]
            baseline_p99 = pct[PERCENTILES.index(99)]
            burst = post_stall_lat > baseline_p99 * 2
            status = "WARN — possible catch-up burst!" if burst else "OK"
            print(f"  Post-stall latency ({stall_duration}s stall) : "
                  f"{post_stall_lat:.3f} ms  [{status}]")

    print("═" * 60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

print(f"Waiting for Godot environment '{args.name}'...")

with IPCClient(args.name) as client:
    state = client.connect()
    n = client.num_envs
    print(f"Connected — {n} envs | "
          f"{client.visual_width}×{client.visual_height}×{client.visual_channels} visual obs | "
          f"{client.scalar_obs_size}B scalar obs | "
          f"{client.action_size}B actions")

    actions = np.zeros((n, client.action_size), dtype=np.uint8)
    latencies: list[float] = []
    stall_injected = False
    first_obs: np.ndarray | None = None

    total = args.warmup + args.steps
    print(f"Running {args.warmup} warmup + {args.steps} measured steps...")

    for i in range(total):
        # Stall injection: sleep *before* posting actions so Godot blocks in
        # sem_wait(act_ready) for the full duration. This is the robustness
        # scenario described in the design doc.
        if args.stall_after is not None and i == args.warmup + args.stall_after and not stall_injected:
            print(f"  [step {i}] Injecting {args.stall_seconds}s stall...")
            time.sleep(args.stall_seconds)
            stall_injected = True
            print(f"  [step {i}] Resuming.")

        t0 = time.perf_counter()
        state = client.step(actions)
        elapsed = time.perf_counter() - t0

        if i == args.warmup:
            first_obs = state.visual_obs.copy()

        if i >= args.warmup:
            latencies.append(elapsed)
            measured = i - args.warmup + 1
            if measured % 100 == 0 or measured == args.steps:
                running_mean_ms = np.mean(latencies) * 1000
                print(f"  step {measured:>5}/{args.steps}   "
                      f"running mean: {running_mean_ms:.2f} ms")

    last_obs = state.visual_obs.copy()
    stall_step = args.stall_after if stall_injected else None

    print_summary(
        np.array(latencies),
        stall_step=stall_step,
        stall_duration=args.stall_seconds if stall_injected else None,
        first_obs=first_obs,
        last_obs=last_obs,
    )
