"""
Warmup profiler for VGA — records per-step throughput from step 0 so you can
identify when performance stabilises and choose a warmup value for
benchmark_scaling.py.

Run this at a few different --num-envs values; plot the output with
plot_scaling.py (record_type == "warmup_profile") to pick a cutoff.

Usage:
    python benchmarks/profile_warmup.py \
        --godot /path/to/godot --project /path/to/project \
        --num-envs 1 4 16 \
        --steps 500 \
        --output warmup_profile.jsonl
"""

import argparse
import json
import subprocess
import time
import uuid
from datetime import datetime, timezone

import numpy as np

from py_vga.ipc_client import IPCClient


# ── Metadata helpers (duplicated from benchmark_scaling to keep scripts self-contained) ──

def _get_git_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _get_gpu_name() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        lines = result.stdout.strip().splitlines()
        return lines[0] if lines else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


# ── Core ──────────────────────────────────────────────────────────────────────

def _run_profile(
    *,
    ipc_name: str,
    godot_binary: str,
    project_path: str | None,
    num_envs: int,
    obs_width: int | None,
    obs_height: int | None,
    obs_channels: int | None,
    steps: int,
) -> dict:
    with IPCClient(ipc_name) as client:
        client.launch_godot(
            godot_binary,
            project_path=project_path,
            num_envs=num_envs,
            obs_width=obs_width,
            obs_height=obs_height,
            obs_channels=obs_channels,
        )

        print(f"    Waiting for Godot ({num_envs} envs)...")
        state = client.connect()

        n = client.num_envs
        actions = np.zeros((n, client.action_size), dtype=np.uint8)

        latencies: list[float] = []
        t_start = time.perf_counter()

        for i in range(steps):
            t0 = time.perf_counter()
            state = client.step(actions)
            latencies.append(time.perf_counter() - t0)

            if (i + 1) % 100 == 0 or (i + 1) == steps:
                recent = latencies[-20:]
                throughput = n / np.mean(recent)
                print(f"    step {i+1:>5}/{steps}  "
                      f"throughput (last 20): {throughput:.1f} env·steps/s")

        total_time_s = time.perf_counter() - t_start

        return {
            "num_envs":         client.num_envs,
            "obs_width":        client.visual_width,
            "obs_height":       client.visual_height,
            "obs_channels":     client.visual_channels,
            "total_frames":     steps,
            "total_time_s":     total_time_s,
            "step_latencies_s": latencies,
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="VGA warmup profiler — identify when throughput stabilises",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--godot",    required=True, metavar="PATH",
                   help="Path to the Godot binary")
    p.add_argument("--project",  default=None, metavar="PATH",
                   help="Path to the Godot project directory (omit for exported binary)")
    p.add_argument("--num-envs", type=int, nargs="+", required=True, metavar="N",
                   help="Environment counts to profile (each gets its own run)")
    p.add_argument("--steps",    type=int, default=500,
                   help="Total steps to record per run (no warmup discarded)")
    p.add_argument("--output",   default="warmup_profile.jsonl",
                   help="Output file (appended to if it already exists)")
    p.add_argument("--name",     default="vga",
                   help="IPC name passed to Godot")
    p.add_argument("--obs-width",  type=int, default=None)
    p.add_argument("--obs-channels", type=int, default=None)
    p.add_argument("--obs-height", type=int, default=None)
    return p


def main() -> None:
    args = _build_parser().parse_args()

    git_hash = _get_git_hash()
    gpu      = _get_gpu_name()
    timestamp = datetime.now(timezone.utc).isoformat()

    print(f"Git : {git_hash or 'unknown'}")
    print(f"GPU : {gpu or 'unknown'}")
    print(f"Out : {args.output}")
    print(f"Profiling warmup over {args.steps} steps for num_envs={args.num_envs}")
    print()

    with open(args.output, "a") as f:
        for num_envs in args.num_envs:
            print(f"[num_envs={num_envs}]")

            data = _run_profile(
                ipc_name=args.name,
                godot_binary=args.godot,
                project_path=args.project,
                num_envs=num_envs,
                obs_width=args.obs_width,
                obs_height=args.obs_height,
                obs_channels=args.obs_channels,
                steps=args.steps,
            )

            record = {
                "record_type": "warmup_profile",
                "run_id":      str(uuid.uuid4()),
                "timestamp":   timestamp,
                "git_hash":    git_hash,
                "gpu":         gpu,
                **data,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

            lats = np.array(data["step_latencies_s"])
            gap_ms = abs(data["total_time_s"] - lats.sum()) * 1000
            print(f"    total_time_s={data['total_time_s']:.4f}  "
                  f"sum(latencies)={lats.sum():.4f}  gap={gap_ms:.2f} ms")
            print()

    print(f"Done. Results appended to {args.output}")


if __name__ == "__main__":
    main()
