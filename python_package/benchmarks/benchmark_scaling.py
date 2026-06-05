"""
Scaling benchmark for VGA — measures throughput as num_envs increases.

Launches and terminates a fresh Godot process for each run. Results are
appended to a JSONL file (one record per run) so repeated invocations
accumulate data. Use plot_scaling.py to visualise the results.

Usage:
    python benchmarks/benchmark_scaling.py \
        --godot /path/to/godot --project /path/to/project \
        --num-envs 1 2 4 8 16 32 \
        --steps 1000 --warmup 100 --runs 3 \
        --output results.jsonl
"""

import argparse
import json
import subprocess
import time
import uuid
from datetime import datetime, timezone

import numpy as np

from py_vga.ipc_client import IPCClient


# ── Metadata helpers ───────────────────────────────────────────────────────────

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


# ── Core benchmark ─────────────────────────────────────────────────────────────

def _run_single(
    *,
    ipc_name: str,
    godot_binary: str,
    project_path: str | None,
    num_envs: int,
    obs_width: int | None,
    obs_height: int | None,
    warmup_steps: int,
    measured_steps: int,
) -> dict:
    """
    Launch Godot, run warmup + measurement, shut Godot down, return raw data.
    Godot is always terminated before this function returns.
    """
    with IPCClient(ipc_name) as client:
        client.launch_godot(
            godot_binary,
            project_path=project_path,
            num_envs=num_envs,
            obs_width=obs_width,
            obs_height=obs_height,
        )

        print(f"    Waiting for Godot ({num_envs} envs)...")
        state = client.connect()

        n = client.num_envs
        actions = np.zeros((n, client.action_size), dtype=np.uint8)

        for _ in range(warmup_steps):
            state = client.step(actions)

        latencies: list[float] = []
        t_start = time.perf_counter()

        for i in range(measured_steps):
            t0 = time.perf_counter()
            state = client.step(actions)
            latencies.append(time.perf_counter() - t0)

            if (i + 1) % 200 == 0 or (i + 1) == measured_steps:
                mean_ms = np.mean(latencies) * 1000
                throughput = n / np.mean(latencies)
                print(f"    step {i+1:>5}/{measured_steps}  "
                      f"mean latency: {mean_ms:.2f} ms  "
                      f"throughput: {throughput:.1f} env·steps/s")

        total_time_s = time.perf_counter() - t_start

        return {
            # actual values from the shared memory header (may differ from args)
            "num_envs":     client.num_envs,
            "obs_width":    client.visual_width,
            "obs_height":   client.visual_height,
            "obs_channels": client.visual_channels,
            "warmup_steps":       warmup_steps,
            "total_frames":       measured_steps,
            "total_time_s":       total_time_s,
            "step_latencies_s":   latencies,
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="VGA scaling benchmark — data collection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--godot",    required=True, metavar="PATH",
                   help="Path to the Godot binary")
    p.add_argument("--project",  default=None, metavar="PATH",
                   help="Path to the Godot project directory (omit for exported binary)")
    p.add_argument("--num-envs", type=int, nargs="+", required=True, metavar="N",
                   help="Environment counts to sweep")
    p.add_argument("--steps",   type=int, default=1000,
                   help="Measured steps per run")
    p.add_argument("--warmup",  type=int, default=500,
                   help="Warmup steps discarded before measurement")
    p.add_argument("--runs",    type=int, default=1,
                   help="Number of independent runs per env count")
    p.add_argument("--output",  default="scaling_results.jsonl",
                   help="Output file (appended to if it already exists)")
    p.add_argument("--name",    default="vga",
                   help="IPC name passed to Godot")
    p.add_argument("--obs-width",  type=int, default=None)
    p.add_argument("--obs-height", type=int, default=None)
    return p


def main() -> None:
    args = _build_parser().parse_args()

    git_hash = _get_git_hash()
    gpu      = _get_gpu_name()
    # One timestamp for the whole invocation so a sweep appears as a group.
    timestamp = datetime.now(timezone.utc).isoformat()

    print(f"Git : {git_hash or 'unknown'}")
    print(f"GPU : {gpu or 'unknown'}")
    print(f"Out : {args.output}")
    print(f"Sweep: {args.num_envs}  runs/count: {args.runs}  "
          f"steps: {args.warmup} warmup + {args.steps} measured")
    print()

    with open(args.output, "a") as f:
        for num_envs in args.num_envs:
            for run_idx in range(args.runs):
                print(f"[num_envs={num_envs}  run {run_idx + 1}/{args.runs}]")

                data = _run_single(
                    ipc_name=args.name,
                    godot_binary=args.godot,
                    project_path=args.project,
                    num_envs=num_envs,
                    obs_width=args.obs_width,
                    obs_height=args.obs_height,
                    warmup_steps=args.warmup,
                    measured_steps=args.steps,
                )

                record = {
                    "run_id":    str(uuid.uuid4()),
                    "timestamp": timestamp,
                    "git_hash":  git_hash,
                    "gpu":       gpu,
                    **data,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()

                # Sanity check: sum of per-step latencies should closely match
                # total wall-clock time (any gap is Python loop overhead).
                lats = np.array(data["step_latencies_s"])
                sum_lat  = lats.sum()
                gap_ms   = abs(data["total_time_s"] - sum_lat) * 1000
                throughput = data["num_envs"] / lats.mean()
                print(f"    total_time_s={data['total_time_s']:.4f}  "
                      f"sum(latencies)={sum_lat:.4f}  "
                      f"gap={gap_ms:.2f} ms  "
                      f"throughput={throughput:.1f} env·steps/s")
                print()

    print(f"Done. Results appended to {args.output}")


if __name__ == "__main__":
    main()
