"""
Plotting script for HPA scaling benchmarks.

Reads JSONL files produced by benchmark_scaling.py and profile_warmup.py and
generates three figures:

  warmup_profile.png     — throughput over step index (one line per num_envs),
                           to identify when performance stabilises
  scaling_throughput.png — total throughput (num_envs × steps/s) vs num_envs,
                           with 95 % CI across runs
  scaling_latency.png    — step latency percentiles (p50 / p95 / p99) vs num_envs

Each plot is a standalone function that accepts a DataFrame and returns a
Figure — copy individual functions into a Jupyter notebook cell as needed.

Usage:
    python benchmarks/plot_scaling.py --input results.jsonl [warmup_profile.jsonl ...]
    python benchmarks/plot_scaling.py --input results.jsonl --show
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(paths: list[str | Path]) -> pd.DataFrame:
    """Load one or more JSONL files into a single DataFrame."""
    records = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    df = pd.DataFrame(records)
    # Normalise: records from benchmark_scaling.py have no record_type field.
    if "record_type" not in df.columns:
        df["record_type"] = "scaling"
    else:
        df["record_type"] = df["record_type"].fillna("scaling")

    # Convert latency lists to numpy arrays for convenience.
    df["step_latencies_s"] = df["step_latencies_s"].apply(np.array)
    return df


# ── Per-run summary ───────────────────────────────────────────────────────────

def _per_run_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for each scaling run row.
    Returns one row per run with derived columns added.
    """
    def _stats(row):
        lats = row["step_latencies_s"]
        mean_lat = lats.mean()
        return pd.Series({
            "mean_latency_s": mean_lat,
            "std_latency_s":  lats.std(),
            "p50_latency_s":  np.percentile(lats, 50),
            "p95_latency_s":  np.percentile(lats, 95),
            "p99_latency_s":  np.percentile(lats, 99),
            "throughput":     row["num_envs"] / mean_lat,
        })

    stats = df.apply(_stats, axis=1)
    return pd.concat([df.drop(columns=["step_latencies_s"]), stats], axis=1)


def _aggregate_by_envs(per_run: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-run stats by num_envs.
    Returns one row per unique num_envs with mean, std, and 95 % CI.
    """
    def _agg(g):
        n = len(g)
        ci_factor = 1.96 / np.sqrt(n) if n > 1 else float("nan")
        return pd.Series({
            "n_runs":           n,
            "mean_throughput":  g["throughput"].mean(),
            "std_throughput":   g["throughput"].std() if n > 1 else float("nan"),
            "ci95_throughput":  g["throughput"].std() * ci_factor if n > 1 else float("nan"),
            "mean_latency_ms":  g["mean_latency_s"].mean() * 1000,
            "std_latency_ms":   g["mean_latency_s"].std() * 1000 if n > 1 else float("nan"),
            "p50_ms":           g["p50_latency_s"].mean() * 1000,
            "p95_ms":           g["p95_latency_s"].mean() * 1000,
            "p99_ms":           g["p99_latency_s"].mean() * 1000,
        })

    return per_run.groupby("num_envs").apply(_agg).reset_index()


# ── Plot: warmup profile ──────────────────────────────────────────────────────

def plot_warmup_profile(df: pd.DataFrame, rolling_window: int = 20) -> plt.Figure:
    """
    Throughput over step index for each num_envs in the warmup-profile records.
    A rolling mean smooths the noisy per-step signal.
    """
    warmup = df[df["record_type"] == "warmup_profile"].copy()
    if warmup.empty:
        raise ValueError("No warmup_profile records found in data.")

    fig, ax = plt.subplots(figsize=(9, 5))

    for _, row in warmup.sort_values("num_envs").iterrows():
        lats = row["step_latencies_s"]
        throughput = row["num_envs"] / lats
        steps = np.arange(len(throughput))

        # Raw (faint) + rolling mean (solid)
        series = pd.Series(throughput)
        rolled = series.rolling(rolling_window, min_periods=1).mean().to_numpy()

        label = f"{row['num_envs']} env{'s' if row['num_envs'] != 1 else ''}"
        (line,) = ax.plot(steps, rolled, linewidth=1.8, label=label)
        ax.plot(steps, throughput, color=line.get_color(), alpha=0.15, linewidth=0.6)

    ax.set_xlabel("Step index")
    ax.set_ylabel("Throughput (env·steps / s)")
    ax.set_title(f"Warmup profile — throughput over time\n"
                 f"(solid = {rolling_window}-step rolling mean)")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    return fig


# ── Plot: frametime profile ───────────────────────────────────────────────────

def plot_frametime_profile(
    df: pd.DataFrame,
    rolling_window: int = 20,
    steady_state_start: int = 1000,
) -> plt.Figure:
    """
    Per-step latency (ms) over step index for each num_envs in the
    warmup-profile records.

    Directly comparable to the frame-spacing visible in nsys: read the
    steady-state mean off the horizontal reference line and cross-check
    against the gap between successive vkWaitForFences completions in the
    nsys timeline.
    """
    warmup = df[df["record_type"] == "warmup_profile"].copy()
    if warmup.empty:
        raise ValueError("No warmup_profile records found in data.")

    fig, ax = plt.subplots(figsize=(9, 5))

    for _, row in warmup.sort_values("num_envs").iterrows():
        lats_ms = row["step_latencies_s"] * 1000
        steps = np.arange(len(lats_ms))

        # Raw (faint) + rolling mean (solid)
        rolled = pd.Series(lats_ms).rolling(rolling_window, min_periods=1).mean().to_numpy()

        label = f"{row['num_envs']} env{'s' if row['num_envs'] != 1 else ''}"
        (line,) = ax.plot(steps, rolled, linewidth=1.8, label=label)
        ax.plot(steps, lats_ms, color=line.get_color(), alpha=0.15, linewidth=0.6)

        # Horizontal reference line at the steady-state mean so the number is
        # easy to read off and compare against nsys.
        if len(lats_ms) > steady_state_start:
            steady_mean = lats_ms[steady_state_start:].mean()
            ax.axhline(steady_mean, color=line.get_color(), linestyle=":", linewidth=1.2)
            ax.text(
                len(lats_ms) - 1, steady_mean,
                f" {steady_mean:.2f} ms",
                color=line.get_color(),
                va="bottom", ha="right", fontsize=8,
            )

    if steady_state_start > 0:
        ax.axvline(steady_state_start, color="gray", linestyle="--",
                   linewidth=0.8, label=f"steady-state start ({steady_state_start})")

    ax.set_xlabel("Step index")
    ax.set_ylabel("Step latency (ms)")
    ax.set_title(f"Frametime profile — latency over time\n"
                 f"(solid = {rolling_window}-step rolling mean, "
                 f"dotted = steady-state mean)")
    ax.legend()
    fig.tight_layout()
    return fig


# ── Plot: throughput vs num_envs ───────────────────────────────────────────────

def plot_throughput_vs_envs(df: pd.DataFrame) -> plt.Figure:
    """
    Total throughput (num_envs × steps/s) vs num_envs with 95 % CI error bars.
    """
    scaling = df[df["record_type"] == "scaling"]
    if scaling.empty:
        raise ValueError("No scaling records found in data.")

    per_run = _per_run_stats(scaling)
    agg = _aggregate_by_envs(per_run)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        agg["num_envs"],
        agg["mean_throughput"],
        yerr=agg["ci95_throughput"].fillna(0),
        fmt="o-",
        capsize=4,
        linewidth=1.8,
        label="mean ± 95 % CI",
    )

    # Linear-scaling reference line from the single-env point.
    if len(agg) > 1:
        base = agg.iloc[0]
        ref_y = base["mean_throughput"] / base["num_envs"] * agg["num_envs"]
        ax.plot(agg["num_envs"], ref_y, "--", color="gray",
                linewidth=1.2, label="linear scaling")

    ax.set_xlabel("Number of environments")
    ax.set_ylabel("Throughput (env·steps / s)")
    ax.set_title("Scaling throughput vs number of environments")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    return fig


# ── Plot: latency percentiles vs num_envs ─────────────────────────────────────

def plot_latency_percentiles(df: pd.DataFrame) -> plt.Figure:
    """
    Per-step latency percentiles (p50, p95, p99) vs num_envs.
    Shaded band shows ± 1 std of the per-run mean latency across runs.
    """
    scaling = df[df["record_type"] == "scaling"]
    if scaling.empty:
        raise ValueError("No scaling records found in data.")

    per_run = _per_run_stats(scaling)
    agg = _aggregate_by_envs(per_run)

    fig, ax = plt.subplots(figsize=(8, 5))

    for col, label, ls in [
        ("p50_ms", "p50", "-"),
        ("p95_ms", "p95", "--"),
        ("p99_ms", "p99", ":"),
    ]:
        ax.plot(agg["num_envs"], agg[col], linestyle=ls, marker="o",
                linewidth=1.8, label=label)

    # Std band around the mean.
    ax.fill_between(
        agg["num_envs"],
        agg["mean_latency_ms"] - agg["std_latency_ms"].fillna(0),
        agg["mean_latency_ms"] + agg["std_latency_ms"].fillna(0),
        alpha=0.15,
        label="mean ± 1 std",
    )

    ax.set_xlabel("Number of environments")
    ax.set_ylabel("Step latency (ms)")
    ax.set_title("Step latency percentiles vs number of environments")
    ax.legend()
    fig.tight_layout()
    return fig


# ── Text summary ──────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    """Print a text table of aggregated scaling stats — useful for sharing results."""
    scaling = df[df["record_type"] == "scaling"]
    if scaling.empty:
        print("No scaling records found.")
        return

    git_hashes = df["git_hash"].dropna().unique()
    gpus = df["gpu"].dropna().unique()
    print(f"git : {', '.join(git_hashes) if len(git_hashes) else 'unknown'}")
    print(f"gpu : {', '.join(gpus) if len(gpus) else 'unknown'}")
    print()

    per_run = _per_run_stats(scaling)
    agg = _aggregate_by_envs(per_run)

    # Scaling efficiency relative to 1-env throughput-per-env.
    base_per_env = agg.iloc[0]["mean_throughput"] / agg.iloc[0]["num_envs"]
    agg["efficiency"] = (agg["mean_throughput"] / agg["num_envs"]) / base_per_env

    col_w = 10
    header = (
        f"{'envs':>{col_w}} {'runs':>{col_w}} "
        f"{'throughput':>{col_w}} {'±ci95':>{col_w}} "
        f"{'lat_mean':>{col_w}} {'lat_std':>{col_w}} "
        f"{'p50':>{col_w}} {'p95':>{col_w}} {'p99':>{col_w}} "
        f"{'efficiency':>{col_w}}"
    )
    print(header)
    print("-" * len(header))

    for _, row in agg.iterrows():
        ci   = f"±{row['ci95_throughput']:.0f}" if not np.isnan(row["ci95_throughput"]) else "n/a"
        lstd = f"{row['std_latency_ms']:.2f}" if not np.isnan(row["std_latency_ms"]) else "n/a"
        print(
            f"{int(row['num_envs']):>{col_w}} {int(row['n_runs']):>{col_w}} "
            f"{row['mean_throughput']:>{col_w}.1f} {ci:>{col_w}} "
            f"{row['mean_latency_ms']:>{col_w}.2f} {lstd:>{col_w}} "
            f"{row['p50_ms']:>{col_w}.2f} {row['p95_ms']:>{col_w}.2f} {row['p99_ms']:>{col_w}.2f} "
            f"{row['efficiency']:>{col_w}.2%}"
        )
    print()
    print("throughput = env·steps/s   latencies in ms   efficiency = per-env throughput vs 1-env baseline")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot HPA scaling benchmark results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", nargs="+", required=True, metavar="FILE",
                   help="One or more JSONL files to load")
    p.add_argument("--output-dir", default=".", metavar="DIR",
                   help="Directory for saved figures")
    p.add_argument("--show", action="store_true",
                   help="Show interactive plots instead of saving to files")
    p.add_argument("--summary", action="store_true",
                   help="Print a text summary table instead of plotting")
    p.add_argument("--rolling-window", type=int, default=20, metavar="N",
                   help="Rolling-mean window for the warmup/frametime profile plots")
    p.add_argument("--steady-state-start", type=int, default=1000, metavar="N",
                   help="Step index after which the frametime is considered steady-state "
                        "(used for the reference line in frametime_profile.png)")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    df = load_data(args.input)
    print(f"Loaded {len(df)} records  "
          f"({(df['record_type'] == 'scaling').sum()} scaling, "
          f"{(df['record_type'] == 'warmup_profile').sum()} warmup profile)")
    print()

    if args.summary:
        print_summary(df)
        return

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plots = [
        ("warmup_profile",      "warmup_profile.png",       "warmup_profile"),
        ("frametime_profile",   "frametime_profile.png",    "warmup_profile"),
        ("scaling_throughput",  "scaling_throughput.png",   "scaling"),
        ("scaling_latency",     "scaling_latency.png",      "scaling"),
    ]

    generators = {
        "warmup_profile":     lambda: plot_warmup_profile(df, args.rolling_window),
        "frametime_profile":  lambda: plot_frametime_profile(df, args.rolling_window, args.steady_state_start),
        "scaling_throughput": lambda: plot_throughput_vs_envs(df),
        "scaling_latency":    lambda: plot_latency_percentiles(df),
    }

    for name, filename, required_type in plots:
        has_data = (df["record_type"] == required_type).any()
        if not has_data:
            print(f"  skip {filename} — no {required_type!r} records")
            continue
        try:
            fig = generators[name]()
            if args.show:
                plt.show()
            else:
                dest = out / filename
                fig.savefig(dest, dpi=150)
                print(f"  saved {dest}")
            plt.close(fig)
        except Exception as exc:
            print(f"  error generating {filename}: {exc}")


if __name__ == "__main__":
    main()
