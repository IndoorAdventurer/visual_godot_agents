#!/usr/bin/env bash
# Training launcher — mirrors what the Slurm script will do.
# On a Slurm cluster: replace the Xvfb block with the cluster's virtual
# display setup (e.g. vglrun / VirtualGL), add #SBATCH directives, and
# point GODOT_BINARY / PROJECT_PATH at the exported release binary.
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
GODOT_BINARY="/home/vincent/.local/bin/godot"
PROJECT_PATH="/home/vincent/Documents/projects/high_performance_godot_agents/godot-rl-compute-shader-demo"
NUM_ENVS=32
NUM_INSTANCES=4
OBS_WIDTH=64
OBS_HEIGHT=64
TOTAL_TIMESTEPS=10000000
RESUME_FROM=""          # path to .cleanrl_model, or empty to start fresh

# ── Performance mode ──────────────────────────────────────────────────────────
if command -v powerprofilesctl &>/dev/null; then
    powerprofilesctl set performance
    echo "Power profile: $(powerprofilesctl | grep -A1 'performance' | grep 'Driver' || echo 'performance set')"
fi

# ── Virtual display ───────────────────────────────────────────────────────────
# Godot needs an X display even when rendering into SubViewports only.
export DISPLAY=:99
Xvfb :99 -screen 0 8x8x24 &>/dev/null &
XVFB_PID=$!
echo "Xvfb started (pid $XVFB_PID)"

# Kill Xvfb when the script exits for any reason.
trap "echo 'Cleaning up Xvfb...'; kill $XVFB_PID 2>/dev/null || true" EXIT

# Give Xvfb a moment to initialise before Godot tries to connect.
sleep 1

# ── Training ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."   # run from python/ so uv picks up pyproject.toml

EXTRA_ARGS=""
if [[ -n "$RESUME_FROM" ]]; then
    EXTRA_ARGS="--resume-from $RESUME_FROM"
fi

uv run python scripts/clean_rl_ppo_test.py \
    --godot-binary "$GODOT_BINARY" \
    --project-path "$PROJECT_PATH" \
    --num-envs "$NUM_ENVS" \
    --num-instances "$NUM_INSTANCES" \
    --obs-width "$OBS_WIDTH" \
    --obs-height "$OBS_HEIGHT" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --save-model \
    $EXTRA_ARGS
