#!/usr/bin/env bash
# Description:

# Usage: 
# Author: Sabine
# Date: 2026-11-11

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$ROOT/experiments-PH-SINDY/results/matplotlib"
export XDG_CACHE_HOME="$ROOT/experiments-PH-SINDY/results/cache"
STAMP=$(date +%Y%m%d_%H%M%S)
SYSTEM=motor
OUT="$ROOT/experiments-PH-SINDY/results/$SYSTEM/original_loss_$STAMP"
TRACKING_DIR="$ROOT/experiments-PH-SINDY/results/tracking"

mkdir -p "$OUT/logs" "$TRACKING_DIR" # create directory 
export MLFLOW_TRACKING_URI="sqlite:///$TRACKING_DIR/mlflow.db"


# default ph model 
N=2
DATASET="$OUT/data/${SYSTEM}_${N}train_100val_seed1000.npy"
"$PYTHON" -u main.py \
  --name "$SYSTEM" \
  --num_trajectories "$N" --num_val_trajectories 100 \
  --time 10 --steps 1000 \
  --val_time 1 --val_steps 10 \
  --hidden_dim 16 --depth 3 \
  --epochs 10 --batch_size 256 --lr 0.001 \
  --J default --R default --G mlp --grad_H gradient \
  --output-weight 0.25 --criterion normalized_mse \
  --seed 1000 --device cuda \
  --dataset "$DATASET" \
  --experiment "${SYSTEM}_smke" \
  --run_name "default_n${N}_seed1000_ep10" \
  --no-forecast \
  2>&1 | tee "$OUT/logs/default_n${N}_seed1000.log"
