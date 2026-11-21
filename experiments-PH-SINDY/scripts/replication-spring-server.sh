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
DEFAULT_RESULTS="$ROOT/experiments-PH-SINDY/results/spring/default_20261112_223405"
OUT="$ROOT/experiments-PH-SINDY/results/spring/original_loss_$STAMP"
TRACKING_DIR="$ROOT/experiments-PH-SINDY/results/tracking"

mkdir -p "$OUT/logs" "$TRACKING_DIR" # create directory 
export MLFLOW_TRACKING_URI="sqlite:///$TRACKING_DIR/mlflow.db"


# default ph model 
for N in 4 8 16 32 64; do
DATASET="$DEFAULT_RESULTS/data/spring_${N}train_100val_seed1000.npy"
if [[ ! -f "$DATASET" ]]; then
  echo "Missing dataset: $DATASET" >&2
  exit 1
fi

"$PYTHON" -u main.py \
  --name spring \
  --num_trajectories "$N" --num_val_trajectories 100 \
  --time 10 --steps 1000 \
  --val_time 100 --val_steps 10000 \
  --hidden_dim 16 --depth 3 \
  --epochs 4000 --batch_size 256 --lr 0.001 \
  --J default --R default --G mlp --grad_H gradient \
  --output-weight 0.25 --criterion normalized_mse \
  --seed 1000 --device cuda \
  --dataset "$DATASET" \
  --experiment spring_original_loss \
  --run_name "default_n${N}_seed1000" \
  --no-forecast \
  2>&1 | tee "$OUT/logs/default_n${N}_seed1000.log"
done

# prior model - J and G are learned constant matrices and the Hamiltonian is quadratic
for N in 4 8 16 32 64; do
DATASET="$DEFAULT_RESULTS/data/spring_${N}train_100val_seed1000.npy"
if [[ ! -f "$DATASET" ]]; then
  echo "Missing dataset: $DATASET" >&2
  exit 1
fi

  "$PYTHON" -u main.py \
    --name spring \
    --num_trajectories "$N" --num_val_trajectories 100 \
    --time 10 --steps 1000 \
    --val_time 100 --val_steps 10000 \
    --hidden_dim 16 --depth 3 \
    --epochs 4000 --batch_size 256 --lr 0.001 \
    --J linear --R default --G linear --grad_H linear \
    --no-normalize-x \
    --output-weight 0.25 --criterion normalized_mse \
    --seed 1000 --device cuda \
    --dataset "$DATASET" \
    --experiment spring_original_loss \
    --run_name "prior_n${N}_seed1000" \
    --no-forecast \
    2>&1 | tee "$OUT/logs/prior_n${N}_seed1000.log"
done