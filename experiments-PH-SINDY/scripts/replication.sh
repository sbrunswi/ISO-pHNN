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
OUT="$ROOT/experiments-PH-SINDY/results/$STAMP"
mkdir -p "$OUT"
export MLFLOW_TRACKING_URI="sqlite:///$OUT/mlflow.db"

# default ph model 
for N in 4 8 16 32 64; do
"$PYTHON" -u main.py \
  --name spring \
  --num_trajectories "$N" --num_val_trajectories 100 \
  --time 10 --steps 1000 \
  --val_time 100 --val_steps 10000 \
  --hidden_dim 16 --depth 3 \
  --epochs 10 --batch_size 256 --lr 0.001 \
  --J default --R default --G mlp --grad_H gradient \
  --output-weight 0.25 --criterion normalized_mse \
  --seed 1000 --device mps \
  --dataset "experiments-PH-SINDY/results/$STAMP/data/spring_${N}train_100val_seed1000.npy" \
  --experiment spring_reproduction \
  --run_name "default_n${N}_seed1000" \
  --no-forecast \
  2>&1 | tee "$OUT/default_n${N}_seed1000.log"
done

# # PRIOR MODEL: J and G are learned constant matrices and the Hamiltonian is quadratic
# for N in 4 8 16 32 64; do
#   "$PYTHON" -u main.py \
#     --name spring \
#     --num_trajectories "$N" --num_val_trajectories 100 \
#     --time 10 --steps 1000 \
#     --val_time 100 --val_steps 10000 \
#     --hidden_dim 16 --depth 3 \
#     --epochs 10 --batch_size 256 --lr 0.001 \
#     --J linear --R default --G linear --grad_H linear \
#     --no-normalize-x \
#     --output-weight 0.25 --criterion normalized_mse \
#     --seed 1000 --device mps \
#     --dataset experiments-PH-SINDY/results//$STAMP/data/spring_${N}train_100val_seed1000.npy \
#     --experiment spring_reproduction \
#     --run_name prior_seed1000 \
#     --no-forecast
#     2>&1 | tee "$OUT/default_n${N}_seed1000.log"
# done