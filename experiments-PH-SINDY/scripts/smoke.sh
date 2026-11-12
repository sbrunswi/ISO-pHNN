#!/usr/bin/env bash
# Small pipeline check; these settings are not a paper reproduction.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$ROOT/experiments-PH-SINDY/smoke/matplotlib"
export XDG_CACHE_HOME="$ROOT/experiments-PH-SINDY/smoke/cache"
# Keep smoke-test runs separate from existing experiment tracking.
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///$ROOT/experiments-s/smoke/mlflow.db}"
mkdir -p experiments-s/smoke

# A missing dataset is generated and saved; subsequent runs load it.
"$PYTHON" -u main.py \
  --name spring \
  --num_trajectories 2 --num_val_trajectories 2 \
  --time 1 --steps 100 --val_time 1 --val_steps 100 \
  --hidden_dim 16 --depth 3 --epochs 3 --batch_size 64 \
  --J default --R default --G mlp --grad_H gradient \
  --output-weight 0.25 --criterion normalized_mse \
  --seed 1000 --device mps --validation-frequency 1 \
  --dataset experiments-PH-SINDY/smoke/spring.npy \
  --experiment smoke --run_name spring_default_smoke \
  --forecast_examples 1 --forecast_time 1 --forecast_steps 100
