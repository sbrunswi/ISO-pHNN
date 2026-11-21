# Script to plot the comparison of spring models
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import numpy as np

# load runs from MLflow 1. accept the data mlflow.db path and an output directory as arguments. select completed runs from spring_reproduction. 
# build a results table. extract _num_trajectories, seed, epocs, and mae_rel. convert parameter strings to numbers.
# validate and summarize
# plot and save figures
# usage:
# python plot_results.py <results_dir>


# 1. Select the results directory.
parser = argparse.ArgumentParser()
parser.add_argument("results_dir", type=Path,
                    help="Directory containing the shared mlflow.db")
parser.add_argument("--system", choices=["spring", "ball", "motor"],
                    default="spring")
args = parser.parse_args()

results_dir = args.results_dir.resolve()

# Define the path to the MLflow database and the system type.
database = results_dir / "mlflow.db"
system = args.system

if not database.is_file():
    raise FileNotFoundError(database)

# 2. Read completed runs from MLflow.
mlflow.set_tracking_uri(f"sqlite:///{database}")

experiment_name = f"{system}_original_loss"
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    raise ValueError(f"Experiment {experiment_name} not found")
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
)

# 3. Extract the values needed for plotting.
columns = {
    "params.num_trajectories": "trajectories",
    "params.seed": "seed",
    "params.epochs": "epochs",
    "metrics.mae_rel": "nmae",
}

table = runs[list(columns)].rename(columns=columns).copy()

table["model"] = runs["params.run_name"].str.extract(
    r"^(default|prior)_", expand=False
)

if table["model"].isna().any():
    raise ValueError("Some run names do not identify default or prior")

table = table.sort_values(["model", "trajectories"])

# Avoid silently combining different training budgets or repeated runs.
if table.empty:
    raise ValueError("No completed runs found")
if table["epochs"].nunique() != 1:
    raise ValueError("Select runs with the same epoch budget")
if table.duplicated(["model", "trajectories", "seed"]).any():
    raise ValueError("Repeated runs found; select the intended experiment batch")
if table["seed"].nunique() != 1:
    raise ValueError("This initial plot expects one seed")
if not (
    np.isfinite(table["nmae"]).all()
    and table["nmae"].gt(0).all()
):
    raise ValueError("NMAE values must be finite and positive")

# 4. Save the table and plot.
output_dir = results_dir.parent / system / "figures_original_loss"
output_dir.mkdir(parents=True, exist_ok=True)
table.to_csv(output_dir / f"{system}_default_table.csv", index=False)

fig, ax = plt.subplots(figsize=(4, 3.5))
for model_name, group in table.groupby("model"):
    ax.plot(
        group["trajectories"],
        group["nmae"],
        marker="o",
        label=f"ISO-pHNN {model_name}",
    )

ax.set_xscale("log", base=2)
ax.set_xticks([4, 8, 16, 32, 64])
ax.set_xticklabels(["4", "8", "16", "32", "64"])
ax.set_yscale("log")
ax.set_title({
    "spring": "spring chain",
    "ball": "magnetic ball",
    "motor": "PMSM",
}[system])
ax.set_xlabel("training trajectories")
ax.set_ylabel("Validation derivative NMAE (pooled)")
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig(output_dir / f"{system}_scaling.png", dpi=300)
fig.savefig(output_dir / f"{system}_scaling.pdf")
plt.close(fig)

