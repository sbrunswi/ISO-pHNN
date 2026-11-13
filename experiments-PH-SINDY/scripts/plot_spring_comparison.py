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


# 1. Select the results directory.
parser = argparse.ArgumentParser()
parser.add_argument("results_dir", type=Path)
args = parser.parse_args()

results_dir = args.results_dir.resolve()
database = results_dir / "mlflow.db"

if not database.is_file():
    raise FileNotFoundError(database)

# 2. Read completed runs from MLflow.
mlflow.set_tracking_uri(f"sqlite:///{database}")

experiment = mlflow.get_experiment_by_name("spring_reproduction")
if experiment is None:
    raise ValueError("Experiment spring_reproduction not found")

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

for column in table.columns:
    table[column] = pd.to_numeric(table[column], errors="raise")

table = table.sort_values("trajectories")
print(table.to_string(index=False))

# Avoid silently combining different training budgets or repeated runs.
if table.empty:
    raise ValueError("No completed runs found")
if table["epochs"].nunique() != 1:
    raise ValueError("Select runs with the same epoch budget")
if table.duplicated(["trajectories", "seed"]).any():
    raise ValueError("Multiple runs found for the same trajectory count and seed")
if table["seed"].nunique() != 1:
    raise ValueError("This initial plot expects one seed")
if not table["nmae"].gt(0).all():
    raise ValueError("Log-scale plotting requires finite, positive errors")

# 4. Save the table and plot.
output_dir = results_dir / "figures"
output_dir.mkdir(exist_ok=True)
table.to_csv(output_dir / "spring_default_table.csv", index=False)

fig, ax = plt.subplots(figsize=(4, 3.5))
ax.plot(
    table["trajectories"],
    table["nmae"],
    "o-",
    label="ISO-pHNN default",
)

ax.set_xscale("log", base=2)
ax.set_xticks([4, 8, 16, 32, 64])
ax.set_xticklabels(["4", "8", "16", "32", "64"])
ax.set_yscale("log")
ax.set_title("spring chain")
ax.set_xlabel("training trajectories")
ax.set_ylabel("Validation derivative NMAE")
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig(output_dir / "spring_scaling.png", dpi=300)
fig.savefig(output_dir / "spring_scaling.pdf")
plt.close(fig)