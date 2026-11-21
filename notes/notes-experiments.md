
### Experiments

| Setting | Value |
|---|---|
| Training trajectories | 32 |
| Training duration | 10 seconds |
| Training points per trajectory | 1,000 |
| Validation trajectories | 100 |
| Validation duration | 100 seconds |
| Validation points per trajectory | 10,000 |
| Hidden layers / width | 3 / 16 |
| Epochs | 4,000 |
| Batch size | 256 |
| Learning rate | 0.001 |
| Output loss weight | 0.25 |

# key commands
.venv/bin/mlflow ui \
  --backend-store-uri sqlite:////home/techstor1/a/sbrunswi/projects/ISO-pHNN/experiments-PH-SINDY/results/20261112_223405/mlflow.db \
  --port 5002

  http://127.0.0.1:5002

# Issue with 

