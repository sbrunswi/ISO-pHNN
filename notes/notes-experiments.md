
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
  --backend-store-uri sqlite:////Users/sbrunswi/projects/ISO-pHNN/experiments-PH-SINDY/results/20261112_223405/mlflow.db \
  --port 5002
# inspecting differences between results: 
One important finding from reading the simulator: the current spring equations deserve comparison with the paper before further training. In data.py, the supplied input matrix acts on components 1 and 3, and the Hamiltonian coefficients contain an additional factor of one-half. These are concrete implementation details to investigate; they were not checked by our earlier array-shape tests. They could affect reproducibility, though we have not yet established that they explain your error gap.

# Ball experiments: 
These experiments use fixed width 16 and 4,000 epochs, whereas the paper’s scaling figures report the best configurations from width and epoch searches. The repository’s ball initial-flux range is [-3, 4], compared with [-3, 5] stated in the paper. Runs use the repository’s original pooled-normalization loss and NMAE, rather than the paper’s written per-component normalization.

# Selecting another GPU
CUDA_VISIBLE_DEVICES=1 \
bash experiments-PH-SINDY/scripts/replication-ball-server.sh


