# Cluster experiments (Phase 2)

Everything runs through `run.py` (config-driven) + `sweep.py` (grid → SLURM jobs).
The estimator is chosen by the `estimator:` block of a YAML config; see
`Utils/estimator_factory.py` and `Utils/config.py`.

## Quick start (per cluster)

1. Mirror the repo on the cluster:
   ```
   cd $HOME
   git clone <repo-url> Poisson_CAE
   cd Poisson_CAE
   conda env create -f environment.yml  # or pip install -r requirements.txt
   pip install torch --index-url https://download.pytorch.org/whl/cu124  # GPU torch
   ```
2. Generate a sweep locally (or on the login node):
   ```
   python sweep.py --config configs/mnist_flat.yaml \
     --name mnist_diffusion \
     --grid train.lam=1e-3,3e-3,1e-2 \
     --grid estimator.k=32,64 \
     --grid data.seed=0,1 \
     --partition <PARTITION> --account <ACCOUNT> \
     --time 02:00:00 --mem 16G --conda-env torch
   ```
3. Inspect one generated script, then submit everything:
   ```
   cd cluster/jobs/mnist_diffusion
   cat run_0000.sh     # sanity check
   ./submit_all.sh     # or: sbatch run_0000.sh ...
   ```
4. Poll status / parse logs / update the tracking table:
   ```
   conda activate torch
   python cluster/status.py --jobs-dir cluster/jobs/mnist_diffusion
   ```
   (or watch `cluster/logs/mnist_diffusion/*.out` by hand)

## Cluster notes

- **SDumont**: compute nodes require `--partition` (e.g. `cpu`, `gpu`, `huge`).
  Use `--account` only if your project allocation requires it. Time budget for
  mirror partitions is short (hours); submit ≤ a-few-hour jobs.
- **ICA**: partitions are machine groups (e.g. `gpu`); same template applies.
- The template requests `--gres=gpu:1`; adjust `--cpus`, `--mem`, `--time` per
  job size in `cluster/template.slurm` defaults (overridable via `sweep.py`).

## Configs (experiment ladder)

| Config                    | dim  | task             | estimator                 |
|---------------------------|------|------------------|---------------------------|
| `configs/mog.yaml`        | 2    | reconstruction   | compact + diffusion       |
| `configs/spirals.yaml`    | 2    | classification   | compact + diffusion       |
| `configs/banana.yaml`     | 2    | classification   | compact + diffusion       |
| `configs/rings.yaml`      | 2    | classification   | compact + diffusion       |
| `configs/breast_cancer.yaml` | 30  | classification   | knn + diffusion           |
| `configs/sinusoid_reg.yaml`  | 50  | regression       | knn + diffusion           |
| `configs/mnist_flat.yaml`    | 784 | reconstruction   | knn + diffusion (t=196)   |

Baseline / ablation runs are generated as sweeps over the `estimator:` block:
`--grid estimator.scheme=global,compact,knn,variational` (Poisson only in low-d),
`--grid estimator.kernel_type=poisson,diffusion`, `--grid estimator.t=...`,
`--grid train.lam=...`, `--grid data.seed=...`.

## Results

Each run writes `results/<name>/<timestamp>/config.yml` + `metrics.json`
(accuracy / MSE). Aggregate them with a small script:

```
python scripts/collect_results.py results/ --out table.csv
```