#!/bin/bash
# Generate every Phase 2 sweep. Run this ON THE CLUSTER LOGIN NODE after
#   git pull   (container ICA_v4.sif must be present at CONTAINER path)
# because generated SLURM scripts bake in the repo path.
#
# Edit the three CLUSTER knobs below for your site, then:
#   bash cluster/sweeps.sh
# It writes job scripts to cluster/jobs/<name>/ and prints a submit command.
# Submit lazily: cd cluster/jobs/<name> && ./submit_all.sh
#
# Cluster notes:
#   SDumont: --partition changes per queue (gpu/huge), may need --account.
#   ICA:     --partition is the machine group; usually no account.

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# ---- EDIT THESE PER CLUSTER ----
PARTITION="gpu"        # e.g. gpu on SDumont / ICA machine group
ACCOUNT=""             # e.g. your allocation if required
CONTAINER="/share_zeta/Proxy-Sim/guillermo.carrillo/envs/ICA_v4.sif"
# ---------------------------------

run_sweep () {
  local name="$1"; shift
  python sweep.py --name "$name" \
    --partition "$PARTITION" --account "$ACCOUNT"     --container "$CONTAINER" \
    "$@"
}

echo "== 1. Ladder: 2D toys (compact + diffusion) =="
for exp in mog spirals banana rings; do
  run_sweep "${exp}_ladder" \
    --config "configs/${exp}.yaml" \
    --grid data.seed=0,1 --time 00:45:00 --mem 8G
done

echo "== 2. Ladder: tabular + time series (kNN + diffusion) =="
run_sweep breast_cancer_ladder \
  --config configs/breast_cancer.yaml \
  --grid data.seed=0,1,2 --time 00:45:00 --mem 8G
run_sweep sinusoid_ladder \
  --config configs/sinusoid_reg.yaml \
  --grid data.seed=0,1,2 --time 00:45:00 --mem 8G

echo "== 3. Ladder: MNIST flat d=784 (kNN + diffusion, t=196) =="
run_sweep mnist_diffusion \
  --config configs/mnist_flat.yaml \
  --grid data.seed=0,1,2 --time 03:00:00 --mem 16G

echo "== 4. Ablation: lambda (banana) =="
run_sweep banana_lambda \
  --config configs/banana.yaml \
  --grid train.lam=0,1e-3,3e-3,1e-2,3e-2,1e-1 \
  --grid data.seed=0,1 --time 00:45:00 --mem 8G

echo "== 5. Ablation: locality radius R (banana) =="
run_sweep banana_radius \
  --config configs/banana.yaml \
  --grid estimator.radius=1.0,1.5,2.0,2.5,3.0 \
  --grid data.seed=0,1 --time 00:45:00 --mem 8G

echo "== 6. Ablation: global vs compact vs knn (banana, Poisson kernel) =="
run_sweep banana_scheme_poisson \
  --config configs/banana.yaml \
  --grid estimator.scheme=global,compact,knn \
  --grid estimator.kernel_type=poisson \
  --grid data.seed=0,1 --time 00:45:00 --mem 8G

echo "== 7. Ablation: diffusion vs Poisson kernel (banana) =="
run_sweep banana_kernel \
  --config configs/banana.yaml \
  --grid estimator.kernel_type=poisson,diffusion \
  --grid data.seed=0,1 --time 00:45:00 --mem 8G

echo "== 8. Ablation: diffusion-kernel vs variational-Ritz solver (banana) =="
run_sweep banana_variational \
  --config configs/banana.yaml \
  --grid estimator.scheme=compact,variational \
  --grid data.seed=0,1 --time 00:45:00 --mem 8G

echo "== 9. Ablation: k neighbours (breast_cancer + MNIST) =="
run_sweep breast_cancer_k \
  --config configs/breast_cancer.yaml \
  --grid estimator.k=8,16,32,64 \
  --grid data.seed=0,1 --time 00:45:00 --mem 8G
run_sweep mnist_k \
  --config configs/mnist_flat.yaml \
  --grid estimator.k=16,32,64 \
  --grid data.seed=0,1 --time 03:00:00 --mem 16G

echo "== 10. Ablation: corruption mode (breast_cancer + banana) =="
run_sweep breast_cancer_corruption \
  --config configs/breast_cancer.yaml \
  --grid train.corruption_mode=gaussian,ddpm,shift_scale \
  --grid estimator.scheme=knn \
  --grid data.seed=0,1 --time 00:45:00 --mem 8G
run_sweep banana_corruption \
  --config configs/banana.yaml \
  --grid train.corruption_mode=gaussian,ddpm,shift_scale \
  --grid estimator.scheme=compact \
  --grid data.seed=0,1 --time 00:45:00 --mem 8G

echo
echo "All sweeps generated under cluster/jobs/."
echo "Submit e.g.: cd cluster/jobs/mnist_diffusion && ./submit_all.sh"