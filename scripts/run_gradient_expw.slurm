#!/bin/bash
#SBATCH --job-name=alpha_expw_grad
#SBATCH --account=punim3024
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --output=/data/gpfs/projects/punim3024/zalsaadi/robust_alpha/logs/expw_gradient_%j.out
#SBATCH --error=/data/gpfs/projects/punim3024/zalsaadi/robust_alpha/logs/expw_gradient_%j.err

set -euo pipefail
module purge
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate /data/gpfs/projects/punim3024/zalsaadi/anaconda3/envs/quonet

PROJECT=/data/gpfs/projects/punim3024/zalsaadi/robust_alpha
cd "$PROJECT"

if [[ ! -f src/plot_grad_expw.py ]]; then
    echo "ERROR: $PROJECT/src/plot_grad_expw.py is missing." >&2
    echo "Upload the Python source into src/ before submitting." >&2
    exit 2
fi
if [[ ! -f src/alpha_rank_full_v2.py ]]; then
    echo "ERROR: $PROJECT/src/alpha_rank_full_v2.py is missing." >&2
    exit 2
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -u src/plot_grad_expw.py \
  --function Gaussian \
  --condition asymmetric \
  --epochs 20000 \
  --warmup 2000 \
  --sigma-every 200 \
  --eval-every 100 \
  --alpha 0.9 \
  --sigma-floor 0.1 \
  --batch 20 \
  --lr-gd 0.5 \
  --lr-sgd 0.4 \
  --seed 0 \
  --out results/gradient_expw_gaussian_asymmetric
