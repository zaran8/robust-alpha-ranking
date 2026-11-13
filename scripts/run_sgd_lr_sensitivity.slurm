#!/bin/bash
#SBATCH --job-name=alpha_lr_sensitivity
#SBATCH --account=punim3024
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/data/gpfs/projects/punim3024/zalsaadi/robust_alpha/logs/lr_sensitivity_%j.out
#SBATCH --error=/data/gpfs/projects/punim3024/zalsaadi/robust_alpha/logs/lr_sensitivity_%j.err

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate /data/gpfs/projects/punim3024/zalsaadi/anaconda3/envs/quonet

cd /data/gpfs/projects/punim3024/zalsaadi/robust_alpha

python -u src/alpha_rank_full_v2.py \
  --stage sigma \
  --sigma-strategies periodic \
  --sigma-floor 0.1 \
  --epochs 20000 \
  --seeds 20 \
  --alpha 0.9 \
  --tune-epochs 2000 \
  --tune-seeds 2 \
  --lr-grid-gd 0.25 0.5 1.0 2.0 \
  --lr-grid-sgd 0.05 0.1 0.2 0.4 0.6 0.8 1.0 \
  --workers 8 \
  --max-jobs 24 \
  --out results/sigma_lr_sensitivity
