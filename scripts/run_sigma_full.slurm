#!/bin/bash
#SBATCH --job-name=alpha_sigma
#SBATCH --account=punim3024
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/data/gpfs/projects/punim3024/zalsaadi/robust_alpha/logs/sigma_full_%j.out
#SBATCH --error=/data/gpfs/projects/punim3024/zalsaadi/robust_alpha/logs/sigma_full_%j.err

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate /data/gpfs/projects/punim3024/zalsaadi/anaconda3/envs/quonet

cd /data/gpfs/projects/punim3024/zalsaadi/robust_alpha/src

python -u alpha_rank_full_v2.py \
    --stage sigma \
    --seeds 20 \
    --workers 8 \
    --out /data/gpfs/projects/punim3024/zalsaadi/robust_alpha/results/sigma_full
