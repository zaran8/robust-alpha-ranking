#!/bin/bash
#SBATCH --job-name=alpha_rank
#SBATCH --account=punim3024
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/data/gpfs/projects/punim3024/zalsaadi/robust_alpha/logs/alpha_rank_%j.out
#SBATCH --error=/data/gpfs/projects/punim3024/zalsaadi/robust_alpha/logs/alpha_rank_%j.err

set -euo pipefail

MODE=${1:-sigma}
STRATEGY=${2:-periodic}

PROJECT=/data/gpfs/projects/punim3024/zalsaadi/robust_alpha

module purge
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate /data/gpfs/projects/punim3024/zalsaadi/anaconda3/envs/quonet

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

cd "$PROJECT/src"

WORKERS=${SLURM_CPUS_PER_TASK:-1}

echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Mode: $MODE"
echo "Workers: $WORKERS"
echo "Started: $(date)"
echo "Python: $(which python)"
python --version
echo "========================================"

case "$MODE" in

    selftest)
        python -u alpha_rank_full_v2.py --self-test
        ;;

    sigma)
        python -u alpha_rank_full_v2.py \
            --stage sigma \
            --seeds 20 \
            --workers "$WORKERS" \
            --out "$PROJECT/results/sigma_full_v2"
        ;;

    ranking)
        python -u alpha_rank_full_v2.py \
            --stage ranking \
            --sigma-strategy "$STRATEGY" \
            --seeds 20 \
            --workers "$WORKERS" \
            --out "$PROJECT/results/ranking_full_v2_${STRATEGY}"
        ;;

    *)
        echo "Unknown mode: $MODE"
        exit 1
        ;;
esac

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
