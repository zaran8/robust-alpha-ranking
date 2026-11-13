#!/bin/bash
#SBATCH --job-name=alpha_rank
#SBATCH --account=punim3024
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/data/gpfs/projects/punim3024/zalsaadi/robust_alpha/logs/alpha_rank_%j.out
#SBATCH --error=/data/gpfs/projects/punim3024/zalsaadi/robust_alpha/logs/alpha_rank_%j.err
#
# Usage:  sbatch run_alpha.slurm MODE [STRATEGY] [TAG] [extra flags ...]
#
#   selftest                          run every verification test of the program
#   sigma    -        TAG  [flags]    Stage 1 (sigma strategies)  -> results/sigma_TAG
#   choose   -        -    [folders]  pick sigma strategy + floor from Stage 1 VALIDATION error
#                                     (default folders: results/sigma_floor005, _floor010, _floor020)
#   ranking  STRATEGY TAG  [flags]    Stage 2 with early | late | periodic
#                                     -> results/ranking_TAG_STRATEGY
#   ranking  auto     TAG  [flags]    Stage 2 with the strategy and floor written by 'choose'
#                                     -> results/ranking_TAG_STRATEGY_floorF
#
# Extra flags go straight to alpha_rank_full_v2.py, after the script's own (the last one wins).
# submit_retrain.sh submits the whole re-run plan with job dependencies.

set -euo pipefail

MODE=${1:-sigma}
STRATEGY=${2:-periodic}
TAG=${3:-v2}
shift $(( $# < 3 ? $# : 3 ))
EXTRA=("$@")

PROJECT=${PROJECT:-/data/gpfs/projects/punim3024/zalsaadi/robust_alpha}
RESULTS="$PROJECT/results"

module purge
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate /data/gpfs/projects/punim3024/zalsaadi/anaconda3/envs/quonet

# One BLAS thread per process: the matrices are tiny; parallelism comes from --workers.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

cd "$PROJECT/src"

WORKERS=${SLURM_CPUS_PER_TASK:-1}

echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Mode: $MODE | strategy: $STRATEGY | tag: $TAG"
echo "Extra flags: ${EXTRA[*]:-none}"
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
            --out "$RESULTS/sigma_${TAG}" \
            "${EXTRA[@]}"
        ;;

    choose)
        if [ ${#EXTRA[@]} -eq 0 ]; then
            EXTRA=("$RESULTS/sigma_floor005" "$RESULTS/sigma_floor010" "$RESULTS/sigma_floor020")
        fi
        python -u choose_sigma.py --out-dir "$RESULTS" "${EXTRA[@]}"
        ;;

    ranking)
        FLOOR_ARGS=()
        if [ "$STRATEGY" = "auto" ]; then
            CHOICE="$RESULTS/sigma_choice.env"
            if [ ! -f "$CHOICE" ]; then
                echo "No $CHOICE yet: run the 'choose' mode first."
                exit 1
            fi
            source "$CHOICE"
            STRATEGY="$SIGMA_STRATEGY"
            FLOOR_ARGS=(--sigma-floor "$SIGMA_FLOOR")
            OUT="$RESULTS/ranking_${TAG}_${STRATEGY}_floor${SIGMA_FLOOR}"
            echo "Using the Stage 1 choice: sigma strategy $STRATEGY, sigma floor $SIGMA_FLOOR"
            mkdir -p "$OUT"
            cp "$RESULTS/sigma_choice.json" "$OUT/"
        else
            OUT="$RESULTS/ranking_${TAG}_${STRATEGY}"
        fi
        python -u alpha_rank_full_v2.py \
            --stage ranking \
            --sigma-strategy "$STRATEGY" \
            --seeds 20 \
            --workers "$WORKERS" \
            --out "$OUT" \
            "${FLOOR_ARGS[@]}" \
            "${EXTRA[@]}"
        ;;

    *)
        echo "Unknown mode: $MODE (use selftest | sigma | choose | ranking)"
        exit 2
        ;;
esac

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
