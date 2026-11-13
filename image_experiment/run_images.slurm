#!/bin/bash
# Image-data experiment: one array task per data set
#   0 mnist   1 frey   2 norb   3 cifar10   4 svhn
# Each task prepares its data set (first time only), tunes the step sizes on pilot runs,
# runs 20 realizations x 4 conditions x {GD, SGD} x {BP, alpha}, and writes a summary.
# Finished runs are cached: if a task hits its time limit, submit it again.
#SBATCH --job-name=alpha_img
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out

module purge
module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/alpha-3.10.4/bin/activate

# One BLAS thread per process: parallelism comes from --workers.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg

DATASETS=(mnist frey norb cifar10 svhn)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}
echo "data set: $DS   node: $(hostname)   started: $(date)"
python alpha_images.py --data "${DATA:-data}" --datasets "$DS" \
    --workers "$SLURM_CPUS_PER_TASK" --out results_images ${EXTRA:-}
echo "finished: $(date)"
