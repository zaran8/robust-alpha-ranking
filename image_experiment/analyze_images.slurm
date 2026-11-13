#!/bin/bash
# Joint analysis of the five data sets (submitted by submit_images.sh after the array job).
#SBATCH --job-name=alpha_img_analyze
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

module purge
module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/alpha-3.10.4/bin/activate
export MPLBACKEND=Agg
python alpha_images.py --data "${DATA:-data}" --analyze-only --out results_images ${EXTRA:-}
bash pack_results.sh
