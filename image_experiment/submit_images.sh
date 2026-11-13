#!/bin/bash
# Submit the five data sets as one array job, and the joint analysis to run after it.
#     bash submit_images.sh
# (If the data are not in ./data:  DATA=/path/to/data bash submit_images.sh)
set -e
mkdir -p logs
JOB=$(sbatch --parsable run_images.slurm)
JOB=${JOB%%;*}
echo "submitted array job $JOB (mnist, frey, norb, cifar10, svhn)"
AJOB=$(sbatch --parsable --dependency=afterany:"$JOB" analyze_images.slurm)
echo "analysis job ${AJOB%%;*} will start when $JOB has finished"
echo "watch:  squeue -u \$USER      logs:  tail -f logs/alpha_img_${JOB}_*.out"
