#!/bin/bash
# Submit the whole re-run plan on Spartan, with job dependencies:
#
#   Stage 1 x4, in parallel   sigma floor 0.05 / 0.10 / 0.20, and contaminated validation
#   choose                    after the three floor runs succeed: sigma strategy + floor
#                             by Stage 1 VALIDATION error (choose_sigma.py, no test data)
#   Stage 2 x2, in parallel   probabilistic and strict ranking, with that choice
#
# Every run uses the wider learning-rate grids. Run from the folder that holds
# run_alpha.slurm:   bash submit_retrain.sh
set -euo pipefail
cd "$(dirname "$0")"
[ -f run_alpha.slurm ] || { echo "run_alpha.slurm not found next to this script" >&2; exit 1; }

GRID=(--lr-grid-sgd 0.1 0.2 0.4 0.8 1.6 --lr-grid-gd 0.125 0.25 0.5 1 2)

submit() {
    local out
    out=$(sbatch --parsable "$@") || { echo "sbatch failed: $*" >&2; return 1; }
    echo "${out%%;*}"
}

j1=$(submit run_alpha.slurm sigma - floor005 "${GRID[@]}" --sigma-floor 0.05)
j2=$(submit run_alpha.slurm sigma - floor010 "${GRID[@]}" --sigma-floor 0.10)
j3=$(submit run_alpha.slurm sigma - floor020 "${GRID[@]}" --sigma-floor 0.20)
j4=$(submit run_alpha.slurm sigma - valcont  "${GRID[@]}" --sigma-floor 0.10 --val-mode contaminated)
jc=$(submit --dependency=afterok:"$j1":"$j2":"$j3" --cpus-per-task=1 --mem=4G --time=00:20:00 \
            run_alpha.slurm choose)
jr=$(submit --dependency=afterok:"$jc" run_alpha.slurm ranking auto v3 "${GRID[@]}")
js=$(submit --dependency=afterok:"$jc" run_alpha.slurm ranking auto v3strict "${GRID[@]}" \
            --rank-mode strict)

cat <<EOF
Submitted:
  Stage 1   $j1 (floor 0.05)   $j2 (floor 0.10)   $j3 (floor 0.20)   $j4 (contaminated validation)
  choose    $jc   starts when $j1, $j2 and $j3 succeed
  Stage 2   $jr (ranking)   $js (strict ranking)   start when $jc succeeds
Watch with:  squeue -u \$USER
If a job fails, the jobs waiting on it never start: scancel them, fix the cause and
run this script again. Finished runs are cached and skipped.
EOF
