#!/bin/bash
# Run ONCE on the Spartan login node:  bash setup_images.sh
# Uses the same Python environment as the earlier alpha_rank runs (created if missing)
# and runs the self-test (about 1 minute).
set -e
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
[ -d ~/venvs/alpha-3.10.4 ] || virtualenv ~/venvs/alpha-3.10.4
source ~/venvs/alpha-3.10.4/bin/activate
pip install --quiet --upgrade pip
pip install --quiet numpy pandas scipy matplotlib
OMP_NUM_THREADS=1 python alpha_images.py --self-test --workers 2
echo "Setup done.  Next:  bash get_data.sh"
