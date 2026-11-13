"""Sensitivity check for the two GD cases where the tuned fixed-sigma step sits at the
bottom of the grid (Gaussian clean / symmetric): full 20-seed runs at other step sizes.
    python3 paper/sigma_fixed_lr_check.py OUT_DIR"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sigma_fixed_run as fr

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
jobs = [("GD", "Gaussian", c, s, lr) for c in ("clean", "symmetric") for lr in (0.25, 0.5, 1.0) for s in range(20)]
todo = [j for j in jobs if not (OUT / f"{j[0]}_{j[1]}_{j[2]}_lr{j[4]}_{j[3]}.json").is_file()]
with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))) as pool:
    for j, res in zip(todo, pool.map(fr.eval_job, todo)):
        (OUT / f"{j[0]}_{j[1]}_{j[2]}_lr{j[4]}_{j[3]}.json").write_text(json.dumps(res, default=float))
print(len(list(OUT.glob("*.json"))), "runs")
