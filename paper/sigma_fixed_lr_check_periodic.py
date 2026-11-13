"""Same step sizes for the PERIODIC model (unchanged program) in the two GD cases, so the
failure counts of fixed and periodic sigma can be compared at equal step size.
    python3 paper/sigma_fixed_lr_check_periodic.py OUT_DIR"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # the repository's src/
import alpha_rank_full_v2 as ar

ARGS = ar.parse_args(["--sigma-floor", "0.05", "--lr-grid-gd", "0.125", "0.25", "0.5", "1", "2",
                      "--lr-grid-sgd", "0.1", "0.2", "0.4", "0.8", "1.6"])
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)


def job(j):
    opt, fn, c, s, lr = j
    spec = dict(name=f"{opt} Alpha[periodic]", opt=opt, loss="Alpha", sigma="periodic", sparse=None, fraction=1.0)
    final = ar.train(spec, ar.make_data(fn, c, s, ARGS), lr, ar.schedule(ARGS, ARGS.epochs), ARGS, s)[0]
    final.update(function=fn, condition=c, seed=s)
    return final


jobs = [("GD", "Gaussian", c, s, lr) for c in ("clean", "symmetric") for lr in (0.25, 1.0, 0.5) for s in range(20)]
todo = [j for j in jobs if not (OUT / f"{j[2]}_lr{j[4]}_{j[3]}.json").is_file()]
with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))) as pool:
    for j, res in zip(todo, pool.map(job, todo)):
        (OUT / f"{j[2]}_lr{j[4]}_{j[3]}.json").write_text(json.dumps(res, default=float))
