#!/usr/bin/env python3
"""Control for the residual scale: sigma FIXED at the floor from the first alpha-update.

Identical to the paper's periodic alpha model -- same program (alpha_rank_full_v2.py,
imported unchanged), data, network, 2,000-epoch squared-error warm-up, checkpoints,
seeds, step-size grids and tuning procedure -- except that sigma is never estimated:
from the first alpha-update on, the alpha-loss uses sigma = sigma_min = 0.05.
The periodic model's sigma ends at this floor in 97.5% of runs, so the comparison asks
whether the re-estimation path matters or only the final value of sigma.

Implementation: the program's 'late' schedule (one scale at --warmup, never re-estimated)
with the MAD estimate replaced by the floor. Nothing else is changed.
Resumable: every finished job is written to its own file; re-run until nothing is left.
    python3 paper/sigma_fixed_run.py OUT_DIR [--workers N] [--budget SECONDS]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # the repository's src/
import alpha_rank_full_v2 as ar  # noqa: E402

# the v3 settings of the Spartan runs (sigma_floor005 / ranking_v3_periodic_floor0.05)
ARGS = ar.parse_args(["--sigma-floor", "0.05",
                      "--lr-grid-gd", "0.125", "0.25", "0.5", "1", "2",
                      "--lr-grid-sgd", "0.1", "0.2", "0.4", "0.8", "1.6"])


def fixed_scale(residuals, floor):
    """Replaces the 1.4826*MAD estimate: the scale is the floor itself."""
    return float(floor)


ar.scale_from_residuals = fixed_scale  # train() looks the estimator up at call time


def spec(opt):
    # 'late' = scale set once at --warmup (2,000 epochs, as for 'periodic'), never re-estimated
    return dict(name=f"{opt} Alpha[fixed]", opt=opt, loss="Alpha", sigma="late", sparse=None,
                fraction=1.0)


def tune_job(job):
    """The program's own tuning rule (choose_lrs): mean over pilot seeds of log(best
    post-alpha validation score); ties go to the smaller step."""
    opt, function, condition = job
    sched = ar.schedule(ARGS, ARGS.tune_epochs)
    grid = ar.lr_grid(ARGS, opt)
    crit = {}
    for lr in grid:
        vals = []
        for k in range(ARGS.tune_seeds):
            data = ar.make_data(function, condition, k, ARGS, pilot=True)
            final = ar.train(spec(opt), data, lr, sched, ARGS, k, pilot=True)[0]
            vals.append(ar.tuning_score(final))
        crit[lr] = float(np.mean(np.log(vals))) if all(np.isfinite(vals)) else float("inf")
    best = min(grid, key=lambda lr: (crit[lr], lr))
    return dict(opt=opt, function=function, condition=condition, lr=best,
                at_edge=best in (min(grid), max(grid)), criterion=crit)


def eval_job(job):
    opt, function, condition, seed, lr = job
    data = ar.make_data(function, condition, seed, ARGS)
    final, cols, _ = ar.train(spec(opt), data, lr, ar.schedule(ARGS, ARGS.epochs), ARGS, seed,
                              record=True)
    final["sigma_strategy"] = "fixed"
    keep = ("method", "opt", "lr", "sigma_strategy", "test_selected", "test_alpha_phase",
            "test_final", "val_post_alpha", "best_post_alpha_epoch", "diverged", "sigma_first",
            "sigma_final", "sigma_updates", "warmup_epoch", "q_outlier_final", "q_inlier_final",
            "alpha_updates", "epochs_completed")
    row = {k: final[k] for k in keep}
    row.update(function=function, condition=condition, seed=seed)
    sel = [i for i, c in enumerate(cols["checkpoint_candidate"]) if c]
    curve = {k: [float(cols[k][i]) for i in sel]
             for k in ("epoch", "test_mse", "val_score", "sigma", "q_outlier", "q_inlier")}
    return row, curve


def run(pool, fn, jobs, paths, budget, start, chunk=8):
    pending = [(j, p) for j, p in zip(jobs, paths) if not p.is_file()]
    for i in range(0, len(pending), chunk):
        if time.time() - start > budget:
            return False
        part = pending[i:i + chunk]
        for (_, p), res in zip(part, pool.map(fn, [j for j, _ in part])):
            p.write_text(json.dumps(res, default=float))
    return all(p.is_file() for p in paths)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", help="output folder; finished jobs are kept, so the script can be re-run")
    p.add_argument("--workers", type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    p.add_argument("--budget", type=float, default=0,
                   help="stop after this many seconds and resume on the next call (0: no limit)")
    a = p.parse_args()
    out, budget, chunk = Path(a.out), (a.budget or float("inf")), max(8, 4 * a.workers)
    start = time.time()
    (out / "tune").mkdir(parents=True, exist_ok=True)
    (out / "eval").mkdir(parents=True, exist_ok=True)
    cases = [(o, f, c) for o in ("GD", "SGD") for f in ar.FUNCTIONS for c in ar.CONDITIONS]
    with Pool(a.workers) as pool:
        tpaths = [out / "tune" / f"{o}_{f}_{c}.json" for o, f, c in cases]
        if not run(pool, tune_job, cases, tpaths, budget, start, chunk):
            print("tuning not finished; run again")
            return
        lr = {tuple(t[k] for k in ("opt", "function", "condition")): t["lr"]
              for t in (json.loads(p.read_text()) for p in tpaths)}
        jobs = [(o, f, c, s, lr[(o, f, c)]) for o in ("SGD", "GD") for f, c in
                [(f, c) for f in ar.FUNCTIONS for c in ar.CONDITIONS] for s in range(ARGS.seeds)]
        epaths = [out / "eval" / f"{o}_{f}_{c}_{s}.json" for o, f, c, s, _ in jobs]
        done = run(pool, eval_job, jobs, epaths, budget, start, chunk)
    n = sum(p.is_file() for p in epaths)
    print(f"{n}/{len(epaths)} evaluation runs done" + ("" if done else "; run again"))


if __name__ == "__main__":
    main()
