#!/usr/bin/env python3
"""Train the paper's alpha model with the update  W <- W - eta * (dL/dW) * exp(W).

Same network, data, alpha = 0.9, periodic sigma, warm-up, checkpoints and seeds as the
paper (alpha_rank_full_v2.py, imported unchanged). The exp(W) factor multiplies the
gradient of every weight matrix during the alpha phase, exactly where the paper's
sparse masks act; biases are updated with the plain gradient, as in the sparse runs.
Step sizes are tuned per optimiser and case with the paper's pilot procedure.

Resumable: every finished job is written to its own file, so the script can be
re-run until it reports that nothing is left.
    python3 paper/expw_run.py OUT_DIR [--workers N] [--budget SECONDS]
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

ARGS = ar.parse_args(["--sigma-floor", "0.05"])
# v3 grids (GD 0.125-2, SGD 0.1-1.6) extended fourfold down and twofold up
GRID = {"GD": [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0],
        "SGD": [0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]}
_plain_update = ar.update


def update(theta, grads, lr, kind=None, fraction=1.0, rng=None, *rest, **kw):
    if kind != "expw":
        return _plain_update(theta, grads, lr, kind, fraction, rng, *rest, **kw)
    for (W, b), (dW, db) in zip(theta, grads):
        W -= lr * dW * np.exp(W)
        b -= lr * db
    return 0, 0.0, 0


ar.update = update  # train() looks the update up in the module at call time


def spec(opt):
    return dict(name=f"{opt} Alpha expw", opt=opt, loss="Alpha", sigma="periodic",
                sparse="expw", fraction=1.0)


def tune_job(job):
    opt, function, condition = job
    sched = ar.schedule(ARGS, ARGS.tune_epochs)
    crit = {}
    for lr in GRID[opt]:
        vals = []
        for k in range(ARGS.tune_seeds):
            data = ar.make_data(function, condition, k, ARGS, pilot=True)
            final = ar.train(spec(opt), data, lr, sched, ARGS, k, pilot=True)[0]
            vals.append(ar.tuning_score(final))
        crit[lr] = float(np.mean(np.log(vals))) if all(np.isfinite(vals)) else float("inf")
    best = min(GRID[opt], key=lambda lr: (crit[lr], lr))
    return dict(opt=opt, function=function, condition=condition, lr=best,
                at_edge=best in (min(GRID[opt]), max(GRID[opt])), criterion=crit)


def eval_job(job):
    opt, function, condition, seed, lr = job
    data = ar.make_data(function, condition, seed, ARGS)
    final, cols, _ = ar.train(spec(opt), data, lr, ar.schedule(ARGS, ARGS.epochs), ARGS, seed,
                              record=True)
    keep = ("method", "opt", "lr", "test_selected", "test_alpha_phase", "test_final",
            "val_post_alpha", "diverged", "sigma_final", "q_outlier_final", "epochs_completed")
    row = {k: final[k] for k in keep}
    row.update(function=function, condition=condition, seed=seed)
    sel = [i for i, c in enumerate(cols["checkpoint_candidate"]) if c]
    curve = {k: [float(cols[k][i]) for i in sel] for k in ("epoch", "test_mse", "val_score")}
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
            print("tuning not finished; run again"); return
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
