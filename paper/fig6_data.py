"""Single-case trajectories for the page figure (Gaussian target, asymmetric outliers,
seed 0): the paper's own train() loop, with the two update rules and tuned step sizes.
At every checkpoint of the plain-gradient run, the full-batch gradient of the loss in
force is taken once and both expressions are measured on the same parameters:
||grad_W L||_2 and ||e^W (.) grad_W L||_2.  Test MSE is recorded for both runs.
    python3 paper/fig6_data.py [OUT_CSV] [--expw-tune FOLDER]
(FOLDER: the tune/ folder of expw_run.py; default paper/data/expw_tune, the tuned step sizes
used in the paper.)"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]
ap = argparse.ArgumentParser()
ap.add_argument("out_csv", nargs="?", default="fig6_trajectories.csv")
ap.add_argument("--expw-tune", default=str(HERE / "data" / "expw_tune"))
A = ap.parse_args()
out_csv = Path(A.out_csv)
import expw_run as ex  # noqa: E402  (patches ar.update with the exp(W) rule)

ar, args = ex.ar, ex.ARGS
LOG = []
_noise = ar.gradient_noise


def gradient_noise(theta, data, alpha_now, sigma_now, parts):
    _, g = ar.loss_and_grad(theta, data["X"], data["y"], alpha_now, sigma_now)
    raw = np.sqrt(sum(float(np.sum(dW ** 2)) for dW, _ in g))
    scaled = np.sqrt(sum(float(np.sum((dW * np.exp(W)) ** 2)) for (W, _), (dW, _) in zip(theta, g)))
    LOG.append((raw, scaled))
    return _noise(theta, data, alpha_now, sigma_now, parts)


ar.gradient_noise = gradient_noise
fn, cond, seed = "Gaussian", "asymmetric", 0
# step sizes: plain = the v3 Spartan choice for this case (individual_results.csv, lr column);
# exp(W) = this study's own tuning (expw_run.py, tune/), both on the paper's pilot procedure
LR_PLAIN = {"GD": 0.5, "SGD": 0.8}
TUNE = Path(A.expw_tune)
LR_EXPW = {o: json.loads((TUNE / f"{o}_{fn}_{cond}.json").read_text())["lr"] for o in ("GD", "SGD")}
data = ar.make_data(fn, cond, seed, args)
sched = ar.schedule(args, args.epochs)
lrs = {("GD", None): LR_PLAIN["GD"], ("GD", "expw"): LR_EXPW["GD"], ("SGD", None): LR_PLAIN["SGD"], ("SGD", "expw"): LR_EXPW["SGD"]}
frames = []
for (opt, kind), lr in lrs.items():
    spec = dict(name=f"{opt} {kind}", opt=opt, loss="Alpha", sigma="periodic", sparse=kind, fraction=1.0)
    LOG.clear()
    final, cols, _ = ar.train(spec, data, lr, sched, args, seed, record=True)
    f = pd.DataFrame({k: cols[k] for k in ("epoch", "test_mse", "sigma")})
    f["raw"], f["scaled"] = [r for r, _ in LOG], [s for _, s in LOG]
    f["opt"], f["rule"], f["lr"] = opt, kind or "plain", lr
    frames.append(f)
    print(opt, kind or "plain", "lr", lr, "test_alpha_phase", round(final["test_alpha_phase"], 6),
          "test_final", round(final["test_final"], 6), "diverged", final["diverged"])
out_csv.parent.mkdir(parents=True, exist_ok=True)
pd.concat(frames).to_csv(out_csv, index=False)
