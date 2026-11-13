#!/usr/bin/env python3
"""Training curves of BP and of the alpha-loss with the three schedules of sigma (early, late,
periodic), from convergence.csv.gz of the sigma-floor-0.05 run (Section IV-D).
For every seed and epoch: test MSE and validation score = geometric mean over the 3 target
functions (per type of contamination), over the 9 contaminated cases ("all") or over the 6
asymmetric and point-mass cases ("gross"); sigma = geometric mean; psi = mean weight of the
outliers / clean points (mean over the cases).
Then median, 25% and 75% over the 20 seeds.
    python3 paper/fig5_data.py results/sigma_floor005 fig5_curves.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R, OUT = sys.argv[1], sys.argv[2]
SCHED = ["early", "late", "periodic"]
NAMES = [f"{o} MSE" for o in ("GD", "SGD")] + [f"{o} Alpha[{s}]" for o in ("GD", "SGD") for s in SCHED]
COND = ["clean", "asymmetric", "point_mass", "symmetric"]
CONT, GROSS = COND[1:], ["asymmetric", "point_mass"]
c = pd.read_csv(R + "/convergence.csv.gz",
                usecols=["epoch", "test_mse", "val_score", "sigma", "q_outlier", "q_inlier",
                         "method", "function", "condition", "seed"])
c = c[c.method.isin(NAMES)]
assert sorted(c.method.unique()) == sorted(NAMES), c.method.unique()
assert c.seed.nunique() == 20 and c.function.nunique() == 3, (c.seed.nunique(), c.function.unique())
logm = lambda s: np.exp(np.log(s).mean()) if s.notna().all() else np.nan  # noqa: E731


def summarize(per_seed, cond):
    rows = []
    for (m, e), s in per_seed.groupby(["method", "epoch"]):
        row = dict(method=m, condition=cond, epoch=e, n_seeds=len(s))
        for k in ("test", "val", "sigma", "q_outlier", "q_inlier"):
            v = s[k].dropna()
            if len(v) == 20:
                row.update({f"{k}_med": v.median(), f"{k}_q25": v.quantile(0.25),
                            f"{k}_q75": v.quantile(0.75)})
        rows.append(row)
    return rows


rows = []
groups = [(cd, [cd]) for cd in COND] + [("all", CONT), ("gross", GROSS)]
for name, conds in groups:
    d = c[c.condition.isin(conds)]
    k = d.groupby(["method", "seed", "epoch"]).size()
    assert k.eq(3 * len(conds)).all(), (name, k.value_counts())
    per_seed = d.groupby(["method", "seed", "epoch"]).agg(
        test=("test_mse", logm), val=("val_score", logm), sigma=("sigma", logm),
        q_outlier=("q_outlier", "mean"), q_inlier=("q_inlier", "mean"))
    rows += summarize(per_seed, name)
out = pd.DataFrame(rows).sort_values(["condition", "method", "epoch"])
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)
print("rows", len(out), "| epochs per method:", out.groupby("method").epoch.nunique().to_dict())
end = out[out.epoch == out.epoch.max()].pivot(index="method", columns="condition", values="test_med")
print("\nmedian test MSE at the last epoch (x1e-4):\n", (end * 1e4).round(2).to_string())
s = out[(out.condition == "all") & out.method.str.contains("Alpha")]
for m in [n for n in NAMES if "Alpha" in n]:
    t = s[s.method == m].set_index("epoch")
    print(m, "sigma med @100/2100/5000/20000:", t.reindex([100, 2100, 5000, 20000]).sigma_med.round(3).tolist())
