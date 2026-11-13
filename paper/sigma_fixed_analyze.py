#!/usr/bin/env python3
"""Fixed-sigma control vs the paper's periodic scale (v3, sigma floor 0.05), paired by seed.
Periodic and MSE runs: Spartan Stage 1 (sigma_floor005). Fixed runs: sigma_fixed_run.py output.
Family declared before looking at results: fixed vs periodic, 24 tests (12 cases x GD/SGD),
two-sided Wilcoxon signed-rank on paired differences (as the program), Holm-corrected.
    python3 paper/sigma_fixed_analyze.py results FIXED_OUT_DIR OUT_DIR"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

R, E, O = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
O.mkdir(parents=True, exist_ok=True)
M = "test_alpha_phase"
CONT = ["asymmetric", "point_mass", "symmetric"]
CONDS = ["clean"] + CONT
gm = lambda x: float(np.exp(np.mean(np.log(np.asarray(x, float)))))  # noqa: E731
nums = {}


def holm(p):
    p = np.asarray(p, float)
    order, adj, run = np.argsort(p), np.empty(len(p)), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - k) * p[i]))
        adj[i] = run
    return adj


s1 = pd.read_csv(R / "sigma_floor005" / "individual_results.csv")
rows, curves = [], []
for f in sorted((E / "eval").glob("*.json")):
    row, curve = json.loads(f.read_text())
    rows.append(row)
    c = pd.DataFrame(curve)
    c["opt"], c["function"], c["condition"], c["seed"] = row["opt"], row["function"], row["condition"], row["seed"]
    curves.append(c)
fx = pd.DataFrame(rows)
tn = pd.DataFrame([json.loads(f.read_text()) for f in sorted((E / "tune").glob("*.json"))])
assert len(fx) == 480 and len(tn) == 24, (len(fx), len(tn))
nums["runs"] = len(fx)
nums["diverged"] = int(fx.diverged.sum())
nums["sigma_all_at_floor"] = bool(np.allclose(fx.sigma_final, 0.05) and np.allclose(fx.sigma_first, 0.05)
                                  and (fx.sigma_updates == 1).all() and (fx.warmup_epoch == 2000).all())
nums["lr_at_edge"] = int(tn.at_edge.sum())
nums["lr_edge_cases"] = tn[tn.at_edge][["opt", "function", "condition", "lr"]].to_dict("records")


def per_case(a_df, b_df, metric=M, b_method=None):
    out = []
    for (o, fn, c), g in a_df.groupby(["opt", "function", "condition"]):
        a = g.set_index("seed")[metric].sort_index()
        bm = b_df[(b_df.method == b_method.format(o=o)) & (b_df.function == fn) & (b_df.condition == c)]
        b = bm.set_index("seed")[metric].loc[a.index]
        assert len(a) == 20 and np.isfinite(a).all() and np.isfinite(b).all()
        d = a.to_numpy() - b.to_numpy()
        out.append(dict(opt=o, function=fn, condition=c, ratio=float(np.exp(np.median(np.log(a / b)))),
                        p=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
                        wins=int((d < 0).sum()), losses=int((d > 0).sum()),
                        med_a=float(a.median()), med_b=float(b.median()),
                        lr_a=float(g.lr.iloc[0]), lr_b=float(bm.lr.iloc[0])))
    st = pd.DataFrame(out)
    st["p_holm"] = holm(st.p)
    st["significant"] = st.p_holm < 0.05
    return st


# ---------------------------------------------------------------- fixed vs periodic (primary)
st = per_case(fx, s1, M, "{o} Alpha[periodic]")
st.to_csv(O / "fixed_vs_periodic.csv", index=False)
for o in ("GD", "SGD"):
    s = st[st.opt == o]
    sig = s[s.significant]
    nums[f"{o} fixed/periodic"] = dict(
        gm_all12=gm(s.ratio), gm_cont9=gm(s[s.condition.isin(CONT)].ratio),
        by_condition={c: gm(s[s.condition == c].ratio) for c in CONDS},
        range=[float(s.ratio.min()), float(s.ratio.max())],
        sig_fixed_better=int((sig.ratio < 1).sum()), sig_fixed_worse=int((sig.ratio > 1).sum()),
        lr_fixed=s.lr_a.value_counts().sort_index().to_dict(),
        lr_same_as_periodic=int((s.lr_a == s.lr_b).sum()))
# final-epoch comparison (secondary)
stf = per_case(fx, s1, "test_final", "{o} Alpha[periodic]")
for o in ("GD", "SGD"):
    s = stf[stf.opt == o]
    nums[f"{o} fixed/periodic, final epoch"] = dict(gm_all12=gm(s.ratio),
                                                    sig_better=int((s.significant & (s.ratio < 1)).sum()),
                                                    sig_worse=int((s.significant & (s.ratio > 1)).sum()))

# ---------------------------------------------------------------- fixed vs squared error (as Table I)
sm = per_case(fx, s1, M, "{o} MSE")
for o in ("GD", "SGD"):
    s = sm[sm.opt == o]
    c9 = s[s.condition.isin(CONT)]
    nums[f"{o} fixed vs MSE"] = dict(gain_cont9=1 / gm(c9.ratio),
                                     clean_ratio=s[s.condition == "clean"].set_index("function").ratio.round(4).to_dict())
    med = fx[fx.opt == o].groupby(["function", "condition"])[M].median()
    medp = s1[s1.method == f"{o} Alpha[periodic]"].groupby(["function", "condition"])[M].median()
    nums[f"{o} Table I row, x1e-4"] = dict(
        fixed=[round(gm(med.xs(c, level=1)) * 1e4, 3) for c in CONDS],
        periodic=[round(gm(medp.xs(c, level=1)) * 1e4, 3) for c in CONDS])
    q = fx[(fx.opt == o) & fx.condition.isin(["asymmetric", "point_mass"])].groupby(["function", "condition"]).q_outlier_final.median()
    nums[f"{o} q gross outliers (fixed, mean of medians)"] = float(q.mean())

# ---------------------------------------------------------------- curves after the first alpha-update
cv = pd.concat(curves)
cv = cv[cv.condition.isin(CONT)]
cf = cv.groupby(["opt", "function", "condition", "epoch"]).test_mse.median().groupby(["opt", "epoch"]).apply(gm).unstack(0)
conv = pd.read_csv(R / "sigma_floor005" / "convergence.csv.gz",
                   usecols=["epoch", "checkpoint_candidate", "test_mse", "sigma", "method", "function", "condition", "seed"])
conv = conv[conv.method.isin(["GD Alpha[periodic]", "SGD Alpha[periodic]"]) & conv.checkpoint_candidate.astype(bool)
            & conv.condition.isin(CONT)]
cp = conv.groupby(["method", "function", "condition", "epoch"]).test_mse.median().groupby(["method", "epoch"]).apply(gm).unstack(0)
sig_p = conv.groupby(["method", "epoch"]).sigma.median().unstack(0)
curve = pd.DataFrame({"GD fixed": cf["GD"], "SGD fixed": cf["SGD"],
                      "GD periodic": cp["GD Alpha[periodic]"], "SGD periodic": cp["SGD Alpha[periodic]"],
                      "GD periodic sigma (median)": sig_p["GD Alpha[periodic]"],
                      "SGD periodic sigma (median)": sig_p["SGD Alpha[periodic]"]})
curve.to_csv(O / "fixed_curves.csv")
for e in (2000, 2100, 2500, 3000, 4000, 5000, 10000, 20000):
    if e in curve.index:
        nums.setdefault("curve", {})[e] = {k: float(v) for k, v in curve.loc[e].items()}
# epoch at which the periodic sigma (median over contaminated runs) first reaches the floor
for o in ("GD", "SGD"):
    s = sig_p[f"{o} Alpha[periodic]"].dropna()
    nums[f"{o} periodic sigma median: first epoch at floor"] = int(s[np.isclose(s, 0.05)].index.min())
    nums[f"{o} periodic sigma median at epoch 2000"] = float(s.loc[2000])

(O / "fixed_numbers.json").write_text(json.dumps(nums, indent=1, default=float))
pd.set_option("display.width", 200)
print(json.dumps(nums, indent=1, default=float))
print(st.round(4).to_string(index=False))
