#!/usr/bin/env python3
"""Compare the exp(W)-scaled update with the plain gradient, paired by seed.
Plain-gradient runs: the paper's Spartan Stage-2 'GD Alpha' / 'SGD Alpha' results
(same code, data, initial weights, batch order). Writes the figure data and numbers.
    python3 paper/expw_analyze.py results/ranking_v3_periodic_floor0.05 EXPW_OUT_DIR OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # the repository's src/
import alpha_rank_full_v2 as ar  # noqa: E402

R, E, F = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
F.mkdir(parents=True, exist_ok=True)
M = "test_alpha_phase"
CONT = ["asymmetric", "point_mass", "symmetric"]
gm = lambda x: float(np.exp(np.mean(np.log(np.asarray(x, float)))))  # noqa: E731
nums = {}

# ---------------------------------------------------------------- results
plain = pd.read_csv(R / "individual_results.csv")
plain = plain[plain.method.isin(["GD Alpha", "SGD Alpha"])].copy()
plain["rule"] = "plain"
rows, curves = [], []
for p in sorted((E / "eval").glob("*.json")):
    row, curve = json.loads(p.read_text())
    rows.append(row)
    c = pd.DataFrame(curve)
    if row["diverged"]:  # count the run as infinitely bad after it diverged
        rest = [e for e in range(0, 20001, 100) if e > c.epoch.max()]
        c = pd.concat([c, pd.DataFrame(dict(epoch=rest, test_mse=np.inf, val_score=np.inf))])
    c["method"], c["function"], c["condition"], c["seed"] = (row["method"], row["function"],
                                                             row["condition"], row["seed"])
    curves.append(c)
expw = pd.DataFrame(rows)
expw["rule"] = "expw"
assert len(expw) == 480, "missing exp(W) runs"
# As in the paper's pipeline, a diverged run keeps its validation-selected checkpoint from
# before the blow-up; its final-epoch error is undefined and left out of final ratios.
nums["expw_diverged"] = expw[expw.diverged][["opt", "function", "condition", "seed", "lr", "epochs_completed"]].to_dict("records")
nums["plain_diverged"] = int(plain.diverged.sum())

# ---------------------------------------------------------------- paired statistics
stats = []
for opt in ("GD", "SGD"):
    a = expw[expw.opt == opt].set_index(["function", "condition", "seed"])
    b = plain[plain.opt == opt].set_index(["function", "condition", "seed"])
    for (fn, cond), g in a.groupby(level=[0, 1]):
        va = g[M]
        vb = b.loc[va.index, M]
        d = va.to_numpy() - vb.to_numpy()
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        lr_a = float(g.lr.iloc[0])
        lr_b = float(b.loc[va.index, "lr"].iloc[0])
        stats.append(dict(opt=opt, function=fn, condition=cond, p=p, lr_expw=lr_a, lr_plain=lr_b,
                          median_ratio=float(np.exp(np.median(np.log(va.to_numpy() / vb.to_numpy())))),
                          wins=int((d < 0).sum()), losses=int((d > 0).sum()),
                          final_ratio=float(np.exp(np.nanmedian(np.log(
                              g["test_final"].to_numpy(float) / b.loc[va.index, "test_final"].to_numpy(float)))))))
st = pd.DataFrame(stats)
# Holm over all 24 tests (12 cases x 2 optimisers), declared before looking at results
order = np.argsort(st.p.to_numpy())
m, run_max = len(st), 0.0
holm = np.empty(m)
for rank, i in enumerate(order):
    run_max = max(run_max, min(1.0, (m - rank) * st.p.iloc[i]))
    holm[i] = run_max
st["p_holm"] = holm
st["significant"] = st.p_holm < 0.05
st.to_csv(F / "expw_stats.csv", index=False)
for opt in ("GD", "SGD"):
    s = st[st.opt == opt]
    sig = s[s.significant]
    nums[opt] = dict(
        gm_ratio_all12=gm(s.median_ratio), gm_ratio_cont9=gm(s[s.condition.isin(CONT)].median_ratio),
        gm_ratio_clean3=gm(s[s.condition == "clean"].median_ratio),
        ratio_range=[float(s.median_ratio.min()), float(s.median_ratio.max())],
        sig_expw_better=int((sig.median_ratio < 1).sum()), sig_expw_worse=int((sig.median_ratio > 1).sum()),
        gm_final_ratio_all12=gm(s.final_ratio),
        lr_expw=s.lr_expw.value_counts().sort_index().to_dict(),
        lr_plain=s.lr_plain.value_counts().sort_index().to_dict())
    for cond in ("clean", *CONT):
        nums[opt][f"gm_ratio_{cond}"] = gm(s[s.condition == cond].median_ratio)

# medians of the two rules per case (for reporting), GM over the 9 contaminated cases
for opt in ("GD", "SGD"):
    for name, df in (("plain", plain), ("expw", expw)):
        med = df[df.opt == opt].groupby(["function", "condition"])[M].median()
        nums[opt][f"gm_test_{name}_cont9"] = gm(med[med.index.get_level_values(1).isin(CONT)])
        nums[opt][f"gm_test_{name}_all12"] = gm(med)

# ---------------------------------------------------------------- curves (panel b)
pc = pd.read_csv(R / "convergence.csv.gz",
                 usecols=["epoch", "checkpoint_candidate", "test_mse", "method", "function",
                          "condition", "seed"])
pc = pc[pc.method.isin(["GD Alpha", "SGD Alpha"]) & pc.checkpoint_candidate.astype(bool)]
ec = pd.concat(curves)
both = pd.concat([pc.drop(columns="checkpoint_candidate"), ec.drop(columns="val_score")])
both = both[both.condition.isin(CONT)]
med = both.groupby(["method", "function", "condition", "epoch"]).test_mse.median().reset_index()
curve = med.groupby(["method", "epoch"]).test_mse.apply(gm).unstack(0)
curve.to_csv(F / "expw_curves.csv")
for col in curve.columns:
    nums.setdefault("curve_end", {})[col] = float(curve[col].iloc[-1])

# ---------------------------------------------------------------- per-weight update (panel a)
# GD network of one case at the first alpha update: the warm-up is plain squared-error GD
# with the plain rule's tuned step size, then sigma is estimated and the alpha gradient taken.
args = ar.parse_args(["--sigma-floor", "0.05"])
fn, cond, seed = "Gaussian", "asymmetric", 0
lr = float(plain[(plain.method == "GD Alpha") & (plain.function == fn) & (plain.condition == cond)
                 & (plain.seed == seed)].lr.iloc[0])
data = ar.make_data(fn, cond, seed, args)
theta = ar.init_params(seed)
for _ in range(args.warmup):
    _, g = ar.loss_and_grad(theta, data["X"], data["y"], 1.0, 1.0)
    ar.update(theta, g, lr)
sigma = ar.scale_from_residuals(ar.predict(theta, data["X"]) - data["y"], args.sigma_floor)
_, g = ar.loss_and_grad(theta, data["X"], data["y"], args.alpha, sigma)
pw = []
for layer, ((W, _), (dW, _)) in enumerate(zip(theta, g), 1):
    for w, d in zip(W.ravel(), dW.ravel()):
        pw.append(dict(layer=layer, w=float(w), plain=float(abs(d)), expw=float(abs(d) * np.exp(w))))
pw = pd.DataFrame(pw)
pw.to_csv(F / "expw_per_weight.csv", index=False)
pos = pw.w > 0
nums["per_weight"] = dict(case=f"{fn}/{cond}/seed {seed}, GD, lr {lr}, sigma {sigma:.3f}",
                          n=len(pw), w_min=float(pw.w.min()), w_max=float(pw.w.max()),
                          w_by_layer={int(k): [float(v.min()), float(v.max())] for k, v in pw.groupby("layer").w},
                          share_pos_plain=float((pw.plain[pos] ** 2).sum() / (pw.plain ** 2).sum()),
                          share_pos_expw=float((pw.expw[pos] ** 2).sum() / (pw.expw ** 2).sum()),
                          factor_range=[float(np.exp(pw.w.min())), float(np.exp(pw.w.max()))])
(F / "expw_numbers.json").write_text(json.dumps(nums, indent=1, default=float))
print(json.dumps(nums, indent=1, default=float))
print(st.round(4).to_string(index=False))
