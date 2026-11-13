"""Independent recomputation of the exp(W) numbers (v3, sigma floor 0.05) from the raw
per-run files: plain runs = Spartan Stage 2 'GD Alpha'/'SGD Alpha'; exp(W) runs = EXPW_OUT_DIR/eval.
    python3 paper/expw_verify.py results/ranking_v3_periodic_floor0.05 EXPW_OUT_DIR"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

P, E = Path(sys.argv[1]), Path(sys.argv[2])
M = "test_alpha_phase"
CONDS = ["clean", "asymmetric", "point_mass", "symmetric"]
gm = lambda x: float(np.exp(np.mean(np.log(np.asarray(x, float)))))  # noqa: E731

plain = pd.read_csv(P / "individual_results.csv")
plain = plain[plain.method.isin(["GD Alpha", "SGD Alpha"])]
ex = pd.DataFrame([json.loads(f.read_text())[0] for f in sorted((E / "eval").glob("*.json"))])
tune = pd.DataFrame([json.loads(f.read_text()) for f in sorted((E / "tune").glob("*.json"))])
print("runs: plain", len(plain), " exp(W)", len(ex), " diverged exp(W):", int(ex.diverged.sum()),
      " plain:", int(plain.diverged.astype(bool).sum()))
print("tuned exp(W) lr at grid edge:", int(tune.at_edge.sum()), "of", len(tune))
print("tuned exp(W) lr differs from plain lr in:")
for _, t in tune.iterrows():
    lp = plain[(plain.opt == t.opt) & (plain.function == t.function) & (plain.condition == t.condition)].lr.iloc[0]
    if lp != t.lr:
        print(f"   {t.opt} {t.function} {t.condition}: plain {lp}, exp(W) {t.lr}")

rows = []
for opt in ("GD", "SGD"):
    for fn in sorted(ex.function.unique()):
        for c in CONDS:
            a = ex[(ex.opt == opt) & (ex.function == fn) & (ex.condition == c)].set_index("seed")[M].sort_index()
            b = plain[(plain.opt == opt) & (plain.function == fn) & (plain.condition == c)].set_index("seed")[M].loc[a.index]
            assert len(a) == 20 and np.all(np.isfinite(a)) and np.all(np.isfinite(b))
            d = a.to_numpy() - b.to_numpy()
            rows.append(dict(opt=opt, function=fn, condition=c, ratio=float(np.exp(np.median(np.log(a / b)))),
                             p=float(wilcoxon(d).pvalue), med_e=float(a.median()), med_p=float(b.median()),
                             fin_ratio=float(np.exp(np.median(np.log(
                                 ex[(ex.opt == opt) & (ex.function == fn) & (ex.condition == c)].set_index("seed").test_final.sort_index()
                                 / plain[(plain.opt == opt) & (plain.function == fn) & (plain.condition == c)].set_index("seed").test_final.loc[a.index]))))))
st = pd.DataFrame(rows)
# Holm over the 24 tests, written out step by step
order = np.argsort(st.p.to_numpy())
adj, running = np.empty(len(st)), 0.0
for k, i in enumerate(order):
    running = max(running, min(1.0, (len(st) - k) * st.p.iloc[i]))
    adj[i] = running
st["p_holm"] = adj
st["sig"] = st.p_holm < 0.05
pd.set_option("display.width", 200)
print(st.round(4).to_string(index=False))
for opt in ("GD", "SGD"):
    s = st[st.opt == opt]
    print(f"\n{opt}: GM ratio (12 cases) {gm(s.ratio):.4f}; range {s.ratio.min():.3f}-{s.ratio.max():.3f}; "
          f"significant worse {int((s.sig & (s.ratio > 1)).sum())}, better {int((s.sig & (s.ratio < 1)).sum())}; "
          f"GM final-epoch ratio {gm(s.fin_ratio):.4f}")
    print("   GM ratio by condition:", {c: round(gm(s[s.condition == c].ratio), 3) for c in CONDS})
    for name, df in (("plain", plain), ("exp(W)", ex)):
        med = df[df.opt == opt].groupby(["function", "condition"])[M].median()
        print(f"   Table II {name:6s}:", " ".join(f"{gm(med.xs(c, level=1)) * 1e4:.3f}" for c in CONDS))
