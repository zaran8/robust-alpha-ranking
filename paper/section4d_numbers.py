"""Table II and every number of Section IV-D (and of the ranking study), from the v3 Spartan results (wider step-size grids,
sigma floors 0.05 / 0.10 / 0.20, contaminated validation, Stage 2 at the chosen
setting). Writes the figure data to OUT and prints the numbers.
    python3 paper/section4d_numbers.py results OUT_DIR    (results/: the Spartan results folder)"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
M = "test_alpha_phase"
CONT = ["asymmetric", "point_mass", "symmetric"]
GROSS = ["asymmetric", "point_mass"]
CONDS = ["clean"] + CONT
gm = lambda x: float(np.exp(np.mean(np.log(np.asarray(x, float)))))  # noqa: E731
nums = {}


def load(name):
    d = R / name
    return (pd.read_csv(d / "individual_results.csv"), pd.read_csv(d / "paired_statistics.csv"),
            pd.read_csv(d / "tuning.csv"))


def paired_ratio(df, a, b, metric=M):
    piv = df.pivot_table(index=["function", "condition", "seed"], columns="method", values=metric)
    r = np.exp(np.log(piv[a] / piv[b]).groupby(level=[0, 1]).median())
    return r


def sig_counts(st):
    s = st["significant_holm_0.05"].astype(bool)
    return int((s & (st.median_ratio < 1)).sum()), int((s & (st.median_ratio > 1)).sum())


s1, p1, t1 = load("sigma_floor005")
print("=" * 30, "STAGE 1, floor 0.05 (chosen)")
# ---------------------------------------------------------------- Table I
med = s1.groupby(["method", "function", "condition"])[M].median()
table = {}
for o in ("GD", "SGD"):
    for m in (f"{o} MSE", f"{o} Alpha[early]", f"{o} Alpha[late]", f"{o} Alpha[periodic]"):
        cells = [gm(med.loc[m].xs(c, level=1)) * 1e4 for c in CONDS]
        st = p1[(p1.metric == M) & (p1.family == "alpha_vs_mse") & (p1.candidate == m) & p1.condition.isin(CONT)]
        gain = 1 / gm(st.median_ratio) if len(st) else 1.0
        better, worse = sig_counts(st) if len(st) else (0, 0)
        table[m] = dict(cells=cells, gain=gain, sig_better=better, sig_worse=worse)
        print(f"  {m:20s}", " ".join(f"{v:9.4f}" for v in cells), f" gain {gain:7.2f} sig {better}/{worse}")
nums["table1"] = table
# clean data, periodic vs MSE
cl = p1[(p1.metric == M) & (p1.family == "alpha_vs_mse") & (p1.condition == "clean")]
print("  clean, alpha vs MSE per target:")
print(cl[["candidate", "function", "median_ratio", "p_holm", "significant_holm_0.05"]].round(4).to_string(index=False))
per = cl[cl.candidate.str.contains("periodic")]
nums["clean_periodic"] = dict(min=float(per.median_ratio.min()), max=float(per.median_ratio.max()),
                              sig=per[per["significant_holm_0.05"].astype(bool)][["candidate", "function", "median_ratio"]].to_dict("records"))
# per contamination type gains for periodic
for o in ("GD", "SGD"):
    st = p1[(p1.metric == M) & (p1.family == "alpha_vs_mse") & (p1.candidate == f"{o} Alpha[periodic]")]
    nums[f"{o}_periodic_gain_by_type"] = {c: 1 / gm(st[st.condition == c].median_ratio) for c in CONT}
    nums[f"{o}_periodic_min_case_gain"] = float(1 / st[st.condition != "clean"].median_ratio.max())
    print(f"  {o} periodic gain by type", {c: round(v, 1) for c, v in nums[f'{o}_periodic_gain_by_type'].items()},
          "min case gain", round(nums[f'{o}_periodic_min_case_gain'], 2))

# ---------------------------------------------------------------- figure data (a), (b)
c1 = pd.read_csv(R / "sigma_floor005" / "convergence.csv.gz")
c1 = c1[c1.checkpoint_candidate.astype(bool) | (c1.epoch == 0)] if "checkpoint_candidate" in c1 else c1
keep = ["GD MSE", "SGD MSE", "GD Alpha[periodic]", "SGD Alpha[periodic]"]
a = c1[c1.method.isin(keep) & c1.condition.isin(CONT)]
ma = a.groupby(["method", "function", "condition", "epoch"]).test_mse.median().reset_index()
curve_a = ma.groupby(["method", "epoch"]).test_mse.apply(gm).unstack(0)
curve_a.to_csv(OUT / "fig_a_test_mse.csv")
for m in ("GD Alpha[periodic]", "SGD Alpha[periodic]"):
    print(f"  fig a {m}: 2000 {curve_a.loc[2000, m]:.5f} -> 2100 {curve_a.loc[2100, m]:.5f} -> 20000 {curve_a.loc[20000, m]:.6f}"
          f"  (fall {curve_a.loc[2000, m] / curve_a.loc[2100, m]:.1f}x)")
for m in ("GD MSE", "SGD MSE"):
    print(f"  fig a {m}: min after 5000 {curve_a[m].loc[5000:].min():.4f}, end {curve_a.loc[20000, m]:.4f}")
alpha = [f"{o} Alpha[{s}]" for o in ("GD", "SGD") for s in ("early", "late", "periodic")]
b = c1[c1.method.isin(alpha) & c1.condition.isin(GROSS)]
mb = b.groupby(["method", "function", "condition", "epoch"]).q_outlier.median().reset_index()
curve_b = mb.groupby(["method", "epoch"]).q_outlier.mean().unstack(0)
curve_b.to_csv(OUT / "fig_b_q_gross.csv")
nums["fig_b_end"] = {c: float(curve_b[c].dropna().iloc[-1]) for c in curve_b.columns}
print("  fig b end:", {c: round(v, 4) for c, v in nums["fig_b_end"].items()})
for m in alpha:
    d = s1[s1.method == m]
    print(f"  {m:20s} sigma_final median: cont {d[d.condition.isin(CONT)].sigma_final.median():.3f}"
          f" asym {d[d.condition == 'asymmetric'].sigma_final.median():.3f} gross {d[d.condition.isin(GROSS)].sigma_final.median():.3f}"
          f" | at floor {np.isclose(d.sigma_final, 0.05).mean():.2f} | q sym median {d[d.condition == 'symmetric'].q_outlier_final.median():.3f}"
          f" | q asym median {d[d.condition == 'asymmetric'].q_outlier_final.median():.4f}")

# ---------------------------------------------------------------- SGD vs GD
for metric in (M, "test_final"):
    b2 = p1[(p1.family == "sgd_vs_gd") & (p1.metric == metric)]
    sgd_better, gd_better = sig_counts(b2)
    print(f"  SGD vs GD {metric}: tests {len(b2)}, SGD better {sgd_better}, GD better {gd_better}")
    nums[f"sgd_vs_gd_{metric}"] = dict(n=len(b2), sgd_better=sgd_better, gd_better=gd_better)
fs = {}
for m in ("GD MSE", "SGD MSE", "GD Alpha[periodic]", "SGD Alpha[periodic]", "SGD Alpha[late]", "SGD Alpha[early]",
          "GD Alpha[late]", "GD Alpha[early]"):
    piv = s1.pivot_table(index=["function", "condition", "seed"], columns="method", values=["test_final", M])
    rr = np.exp(np.nanmedian if False else np.log(piv["test_final"][m] / piv[M][m]).groupby(level=[0, 1]).median())
    fs[m] = gm(rr)
print("  final/selected (paired GM over 12 cases):", {k: round(v, 2) for k, v in fs.items()})
nums["final_over_selected"] = fs
sine = s1[(s1.function == "Sine") & (s1.condition == "clean")].groupby("method")[M].median()
print("  clean sine GD/SGD: MSE %.1f, periodic %.1f" % (sine["GD MSE"] / sine["SGD MSE"], sine["GD Alpha[periodic]"] / sine["SGD Alpha[periodic]"]))

# ---------------------------------------------------------------- step sizes and divergence
ch = t1[t1.chosen.astype(bool)].drop_duplicates(["function", "condition", "method"])
grid = {"GD": [0.125, 0.25, 0.5, 1.0, 2.0], "SGD": [0.1, 0.2, 0.4, 0.8, 1.6]}
print("  chosen step sizes (stage 1, floor 0.05):")
for m, g in ch.groupby("method"):
    o = m.split()[0]
    top = int((g.lr == max(grid[o])).sum()); bot = int((g.lr == min(grid[o])).sum())
    print(f"    {m:20s} {dict(g.lr.value_counts().sort_index())}  top {top} bottom {bot}"
          + (f"  edge cases: {g[(g.lr == max(grid[o])) | (g.lr == min(grid[o]))][['function', 'condition', 'lr']].values.tolist()}" if top + bot else ""))
print("  diverged runs:", s1.groupby("method").diverged.sum()[lambda v: v > 0].to_dict())

# ---------------------------------------------------------------- floor study
print("=" * 30, "FLOOR STUDY")
choice = json.loads((R / "sigma_choice.json").read_text())
cand = pd.DataFrame(choice["candidates"])
print(cand[["strategy", "sigma_floor", "score", "score_contaminated", "score_gd", "score_sgd", "share_sigma_at_floor",
            "lr_at_grid_edge"]].round(6).to_string(index=False))
floors = {0.05: s1}
floors[0.10] = load("sigma_floor010")[0]
floors[0.20] = load("sigma_floor020")[0]
fl = {}
for f, d in floors.items():
    md = d.groupby(["method", "function", "condition"])[M].median()
    for o in ("GD", "SGD"):
        m = f"{o} Alpha[periodic]"
        fl[(f, o)] = dict(cont=gm(md.loc[m][md.loc[m].index.get_level_values(1) != "clean"]) * 1e4,
                          clean=gm(md.loc[m].xs("clean", level=1)) * 1e4,
                          at_floor=float(np.isclose(d[d.method == m].sigma_final, f).mean()))
for (f, o), v in fl.items():
    print(f"  floor {f:.2f} {o}: test GM contaminated {v['cont']:.3f}e-4, clean {v['clean']:.3f}e-4, sigma at floor {v['at_floor']:.2f}")
nums["floor_study"] = {f"{f}_{o}": v for (f, o), v in fl.items()}
# paired floor comparison, periodic alpha: 0.05 vs 0.10 and 0.20 (same data, weights, batches)
for f in (0.10, 0.20):
    for o in ("GD", "SGD"):
        m = f"{o} Alpha[periodic]"
        A = floors[0.05][floors[0.05].method == m].set_index(["function", "condition", "seed"])[M]
        B = floors[f][floors[f].method == m].set_index(["function", "condition", "seed"])[M].loc[A.index]
        r = np.exp(np.log(A / B).groupby(level=[0, 1]).median())
        pv = [wilcoxon(np.log(A.loc[k]) - np.log(B.loc[k])).pvalue for k in r.index]
        print(f"  floor 0.05 / {f:.2f} {o}: GM ratio all12 {gm(r):.3f}, cont9 {gm(r[r.index.get_level_values(1) != 'clean']):.3f},"
              f" range {r.min():.2f}-{r.max():.2f}, unadjusted p<0.05 better {sum(1 for x, p in zip(r, pv) if p < .05 and x < 1)} worse {sum(1 for x, p in zip(r, pv) if p < .05 and x > 1)}")
        nums[f"floor005_vs_{f}_{o}"] = dict(gm_all12=gm(r), gm_cont9=gm(r[r.index.get_level_values(1) != "clean"]))

# ---------------------------------------------------------------- contaminated validation
print("=" * 30, "CONTAMINATED VALIDATION (floor 0.10)")
vc, pvc, tvc = load("sigma_valcont")
s10 = floors[0.10]
for m in ("GD MSE", "SGD MSE", "GD Alpha[periodic]", "SGD Alpha[periodic]"):
    A = vc[vc.method == m].set_index(["function", "condition", "seed"])[M]
    B = s10[s10.method == m].set_index(["function", "condition", "seed"])[M].loc[A.index]
    r = np.exp(np.log(A / B).groupby(level=[0, 1]).median())
    print(f"  {m:20s} contaminated-val / clean-val test MSE: GM all12 {gm(r):.3f}, cont9 {gm(r[r.index.get_level_values(1) != 'clean']):.3f}, range {r.min():.2f}-{r.max():.2f}")
    nums[f"valcont_{m}"] = dict(gm_all12=gm(r), gm_cont9=gm(r[r.index.get_level_values(1) != "clean"]))
for o in ("GD", "SGD"):
    st = pvc[(pvc.metric == M) & (pvc.family == "alpha_vs_mse") & (pvc.candidate == f"{o} Alpha[periodic]") & pvc.condition.isin(CONT)]
    better, worse = sig_counts(st)
    print(f"  valcont {o} periodic gain vs MSE {1 / gm(st.median_ratio):.1f}, sig {better}/{worse}")
    nums[f"valcont_{o}_gain"] = dict(gain=1 / gm(st.median_ratio), sig=better)

# ---------------------------------------------------------------- stage 2
for name, tag in (("ranking_v3_periodic_floor0.05", "probabilistic"), ("ranking_v3strict_periodic_floor0.05", "strict")):
    print("=" * 30, "STAGE 2", tag)
    s2, p2, t2 = load(name)
    pr = p2[(p2.metric == M) & p2.family.str.startswith("PRIMARY")].copy()
    pr["opt"] = pr.candidate.str.split().str[0]
    pr["fraction"] = pr.candidate.str.extract(r"(\d+)%")[0].astype(int)
    if tag == "probabilistic":
        pr[["opt", "fraction", "function", "condition", "median_ratio", "significant_holm_0.05", "p_holm"]].to_csv(
            OUT / "fig_c_ratio_cases.csv", index=False)
    rows = []
    for (o, f), g in pr.groupby(["opt", "fraction"]):
        rb, uw = sig_counts(g)
        rows.append(dict(opt=o, fraction=f, gm=gm(g.median_ratio), ranked_better=rb, ranked_worse=uw,
                         mean_rank=float(s2[s2.method == f"{o} Ranked {f}%"].mean_selected_rank.mean())))
    rk = pd.DataFrame(rows)
    print(rk.round(3).to_string(index=False))
    print(f"  totals: tests {len(pr)}, ranked better {rk.ranked_better.sum()}, ranked worse {rk.ranked_worse.sum()}"
          f" (GD {rk[rk.opt == 'GD'].ranked_worse.sum()}, SGD {rk[rk.opt == 'SGD'].ranked_worse.sum()})")
    sa = []
    for o in ("GD", "SGD"):
        for f in (10, 30, 50, 70):
            for k in ("Ranked", "Uniform"):
                r = paired_ratio(s2, f"{o} {k} {f}%", f"{o} Alpha")
                sa.append(dict(opt=o, fraction=f, kind=k, gm=gm(r)))
    sa = pd.DataFrame(sa).pivot_table(index=["opt", "fraction"], columns="kind", values="gm")
    print("  sparse / full alpha:\n", sa.round(3).to_string())
    print("  diverged:", s2.groupby("method").diverged.sum()[lambda v: v > 0].to_dict())
    ch2 = t2[t2.chosen.astype(bool)].drop_duplicates(["function", "condition", "method"])
    edge = ch2[ch2.chosen_at_grid_edge.astype(bool)]
    print(f"  step sizes at grid edge: {len(edge)} of {len(ch2)}; by method: {edge.method.value_counts().to_dict()}")
    nums[f"stage2_{tag}"] = dict(rows=rows, sparse_vs_alpha=sa.reset_index().to_dict("records"),
                                 total_ranked_better=int(rk.ranked_better.sum()), total_ranked_worse=int(rk.ranked_worse.sum()))
    # full alpha in stage 2 equals stage 1 periodic?
    if tag == "probabilistic":
        for o in ("GD", "SGD"):
            A = s2[s2.method == f"{o} Alpha"].set_index(["function", "condition", "seed"])[M].sort_index()
            B = s1[s1.method == f"{o} Alpha[periodic]"].set_index(["function", "condition", "seed"])[M].sort_index()
            print(f"  stage 2 '{o} Alpha' == stage 1 '{o} Alpha[periodic]': max rel diff {float(np.max(np.abs(A / B - 1))):.2e}")
(OUT / "v3_numbers.json").write_text(json.dumps(nums, indent=1, default=float))
