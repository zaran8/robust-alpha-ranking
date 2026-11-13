#!/usr/bin/env python3
"""Section IV-D figures at one-column width (3.5 in): BP and the alpha-loss with the three
schedules of sigma (early, late, periodic), GD in the left column and SGD in the right one.
Median over the 20 realizations.

  fig_gdsgd_schedules  drop-in replacement for Fig. 5 (same panels): (a) test MSE and (b) scale
                       sigma, geometric mean over the 9 contaminated data sets; (c) weight psi of
                       the outliers (thick) and of the clean points (thin), mean over the
                       asymmetric and point-mass data sets; median line and interquartile band
  fig_gdsgd_types      test MSE for each type of contamination, clean data included (rows),
                       geometric mean over the 3 target functions

    python3 paper/fig5_plot.py [CSV] [--metric test|val] [--bands schedules|both|none]
                         [--no-clean-weights] [--out DIR]
CSV (default paper/data/fig5_curves.csv) is made by fig5_data.py.  --metric val draws the validation score (mean squared
residual on the clean validation set, internal scale) instead of the test MSE."""
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="?", default=str(Path(__file__).resolve().parent / "data" / "fig5_curves.csv"))
ap.add_argument("--metric", choices=["test", "val"], default="test")
ap.add_argument("--bands", choices=["schedules", "both", "none"], default="schedules",
                help="interquartile bands: in fig_gdsgd_schedules only (default), both, none")
ap.add_argument("--no-clean-weights", action="store_true",
                help="panel (c) of fig_gdsgd_schedules without the weight of the clean points")
ap.add_argument("--out", default=".")
A = ap.parse_args()
OUT = Path(A.out)
OUT.mkdir(parents=True, exist_ok=True)

# BP blue and alpha-loss green as in the paper's other figures; plum for the early schedule.
# Checked with the dataviz palette validator (all pairs pass; plum/blue is in the colour-blind
# floor band, so the early schedule also has its own line style).
MODELS = [  # method suffix, label, colour, line style, line width
    ("MSE", "BP", "#1f77b4", "-", 1.0),
    ("Alpha[early]", r"$\alpha$-loss, $\sigma$ early", "#a6488f", (0, (1.1, 1.4)), 1.25),
    ("Alpha[late]", r"$\alpha$-loss, $\sigma$ late", "#4fa34a", (0, (3.6, 1.6)), 1.1),
    ("Alpha[periodic]", r"$\alpha$-loss, $\sigma$ periodic", "#1d6b1d", "-", 1.3),
]
ALPHAS = [m[0] for m in MODELS[1:]]
OPTS = ["GD", "SGD"]
INK, INK2, MUTED, GRIDC, AXIS = "#111111", "#3d3d3a", "#8a8984", "#e4e3dc", "#9d9c94"
WARM = 2.0                                        # end of the warm-up (late, periodic), x1000
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7.5, "axes.titlesize": 7.8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.2,
    "ytick.major.size": 2.2, "ytick.minor.size": 1.3, "ytick.minor.width": 0.4,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "text.color": INK, "pdf.fonttype": 42,
    "ps.fonttype": 42, "legend.frameon": False, "savefig.dpi": 300, "axes.titlepad": 2.5,
    "axes.labelpad": 2.0,
})
D = pd.read_csv(A.csv)
K = A.metric
YLAB = {"test": "Test MSE", "val": "Validation score"}[K]
SUFFIX = "" if K == "test" else "_val"


def series(cond, method, k):
    t = D[(D.condition == cond) & (D.method == method)].set_index("epoch").sort_index()
    return t[[f"{k}_med", f"{k}_q25", f"{k}_q75"]].dropna()


def draw(ax, cond, opt, k, which=None, bands=False, lw_scale=1.0):
    for suffix, _, color, ls, lw in MODELS:
        if which is not None and suffix not in which:
            continue
        t = series(cond, f"{opt} {suffix}", k)
        if t.empty:
            continue
        x = t.index / 1000
        if bands:
            ax.fill_between(x, t[f"{k}_q25"], t[f"{k}_q75"], color=color, alpha=0.13, lw=0,
                            zorder=1)
        ax.plot(x, t[f"{k}_med"], color=color, ls=ls, lw=lw * lw_scale,
                zorder=4 if suffix == "Alpha[periodic]" else 3)


def frame(ax, ylog=True):
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.axvline(WARM, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=0)
    ax.grid(True, color=GRIDC, lw=0.45, which="major")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if ylog:
        ax.set_yscale("log")


def log_ticks(ax, lo, hi, majors, labels):
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(FixedLocator(majors))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticklabels(labels)


def yrange(conds, k, pad_lo=0.8, pad_hi=1.25):
    """Limits holding every median curve from epoch 100 on (the first epochs, where all the
    models start from the same initial network, are cut at the top), extended down to the
    decade below when it is close."""
    v = D[D.condition.isin(conds) & D.epoch.ge(100)][f"{k}_med"].dropna()
    lo, hi = v.min() * pad_lo, v.max() * pad_hi
    dec = 10.0 ** math.floor(math.log10(lo))
    lo = dec * 0.85 if lo / dec < 2.0 else lo
    es = range(math.ceil(math.log10(lo)) - 1, math.floor(math.log10(hi)) + 1)
    ms = (1,) if math.log10(hi / lo) >= 2.5 else (1, 3)    # add 3x ticks on short ranges
    ticks = [(m, e) for e in es for m in ms if lo <= m * 10.0 ** e <= hi]
    return lo, hi, [m * 10.0 ** e for m, e in ticks], [
        rf"$10^{{{e}}}$" if m == 1 else rf"${m}{{\times}}10^{{{e}}}$" for m, e in ticks]


def legend(fig):
    handles = [Line2D([], [], color=c, ls=ls, lw=lw + 0.2, label=lab) for _, lab, c, ls, lw in MODELS]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.6, columnspacing=1.6, handletextpad=0.5, labelspacing=0.3)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    plt.close(fig)
    print("saved", OUT / f"{stem}.pdf")


# ------------------------------------------- fig_gdsgd_schedules: (a) error, (b) sigma, (c) psi
bands = A.bands in ("schedules", "both")
fig, axs = plt.subplots(3, 2, figsize=(3.5, 4.55), sharex=True, sharey="row")
for j, opt in enumerate(OPTS):
    draw(axs[0, j], "all", opt, K, bands=bands)
    draw(axs[1, j], "all", opt, "sigma", which=ALPHAS, bands=bands)
    if not A.no_clean_weights:
        draw(axs[2, j], "gross", opt, "q_inlier", which=ALPHAS, lw_scale=0.6)
    draw(axs[2, j], "gross", opt, "q_outlier", which=ALPHAS, bands=bands)
    frame(axs[0, j])
    frame(axs[1, j])
    frame(axs[2, j], ylog=False)
    axs[0, j].set_title(opt, fontsize=8.2, pad=3)
    axs[1, j].axhline(0.05, color=INK2, lw=0.6, ls=(0, (1, 1.6)), zorder=2)
    axs[1, j].text(19.6, 0.0478, "floor", color=INK2, fontsize=6.8, ha="right", va="top")
    axs[2, j].set_xlabel(r"Epoch ($\times 10^3$)")
lo, hi, ds, labs = yrange(["all"], K)
log_ticks(axs[0, 0], lo, hi, ds, labs)
log_ticks(axs[1, 0], 0.037, 0.55, [0.05, 0.1, 0.2, 0.4], ["0.05", "0.1", "0.2", "0.4"])
axs[2, 0].set_ylim(-0.03, 1.07)
axs[2, 0].set_yticks([0, 0.25, 0.5, 0.75, 1])
axs[2, 0].set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
axs[0, 0].set_ylabel("(a) " + YLAB)
axs[1, 0].set_ylabel(r"(b) Scale $\sigma$")
if A.no_clean_weights:
    axs[2, 0].set_ylabel(r"(c) Weight $\psi$ of outliers")
else:
    axs[2, 0].set_ylabel(r"(c) Weight $\psi$")
    axs[2, 1].text(19.6, 0.955, "clean points", color=INK2, fontsize=6.8, ha="right", va="top")
legend(fig)
fig.tight_layout(pad=0.2, h_pad=0.55, w_pad=0.6, rect=(0, 0, 1, 0.915))
save(fig, "fig_gdsgd_schedules" + SUFFIX)

# ------------------------------------------- fig_gdsgd_types: one row per type of contamination
CONDS = [("clean", "Clean"), ("asymmetric", "Asymmetric"), ("point_mass", "Point mass"),
         ("symmetric", "Symmetric")]
fig, axs = plt.subplots(4, 2, figsize=(3.5, 5.3), sharex=True, sharey=True)
for i, (cond, name) in enumerate(CONDS):
    for j, opt in enumerate(OPTS):
        ax = axs[i, j]
        draw(ax, cond, opt, K, bands=A.bands == "both")
        frame(ax)
        if i == 0:
            ax.set_title(opt, fontsize=8.2, pad=3)
        if j == 1:                                # row label on the right, as in a facet grid
            ax.text(1.03, 0.5, name, transform=ax.transAxes, ha="left", va="center",
                    rotation=270, fontsize=7.6, color=INK)
lo, hi, ds, labs = yrange([c for c, _ in CONDS], K)
log_ticks(axs[0, 0], lo, hi, ds, labs)
for ax in axs[:, 0]:
    ax.set_ylabel(YLAB)
for ax in axs[3]:
    ax.set_xlabel(r"Epoch ($\times 10^3$)")
legend(fig)
fig.tight_layout(pad=0.2, h_pad=0.45, w_pad=0.6, rect=(0, 0, 1, 0.93))
save(fig, "fig_gdsgd_types" + SUFFIX)
