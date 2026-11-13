#!/usr/bin/env python3
"""Fig. 6 at one-column width (3.5 in): plain vs exp(W)-scaled gradient of the alpha-loss for one
case (Gaussian target, asymmetric outliers, one realization).  Left column GD, right column SGD;
rows: (a, d) norms of both expressions at the parameters of the plain run, (b, e) their ratio,
(c, f) test MSE when training with each rule.
    python3 paper/fig6_plot.py [CSV] [OUTPUT_STEM]    (CSV: made by fig6_data.py;
                                                       default paper/data/fig6_trajectories.csv)"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent / "data" / "fig6_trajectories.csv")
STEM = sys.argv[2] if len(sys.argv) > 2 else "fig_expw"
PLAIN, EXPW = "#1d6b1d", "#6250d6"     # alpha-loss green (plain rule), violet (e^W rule)
INK, INK2, MUTED, GRIDC, AXIS = "#111111", "#3d3d3a", "#8a8984", "#e4e3dc", "#9d9c94"
DASH = (0, (4.0, 1.8))
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
plain_num = FuncFormatter(lambda v, _: f"{v:g}")

d = pd.read_csv(CSV)
d = d[d.epoch > 0]
fig, axs = plt.subplots(3, 2, figsize=(3.5, 4.3), sharex=True)
letters = {("GD", 0): "a", ("GD", 1): "b", ("GD", 2): "c",
           ("SGD", 0): "d", ("SGD", 1): "e", ("SGD", 2): "f"}
for col, opt in enumerate(("GD", "SGD")):
    p = d[(d.opt == opt) & (d.rule == "plain")]
    e = d[(d.opt == opt) & (d.rule == "expw")]
    x = p.epoch / 1000
    ax = axs[0, col]                                   # (a, d) both expressions, same theta
    ax.plot(x, p.raw, color=PLAIN, lw=1.0)
    ax.plot(x, p.scaled, color=EXPW, lw=1.0, ls=DASH)
    ax.set_yscale("log")
    ax = axs[1, col]                                   # (b, e) ratio
    ax.plot(x, p.scaled / p.raw, color=EXPW, lw=1.0)
    ax.axhline(1, color=MUTED, lw=0.6, ls=(0, (1, 1.6)))
    ax.set_ylim(0.9, 2.6)
    ax.yaxis.set_major_locator(FixedLocator([1, 1.5, 2, 2.5]))
    ax = axs[2, col]                                   # (c, f) training outcome
    ax.plot(x, p.test_mse, color=PLAIN, lw=1.0)
    ax.plot(e.epoch / 1000, e.test_mse, color=EXPW, lw=1.0, ls=DASH)
    ax.set_yscale("log")
    for r in range(3):
        ax = axs[r, col]
        ax.set_title(f"({letters[(opt, r)]}) {opt}", loc="left")
        ax.axvline(2, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
        ax.set_xlim(0, 20)
        ax.set_xticks([0, 10, 20])
        ax.grid(True, color=GRIDC, lw=0.45, which="major")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
for r, lab in enumerate(("Gradient norm", r"Scaled $\div$ plain", "Test MSE")):
    axs[r, 0].set_ylabel(lab)
for ax in axs[2]:
    ax.set_xlabel(r"Epoch ($\times 10^3$)")
handles = [Line2D([], [], color=PLAIN, lw=1.4, label=r"Plain: $\nabla_\theta L$"),
           Line2D([], [], color=EXPW, lw=1.4, ls=DASH, label=r"Scaled: $e^{W}\odot\nabla_W L$"),
           Line2D([], [], color=MUTED, lw=0.8, ls=(0, (2, 2)), label=r"First $\alpha$-update")]
fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0),
           handlelength=1.9, columnspacing=0.9, handletextpad=0.4)
fig.tight_layout(pad=0.2, h_pad=0.45, w_pad=0.7, rect=(0, 0, 1, 0.94))
Path(STEM).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(STEM + ".pdf")
fig.savefig(STEM + ".png", dpi=300)
print("saved", STEM + ".pdf")
