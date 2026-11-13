#!/usr/bin/env python3
"""plot_images.py -- figures of the image-data experiment (Section IV-F) from the Spartan
results of alpha_images.py.

  python plot_images.py                      # reads results_images/, writes results_images/figures/
  python plot_images.py --condition asymmetric

Figure 1  fig_images_gain.pdf      improvement factor R = test MSE of BP / test MSE of the
                                   alpha-loss (median over the 20 realizations), per data set,
                                   GD and SGD, one panel per type of contamination; the value
                                   is written on every bar.
Figure 2  fig_images_training.pdf  test MSE during training (median and interquartile range
                                   over the 20 realizations) of BP and of the alpha-loss, GD
                                   and SGD, one panel per data set, for one type of
                                   contamination (default: point mass).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
import numpy as np
import pandas as pd

DATASETS = [("mnist", "MNIST"), ("frey", "Frey faces"), ("norb", "Small NORB"),
            ("cifar10", "CIFAR-10"), ("svhn", "SVHN")]
CONDS = [("asymmetric", "Asymmetric outliers"), ("point_mass", "Point-mass outliers"),
         ("symmetric", "Symmetric outliers")]
# Colours of the paper: BP blue, alpha-loss green; GD dark/solid, SGD light/dashed.
C = {("BP", "GD"): "#1f77b4", ("BP", "SGD"): "#5aa7f2",
     ("alpha", "GD"): "#1d6b1d", ("alpha", "SGD"): "#56a94f"}
LS = {"GD": "-", "SGD": (0, (4.5, 2))}
INK, INK2, MUTED, GRID, AXIS = "#111111", "#3d3d3a", "#8a8984", "#e4e3dc", "#9d9c94"


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9.5,
        "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 9, "axes.edgecolor": AXIS, "axes.linewidth": 0.7,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.7,
        "ytick.major.width": 0.7, "xtick.major.size": 3, "ytick.major.size": 3,
        "ytick.minor.size": 1.8, "ytick.minor.width": 0.5, "axes.labelcolor": INK,
        "axes.titlecolor": INK, "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
        "legend.frameon": False, "savefig.dpi": 300, "axes.titlepad": 4,
    })


def clean_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)


def log_axis(ax, lo, hi, pad=1.15, include_one=True):
    """Log y-axis from lo to hi (with padding), 1 always shown, plain-number ticks."""
    if include_one:
        lo, hi = min(lo, 1.0), max(hi, 1.0)
    lo, hi = lo / pad, hi * pad
    dec = np.log10(hi / lo)
    subs = (1.0, 1.5, 2.0, 3.0, 5.0) if dec < 0.9 else ((1.0, 2.0, 5.0) if dec < 1.6 else
                                                     ((1.0, 3.0) if dec < 2.7 else (1.0,)))
    ticks = [t for t in (s_ * 10.0 ** k for k in range(-6, 4) for s_ in subs) if lo <= t <= hi]
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(lo, hi)


def fmt(r):
    return f"{r:.1f}" if r >= 9.95 else f"{r:.2f}"


def figure_gain(summary, out):
    s = summary[summary.metric == "test_selected"]
    fig, axs = plt.subplots(1, len(CONDS), figsize=(7.16, 2.7), sharey=True)
    w = 0.38
    for ax, (cond, title) in zip(axs, CONDS):
        for k, opt in enumerate(("GD", "SGD")):
            x0 = np.arange(len(DATASETS)) + (k - 0.5) * (w + 0.03)
            for x, (key, _) in zip(x0, DATASETS):
                r = s[(s.dataset == key) & (s.opt == opt) & (s.condition == cond)].iloc[0]
                R = float(r.ratio_BP_over_alpha)
                ax.bar(x, R - 1, bottom=1, width=w, color=C[("alpha", opt)], linewidth=0,
                       zorder=2)
                ax.text(x, max(R, 1) * 1.12, fmt(R), rotation=90, ha="center", va="bottom",
                        fontsize=8, color=INK2)
        ax.axhline(1, color=MUTED, lw=0.8, zorder=3)
        ax.set_title(title)
        ax.set_yscale("log")
        ax.set_ylim(0.85, 320)
        ax.yaxis.set_major_locator(FixedLocator([1, 10, 100]))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_yticklabels(["1", "10", "100"])
        ax.set_xticks(range(len(DATASETS)), [d[1].replace("Small ", "") .replace(" faces", "")
                                             for d in DATASETS], rotation=30, ha="right",
                      rotation_mode="anchor")
        ax.tick_params(axis="x", length=0, pad=2)
        ax.set_xlim(-0.6, len(DATASETS) - 0.4)
        ax.grid(True, axis="y", which="major", color=GRID, lw=0.6)
        clean_axes(ax)
    for ax in axs[1:]:
        ax.tick_params(axis="y", which="both", left=False)
        ax.spines["left"].set_visible(False)
    axs[0].set_ylabel("Test MSE of BP /\ntest MSE of $\\alpha$-loss")
    fig.legend(handles=[Patch(color=C[("alpha", "GD")], label="GD"),
                        Patch(color=C[("alpha", "SGD")], label="SGD")],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), handlelength=1.2,
               columnspacing=1.6)
    fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0, 1, 0.92))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_images_gain.{ext}")
    plt.close(fig)


COND_NAME = dict(clean="Clean data", asymmetric="Asymmetric outliers",
                 point_mass="Point-mass outliers", symmetric="Symmetric outliers")


def figure_training(curves, out, cond, layout="grid", warmup=200):
    c = curves[curves.condition == cond]
    if layout == "grid":     # 2 x 3: five data sets, legend in the sixth cell
        fig, grid = plt.subplots(2, 3, figsize=(7.16, 4.3))
        axs, leg_ax = list(grid.flat[:5]), grid.flat[5]
        below = {0: True, 1: True, 2: False, 3: False, 4: False}   # a panel below it?
    else:                    # 1 x 5 row
        fig, row = plt.subplots(1, len(DATASETS), figsize=(7.16, 2.45))
        axs, leg_ax, below = list(row), None, {i: False for i in range(5)}
    for i, (ax, (key, title)) in enumerate(zip(axs, DATASETS)):
        for loss in ("BP", "alpha"):
            for opt in ("GD", "SGD"):
                d = c[(c.dataset == key) & (c.opt == opt) & (c.loss == loss)].sort_values("epoch")
                d = d[d.epoch > 0]
                ax.fill_between(d.epoch, d.q25, d.q75, color=C[(loss, opt)], alpha=0.16, lw=0,
                                zorder=1)
                ax.plot(d.epoch, d["median"], color=C[(loss, opt)], ls=LS[opt], lw=1.3,
                        zorder=3)
        ax.axvline(warmup, color=MUTED, lw=0.7, ls=":", zorder=0)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlim(0, 2000)
        ax.set_xticks([0, 1000, 2000], ["0", "1000", "2000"])
        lo, hi = ax.get_ylim()                       # plain-number ticks, 3-5 per panel
        dec = np.log10(hi / lo)
        subs = (1.0, 2.0, 5.0) if dec < 1.5 else ((1.0, 3.0) if dec < 2.6 else (1.0,))
        ticks = [s * 10.0 ** k for k in range(-6, 2) for s in subs]
        below_lo = [t for t in ticks if t <= lo]
        if below_lo and lo / max(below_lo) < 1.35:   # extend slightly to show one more tick
            lo = max(below_lo) * 0.97
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=subs))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(lo, hi)
        ax.grid(True, axis="y", which="major", color=GRID, lw=0.6)
        clean_axes(ax)
        if not below[i]:
            ax.set_xlabel("Epoch")
    for i, ax in enumerate(axs):
        if layout != "grid" and i == 0 or layout == "grid" and i in (0, 3):
            ax.set_ylabel("Test MSE")
    handles = [Line2D([], [], color=C[(l, o)], ls=LS[o], lw=1.8,
                      label=("BP" if l == "BP" else r"$\alpha$-loss") + f", {o}")
               for l in ("BP", "alpha") for o in ("GD", "SGD")]
    handles.append(Line2D([], [], color=MUTED, lw=1.0, ls=":", label="End of warm-up"))
    if leg_ax is not None:
        leg_ax.axis("off")
        leg = leg_ax.legend(handles=handles, loc="center", ncol=1, handlelength=2.6,
                            labelspacing=0.75, title=COND_NAME[cond], fontsize=9.5,
                            title_fontsize=10)
        leg._legend_box.align = "left"
        fig.tight_layout(pad=0.3, w_pad=1.0, h_pad=1.1)
    else:
        fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0),
                   handlelength=2.2, columnspacing=1.3)
        fig.tight_layout(pad=0.3, w_pad=0.4, rect=(0, 0, 1, 0.9))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_images_training.{ext}")
    plt.close(fig)


def figure_all(curves, summary, out, conds=("asymmetric", "point_mass", "symmetric"),
               warmup=200):
    """All data sets in one panel: test MSE divided by that of BP on clean data (same data
    set and optimizer, selected checkpoint), geometric mean over the 5 data sets x the
    contaminated conditions of the median (line) and of the quartiles (band) over the seeds,
    as in Fig. 5.  A curve stops when one of its cases has no run left (divergence)."""
    s = summary[(summary.metric == "test_selected") & (summary.condition == "clean")]
    ref = {(r.dataset, r.opt): r.median_BP for r in s.itertuples()}
    c = curves[curves.epoch > 0]
    fig, ax = plt.subplots(figsize=(3.5, 2.75))
    for loss in ("BP", "alpha"):
        for opt in ("GD", "SGD"):
            logs = []
            for key, _ in DATASETS:
                for cond in conds:
                    d = c[(c.dataset == key) & (c.opt == opt) & (c.condition == cond)
                          & (c.loss == loss)].set_index("epoch")[["median", "q25", "q75"]]
                    logs.append(np.log(d / ref[(key, opt)]))
            stacked = pd.concat(logs, axis=1, keys=range(len(logs))).dropna()  # all cases present
            gm = {q: np.exp(stacked.xs(q, axis=1, level=1).mean(axis=1))
                  for q in ("median", "q25", "q75")}
            e = stacked.index.to_numpy()
            ax.fill_between(e, gm["q25"], gm["q75"], color=C[(loss, opt)], alpha=0.18, lw=0,
                            zorder=1)
            ax.plot(e, gm["median"], color=C[(loss, opt)], ls=LS[opt], lw=1.4, zorder=3)
    ax.axhline(1, color=INK2, lw=0.8, zorder=2)
    ax.axvline(warmup, color=MUTED, lw=0.8, ls=":", zorder=0)
    ax.set_xlim(0, 2000)
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ys = np.concatenate([l.get_ydata() for l in ax.get_lines() if len(l.get_ydata()) > 2])
    log_axis(ax, np.nanmin(ys), np.nanmax(ys))
    ax.grid(True, axis="y", which="major", color=GRID, lw=0.6)
    clean_axes(ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test MSE / test MSE of BP\ntrained on clean data")
    handles = [Line2D([], [], color=C[(l, o)], ls=LS[o], lw=1.8,
                      label=("BP" if l == "BP" else r"$\alpha$-loss") + f", {o}")
               for l in ("BP", "alpha") for o in ("GD", "SGD")]
    handles += [Line2D([], [], color=INK2, lw=0.8, label="Reference: BP trained on clean data"),
                Line2D([], [], color=MUTED, lw=1.0, ls=":", label="End of warm-up")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0),
               fontsize=8, handlelength=2.2, columnspacing=0.9, handletextpad=0.5)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.83))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_images_all.{ext}")
    plt.close(fig)


# One colour per data set (validated: all pairs separable, also with colour-vision deficiency);
# no blue or green, which mean BP and the alpha-loss in the other figures.
DS_COL = dict(mnist="#6250d6", frey="#D55E00", norb="#E69F00", cifar10="#a6488f",
              svhn="#e87ba4")


def figure_datasets(curves, summary, out, cond="point_mass", warmup=200):
    """All five data sets in one plot per optimizer: test MSE divided by that of BP on clean
    data, median over the seeds; colour = data set, dashed = BP, solid = alpha-loss."""
    s = summary[(summary.metric == "test_selected") & (summary.condition == "clean")]
    ref = {(r.dataset, r.opt): r.median_BP for r in s.itertuples()}
    c = curves[(curves.epoch > 0) & (curves.condition == cond)]
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 3.0), sharey=True)
    for ax, opt in zip(axs, ("GD", "SGD")):
        ends = []
        for key, name in DATASETS:
            for loss in ("alpha", "BP"):
                d = c[(c.dataset == key) & (c.opt == opt) & (c.loss == loss)].sort_values("epoch")
                y = d["median"] / ref[(key, opt)]
                ax.plot(d.epoch, y, color=DS_COL[key], lw=1.5 if loss == "alpha" else 1.3,
                        ls="-" if loss == "alpha" else (0, (4, 1.6)), zorder=3)
                if loss == "BP":
                    ends.append([np.log10(y.iloc[-1]), d.epoch.iloc[-1], name, DS_COL[key]])
        # direct labels at the end of the BP curves, pushed apart vertically
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.15)
        for ly, x, name, col in ends:
            ax.text(x + 30, 10 ** ly, name.replace("Small ", ""), color=col, fontsize=8.5,
                    va="center", ha="left", clip_on=False)
        ax.axhline(1, color=INK2, lw=0.8, zorder=2)
        ax.axvline(warmup, color=MUTED, lw=0.8, ls=":", zorder=0)
        ax.set_xlim(0, 2000)
        ax.set_xticks([0, 500, 1000, 1500, 2000])
        ax.grid(True, axis="y", which="major", color=GRID, lw=0.6)
        ax.set_title({"GD": "Gradient descent (GD)", "SGD": "Stochastic gradient descent (SGD)"}[opt])
        ax.set_xlabel("Epoch")
        clean_axes(ax)
    ys = np.concatenate([np.asarray(l.get_ydata(), float) for a_ in axs for l in a_.get_lines()
                         if len(l.get_ydata()) > 2])
    for ax in axs:                                  # both panels share the vertical scale
        log_axis(ax, np.nanmin(ys), np.nanmax(ys))
    axs[0].set_ylabel("Test MSE / test MSE of BP\ntrained on clean data")
    data_h = [Line2D([], [], color=DS_COL[k], lw=2.2, label=n) for k, n in DATASETS]
    style_h = [Line2D([], [], color=INK2, lw=1.3, ls=(0, (4, 1.6)), label="BP"),
               Line2D([], [], color=INK2, lw=1.5, label=r"$\alpha$-loss"),
               Line2D([], [], color=INK2, lw=0.8, label="Reference: BP trained on clean data"),
               Line2D([], [], color=MUTED, lw=1.0, ls=":", label="End of warm-up")]
    fig.legend(handles=data_h, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.8, columnspacing=1.5)
    fig.legend(handles=style_h, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.925),
               handlelength=2.4, columnspacing=1.5)
    fig.tight_layout(pad=0.3, w_pad=2.2, rect=(0, 0, 0.98, 0.83))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_images_datasets.{ext}")
    plt.close(fig)


def figure_split(curves, summary, out, cond="point_mass", warmup=200, layout="2x2",
                 width="column"):
    """BP (squared error) and the alpha-loss in separate panels, all five data sets in each:
    test MSE divided by that of BP on clean data, median over the seeds.
    layout '2x2': rows BP / alpha-loss, columns GD / SGD (5 curves per panel);
    layout '1x2': BP / alpha-loss, GD solid and SGD dashed (10 curves per panel).
    width 'column' (3.5 in, one column of the paper) or 'full' (7.16 in, figure*).
    The alpha-loss is drawn from its first checkpoint after the warm-up (shaded)."""
    s = summary[(summary.metric == "test_selected") & (summary.condition == "clean")]
    ref = {(r.dataset, r.opt): r.median_BP for r in s.itertuples()}
    c = curves[(curves.epoch > 0) & (curves.condition == cond)]
    col = width == "column"
    rc = {"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
          "ytick.labelsize": 7, "legend.fontsize": 7.2, "axes.titlepad": 2.5,
          "xtick.major.size": 2.2, "ytick.major.size": 2.2, "ytick.minor.size": 1.3} if col else {}
    with plt.rc_context(rc):
        if layout == "2x2":
            fig, grid = plt.subplots(2, 2, figsize=(3.5, 3.9) if col else (7.16, 4.6),
                                     sharex=True)
            cells = [(grid[0, 0], "BP", ["GD"]), (grid[0, 1], "BP", ["SGD"]),
                     (grid[1, 0], "alpha", ["GD"]), (grid[1, 1], "alpha", ["SGD"])]
        else:
            fig, row = plt.subplots(1, 2, figsize=(3.5, 2.3) if col else (7.16, 3.0))
            cells = [(row[0], "BP", ["GD", "SGD"]), (row[1], "alpha", ["GD", "SGD"])]
        lw = 1.05 if col else 1.5
        yrange = {}                                # one vertical scale per loss (row)
        for ax, loss, opts in cells:
            for key, name in DATASETS:
                for opt in opts:
                    d = c[(c.dataset == key) & (c.opt == opt) & (c.loss == loss)].sort_values("epoch")
                    if loss == "alpha":            # from the first checkpoint of the alpha-loss
                        d = d[d.epoch > warmup]
                    y = (d["median"] / ref[(key, opt)]).to_numpy()
                    ax.plot(d.epoch, y, color=DS_COL[key], lw=lw,
                            ls=LS[opt] if len(opts) > 1 else "-", zorder=3)
                    if np.isfinite(y).any():
                        lo, hi = yrange.get(loss, (np.inf, -np.inf))
                        yrange[loss] = (min(lo, np.nanmin(y)), max(hi, np.nanmax(y)))
        for ax, loss, opts in cells:
            ax.axhline(1, color=INK2, lw=0.7, zorder=2)
            if loss == "alpha":
                ax.axvspan(0, warmup, color="#efeee9", lw=0, zorder=0)
                ax.text(warmup / 2, 0.5, "warm-up", transform=ax.get_xaxis_transform(),
                        rotation=90, ha="center", va="center", fontsize=6.5 if col else 8,
                        color=MUTED)
            else:
                ax.axvline(warmup, color=MUTED, lw=0.7, ls=":", zorder=0)
            log_axis(ax, *yrange[loss])
            ax.set_xlim(0, 2000)
            ax.set_xticks([0, 1000, 2000] if col else [0, 500, 1000, 1500, 2000])
            ax.grid(True, axis="y", which="major", color=GRID, lw=0.5)
            clean_axes(ax)
            who = ("BP" if col else "BP (squared error)") if loss == "BP" else r"$\alpha$-loss"
            ax.set_title(who + (f", {opts[0]}" if len(opts) == 1 else ""))
            if layout != "2x2" or loss == "alpha":
                ax.set_xlabel("Epoch")
            if layout != "2x2" or opts == ["GD"]:
                ax.set_ylabel("Test MSE / BP on clean data" if col else
                              "Test MSE / test MSE of BP\ntrained on clean data")
        names = {"norb": "NORB"} if col else {}
        handles = [Line2D([], [], color=DS_COL[k], lw=2.0, label=names.get(k, n))
                   for k, n in DATASETS]
        if layout != "2x2":
            handles += [Line2D([], [], color=INK2, lw=1.2, ls=LS[o], label=o)
                        for o in ("GD", "SGD")]
        handles.append(Line2D([], [], color=INK2, lw=0.8,
                              label="BP on clean data" if col else
                              "Reference: BP trained on clean data"))
        title = (f"Training data: 20% {COND_NAME[cond].lower()}" if col else
                 f"Training data with 20% {COND_NAME[cond].lower()}; test data clean")
        fig.text(0.5, 0.995, title, ha="center", va="top", fontsize=8 if col else 9.5, color=INK)
        if col:
            fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.955),
                       handlelength=1.5, columnspacing=0.9, handletextpad=0.4, labelspacing=0.25)
            top = 0.86 if layout == "2x2" else 0.72
            fig.tight_layout(pad=0.2, w_pad=0.6, h_pad=0.5, rect=(0, 0, 1, top))
        else:
            fig.legend(handles=handles, loc="upper center",
                       ncol=len(handles) if layout == "2x2" else 4,
                       bbox_to_anchor=(0.5, 0.965 if layout == "2x2" else 0.94),
                       handlelength=1.8, columnspacing=1.2, fontsize=8.5)
            top = 0.885 if layout == "2x2" else 0.74
            fig.tight_layout(pad=0.3, w_pad=1.2, h_pad=1.0, rect=(0, 0, 1, top))
        for ext in ("pdf", "png"):
            fig.savefig(out / f"fig_images_split.{ext}")
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results", default="results_images")
    p.add_argument("--out", default=None, help="default: RESULTS/figures")
    p.add_argument("--condition", default="point_mass",
                   choices=["clean", "asymmetric", "point_mass", "symmetric"],
                   help="contamination shown in the training-curve figure")
    p.add_argument("--layout", default="grid", choices=["grid", "row"],
                   help="training-curve figure: 2 x 3 grid (default) or one row of 5 panels")
    p.add_argument("--split-width", default="column", choices=["column", "full"],
                   help="fig_images_split: one column of the paper (3.5 in, default) or the "
                        "full page width (7.16 in)")
    p.add_argument("--split-layout", default="2x2", choices=["2x2", "1x2"],
                   help="fig_images_split: rows BP / alpha-loss and columns GD / SGD (default), "
                        "or two panels with GD and SGD together")
    a = p.parse_args()
    res = Path(a.results)
    out = Path(a.out) if a.out else res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    style()
    figure_gain(pd.read_csv(res / "summary.csv"), out)
    curves, summary = pd.read_csv(res / "curves.csv"), pd.read_csv(res / "summary.csv")
    figure_training(curves, out, a.condition, a.layout)
    figure_all(curves, summary, out)
    figure_datasets(curves, summary, out, a.condition)
    figure_split(curves, summary, out, a.condition, layout=a.split_layout, width=a.split_width)
    print(f"wrote fig_images_split, fig_images_datasets, fig_images_all, fig_images_training, "
          f"fig_images_gain "
          f"(.pdf, .png) in {out}")


if __name__ == "__main__":
    main()
