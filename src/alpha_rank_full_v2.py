#!/usr/bin/env python3
"""alpha_rank_full_v2.py -- robust nonlinear regression with alpha-divergence,
GD vs mini-batch SGD, residual-scale (sigma) strategies, and small-|W| ranked
sparse weight updates.

Network 1 -> 21 -> 14 -> 7 -> 1, sigmoid hidden units, linear output.
Targets Gaussian, Sigmoid, Sine.  Conditions: clean control and 20 %
asymmetric / point-mass / symmetric label contamination.  20,000 epochs.

TWO STAGES, run in this order
  --stage sigma    Does the alpha-loss work, and with which residual scale?
                   GD MSE, SGD MSE, and alpha-GD / alpha-SGD with
                     early    : sigma estimated once after --early-warmup epochs
                     late     : sigma estimated once after --warmup epochs (~10 %)
                     periodic : sigma estimated after --warmup, then re-estimated
                                every --sigma-every epochs.  Each re-estimate
                                CHANGES THE OBJECTIVE being minimised; this is a
                                separate algorithm, not annealing of alpha.
  --stage ranking  With ONE sigma strategy (--sigma-strategy):
                     GD MSE, SGD MSE, alpha-GD, alpha-SGD,
                     ranked alpha-GD/SGD  (smallest |W| within each layer),
                     uniform alpha-GD/SGD (matched control: same k per layer)
                   at --fractions 10/30/50/70/100 %.  100 % is alpha itself
                   (bit-identical, verified in the self-test, not re-run).

Loss (residual r_i = g(x_i) - y_i, batch B, tau = 1 - alpha):
    L_alpha = sigma^2/tau * (1/B) sum_i [1 - exp(-tau r_i^2 / (2 sigma^2))]
    dL_alpha/dg_i = (1/B) r_i q_i,   q_i = exp(-tau r_i^2 / (2 sigma^2))
alpha = 1 recovers half-MSE exactly, so both losses share one step-size scale.
  GD : theta <- theta - eta (1/n) sum_{i=1..n} grad l_i
  SGD: theta <- theta - eta (1/q) sum_{i in B_t} grad l_i, mini-batches of size q
       reshuffled every epoch, constant step size.
sigma = 1.4826 * MAD of the full-training-set residuals, floored at
--sigma-floor; never estimated per mini-batch.  During the warmup alpha methods
use the half-MSE gradient on all weights; alpha-loss and sparse selection begin
together at the first update of epoch `warmup`.

Bookkeeping conventions
  * A checkpoint at epoch e is evaluated BEFORE the updates of epoch e.  The
    checkpoint at epoch == warmup has therefore had ZERO alpha-gradient updates.
  * For every run three checkpoints are recorded, all chosen by VALIDATION only:
      overall     best validation score over all checkpoints
      post_alpha  best validation score among checkpoints with >= 1 alpha update
      final       the last epoch
    plus best epochs, alpha updates before each, and whether the overall best
    lies in the MSE warmup.  For MSE methods post_alpha is undefined (NaN); the
    derived metric `test_alpha_phase` equals post_alpha for alpha methods and
    the overall best for MSE methods.
  * sigma_final is the sigma used by the LAST optimiser update; q_outlier_final
    and q_inlier_final are the mean weights q_i at the final parameters under
    that same sigma.  sigma is never re-estimated after the last update.
  * Cost is reported separately as epochs, optimiser steps, alpha updates,
    backprop calls, per-example gradient evaluations and wall-clock seconds.
    Sparse updates still compute the FULL gradient: no computational saving.

Protocol
  * Learning rates are tuned separately for EVERY method (each sparse fraction
    included) on pilot datasets with disjoint random streams, using the same
    schedule shrunk proportionally to --tune-epochs.  Tuning criterion: mean
    over pilot seeds of log(best validation score), using the post-alpha
    checkpoint for alpha methods and the overall checkpoint for MSE methods.
  * Validation is clean by default (--val-mode clean): an explicit simulation
    assumption that may fail for real contaminated data.  --val-mode
    contaminated contaminates the validation set the same way and scores by
    the median squared residual.
  * Test MSE: exact clean target on a 401-point grid, original units.  Used only
    for reporting, never for checkpoint or learning-rate selection.
  * Paired design: per seed every method shares training and validation data,
    initial weights and SGD mini-batch order; masks use their own stream.
  * Initialisation W ~ N(0, 1/fan_in): VARIANCE 1/m (std 1/sqrt(m)).
  * Statistics: paired Wilcoxon signed-rank tests, Holm correction within each
    pre-declared family (see families()), with effect sizes: median ratio,
    bootstrap 95 % CI of the median ratio, and matched-pairs rank-biserial
    correlation.  A non-significant test is NOT evidence of equivalence.

Caching: one gzipped JSON file per tuning job and per (function, condition,
seed), in a folder named after a digest of every setting AND the program
code; each file also stores the digest and is verified on load.  Interrupted
runs resume; corrupt or foreign files are quarantined and re-run.

Requirements: numpy pandas scipy matplotlib

Commands
  python alpha_rank_full_v2.py --self-test                        # all verification tests
  python alpha_rank_full_v2.py --stage sigma --quick --show-plots # 1-min smoke run
  python alpha_rank_full_v2.py --stage sigma --workers 32 --out results_sigma
  python alpha_rank_full_v2.py --stage ranking --sigma-strategy periodic \\
      --workers 32 --out results_ranking_periodic
  python alpha_rank_full_v2.py --stage sigma --out results_sigma --plots-only --show-plots
  python alpha_rank_full_v2.py --analyze-cache results_sigma/cache/sigma_<digest> --out replot
      # re-plot an existing cache after editing plotting code (no training, no mixing)
  # resume / limit work per call: add --max-jobs 10 and rerun the same command
  # Jupyter: %run alpha_rank_full_v2.py --stage sigma --quick --show-plots
  #          (keep --workers 1 inside Jupyter on macOS / Windows)
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import io
import json
import math
import multiprocessing
import os
import platform
import shutil
import sys
import tempfile
import time
import types
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # tiny matrices: parallelism comes from --workers

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata, wilcoxon

FUNCTIONS = ("Gaussian", "Sigmoid", "Sine")
CONDITIONS = ("clean", "asymmetric", "point_mass", "symmetric")
OPTIMIZERS = ("GD", "SGD")
SIGMA_STRATEGIES = ("early", "late", "periodic")
WIDTHS = (1, 21, 14, 7, 1)
BASE_SEED = 20261106
GRID = np.linspace(-1.0, 1.0, 401)
XG = (4.0 * GRID)[:, None]
CACHE_FORMAT = 3
OPERATIONAL = {"out", "seeds", "workers", "max_jobs", "plots_only", "show_plots",
               "self_test", "quick", "dpi", "analyze_cache"}
METRICS = ("test_selected", "test_alpha_phase", "test_final")


# ==========================================================================
# Data
# ==========================================================================
def target(name, x):
    x = np.asarray(x, dtype=float)
    if name == "Gaussian":
        return np.exp(-x * x / (2 * 0.5 ** 2))
    if name == "Sigmoid":
        return expit(4 * x)
    if name == "Sine":
        return 0.5 * (np.sin(np.pi * x) + 1)
    raise ValueError(name)


def y_internal(y):
    return 0.1 + 0.8 * np.asarray(y, dtype=float)


def y_original(y):
    return (np.asarray(y, dtype=float) - 0.1) / 0.8


def _rng(*key):
    """Independent reproducible stream per purpose (SeedSequence keys)."""
    return np.random.default_rng(np.random.SeedSequence([BASE_SEED, *map(int, key)]))


def corrupt(y, idx, condition, rng):
    y = y.copy()
    if condition == "asymmetric":
        y[idx] += 0.80
    elif condition == "point_mass":
        y[idx] = 1.50 + rng.normal(0, 0.008, len(idx))
    elif condition == "symmetric":
        y[idx] += rng.normal(0, 0.80, len(idx))
    elif condition != "clean":
        raise ValueError(condition)
    return y


def make_data(function, condition, seed, args, pilot=False):
    """Inputs, base noise and outlier positions are shared by the four
    conditions of one (function, seed); pilot data use disjoint streams.
    `true` (test grid target) is consumed only by evaluate()'s test output."""
    f, c, p = FUNCTIONS.index(function), CONDITIONS.index(condition), int(pilot)
    r = _rng(10, p, f, seed)
    x = np.sort(r.uniform(-1, 1, args.n))
    y = y_internal(target(function, x)) + r.normal(0, args.noise, args.n)
    idx = _rng(11, p, f, seed).permutation(args.n)[:int(round(args.n * args.contamination))]
    bad = np.zeros(args.n, dtype=bool)
    if condition != "clean":
        bad[idx] = True
        y = corrupt(y, idx, condition, _rng(12, p, f, seed, c))
    rv = _rng(13, p, f, seed)
    xv = rv.uniform(-1, 1, args.n_val)
    yv = y_internal(target(function, xv)) + rv.normal(0, args.noise, args.n_val)
    if args.val_mode == "contaminated" and condition != "clean":
        iv = _rng(14, p, f, seed).permutation(args.n_val)[:int(round(args.n_val * args.contamination))]
        yv = corrupt(yv, iv, condition, _rng(15, p, f, seed, c))
    return dict(X=4 * x[:, None], y=y, x=x, bad=bad, Xv=4 * xv[:, None], yv=yv,
                true=target(function, GRID))


# ==========================================================================
# Network, alpha-divergence loss and gradient
# ==========================================================================
def init_params(seed, pilot=False):
    """W ~ N(0, 1/fan_in): variance 1/m, standard deviation 1/sqrt(m); b = 0."""
    rng = _rng(20, int(pilot), seed)
    return [[rng.normal(0, 1 / np.sqrt(WIDTHS[j]), (WIDTHS[j], WIDTHS[j + 1])),
             np.zeros(WIDTHS[j + 1])] for j in range(len(WIDTHS) - 1)]


def forward(theta, X):
    acts = [np.asarray(X, dtype=float)]
    for j, (W, b) in enumerate(theta):
        h = acts[-1] @ W + b
        acts.append(expit(h) if j < len(theta) - 1 else h)
    return acts[-1].ravel(), acts


def predict(theta, X):
    return forward(theta, X)[0]


def alpha_weights(r, alpha, sigma):
    """q_i = exp(-(1-alpha) r_i^2 / (2 sigma^2)); all ones at alpha = 1."""
    tau = 1.0 - alpha
    r = np.asarray(r, dtype=float)
    return np.exp(-0.5 * tau * np.square(r / sigma)) if tau > 1e-12 else np.ones_like(r)


def loss_and_grad(theta, X, y, alpha=1.0, sigma=1.0):
    """L_alpha = sigma^2/tau * mean(1 - exp(-tau r^2/(2 sigma^2))), half-MSE at
    alpha = 1.  sigma is held fixed while differentiating."""
    if not (0 < alpha <= 1 and sigma > 0):
        raise ValueError("alpha in (0, 1] and sigma > 0 are required")
    out, acts = forward(theta, X)
    r = out - np.asarray(y, dtype=float)
    tau = 1.0 - alpha
    if tau > 1e-12:
        z = -0.5 * tau * np.square(r / sigma)
        weight = np.exp(z)
        obj = float(sigma ** 2 / tau * np.mean(-np.expm1(z)))
    else:
        weight = np.ones_like(r)
        obj = float(0.5 * np.mean(np.square(r)))
    delta = ((r * weight) / len(r))[:, None]
    grads = [None] * len(theta)
    for j in reversed(range(len(theta))):
        W, _ = theta[j]
        grads[j] = (acts[j].T @ delta, delta.sum(axis=0))
        if j:
            a = acts[j]
            delta = (delta @ W.T) * a * (1 - a)
    return obj, grads


def flat(grads):
    return np.concatenate([g.ravel() for pair in grads for g in pair])


def scale_from_residuals(residuals, floor):
    r = np.asarray(residuals, dtype=float)
    return max(float(1.4826 * np.median(np.abs(r - np.median(r)))), float(floor))


# ==========================================================================
# Sparse weight selection
# ==========================================================================
def selection_mask(W, fraction, kind, rng, exploration=0.2, rank_strength=3.0,
                   rank_mode="probabilistic"):
    """Choose exactly k = ceil(fraction * size) weights of ONE layer.
      ranked / strict        the k smallest |W|
      ranked / probabilistic weighted sampling without replacement (Gumbel
                             top-k, same law as sequential sampling) favouring
                             small |W|; `exploration` keeps every p > 0
      uniform                k weights uniformly at random (matched control)
    Returns (mask, mean normalised within-layer rank of chosen, 0 = smallest)."""
    w = np.abs(np.asarray(W, dtype=float)).ravel()
    n = w.size
    k = max(1, min(n, int(math.ceil(fraction * n - 1e-12))))
    if k == n:
        return np.ones(W.shape, dtype=bool), 0.5
    order = np.argsort(w, kind="stable")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n)
    if kind == "uniform":
        chosen = rng.permutation(n)[:k]
    elif kind == "ranked" and rank_mode == "strict":
        chosen = order[:k]
    elif kind == "ranked" and rank_mode == "probabilistic":
        scores = np.power(n - ranks.astype(float), rank_strength)
        p = (1 - exploration) * scores / scores.sum() + exploration / n
        keys = np.log(p) + rng.gumbel(size=n)
        chosen = np.argpartition(-keys, k - 1)[:k]
    else:
        raise ValueError((kind, rank_mode))
    mask = np.zeros(n, dtype=bool)
    mask[chosen] = True
    return mask.reshape(W.shape), float(np.mean(ranks[chosen] / (n - 1)))


def update(theta, grads, lr, kind=None, fraction=1.0, rng=None, exploration=0.2,
           rank_strength=3.0, rank_mode="probabilistic"):
    """Gradient step. Only weight matrices are masked; biases always update."""
    k_total, rank_sum, size_total = 0, 0.0, 0
    for (W, b), (dW, db) in zip(theta, grads):
        if kind is None:
            W -= lr * dW
        else:
            mask, mean_rank = selection_mask(W, fraction, kind, rng, exploration,
                                             rank_strength, rank_mode)
            k = int(mask.sum())
            W -= lr * dW * mask
            k_total += k
            rank_sum += k * mean_rank
            size_total += W.size
        b -= lr * db
    return k_total, rank_sum, size_total


# ==========================================================================
# Methods, schedules, checkpoints, training
# ==========================================================================
def methods_for(stage, args):
    specs = [dict(name=f"{o} MSE", opt=o, loss="MSE", sigma=None, sparse=None, fraction=1.0)
             for o in OPTIMIZERS]
    if stage == "sigma":
        for o in OPTIMIZERS:
            for s in args.sigma_strategies:
                specs.append(dict(name=f"{o} Alpha[{s}]", opt=o, loss="Alpha", sigma=s,
                                  sparse=None, fraction=1.0))
    else:
        s = args.sigma_strategy
        for o in OPTIMIZERS:
            specs.append(dict(name=f"{o} Alpha", opt=o, loss="Alpha", sigma=s, sparse=None,
                              fraction=1.0))
            for f in args.fractions:
                if f < 1:
                    for kind in ("ranked", "uniform"):
                        specs.append(dict(name=f"{o} {kind.capitalize()} {round(100 * f)}%",
                                          opt=o, loss="Alpha", sigma=s, sparse=kind,
                                          fraction=float(f)))
    return specs


def schedule(args, total_epochs):
    """All lengths are defined relative to --epochs; tuning shrinks them in
    proportion so it sees the same shape of schedule."""
    def s(e):
        return max(1, int(round(e * total_epochs / args.epochs)))
    return dict(epochs=int(total_epochs),
                warmup={"early": s(args.early_warmup), "late": s(args.warmup),
                        "periodic": s(args.warmup)},
                sigma_every=s(args.sigma_every), eval_every=s(args.eval_every))


def validation_criterion(args):
    return ("mean squared residual on CLEAN validation set (internal scale)"
            if args.val_mode == "clean" else
            "median squared residual on CONTAMINATED validation set (internal scale)")


def evaluate(theta, data, args):
    """Returns (validation score, test MSE, test curve).  Selection code may use
    ONLY the first element."""
    rv2 = np.square(predict(theta, data["Xv"]) - data["yv"])
    score = float(np.mean(rv2) if args.val_mode == "clean" else np.median(rv2))
    curve = y_original(predict(theta, XG))
    return score, float(np.mean(np.square(curve - data["true"]))), curve


class CheckpointTracker:
    """Validation-only checkpoint bookkeeping.  `alpha_updates` is the number of
    alpha-gradient optimiser updates performed BEFORE the checkpoint."""

    def __init__(self):
        self.overall = self.post_alpha = self.final = None

    def observe(self, epoch, score, test, curve, alpha_updates):
        rec = dict(epoch=int(epoch), score=float(score), test=float(test), curve=curve,
                   alpha_updates=int(alpha_updates))
        if np.isfinite(score):
            if self.overall is None or score < self.overall["score"]:
                self.overall = rec
            if alpha_updates > 0 and (self.post_alpha is None or score < self.post_alpha["score"]):
                self.post_alpha = rec
        self.final = rec

    @staticmethod
    def get(rec, key, default=np.nan):
        return default if rec is None else rec[key]


def gradient_noise(theta, data, alpha_now, sigma_now, parts):
    """Mean squared distance between mini-batch and full gradients of the loss
    currently being optimised, on a fixed partition (diagnostic only)."""
    X, y = data["X"], data["y"]
    g = flat(loss_and_grad(theta, X, y, alpha_now, sigma_now)[1])
    noise = np.mean([np.sum(np.square(flat(loss_and_grad(theta, X[p], y[p], alpha_now,
                                                          sigma_now)[1]) - g)) for p in parts])
    return float(noise), float(np.sum(g * g))


PATH_COLUMNS = ("epoch", "checkpoint_candidate", "optimizer_steps", "alpha_updates",
                "val_score", "test_mse", "sigma", "q_outlier", "q_inlier", "grad_noise",
                "grad_norm2")


def train(spec, data, lr, sched, args, seed, pilot=False, record=False):
    theta = init_params(seed, pilot)
    batch_rng = _rng(30, int(pilot), seed)   # identical for every SGD method
    mask_rng = _rng(40, int(pilot), seed)    # never touches data order
    n, E = len(data["y"]), sched["epochs"]
    is_alpha, sgd = spec["loss"] == "Alpha", spec["opt"] == "SGD"
    warm = sched["warmup"][spec["sigma"]] if is_alpha else None
    periodic = is_alpha and spec["sigma"] == "periodic"
    step = args.batch if sgd else n
    parts = np.array_split(_rng(50, seed).permutation(n), max(1, math.ceil(n / args.batch)))
    # Checkpoint candidates are the same grid in tuning and evaluation; the extra early
    # epochs of recorded runs are diagnostics only and can never be selected.
    select_epochs = set(range(0, E + 1, sched["eval_every"])) | {E}
    evals = select_epochs | ({e for e in (1, 2, 5, 10, 20, 50) if e < E} if record else set())
    tracker = CheckpointTracker()
    sigma = None                 # sigma currently in force for alpha updates
    sigma_last_used = None       # sigma used by the most recent alpha update
    sigma_history = []           # (epoch, sigma) at every (re-)estimation
    cols = {k: [] for k in PATH_COLUMNS}
    k_tot = rank_tot = size_tot = steps = alpha_updates = examples = 0
    diverged, epochs_done = False, 0
    start = time.perf_counter()
    for epoch in range(E + 1):
        updating = epoch < E
        # sigma is (re-)estimated only when an update follows, never after the last one
        if is_alpha and updating and epoch >= warm and (
                sigma is None or (periodic and (epoch - warm) % sched["sigma_every"] == 0)):
            sigma = scale_from_residuals(predict(theta, data["X"]) - data["y"], args.sigma_floor)
            sigma_history.append((epoch, sigma))
        active = is_alpha and updating and epoch >= warm
        alpha_now, sigma_now = (args.alpha, sigma) if active else (1.0, 1.0)
        if epoch in evals:
            score, test, curve = evaluate(theta, data, args)
            if epoch in select_epochs:
                tracker.observe(epoch, score, test, curve, alpha_updates)
            if record:
                # Diagnostics use the loss in force from this checkpoint on: alpha with the
                # current sigma once sigma exists (at epoch E: the sigma of the last update).
                diag_sigma = sigma if is_alpha and sigma is not None else None
                q = (alpha_weights(predict(theta, data["X"]) - data["y"], args.alpha, diag_sigma)
                     if diag_sigma is not None else None)
                gn, g2 = gradient_noise(theta, data,
                                        args.alpha if diag_sigma is not None else 1.0,
                                        diag_sigma if diag_sigma is not None else 1.0, parts)
                bad = data["bad"]
                for key, val in (
                        ("epoch", epoch), ("checkpoint_candidate", epoch in select_epochs),
                        ("optimizer_steps", steps),
                        ("alpha_updates", alpha_updates), ("val_score", score), ("test_mse", test),
                        ("sigma", np.nan if diag_sigma is None else diag_sigma),
                        ("q_outlier", float(q[bad].mean()) if q is not None and bad.any() else np.nan),
                        ("q_inlier", float(q[~bad].mean()) if q is not None else np.nan),
                        ("grad_noise", gn), ("grad_norm2", g2)):
                    cols[key].append(val)
        if not updating:
            break
        order = batch_rng.permutation(n) if sgd else np.arange(n)
        kind = spec["sparse"] if active else None
        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(0, n, step):
                ids = order[i:i + step]
                _, grads = loss_and_grad(theta, data["X"][ids], data["y"][ids], alpha_now, sigma_now)
                k, rs, st = update(theta, grads, lr, kind, spec["fraction"], mask_rng,
                                   args.exploration, args.rank_strength, args.rank_mode)
                k_tot, rank_tot, size_tot = k_tot + k, rank_tot + rs, size_tot + st
                steps += 1
                examples += len(ids)
                if active:
                    alpha_updates += 1
                    sigma_last_used = sigma
        epochs_done = epoch + 1
        if not all(np.isfinite(z).all() for W, b in theta for z in (W, b)):
            diverged = True
            break
    q_out = q_in = msr = np.nan
    if not diverged:
        resid = predict(theta, data["X"]) - data["y"]
        msr = float(np.mean(np.square(resid)))
        if sigma_last_used is not None:
            q = alpha_weights(resid, args.alpha, sigma_last_used)
            q_out = float(q[data["bad"]].mean()) if data["bad"].any() else np.nan
            q_in = float(q[~data["bad"]].mean())
    ov, pa, fi = tracker.overall, tracker.post_alpha, (None if diverged else tracker.final)
    get = CheckpointTracker.get
    test_alpha_phase = get(pa, "test") if is_alpha else get(ov, "test")
    final = dict(
        method=spec["name"], opt=spec["opt"], loss=spec["loss"], sigma_strategy=spec["sigma"],
        sparse=spec["sparse"], fraction=spec["fraction"], lr=lr,
        validation_criterion=validation_criterion(args),
        # overall-best validation checkpoint
        test_selected=get(ov, "test"), val_selected=get(ov, "score"),
        best_epoch=get(ov, "epoch", -1), alpha_updates_at_best=get(ov, "alpha_updates", 0),
        selected_in_warmup=bool(is_alpha and ov is not None and ov["alpha_updates"] == 0),
        # best validation checkpoint after >= 1 alpha update
        test_post_alpha=get(pa, "test"), val_post_alpha=get(pa, "score"),
        best_post_alpha_epoch=get(pa, "epoch", -1),
        alpha_updates_at_post_alpha_best=get(pa, "alpha_updates", 0),
        test_alpha_phase=test_alpha_phase,
        # final epoch
        test_final=get(fi, "test"), val_final=get(fi, "score"), train_msr_final=msr,
        # residual scale actually used
        warmup_epoch=warm if is_alpha else np.nan, sigma_updates=len(sigma_history),
        sigma_first=sigma_history[0][1] if sigma_history else np.nan,
        sigma_final=np.nan if sigma_last_used is None else sigma_last_used,
        q_outlier_final=q_out, q_inlier_final=q_in,
        # sparse selection diagnostics
        selected_fraction_actual=k_tot / size_tot if size_tot else 1.0,
        mean_selected_rank=rank_tot / k_tot if k_tot else np.nan,
        # cost, kept separate
        epochs=E, epochs_completed=epochs_done, optimizer_steps=steps,
        alpha_updates=alpha_updates, backprop_calls=steps, gradient_evaluations=examples,
        elapsed_s=time.perf_counter() - start, diverged=diverged)
    report = pa if is_alpha else ov
    return final, (dict(cols, sigma_history=sigma_history) if record else None), (
        None if report is None else report["curve"])


def tuning_score(final):
    """Learning-rate selection criterion (validation only)."""
    v = final["val_post_alpha"] if final["loss"] == "Alpha" else final["val_selected"]
    return np.inf if final["diverged"] or not np.isfinite(v) else float(v)


# ==========================================================================
# Cache, jobs, orchestration
# ==========================================================================
def _clean(v):
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return [_clean(x) for x in v.tolist()]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return v if math.isfinite(v) else (None if math.isnan(v) else ("inf" if v > 0 else "-inf"))
    return v


def _num(v):
    if v is None:
        return np.nan
    if v == "inf":
        return np.inf
    if v == "-inf":
        return -np.inf
    return v


def save_gz(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + f".{platform.node()}.{os.getpid()}.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fp:
        json.dump(_clean(obj), fp, separators=(",", ":"))
    tmp.replace(path)                       # atomic on POSIX file systems


def load_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as fp:
        return json.load(fp)


def _code_digest(code, h):
    h.update(code.co_code)
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            _code_digest(c, h)
        else:
            h.update(repr(c).encode())
    h.update(repr(code.co_names).encode())


def source_hash():
    """Hash of this file; if unavailable (code pasted into a notebook cell),
    a hash of the compiled code of every function defined here."""
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    except (NameError, OSError):
        h = hashlib.sha256()
        for name, obj in sorted(globals().items()):
            code = getattr(obj, "__code__", None)
            if code is not None and getattr(obj, "__module__", None) == __name__:
                h.update(name.encode())
                _code_digest(code, h)
        return "code-" + h.hexdigest()[:12]


def settings(args):
    cfg = {k: v for k, v in sorted(vars(args).items()) if k not in OPERATIONAL}
    if args.stage == "sigma":
        cfg.pop("sigma_strategy", None)
        cfg.pop("fractions", None)
        cfg.pop("plot_fraction", None)
    else:
        cfg.pop("sigma_strategies", None)
    cfg["cache_format"] = CACHE_FORMAT
    cfg["source"] = source_hash()
    return cfg


def cache_dir(args):
    cfg = settings(args)
    digest = hashlib.sha256(json.dumps(_clean(cfg), sort_keys=True).encode()).hexdigest()[:14]
    d = Path(args.out) / "cache" / f"{args.stage}_{digest}"
    d.mkdir(parents=True, exist_ok=True)
    stamp = d / "settings.json"
    if not stamp.is_file():
        stamp.write_text(json.dumps(_clean(cfg), indent=2))
    return d, cfg, digest


def valid_cached(path, digest):
    """True if `path` is a readable cache file of this exact experiment version.
    Unreadable or foreign files are moved aside (never silently mixed)."""
    path = Path(path)
    if not path.is_file():
        return False
    try:
        d = load_gz(path)
        ok = d.get("digest") == digest and d.get("cache_format") == CACHE_FORMAT
    except (OSError, EOFError, ValueError, json.JSONDecodeError):
        ok = False
    if not ok:
        bad = path.with_name(path.name + ".quarantined")
        path.replace(bad)
        print(f"  WARNING: {path.name} is corrupt or from another version; moved to "
              f"{bad.name} and will be re-run.", flush=True)
    return ok


def lr_grid(args, opt):
    return list(args.lr_grid_gd if opt == "GD" else args.lr_grid_sgd)


def tune_job(args, digest, function, condition, pilot_seed, path):
    data = make_data(function, condition, pilot_seed, args, pilot=True)
    sched = schedule(args, args.tune_epochs)
    scores = {}
    for spec in methods_for(args.stage, args):
        scores[spec["name"]] = {}
        for lr in lr_grid(args, spec["opt"]):
            final, _, _ = train(spec, data, lr, sched, args, pilot_seed, pilot=True)
            scores[spec["name"]][repr(float(lr))] = dict(
                score=tuning_score(final), val_selected=final["val_selected"],
                val_post_alpha=final["val_post_alpha"], diverged=final["diverged"])
    save_gz(path, dict(digest=digest, cache_format=CACHE_FORMAT, kind="tune",
                       function=function, condition=condition, pilot_seed=pilot_seed,
                       tune_epochs=args.tune_epochs, schedule=sched, scores=scores))
    return str(path)


def eval_job(args, digest, function, condition, seed, lrs, path):
    data = make_data(function, condition, seed, args)
    sched = schedule(args, args.epochs)
    finals, paths, curves = [], {}, {}
    for spec in methods_for(args.stage, args):
        final, cols, curve = train(spec, data, lrs[spec["name"]], sched, args, seed, record=True)
        final.update(function=function, condition=condition, seed=seed)
        finals.append(final)
        paths[spec["name"]] = cols
        if seed == 0 and curve is not None:
            curves[spec["name"]] = curve
    preview = (dict(x=data["x"], y=data["y"], bad=data["bad"], curves=curves)
               if seed == 0 else None)
    save_gz(path, dict(digest=digest, cache_format=CACHE_FORMAT, kind="eval",
                       function=function, condition=condition, seed=seed, lrs=lrs,
                       schedule=sched, finals=finals, path=paths, preview=preview))
    return str(path)


def run_jobs(jobs, workers, quiet=False):
    """jobs: list of (callable, args tuple); the last arg is the output path."""
    if not jobs:
        return
    t0 = time.perf_counter()

    def report(i, p):
        if not quiet:
            print(f"  [{i}/{len(jobs)}] {Path(p).name}  ({time.perf_counter() - t0:.0f}s)",
                  flush=True)
    if workers <= 1:
        for i, (fn, a) in enumerate(jobs, 1):
            report(i, fn(*a))
        return
    # fork on Linux keeps workers usable after `%run` in Jupyter (Spartan, Python >= 3.14
    # default changes); macOS / Windows use their safe default start method.
    ctx = multiprocessing.get_context("fork") if sys.platform.startswith("linux") else None
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futures = [ex.submit(fn, *a) for fn, a in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            report(i, fut.result())


def tune_path(cdir, fn, cond, k):
    return cdir / f"tune_{fn}_{cond}_p{k}.json.gz"


def eval_path(cdir, fn, cond, s):
    return cdir / f"eval_{fn}_{cond}_s{s}.json.gz"


def choose_lrs(args, cdir, function, condition):
    """Per method: the learning rate minimising mean log(tuning score) over the
    pilot seeds.  Returns (lrs, rows), or (None, rows) if tuning is incomplete."""
    files = [tune_path(cdir, function, condition, k) for k in range(args.tune_seeds)]
    if not all(f.is_file() for f in files):
        return None, []
    loaded = [load_gz(f) for f in files]
    lrs, rows = {}, []
    for spec in methods_for(args.stage, args):
        grid = lr_grid(args, spec["opt"])
        crit, block = {}, []
        for lr in grid:
            vals = [float(_num(d["scores"][spec["name"]][repr(float(lr))]["score"])) for d in loaded]
            crit[lr] = float(np.mean(np.log(vals))) if all(np.isfinite(vals)) else np.inf
            for k, v in enumerate(vals):
                block.append(dict(function=function, condition=condition, method=spec["name"],
                                  lr=lr, pilot_seed=k, tuning_score=v, criterion=crit[lr]))
        best = min(grid, key=lambda lr: (crit[lr], lr))
        lrs[spec["name"]] = best
        for r in block:
            r.update(chosen=r["lr"] == best, chosen_at_grid_edge=best in (min(grid), max(grid)),
                     all_diverged=not np.isfinite(crit[best]))
        rows += block
    return lrs, rows


def open_cache(args, say):
    """--analyze-cache: analyse ONE explicitly named cache folder with the settings
    it was produced with (no training).  Results are never mixed across folders."""
    cdir = Path(args.analyze_cache)
    st = json.loads((cdir / "settings.json").read_text())
    if st.get("cache_format") != CACHE_FORMAT:
        raise RuntimeError(f"{cdir} uses cache format {st.get('cache_format')}; this program "
                           f"reads format {CACHE_FORMAT}")
    new = argparse.Namespace(**vars(args))
    for k, v in st.items():
        if hasattr(new, k) and k not in OPERATIONAL:
            setattr(new, k, v)
    if st.get("source") != source_hash():
        say(f"NOTE: {cdir.name} was produced by code version {st.get('source')}; its results "
            "are analysed with the current analysis and plotting code.")
    return new, cdir, dict(st), cdir.name.split("_", 1)[1]


def experiment(args, quiet=False):
    say = (lambda *a, **k: None) if quiet else print
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.analyze_cache:
        args, cdir, cfg, digest = open_cache(args, say)
        training = False
    else:
        cdir, cfg, digest = cache_dir(args)
        training = not args.plots_only
    others = [d.name for d in (out / "cache").glob(f"{args.stage}_*") if d != cdir]
    say(f"Stage '{args.stage}': {len(methods_for(args.stage, args))} methods; cache {cdir}")
    if others:
        say(f"  (ignoring {len(others)} cache folder(s) from other settings or code versions)")
    budget = args.max_jobs if args.max_jobs > 0 else 10 ** 9

    if training:
        jobs = [(tune_job, (args, digest, fn, cond, k, tune_path(cdir, fn, cond, k)))
                for fn in FUNCTIONS for cond in CONDITIONS for k in range(args.tune_seeds)
                if not valid_cached(tune_path(cdir, fn, cond, k), digest)][:budget]
        if jobs:
            say(f"Tuning learning rates on pilot data: {len(jobs)} job(s)")
        run_jobs(jobs, args.workers, quiet)
        budget -= len(jobs)

    lrs, tune_rows = {}, []
    for fn in FUNCTIONS:
        for cond in CONDITIONS:
            chosen, rows = choose_lrs(args, cdir, fn, cond)
            tune_rows += rows
            if chosen is not None:
                lrs[(fn, cond)] = chosen

    if training and budget > 0:
        jobs = [(eval_job, (args, digest, fn, cond, s, lrs[(fn, cond)], eval_path(cdir, fn, cond, s)))
                for s in range(args.seeds) for fn in FUNCTIONS for cond in CONDITIONS
                if (fn, cond) in lrs and not valid_cached(eval_path(cdir, fn, cond, s), digest)
                ][:budget]
        if jobs:
            say(f"Evaluation runs: {len(jobs)} job(s)")
        run_jobs(jobs, args.workers, quiet)

    expected = len(FUNCTIONS) * len(CONDITIONS) * args.seeds
    done = [eval_path(cdir, fn, cond, s) for s in range(args.seeds) for fn in FUNCTIONS
            for cond in CONDITIONS if eval_path(cdir, fn, cond, s).is_file()]
    if tune_rows:
        pd.DataFrame(tune_rows).to_csv(out / "tuning.csv", index=False)
    cfg.update(digest=digest, cache=str(cdir), completed_eval_jobs=len(done),
               expected_eval_jobs=expected, seeds=args.seeds,
               methods=[s["name"] for s in methods_for(args.stage, args)],
               families=families(args.stage, args),
               validation_criterion=validation_criterion(args),
               main_schedule=schedule(args, args.epochs),
               tuning_schedule=schedule(args, args.tune_epochs),
               tuning_criterion="mean over pilot seeds of log(best validation score); "
                                "post-alpha checkpoint for alpha methods, overall for MSE")
    (out / "config.json").write_text(json.dumps(_clean(cfg), indent=2))
    if not done:
        say("No finished evaluation runs yet; rerun the same command to continue.")
        return None
    if len(done) < expected:
        say(f"PARTIAL RESULTS: {len(done)}/{expected} evaluation jobs finished. "
            "Rerun the same command to continue.")
    return analyze(args, cdir, digest, out, quiet=quiet)


# ==========================================================================
# Statistics
# ==========================================================================
def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adj = np.maximum.accumulate(np.clip((m - np.arange(m)) * p[order], 0, 1))
    out = np.empty(m)
    out[order] = adj
    return out


def families(stage, args):
    """Pre-declared comparison families (candidate, baseline).  Holm is applied
    within each family across all function x condition cases, per metric."""
    F = {}
    if stage == "sigma":
        S = list(args.sigma_strategies)
        F["alpha_vs_mse"] = [(f"{o} Alpha[{s}]", f"{o} MSE") for o in OPTIMIZERS for s in S]
        F["sigma_strategy"] = [(f"{o} Alpha[{a}]", f"{o} Alpha[{b}]") for o in OPTIMIZERS
                               for a, b in (("late", "early"), ("periodic", "early"),
                                            ("periodic", "late")) if a in S and b in S]
        F["sgd_vs_gd"] = [(f"SGD {m}", f"GD {m}") for m in ["MSE"] + [f"Alpha[{s}]" for s in S]]
    else:
        P = [round(100 * f) for f in args.fractions if f < 1]
        for o in OPTIMIZERS:
            F[f"PRIMARY ranked_vs_uniform {o}"] = [(f"{o} Ranked {p}%", f"{o} Uniform {p}%")
                                                   for p in P]
        F["ranked_vs_alpha"] = [(f"{o} Ranked {p}%", f"{o} Alpha") for o in OPTIMIZERS for p in P]
        F["alpha_vs_mse"] = [(f"{o} Alpha", f"{o} MSE") for o in OPTIMIZERS]
        F["sgd_vs_gd"] = [("SGD MSE", "GD MSE"), ("SGD Alpha", "GD Alpha")]
    return F


def effect_sizes(a, b, rng):
    """a = candidate, b = baseline (positive MSEs, paired).  Returns median ratio,
    bootstrap 95 % CI of the median ratio, and matched-pairs rank-biserial r
    (r > 0: candidate has LOWER error)."""
    lr = np.log(a / b)
    boot = np.median(rng.choice(lr, size=(2000, len(lr)), replace=True), axis=1)
    d = a - b
    nz = d[d != 0]
    if len(nz):
        rk = rankdata(np.abs(nz))
        r_rb = float((rk[nz < 0].sum() - rk[nz > 0].sum()) / rk.sum())
    else:
        r_rb = 0.0
    return (float(np.exp(np.median(lr))), float(np.exp(np.percentile(boot, 2.5))),
            float(np.exp(np.percentile(boot, 97.5))), r_rb)


def paired_tests(final, fams):
    rng = np.random.default_rng(BASE_SEED)
    rows = []
    for fam, pairs in fams.items():
        for metric in METRICS:
            block = []
            for fn in FUNCTIONS:
                for cond in CONDITIONS:
                    sub = final[(final.function == fn) & (final.condition == cond)]
                    if sub.empty:
                        continue
                    tab = sub.pivot_table(index="seed", columns="method", values=metric,
                                          aggfunc="first")
                    for a, b in pairs:
                        if a not in tab or b not in tab:
                            continue
                        both = tab[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
                        both = both[(both[a] > 0) & (both[b] > 0)]
                        if len(both) < 2:
                            continue
                        va, vb = both[a].to_numpy(float), both[b].to_numpy(float)
                        d = va - vb
                        nz = int(np.sum(d != 0))
                        if nz == 0:
                            p = 1.0
                        else:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                p = float(wilcoxon(d).pvalue)
                        ratio, lo, hi, rrb = effect_sizes(va, vb, rng)
                        block.append(dict(
                            family=fam, metric=metric, function=fn, condition=cond,
                            candidate=a, baseline=b, n_pairs=len(d), n_nonzero=nz,
                            median_candidate=float(np.median(va)),
                            median_baseline=float(np.median(vb)),
                            median_difference=float(np.median(d)),
                            median_ratio=ratio, ratio_ci95_low=lo, ratio_ci95_high=hi,
                            rank_biserial=rrb, wins=int(np.sum(d < 0)), losses=int(np.sum(d > 0)),
                            p_raw=p, min_possible_p=(min(1.0, 2.0 / 2 ** nz) if nz else 1.0)))
            if block:
                for r, adj in zip(block, holm([r["p_raw"] for r in block])):
                    r["family_size"] = len(block)
                    r["p_holm"] = float(adj)
                    r["significant_holm_0.05"] = bool(adj < 0.05)
                rows += block
    return pd.DataFrame(rows)


# ==========================================================================
# Loading, analysis
# ==========================================================================
def load_results(cdir, digest, seeds):
    finals, paths, previews, sig = [], [], {}, []
    for f in sorted(cdir.glob("eval_*.json.gz")):
        try:
            d = load_gz(f)
        except (OSError, EOFError, ValueError) as exc:
            print(f"  WARNING: skipping unreadable {f.name} ({exc.__class__.__name__}); "
                  "a training invocation will quarantine and re-run it.")
            continue
        if d.get("digest") != digest or d.get("cache_format") != CACHE_FORMAT:
            raise RuntimeError(f"{f} belongs to a different experiment version; refusing to mix.")
        if d["seed"] >= seeds:
            continue
        finals += d["finals"]
        for m, cols in d["path"].items():
            hist = cols.pop("sigma_history", [])
            df = pd.DataFrame({k: [_num(v) for v in vals] for k, vals in cols.items()})
            df["method"], df["function"], df["condition"], df["seed"] = (
                m, d["function"], d["condition"], d["seed"])
            paths.append(df)
            sig += [dict(function=d["function"], condition=d["condition"], seed=d["seed"],
                         method=m, epoch=e, sigma=s) for e, s in hist]
        if d.get("preview"):
            previews[(d["function"], d["condition"])] = d["preview"]
    final = pd.DataFrame(finals)
    text = {"method", "opt", "loss", "sigma_strategy", "sparse", "function", "condition",
            "validation_criterion"}
    for c in final.columns:
        if c not in text and final[c].dtype == object:
            final[c] = pd.to_numeric(final[c].map(_num), errors="coerce")
    return (final, pd.concat(paths, ignore_index=True) if paths else pd.DataFrame(),
            previews, pd.DataFrame(sig))


def _summary(final):
    q25, q75 = (lambda s: s.quantile(.25)), (lambda s: s.quantile(.75))
    agg = dict(seeds=("seed", "nunique"), lr=("lr", "first"))
    for m in METRICS + ("test_post_alpha",):
        agg[f"median_{m}"] = (m, "median")
        agg[f"q25_{m}"] = (m, q25)
        agg[f"q75_{m}"] = (m, q75)
    agg.update(
        median_train_msr=("train_msr_final", "median"),
        median_best_epoch=("best_epoch", "median"),
        median_best_post_alpha_epoch=("best_post_alpha_epoch", "median"),
        share_selected_in_warmup=("selected_in_warmup", "mean"),
        median_sigma_first=("sigma_first", "median"),
        median_sigma_final=("sigma_final", "median"),
        median_sigma_updates=("sigma_updates", "median"),
        median_q_outlier=("q_outlier_final", "median"),
        median_q_inlier=("q_inlier_final", "median"),
        selected_fraction_actual=("selected_fraction_actual", "mean"),
        mean_selected_rank=("mean_selected_rank", "mean"),
        optimizer_steps=("optimizer_steps", "median"), alpha_updates=("alpha_updates", "median"),
        gradient_evaluations=("gradient_evaluations", "median"),
        median_runtime_s=("elapsed_s", "median"), diverged=("diverged", "sum"))
    s = final.groupby(["function", "condition", "method"], sort=False, observed=True).agg(**agg)
    return s.reset_index()


def analyze(args, cdir, digest, out, quiet=False):
    final, paths, previews, sig = load_results(cdir, digest, args.seeds)
    order = {m: i for i, m in enumerate(s["name"] for s in methods_for(args.stage, args))}
    final["_f"] = final.function.map(FUNCTIONS.index)
    final["_c"] = final.condition.map(CONDITIONS.index)
    final["_m"] = final.method.map(order)
    final = final.sort_values(["_f", "_c", "_m", "seed"]).drop(columns=["_f", "_c", "_m"])
    final = final.reset_index(drop=True)
    final.to_csv(out / "individual_results.csv", index=False)
    if not paths.empty:
        paths.to_csv(out / "convergence.csv.gz", index=False)
    if not sig.empty:
        sig.to_csv(out / "sigma_history.csv", index=False)
    summary = _summary(final)
    summary.to_csv(out / "summary.csv", index=False)
    stats = paired_tests(final, families(args.stage, args))
    stats.to_csv(out / "paired_statistics.csv", index=False)
    figs = plots(args, final, paths, previews, summary, stats, out)
    if not quiet:
        print_report(args, summary, stats)
        print("\nFigures:", ", ".join(figs))
        print("Outputs saved to", out.resolve())
    return summary


def print_report(args, summary, stats):
    shown = [m for m in shown_methods(args) if m in set(summary.method)]
    pd.set_option("display.width", 250)
    idx = pd.MultiIndex.from_product([FUNCTIONS, CONDITIONS])

    def table(col, fmt):
        t = summary.pivot_table(index=["function", "condition"], columns="method", values=col,
                                sort=False)
        cols = [m for m in shown if m in t.columns]
        t = t.reindex(idx).dropna(how="all")
        return t[cols].to_string(float_format=fmt) if cols and len(t) else "  (none)"
    print("\nMedian test MSE, alpha-phase checkpoint (post-alpha for alpha methods, "
          "overall for MSE), original units:")
    print(table("median_test_alpha_phase", lambda v: f"{v:.5f}"))
    print("\nShare of runs whose OVERALL best checkpoint lies in the MSE warmup:")
    print(table("share_selected_in_warmup", lambda v: f"{v:.2f}"))
    print("\nMedian sigma used by the LAST optimiser update (alpha methods):")
    print(table("median_sigma_final", lambda v: f"{v:.3f}"))
    print("\nMedian mean-q on CORRUPTED points at the final parameters, under that sigma:")
    print(table("median_q_outlier", lambda v: f"{v:.3f}"))
    if len(stats):
        s = stats[stats.metric == "test_alpha_phase"]
        print("\nPaired Wilcoxon, Holm within family, alpha-phase checkpoint "
              "(full table incl. other metrics: paired_statistics.csv).")
        print("Effect: median ratio candidate/baseline [bootstrap 95% CI]; r = rank-biserial "
              "(>0 favours candidate). Non-significance is NOT equivalence.")
        for fam, block in s.groupby("family", sort=False):
            sig = block[block["significant_holm_0.05"]]
            better = int((sig.median_ratio < 1).sum())
            print(f"  {fam}: {len(block)} tests, {len(sig)} significant "
                  f"({better} favour the candidate, {len(sig) - better} the baseline)")
            show = block if fam.startswith("PRIMARY") else sig
            for _, r in show.head(30).iterrows():
                print(f"      {r.function:8s} {r.condition:10s} {r.candidate} vs {r.baseline}: "
                      f"ratio {r.median_ratio:.3g} [{r.ratio_ci95_low:.3g}, {r.ratio_ci95_high:.3g}]"
                      f", r={r.rank_biserial:+.2f}, wins {r.wins}/{r.n_pairs}, "
                      f"p_holm {r.p_holm:.2g}{' *' if r['significant_holm_0.05'] else ''}")
        m = int(stats.family_size.max())
        print(f"  Smallest attainable raw p with these seeds: {stats.min_possible_p.min():.3g}; "
              f"the largest family ({m} tests) needs >= {math.ceil(math.log2(2 * m / 0.05))} "
              f"non-zero pairs, all in one direction, to reach p_holm < 0.05.")


# ==========================================================================
# Plots (all figures: PNG files; displayed inline in Jupyter by show_plots)
# ==========================================================================
def shown_methods(args):
    if args.stage == "sigma":
        return [s["name"] for s in methods_for("sigma", args)]
    fr = [f for f in args.fractions if f < 1]
    p = round(100 * (args.plot_fraction if args.plot_fraction in fr else (fr[0] if fr else 1)))
    names = ["GD MSE", "SGD MSE", "GD Alpha", "SGD Alpha"]
    if fr:
        names += [f"{o} {k} {p}%" for o in OPTIMIZERS for k in ("Ranked", "Uniform")]
    return names


def colors_for(names):
    """tab10 without its grey, so no method is confused with the black truth."""
    cmap = plt.get_cmap("tab10")
    palette = [cmap(i) for i in (0, 1, 2, 3, 4, 5, 6, 8, 9)]
    return {m: palette[i % len(palette)] for i, m in enumerate(names)}


def style(m):
    return "--" if "Uniform" in m else (":" if "MSE" in m else "-")


def _positive(v):
    v = np.asarray(v, dtype=float)
    return v[np.isfinite(v) & (v > 0)]


def set_log_if_valid(ax):
    """Log y-axis only when the panel holds at least one finite positive value."""
    vals = []
    for line in ax.get_lines():
        vals.append(_positive(line.get_ydata()))
    for coll in ax.collections:
        off = coll.get_offsets()
        if len(off):
            vals.append(_positive(np.asarray(off)[:, 1]))
    if vals and any(len(v) for v in vals):
        ax.set_yscale("log", nonpositive="mask")
        return True
    return False


def mark_empty(ax, text="no finished runs"):
    if not ax.get_lines() and not ax.collections and not ax.containers:
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes,
                color="gray", fontsize=10)
        ax.set_yticks([])
        return True
    return False


def _grid_figure(title):
    fig, axs = plt.subplots(len(FUNCTIONS), len(CONDITIONS), figsize=(20, 12), sharex=True,
                            squeeze=False)
    fig.suptitle(title, fontsize=14)
    for i, fn in enumerate(FUNCTIONS):
        for j, cond in enumerate(CONDITIONS):
            axs[i, j].set_title(f"{fn} | {cond.replace('_', ' ')}", fontsize=10)
            axs[i, j].grid(alpha=.15)
    return fig, axs


def _finish(fig, axs, out, name, xlabel, dpi):
    for ax in np.ravel(axs)[-len(CONDITIONS):] if np.ndim(axs) == 2 else np.ravel(axs):
        ax.set_xlabel(xlabel)
    handles, labels = [], []
    for ax in np.ravel(axs):
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    if labels:
        fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)), fontsize=9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(out / name, dpi=dpi)
    plt.close(fig)
    return name


def _path_plot(paths, methods, colors, column, out, name, title, ylabel, dpi, log=True,
               conditions_with_data=CONDITIONS):
    fig, axs = _grid_figure(title)
    for i, fn in enumerate(FUNCTIONS):
        for j, cond in enumerate(CONDITIONS):
            ax = axs[i, j]
            if not paths.empty and column in paths and cond in conditions_with_data:
                sub = paths[(paths.function == fn) & (paths.condition == cond)]
                for m in methods:
                    med = sub[sub.method == m].groupby("epoch")[column].median()
                    med = med[np.isfinite(med.values)]
                    if log:
                        med = med[med.values > 0]
                    if len(med):
                        ax.plot(med.index, med.values, color=colors[m], ls=style(m), lw=1.5,
                                label=m)
            if not mark_empty(ax, "no data" if cond in conditions_with_data else
                              "no contaminated points") and log:
                set_log_if_valid(ax)
            if j == 0:
                ax.set_ylabel(ylabel)
    return _finish(fig, axs, out, name,
                   "Epoch (equal data passes; SGD makes n/q optimiser steps per epoch)", dpi)


def _effects_plot(stats, fams, metric, out, name, title, dpi):
    """Median ratio candidate/baseline with bootstrap 95 % CI, per case."""
    s = stats[(stats.metric == metric) & (stats.family.isin(fams))] if len(stats) else stats
    fig, axs = _grid_figure(title)
    pairs = [] if s is None or not len(s) else list(dict.fromkeys(zip(s.candidate, s.baseline)))
    cmap = plt.get_cmap("tab10")
    for i, fn in enumerate(FUNCTIONS):
        for j, cond in enumerate(CONDITIONS):
            ax = axs[i, j]
            sub = s[(s.function == fn) & (s.condition == cond)] if len(pairs) else s
            for k, (a, b) in enumerate(pairs):
                r = sub[(sub.candidate == a) & (sub.baseline == b)]
                if not len(r):
                    continue
                r = r.iloc[0]
                yerr = [[r.median_ratio - r.ratio_ci95_low], [r.ratio_ci95_high - r.median_ratio]]
                ax.errorbar([k], [r.median_ratio], yerr=yerr, fmt="o",
                            mfc=cmap(k % 10) if r["significant_holm_0.05"] else "white",
                            color=cmap(k % 10), capsize=3, label=f"{a} / {b}")
            if not mark_empty(ax):
                ax.axhline(1.0, color="k", lw=.8)
                set_log_if_valid(ax)
                ax.set_xticks([])
            if j == 0:
                ax.set_ylabel("median ratio (<1 favours candidate)")
    return _finish(fig, axs, out, name, "filled marker = Holm-significant at 0.05", dpi)


def plots(args, final, paths, previews, summary, stats, out):
    dpi = args.dpi
    methods = [m for m in shown_methods(args) if m in set(final.method)]
    colors = colors_for(methods)
    alpha_methods = [m for m in methods if "MSE" not in m]
    made = []
    plt.close("all")

    # 1. fits of seed 0 at the reporting checkpoint
    fig, axs = _grid_figure(f"Seed 0 fits at the reporting checkpoint (post-alpha best for alpha "
                            f"methods, overall best for MSE) | alpha={args.alpha} | "
                            f"stage={args.stage}")
    for i, fn in enumerate(FUNCTIONS):
        for j, cond in enumerate(CONDITIONS):
            ax, pv = axs[i, j], previews.get((fn, cond))
            if pv is not None:
                x, y, bad = np.array(pv["x"]), np.array(pv["y"]), np.array(pv["bad"], bool)
                ax.plot(GRID, target(fn, GRID), "k", lw=2.6, label="True function", zorder=1)
                ax.scatter(x[~bad], y_original(y[~bad]), facecolors="none", edgecolors="gray",
                           s=14, lw=.7, label="Inlier")
                if bad.any():
                    ax.scatter(x[bad], y_original(y[bad]), c="#b3262b", marker="x", s=18,
                               label="Outlier")
                for m in methods:
                    c = pv["curves"].get(m)
                    if c is not None:
                        ax.plot(GRID, np.array([_num(v) for v in c], float), color=colors[m],
                                ls=style(m), lw=1.4, label=m)
                ax.set_ylim(-0.6, 2.2)
            mark_empty(ax, "seed 0 not finished")
            if j == 0:
                ax.set_ylabel("y (original units)")
    made.append(_finish(fig, axs, out, "fitted_curves.png", "x", dpi))

    # 2-6. trajectories
    n_seeds = final.seed.nunique() if len(final) else 0
    made.append(_path_plot(paths, methods, colors, "test_mse", out, "convergence.png",
                           f"Median clean-function test MSE ({n_seeds} paired seeds; "
                           f"reporting only, never used for selection)", "Median test MSE", dpi))
    made.append(_path_plot(paths, methods, colors, "val_score", out, "validation.png",
                           f"Median validation score ({validation_criterion(args)})",
                           "Median validation score", dpi))
    made.append(_path_plot(paths, methods, colors, "grad_noise", out, "gradient_noise.png",
                           "Mini-batch gradient noise E||g_B - g||^2 of the loss being optimised "
                           "(fixed partition)", "Median gradient noise", dpi))
    made.append(_path_plot(paths, alpha_methods, colors, "sigma", out, "sigma_trajectory.png",
                           "Residual scale sigma in force (median over seeds; periodic changes "
                           "the objective)", "sigma", dpi))
    made.append(_path_plot(paths, alpha_methods, colors, "q_outlier", out, "outlier_weight.png",
                           "Mean alpha-weight q on CORRUPTED points (1 = MSE-like, 0 = ignored)",
                           "Median q on outliers", dpi, log=False,
                           conditions_with_data=CONDITIONS[1:]))
    made.append(_path_plot(paths, alpha_methods, colors, "q_inlier", out, "inlier_weight.png",
                           "Mean alpha-weight q on CLEAN points", "Median q on inliers", dpi,
                           log=False))

    # 7. summary across conditions
    fig, axs = plt.subplots(1, len(FUNCTIONS), figsize=(18, 5.5), squeeze=False)
    for ax, fn in zip(axs[0], FUNCTIONS):
        sub = summary[summary.function == fn]
        for m in methods:
            vals = []
            for c in CONDITIONS:
                v = sub[(sub.condition == c) & (sub.method == m)].median_test_alpha_phase
                vals.append(v.iloc[0] if len(v) else np.nan)
            vals = np.array(vals, float)
            if np.isfinite(vals).any():
                ax.plot(range(len(CONDITIONS)), np.where(vals > 0, vals, np.nan), marker="o",
                        ms=4, color=colors[m], ls=style(m), label=m)
        ax.set_xticks(range(len(CONDITIONS)), ["Clean", "Asym.", "Point-mass", "Sym."])
        if not mark_empty(ax):
            set_log_if_valid(ax)
        ax.set_title(fn)
        ax.set_ylabel("Median test MSE (alpha-phase checkpoint)")
        ax.grid(alpha=.2)
    made.append(_finish(fig, axs, out, "summary_mse.png", "Condition", dpi))

    # 8. effect sizes of the key family
    if args.stage == "sigma":
        fams, title = ["alpha_vs_mse"], "Effect size: alpha / MSE (same optimiser)"
    else:
        fams = [f for f in families(args.stage, args) if f.startswith("PRIMARY")]
        title = "PRIMARY effect size: Ranked / Uniform at identical fractions"
    made.append(_effects_plot(stats, fams, "test_alpha_phase", out, "effect_sizes.png",
                              title + " | alpha-phase checkpoint | bootstrap 95% CI", dpi))

    if args.stage == "ranking":
        # 9. fraction sweep
        fr = sorted({f for f in args.fractions if f < 1} | {1.0})
        fig, axs = _grid_figure(f"Fraction of W updated per step | sigma strategy "
                                f"'{args.sigma_strategy}' | 100% = alpha (all weights)")
        pal = {("GD", "Ranked"): "#b42c65", ("GD", "Uniform"): "#a78a23",
               ("SGD", "Ranked"): "#7040a7", ("SGD", "Uniform"): "#548a22"}
        for i, fn in enumerate(FUNCTIONS):
            for j, cond in enumerate(CONDITIONS):
                ax = axs[i, j]
                sub = summary[(summary.function == fn) & (summary.condition == cond)]

                def med(name):
                    v = sub[sub.method == name].median_test_alpha_phase
                    return float(v.iloc[0]) if len(v) else np.nan
                for (o, k), c in pal.items():
                    ys = np.array([med(f"{o} Alpha") if f == 1.0 else
                                   med(f"{o} {k} {round(100 * f)}%") for f in fr])
                    ok = np.isfinite(ys) & (ys > 0)
                    if ok.any():
                        ax.plot(np.array(fr)[ok], ys[ok], marker="o", ms=4, color=c,
                                ls="--" if k == "Uniform" else "-", label=f"{o} {k}")
                for o, c in (("GD", "#1b1b1b"), ("SGD", "#295bb5")):
                    v = med(f"{o} MSE")
                    if np.isfinite(v) and v > 0:
                        ax.axhline(v, color=c, ls=":", lw=1.2, label=f"{o} MSE")
                if not mark_empty(ax):
                    set_log_if_valid(ax)
                if j == 0:
                    ax.set_ylabel("Median test MSE (alpha-phase)")
        made.append(_finish(fig, axs, out, "fraction_sweep.png",
                            "Fraction of each layer's W updated", dpi))
        # 10. ranking diagnostic
        diag = final[final["sparse"].notna()].groupby("method", sort=False).mean_selected_rank.mean()
        diag = diag.dropna()
        fig, ax = plt.subplots(figsize=(max(7, .6 * max(1, len(diag))), 4.5))
        if len(diag):
            ax.bar(range(len(diag)), diag.values,
                   color=["#b42c65" if "Ranked" in m else "#a78a23" for m in diag.index])
            ax.set_xticks(range(len(diag)), diag.index, rotation=60, ha="right", fontsize=8)
            ax.axhline(.5, color="gray", ls="--", lw=1, label="Uniform expectation 0.5")
            ax.legend()
        mark_empty(ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean within-layer rank of updated W (0 = smallest)")
        ax.set_title("Ranking diagnostic")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.tight_layout()
        fig.savefig(out / "ranking_diagnostics.png", dpi=dpi)
        plt.close(fig)
        made.append("ranking_diagnostics.png")
    return made


FIGURES = [("Fitted curves (seed 0)", "fitted_curves.png"), ("Test-MSE convergence", "convergence.png"),
           ("Validation score", "validation.png"), ("Median test MSE", "summary_mse.png"),
           ("Effect sizes", "effect_sizes.png"), ("Residual scale sigma", "sigma_trajectory.png"),
           ("Alpha-weight on corrupted points", "outlier_weight.png"),
           ("Alpha-weight on clean points", "inlier_weight.png"),
           ("Gradient noise", "gradient_noise.png"), ("Fraction sweep", "fraction_sweep.png"),
           ("Ranking diagnostics", "ranking_diagnostics.png")]


def show_plots(directory):
    """Display every saved figure inline in Jupyter, or print the file paths."""
    out = Path(directory)
    try:
        from IPython import get_ipython
        from IPython.display import Image, Markdown, display
        shell = get_ipython()
        notebook = shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        notebook = False
    for title, filename in FIGURES:
        p = out / filename
        if not p.is_file():
            continue
        if notebook:
            display(Markdown("### " + title))
            display(Image(filename=str(p)))
        else:
            print(f"{title}: {p.resolve()}")


# ==========================================================================
# Verification tests
# ==========================================================================
def _tiny_args(**kw):
    a = dict(n=100, n_val=50, noise=.025, contamination=.2, val_mode="clean", epochs=40,
             early_warmup=2, warmup=8, sigma_every=4, eval_every=5, batch=20, alpha=.9,
             sigma_floor=.1, exploration=.2, rank_strength=3., rank_mode="probabilistic",
             sigma_strategies=list(SIGMA_STRATEGIES), sigma_strategy="late",
             fractions=[.1, .3, .5, .7, 1.0], plot_fraction=.3)
    a.update(kw)
    return argparse.Namespace(**a)


def _spec(opt="GD", loss="Alpha", sigma="late", sparse=None, fraction=1.0):
    return dict(name="t", opt=opt, loss=loss, sigma=sigma, sparse=sparse, fraction=fraction)


def unit_tests():
    """Fast mathematical / bookkeeping tests (run before every experiment)."""
    passed = []
    # 1. finite-difference gradients
    xx = np.array([[-2.0], [0.0], [1.0], [2.0]])
    yy = np.array([0.1, 0.9, 0.4, 8.0])
    for alpha in (1.0, 0.9, 0.5):
        params = init_params(4)
        _, grads = loss_and_grad(params, xx, yy, alpha, sigma=0.8)
        for j, r, c in ((0, 0, 0), (1, 1, 2), (2, 0, 4), (3, 1, 0)):
            W = params[j][0]
            old, eps = W[r, c], 1e-5
            W[r, c] = old + eps
            plus = loss_and_grad(params, xx, yy, alpha, 0.8)[0]
            W[r, c] = old - eps
            minus = loss_and_grad(params, xx, yy, alpha, 0.8)[0]
            W[r, c] = old
            assert np.isclose((plus - minus) / (2 * eps), grads[j][0][r, c],
                              rtol=2e-4, atol=2e-7), (alpha, j, r, c)
        b = params[2][1]
        old = b[3]
        b[3] = old + 1e-5
        plus = loss_and_grad(params, xx, yy, alpha, 0.8)[0]
        b[3] = old - 1e-5
        minus = loss_and_grad(params, xx, yy, alpha, 0.8)[0]
        b[3] = old
        assert np.isclose((plus - minus) / 2e-5, grads[2][1][3], rtol=2e-4, atol=2e-7)
    passed.append("1 finite-difference gradients (weights and biases; alpha = 1, .9, .5)")
    # 2. alpha = 1 equals half-MSE; alpha -> 1 limit is continuous
    params = init_params(1)
    o1, g1 = loss_and_grad(params, xx, yy, 1.0, 0.3)
    out = predict(params, xx)
    assert np.isclose(o1, 0.5 * np.mean((out - yy) ** 2))
    o2, g2 = loss_and_grad(params, xx, yy, 1 - 1e-9, 0.3)
    assert np.isclose(o1, o2, rtol=1e-6) and np.allclose(flat(g1), flat(g2), rtol=1e-6)
    passed.append("2 alpha = 1 is half-MSE and the alpha -> 1 limit is continuous")

    # shared tiny data
    args = _tiny_args()
    data = make_data("Gaussian", "asymmetric", 0, args)
    sched = schedule(args, args.epochs)
    E, W = sched["epochs"], sched["warmup"]

    # 3. warmup -> alpha transition
    mse = train(_spec(loss="MSE", sigma=None), data, 1.0, sched, args, 0, record=True)
    alp = train(_spec(sigma="late"), data, 1.0, sched, args, 0, record=True)
    def path_df(cols):
        return pd.DataFrame({k: v for k, v in cols.items() if k != "sigma_history"})
    pm, pa = path_df(mse[1]), path_df(alp[1])
    before = pm.epoch <= W["late"]
    assert np.array_equal(pm.test_mse[before], pa.test_mse[before]), "warmup must be pure MSE"
    assert not np.allclose(pm.test_mse[~before], pa.test_mse[~before]), "alpha must act after warmup"
    assert alp[0]["alpha_updates"] == E - W["late"] and alp[0]["optimizer_steps"] == E
    assert (pa.alpha_updates[pa.epoch <= W["late"]] == 0).all()
    sgd = train(_spec(opt="SGD"), data, .2, sched, args, 0)[0]
    assert sgd["alpha_updates"] == (E - W["late"]) * 5 and sgd["optimizer_steps"] == E * 5
    assert sgd["gradient_evaluations"] == E * args.n
    passed.append("3 MSE warmup is identical to MSE; alpha updates start exactly after warmup")

    # 4 + 5. sigma strategies and final-sigma bookkeeping
    for strat in SIGMA_STRATEGIES:
        f, cols, _ = train(_spec(sigma=strat), data, 1.0, sched, args, 0, record=True)
        hist = cols["sigma_history"]
        w = W[strat]
        expected = ([e for e in range(w, E) if (e - w) % sched["sigma_every"] == 0]
                    if strat == "periodic" else [w])
        assert [e for e, _ in hist] == expected, (strat, hist)
        assert all(e < E for e, _ in hist), "sigma must not be re-estimated after the last update"
        assert f["sigma_final"] == hist[-1][1] and f["sigma_updates"] == len(expected)
        # independent check: the last checkpoint (epoch E, final parameters) reports
        # sigma and q under the sigma in force, which must equal sigma_final / q_*_final
        assert cols["epoch"][-1] == E and cols["sigma"][-1] == f["sigma_final"]
        assert np.isclose(cols["q_outlier"][-1], f["q_outlier_final"])
        assert np.isclose(cols["q_inlier"][-1], f["q_inlier_final"])
        assert 0 < f["q_outlier_final"] <= 1
        if strat != "periodic":
            s_path = np.array([v for v in cols["sigma"] if np.isfinite(v)])
            assert np.allclose(s_path, s_path[0]), "fixed strategies keep sigma constant"
    # E - warm divisible by sigma_every: the old code re-estimated at epoch E
    assert (E - W["periodic"]) % sched["sigma_every"] == 0
    passed.append("4 early / late / periodic sigma estimated exactly at the scheduled epochs")
    passed.append("5 sigma_final is the sigma of the last update (no re-estimate at epoch E)")

    # 6. no validation/test leakage: scrambling the test target changes nothing in selection
    d2 = dict(data, true=np.random.default_rng(1).uniform(-5, 5, GRID.size))
    for spec in (_spec(), _spec(opt="SGD", sigma="periodic", sparse="ranked", fraction=.3)):
        a1 = train(spec, data, 1.0 if spec["opt"] == "GD" else .2, sched, args, 0)[0]
        a2 = train(spec, d2, 1.0 if spec["opt"] == "GD" else .2, sched, args, 0)[0]
        for k in ("best_epoch", "best_post_alpha_epoch", "val_selected", "val_post_alpha",
                  "sigma_final"):
            assert a1[k] == a2[k], k
        assert a1["test_selected"] != a2["test_selected"]
    passed.append("6 checkpoint selection is invariant to the test target (no leakage)")

    # 7. checkpoint classification
    t = CheckpointTracker()
    for e, s, au in ((0, 5., 0), (4, 1., 0), (8, 3., 0), (12, 2., 4), (16, 2.5, 8)):
        t.observe(e, s, 10 * s, None, au)        # epoch 8 == warmup: zero alpha updates
    assert t.overall["epoch"] == 4 and t.overall["alpha_updates"] == 0
    assert t.post_alpha["epoch"] == 12 and t.final["epoch"] == 16
    t = CheckpointTracker()
    t.observe(8, 0.1, 1., None, 0)
    assert t.post_alpha is None, "epoch == warmup is not a post-alpha checkpoint"
    f = alp[0]
    assert f["best_post_alpha_epoch"] > W["late"] and f["alpha_updates_at_post_alpha_best"] > 0
    assert f["selected_in_warmup"] == (f["alpha_updates_at_best"] == 0)
    assert f["test_alpha_phase"] == f["test_post_alpha"]
    fm = mse[0]
    assert not fm["selected_in_warmup"] and np.isnan(fm["test_post_alpha"])
    assert fm["test_alpha_phase"] == fm["test_selected"]
    passed.append("7 overall / post-alpha / final checkpoints classified correctly")

    # 8 + 9. selection masks
    rng = np.random.default_rng(42)
    Wm = np.arange(1, 101, dtype=float).reshape(10, 10)
    mask, _ = selection_mask(Wm, 0.3, "ranked", rng, rank_mode="strict")
    assert mask.sum() == 30 and np.array_equal(np.flatnonzero(mask.ravel()), np.arange(30))
    rr, ru = [], []
    for _ in range(300):
        m1, r1 = selection_mask(Wm, 0.3, "ranked", rng)
        m2, r2 = selection_mask(Wm, 0.3, "uniform", rng)
        assert m1.sum() == 30 and m2.sum() == 30
        rr.append(r1)
        ru.append(r2)
    assert np.mean(rr) < 0.4 and abs(np.mean(ru) - 0.5) < 0.05, (np.mean(rr), np.mean(ru))
    counts = np.zeros(100)
    for _ in range(3000):
        counts += selection_mask(Wm, 0.1, "ranked", rng)[0].ravel()
    assert counts.min() > 0 and counts[:10].mean() > 3 * counts[-10:].mean()
    passed.append("8 ranked selection favours small |W| (strict = exact k smallest)")
    for kind in ("ranked", "uniform"):
        for frac in (.1, .3, .5, .7):
            params = init_params(8)
            before = [(w.copy(), b.copy()) for w, b in params]
            fake = [(np.ones_like(w), np.ones_like(b)) for w, b in params]
            update(params, fake, 0.01, kind, frac, np.random.default_rng(0))
            for j in range(4):
                changed = params[j][0] != before[j][0]
                assert changed.sum() == math.ceil(frac * params[j][0].size - 1e-12)
                assert np.allclose(params[j][1], before[j][1] - 0.01)
    passed.append("9 ranked and uniform update the same k per layer; biases always updated")

    # 10. 100 % selection is bit-identical to full alpha training
    for o, lr in (("GD", 1.0), ("SGD", 0.2)):
        ref = train(_spec(opt=o, sigma="periodic"), data, lr, sched, args, 0)[0]
        for kind in ("ranked", "uniform"):
            alt = train(_spec(opt=o, sigma="periodic", sparse=kind), data, lr, sched, args, 0)[0]
            for k in ("test_final", "test_selected", "test_post_alpha", "sigma_final"):
                assert alt[k] == ref[k], (o, kind, k)
    passed.append("10 Ranked/Uniform at 100 % == alpha, bit for bit (GD and SGD)")

    # 11. every method of both stages trains, finite outputs
    for stage in ("sigma", "ranking"):
        for spec in methods_for(stage, args):
            lr = 1.0 if spec["opt"] == "GD" else 0.2
            final, cols, _ = train(spec, data, lr, sched, args, 0, record=True)
            assert np.isfinite(final["test_selected"]) and np.isfinite(final["test_final"]), spec
            assert len(cols["epoch"]) == len(set(cols["epoch"]))
            if spec["sparse"]:
                k_exp = sum(math.ceil(spec["fraction"] * WIDTHS[j] * WIDTHS[j + 1] - 1e-12)
                            for j in range(4)) / sum(WIDTHS[j] * WIDTHS[j + 1] for j in range(4))
                assert np.isclose(final["selected_fraction_actual"], k_exp), final
    passed.append(f"11 all {len(methods_for('ranking', args)) + 6} GD/SGD variants of both "
                  "stages train with finite results")

    # 14. numerical stability: divergence is detected, nothing invented
    f = train(_spec(loss="MSE", sigma=None), data, 1e6, sched, args, 0)[0]
    assert f["diverged"] and np.isnan(f["test_final"]) and np.isnan(f["train_msr_final"])
    assert tuning_score(f) == np.inf
    big = loss_and_grad(init_params(2), xx * 1e3, yy * 1e6, .9, .1)
    assert np.isfinite(big[0]) and np.isfinite(flat(big[1])).all()
    passed.append("14 divergence detected and reported as NaN/inf; extreme residuals stay finite")

    # misc: Holm, paired data, tuning schedule
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    a = _tiny_args()
    dc, da = make_data("Sine", "clean", 3, a), make_data("Sine", "asymmetric", 3, a)
    dp = make_data("Sine", "asymmetric", 3, a, pilot=True)
    assert np.array_equal(dc["x"], da["x"]) and np.array_equal(dc["yv"], da["yv"])
    assert not dc["bad"].any() and da["bad"].sum() == 20
    assert np.allclose(da["y"][da["bad"]] - dc["y"][da["bad"]], 0.8)
    assert not np.allclose(dp["x"], da["x"]) and not np.allclose(dp["yv"], da["yv"])
    big = argparse.Namespace(epochs=20000, early_warmup=40, warmup=2000, sigma_every=200,
                             eval_every=100)
    ts = schedule(big, 2000)
    assert ts["warmup"] == {"early": 4, "late": 200, "periodic": 200} and ts["sigma_every"] == 20
    passed.append("+ Holm; paired data across conditions; pilot data disjoint; proportional tuning")
    return passed


def workflow_tests():
    """Tests 12 and 13: complete and partial experiment workflows, plotting,
    interruption / resume, version mismatch, corrupt files."""
    passed = []
    tmp = Path(tempfile.mkdtemp(prefix="alpha_rank_test_"))
    common = ["--seeds", "2", "--epochs", "24", "--early-warmup", "2", "--warmup", "6",
              "--sigma-every", "4", "--eval-every", "4", "--tune-seeds", "1", "--tune-epochs", "12",
              "--lr-grid-gd", "0.5", "1.0", "--lr-grid-sgd", "0.2", "--dpi", "40"]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            # complete sigma-stage workflow
            a = parse_args(["--stage", "sigma", "--out", str(tmp / "s")] + common)
            summ = experiment(a, quiet=True)
            assert summ is not None and summ.seeds.min() == 2
            for _, f in FIGURES[:9]:
                assert (tmp / "s" / f).is_file(), f
            for f in ("config.json", "individual_results.csv", "summary.csv", "tuning.csv",
                      "paired_statistics.csv", "convergence.csv.gz", "sigma_history.csv"):
                assert (tmp / "s" / f).is_file(), f
            tun = pd.read_csv(tmp / "s" / "tuning.csv")
            assert tun.groupby(["function", "condition", "method"]).chosen.sum().eq(1).all()
            # partial ranking workflow: interrupted after a few jobs
            r = ["--stage", "ranking", "--out", str(tmp / "r"), "--fractions", "0.3", "1.0"] + common
            assert experiment(parse_args(r + ["--max-jobs", "5"]), quiet=True) is None
            cdir = cache_dir(parse_args(r))[0]
            assert len(list(cdir.glob("tune_*"))) == 5 and not list(cdir.glob("eval_*"))
            experiment(parse_args(r + ["--max-jobs", "12"]), quiet=True)   # 7 tune + 5 eval
            evals = sorted(cdir.glob("eval_*.json.gz"))
            assert len(evals) == 5
            for _, f in FIGURES:
                assert (tmp / "r" / f).is_file(), f"partial plots: {f}"
            mtimes = {p.name: p.stat().st_mtime_ns for p in cdir.glob("*.json.gz")}
            # plots-only does no training
            experiment(parse_args(r + ["--plots-only"]), quiet=True)
            assert len(list(cdir.glob("eval_*.json.gz"))) == 5
            # resume to completion, earlier files untouched
            summ = experiment(parse_args(r), quiet=True)
            assert len(list(cdir.glob("eval_*.json.gz"))) == 24 and summ.seeds.min() == 2
            for name, mt in mtimes.items():
                assert (cdir / name).stat().st_mtime_ns == mt, f"{name} was recomputed"
            # a corrupt file is quarantined and re-run
            victim = cdir / "eval_Sine_symmetric_s1.json.gz"
            victim.write_bytes(b"not gzip")
            experiment(parse_args(r), quiet=True)
            assert valid_cached(victim, cache_dir(parse_args(r))[2])
            assert (cdir / (victim.name + ".quarantined")).is_file()
            # a foreign-version file is refused at analysis time
            d = load_gz(victim)
            d["digest"] = "someone-else"
            save_gz(victim, d)
            try:
                load_results(cdir, cache_dir(parse_args(r))[2], 2)
                raise AssertionError("mixed versions were not detected")
            except RuntimeError:
                pass
            experiment(parse_args(r), quiet=True)          # re-runs the foreign file
            load_results(cdir, cache_dir(parse_args(r))[2], 2)
            # a different setting uses a different cache folder
            assert cache_dir(parse_args(r + ["--alpha", "0.8"]))[0] != cdir
            # an explicitly named cache folder can be re-analysed without training
            experiment(parse_args(["--analyze-cache", str(cdir), "--out", str(tmp / "r2"),
                                   "--seeds", "2", "--dpi", "40"]), quiet=True)
            assert (tmp / "r2" / "summary.csv").is_file()
            assert (tmp / "r2" / "fraction_sweep.png").is_file()
            assert len(list(cdir.glob("eval_*.json.gz"))) == 24
            # plotting with all-NaN / empty data never raises
            empty_stats = pd.DataFrame()
            fake = pd.read_csv(tmp / "r" / "individual_results.csv").head(0)
            plots(parse_args(r), fake, pd.DataFrame(), {}, _summary(fake) if len(fake) else
                  pd.DataFrame(columns=["function", "condition", "method",
                                        "median_test_alpha_phase"]), empty_stats, tmp / "r")
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, -1])
            assert not set_log_if_valid(ax)
            plt.close(fig)
        passed.append("12 complete and partial workflows, plots-only and all-empty data "
                      "produce every figure without exceptions")
        passed.append("13 interrupted runs resume without recomputing finished jobs; corrupt "
                      "and foreign-version files are quarantined/refused; settings change the "
                      "cache key")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return passed


def self_test(full=True):
    t0 = time.perf_counter()
    passed = unit_tests() + (workflow_tests() if full else [])
    if full:
        print("PASS self-test (%.0f s):" % (time.perf_counter() - t0))
        for p in sorted(passed, key=lambda s: int(s.split()[0]) if s.split()[0].isdigit() else 99):
            print("  " + p)
    else:
        print(f"PASS quick self-test ({len(passed)} checks, {time.perf_counter() - t0:.0f} s)")


# ==========================================================================
# CLI
# ==========================================================================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("run control")
    g.add_argument("--stage", choices=("sigma", "ranking"), default="sigma")
    g.add_argument("--out", default="alpha_rank_results")
    g.add_argument("--seeds", type=int, default=20)
    g.add_argument("--workers", type=int, default=1, help="parallel processes")
    g.add_argument("--max-jobs", type=int, default=0,
                   help="new jobs per invocation (0 = all); rerun to resume")
    g.add_argument("--plots-only", action="store_true", help="no training; analyse the cache")
    g.add_argument("--analyze-cache", default=None, metavar="CACHE_FOLDER",
                   help="analyse / re-plot one existing cache folder with its own settings "
                        "(no training), e.g. after editing plotting code")
    g.add_argument("--show-plots", action="store_true")
    g.add_argument("--self-test", action="store_true", help="run all verification tests")
    g.add_argument("--quick", action="store_true", help="tiny smoke-test configuration")
    g.add_argument("--dpi", type=int, default=150)
    g = ap.add_argument_group("data")
    g.add_argument("--n", type=int, default=100)
    g.add_argument("--n-val", type=int, default=100)
    g.add_argument("--noise", type=float, default=0.025)
    g.add_argument("--contamination", type=float, default=0.20)
    g.add_argument("--val-mode", choices=("clean", "contaminated"), default="clean")
    g = ap.add_argument_group("loss and schedule (in epochs of the main run)")
    g.add_argument("--alpha", type=float, default=0.9)
    g.add_argument("--sigma-floor", type=float, default=0.10)
    g.add_argument("--epochs", type=int, default=20000)
    g.add_argument("--early-warmup", type=int, default=40, help="warmup of sigma strategy 'early'")
    g.add_argument("--warmup", type=int, default=2000, help="warmup of 'late' and 'periodic'")
    g.add_argument("--sigma-every", type=int, default=200, help="re-estimation period, 'periodic'")
    g.add_argument("--eval-every", type=int, default=100)
    g.add_argument("--batch", type=int, default=20, help="SGD mini-batch size q")
    g = ap.add_argument_group("learning-rate tuning on pilot data")
    g.add_argument("--tune-seeds", type=int, default=2)
    g.add_argument("--tune-epochs", type=int, default=2000)
    g.add_argument("--lr-grid-gd", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    g.add_argument("--lr-grid-sgd", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.4])
    g = ap.add_argument_group("stage sigma")
    g.add_argument("--sigma-strategies", nargs="+", choices=SIGMA_STRATEGIES,
                   default=list(SIGMA_STRATEGIES))
    g = ap.add_argument_group("stage ranking")
    g.add_argument("--sigma-strategy", choices=SIGMA_STRATEGIES, default="late")
    g.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 1.0])
    g.add_argument("--plot-fraction", type=float, default=0.3)
    g.add_argument("--rank-mode", choices=("probabilistic", "strict"), default="probabilistic")
    g.add_argument("--exploration", type=float, default=0.2)
    g.add_argument("--rank-strength", type=float, default=3.0)
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        jupyter = (len(unknown) == 2 and unknown[0] == "-f"
                   and Path(unknown[1]).name.startswith("kernel-"))
        if not jupyter:
            ap.error("unrecognized arguments: " + " ".join(unknown))
    if args.quick:
        args.seeds, args.epochs, args.eval_every = min(args.seeds, 2), 400, 20
        args.early_warmup, args.warmup, args.sigma_every = 4, 40, 20
        args.tune_seeds, args.tune_epochs = 1, 200
        args.lr_grid_gd, args.lr_grid_sgd = [0.5, 1.0], [0.1, 0.2]
        args.fractions = [0.3, 1.0]
    args.fractions = sorted(set(float(f) for f in args.fractions))
    errors = []
    if not 0 < args.alpha <= 1:
        errors.append("--alpha must be in (0, 1]")
    if not all(0 < f <= 1 for f in args.fractions):
        errors.append("--fractions must be in (0, 1]")
    if not (0 <= args.exploration <= 1 and args.rank_strength >= 0 and 0 <= args.contamination < 1):
        errors.append("invalid exploration / rank-strength / contamination")
    if not (1 <= args.early_warmup < args.epochs and 1 <= args.warmup < args.epochs):
        errors.append("warmups must satisfy 1 <= warmup < --epochs")
    if not (1 <= args.sigma_every and 1 <= args.eval_every <= args.epochs):
        errors.append("invalid --sigma-every / --eval-every")
    if not (args.batch >= 1 and args.n >= 4 and args.n_val >= 4 and args.seeds >= 1
            and args.tune_seeds >= 1 and args.sigma_floor > 0 and args.workers >= 1):
        errors.append("invalid batch / n / seeds / workers / sigma-floor")
    if not (1 <= args.tune_epochs <= args.epochs):
        errors.append("--tune-epochs must be between 1 and --epochs")
    else:
        t = schedule(args, args.tune_epochs)
        if max(t["warmup"].values()) >= args.tune_epochs:
            errors.append(f"scaled warmup {max(t['warmup'].values())} >= --tune-epochs "
                          f"{args.tune_epochs}; increase --tune-epochs")
    if min(args.lr_grid_gd + args.lr_grid_sgd) <= 0:
        errors.append("learning rates must be positive")
    if errors:
        ap.error("; ".join(errors))
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        self_test(full=True)
        return None
    self_test(full=False)
    result = experiment(args)
    if args.show_plots:
        show_plots(args.out)
    return result


if __name__ == "__main__":
    main()
