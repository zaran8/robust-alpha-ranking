#!/usr/bin/env python3
"""alpha_images.py -- the alpha-loss of "Robust Backpropagation Learning via alpha-Divergence"
on the five image data sets of Rezende, Mohamed and Wierstra (ICML 2014):
MNIST, Frey faces, small NORB, CIFAR-10 and SVHN.  Gradient descent (GD) and stochastic
gradient descent (SGD) only.

Task (every data set): inpainting of 8x8 patches.  A patch of 8x8 pixels (x C channels) is cut
at a random position of a random image, and the network predicts the central 4x4 square
(16C outputs) from the 48 surrounding pixels (48C inputs).  Pixel values are in [0, 1].
Patches are cut from disjoint sets of training, validation and test images.

Contamination (training targets only; validation and test targets are always clean).
20% of the training patches have their target square corrupted, with the magnitudes of
Section IV-D (alpha_rank_full_v2.py, internal scale):
    asymmetric : +0.80 on every target value
    point_mass : every target value set to 1.50 + N(0, 0.008^2)
    symmetric  : + N(0, 0.80^2), independently on every target value
    clean      : no contamination (control)
Within a seed, the four conditions share the training patches and the corrupted positions.

Network: 48C -> 64 -> 64 -> 16C, tanh hidden units, linear outputs.  W ~ N(0, 1/fan_in), i.e.
variance 1/m; hidden biases 0; output biases = mean of the observed training targets.
Inputs are centred with the mean of the training inputs.

Loss per target value, residual r = g(x) - y, tau = 1 - alpha:
    BP    : r^2 / 2
    alpha : sigma^2/tau (1 - exp(-tau r^2 / (2 sigma^2)))   (sigma^2 times criterion (4))
            dl/dg = r q(r),   q(r) = exp(-tau r^2 / (2 sigma^2)),  -> r^2/2 as alpha -> 1
summed over the 16C outputs and averaged over the patches of the step.
  GD : theta <- theta - eta (1/n) sum over all n training patches
  SGD: theta <- theta - eta (1/B) sum over a mini-batch of B patches, reshuffled every epoch
Constant step sizes; both optimizers run the same number of epochs.
sigma = max(1.4826 MAD(all n x 16C training residuals), floor) is estimated at the end of an
MSE warm-up (epochs/10) and re-estimated every epochs/100 epochs (periodic estimation).

Protocol (as in Section IV-D)
  * Step size, and for the alpha-loss the floor of sigma, chosen for every data set, condition,
    optimizer and loss on 2 pilot realizations (independent random streams), minimising the
    mean log of the best validation MSE; the schedule is shrunk to --tune-epochs.
  * Checkpoints every epochs/200; the reported network is the checkpoint with the smallest
    MSE on the clean validation patches ("selected"); the last epoch is reported too ("final").
  * Test MSE on clean test patches cut from the test images; never used for any selection.
  * Paired design: per seed, BP and alpha share the patches, the contamination, the initial
    weights and the mini-batch order.  Two-sided Wilcoxon signed-rank tests on the paired
    differences, Holm correction over all data set x optimizer x condition cases.

Usage (see README.txt)
  python alpha_images.py --self-test
  python alpha_images.py --data data --datasets mnist --workers 32 --out results_images
  python alpha_images.py --data data --analyze-only --out results_images
Raw files are read from DATA/raw (get_data.sh) and converted once into DATA/prepared.
"""
import argparse
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import struct
import sys
import tarfile
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

DATASETS = ("mnist", "frey", "norb", "cifar10", "svhn")
LABELS = dict(mnist="MNIST", frey="Frey faces", norb="small NORB", cifar10="CIFAR-10", svhn="SVHN")
CONDITIONS = ("clean", "asymmetric", "point_mass", "symmetric")
OPTIMIZERS = ("GD", "SGD")
LOSSES = ("BP", "alpha")
BASE_SEED = 20261112
PATCH, HOLE = 8, 4
MASK = np.zeros((PATCH, PATCH), dtype=bool)
_a, _b = (PATCH - HOLE) // 2, (PATCH + HOLE) // 2
MASK[_a:_b, _a:_b] = True
# Settings that change how the program runs, not what it computes (not part of the cache key).
OPERATIONAL = {"data", "out", "datasets", "workers", "prepare", "self_test", "analyze_only",
               "reuse_across_code_changes", "seeds", "quick", "conditions"}

RAW = {
    "mnist": ["train-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz"],
    "frey": ["frey_rawface.mat"],
    "norb": ["smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz",
             "smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz"],
    "cifar10": ["cifar-10-binary.tar.gz"],
    "svhn": ["train_32x32.mat", "test_32x32.mat"],
}
# Number of images in the official files (checked after reading; a mismatch is reported).
OFFICIAL = {"mnist": (60000, 10000), "frey": (1965,), "norb": (24300, 24300),
            "cifar10": (50000, 10000), "svhn": (73257, 26032)}


def _rng(*key):
    """Independent, reproducible random stream for each purpose."""
    return np.random.default_rng(np.random.SeedSequence([BASE_SEED, *map(int, key)]))


# --------------------------------------------------------------------------
# Raw readers (no pickle is ever loaded)
# --------------------------------------------------------------------------
def read_idx_images(path):
    """MNIST IDX file (gzip): magic 2051, n, rows, cols (big-endian), then uint8 pixels."""
    with gzip.open(path, "rb") as fp:
        raw = fp.read()
    magic, n, h, w = struct.unpack(">IIII", raw[:16])
    if magic != 2051:
        raise ValueError(f"{path}: not an IDX image file (magic {magic})")
    a = np.frombuffer(raw, dtype=np.uint8, offset=16)
    if a.size != n * h * w:
        raise ValueError(f"{path}: {a.size} bytes of pixels, header says {n}x{h}x{w}")
    return a.reshape(n, h, w, 1)


def read_frey(path):
    """frey_rawface.mat: variable 'ff' of shape (560, N); each column is a 28x20 image, row-major."""
    from scipy.io import loadmat
    ff = np.asarray(loadmat(str(path))["ff"])
    if ff.ndim != 2 or ff.shape[0] != 28 * 20:
        raise ValueError(f"{path}: expected 'ff' of shape (560, N), got {ff.shape}")
    ff = ff.astype(np.float64)
    if ff.max() <= 1.0:
        ff = ff * 255.0
    return np.ascontiguousarray(np.rint(ff).T.reshape(-1, 28, 20, 1)).astype(np.uint8)


def read_norb_dat(path):
    """small-NORB -dat file (gzip): little-endian 'binary matrix', magic 0x1E3D4C55 (bytes),
    ndim, max(3, ndim) dimensions, then the data in row-major order: (pairs, 2, 96, 96)."""
    with gzip.open(path, "rb") as fp:
        raw = fp.read()
    magic, ndim = struct.unpack("<ii", raw[:8])
    if magic != 0x1E3D4C55:
        raise ValueError(f"{path}: not a small-NORB byte matrix (magic {magic:#x})")
    k = max(3, ndim)
    dims = struct.unpack("<" + "i" * k, raw[8:8 + 4 * k])[:ndim]
    a = np.frombuffer(raw, dtype=np.uint8, offset=8 + 4 * k)
    if ndim != 4 or dims[1] != 2 or a.size != int(np.prod(dims)):
        raise ValueError(f"{path}: unexpected dimensions {dims}")
    return a.reshape(dims)


def read_cifar10(path):
    """cifar-10-binary.tar.gz: records of 1 label byte + 1024 R + 1024 G + 1024 B bytes
    (row-major 32x32 planes).  Members are read from the archive, never extracted."""
    out = {}
    with tarfile.open(path, "r:gz") as tf:
        for m in tf.getmembers():
            name = Path(m.name).name
            if m.isfile() and (name.startswith("data_batch_") or name == "test_batch.bin"):
                raw = tf.extractfile(m).read()
                if len(raw) % 3073:
                    raise ValueError(f"{path}:{m.name}: size {len(raw)} is not a multiple of 3073")
                rec = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3073)
                out[name] = rec[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    names = [f"data_batch_{i}.bin" for i in range(1, 6)]
    missing = [n for n in names + ["test_batch.bin"] if n not in out]
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    return (np.ascontiguousarray(np.concatenate([out[n] for n in names])),
            np.ascontiguousarray(out["test_batch.bin"]))


def read_svhn(path):
    """SVHN *_32x32.mat: variable 'X' of shape (32, 32, 3, N)."""
    from scipy.io import loadmat
    X = np.asarray(loadmat(str(path), variable_names=["X"])["X"])
    if X.ndim != 4 or X.shape[:3] != (32, 32, 3):
        raise ValueError(f"{path}: expected X of shape (32, 32, 3, N), got {X.shape}")
    return np.ascontiguousarray(X.transpose(3, 0, 1, 2)).astype(np.uint8)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _random_split(n, frac, name):
    perm = np.random.default_rng(np.random.SeedSequence([BASE_SEED, 99, DATASETS.index(name)])) \
        .permutation(n)
    nv = int(round(n * frac))
    return np.sort(perm[nv:]), np.sort(perm[:nv])


def prepared_ok(data_dir, name):
    prep = Path(data_dir) / "prepared"
    return all((prep / f"{name}_{s}.npy").is_file() for s in ("train", "val", "test"))


def prepare_one(data_dir, name):
    """Read the raw files of one data set, split the images into training, validation and test
    pools, and save them as uint8 arrays (N, H, W, C) in DATA/prepared."""
    raw, prep = Path(data_dir) / "raw", Path(data_dir) / "prepared"
    files = [raw / f for f in RAW[name]]
    missing = [f.name for f in files if not f.is_file()]
    if missing:
        raise FileNotFoundError(f"{name}: raw files missing in {raw}: {', '.join(missing)} "
                                "(run get_data.sh first)")
    counts = []
    if name == "mnist":            # official split; validation = last 1/6 of the training images
        tr, te = read_idx_images(files[0]), read_idx_images(files[1])
        counts = [len(tr), len(te)]
        nv = len(tr) // 6
        pools = dict(train=tr[:len(tr) - nv], val=tr[len(tr) - nv:], test=te)
    elif name == "frey":           # video frames: contiguous 70 / 10 / 20 % blocks
        x = read_frey(files[0])
        counts = [len(x)]
        a, b = int(0.7 * len(x)), int(0.8 * len(x))
        pools = dict(train=x[:a], val=x[a:b], test=x[b:])
    elif name == "norb":           # official split (different object instances); validation =
        tr, te = read_norb_dat(files[0]), read_norb_dat(files[1])   # 10% of the training pairs
        counts = [len(tr), len(te)]
        itr, iva = _random_split(len(tr), 0.1, name)
        pools = {k: v.reshape(-1, v.shape[2], v.shape[3], 1)          # both cameras as images
                 for k, v in dict(train=tr[itr], val=tr[iva], test=te).items()}
    elif name == "cifar10":        # official split; validation = last 1/10 of the training images
        tr, te = read_cifar10(files[0])
        counts = [len(tr), len(te)]
        nv = len(tr) // 10
        pools = dict(train=tr[:len(tr) - nv], val=tr[len(tr) - nv:], test=te)
    elif name == "svhn":           # official split; validation = random 10% of the training images
        tr, te = read_svhn(files[0]), read_svhn(files[1])
        counts = [len(tr), len(te)]
        itr, iva = _random_split(len(tr), 0.1, name)
        pools = dict(train=tr[itr], val=tr[iva], test=te)
    else:
        raise ValueError(name)
    prep.mkdir(parents=True, exist_ok=True)
    for split, arr in pools.items():
        tmp = prep / f"{name}_{split}.tmp.npy"
        np.save(tmp, np.ascontiguousarray(arr, dtype=np.uint8))
        tmp.replace(prep / f"{name}_{split}.npy")
    official = tuple(counts) == OFFICIAL[name]
    meta = dict(dataset=name, images_in_raw_files=counts, official_counts=list(OFFICIAL[name]),
                matches_official=official,
                pools={k: list(v.shape) for k, v in pools.items()},
                raw_sha256={f.name: _sha256(f) for f in files})
    (prep / f"{name}.json").write_text(json.dumps(meta, indent=1))
    print(f"{name}: prepared " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in pools.items())
          + ("" if official else f"   WARNING: {counts} images, official {OFFICIAL[name]}"),
          flush=True)
    return meta


# --------------------------------------------------------------------------
# Patches and contamination
# --------------------------------------------------------------------------
_POOLS, _FIXED = {}, {}


def pools(data_dir, name):
    key = (str(data_dir), name)
    if key not in _POOLS:
        prep = Path(data_dir) / "prepared"
        _POOLS[key] = {s: np.load(prep / f"{name}_{s}.npy", mmap_mode="r")
                       for s in ("train", "val", "test")}
    return _POOLS[key]


def sample_patches(pool, n, rng):
    """n patches of PATCH x PATCH pixels: image, row and column drawn uniformly."""
    N, H, W, _ = pool.shape
    i = rng.integers(0, N, n)
    r = rng.integers(0, H - PATCH + 1, n)
    c = rng.integers(0, W - PATCH + 1, n)
    k = np.arange(PATCH)
    return np.asarray(pool[i[:, None, None], r[:, None, None] + k[None, :, None],
                           c[:, None, None] + k[None, None, :]])


def split_patches(p):
    """(n, P, P, C) uint8 -> inputs (n, 48C) and targets (n, 16C) in [0, 1], float32."""
    n = len(p)
    x = p[:, ~MASK, :].reshape(n, -1).astype(np.float32) / np.float32(255)
    y = p[:, MASK, :].reshape(n, -1).astype(np.float32) / np.float32(255)
    return x, y


def corrupt(Y, idx, condition, rng):
    Y = Y.copy()
    if condition == "asymmetric":
        Y[idx] += np.float32(0.80)
    elif condition == "point_mass":
        Y[idx] = 1.50 + rng.normal(0, 0.008, (len(idx), Y.shape[1]))
    elif condition == "symmetric":
        Y[idx] += rng.normal(0, 0.80, (len(idx), Y.shape[1])).astype(np.float32)
    elif condition != "clean":
        raise ValueError(condition)
    return Y.astype(np.float32)


def fixed_patches(args, name, split, pilot):
    n = args.n_val if split == "val" else args.n_test
    key = (str(args.data), name, split, int(pilot), n)
    if key not in _FIXED:
        d = DATASETS.index(name)
        stream = _rng(13, int(pilot), d) if split == "val" else _rng(16, d)
        _FIXED[key] = split_patches(sample_patches(pools(args.data, name)[split], n, stream))
    return _FIXED[key]


def make_data(args, name, condition, seed, pilot=False):
    d, c, p = DATASETS.index(name), CONDITIONS.index(condition), int(pilot)
    X, Y = split_patches(sample_patches(pools(args.data, name)["train"], args.n_train,
                                        _rng(10, p, d, seed)))
    m = int(round(args.n_train * args.contamination))
    idx = np.sort(_rng(11, p, d, seed).permutation(args.n_train)[:m])
    bad = np.zeros(args.n_train, dtype=bool)
    if condition != "clean":
        bad[idx] = True
        Y = corrupt(Y, idx, condition, _rng(12, p, d, seed, c))
    Xv, Yv = fixed_patches(args, name, "val", pilot)
    data = dict(X=X, Y=Y, bad=bad, Xv=Xv, Yv=Yv, mu=X.mean(axis=0))
    if not pilot:
        data["Xt"], data["Yt"] = fixed_patches(args, name, "test", False)
    return data


# --------------------------------------------------------------------------
# Network, loss, gradient
# --------------------------------------------------------------------------
def init_params(widths, seed, pilot, out_bias, dtype=np.float32):
    """W ~ N(0, 1/fan_in): variance 1/m, standard deviation 1/sqrt(m)."""
    rng = _rng(20, int(pilot), seed)
    theta = []
    for j in range(len(widths) - 1):
        W = rng.normal(0.0, 1.0 / np.sqrt(widths[j]), (widths[j], widths[j + 1])).astype(dtype)
        theta.append([W, np.zeros(widths[j + 1], dtype)])
    theta[-1][1] = np.array(out_bias, dtype=dtype)
    return theta


def forward(theta, X):
    acts = [X]
    for j, (W, b) in enumerate(theta):
        h = acts[-1] @ W + b
        acts.append(np.tanh(h) if j < len(theta) - 1 else h)
    return acts


def q_weights(r, tau, sigma):
    return np.exp(-0.5 * tau * np.square(r / sigma))


def objective(theta, X, Y, tau=0.0, sigma=1.0):
    """(1/n) sum_i sum_k l(r_ik); used by the self-test only."""
    r = forward(theta, X)[-1] - Y
    if tau > 0:
        return float(sigma ** 2 / tau * np.sum(-np.expm1(-0.5 * tau * np.square(r / sigma))) / len(X))
    return float(0.5 * np.sum(np.square(r)) / len(X))


def loss_grad(theta, X, Y, tau=0.0, sigma=1.0):
    """Gradient of objective() with sigma held fixed."""
    acts = forward(theta, X)
    r = acts[-1] - Y
    delta = (r * q_weights(r, tau, sigma) if tau > 0 else r) / len(X)
    grads = [None] * len(theta)
    for j in reversed(range(len(theta))):
        W, _ = theta[j]
        grads[j] = (acts[j].T @ delta, delta.sum(axis=0))
        if j:
            a = acts[j]
            delta = (delta @ W.T) * (1 - a * a)
    return r, grads


def scale_from_residuals(r, floor):
    r = np.asarray(r).ravel()
    med = np.median(r)
    return max(float(1.4826 * np.median(np.abs(r - med))), float(floor))


def mse(theta, X, Y):
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.mean(np.square(forward(theta, X)[-1] - Y)))


def schedule(args, epochs):
    """Every length is defined relative to --epochs and shrunk proportionally for tuning."""
    def s(e):
        return max(1, int(round(e * epochs / args.epochs)))
    return dict(epochs=int(epochs), warmup=s(args.warmup), sigma_every=s(args.sigma_every),
                eval_every=s(args.eval_every))


def train(args, data, opt, loss, lr, floor, sched, seed, pilot=False, record=False):
    X, Y, bad = data["X"] - data["mu"], data["Y"], data["bad"]
    Xv, Yv = data["Xv"] - data["mu"], data["Yv"]
    Xt, Yt = (data["Xt"] - data["mu"], data["Yt"]) if "Xt" in data else (None, None)
    n = len(Y)
    theta = init_params((X.shape[1], *args.hidden, Y.shape[1]), seed, pilot, Y.mean(axis=0))
    batch_rng = _rng(30, int(pilot), seed)          # identical for BP and alpha
    is_alpha, E = loss == "alpha", sched["epochs"]
    tau = 1.0 - args.alpha
    warm = sched["warmup"] if is_alpha else E + 1
    step = args.batch if opt == "SGD" else n
    evals = set(range(0, E + 1, sched["eval_every"])) | {E} | ({warm} if is_alpha else set())
    lr32 = np.float32(lr)
    sigma, diverged = None, False
    best = dict(val=np.inf, epoch=-1, test=np.nan)
    last = dict(val=np.nan, test=np.nan)
    test_at_warmup = np.nan
    curve = {k: [] for k in ("epoch", "val_mse", "test_mse")}
    scale_log = {k: [] for k in ("epoch", "sigma", "q_inlier", "q_outlier", "mse_inlier")}
    updates, t0 = 0, time.perf_counter()
    for epoch in range(E + 1):
        if is_alpha and epoch >= warm and (epoch - warm) % sched["sigma_every"] == 0:
            with np.errstate(over="ignore", invalid="ignore"):
                r = forward(theta, X)[-1] - Y
            sigma = scale_from_residuals(r, floor)
            if record:
                q = q_weights(r, tau, sigma)
                for k, v in (("epoch", epoch), ("sigma", sigma), ("q_inlier", q[~bad].mean()),
                             ("q_outlier", q[bad].mean() if bad.any() else np.nan),
                             ("mse_inlier", np.mean(np.square(r[~bad])))):
                    scale_log[k].append(float(v))
        active = is_alpha and epoch >= warm
        if epoch in evals:
            v = mse(theta, Xv, Yv)
            t = mse(theta, Xt, Yt) if Xt is not None else np.nan
            if not np.isfinite(v):
                diverged = True
                last = dict(val=np.nan, test=np.nan)
                break
            if v < best["val"]:
                best = dict(val=v, epoch=epoch, test=t)
            last = dict(val=v, test=t)
            if is_alpha and epoch == warm:
                test_at_warmup = t
            if record:
                for k, x in (("epoch", epoch), ("val_mse", v), ("test_mse", t)):
                    curve[k].append(float(x))
        if epoch == E:
            break
        order = batch_rng.permutation(n) if opt == "SGD" else None
        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(0, n, step):
                if order is None:
                    xb, yb = X, Y
                else:
                    ids = order[i:i + step]
                    xb, yb = X[ids], Y[ids]
                _, grads = loss_grad(theta, xb, yb, tau if active else 0.0,
                                     sigma if active else 1.0)
                for (W, b), (gW, gb) in zip(theta, grads):
                    W -= lr32 * gW
                    b -= lr32 * gb
                updates += 1
        if not all(np.isfinite(W).all() and np.isfinite(b).all() for W, b in theta):
            diverged = True
            last = dict(val=np.nan, test=np.nan)
            break
    q_in = q_out = mse_in = np.nan
    if not diverged:
        r = forward(theta, X)[-1] - Y
        mse_in = float(np.mean(np.square(r[~bad])))
        if is_alpha and sigma is not None:
            q = q_weights(r, tau, sigma)
            q_in = float(q[~bad].mean())
            q_out = float(q[bad].mean()) if bad.any() else np.nan
    final = dict(opt=opt, loss=loss, lr=float(lr), floor=float(floor) if is_alpha else np.nan,
                 test_selected=best["test"], val_selected=best["val"], best_epoch=best["epoch"],
                 warmup_epoch=warm if is_alpha else np.nan,
                 selected_in_warmup=bool(is_alpha and 0 <= best["epoch"] <= warm),
                 test_final=last["test"], val_final=last["val"], test_at_warmup=test_at_warmup,
                 sigma_final=sigma if sigma is not None else np.nan,
                 q_inlier_final=q_in, q_outlier_final=q_out, train_mse_inlier_final=mse_in,
                 updates=updates, elapsed_s=time.perf_counter() - t0, diverged=diverged)
    return final, curve, scale_log


# --------------------------------------------------------------------------
# Jobs, cache, orchestration
# --------------------------------------------------------------------------
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
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fp:
        json.dump(_clean(obj), fp, separators=(",", ":"))
    tmp.replace(path)


def load_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as fp:
        return json.load(fp)


def source_hash():
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    except (NameError, OSError):
        return "unknown-source"


def settings(args):
    return {k: v for k, v in sorted(vars(args).items()) if k not in OPERATIONAL}


def cache_dir(args):
    cfg = settings(args)
    cfg["source"] = "any" if args.reuse_across_code_changes else source_hash()
    digest = hashlib.sha256(json.dumps(_clean(cfg), sort_keys=True).encode()).hexdigest()[:14]
    d = Path(args.out) / "cache" / f"images_{digest}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "settings.json").write_text(json.dumps(_clean(cfg), indent=1, sort_keys=True))
    return d


def tag(lr, floor):
    return f"lr{lr:g}" + ("" if floor is None else f"_f{floor:g}")


def tune_path(cdir, name, cond, opt, loss, lr, floor, k):
    return cdir / f"tune_{name}_{cond}_{opt}_{loss}_{tag(lr, floor)}_p{k}.json.gz"


def eval_path(cdir, name, cond, opt, loss, lr, floor, seed):
    return cdir / f"eval_{name}_{cond}_{opt}_{loss}_{tag(lr, floor)}_s{seed}.json.gz"


def _ext_file(cdir, name):
    return cdir / f"grid_extensions_{name}.json"


def lr_grid(args, cdir, name, cond, opt, loss):
    """Base grid of step sizes plus the values added when a choice fell on the edge."""
    base = [float(x) for x in (args.lr_grid_gd if opt == "GD" else args.lr_grid_sgd)]
    f = _ext_file(cdir, name)
    ext = json.loads(f.read_text()).get(f"{cond}|{opt}|{loss}", []) if f.is_file() else []
    return sorted(set(base) | {float(x) for x in ext})


def grid(args, cdir, name, cond, opt, loss):
    floors = args.floors if loss == "alpha" else [None]
    return [(lr, None if f is None else float(f))
            for lr in lr_grid(args, cdir, name, cond, opt, loss) for f in floors]


def tune_task(args, name, cond, opt, loss, lr, floor, k, path):
    data = make_data(args, name, cond, k, pilot=True)
    final, _, _ = train(args, data, opt, loss, lr, floor, schedule(args, args.tune_epochs), k,
                        pilot=True)
    final.update(dataset=name, condition=cond, pilot_seed=k)
    save_gz(path, dict(final=final))
    return str(path)


def eval_task(args, name, cond, opt, loss, lr, floor, seed, path):
    data = make_data(args, name, cond, seed)
    final, curve, scale_log = train(args, data, opt, loss, lr, floor,
                                    schedule(args, args.epochs), seed, record=True)
    final.update(dataset=name, condition=cond, seed=seed)
    save_gz(path, dict(final=final, curve=curve, scale=scale_log))
    return str(path)


def run_jobs(jobs, workers, label):
    if not jobs:
        print(f"{label}: nothing to run (all cached)", flush=True)
        return
    print(f"{label}: {len(jobs)} runs on {workers} workers", flush=True)
    t0 = time.perf_counter()
    every = max(1, len(jobs) // 40)
    if workers <= 1:
        for i, (fn, a) in enumerate(jobs, 1):
            fn(*a)
            if i % every == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] {Path(a[-1]).name}  ({time.perf_counter() - t0:.0f}s)",
                      flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fn, *a) for fn, a in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if i % every == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] {Path(res).name}  ({time.perf_counter() - t0:.0f}s)",
                      flush=True)


def choose(args, cdir, name, cond, opt, loss):
    """Configuration (step size, floor) minimising the mean log of the best validation MSE
    over the pilot seeds; a run that diverged counts as infinite."""
    crit, rows = {}, []
    for lr, f in grid(args, cdir, name, cond, opt, loss):
        vals = []
        for k in range(args.tune_seeds):
            p = tune_path(cdir, name, cond, opt, loss, lr, f, k)
            if not p.is_file():
                return None, []
            fin = load_gz(p)["final"]
            v = _num(fin["val_selected"])
            vals.append(np.inf if fin["diverged"] or not np.isfinite(v) else float(v))
            rows.append(dict(dataset=name, condition=cond, opt=opt, loss=loss, lr=lr, floor=f,
                             pilot_seed=k, val_selected=vals[-1], best_epoch=fin["best_epoch"],
                             diverged=fin["diverged"]))
        crit[(lr, f)] = float(np.mean(np.log(vals))) if all(np.isfinite(vals)) else np.inf
    lr, f = min(crit, key=crit.get)
    if not np.isfinite(crit[(lr, f)]):
        raise RuntimeError(f"{name} {cond} {opt} {loss}: every configuration diverged")
    lrs = lr_grid(args, cdir, name, cond, opt, loss)
    for r in rows:
        r["chosen"] = (r["lr"], r["floor"]) == (lr, f)
    return dict(lr=lr, floor=f, at_lr_edge=lr in (min(lrs), max(lrs)),
                at_floor_edge=(f is not None and len(args.floors) > 1
                               and f in (min(args.floors), max(args.floors))),
                criterion=crit[(lr, f)], lr_grid=lrs), rows


def experiment(args):
    cdir = cache_dir(args)
    print(f"cache: {cdir}", flush=True)
    for name in args.datasets:
        if not prepared_ok(args.data, name):
            prepare_one(args.data, name)
    cases = [(n, c, o, l) for n in args.datasets for c in args.conditions for o in OPTIMIZERS
             for l in LOSSES]
    # Tuning; whenever the chosen step size lies on the edge of its grid, the grid is extended
    # by a factor of 2 beyond that edge and the new values are tuned too.
    for rnd in range(args.max_grid_extensions + 1):
        jobs = [(tune_task, (args, n, c, o, l, lr, f, k, tune_path(cdir, n, c, o, l, lr, f, k)))
                for (n, c, o, l) in cases for lr, f in grid(args, cdir, n, c, o, l)
                for k in range(args.tune_seeds)]
        run_jobs([j for j in jobs if not Path(j[1][-1]).is_file()], args.workers,
                 f"tuning ({args.tune_epochs} epochs)" + (f", grid extension {rnd}" if rnd else ""))
        if rnd == args.max_grid_extensions:
            break
        added = 0
        for name in args.datasets:
            f = _ext_file(cdir, name)
            ext = json.loads(f.read_text()) if f.is_file() else {}
            for (n, c, o, l) in [x for x in cases if x[0] == name]:
                ch, _ = choose(args, cdir, n, c, o, l)
                lrs = lr_grid(args, cdir, n, c, o, l)
                new = 2 * lrs[-1] if ch["lr"] == lrs[-1] else (lrs[0] / 2 if ch["lr"] == lrs[0]
                                                                else None)
                # at most --max-grid-extensions values per case, over all invocations
                if new is not None and len(ext.get(f"{c}|{o}|{l}", [])) < args.max_grid_extensions:
                    ext.setdefault(f"{c}|{o}|{l}", []).append(new)
                    added += 1
                    print(f"  {n} {c} {o} {l}: chosen step size {ch['lr']:g} on the edge of the "
                          f"grid; adding {new:g}", flush=True)
            if ext:
                f.write_text(json.dumps(ext, indent=1))
        if not added:
            break
    jobs, tune_rows = [], []
    for name in args.datasets:
        cf = cdir / f"chosen_{name}.json"
        chosen = json.loads(cf.read_text()) if cf.is_file() else {}
        for (n, c, o, l) in [x for x in cases if x[0] == name]:
            ch, rows = choose(args, cdir, n, c, o, l)
            tune_rows += rows
            chosen[f"{c}|{o}|{l}"] = ch
            for s in range(args.seeds):
                p = eval_path(cdir, n, c, o, l, ch["lr"], ch["floor"], s)
                if not p.is_file():
                    jobs.append((eval_task, (args, n, c, o, l, ch["lr"], ch["floor"], s, p)))
        cf.write_text(json.dumps(_clean(chosen), indent=1))
    try:
        import pandas as pd
        pd.DataFrame(tune_rows).to_csv(cdir / f"tuning_{'_'.join(args.datasets)}.csv", index=False)
    except ImportError:
        pass
    run_jobs(jobs, args.workers, f"evaluation ({args.epochs} epochs, {args.seeds} seeds)")
    analyze(args, cdir)


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
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


def gmean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.mean(np.log(x)))) if len(x) else np.nan


def load_runs(args, cdir):
    finals, curves, scales = [], [], []
    for name in DATASETS:
        cf = cdir / f"chosen_{name}.json"
        if not cf.is_file():
            continue
        for key, ch in json.loads(cf.read_text()).items():
            cond, opt, loss = key.split("|")
            for s in range(args.seeds):
                p = eval_path(cdir, name, cond, opt, loss, ch["lr"], ch["floor"], s)
                if not p.is_file():
                    continue
                d = load_gz(p)
                f = {k: _num(v) for k, v in d["final"].items()}
                f.update(at_lr_edge=ch["at_lr_edge"], at_floor_edge=ch["at_floor_edge"])
                finals.append(f)
                c = d["curve"]
                for e, v, t in zip(c["epoch"], c["val_mse"], c["test_mse"]):
                    curves.append((name, opt, cond, loss, s, e, _num(v), _num(t)))
                sc = d["scale"]
                for row in zip(sc["epoch"], sc["sigma"], sc["q_inlier"], sc["q_outlier"],
                               sc["mse_inlier"]):
                    scales.append((name, opt, cond, s, *[_num(x) for x in row]))
    return finals, curves, scales


def analyze(args, cdir, out=None):
    import pandas as pd
    from scipy.stats import wilcoxon
    out = Path(out or args.out)
    out.mkdir(parents=True, exist_ok=True)
    finals, curves, scales = load_runs(args, cdir)
    if not finals:
        print("analysis: no evaluation results yet", flush=True)
        return None
    runs = pd.DataFrame(finals)
    runs.to_csv(out / "runs.csv", index=False)
    tests = []
    for metric in ("test_selected", "test_final"):
        block = []
        for name in DATASETS:
            for opt in OPTIMIZERS:
                for cond in CONDITIONS:
                    sub = runs[(runs.dataset == name) & (runs.opt == opt) & (runs.condition == cond)]
                    if sub.empty:
                        continue
                    tab = sub.pivot_table(index="seed", columns="loss", values=metric,
                                          aggfunc="first")
                    if "BP" not in tab or "alpha" not in tab:
                        continue
                    both = tab[["BP", "alpha"]].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(both) < 2:
                        continue
                    d = (both["BP"] - both["alpha"]).to_numpy()
                    nz = int(np.sum(d != 0))
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        p = float(wilcoxon(d).pvalue) if nz else 1.0
                    a = sub[sub.loss == "alpha"]
                    b = sub[sub.loss == "BP"]
                    block.append(dict(
                        metric=metric, dataset=name, opt=opt, condition=cond, n_pairs=len(d),
                        median_BP=float(both["BP"].median()),
                        median_alpha=float(both["alpha"].median()),
                        ratio_BP_over_alpha=float(np.median(both["BP"] / both["alpha"])),
                        ratio_min=float((both["BP"] / both["alpha"]).min()),
                        ratio_max=float((both["BP"] / both["alpha"]).max()),
                        alpha_better=int(np.sum(d > 0)), alpha_worse=int(np.sum(d < 0)),
                        p_raw=p, min_possible_p=min(1.0, 2.0 / 2 ** nz) if nz else 1.0,
                        lr_BP=float(b.lr.iloc[0]), lr_alpha=float(a.lr.iloc[0]),
                        floor_alpha=float(a.floor.iloc[0]),
                        lr_edge_BP=bool(b.at_lr_edge.iloc[0]),
                        lr_edge_alpha=bool(a.at_lr_edge.iloc[0]),
                        floor_edge_alpha=bool(a.at_floor_edge.iloc[0]),
                        alpha_selected_in_warmup=float(a.selected_in_warmup.mean()),
                        sigma_final_median=float(a.sigma_final.median()),
                        sigma_at_floor=float(np.mean(np.isclose(a.sigma_final, a.floor))),
                        q_inlier_final=float(a.q_inlier_final.median()),
                        q_outlier_final=float(a.q_outlier_final.median()),
                        best_epoch_BP=float(b.best_epoch.median()),
                        best_epoch_alpha=float(a.best_epoch.median()),
                        diverged_BP=int(b.diverged.sum()), diverged_alpha=int(a.diverged.sum())))
        adj = holm([r["p_raw"] for r in block])
        for r, q in zip(block, adj):
            r["family_size"] = len(block)
            r["p_holm"] = float(q)
            r["significant"] = bool(q < 0.05)
        tests += block
    tests = pd.DataFrame(tests)
    tests.to_csv(out / "summary.csv", index=False)
    # Geometric means of the per-case median ratios.
    sel = tests[tests.metric == "test_selected"]
    gm_rows = []
    for opt in OPTIMIZERS:
        for cond in CONDITIONS:
            s = sel[(sel.opt == opt) & (sel.condition == cond)]
            if len(s):
                gm_rows.append(dict(opt=opt, condition=cond, datasets=len(s),
                                    gm_ratio=gmean(s.ratio_BP_over_alpha),
                                    min_ratio=float(s.ratio_BP_over_alpha.min()),
                                    max_ratio=float(s.ratio_BP_over_alpha.max()),
                                    alpha_significantly_better=int(
                                        (s.significant & (s.median_alpha < s.median_BP)).sum()),
                                    alpha_significantly_worse=int(
                                        (s.significant & (s.median_alpha > s.median_BP)).sum())))
        s = sel[(sel.opt == opt) & (sel.condition != "clean")]
        if len(s):
            gm_rows.append(dict(opt=opt, condition="all contaminated", datasets=len(s),
                                gm_ratio=gmean(s.ratio_BP_over_alpha),
                                min_ratio=float(s.ratio_BP_over_alpha.min()),
                                max_ratio=float(s.ratio_BP_over_alpha.max()),
                                alpha_significantly_better=int(
                                    (s.significant & (s.median_alpha < s.median_BP)).sum()),
                                alpha_significantly_worse=int(
                                    (s.significant & (s.median_alpha > s.median_BP)).sum())))
    gm = pd.DataFrame(gm_rows)
    gm.to_csv(out / "gain_by_condition.csv", index=False)
    # Training curves and scale/weights, summarised over seeds.
    cv = pd.DataFrame(curves, columns=["dataset", "opt", "condition", "loss", "seed", "epoch",
                                       "val_mse", "test_mse"])
    q = cv.groupby(["dataset", "opt", "condition", "loss", "epoch"]).test_mse
    pd.DataFrame({"median": q.median(), "q25": q.quantile(0.25), "q75": q.quantile(0.75),
                  "n": q.count()}).reset_index().to_csv(out / "curves.csv", index=False)
    if scales:
        sc = pd.DataFrame(scales, columns=["dataset", "opt", "condition", "seed", "epoch", "sigma",
                                           "q_inlier", "q_outlier", "mse_inlier"])
        sc.groupby(["dataset", "opt", "condition", "epoch"])[
            ["sigma", "q_inlier", "q_outlier", "mse_inlier"]].median().reset_index() \
            .to_csv(out / "sigma_weights.csv", index=False)
    write_table(sel, out / "table_images.tex")
    quick_figure(sel, out / "ratios.png")
    print_summary(sel, gm)
    return tests


def write_table(sel, path):
    """Draft LaTeX table: test MSE of BP on clean data and ratios BP/alpha per condition."""
    lines = [r"% Draft generated by alpha_images.py.  Ratio = test MSE of BP / test MSE of the",
             r"% alpha-loss (median over the paired realizations); * : significant after Holm.",
             r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
             r" & & BP clean & \multicolumn{4}{c}{Ratio BP / $\alpha$-loss}\\",
             r"\cmidrule(l){4-7}",
             r"Data set & Opt. & ($\times10^{-3}$) & Clean & Asym. & P.-mass & Sym.\\", r"\midrule"]
    blocks = []
    for name in DATASETS:
        rows = []
        for opt in OPTIMIZERS:
            s = sel[(sel.dataset == name) & (sel.opt == opt)].set_index("condition")
            if s.empty:
                continue
            cells = []
            for cond in CONDITIONS:
                if cond not in s.index:
                    cells.append("--")
                    continue
                r = s.loc[cond]
                mark = "$^{*}$" if r.significant else ""
                cells.append(f"{r.ratio_BP_over_alpha:.2f}{mark}")
            clean = (f"{1e3 * s.loc['clean'].median_BP:.2f}" if "clean" in s.index else "--")
            rows.append(f"{'' if rows else LABELS[name]} & {opt} & {clean} & "
                        + " & ".join(cells) + r"\\")
        if rows:
            blocks.append(rows)
    for i, rows in enumerate(blocks):
        if i:
            lines.append(r"\addlinespace[2pt]")
        lines += rows
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(path).write_text("\n".join(lines) + "\n")


def quick_figure(sel, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    marks = dict(clean="o", asymmetric="^", point_mass="s", symmetric="D")
    for ax, opt in zip(axs, OPTIMIZERS):
        for j, cond in enumerate(CONDITIONS):
            s = sel[(sel.opt == opt) & (sel.condition == cond)]
            x = [DATASETS.index(d) + (j - 1.5) * 0.12 for d in s.dataset]
            ax.scatter(x, s.ratio_BP_over_alpha, marker=marks[cond], label=cond, s=22,
                       facecolors=["k" if v else "none" for v in s.significant], edgecolors="k")
        ax.axhline(1, color="0.6", lw=0.8)
        ax.set_xlim(-0.5, len(DATASETS) - 0.5)
        ax.set_yscale("log")
        ax.set_xticks(range(len(DATASETS)), [LABELS[d] for d in DATASETS], fontsize=8)
        ax.set_title(opt, fontsize=9)
    axs[0].set_ylabel("test MSE of BP / alpha-loss")
    axs[1].legend(fontsize=7, frameon=False)
    fig.text(0.5, 0.01, "filled: significant after Holm", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_summary(sel, gm):
    print("\nTest MSE (x1e-3) at the selected checkpoint, median over seeds; ratio BP/alpha "
          "(* Holm-significant)", flush=True)
    for name in DATASETS:
        for opt in OPTIMIZERS:
            s = sel[(sel.dataset == name) & (sel.opt == opt)]
            if s.empty:
                continue
            cells = []
            for cond in CONDITIONS:
                r = s[s.condition == cond]
                if len(r):
                    r = r.iloc[0]
                    cells.append(f"{cond[:5]} {1e3 * r.median_BP:7.3f}/{1e3 * r.median_alpha:7.3f}"
                                 f" {r.ratio_BP_over_alpha:6.2f}{'*' if r.significant else ' '}")
            print(f"  {name:8s} {opt:3s} " + "  ".join(cells), flush=True)
    if len(gm):
        print("\nGeometric mean over data sets of the ratio BP/alpha:", flush=True)
        for _, r in gm.iterrows():
            print(f"  {r.opt:3s} {r.condition:17s} {r.gm_ratio:6.2f}  (range {r.min_ratio:.2f}-"
                  f"{r.max_ratio:.2f}; alpha significantly better in "
                  f"{r.alpha_significantly_better}, worse in {r.alpha_significantly_worse} "
                  f"of {r.datasets})", flush=True)


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
def _write_fake_raw(raw, rng, sizes):
    """Small files in the exact formats of the real downloads, with known pixel patterns."""
    raw.mkdir(parents=True, exist_ok=True)
    from scipy.io import savemat
    imgs = {}

    def pattern(n, h, w, c, off):
        g = (np.arange(n)[:, None, None, None] * 7 + np.arange(h)[None, :, None, None] * 3
             + np.arange(w)[None, None, :, None] * 5 + np.arange(c)[None, None, None, :] * 11 + off)
        smooth = (g % 256).astype(np.uint8)
        noise = rng.integers(0, 256, (n, h, w, c), dtype=np.uint8)
        return np.where(rng.random((n, 1, 1, 1)) < 0.5, smooth, noise).astype(np.uint8)

    for fname, n in (("train-images-idx3-ubyte.gz", sizes["mnist"][0]),
                     ("t10k-images-idx3-ubyte.gz", sizes["mnist"][1])):
        x = pattern(n, 28, 28, 1, len(fname))
        with gzip.open(raw / fname, "wb") as fp:
            fp.write(struct.pack(">IIII", 2051, n, 28, 28) + x.tobytes())
        imgs[fname] = x
    x = pattern(sizes["frey"], 28, 20, 1, 3)
    savemat(raw / "frey_rawface.mat", {"ff": x.reshape(len(x), -1).T.copy()})
    imgs["frey"] = x
    for fname, n in ((RAW["norb"][0], sizes["norb"][0]), (RAW["norb"][1], sizes["norb"][1])):
        x = pattern(n * 2, 96, 96, 1, 9).reshape(n, 2, 96, 96)
        with gzip.open(raw / fname, "wb") as fp:
            fp.write(struct.pack("<ii", 0x1E3D4C55, 4) + struct.pack("<iiii", n, 2, 96, 96)
                     + x.tobytes())
        imgs[fname] = x
    tr = pattern(sizes["cifar10"][0], 32, 32, 3, 5)
    te = pattern(sizes["cifar10"][1], 32, 32, 3, 6)
    with tarfile.open(raw / "cifar-10-binary.tar.gz", "w:gz") as tf:
        parts = np.array_split(tr, 5)
        for name, arr in [(f"data_batch_{i + 1}.bin", parts[i]) for i in range(5)] + \
                         [("test_batch.bin", te)]:
            rec = np.concatenate([np.zeros((len(arr), 1), np.uint8),
                                  arr.transpose(0, 3, 1, 2).reshape(len(arr), -1)], axis=1)
            b = rec.tobytes()
            info = tarfile.TarInfo(f"cifar-10-batches-bin/{name}")
            info.size = len(b)
            tf.addfile(info, io.BytesIO(b))
    imgs["cifar_train"], imgs["cifar_test"] = tr, te
    for fname, n in (("train_32x32.mat", sizes["svhn"][0]), ("test_32x32.mat", sizes["svhn"][1])):
        x = pattern(n, 32, 32, 3, 8)
        savemat(raw / fname, {"X": x.transpose(1, 2, 3, 0).copy(),
                              "y": np.ones((n, 1), np.uint8)})
        imgs[fname] = x
    return imgs


def self_test():
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    # 1. Gradient of the alpha-loss and of BP against finite differences (float64).
    X = rng.normal(size=(7, 12))
    Y = rng.normal(size=(7, 4))
    th = init_params((12, 5, 6, 4), 0, False, np.zeros(4), dtype=np.float64)
    for tau, sigma in ((0.1, 0.3), (0.5, 1.7), (0.0, 1.0)):
        _, g = loss_grad(th, X, Y, tau, sigma)
        for j in range(len(th)):
            for which in (0, 1):
                P = th[j][which]
                for idx in [tuple(rng.integers(0, s) for s in P.shape) for _ in range(4)]:
                    old = P[idx]
                    P[idx] = old + 1e-6
                    fp = objective(th, X, Y, tau, sigma)
                    P[idx] = old - 1e-6
                    fm = objective(th, X, Y, tau, sigma)
                    P[idx] = old
                    num = (fp - fm) / 2e-6
                    assert abs(num - g[j][which][idx]) < 1e-6 * max(1, abs(num)), \
                        (tau, sigma, j, which, num, g[j][which][idx])
    # alpha -> 1 gives the BP gradient
    _, g0 = loss_grad(th, X, Y, 0.0, 1.0)
    _, g1 = loss_grad(th, X, Y, 1e-9, 1.0)
    assert all(np.allclose(a, b, atol=1e-8) for p, q in zip(g0, g1) for a, b in zip(p, q))
    # MAD scale and floor
    r = rng.normal(0, 0.3, 200001)
    assert abs(scale_from_residuals(r, 0.05) - 0.3) < 0.01
    assert scale_from_residuals(np.zeros(10), 0.05) == 0.05
    print("  gradient, alpha->1 limit and scale: ok", flush=True)

    tmp = Path(tempfile.mkdtemp(prefix="alpha_images_selftest_"))
    try:
        # 2. Readers on files in the exact raw formats, orientation checks.
        sizes = dict(mnist=(60, 12), frey=50, norb=(10, 6), cifar10=(50, 10), svhn=(40, 12))
        imgs = _write_fake_raw(tmp / "raw", rng, sizes)
        raw = tmp / "raw"
        assert np.array_equal(read_idx_images(raw / RAW["mnist"][0]), imgs[RAW["mnist"][0]])
        assert np.array_equal(read_frey(raw / "frey_rawface.mat"), imgs["frey"])
        assert np.array_equal(read_norb_dat(raw / RAW["norb"][0]), imgs[RAW["norb"][0]])
        ctr, cte = read_cifar10(raw / "cifar-10-binary.tar.gz")
        assert np.array_equal(ctr, imgs["cifar_train"]) and np.array_equal(cte, imgs["cifar_test"])
        assert np.array_equal(read_svhn(raw / "train_32x32.mat"), imgs["train_32x32.mat"])
        for name in DATASETS:
            prepare_one(tmp, name)
        P = {n: pools(tmp, n) for n in DATASETS}
        assert P["mnist"]["train"].shape == (50, 28, 28, 1) and P["mnist"]["val"].shape[0] == 10
        assert P["frey"]["train"].shape == (35, 28, 20, 1) and P["frey"]["test"].shape[0] == 10
        assert P["norb"]["train"].shape == (18, 96, 96, 1) and P["norb"]["test"].shape[0] == 12
        assert np.array_equal(P["norb"]["test"][1, :, :, 0], imgs[RAW["norb"][1]][0, 1])
        assert P["cifar10"]["train"].shape == (45, 32, 32, 3)
        assert P["svhn"]["train"].shape[0] + P["svhn"]["val"].shape[0] == 40
        print("  readers, orientation and splits: ok", flush=True)
        # 3. Patches: the target is the central 4x4 square of every channel.
        pool = np.arange(2 * 10 * 12 * 3).reshape(2, 10, 12, 3).astype(np.uint8)
        g = np.random.default_rng(5)
        pat = sample_patches(pool, 50, g)
        g = np.random.default_rng(5)
        i, rr, cc = g.integers(0, 2, 50), g.integers(0, 3, 50), g.integers(0, 5, 50)
        for k in range(50):
            assert np.array_equal(pat[k], pool[i[k], rr[k]:rr[k] + 8, cc[k]:cc[k] + 8])
        x, y = split_patches(pat)
        assert x.shape == (50, 144) and y.shape == (50, 48)
        assert np.allclose(y[0].reshape(4, 4, 3) * 255, pat[0, 2:6, 2:6])
        # 4. Contamination.
        Y = rng.random((1000, 16)).astype(np.float32)
        idx = np.arange(200)
        a = corrupt(Y, idx, "asymmetric", rng)
        assert np.allclose(a[:200] - Y[:200], 0.8) and np.array_equal(a[200:], Y[200:])
        pm = corrupt(Y, idx, "point_mass", rng)
        assert abs(pm[:200].mean() - 1.5) < 1e-3 and pm[:200].std() < 0.01
        sy = corrupt(Y, idx, "symmetric", rng)
        assert abs((sy[:200] - Y[:200]).std() - 0.8) < 0.03
        print("  patches and contamination: ok", flush=True)
        # 5. Small end-to-end run on two data sets (parallel workers), then the analysis.
        args = parse_args(["--data", str(tmp), "--out", str(tmp / "out"), "--datasets",
                           "cifar10,mnist", "--workers", "2", "--n-train", "400", "--n-val", "200",
                           "--n-test", "300", "--epochs", "40", "--tune-epochs", "20",
                           "--warmup", "4", "--sigma-every", "2", "--eval-every", "2",
                           "--seeds", "3", "--lr-grid-gd", "0.05", "0.1",
                           "--lr-grid-sgd", "0.01", "0.02"])
        experiment(args)
        s = __import__("pandas").read_csv(tmp / "out" / "summary.csv")
        assert len(s) == 2 * 2 * 2 * 4 and s.n_pairs.min() == 3, s
        for f in ("runs.csv", "curves.csv", "sigma_weights.csv", "gain_by_condition.csv",
                  "table_images.tex"):
            assert (tmp / "out" / f).is_file(), f
        # Re-running uses the cache and changes nothing.
        before = sorted(p.name for p in (tmp / "out" / "cache").rglob("*.json.gz"))
        experiment(args)
        assert sorted(p.name for p in (tmp / "out" / "cache").rglob("*.json.gz")) == before
        # Paired design: BP and alpha share everything up to the end of the warm-up.
        d = make_data(args, "mnist", "asymmetric", 0)
        sched = schedule(args, args.epochs)
        fb, cb, _ = train(args, d, "SGD", "BP", 0.02, None, sched, 0, record=True)
        fa, ca, _ = train(args, d, "SGD", "alpha", 0.02, 0.1, sched, 0, record=True)
        k = cb["epoch"].index(sched["warmup"])
        assert cb["val_mse"][:k + 1] == ca["val_mse"][:k + 1]
        assert cb["val_mse"][k + 1] != ca["val_mse"][k + 1]
        print("  end-to-end run, cache and pairing: ok", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"self-test passed ({time.perf_counter() - t0:.0f}s)", flush=True)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", default="data", help="folder with raw/ (downloads) and prepared/")
    p.add_argument("--out", default="results_images")
    p.add_argument("--datasets", default=",".join(DATASETS),
                   help="comma-separated subset of " + ",".join(DATASETS))
    p.add_argument("--workers", type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    p.add_argument("--prepare", action="store_true", help="only convert the raw files")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="small smoke test: 2000 patches, 200 epochs, 3 seeds")
    p.add_argument("--reuse-across-code-changes", action="store_true",
                   help="keep cached runs after this file is edited")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--tune-seeds", type=int, default=2)
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--n-val", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=10000)
    p.add_argument("--contamination", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--floors", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--tune-epochs", type=int, default=500)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--sigma-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--lr-grid-gd", type=float, nargs="+",
                   default=[0.01, 0.02, 0.04, 0.08, 0.16, 0.32])
    p.add_argument("--lr-grid-sgd", type=float, nargs="+",
                   default=[0.01, 0.02, 0.04, 0.08, 0.16, 0.32])
    p.add_argument("--max-grid-extensions", type=int, default=3,
                   help="times a step-size grid may be extended when the choice is on its edge")
    p.add_argument("--conditions", default=",".join(CONDITIONS),
                   help="comma-separated subset of " + ",".join(CONDITIONS) + " (for checks)")
    a = p.parse_args(argv)
    a.datasets = [d.strip() for d in a.datasets.split(",") if d.strip()]
    bad = [d for d in a.datasets if d not in DATASETS]
    if bad:
        p.error(f"unknown data sets {bad}; choose from {DATASETS}")
    a.conditions = [c.strip() for c in a.conditions.split(",") if c.strip()]
    bad = [c for c in a.conditions if c not in CONDITIONS]
    if bad:
        p.error(f"unknown conditions {bad}; choose from {CONDITIONS}")
    if not 0 < a.alpha < 1:
        p.error("--alpha must be in (0, 1)")
    if a.quick:
        a.n_train, a.n_val, a.n_test, a.seeds = 2000, 500, 1000, 3
        a.epochs, a.tune_epochs, a.warmup, a.sigma_every, a.eval_every = 200, 50, 20, 2, 1
    a.hidden = list(a.hidden)
    return a


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return
    if args.prepare:
        for name in args.datasets:
            prepare_one(args.data, name)
        return
    if args.analyze_only:
        analyze(args, cache_dir(args))
        return
    experiment(args)


if __name__ == "__main__":
    main()
