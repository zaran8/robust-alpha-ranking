#!/usr/bin/env python3
"""Gradient vs exp(W)-scaled gradient, using alpha_rank_full_v2.py.

Place this file alongside src/alpha_rank_full_v2.py on Spartan. Outputs a
publication-quality 2x3 plot (GD and SGD) and a parameter-level diagnostic.

The diagnostic compares g and exp(W)*g AT THE SAME baseline parameters;
training curves compare separate runs with identical data and initialization.
The exp(W) factor applies to WEIGHT MATRICES only; biases remain unscaled.

Example (quick illustration):
  python src/plot_grad_expw.py --function Gaussian --condition asymmetric \
      --epochs 1000 --warmup 100 --sigma-every 20 --eval-every 10 \
      --out results/gradient_expw_demo
Paper-scale single-seed diagnostic:
  python src/plot_grad_expw.py --function Gaussian --condition asymmetric \
      --epochs 20000 --warmup 2000 --sigma-every 200 --eval-every 100 \
      --lr-gd 0.5 --lr-sgd 0.4 --out results/gradient_expw_paper
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import alpha_rank_full_v2 as core
except ImportError as exc:
    raise SystemExit(
        'Put plot_grad_expw.py in the same src/ directory as alpha_rank_full_v2.py. '
        'Run it as: python src/plot_grad_expw.py ...'
    ) from exc

# Colorblind-considerate, print-friendly palette, with contrasting line styles.
NAVY = '#163A63'    # ordinary gradient
CORAL = '#E76F51'   # exp(W)-scaled gradient
TEAL = '#0C9888'    # ratio / warmup
PURPLE = '#7651A8'
GREY = '#64748B'
GRID = '#E4EAF0'


def copy_params(theta):
    return [[W.copy(), b.copy()] for W, b in theta]


def weight_gradient_vectors(theta, grads):
    """Exclude biases from both norms: exp(W) acts only on weight matrices."""
    g = np.concatenate([dW.ravel() for dW, db in grads])
    multipliers = np.concatenate([np.exp(W).ravel() for W, b in theta])
    return g, g * multipliers


def update_parameters(theta, grads, lr, use_exp):
    """Full-gradient/mini-batch update. Bias updates are identical by design."""
    with np.errstate(over='raise', invalid='raise'):
        for (W, b), (dW, db) in zip(theta, grads):
            if use_exp:
                W -= lr * (np.exp(W) * dW)
            else:
                W -= lr * dW
            b -= lr * db
    return all(np.all(np.isfinite(p)) for layer in theta for p in layer)


def clean_test_mse(theta, data):
    pred = core.y_original(core.predict(theta, core.XG))
    return float(np.mean(np.square(pred - data['true'])))


def run_pair(function, condition, optimizer, args):
    """Two paths: ordinary alpha gradient and exp(W)-scaled alpha gradient.

    The two methods use paired samples, initialization, and SGD permutations.
    Sigma is independently estimated from each model's full training residuals.
    Gradient-norm comparison itself is evaluated on the BASELINE path only.
    """
    data_args = argparse.Namespace(n=args.n, n_val=args.n_val,
        noise=args.noise, contamination=args.contamination, val_mode='clean')
    data = core.make_data(function, condition, args.seed, data_args)
    vanilla = core.init_params(args.seed)
    expw = copy_params(vanilla)
    models = {'raw': vanilla, 'expw': expw}
    sigma = {'raw': None, 'expw': None}
    alive = {'raw': True, 'expw': True}
    batch_rng = {name: core._rng(30, 0, args.seed) for name in models}
    lr = args.lr_gd if optimizer == 'GD' else args.lr_sgd
    n = len(data['y'])
    records = []
    snapshots = None

    for epoch in range(args.epochs + 1):
        # Update the scale ONLY if there is another optimizer step to take.
        if args.warmup <= epoch < args.epochs:
            if (epoch - args.warmup) % args.sigma_every == 0:
                for name in models:
                    if alive[name]:
                        r = core.predict(models[name], data['X']) - data['y']
                        sigma[name] = core.scale_from_residuals(r, args.sigma_floor)

        a = args.alpha if epoch >= args.warmup else 1.0
        if epoch % args.eval_every == 0 or epoch in (args.warmup, args.epochs):
            # Critical: g and exp(W)*g are evaluated at the SAME theta.
            graw = gexp = None
            if alive['raw']:
                _, grads = core.loss_and_grad(
                    vanilla, data['X'], data['y'], a, sigma['raw'] or 1.0)
                graw, gexp = weight_gradient_vectors(vanilla, grads)
                snapshots = (np.concatenate([W.ravel() for W, _ in vanilla]),
                             graw.copy(), gexp.copy())
            records.append({
                'optimizer': optimizer, 'epoch': epoch,
                'loss_in_use': 'alpha' if epoch > args.warmup else 'MSE warmup',
                'raw_gradient_norm': float(np.linalg.norm(graw)) if graw is not None else np.nan,
                'expw_gradient_norm': float(np.linalg.norm(gexp)) if gexp is not None else np.nan,
                'gradient_norm_ratio': float(np.linalg.norm(gexp)/np.linalg.norm(graw))
                                       if graw is not None and np.linalg.norm(graw)>0 else np.nan,
                'mse_raw': clean_test_mse(vanilla, data) if alive['raw'] else np.nan,
                'mse_expw': clean_test_mse(expw, data) if alive['expw'] else np.nan,
                'sigma_raw': sigma['raw'] if sigma['raw'] is not None else np.nan,
                'sigma_expw': sigma['expw'] if sigma['expw'] is not None else np.nan,
            })
        if epoch == args.epochs:
            break

        step = n if optimizer == 'GD' else args.batch
        for name, theta in models.items():
            if not alive[name]:
                continue
            order = np.arange(n) if optimizer == 'GD' else batch_rng[name].permutation(n)
            for start in range(0, n, step):
                idx = order[start:start+step]
                _, grads = core.loss_and_grad(theta, data['X'][idx], data['y'][idx],
                                              a, sigma[name] or 1.0)
                try:
                    ok = update_parameters(theta, grads, lr,
                                           use_exp=(name == 'expw' and epoch >= args.warmup))
                except (FloatingPointError, OverflowError):
                    ok = False
                if not ok:
                    alive[name] = False
                    warnings.warn(f'{optimizer} {name} diverged at epoch {epoch}; '
                                  'later measurements will be omitted.', stacklevel=1)
                    break
    return pd.DataFrame(records), snapshots


def prettify(ax):
    ax.set_facecolor('#FFFFFF')
    ax.spines[['top', 'right']].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color('#C9D4DF')
    ax.grid(True, which='major', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors='#334155', labelsize=9)


def draw_main(df, args, folder):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titleweight': 'bold', 'savefig.facecolor': 'white'})
    fig, axs = plt.subplots(2, 3, figsize=(15.6, 8.1), constrained_layout=True)
    for row, optimizer in enumerate(('GD', 'SGD')):
        d = df[df['optimizer'] == optimizer].sort_values('epoch')
        if d.empty: continue
        x = d['epoch'].to_numpy()
        a, b, c = axs[row]
        # (1) Exactly the two requested gradient directions, same theta.
        a.plot(x, d['raw_gradient_norm'], color=NAVY, lw=2.3,
               label=r'$\|\nabla_W L\|_2$')
        a.plot(x, d['expw_gradient_norm'], color=CORAL, ls='--', lw=2.3,
               label=r'$\|e^W \odot \nabla_W L\|_2$')
        a.set_yscale('log')
        a.set_ylabel('Weight-gradient norm')
        a.set_title(f'{optimizer}  |  Raw vs. scaled gradient')
        a.legend(loc='best', fontsize=9, frameon=False)
        # (2) Amplification or suppression is measured, never assumed.
        b.plot(x, d['gradient_norm_ratio'], color=TEAL, lw=2.35)
        b.axhline(1.0, color=GREY, ls=':', lw=1.4)
        b.set_ylabel('Scaled / raw gradient norm')
        b.set_title(f'{optimizer}  |  Effect of $e^W$')
        # (3) Two training trajectories; not used to choose hyperparameters.
        c.plot(x, d['mse_raw'], color=NAVY, lw=2.3,
               label=r'$\nabla_\theta L_\alpha$ update')
        c.plot(x, d['mse_expw'], color=CORAL, ls='--', lw=2.3,
               label=r'$e^W \odot \nabla_W L_\alpha$ update')
        c.set_yscale('log')
        c.set_ylabel('Clean-target test MSE')
        c.set_title(f'{optimizer}  |  Training outcomes')
        c.legend(loc='best', fontsize=9, frameon=False)
        for ax in (a, b, c):
            ax.axvline(args.warmup, color=PURPLE, ls=':', lw=1.3, alpha=.8)
            ax.set_xlabel('Epoch')
            prettify(ax)
    fig.suptitle('Unscaled vs. $e^W$-scaled alpha-gradient  •  '
                 f'{args.function}, {args.condition}  •  alpha={args.alpha:g}',
                 fontsize=15, fontweight='bold', color=NAVY)
    fig.text(.5, -.01, 'Dashed coral: exp(W) scales weight gradients; biases are unchanged. '
             'Gradient diagnostics use the same parameters for both expressions. '
             'Purple vertical line: alpha training begins.',
             ha='center', fontsize=9, color=GREY)
    for ext in ('png', 'svg'):
        fig.savefig(folder/f'gradient_expw_comparison.{ext}', dpi=240, bbox_inches='tight')
    plt.close(fig)


def draw_coordinates(snap, args, folder):
    if snap is None: return
    w, g, ge = snap
    multipliers = np.exp(w)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    a.scatter(w, multipliers, s=24, color=TEAL, alpha=.55, edgecolor='none')
    wx = np.linspace(np.min(w), np.max(w), 250)
    a.plot(wx, np.exp(wx), color=NAVY, lw=1.7, label=r'$e^w$')
    a.axhline(1, color=GREY, ls=':', lw=1.2)
    a.set_xlabel('Weight value  $w_j$')
    a.set_ylabel('Multiplication factor  $e^{w_j}$')
    a.set_title('Positive weights amplify; negative weights attenuate')
    a.legend(frameon=False); prettify(a)
    b.scatter(g, ge, color=CORAL, alpha=.55, s=24, edgecolor='none')
    lim = max(float(np.max(np.abs(g))), float(np.max(np.abs(ge))), 1e-12)
    b.plot([-lim,lim], [-lim,lim], color=NAVY, ls='--', lw=1.5, label='No change ($y=x$)')
    b.set_xlim(-lim*1.06, lim*1.06); b.set_ylim(-lim*1.06, lim*1.06)
    b.set_xlabel(r'Raw coordinate  $[\nabla_W L]_j$')
    b.set_ylabel(r'Scaled coordinate  $e^{w_j}[\nabla_W L]_j$')
    b.set_title('Weight-gradient coordinates at the same parameters')
    b.legend(frameon=False); prettify(b)
    fig.suptitle(f'Parameter-wise interpretation  •  {args.function}, {args.condition}',
                 fontsize=13, fontweight='bold', color=NAVY)
    for ext in ('png','svg'):
        fig.savefig(folder/f'gradient_expw_coordinates.{ext}', dpi=240, bbox_inches='tight')
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--function', choices=core.FUNCTIONS, default='Gaussian')
    p.add_argument('--condition', choices=core.CONDITIONS, default='asymmetric')
    p.add_argument('--epochs', type=int, default=20000)
    p.add_argument('--warmup', type=int, default=2000)
    p.add_argument('--sigma-every', type=int, default=200)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--alpha', type=float, default=0.9)
    p.add_argument('--sigma-floor', type=float, default=0.1)
    p.add_argument('--lr-gd', type=float, default=0.5)
    p.add_argument('--lr-sgd', type=float, default=0.4)
    p.add_argument('--batch', type=int, default=20)
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--n-val', type=int, default=100)
    p.add_argument('--noise', type=float, default=0.025)
    p.add_argument('--contamination', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='results/gradient_expw_example')
    args = p.parse_args(argv)
    if not (0 < args.alpha <= 1 and args.sigma_floor > 0
            and 0 <= args.warmup < args.epochs and args.sigma_every > 0
            and args.eval_every > 0 and args.lr_gd > 0 and args.lr_sgd > 0
            and args.epochs > 0 and args.batch > 0):
        p.error('Check alpha, sigma floor, warmup, epoch count, and learning rates.')
    return args


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results=[]; snaps={}
    for optimizer in ('GD','SGD'):
        d, snap = run_pair(args.function, args.condition, optimizer, args)
        results.append(d); snaps[optimizer]=snap
        end=d.iloc[-1]
        print(f'{optimizer}: raw final test MSE={end.mse_raw:.6g}; '
              f'exp(W) final test MSE={end.mse_expw:.6g}; '
              f'gradient ratio at end={end.gradient_norm_ratio:.4g}', flush=True)
    df = pd.concat(results, ignore_index=True)
    df.to_csv(out/'gradient_expw_trajectories.csv', index=False)
    (out/'settings.json').write_text(json.dumps(vars(args), indent=2))
    draw_main(df, args, out)
    draw_coordinates(snaps['GD'], args, out)
    print('Saved PNG + SVG figures and CSV to', out.resolve(), flush=True)
    return df


if __name__ == '__main__':
    main()
