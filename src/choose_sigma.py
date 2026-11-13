#!/usr/bin/env python3
"""choose_sigma.py -- choose the Stage 2 residual-scale setting (sigma strategy
and sigma floor) from Stage 1 VALIDATION error only. Test error is never read:
the results file is loaded without any test_* column.

Usage
  python choose_sigma.py --out-dir RESULTS_DIR STAGE1_FOLDER [STAGE1_FOLDER ...]

Each STAGE1_FOLDER is the --out folder of one finished Stage 1 run (one sigma
floor; the strategies early / late / periodic inside it). All folders must come
from the same program version with identical settings except --sigma-floor.

Pre-declared rule
  candidate  = (sigma strategy, sigma floor)
  per case   = median over seeds of val_post_alpha (the validation score of the
               best checkpoint after at least one alpha update, which is the
               checkpoint the paper reports), for "GD Alpha[s]" and "SGD Alpha[s]"
  score      = geometric mean of those medians over 2 optimisers x 12 cases
  choice     = lowest score; ties go to the larger floor, then early < late < periodic
  A candidate with a missing or diverged run is excluded.

Writes RESULTS_DIR/sigma_choice.json (rule, every candidate, choice) and
RESULTS_DIR/sigma_choice.env (SIGMA_STRATEGY=..., SIGMA_FLOOR=...) for Slurm.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

STRATEGIES = ("early", "late", "periodic")
OPTIMISERS = ("GD", "SGD")
N_CASES = 12                                   # 3 functions x 4 conditions
# Keys of config.json allowed to differ between candidate folders.
MAY_DIFFER = {"sigma_floor", "digest", "cache", "completed_eval_jobs", "expected_eval_jobs"}
# The only columns read from individual_results.csv: no test_* column is loaded.
COLUMNS = ["function", "condition", "seed", "method", "sigma_strategy", "val_post_alpha",
           "sigma_final", "selected_in_warmup", "diverged"]
RULE = ("score = geometric mean, over GD and SGD and all 12 cases, of the median over "
        "seeds of val_post_alpha; lowest score wins; ties -> larger floor, then "
        "early < late < periodic; candidates with missing or diverged runs are excluded")


def gmean(values):
    v = np.asarray(values, dtype=float)
    return float(np.exp(np.mean(np.log(v))))


def load(folder):
    folder = Path(folder)
    cfg_path = folder / "config.json"
    if not cfg_path.is_file():
        raise SystemExit(f"{folder}: no config.json; is this a Stage 1 --out folder?")
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("stage") != "sigma":
        raise SystemExit(f"{folder}: stage is {cfg.get('stage')!r}, expected 'sigma'")
    if cfg.get("completed_eval_jobs") != cfg.get("expected_eval_jobs"):
        raise SystemExit(f"{folder}: incomplete ({cfg.get('completed_eval_jobs')}/"
                         f"{cfg.get('expected_eval_jobs')} evaluation jobs); finish that run first")
    results = pd.read_csv(folder / "individual_results.csv", usecols=COLUMNS)
    tuning = pd.read_csv(folder / "tuning.csv")
    return folder, cfg, results, tuning


def check_comparable(loaded):
    ref_folder, ref_cfg = loaded[0][0], loaded[0][1]
    for folder, cfg, _, _ in loaded[1:]:
        diff = sorted(k for k in set(ref_cfg) | set(cfg)
                      if k not in MAY_DIFFER and ref_cfg.get(k) != cfg.get(k))
        if diff:
            raise SystemExit(f"{folder} differs from {ref_folder} in more than the sigma floor: "
                             + ", ".join(diff))
    floors = [cfg["sigma_floor"] for _, cfg, _, _ in loaded]
    if len(set(floors)) != len(floors):
        raise SystemExit(f"two folders share the same sigma floor: {floors}")


def score_candidates(loaded):
    rows = []
    for folder, cfg, res, tun in loaded:
        floor = float(cfg["sigma_floor"])
        for s in cfg["sigma_strategies"]:
            names = [f"{o} Alpha[{s}]" for o in OPTIMISERS]
            d = res[res.method.isin(names)]
            expected_runs = len(OPTIMISERS) * N_CASES * int(cfg["seeds"])
            bad = int(d.val_post_alpha.isna().sum() + d.diverged.astype(bool).sum())
            med = d.groupby(["method", "function", "condition"]).val_post_alpha.median()
            ok = len(d) == expected_runs and bad == 0 and len(med) == len(OPTIMISERS) * N_CASES
            cont = med[med.index.get_level_values("condition") != "clean"]
            per_opt = {o: med[med.index.get_level_values("method") == f"{o} Alpha[{s}]"]
                       for o in OPTIMISERS}
            chosen = tun[tun.method.isin(names) & tun.chosen.astype(bool)].drop_duplicates(
                ["function", "condition", "method"])
            rows.append(dict(
                folder=str(folder), sigma_floor=floor, strategy=s,
                score=gmean(med) if ok else math.inf,
                score_contaminated=gmean(cont) if ok else math.inf,
                score_gd=gmean(per_opt["GD"]) if ok else math.inf,
                score_sgd=gmean(per_opt["SGD"]) if ok else math.inf,
                runs=int(len(d)), expected_runs=expected_runs, missing_or_diverged=bad,
                share_sigma_at_floor=float(np.isclose(d.sigma_final, floor).mean()),
                median_sigma_final=float(d.sigma_final.median()),
                share_best_in_warmup=float(d.selected_in_warmup.astype(bool).mean()),
                lr_choices=int(len(chosen)),
                lr_at_grid_edge=int(chosen.chosen_at_grid_edge.astype(bool).sum()),
                excluded=not ok))
    table = pd.DataFrame(rows)
    table["_strategy_order"] = table.strategy.map({s: i for i, s in enumerate(STRATEGIES)})
    table = table.sort_values(["score", "sigma_floor", "_strategy_order"],
                              ascending=[True, False, True]).drop(columns="_strategy_order")
    return table.reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", help="Stage 1 --out folders, one per sigma floor")
    ap.add_argument("--out-dir", required=True, help="where sigma_choice.json/.env are written")
    args = ap.parse_args(argv)

    loaded = [load(f) for f in args.folders]
    check_comparable(loaded)
    table = score_candidates(loaded)
    valid = table[~table.excluded]
    if valid.empty:
        raise SystemExit("every candidate has missing or diverged runs; nothing to choose")
    best = valid.iloc[0]
    ref_cfg = loaded[0][1]

    pd.set_option("display.width", 200)
    show = table[["strategy", "sigma_floor", "score", "score_contaminated", "score_gd", "score_sgd",
                  "share_sigma_at_floor", "median_sigma_final", "share_best_in_warmup",
                  "lr_at_grid_edge", "lr_choices", "excluded"]]
    print("Candidates, best first (scores = validation error; test error is never read):")
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nRule: {RULE}")
    print(f"Validation criterion: {ref_cfg.get('validation_criterion')}")
    print(f"\nCHOICE: sigma strategy '{best.strategy}', sigma floor {best.sigma_floor:g} "
          f"(score {best.score:.4g}, from {best.folder})")
    edges = table[table.lr_at_grid_edge > 0]
    for _, r in edges.iterrows():
        print(f"WARNING: {r.lr_at_grid_edge} of {r.lr_choices} learning-rate choices for "
              f"'{r.strategy}' at floor {r.sigma_floor:g} sit at the edge of the grid; "
              "consider widening it before trusting these numbers.")
    if best.share_sigma_at_floor > 0.5:
        print(f"NOTE: in {best.share_sigma_at_floor:.0%} of the chosen setting's runs sigma ends "
              "at the floor, so the floor is part of the method: report it.")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    record = dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        rule=RULE,
        validation_criterion=ref_cfg.get("validation_criterion"),
        test_error_used=False,
        choice=dict(strategy=best.strategy, sigma_floor=float(best.sigma_floor),
                    score=float(best.score), folder=best.folder),
        shared_settings={k: v for k, v in ref_cfg.items() if k not in MAY_DIFFER},
        candidates=json.loads(table.to_json(orient="records")))
    (out / "sigma_choice.json").write_text(json.dumps(record, indent=2))
    (out / "sigma_choice.env").write_text(
        f"SIGMA_STRATEGY={best.strategy}\nSIGMA_FLOOR={best.sigma_floor:g}\n")
    print(f"\nWrote {out / 'sigma_choice.json'} and {out / 'sigma_choice.env'}")


if __name__ == "__main__":
    main()
