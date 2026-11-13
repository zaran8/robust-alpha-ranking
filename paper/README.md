# Scripts behind the paper's figures, tables and numbers

Run every command from the repository root. The scripts import `src/alpha_rank_full_v2.py`
unchanged. `results/` is the results folder of the Spartan runs (made by
`scripts/run_alpha.slurm` and `scripts/submit_retrain.sh`; not stored on GitHub), with the
sub-folders `sigma_floor005`, `sigma_floor010`, `sigma_floor020`, `sigma_valcont`,
`ranking_v3_periodic_floor0.05`, `ranking_v3strict_periodic_floor0.05` and `sigma_choice.json`.

| Paper | Scripts | Needs |
|---|---|---|
| Table II and all numbers of Section IV-D | `section4d_numbers.py` | `results/` |
| Fig. 5 | `fig5_data.py`, `fig5_plot.py` | `results/sigma_floor005`, or `data/fig5_curves.csv` |
| Section IV-D, sigma fixed at the floor | `sigma_fixed_run.py`, `sigma_fixed_analyze.py` | `results/` |
| Table III and Section IV-E | `expw_run.py`, `expw_analyze.py`, `expw_verify.py` | `results/ranking_v3_periodic_floor0.05` |
| Fig. 6 | `fig6_data.py`, `fig6_plot.py` | `data/expw_tune`, or `data/fig6_trajectories.csv` |
| Fig. 7 | `../image_experiment/plot_images.py` | `results_images/` of the image experiment, or `data/fig7` |

The lower bound of sigma and the schedule are selected by `src/choose_sigma.py`.
`sigma_fixed_lr_check.py` and `sigma_fixed_lr_check_periodic.py` repeat the two GD cases in
which sigma fixed at the floor stalls, at other step sizes.

## Commands

```bash
# Section IV-D: Table II and the numbers in the text
python paper/section4d_numbers.py results out/section4d

# Fig. 5 (fig_gdsgd_schedules.pdf; fig_gdsgd_types.pdf is the same per type of contamination)
python paper/fig5_data.py results/sigma_floor005 out/fig5_curves.csv
python paper/fig5_plot.py out/fig5_curves.csv --out out/figures

# Section IV-D: sigma fixed at the floor (tuning + 480 training runs)
python paper/sigma_fixed_run.py out/sigma_fixed --workers 8
python paper/sigma_fixed_analyze.py results out/sigma_fixed out/sigma_fixed_analysis

# Section IV-E and Table III: gradient scaled by exp(w) (tuning + 480 training runs)
python paper/expw_run.py out/expw --workers 8
python paper/expw_analyze.py results/ranking_v3_periodic_floor0.05 out/expw out/expw_analysis
python paper/expw_verify.py results/ranking_v3_periodic_floor0.05 out/expw

# Fig. 6 (4 training runs, under a minute)
python paper/fig6_data.py out/fig6_trajectories.csv      # --expw-tune out/expw/tune to use your own tuning
python paper/fig6_plot.py out/fig6_trajectories.csv out/figures/fig_expw

# Fig. 7 (fig_images_split.pdf)
python image_experiment/plot_images.py --results paper/data/fig7 --out out/figures
```

Without `results/`, the plots can be drawn from the files in `data/`:
`python paper/fig5_plot.py --out out/figures` and `python paper/fig6_plot.py`.

`sigma_fixed_run.py` and `expw_run.py` take about 40 CPU-minutes each. They keep every
finished job, so they can be stopped and started again. On Spartan, run them on a compute node
with the Python environment of `scripts/run_alpha.slurm`, for example:

```bash
sbatch --account=YOUR_PROJECT --cpus-per-task=8 --mem=8G --time=02:00:00 \
       --wrap "python paper/expw_run.py out/expw"
```

Requirements: numpy, pandas, scipy, matplotlib.

Checked: from the Spartan results, these scripts reproduce the paper's numbers, Table III and
Figs. 5-7 exactly, and the run scripts reproduce existing training runs exactly.
