Image-data experiment for the ICAIS paper (new Section IV-F)
=============================================================
The alpha-loss versus BP on the five data sets of Rezende et al. (2014): MNIST, Frey faces,
small NORB, CIFAR-10 and SVHN, trained with GD and SGD only.

What it does
  Task      inpainting of 8x8 patches: the network predicts the central 4x4 square from the
            48 pixels around it (x3 for the colour data sets CIFAR-10 and SVHN); pixels in [0, 1].
  Outliers  20% of the training targets corrupted as in Section IV-D: asymmetric (+0.8),
            point mass (1.5) and symmetric (N(0, 0.8^2)); clean training data as the control.
            Validation and test targets are always clean.
  Network   48C -> 64 -> 64 -> 16C, tanh hidden units, W ~ N(0, 1/m).
  Training  GD (all 20,000 training patches per step) and SGD (mini-batches of 100),
            2,000 epochs each; alpha = 0.9 after a 200-epoch warm-up, sigma re-estimated
            every 20 epochs with a floor chosen from {0.05, 0.10, 0.20}.
  Protocol  step sizes (and the floor) chosen on 2 pilot runs per case; if a chosen step
            size is on the edge of its grid, the grid is extended automatically;
            best checkpoint on clean validation patches; test MSE on clean test patches;
            20 paired realizations; Wilcoxon signed-rank tests with Holm's correction.

Steps on Spartan
  1) Copy this folder to Spartan (from your own computer):
       scp -r alpha_images YOUR_USERNAME@spartan.hpc.unimelb.edu.au:~/alpha_project/
  2) Log in, then set up and test (about 1 minute; reuses ~/venvs/alpha-3.10.4):
       cd ~/alpha_project/alpha_images
       bash setup_images.sh
  3) Download the data on the LOGIN node (about 1.2 GB, a few minutes):
       bash get_data.sh
     (Little space in your home folder? Use the project folder for the data:
        bash get_data.sh /data/gpfs/projects/punim3024/alpha_data
      and then also start step 5 with DATA=/data/gpfs/projects/punim3024/alpha_data in front.)
  4) Optional 10-minute trial on the smallest data set (Frey faces):
       mkdir -p logs
       EXTRA="--quick --out results_quick" sbatch --array=1 --time=00:30:00 run_images.slurm
  5) Submit everything (5 array tasks of 32 cores, then the analysis):
       bash submit_images.sh
       squeue -u $USER
     Expected time: about 1 hour per data set (limit 4 hours).  If a task stops at the
     time limit, run   sbatch --array=N run_images.slurm   again for that data set
     (N = 0 mnist, 1 frey, 2 norb, 3 cifar10, 4 svhn): finished runs are kept.
  6) When the analysis job has finished, send me this file:
       results_images_pack.tar.gz
     (made by pack_results.sh; run   bash pack_results.sh   yourself if needed)

Outputs (results_images/)
  summary.csv            per data set x optimizer x condition: medians, ratio BP/alpha,
                         Wilcoxon p, Holm p, chosen step sizes and floor, sigma, weights
  gain_by_condition.csv  geometric mean of the ratio over the data sets
  table_images.tex       draft LaTeX table
  curves.csv             test MSE during training (median and quartiles over seeds)
  sigma_weights.csv      sigma and the weights of clean and corrupted targets during training
  runs.csv               every run
  ratios.png             quick-look figure

Files
  alpha_images.py        the program (python alpha_images.py --help)
  get_data.sh            downloads with checksums (MNIST, CIFAR-10, SVHN) and mirrors
  setup_images.sh        environment and self-test
  run_images.slurm       one array task per data set
  analyze_images.slurm   joint analysis
  submit_images.sh       submits both
  pack_results.sh        packs the results to send back
