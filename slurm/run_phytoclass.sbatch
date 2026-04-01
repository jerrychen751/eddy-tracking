#!/bin/bash
#SBATCH --job-name=run-phytoclass
#SBATCH --account=math
#SBATCH --partition=ice-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=logs/run_phytoclass_%j.log

# Phytoclass PFT decomposition — processes eddies in parallel via
# ProcessPoolExecutor (12 workers). Each eddy runs SA + NNLS
# independently. Largest eddies (~35k pixels) take ~4h each;
# 12-way parallelism brings wall time to ~5h.

source /usr/local/pace-apps/manual/packages/anaconda3/2022.05.0.1/etc/profile.d/conda.sh
conda activate eddy
cd ~/projects/eddy-tracking
: "${EXPERIMENT:?EXPERIMENT env var is not set}"
python -u run_phytoclass.py "$EXPERIMENT"
