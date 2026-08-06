#!/bin/bash
#SBATCH --job-name=run-phytoclass
#SBATCH --account=gts-ldove6
#SBATCH --partition=cpu-small
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=logs/run_phytoclass_%j.log

# Phytoclass PFT decomposition - processes eddies in parallel via
# ProcessPoolExecutor (12 workers). Each eddy runs SA + NNLS
# independently. Largest eddies (~35k pixels) take ~4h each;
# 12-way parallelism brings wall time to ~5h.

cd ~/projects/eddy-tracking
source .venv/bin/activate
: "${EXPERIMENT:?EXPERIMENT env var is not set}"
python -u run_phytoclass.py "$EXPERIMENT"
