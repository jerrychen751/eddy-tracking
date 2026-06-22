#!/bin/bash
#SBATCH --job-name=eddy-id
#SBATCH --account=gts-ldove6
#SBATCH --partition=cpu-small
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/eddy_id_%j.log

# Eddy identification — CPU-bound, uses ProcessPoolExecutor internally.
# 270 daily SSH files processed in parallel across cpus-per-task cores.
# ~10 minutes with 6 workers on the login node; should be faster here.

cd ~/projects/eddy-tracking
source .venv/bin/activate
: "${EXPERIMENT:?EXPERIMENT env var is not set}"
python -u eddy_id.py "$EXPERIMENT"
