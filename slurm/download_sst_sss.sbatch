#!/bin/bash
#SBATCH --job-name=dl-sst-sss
#SBATCH --account=gts-ldove6
#SBATCH --partition=cpu-small
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/download_sst_sss_%j.log

# Download SST (AQUA MODIS via earthaccess) and SSS (SMAP via Harmony).
# Both are 8-day composites, so fewer files than daily products.

cd ~/projects/eddy-tracking
source .venv/bin/activate
: "${EXPERIMENT:?EXPERIMENT env var is not set}"
python -u download_sst_sss.py "$EXPERIMENT"
