#!/bin/bash
#SBATCH --job-name=dl-sst-sss
#SBATCH --account=math
#SBATCH --partition=ice-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/download_sst_sss_%j.log

# Download SST (AQUA MODIS via earthaccess) and SSS (SMAP via Harmony).
# Both are 8-day composites, so fewer files than daily products.

source /usr/local/pace-apps/manual/packages/anaconda3/2022.05.0.1/etc/profile.d/conda.sh
conda activate eddy
cd ~/projects/eddy-tracking
: "${EXPERIMENT:?EXPERIMENT env var is not set}"
python -u download_sst_sss.py "$EXPERIMENT"
