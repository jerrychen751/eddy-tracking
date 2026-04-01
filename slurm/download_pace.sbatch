#!/bin/bash
#SBATCH --job-name=dl-pace
#SBATCH --account=math
#SBATCH --partition=ice-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/download_pace_%j.log

# Download PACE OCI L3 daily Rrs via earthaccess S3 streaming.
# Slowest download — 270 individual files at ~13 MB each.
# Needs earthaccess credentials (~/.netrc or interactive login).

source /usr/local/pace-apps/manual/packages/anaconda3/2022.05.0.1/etc/profile.d/conda.sh
conda activate eddy
cd ~/projects/eddy-tracking
: "${EXPERIMENT:?EXPERIMENT env var is not set}"
python -u download_pace.py "$EXPERIMENT"
