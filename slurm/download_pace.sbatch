#!/bin/bash
#SBATCH --job-name=dl-pace
#SBATCH --account=gts-ldove6
#SBATCH --partition=cpu-small
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/download_pace_%j.log

# Download PACE OCI L3 daily Rrs via earthaccess S3 streaming.
# Slowest download — 270 individual files at ~13 MB each.
# Needs earthaccess credentials (~/.netrc or interactive login).

cd ~/projects/eddy-tracking
source .venv/bin/activate
: "${EXPERIMENT:?EXPERIMENT env var is not set}"
python -u download_pace.py "$EXPERIMENT"
