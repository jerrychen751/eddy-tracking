#!/bin/bash
#SBATCH --job-name=dl-cmems
#SBATCH --account=gts-ldove6
#SBATCH --partition=cpu-small
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/download_cmems_%j.log

# Download the daily Copernicus Marine plankton fields, one file per calendar month.

cd ~/projects/eddy-tracking
source .venv/bin/activate
: "${EXPERIMENT:?EXPERIMENT env var is not set}"
python -u -m eddy_tracking.downloads.cmems "$EXPERIMENT"
