#!/bin/bash
#SBATCH --job-name=eddy-track
#SBATCH --account=gts-ldove6
#SBATCH --partition=cpu-small
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eddy_track_%j.log

# Eddy tracking - sequential graph-linking across all daily ID files.
# Fast (<1 min), but needs all eddy_id outputs to exist first.

cd ~/projects/eddy-tracking
source .venv/bin/activate
: "${EXPERIMENT:?EXPERIMENT env var is not set}"
python -u eddy_track.py "$EXPERIMENT"
