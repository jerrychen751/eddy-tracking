#!/bin/bash
#SBATCH --job-name=collocate-pace
#SBATCH --account=gts-ldove6
#SBATCH --partition=cpu-small
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/collocate_pace_%j.log

# PACE collocation — iterates daily L3 files, extracts Rrs pixels inside
# eddy contours. Memory-bound (loads full L3 grid per file), not CPU-bound.

cd ~/projects/eddy-tracking
source .venv/bin/activate
: "${EXPERIMENT:?EXPERIMENT env var is not set}"
python -u collocate_pace.py "$EXPERIMENT"
