#!/bin/bash
#SBATCH --job-name=run-sdp
#SBATCH --account=gts-ldove6
#SBATCH --partition=cpu-small
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/run_sdp_%j.log

# SDP pigment inversion - slowest stage. Per-eddy iteration with
# 100-ensemble GSM inversion per pixel. Currently sequential across
# eddies; each eddy takes ~30s-1min depending on pixel count.
#
# Future optimization: could be converted to a Slurm array job
# (--array=0-N) where each task processes one eddy independently.

cd ~/projects/eddy-tracking
source .venv/bin/activate
: "${EXPERIMENT:?EXPERIMENT env var is not set}"
python -u run_sdp.py "$EXPERIMENT"
