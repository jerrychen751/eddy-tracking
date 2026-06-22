#!/bin/bash
#SBATCH --job-name=dl-swot
#SBATCH --account=gts-ldove6
#SBATCH --partition=cpu-small
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/download_swot_%j.log

# Download SWOT L4 SSH from AVISO FTP — I/O-bound, uses ThreadPoolExecutor.
# Needs .env with FTP_HOST, FTP_USER, FTP_PASSWORD in the project root.

cd ~/projects/eddy-tracking
source .venv/bin/activate
: "${EXPERIMENT:?EXPERIMENT env var is not set}"
python -u download_swot.py "$EXPERIMENT"
