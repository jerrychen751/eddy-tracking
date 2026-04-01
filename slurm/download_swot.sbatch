#!/bin/bash
#SBATCH --job-name=dl-swot
#SBATCH --account=math
#SBATCH --partition=ice-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/download_swot_%j.log

# Download SWOT L4 SSH from AVISO FTP — I/O-bound, uses ThreadPoolExecutor.
# Needs .env with FTP_HOST, FTP_USER, FTP_PASSWORD in the project root.

source /usr/local/pace-apps/manual/packages/anaconda3/2022.05.0.1/etc/profile.d/conda.sh
conda activate eddy
cd ~/projects/eddy-tracking
: "${EXPERIMENT:?EXPERIMENT env var is not set}"
python -u download_swot.py "$EXPERIMENT"
