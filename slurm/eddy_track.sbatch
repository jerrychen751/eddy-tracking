#!/bin/bash
#SBATCH --job-name=eddy-track
#SBATCH --account=math
#SBATCH --partition=ice-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eddy_track_%j.log

# Eddy tracking — sequential graph-linking across all daily ID files.
# Fast (<1 min), but needs all eddy_id outputs to exist first.

source /usr/local/pace-apps/manual/packages/anaconda3/2022.05.0.1/etc/profile.d/conda.sh
conda activate eddy
cd ~/projects/eddy-tracking
: "${EXPERIMENT:?EXPERIMENT env var is not set}"
python -u eddy_track.py "$EXPERIMENT"
