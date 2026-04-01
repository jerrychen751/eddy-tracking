#!/bin/bash
#SBATCH --job-name=collocate-pace
#SBATCH --account=math
#SBATCH --partition=ice-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/collocate_pace_%j.log

# PACE collocation — iterates daily L3 files, extracts Rrs pixels inside
# eddy contours. Memory-bound (loads full L3 grid per file), not CPU-bound.

source /usr/local/pace-apps/manual/packages/anaconda3/2022.05.0.1/etc/profile.d/conda.sh
conda activate eddy
cd ~/projects/eddy-tracking
: "${EXPERIMENT:?EXPERIMENT env var is not set}"
python -u collocate_pace.py "$EXPERIMENT"
