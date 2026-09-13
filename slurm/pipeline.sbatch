#!/bin/bash
#SBATCH --job-name=eddy-pipeline
#SBATCH --account=gts-ldove6
#SBATCH --partition=cpu-small
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pipeline_%j.log

cd ~/projects/eddy-tracking
source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -m eddy_tracking.pipeline "$@"
