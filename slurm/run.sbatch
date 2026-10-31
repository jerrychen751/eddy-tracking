#!/bin/bash
#SBATCH --job-name=eddy-tracking
#SBATCH --account=gts-ldove6
#SBATCH --partition=cpu-small
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eddy_tracking_%j.log

# Runs the command after the script name inside the project venv, for example
# sbatch slurm/run.sbatch eddy-tracking identify-eddies --ssh ... or sbatch slurm/run.sbatch bash configs/<study>/run.sh
cd ~/projects/eddy-tracking
source .venv/bin/activate
export PYTHONUNBUFFERED=1
"$@"
