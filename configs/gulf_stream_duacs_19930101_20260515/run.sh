#!/bin/bash
# Gulf Stream eddies in the reprocessed nadir-altimetry DUACS maps, 1993-01-01 to 2026-05-15 (the last reprocessed map), matched to every Argo profile in the box.
# Detection, tracking and axis settings equal the canonical study's (configs/gulf_stream_20240305_20260531/run.sh), so the two eddy sets compare directly over 2023-03-28 to 2025-12-31.
# The maps keep their own grid (cell centers half a cell off the AVISO grid) and get their own open-ocean mask, with the Great Lakes box excluded as in the canonical study.
# Run from the repository root, on Phoenix: sbatch --qos=embers --cpus-per-task=24 --mem=64G --time=08:00:00 slurm/run.sbatch bash configs/gulf_stream_duacs_19930101_20260515/run.sh
# embers stops a job at 8 hours; submit it again and the SSH download and identify-eddies skip the days they already wrote, download-argo resumes at its first missing year, and track-eddies (about 2 hours per polarity here), trace-axis and match-argo redo their whole output.
set -euo pipefail

D=data/gulf_stream_duacs_19930101_20260515
REGION="--lon -81 -56 --lat 29 44 --dates 1993-01-01 2026-05-15"

eddy-tracking download-cmems-ssh $REGION \
  --dataset cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D --version 202411 \
  --exclude-box -81 -75 40 44 --store arco-geo-series --block-days 31 \
  --out $D/bronze/ssh
eddy-tracking identify-eddies $REGION --ssh $D/bronze/ssh \
  --bessel-km 500 --step-m 0.004 --shape-error 55 --min-amplitude-m 0.008 --workers 24 \
  --out $D/silver/eddy_id
eddy-tracking track-eddies --eddies $D/silver/eddy_id \
  --virtual-days 5 --max-speed-km-per-day 10 --min-track-days 30 \
  --out $D/silver/eddy_track
eddy-tracking trace-axis --ssh $D/bronze/ssh --tracks $D/silver/eddy_track \
  --adt-levels 0.10 0.60 \
  --out $D/silver/gulf_stream
eddy-tracking download-argo $REGION --profiles core --chunk-days 90 \
  --out $D/bronze/argo
eddy-tracking match-argo --tracks $D/silver/eddy_track --argo $D/bronze/argo/core_profiles.nc \
  --axis $D/silver/gulf_stream --max-radius-fraction 1.5 \
  --out $D/gold/argo_matchups.parquet
