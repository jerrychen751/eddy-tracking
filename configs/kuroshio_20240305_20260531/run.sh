#!/bin/bash
# Kuroshio Extension copy of the canonical Gulf Stream study: its own box, the Sea of Japan excluded instead of the Great Lakes, and the jet searched at ADT 0.60 to 1.20 m. SSH is AVISO only, to 2025-12-31.
# Run from the repository root. PACE and SST downloads must run on the laptop; the other steps run on Phoenix through slurm/run.sbatch.
set -euo pipefail

D=data/kuroshio_20240305_20260531
LONLAT="--lon 135 160 --lat 29 44"
EDDY_DATES="--dates 2023-03-28 2025-12-31"
RRS_DATES="--dates 2024-03-05 2026-05-31"
COLLOCATION_DATES="--dates 2024-03-05 2025-12-31"

eddy-tracking download-aviso-ssh $LONLAT $EDDY_DATES \
  --ftp-dir swot_products/l4_karin_nadir/v3_0/miost --mask-open-ocean --exclude-box 135 140.5 35.5 44 --workers 8 \
  --out $D/bronze/swot_l4_open_ocean
eddy-tracking download-pace $LONLAT $RRS_DATES --product AOP --collection-id C4184125822-OB_CLOUD --version V3_2 --temporal-resolution 8D \
  --out $D/bronze/pace_l3_8d
eddy-tracking download-pace $LONLAT $RRS_DATES --product BGC --collection-id C4184125847-OB_CLOUD --version V3_2 --temporal-resolution 8D \
  --out $D/bronze/pace_l3_8d_bgc
eddy-tracking download-sst $LONLAT $RRS_DATES --collection-id C1615905770-OB_DAAC \
  --out $D/bronze/sst
eddy-tracking download-sss $LONLAT $RRS_DATES --collection-id C2208422957-POCLOUD --requests-workers 4 --download-chunk-size-mb 16 \
  --out $D/bronze/sss
eddy-tracking download-plankton $LONLAT $EDDY_DATES \
  --dataset-id cmems_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D --dataset-version 202603 \
  --out $D/bronze/plankton

eddy-tracking identify-eddies $LONLAT $EDDY_DATES --ssh $D/bronze/swot_l4_open_ocean \
  --bessel-km 500 --step-m 0.004 --shape-error 55 --min-amplitude-m 0.008 --workers 6 \
  --out $D/silver/eddy_id
eddy-tracking track-eddies --eddies $D/silver/eddy_id \
  --virtual-days 5 --max-speed-km-per-day 10 --min-track-days 30 \
  --out $D/silver/eddy_track
eddy-tracking trace-axis --ssh $D/bronze/swot_l4_open_ocean --tracks $D/silver/eddy_track \
  --adt-levels 0.60 1.20 \
  --out $D/silver/gulf_stream
eddy-tracking collocate-pace --tracks $D/silver/eddy_track --pace $D/bronze/pace_l3_8d $COLLOCATION_DATES \
  --temporal-resolution 8D --min-coverage 0.8 --max-radius 2 --track-ids all \
  --out $D/silver/collocate_pace
eddy-tracking run-sdp --collocated $D/silver/collocate_pace --sst $D/bronze/sst --sss $D/bronze/sss --workers 4 \
  --out $D/silver/pigments
eddy-tracking collocate-plankton --plankton $D/bronze/plankton --tracks $D/silver/eddy_track --axis $D/silver/gulf_stream $EDDY_DATES \
  --min-coverage 0.5 --min-pixels 10 \
  --out $D/gold/eddy_plankton_table.parquet
eddy-tracking measure-dynamics --ssh $D/bronze/swot_l4_open_ocean --tracks $D/silver/eddy_track \
  --out $D/silver/eddy_dynamics
eddy-tracking sample-background --ssh $D/bronze/swot_l4_open_ocean --pace $D/bronze/pace_l3_8d --sst $D/bronze/sst --sss $D/bronze/sss \
  --tracks $D/silver/eddy_track --temporal-resolution 8D --subsample 2000 \
  --out $D/silver/pigments/background
eddy-tracking build-gold-table --pigments $D/silver/pigments --tracks $D/silver/eddy_track --dynamics $D/silver/eddy_dynamics \
  --axis $D/silver/gulf_stream --background $D/silver/pigments/background $EDDY_DATES \
  --out $D/gold/eddy_pigment_table.parquet
