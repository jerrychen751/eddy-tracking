#!/bin/bash
# Sensitivity run of the canonical Gulf Stream study with a 1 cm minimum eddy amplitude instead of 0.8 cm, everything else equal, over the SSH record as it stood before the 2026 Copernicus extension (to 2025-12-31).
# It reads the canonical study's bronze data and writes only its own silver and gold.
set -euo pipefail

B=data/gulf_stream_20240305_20260531/bronze
D=data/gulf_stream_20240305_20260531_amplitude_1cm
LONLAT="--lon -81 -56 --lat 29 44"
EDDY_DATES="--dates 2023-03-28 2025-12-31"
RRS_DATES="--dates 2024-03-05 2025-12-31"

eddy-tracking identify-eddies $LONLAT $EDDY_DATES --ssh $B/swot_l4_open_ocean \
  --bessel-km 500 --step-m 0.004 --shape-error 55 --min-amplitude-m 0.01 --workers 6 \
  --out $D/silver/eddy_id
eddy-tracking track-eddies --eddies $D/silver/eddy_id \
  --virtual-days 5 --max-speed-km-per-day 10 --min-track-days 30 \
  --out $D/silver/eddy_track
eddy-tracking trace-axis --ssh $B/swot_l4_open_ocean --tracks $D/silver/eddy_track \
  --adt-levels 0.10 0.60 \
  --out $D/silver/gulf_stream
eddy-tracking collocate-pace --tracks $D/silver/eddy_track --pace $B/pace_l3_8d $RRS_DATES \
  --temporal-resolution 8D --min-coverage 0.8 --max-radius 2 --track-ids all \
  --out $D/silver/collocate_pace
eddy-tracking run-sdp --collocated $D/silver/collocate_pace --sst $B/sst --sss $B/sss --workers 4 \
  --out $D/silver/pigments
eddy-tracking collocate-plankton --plankton $B/plankton --tracks $D/silver/eddy_track --axis $D/silver/gulf_stream $EDDY_DATES \
  --min-coverage 0.5 --min-pixels 10 \
  --out $D/gold/eddy_plankton_table.parquet
eddy-tracking measure-dynamics --ssh $B/swot_l4_open_ocean --tracks $D/silver/eddy_track \
  --out $D/silver/eddy_dynamics
eddy-tracking sample-background --ssh $B/swot_l4_open_ocean --pace $B/pace_l3_8d --sst $B/sst --sss $B/sss \
  --tracks $D/silver/eddy_track --temporal-resolution 8D --subsample 2000 \
  --out $D/silver/pigments/background
eddy-tracking build-gold-table --pigments $D/silver/pigments --tracks $D/silver/eddy_track --dynamics $D/silver/eddy_dynamics \
  --axis $D/silver/gulf_stream --background $D/silver/pigments/background $EDDY_DATES \
  --out $D/gold/eddy_pigment_table.parquet
