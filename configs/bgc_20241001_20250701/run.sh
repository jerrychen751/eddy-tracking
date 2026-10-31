#!/bin/bash
# The SE BGC poster study (conferences/se_bgc.ipynb): Gulf Stream eddies in AVISO v2.0.1 SSH without the open-ocean mask, 2024-10-01 to 2025-07-01, with daily PACE Rrs and SDP pigments.
# Its daily PACE Rrs in bronze/pace_l3 came from an earlier download code whose config named no collection, so no download-pace line can reproduce it.
set -euo pipefail

D=data/bgc_20241001_20250701
LONLAT="--lon -81 -56 --lat 29 44"
DATES="--dates 2024-10-01 2025-07-01"

eddy-tracking download-aviso-ssh $LONLAT $DATES \
  --ftp-dir swot_products/l4_karin_nadir/v2_0_1/miost/science --no-mask-open-ocean --exclude-box -81 -75 40 44 --workers 2 \
  --out $D/bronze/swot_l4
eddy-tracking download-sst $LONLAT $DATES --collection-id C1615905770-OB_DAAC \
  --out $D/bronze/sst
eddy-tracking download-sss $LONLAT $DATES --collection-id C2208422957-POCLOUD --requests-workers 4 --download-chunk-size-mb 16 \
  --out $D/bronze/sss

eddy-tracking identify-eddies $LONLAT $DATES --ssh $D/bronze/swot_l4 \
  --bessel-km 500 --step-m 0.004 --shape-error 55 --min-amplitude-m 0.008 --workers 6 \
  --out $D/silver/eddy_id
eddy-tracking track-eddies --eddies $D/silver/eddy_id \
  --virtual-days 3 --max-speed-km-per-day 10 --min-track-days 60 \
  --out $D/silver/eddy_track
eddy-tracking collocate-pace --tracks $D/silver/eddy_track --pace $D/bronze/pace_l3 $DATES \
  --temporal-resolution DAY --min-coverage 0.8 --max-radius 2 --track-ids all \
  --out $D/silver/collocate_pace
eddy-tracking run-sdp --collocated $D/silver/collocate_pace --sst $D/bronze/sst --sss $D/bronze/sss --workers 1 \
  --out $D/silver/pigments
