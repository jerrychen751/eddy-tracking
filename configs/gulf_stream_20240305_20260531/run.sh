#!/bin/bash
# Gulf Stream eddies in SWOT-era SSH, 2023-03-28 to 2026-10-10, with PACE pigments, Copernicus plankton, and the Argo, glider, GLORYS and WOA context. The name carries the PACE Rrs period.
# SSH comes from AVISO's experimental DUACS-MIOST maps with SWOT through 2025-12-31, then from the Copernicus near-real-time nadir plus SWOT maps, interpolated onto the AVISO grid and given the AVISO open-ocean label (over the 62 shared days the two differ by 1.6 cm RMS in SLA, less than AVISO changes from one day to the next).
# The open-ocean mask keeps cells 8 or more cells from invalid data and outside the Great Lakes box; eddy boundaries are PET's speed contour.
# Run from the repository root. PACE and SST downloads must run on the laptop, because Phoenix cannot reach oceandata.sci.gsfc.nasa.gov; the other steps run on Phoenix through slurm/run.sbatch.
set -euo pipefail

D=data/gulf_stream_20240305_20260531
LONLAT="--lon -81 -56 --lat 29 44"
EDDY_DATES="--dates 2023-03-28 2026-10-10"
# The PACE Rrs period; collocation keeps eddy-days to 2026-05-31
RRS_DATES="--dates 2024-03-05 2026-05-31"
# The analysis period, also in analysis.yaml
ANALYSIS_DATES="--dates 2024-03-05 2026-06-01"

eddy-tracking download-aviso-ssh $LONLAT --dates 2023-03-28 2025-12-31 \
  --ftp-dir swot_products/l4_karin_nadir/v3_0/miost --mask-open-ocean --exclude-box -81 -75 40 44 --workers 8 \
  --out $D/bronze/swot_l4_open_ocean
eddy-tracking download-cmems-ssh $LONLAT --dates 2026-01-01 2026-10-10 \
  --dataset cmems_obs-sl_glo_phy-ssh_nrt_demo-allsat-swos-l4-duacs-0.125deg_P1D-i --version 202511 \
  --grid-from $D/bronze/swot_l4_open_ocean \
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
eddy-tracking download-glorys $LONLAT $ANALYSIS_DATES \
  --dataset-id cmems_mod_glo_phy_my_0.083deg_P1D-m --dataset-version 202311 \
  --surface-variables mlotst zos bottomT --current-variables uo vo --profile-variables thetao so --max-depth-m 1000 \
  --out $D/bronze/glorys
eddy-tracking download-argo $LONLAT $ANALYSIS_DATES --profiles core bgc --chunk-days 30 \
  --out $D/bronze/argo
eddy-tracking download-insitu $LONLAT $ANALYSIS_DATES --platform-types GL CT XB --min-depth-m 200 --thin-hours 3 --max-levels 600 \
  --out $D/bronze/insitu
eddy-tracking download-woa $LONLAT --opendap-root https://www.ncei.noaa.gov/thredds-ocean/dodsC/woa23/DATA --period decav91C0 \
  --out $D/bronze/woa23

eddy-tracking identify-eddies $LONLAT $EDDY_DATES --ssh $D/bronze/swot_l4_open_ocean \
  --bessel-km 500 --step-m 0.004 --shape-error 55 --min-amplitude-m 0.008 --workers 6 \
  --out $D/silver/eddy_id
eddy-tracking track-eddies --eddies $D/silver/eddy_id \
  --virtual-days 5 --max-speed-km-per-day 10 --min-track-days 30 \
  --out $D/silver/eddy_track
eddy-tracking trace-axis --ssh $D/bronze/swot_l4_open_ocean --tracks $D/silver/eddy_track \
  --adt-levels 0.10 0.60 \
  --out $D/silver/gulf_stream
eddy-tracking collocate-pace --tracks $D/silver/eddy_track --pace $D/bronze/pace_l3_8d $RRS_DATES \
  --temporal-resolution 8D --min-coverage 0.8 --max-radius 2 --track-ids all \
  --out $D/silver/collocate_pace
eddy-tracking run-sdp --collocated $D/silver/collocate_pace --sst $D/bronze/sst --sss $D/bronze/sss --workers 4 \
  --out $D/silver/pigments
# Standard PACE chlorophyll; it stayed out of the default stage list before the tools existed
eddy-tracking collocate-chlorophyll --pace-bgc $D/bronze/pace_l3_8d_bgc --tracks $D/silver/eddy_track --axis $D/silver/gulf_stream $RRS_DATES \
  --track-ids all --min-coverage 0.5 --min-pixels 10 \
  --out $D/silver/pace_chl
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
