"""
Download SST (AQUA MODIS) and SSS (SMAP) for a configured experiment.
Thin orchestrator — all logic lives in utils/download_ancillary.py.
"""

import argparse
import sys

from utils.config import load_config, resolve_data_dir
from utils.download_ancillary import (
    download_aqua_sst_8d_4km,
    download_smap_sss_8d,
)

parser = argparse.ArgumentParser()
parser.add_argument("experiment")
parser.add_argument("--only", choices=["sst", "sss"])
args = parser.parse_args()

cfg = load_config(args.experiment, "base.yaml")

lon_range = tuple(cfg["base"]["region"]["lon_range"])
lat_range = tuple(cfg["base"]["region"]["lat_range"])
date_range = tuple(cfg["base"]["time"]["rrs_date_range"])

run_sst = args.only in (None, "sst")
run_sss = args.only in (None, "sss")

sst_count = 0
sss_saved, sss_failed = 0, 0

if run_sst:
    sst_count = download_aqua_sst_8d_4km(
        date_range=date_range,
        lon_range=lon_range,
        lat_range=lat_range,
        out_dir=resolve_data_dir(cfg, "sst_dir"),
        raw_tmp=resolve_data_dir(cfg, "sst_raw_tmp"),
        collection_id=cfg["base"]["download"]["sst"]["collection_id"],
        download_threads=cfg["base"]["download"]["sst"]["download_threads"],
    )

if run_sss:
    sss_saved, sss_failed = download_smap_sss_8d(
        date_range=date_range,
        lon_range=lon_range,
        lat_range=lat_range,
        out_dir=resolve_data_dir(cfg, "sss_dir"),
        raw_tmp=resolve_data_dir(cfg, "sss_raw_tmp"),
        collection_id=cfg["base"]["download"]["sss"]["collection_id"],
        batch_days=10,
        num_requests_workers=cfg["base"]["download"]["sss"]["num_requests_workers"],
        download_chunk_size_mb=cfg["base"]["download"]["sss"]["download_chunk_size_mb"],
    )

print(f"Done. {sst_count} SST + {sss_saved} SSS files saved.")
if sss_failed:
    sys.exit(f"{sss_failed} SSS window(s) failed to download")
