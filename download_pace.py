"""
Download PACE OCI L3 Mapped Rrs for a configured experiment.
Thin orchestrator — all logic lives in utils/download_pace_l3.py.
"""

import argparse
import sys

from utils.config import load_config, resolve_data_dir
from utils.download_pace_l3 import download_pace_l3

parser = argparse.ArgumentParser()
parser.add_argument("experiment")
args = parser.parse_args()

cfg = load_config(args.experiment, "base.yaml")

lon_range = tuple(cfg["base"]["region"]["lon_range"])
lat_range = tuple(cfg["base"]["region"]["lat_range"])
date_range = tuple(cfg["base"]["time"]["rrs_date_range"])
temporal_res = cfg["base"]["download"]["pace"].get("temporal_resolution", "DAY")

saved, skipped, errors = download_pace_l3(
    date_range=date_range,
    lon_range=lon_range,
    lat_range=lat_range,
    out_dir=resolve_data_dir(cfg, "pace_dir"),
    temporal_res=temporal_res,
)

print(f"Done. {saved} saved, {skipped} skipped, {errors} errors.")
if errors:
    sys.exit(f"{errors} date(s) failed to download")
