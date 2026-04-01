"""
Download PACE OCI L3 Mapped Daily Rrs data via earthaccess.

Downloads full granules via HTTPS, subsets to the study region, and
saves with compression. Uses earthaccess.download() instead of
earthaccess.open() (S3 streaming) because S3 access is blocked on
some HPC compute nodes.
"""

import argparse
import datetime as dt
import re
import shutil
import sys
import tempfile
from pathlib import Path

import earthaccess
import xarray as xr

from utils.config import load_config, resolve_data_dir

parser = argparse.ArgumentParser()
parser.add_argument("experiment")
args = parser.parse_args()

cfg = load_config(args.experiment, "base.yaml")

LON_RANGE = tuple(cfg["base"]["region"]["lon_range"])
LAT_RANGE = tuple(cfg["base"]["region"]["lat_range"])
DATE_RANGE = tuple(cfg["base"]["time"]["date_range"])
OUT_DIR = resolve_data_dir(cfg, "pace_dir")

# Only keep daily 4km Rrs files (collection also contains 8-day, monthly, etc.)
DAILY_4KM_RE = re.compile(r"PACE_OCI\.(\d{8})\.L3m\.DAY\.RRS\..*\.Rrs\.4km\.nc")


def subset_and_save(ds: xr.Dataset, out_path: Path) -> None:
    """
    Subset to study region with compression and atomic write.

    L3 latitude runs 90→-90 (descending), so lat slice goes high-to-low.
    Drops 'palette' (visualization artifact). Writes with zlib level 4
    (~3x reduction on NaN-heavy arrays). Uses temp file + rename to
    prevent corrupt partial writes.
    """
    subset = ds.sel(
        lat=slice(LAT_RANGE[1], LAT_RANGE[0]),
        lon=slice(LON_RANGE[0], LON_RANGE[1]),
    )
    if "palette" in subset:
        subset = subset.drop_vars("palette")
    subset.load()
    tmp_path = out_path.with_suffix(".tmp.nc")
    encoding = {v: {"zlib": True, "complevel": 4} for v in subset.data_vars}
    subset.to_netcdf(tmp_path, encoding=encoding)
    tmp_path.rename(out_path)


def main():
    earthaccess.login()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    start = dt.datetime.strptime(DATE_RANGE[0], "%Y-%m-%d").date()
    end = dt.datetime.strptime(DATE_RANGE[1], "%Y-%m-%d").date()

    # Search once for all PACE L3 Rrs granules across the full date range.
    # The collection contains multiple temporal resolutions (daily, 8-day,
    # monthly, rolling composites) and spatial resolutions (4km, 9km, etc.),
    # so we filter by filename after the search.
    print("Searching for PACE L3 Rrs granules...")
    results = earthaccess.search_data(
        short_name="PACE_OCI_L3M_RRS",
        temporal=(DATE_RANGE[0], DATE_RANGE[1]),
        count=5000,
    )
    print("CMR returned {} granules (all resolutions)".format(len(results)))

    # Build a date -> granule mapping for daily 4km files only
    daily_granules = {}
    for granule in results:
        for link in granule.data_links():
            filename = link.split("/")[-1]
            m = DAILY_4KM_RE.match(filename)
            if m:
                date_str = m.group(1)
                daily_granules[date_str] = (filename, granule)
                break

    print("Matched {} daily 4km granules".format(len(daily_granules)))

    total_saved = 0
    total_skipped = 0
    total_missing = 0
    total_errors = 0
    current = start

    while current <= end:
        date_str = current.strftime("%Y%m%d")

        if date_str not in daily_granules:
            total_missing += 1
            current += dt.timedelta(days=1)
            continue

        filename, granule = daily_granules[date_str]
        out_path = OUT_DIR / filename  # use actual granule filename, not a hardcoded template
        if out_path.exists():
            total_skipped += 1
            current += dt.timedelta(days=1)
            continue

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                paths = earthaccess.download([granule], local_path=tmp_dir)
                if not paths:
                    print("No file returned for {}".format(current))
                    current += dt.timedelta(days=1)
                    continue
                raw_path = Path(paths[0])
                with xr.open_dataset(raw_path) as ds:
                    subset_and_save(ds, out_path)
            size_mb = out_path.stat().st_size / (1024 * 1024)
            total_saved += 1
            print("Saved: {} ({:.1f} MB)".format(filename, size_mb))
        except Exception as e:
            print("Error on {}: {}".format(current, e))
            total_errors += 1

        current += dt.timedelta(days=1)

    print(
        "Done. {} saved, {} skipped, {} not on server, {} errors. "
        "Files in {}".format(total_saved, total_skipped, total_missing, total_errors, OUT_DIR)
    )
    if total_errors:
        sys.exit(f"{total_errors} date(s) failed to download")


if __name__ == "__main__":
    main()
