"""
Download SST (AQUA MODIS) and SSS (SMAP) data for the Kramer SDP model.
"""

import argparse
import datetime as dt
import os
import re
import shutil
import sys
from pathlib import Path

import earthaccess
import xarray as xr

from utils.config import load_config, resolve_data_dir

parser = argparse.ArgumentParser()
parser.add_argument("experiment")
parser.add_argument("--only", choices=["sst", "sss"])
args = parser.parse_args()

# Config from YAML — must load before importing harmony
cfg = load_config(args.experiment, "base.yaml")

# Set Harmony worker count before importing harmony (reads env at Config init)
os.environ['NUM_REQUESTS_WORKERS'] = str(cfg["base"]["download"]["sss"]["num_requests_workers"])
os.environ['DOWNLOAD_CHUNK_SIZE'] = str(cfg["base"]["download"]["sss"]["download_chunk_size_mb"] * 1024 * 1024)

from harmony import BBox, Client, Collection, Request  # noqa: E402

# Derived from config
LON_RANGE = tuple(cfg["base"]["region"]["lon_range"])
LAT_RANGE = tuple(cfg["base"]["region"]["lat_range"])
DATE_RANGE = tuple(cfg["base"]["time"]["date_range"])

SST_DIR = resolve_data_dir(cfg, "sst_dir")
SSS_DIR = resolve_data_dir(cfg, "sss_dir")
SST_RAW_TMP = resolve_data_dir(cfg, "sst_raw_tmp")
SSS_RAW_TMP = resolve_data_dir(cfg, "sss_raw_tmp")

SST_COLLECTION_ID = cfg["base"]["download"]["sst"]["collection_id"]
SSS_COLLECTION_ID = cfg["base"]["download"]["sss"]["collection_id"]

# Only keep 8-day, 4km SST composites (CMR returns many product types)
SST_FILENAME_RE = re.compile(
    r"^AQUA_MODIS\.\d{8}_\d{8}\.L3m\.8D\.SST\.sst\.4km\.nc$"
)


def strip_harmony_prefix(filename: str) -> str:
    """
    Harmony names staged files as '{item_id}_{original_name}'.

    The item_id changes per job, so strip it for stable filenames.
    """
    return re.sub(r'^\d+_', '', filename)


def infer_lat_lon_names(ds: xr.Dataset) -> tuple[str, str]:
    """
    SST uses 'lat'/'lon', SSS uses 'latitude'/'longitude' — this
    normalizes both so downstream code doesn't need to know.
    """
    lat_name = next(
        (n for n in ("lat", "latitude") if n in ds.coords), None
    )
    lon_name = next(
        (n for n in ("lon", "longitude") if n in ds.coords), None
    )
    if lat_name is None or lon_name is None:
        raise KeyError(
            f"Cannot infer lat/lon coords from {list(ds.coords)}"
        )
    return lat_name, lon_name


def subset_to_bbox(
    input_path: Path,
    output_path: Path,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
) -> None:
    """
    Handles both ascending and descending latitude coordinates.
    """
    with xr.open_dataset(input_path) as ds:
        subset_to_bbox_ds(ds, output_path, lon_range, lat_range)


def subset_to_bbox_ds(
    ds: xr.Dataset,
    output_path: Path,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
) -> None:
    """
    Handles both ascending and descending latitude coordinates.
    Loads into memory before writing (necessary for S3-streamed datasets).
    """
    lat_name, lon_name = infer_lat_lon_names(ds)
    lat_vals = ds[lat_name].values
    lat_asc = lat_vals[0] < lat_vals[-1]

    lat_slice = (
        slice(lat_range[0], lat_range[1]) if lat_asc
        else slice(lat_range[1], lat_range[0])
    )
    lon_slice = slice(lon_range[0], lon_range[1])

    subset = ds.sel({lat_name: lat_slice, lon_name: lon_slice})
    subset.load()
    tmp_path = output_path.with_suffix(".tmp.nc")
    subset.to_netcdf(tmp_path)
    tmp_path.rename(output_path)


def download_sst() -> int:
    """
    Download AQUA MODIS 8-day 4km SST via earthaccess, subset locally.

    Downloads full files to a temp dir via HTTPS, subsets to the study
    region, then cleans up. Note: this requires direct HTTPS access to
    oceandata.sci.gsfc.nasa.gov (blocked from some HPC clusters — in
    that case, download locally and rsync).
    """
    print("Downloading SST (AQUA MODIS L3m 8D 4km)")

    earthaccess.login()

    granules = earthaccess.search_data(
        concept_id=SST_COLLECTION_ID,
        temporal=(DATE_RANGE[0], DATE_RANGE[1]),
        count=-1,
    )
    print(f"CMR returned {len(granules)} granules")

    if not granules:
        print("No SST granules found")
        return 0

    # Filter to 8-day 4km composites, collect download URLs
    wanted_urls: list[str] = []
    for granule in granules:
        for url in granule.data_links():
            filename = url.split("?", 1)[0].rsplit("/", 1)[-1]
            if SST_FILENAME_RE.match(filename):
                if (SST_DIR / filename).exists():
                    print(f"Already exists: {filename}")
                    continue
                wanted_urls.append(url)

    if not wanted_urls:
        print("No new SST files to download")
        return 0

    wanted_urls.sort()
    print(f"Downloading {len(wanted_urls)} SST files to temp dir...")

    SST_RAW_TMP.mkdir(parents=True, exist_ok=True)
    try:
        paths = earthaccess.download(
            wanted_urls, local_path=str(SST_RAW_TMP),
            threads=cfg["base"]["download"]["sst"]["download_threads"],
        )

        count = 0
        for path in paths:
            raw_path = Path(path)
            out_path = SST_DIR / raw_path.name
            if out_path.exists():
                continue

            print(f"Subsetting: {raw_path.name}")
            subset_to_bbox(raw_path, out_path, LON_RANGE, LAT_RANGE)
            count += 1
    finally:
        shutil.rmtree(SST_RAW_TMP, ignore_errors=True)

    print(f"Saved {count} subsetted SST files")
    return count


def download_sss() -> tuple[int, int]:
    """
    Batches into 10-day windows to avoid Harmony request size limits.
    Large single requests can hang or return partial results.

    Returns:
        (files_saved, failed_windows)
    """
    print("Downloading SSS (SMAP L3 8-day running mean)")

    harmony_client = Client()
    collection = Collection(id=SSS_COLLECTION_ID)

    start = dt.datetime.strptime(DATE_RANGE[0], "%Y-%m-%d")
    end = dt.datetime.strptime(DATE_RANGE[1], "%Y-%m-%d")

    # Batch into small 10-day windows to stay within Harmony limits.
    # Large requests (even monthly) can hang or return partial results.
    batch_days = 10
    windows = []
    window_start = start
    while window_start < end:
        window_end = min(window_start + dt.timedelta(days=batch_days), end)
        windows.append((window_start, window_end))
        window_start = window_end

    SSS_RAW_TMP.mkdir(parents=True, exist_ok=True)
    total_count = 0
    failed_windows = 0

    for i, (win_start, win_end) in enumerate(windows, 1):
        label = win_start.strftime("%Y-%m")
        print(f"[{i}/{len(windows)}] {label} ({win_start.date()} to {win_end.date()})")

        request = Request(
            collection=collection,
            spatial=BBox(LON_RANGE[0], LAT_RANGE[0], LON_RANGE[1], LAT_RANGE[1]),
            temporal={"start": win_start, "stop": win_end},
            granule_name=["*8DAYS*"],
            max_results=200,
            skip_preview=True,
        )

        try:
            job_id = harmony_client.submit(request)
            print(f"Job submitted: {job_id}")

            futures = harmony_client.download_all(
                job_id, directory=str(SSS_RAW_TMP), overwrite=True
            )
            raw_files = [Path(f.result()) for f in futures]
            print(f"Downloaded {len(raw_files)} files")

        except Exception as e:
            print(f"Error: {e}")
            failed_windows += 1
            continue

        for raw_path in raw_files:
            stable_name = strip_harmony_prefix(raw_path.name)
            out_path = SSS_DIR / stable_name

            if out_path.exists():
                continue

            shutil.copy2(raw_path, out_path)
            total_count += 1
            print(f"Saved: {stable_name}")

        # Clean tmp between batches to avoid accumulating files
        for f in SSS_RAW_TMP.iterdir():
            f.unlink()

    shutil.rmtree(SSS_RAW_TMP, ignore_errors=True)
    print(f"Saved {total_count} new SSS files")
    return total_count, failed_windows


def main():
    run_sst = args.only in (None, "sst")
    run_sss = args.only in (None, "sss")

    if run_sst:
        SST_DIR.mkdir(parents=True, exist_ok=True)
    if run_sss:
        SSS_DIR.mkdir(parents=True, exist_ok=True)

    sst_count = download_sst() if run_sst else 0
    sss_count, sss_failures = download_sss() if run_sss else (0, 0)

    print(f"Done. {sst_count} SST + {sss_count} SSS files saved.")
    if run_sst:
        print(f"SST: {SST_DIR}")
    if run_sss:
        print(f"SSS: {SSS_DIR}")

    if sss_failures:
        sys.exit(f"{sss_failures} SSS window(s) failed to download")


if __name__ == "__main__":
    main()
