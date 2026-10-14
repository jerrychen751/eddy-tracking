from __future__ import annotations

import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory

import earthaccess
import numpy as np
import xarray as xr
from earthaccess import DataGranule

from eddy_tracking.config import load_config, resolve_data_dir
from eddy_tracking.downloads.auth import login_earthdata
from eddy_tracking.preprocess.pace import read_pace_l2
from eddy_tracking.utils.files import write_netcdf_atomically
from eddy_tracking.validation.quality import apply_l2_quality_flags


def grid_pace_l2_day(
    granules: list[DataGranule],
    out_path: Path,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    excluded_flags: list[str],
) -> int:
    # The region on the PACE L3m 4 km grid: 1/24 degree cells with latitude descending, so a daily file reads like an L3m file.
    n_lon = round((lon_range[1] - lon_range[0]) * 24)
    n_lat = round((lat_range[1] - lat_range[0]) * 24)
    n_pixels = np.zeros(n_lat * n_lon, dtype=np.int64)
    rrs_sum = None
    wavelengths = None

    # The swath files of one day run to about 1 GB, so they live only until the day is gridded.
    with TemporaryDirectory(prefix="pace-l2-") as temporary_dir:
        swath_paths = earthaccess.download(granules, local_path=temporary_dir, threads=1, pqdm_kwargs={"disable": True})
        for swath_path in swath_paths:
            with xr.open_dataset(swath_path, group="navigation_data") as navigation:
                longitude = navigation["longitude"].to_numpy()
                latitude = navigation["latitude"].to_numpy()
            in_region = (
                (longitude >= lon_range[0]) & (longitude < lon_range[1])
                & (latitude > lat_range[0]) & (latitude <= lat_range[1])
            )
            if not in_region.any():
                continue
            lines = np.flatnonzero(in_region.any(axis=1))  # (n_lines, n_pixels) -> (n_lines_in_region,)
            pixels = np.flatnonzero(in_region.any(axis=0))  # (n_lines, n_pixels) -> (n_pixels_in_region,)
            swath_df = read_pace_l2(
                swath_path,
                line_indexer=slice(lines[0], lines[-1] + 1),
                pixel_indexer=slice(pixels[0], pixels[-1] + 1),
            )
            swath_df = apply_l2_quality_flags(swath_df, excluded_flags, to_nan=False)
            if rrs_sum is None:
                wavelengths = np.array(swath_df.attrs["wavelengths_nm"], dtype=np.float32)
                rrs_sum = np.zeros((n_lat * n_lon, len(wavelengths)))

            longitude = swath_df["longitude"].to_numpy()
            latitude = swath_df["latitude"].to_numpy()
            rrs = swath_df[list(swath_df.attrs["rrs_columns"])].to_numpy()  # (n_rows, n_wavelengths)
            keep = (
                (longitude >= lon_range[0]) & (longitude < lon_range[1])
                & (latitude > lat_range[0]) & (latitude <= lat_range[1])
                & np.isfinite(rrs).all(axis=1)
            )
            # A pixel goes to the cell that holds its center. NASA's L3 binning also spreads a pixel into the neighboring cells its footprint overlaps, so an L3m daily file fills more cells along a cloud edge than this grid does.
            cell = (
                np.floor((lat_range[1] - latitude[keep]) * 24).astype(int) * n_lon
                + np.floor((longitude[keep] - lon_range[0]) * 24).astype(int)
            )
            np.add.at(rrs_sum, cell, rrs[keep])
            n_pixels += np.bincount(cell, minlength=n_lat * n_lon)

    # A day whose swaths all miss the region writes no file.
    if rrs_sum is None:
        return 0

    filled = n_pixels > 0
    daily_rrs = np.full(rrs_sum.shape, np.nan, dtype=np.float32)
    daily_rrs[filled] = rrs_sum[filled] / n_pixels[filled, None]
    daily = xr.Dataset(
        {"Rrs": (("lat", "lon", "wavelength"), daily_rrs.reshape(n_lat, n_lon, -1))},  # (n_lat*n_lon, n_wavelengths) -> (n_lat, n_lon, n_wavelengths)
        coords={
            "lat": (lat_range[1] - (np.arange(n_lat) + 0.5) / 24).astype(np.float32),
            "lon": (lon_range[0] + (np.arange(n_lon) + 0.5) / 24).astype(np.float32),
            "wavelength": wavelengths,
        },
    )
    write_netcdf_atomically(daily, out_path, encoding={"Rrs": {"zlib": True, "complevel": 4}})
    return int(filled.sum())


def main(experiment: str) -> None:
    cfg = load_config(experiment)
    lon_range = tuple(cfg["base"]["region"]["lon_range"])
    lat_range = tuple(cfg["base"]["region"]["lat_range"])
    date_range = cfg["base"]["time"]["rrs_date_range"]
    pace_cfg = cfg["base"]["download"]["pace_l2"]
    out_dir = resolve_data_dir(cfg, "pace_dir")

    login_earthdata()
    results = earthaccess.search_data(
        concept_id=pace_cfg["collection_id"],
        temporal=(date_range[0], date_range[1]),
        bounding_box=(lon_range[0], lat_range[0], lon_range[1], lat_range[1]),
        count=-1,
    )
    # The version is pinned and a near-real-time name fails the match, so one day never mixes two processings.
    regex = re.compile(rf"PACE_OCI\.(\d{{8}})T\d{{6}}\.L2\.OC_AOP\.{re.escape(pace_cfg['version'])}\.nc")
    granules_by_day: dict[str, list[DataGranule]] = defaultdict(list)
    for granule in results:
        m = regex.fullmatch(granule.data_links()[0].split("/")[-1])
        if m:
            granules_by_day[m.group(1)].append(granule)
    print(
        f"matched_granules: {sum(len(granules) for granules in granules_by_day.values())}\n"
        f"days: {len(granules_by_day)}\n"
        f"collection_id: {pace_cfg['collection_id']}\n"
        f"version: {pace_cfg['version']}\n"
        f"excluded_flags: {' '.join(pace_cfg['excluded_flags'])}"
    )

    n_saved = n_skipped = n_errors = 0
    with ProcessPoolExecutor(max_workers=pace_cfg["max_workers"], initializer=login_earthdata) as executor:
        futures = {}
        for day, granules in sorted(granules_by_day.items()):
            out_path = out_dir / f"PACE_OCI.{day}.L2.DAY.AOP.{pace_cfg['version']}.4km.nc"
            if out_path.exists():
                n_skipped += 1
                continue
            future = executor.submit(grid_pace_l2_day, granules, out_path, lon_range, lat_range, pace_cfg["excluded_flags"])
            futures[future] = day
        for future in as_completed(futures):
            try:
                n_cells = future.result()
            except Exception as exc:
                print(
                    "status: error\n"
                    f"date: {futures[future]}\n"
                    f"error: {exc}",
                    file=sys.stderr,
                )
                n_errors += 1
                continue
            print(
                "status: saved\n"
                f"date: {futures[future]}\n"
                f"valid_cells: {n_cells}"
            )
            n_saved += 1

    print(
        "status: download_finished\n"
        f"days_saved: {n_saved}\n"
        f"days_skipped: {n_skipped}\n"
        f"errors: {n_errors}"
    )
    if n_errors:
        raise SystemExit(f"{n_errors} day(s) failed to download")


if __name__ == "__main__":
    main(sys.argv[1])
