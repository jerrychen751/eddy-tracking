from __future__ import annotations

import datetime as dt
import re
import shutil
from pathlib import Path

import earthaccess
import xarray as xr


def strip_harmony_prefix(filename: str) -> str:
    """
    Harmony stages downloaded files as '{item_id}_{original_name}'.

    The item_id changes per job, so strip it for stable filenames.
    """
    return re.sub(r"^\d+_", "", filename)


def infer_lat_lon_names(ds: xr.Dataset) -> tuple[str, str]:
    """
    SST uses 'lat'/'lon', SSS uses 'latitude'/'longitude' — this normalizes
    both so downstream code does not need to know which product is loaded.
    """
    lat_name = next((n for n in ("lat", "latitude") if n in ds.coords), None)
    lon_name = next((n for n in ("lon", "longitude") if n in ds.coords), None)
    if lat_name is None or lon_name is None:
        raise KeyError(f"Cannot infer lat/lon coords from {list(ds.coords)}")
    return lat_name, lon_name


def subset_to_bbox_ds(
    ds: xr.Dataset,
    output_path: Path,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
) -> None:
    """
    Handles both ascending and descending latitude coordinates. Calls
    ``.load()`` before writing because S3-streamed xarray datasets need
    to be materialized before netcdf serialization.
    """
    lat_name, lon_name = infer_lat_lon_names(ds)
    lat_vals = ds[lat_name].values
    lat_ascending = lat_vals[0] < lat_vals[-1]

    lat_slice = (
        slice(lat_range[0], lat_range[1]) if lat_ascending
        else slice(lat_range[1], lat_range[0])
    )
    lon_slice = slice(lon_range[0], lon_range[1])

    subset = ds.sel({lat_name: lat_slice, lon_name: lon_slice})
    subset.load()
    tmp_path = output_path.with_suffix(".tmp.nc")
    subset.to_netcdf(tmp_path)
    tmp_path.rename(output_path)


def subset_to_bbox(
    input_path: Path,
    output_path: Path,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
) -> None:
    with xr.open_dataset(input_path) as ds:
        subset_to_bbox_ds(ds, output_path, lon_range, lat_range)


SST_FILENAME_RE = re.compile(
    r"^AQUA_MODIS\.\d{8}_\d{8}\.L3m\.8D\.SST\.sst\.4km\.nc$"
)


def download_aqua_sst_8d_4km(
    date_range: tuple[str, str],
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    out_dir: Path,
    raw_tmp: Path,
    collection_id: str,
    download_threads: int = 4,
) -> int:
    """
    Download AQUA MODIS 8-day 4km SST composites covering a date range and
    bbox. Returns the number of new files saved. Skips files already present
    in out_dir.
    """
    print("Downloading SST (AQUA MODIS L3m 8D 4km)")
    earthaccess.login()

    granules = earthaccess.search_data(
        concept_id=collection_id,
        temporal=(date_range[0], date_range[1]),
        count=-1,
    )
    print(f"CMR returned {len(granules)} granules")

    if not granules:
        return 0

    out_dir = Path(out_dir)
    raw_tmp = Path(raw_tmp)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted_urls: list[str] = []
    for granule in granules:
        for url in granule.data_links():
            filename = url.split("?", 1)[0].rsplit("/", 1)[-1]
            if SST_FILENAME_RE.match(filename):
                if (out_dir / filename).exists():
                    continue
                wanted_urls.append(url)

    if not wanted_urls:
        return 0
    wanted_urls.sort()

    raw_tmp.mkdir(parents=True, exist_ok=True)
    try:
        paths = earthaccess.download(
            wanted_urls, local_path=str(raw_tmp), threads=download_threads,
        )
        saved = 0
        for path in paths:
            raw_path = Path(path)
            out_path = out_dir / raw_path.name
            if out_path.exists():
                continue
            subset_to_bbox(raw_path, out_path, lon_range, lat_range)
            saved += 1
    finally:
        shutil.rmtree(raw_tmp, ignore_errors=True)

    return saved
