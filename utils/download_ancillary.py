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
