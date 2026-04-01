from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import numpy as np
import xarray as xr


def _parse_sst_time_window(filename: str) -> tuple[dt.date, dt.date]:
    match = re.search(r'(\d{8})_(\d{8})', filename)
    if not match:
        raise ValueError(f"Cannot parse SST date window from: {filename}")
    start = dt.datetime.strptime(match.group(1), '%Y%m%d').date()
    end = dt.datetime.strptime(match.group(2), '%Y%m%d').date()
    return start, end


def _parse_sss_center_date(filename: str) -> dt.date:
    match = re.search(r'SSS_(\d{8})_', filename)
    if not match:
        raise ValueError(f"Cannot parse SSS date from: {filename}")
    return dt.datetime.strptime(match.group(1), '%Y%m%d').date()


def load_sst_dataset(sst_dir: Path) -> xr.DataArray:
    sst_dir = Path(sst_dir)
    files = sorted(sst_dir.glob('*.nc'))
    if not files:
        raise FileNotFoundError(f"No SST .nc files in {sst_dir}")

    arrays = []
    for f in files:
        start, end = _parse_sst_time_window(f.name)
        midpoint = start + (end - start) / 2
        with xr.open_dataset(f) as ds:
            da = ds['sst'].load().expand_dims(time=[np.datetime64(midpoint)])
        arrays.append(da)

    return xr.concat(arrays, dim='time')


def load_sss_dataset(sss_dir: Path) -> xr.DataArray:
    sss_dir = Path(sss_dir)
    files = sorted(sss_dir.glob('*.nc4'))
    if not files:
        raise FileNotFoundError(f"No SSS .nc4 files in {sss_dir}")

    arrays = []
    for f in files:
        center = _parse_sss_center_date(f.name)
        with xr.open_dataset(f) as ds:
            da = ds['smap_sss'].load().expand_dims(time=[np.datetime64(center)])
        arrays.append(da)

    return xr.concat(arrays, dim='time')


def sample_ancillary(
    sst_da: xr.DataArray,
    sss_da: xr.DataArray,
    lons: np.ndarray,
    lats: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample SST and SSS at given (lon, lat, time) points via nearest-neighbor.

    Uses xarray's vectorized indexing with method='nearest' in all three
    dimensions. Returns NaN where no data is available (land mask, gaps).

    Note:
        SST grid uses coord names 'lat'/'lon'.
        SSS grid uses 'latitude'/'longitude'.
    """
    # SST grid uses 'lat'/'lon'
    sst_vals = sst_da.sel(
        lat=xr.DataArray(lats, dims='points'),
        lon=xr.DataArray(lons, dims='points'),
        time=xr.DataArray(times, dims='points'),
        method='nearest',
    ).values

    # SSS grid uses 'latitude'/'longitude'
    sss_vals = sss_da.sel(
        latitude=xr.DataArray(lats, dims='points'),
        longitude=xr.DataArray(lons, dims='points'),
        time=xr.DataArray(times, dims='points'),
        method='nearest',
    ).values

    return sst_vals.astype(float), sss_vals.astype(float)
