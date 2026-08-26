"""Spatial and temporal subsetting predicates plus SWOT field loading."""
import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr


def parse_date_range(date_range: list[str] | None) -> tuple[dt.date, dt.date] | None:
    """
    Parse a ["YYYY-MM-DD", "YYYY-MM-DD"] config value, such as ["2024-10-01", "2025-12-31"], into (start, end) dates.

    Returns None when no window is configured, which disables temporal filtering.
    """
    if not date_range:
        return None
    return dt.date.fromisoformat(date_range[0]), dt.date.fromisoformat(date_range[1])


def is_in_subset(
    center_lon: float,
    center_lat: float,
    day: dt.date,
    region: dict | None,
    date_range: tuple[dt.date, dt.date] | None,
) -> bool:
    """
    Whether an eddy observation falls within the optional box and date window.

    region is {"lon_range": [lo, hi], "lat_range": [lo, hi]} in -180/180 longitude, such as {"lon_range": [-81, -60], "lat_range": [30, 45]}, or None for no spatial filter. date_range is an inclusive (start, end) pair or None. Every bound is inclusive, and a None filter always passes.
    """
    if date_range is not None:
        start, end = date_range
        if not (start <= day <= end):
            return False
    if region is not None:
        lon_lo, lon_hi = region["lon_range"]
        lat_lo, lat_hi = region["lat_range"]
        if not (lon_lo <= center_lon <= lon_hi and lat_lo <= center_lat <= lat_hi):
            return False
    return True


def load_rossby_field(swot_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return longitude, latitude, and the saved Rossby-number field."""
    with xr.open_dataset(swot_path) as dataset:
        if "time" in dataset["relative_vorticity"].dims:
            # (1, n_lat, n_lon) -> (n_lat, n_lon)
            dataset = dataset.isel(time=0)
        longitude = dataset["longitude"].to_numpy()
        latitude = dataset["latitude"].to_numpy()
        rossby_number = dataset["relative_vorticity"].to_numpy()
    return longitude, latitude, rossby_number
