"""Spatial and temporal subsetting predicates, plus SWOT field loading, for
stages that need to know whether a location is open ocean.

Originally extracted from collocate_pace.py so the logic is unit-testable —
that script parses CLI args at import time and cannot be imported directly.
"""
import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt

# SWOT's grid is 0.125 deg/pixel, so 8 pixels is ~1 deg (~90-111 km depending
# on direction at this project's latitudes).
COAST_MIN_DISTANCE_PIXELS = 8


def parse_date_range(date_range: list[str] | None) -> tuple[dt.date, dt.date] | None:
    """Parse a ["YYYY-MM-DD", "YYYY-MM-DD"] config value into (start, end) dates.

    Returns None when no window is configured, which disables temporal filtering.
    """
    if not date_range:
        return None
    return dt.date.fromisoformat(date_range[0]), dt.date.fromisoformat(date_range[1])


def in_subset(
    center_lon: float,
    center_lat: float,
    day: dt.date,
    region: dict | None,
    date_range: tuple[dt.date, dt.date] | None,
) -> bool:
    """Whether an eddy observation falls within the optional box and date window.

    region is {"lon_range": [lo, hi], "lat_range": [lo, hi]} in -180/180 longitude,
    or None for no spatial filter; date_range is an inclusive (start, end) or None.
    Both bounds inclusive; a None filter always passes, so omitting both reproduces
    the original unfiltered behavior.
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


def build_exclusion_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
) -> np.ndarray:
    """True where (lon, lat) falls outside the box - i.e., where the point survives."""
    inside = (lon >= lon_range[0]) & (lon <= lon_range[1]) & (lat >= lat_range[0]) & (lat <= lat_range[1])
    return ~inside


def swot_is_valid(ds: xr.Dataset) -> np.ndarray:
    """True where adt, ugos, vgos, and relative_vorticity are all finite."""
    return (
        np.isfinite(ds["adt"].values)
        & np.isfinite(ds["ugos"].values)
        & np.isfinite(ds["vgos"].values)
        & np.isfinite(ds["relative_vorticity"].values)
    )


def load_rossby_field(
    swot_fp: Path, min_distance_pixels: int = COAST_MIN_DISTANCE_PIXELS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lon, lat, and Rossby number from one SWOT file, with pixels closer than min_distance_pixels to the nearest invalid cell (see swot_is_valid) set to NaN."""
    with xr.open_dataset(swot_fp) as ds:
        if "time" in ds["relative_vorticity"].dims:
            ds = ds.isel(time=0)
        lon = ds["longitude"].to_numpy()
        lat = ds["latitude"].to_numpy()
        coast_ok = distance_transform_edt(swot_is_valid(ds)) >= min_distance_pixels
        # Source variable name is relative_vorticity, but these DUACS/MIOST
        # values are already normalized by Coriolis: Ro = zeta / f.
        rossby_number = np.where(coast_ok, ds["relative_vorticity"].to_numpy(), np.nan)
    return lon, lat, rossby_number
