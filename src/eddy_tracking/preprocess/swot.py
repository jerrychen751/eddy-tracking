import datetime as dt
import re
from pathlib import Path

import numpy as np
import xarray as xr

SWOT_SEARCH_DAYS = 4


def index_swot_files_by_date(swot_dir: Path) -> dict[dt.date, Path]:
    """Map measurement date (first 8-digit token in the name) to SWOT file path."""
    swot_date_re = re.compile(r"\d{8}")
    files = {}
    for fp in sorted(swot_dir.glob("*.nc")):
        m = swot_date_re.search(fp.name)
        if m:
            files[dt.datetime.strptime(m.group(), "%Y%m%d").date()] = fp
    return files


def find_nearest_swot_file(files: dict[dt.date, Path], target: dt.date) -> Path | None:
    """SWOT file on target, else the closest within SWOT_SEARCH_DAYS, else None."""
    for delta in range(SWOT_SEARCH_DAYS + 1):
        for day in (target - dt.timedelta(delta), target + dt.timedelta(delta)):
            if day in files:
                return files[day]
    return None


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


def compute_calm_mask_on_pace(swot_fp, pace_lon: np.ndarray, pace_lat: np.ndarray) -> np.ndarray:
    """
    Boolean (lat, lon) PACE-grid mask of calm water for one SWOT day.

    Interpolates |Ro| from the coarser SWOT grid onto the PACE pixels and thresholds it. Pixels saved as NaN in the SWOT bronze file interpolate to NaN and fail the comparison, so they are treated as not-calm.
    """
    # The DUACS/MIOST source variable is named relative_vorticity, but these files store normalized relative vorticity, not raw zeta in s^-1.
    # The values are Rossby number (Ro = zeta/f), so "calm" water is a direct threshold on |Ro|.
    bg_threshold_rossby = 0.1
    swot_lon, swot_lat, rossby_number = load_rossby_field(swot_fp)
    abs_rossby_number = xr.DataArray(
        np.abs(rossby_number),
        coords={"latitude": swot_lat, "longitude": swot_lon},
        dims=["latitude", "longitude"],
    )
    on_pace = abs_rossby_number.interp(
        latitude=pace_lat, longitude=pace_lon, method="linear"
    ).values
    return on_pace < bg_threshold_rossby
