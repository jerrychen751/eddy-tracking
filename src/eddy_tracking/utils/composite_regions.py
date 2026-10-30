# Pixel regions of the 8-day composites that the satellite notebooks average: the inside of an eddy's speed contour, the regional means with their coverage gate, and the composites that hold target eddies
import datetime as dt
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from eddy_tracking.preprocess.tracks import EddyObs, collect_eddies_for_date_range, mask_pixels_inside_contour


def locate_interior(lon: np.ndarray, lat: np.ndarray, contour_lon: np.ndarray, contour_lat: np.ndarray) -> np.ndarray:
    # The pixels inside a speed contour, the interior every collocation stage uses; only the grid box around the contour can fall inside it, which keeps the 4 km grid fast
    inside = np.zeros((lat.size, lon.size), dtype=bool)
    rows = np.flatnonzero((lat >= contour_lat.min()) & (lat <= contour_lat.max()))
    columns = np.flatnonzero((lon >= contour_lon.min()) & (lon <= contour_lon.max()))
    inside[np.ix_(rows, columns)] = mask_pixels_inside_contour(lon[columns], lat[rows], contour_lon, contour_lat)
    return inside


def average_regions(field: np.ndarray, regions: dict[str, np.ndarray], min_pixels: int, min_coverage: float) -> dict[str, np.ndarray]:
    # field is (n_lat, n_lon) or (n_lat, n_lon, n_wavelength); a pixel counts only where every value is finite
    means = {}
    for region, inside in regions.items():
        values = field.reshape(*inside.shape, -1)[inside]  # (n_lat, n_lon, n_values) -> (n_pixels, n_values), also when a small contour holds no pixel
        valid = np.isfinite(values).all(axis=1)
        covered = valid.sum() >= min_pixels and valid.mean() >= min_coverage
        means[region] = values[valid].mean(axis=0) if covered else np.full(values.shape[1], np.nan)
    return means


def list_composites(sst_dir: Path, date_index: dict[dt.date, list[EddyObs]], target_keys: set[tuple[str, int]]) -> list[tuple[Path, pd.Timestamp, list[EddyObs], list[tuple[np.ndarray, np.ndarray]]]]:
    # Each 8-day composite with its midpoint, each target eddy present at its observation nearest the midpoint as the collocation stages pick it, and the contours of every tracked eddy during its 8 days
    composites = []
    for path in sorted(sst_dir.glob('*.nc')):
        start, end = (pd.Timestamp(part) for part in path.name.split('.')[1].split('_'))
        midpoint = cast(pd.Timestamp, start + cast(pd.Timedelta, end - start) / 2)
        eddies = [eddy for eddy in collect_eddies_for_date_range(date_index, cast(pd.Timestamp, start).date(), cast(pd.Timestamp, end).date()) if (eddy.polarity, eddy.track_id) in target_keys]
        if eddies:
            contours = [(eddy.contour_lon, eddy.contour_lat) for day in pd.date_range(start, end) for eddy in date_index.get(day.date(), [])]
            composites.append((path, midpoint, eddies, contours))
    return composites
