import datetime as dt
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import xarray as xr
from contourpy import contour_generator
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import KDTree

KM_PER_DEG_LAT = 111.0


class GulfStreamCenterline:
    """
    Ordered Gulf Stream centerline points.

    The points are stored in streamline order, not sorted by longitude. This allows meanders and repeated longitude values.
    """

    def __init__(self, lon: np.ndarray, lat: np.ndarray) -> None:
        lon_arr = np.asarray(lon, dtype=float)
        lat_arr = np.asarray(lat, dtype=float)
        if lon_arr.shape != lat_arr.shape:
            raise ValueError("lon and lat must have the same shape")
        self.lon = lon_arr
        self.lat = lat_arr


def load_axis_fields(fp: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(fp) as ds:
        ds = ds.isel(time=0)
        speed = np.hypot(ds.ugos_full.to_numpy(), ds.vgos_full.to_numpy())
        return ds.longitude.to_numpy(), ds.latitude.to_numpy(), ds.adt_full.to_numpy(), speed


def find_fastest_adt_level(
    lon: np.ndarray, lat: np.ndarray, adt: np.ndarray, speed: np.ndarray, adt_level_range: tuple[float, float]
) -> float:
    speed_at = RegularGridInterpolator((lat, lon), np.nan_to_num(speed), bounds_error=False, fill_value=0.0)
    contours = contour_generator(lon, lat, np.ma.masked_invalid(adt))
    levels = np.arange(adt_level_range[0], adt_level_range[1] + 0.0125, 0.025)
    scores = np.full(levels.size, -np.inf)
    for index, level in enumerate(levels):
        lines = cast(list[np.ndarray], contours.lines(level))
        if not lines:
            continue
        line = max(lines, key=len)
        if line[:, 0].max() >= lon.max() - 0.5:
            scores[index] = speed_at(line[:, ::-1]).mean()
    return float(levels[np.argmax(scores)])


def trace_adt_contour(lon: np.ndarray, lat: np.ndarray, adt: np.ndarray, level: float) -> GulfStreamCenterline:
    line = max(cast(list[np.ndarray], contour_generator(lon, lat, np.ma.masked_invalid(adt)).lines(level)), key=len)
    if line[0, 0] > line[-1, 0]:
        line = line[::-1]
    step_km = np.hypot(np.diff(line[:, 0]) * np.cos(np.radians(line[:-1, 1])), np.diff(line[:, 1])) * KM_PER_DEG_LAT
    along_km = np.concatenate([[0.0], np.cumsum(step_km)])
    sample_km = np.arange(0.0, along_km[-1], 5.0)
    return GulfStreamCenterline(np.interp(sample_km, along_km, line[:, 0]), np.interp(sample_km, along_km, line[:, 1]))


def trace_mean_streamline(swot_files: list[Path], adt_level_range: tuple[float, float]) -> GulfStreamCenterline:
    lon, lat, adt, speed = load_axis_fields(swot_files[0])
    adt_sum = np.zeros(adt.shape)
    speed_sum = np.zeros(adt.shape)
    days = np.zeros(adt.shape)
    for fp in swot_files:
        _, _, adt, speed = load_axis_fields(fp)
        finite = np.isfinite(adt) & np.isfinite(speed)
        adt_sum += np.where(finite, adt, 0.0)
        speed_sum += np.where(finite, speed, 0.0)
        days += finite
    days = np.where(days > 0, days, np.nan)
    mean_adt = adt_sum / days
    return trace_adt_contour(lon, lat, mean_adt, find_fastest_adt_level(lon, lat, mean_adt, speed_sum / days, adt_level_range))


def index_centerlines_by_date(streamline_df: pd.DataFrame) -> dict[dt.date, GulfStreamCenterline]:
    """Read silver streamline rows into ordered centerlines keyed by date."""
    centerlines: dict[dt.date, GulfStreamCenterline] = {}
    for date, grp in streamline_df.groupby("date"):
        ordered = grp.sort_values("point_idx")
        centerlines[pd.Timestamp(date).date()] = GulfStreamCenterline(  # pyright: ignore[reportArgumentType]
            ordered["lon"].to_numpy(),
            ordered["lat"].to_numpy(),
        )
    return centerlines


def compute_signed_distance_km(
    streamline_lon: np.ndarray, streamline_lat: np.ndarray, center_lon: float, center_lat: float
) -> tuple[float, str]:
    """
    Signed shortest distance from an eddy center to an ordered streamline.

    Positive (side 'N') means the eddy is to the left of the flow at the nearest point on the jet, which is north where the jet flows east, negative ('S') to the right. Returns (nan, '') if the streamline has too few finite points.
    """
    streamline_lon = np.asarray(streamline_lon, dtype=float)
    streamline_lat = np.asarray(streamline_lat, dtype=float)
    finite = np.isfinite(streamline_lon) & np.isfinite(streamline_lat)
    if finite.sum() < 2:
        return np.nan, ""

    lon = streamline_lon[finite]
    lat = streamline_lat[finite]
    scale_x = KM_PER_DEG_LAT * np.cos(np.radians(center_lat))
    x = (lon - center_lon) * scale_x
    y = (lat - center_lat) * KM_PER_DEG_LAT

    # x and y (n_finite,) -> start and end (n_finite - 1, 2) holding (x_km, y_km) per segment
    start = np.column_stack([x[:-1], y[:-1]])
    end = np.column_stack([x[1:], y[1:]])
    seg = end - start
    seg_len2 = np.einsum("ij,ij->i", seg, seg)
    nonzero = seg_len2 > 0
    if not np.any(nonzero):
        return np.nan, ""

    start = start[nonzero]
    seg = seg[nonzero]
    seg_len2 = seg_len2[nonzero]

    # The eddy center is the origin in this local km coordinate frame.
    t = np.clip(-np.einsum("ij,ij->i", start, seg) / seg_len2, 0.0, 1.0)
    # t (n_segments,) -> (n_segments, 1) to scale seg (n_segments, 2), then closest (n_segments, 2) -> dist (n_segments,)
    closest = start + t[:, np.newaxis] * seg
    dist = np.hypot(closest[:, 0], closest[:, 1])
    nearest_idx = int(np.argmin(dist))
    left_of_flow = seg[nearest_idx, 1] * closest[nearest_idx, 0] - seg[nearest_idx, 0] * closest[nearest_idx, 1]
    side = "N" if left_of_flow >= 0 else "S"
    signed = dist[nearest_idx] if side == "N" else -dist[nearest_idx]
    return float(signed), side


def compute_signed_distance_grid_km(
    streamline_lon: np.ndarray, streamline_lat: np.ndarray, lon: np.ndarray, lat: np.ndarray
) -> np.ndarray:
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    chord, nearest = KDTree(_convert_to_unit_sphere(streamline_lon, streamline_lat)).query(
        _convert_to_unit_sphere(lon_grid.ravel(), lat_grid.ravel())
    )
    nearest = nearest.reshape(lon_grid.shape)
    distance = np.degrees(2 * np.arcsin(chord / 2)).reshape(lon_grid.shape) * KM_PER_DEG_LAT

    last = streamline_lon.size - 1
    ahead = np.minimum(nearest + 1, last)
    behind = np.maximum(nearest - 1, 0)
    scale_x = np.cos(np.radians(streamline_lat[nearest]))
    flow_x = (streamline_lon[ahead] - streamline_lon[behind]) * scale_x
    flow_y = streamline_lat[ahead] - streamline_lat[behind]
    pixel_x = (lon_grid - streamline_lon[nearest]) * scale_x
    pixel_y = lat_grid - streamline_lat[nearest]
    left_of_flow = flow_x * pixel_y - flow_y * pixel_x
    signed = np.where(left_of_flow >= 0, distance, -distance)
    return np.where((nearest == 0) | (nearest == last), np.nan, signed)


def _convert_to_unit_sphere(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    return np.column_stack([np.cos(lat_rad) * np.cos(lon_rad), np.cos(lat_rad) * np.sin(lon_rad), np.sin(lat_rad)])
