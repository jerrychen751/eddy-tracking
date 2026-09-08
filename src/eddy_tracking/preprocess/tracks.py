import datetime as dt
from collections import defaultdict
from typing import NamedTuple, cast

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

from eddy_tracking.config import resolve_output_dir
from eddy_tracking.packages.py_eddy_tracker.observations.tracking import (
    TrackEddiesObservations,
)
from eddy_tracking.utils.subset import is_in_subset

PET_EPOCH = dt.date(1950, 1, 1)


def load_tracks(experiment: str, polarity: str) -> TrackEddiesObservations:
    track_dir = resolve_output_dir(experiment, "eddy_track", polarity)
    return TrackEddiesObservations.load_file(str(track_dir / f"{polarity}_tracks.zarr"))


class EddyObs(NamedTuple):
    """Single eddy observation on one date: contour + center coordinates."""

    track_id: int
    polarity: str
    contour_lon: np.ndarray
    contour_lat: np.ndarray
    center_lon: float
    center_lat: float
    radius_km: float


def build_date_eddy_index(
    tracked: TrackEddiesObservations,
    polarity: str,
    track_ids: set[int] | None = None,
    region: dict | None = None,
    date_range: tuple[dt.date, dt.date] | None = None,
) -> dict[dt.date, list[EddyObs]]:
    """
    Index detected observations by date, excluding interpolated track gaps.

    Contour longitudes are converted from PET's 0 to 360 convention to -180 to 180.
    """
    date_index: dict[dt.date, list[EddyObs]] = defaultdict(list)

    unique_track_ids = np.unique(tracked.track)
    for track_id in unique_track_ids:
        if track_ids is not None and track_id not in track_ids:
            continue

        mask = tracked.track == track_id
        times = tracked.time[mask]
        virtuals = tracked.virtual[mask]
        contour_lons = tracked.contour_lon_s[mask]
        contour_lats = tracked.contour_lat_s[mask]
        center_lons = tracked.longitude[mask]
        center_lats = tracked.latitude[mask]
        radii = tracked.radius_s[mask]

        for obs_idx in range(len(times)):
            if virtuals[obs_idx]:
                continue

            day = PET_EPOCH + dt.timedelta(days=int(times[obs_idx]))
            center_lon = float((center_lons[obs_idx] + 180) % 360 - 180)
            center_lat = float(center_lats[obs_idx])
            if not is_in_subset(center_lon, center_lat, day, region, date_range):
                continue

            obs = EddyObs(
                track_id=int(track_id),
                polarity=polarity,
                contour_lon=(contour_lons[obs_idx] + 180) % 360 - 180,
                contour_lat=contour_lats[obs_idx],
                center_lon=center_lon,
                center_lat=center_lat,
                radius_km=float(radii[obs_idx]) / 1000.0,
            )
            date_index[day].append(obs)

    return date_index


def collect_eddies_for_window(
    date_index: dict[dt.date, list["EddyObs"]],
    start: dt.date,
    end: dt.date,
) -> list["EddyObs"]:
    """Select each eddy's observation nearest an 8-day window midpoint."""
    midpoint = start + (end - start) / 2
    best: dict[tuple[int, str], tuple[EddyObs, float]] = {}

    day = start
    while day <= end:
        for obs in date_index.get(day, []):
            key = (obs.track_id, obs.polarity)
            midpoint_dist = abs((day - midpoint).days)
            if key not in best or midpoint_dist < best[key][1]:
                best[key] = (obs, midpoint_dist)
        day += dt.timedelta(days=1)

    return [obs for obs, _ in best.values()]


def load_track_observations(experiment: str) -> pd.DataFrame:
    """
    Non-virtual eddy observations across both polarities, one row each.

    Columns: polarity, track_id, date, center_lon (-180/180), center_lat, radius_km, amplitude_cm, contour_lon (-180/180), contour_lat.
    """
    frames = []
    for polarity in ("cyclone", "anticyclone"):
        tracked = load_tracks(experiment, polarity)
        keep = ~tracked.virtual.astype(bool)
        longitudes = tracked.longitude[keep]
        latitudes = tracked.latitude[keep]
        if not (
            np.isfinite(longitudes).all()
            and np.isfinite(latitudes).all()
            and ((latitudes >= -90) & (latitudes <= 90)).all()
        ):
            raise ValueError(f"Invalid physical center coordinates in {polarity} tracks of {experiment}")
        days = [PET_EPOCH + dt.timedelta(days=int(t)) for t in tracked.time[keep]]
        frames.append(pd.DataFrame({
            "polarity": polarity,
            "track_id": tracked.track[keep].astype(int),
            "date": pd.to_datetime(days),
            "center_lon": (longitudes + 180) % 360 - 180,
            "center_lat": latitudes,
            "radius_km": tracked.radius_s[keep] / 1000.0,
            "amplitude_cm": tracked.amplitude[keep] * 100.0,
            "contour_lon": list((tracked.contour_lon_s[keep] + 180) % 360 - 180),
            "contour_lat": list(tracked.contour_lat_s[keep]),
        }))
    observations = pd.concat(frames, ignore_index=True)
    if observations.empty:
        raise ValueError("Track files contain no physical observations")
    if observations.duplicated(["polarity", "track_id", "date"]).any():
        raise ValueError("Track files contain duplicate physical track dates")
    physical_dates = observations.groupby(["polarity", "track_id"])["date"]
    if (cast(pd.Series, physical_dates.max()) <= cast(pd.Series, physical_dates.min())).any():
        raise ValueError("Physical track lifetimes must exceed zero days")
    return observations


def mask_pixels_inside_contour(
    lon: np.ndarray, lat: np.ndarray, contour_lon: np.ndarray, contour_lat: np.ndarray
) -> np.ndarray:
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    polygon = MplPath(np.column_stack([contour_lon, contour_lat]))
    inside = polygon.contains_points(np.column_stack([lon_grid.ravel(), lat_grid.ravel()]))
    return inside.reshape(lon_grid.shape)


def is_in_any_contour(
    contours: list[tuple[np.ndarray, np.ndarray]], lons: np.ndarray, lats: np.ndarray
) -> np.ndarray:
    """Boolean over the given points: inside at least one eddy contour polygon."""
    points = np.column_stack([lons, lats])  # (n_points,) + (n_points,) -> (n_points, 2)
    inside = np.zeros(points.shape[0], dtype=bool)
    for contour_lon, contour_lat in contours:
        polygon = MplPath(np.column_stack([contour_lon, contour_lat]))  # (n_vertices,) + (n_vertices,) -> (n_vertices, 2)
        inside |= polygon.contains_points(points)
    return inside
