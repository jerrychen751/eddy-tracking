"""
Collocate PACE L3 Rrs observations with tracked eddy contours.

For each PACE file (daily or 8-day composite), finds all eddies that
were detected during the observation period, extracts valid Rrs pixels
within each eddy's contour, and writes per-eddy Parquet files.

Temporal resolution is set via the collocate_pace config section:
  - "DAY" (default): exact date match between PACE file and eddy detection
  - "8D": for each 8-day composite, picks the eddy contour from the day
    closest to the window midpoint (since the Rrs is a temporal average)
"""

import argparse
import datetime as dt
import re
from collections import defaultdict
from typing import NamedTuple

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path as MplPath

from utils.config import (
    METADATA_COLS,
    load_config,
    resolve_data_dir,
    resolve_output_dir,
)
from eddy_tracking.packages.py_eddy_tracker.observations.tracking import (
    TrackEddiesObservations,
)
from eddy_tracking.utils.subset import in_subset, parse_date_range

PET_EPOCH = dt.date(1950, 1, 1)
PACE_DAILY_RE = re.compile(r"PACE_OCI\.(\d{8})\.L3m\.DAY\.AOP\.")
PACE_8DAY_RE = re.compile(r"PACE_OCI\.(\d{8})_(\d{8})\.L3m\.8D\.AOP\.")


class EddyObs(NamedTuple):
    """Single eddy observation on one date: contour + center coordinates."""

    track_id: int
    polarity: str
    contour_lon: np.ndarray
    contour_lat: np.ndarray
    center_lon: float
    center_lat: float


def collocate_one_observation(
    lon: np.ndarray,
    lat: np.ndarray,
    rrs: np.ndarray,
    contour_lon: np.ndarray,
    contour_lat: np.ndarray,
    min_coverage: float,
) -> dict[str, np.ndarray | float] | None:
    """
    Return valid Rrs pixels inside a contour when coverage meets the threshold.

    NASA L3 quality flags are already represented as NaN values in ``rrs``.
    """
    longitude_grid, latitude_grid = np.meshgrid(lon, lat)
    n_grid_cells = longitude_grid.size
    grid_longitudes = longitude_grid.ravel()
    grid_latitudes = latitude_grid.ravel()
    flattened_rrs = rrs.reshape(n_grid_cells, -1)

    polygon = MplPath(np.column_stack([contour_lon, contour_lat]))
    inside_contour = polygon.contains_points(
        np.column_stack([grid_longitudes, grid_latitudes])
    )

    finite_spectra = np.all(np.isfinite(flattened_rrs), axis=1)
    valid_pixels = inside_contour & finite_spectra

    n_inside = int(np.sum(inside_contour))
    if n_inside == 0:
        return None

    n_valid = int(np.sum(valid_pixels))
    coverage = float(n_valid) / n_inside
    if coverage < min_coverage:
        return None

    return {
        "rrs": flattened_rrs[valid_pixels],
        "lon": grid_longitudes[valid_pixels],
        "lat": grid_latitudes[valid_pixels],
        "coverage": coverage,
    }


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

        for obs_idx in range(len(times)):
            if virtuals[obs_idx]:
                continue

            day = PET_EPOCH + dt.timedelta(days=int(times[obs_idx]))
            center_lon = float((center_lons[obs_idx] + 180) % 360 - 180)
            center_lat = float(center_lats[obs_idx])
            if not in_subset(center_lon, center_lat, day, region, date_range):
                continue

            obs = EddyObs(
                track_id=int(track_id),
                polarity=polarity,
                contour_lon=(contour_lons[obs_idx] + 180) % 360 - 180,
                contour_lat=contour_lats[obs_idx],
                center_lon=center_lon,
                center_lat=center_lat,
            )
            date_index[day].append(obs)

    return date_index


def parse_pace_date(filename: str) -> dt.date | None:
    """Extract date from a daily PACE L3 filename."""
    match = PACE_DAILY_RE.search(filename)
    if match is None:
        return None
    return dt.datetime.strptime(match.group(1), "%Y%m%d").date()


def parse_pace_date_range(filename: str) -> tuple[dt.date, dt.date] | tuple[None, None]:
    """Extract start/end dates from an 8-day composite filename."""
    match = PACE_8DAY_RE.search(filename)
    if match is None:
        return None, None
    start = dt.datetime.strptime(match.group(1), "%Y%m%d").date()
    end = dt.datetime.strptime(match.group(2), "%Y%m%d").date()
    return start, end


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    return parser.parse_args()


def main(experiment: str | None = None) -> None:
    """Collocate PACE observations and write one Parquet file per tracked eddy."""
    if experiment is None:
        experiment = _parse_args().experiment

    cfg = load_config(experiment)
    collocation_cfg = cfg["collocate_pace"]
    pace_dir = resolve_data_dir(cfg, "pace_dir")
    min_coverage = collocation_cfg["min_coverage"]
    configured_track_ids = collocation_cfg.get("track_ids")
    track_ids = set(configured_track_ids) if configured_track_ids else None
    temporal_resolution = collocation_cfg.get("temporal_resolution", "DAY")
    region = collocation_cfg.get("region")
    date_range = parse_date_range(collocation_cfg.get("date_range"))
    track_dirs = {
        polarity: resolve_output_dir(experiment, "eddy_track", polarity)
        for polarity in ("cyclone", "anticyclone")
    }
    output_dirs = {
        polarity: resolve_output_dir(experiment, "collocate_pace", polarity)
        for polarity in ("cyclone", "anticyclone")
    }

    date_index: dict[dt.date, list[EddyObs]] = defaultdict(list)

    for polarity, track_dir in track_dirs.items():
        zarr_path = track_dir / f"{track_dir.name}_tracks.zarr"
        if not zarr_path.exists():
            print(
                f"polarity: {polarity}\n"
                "status: skipped\n"
                "reason: no_tracks_zarr\n"
                f"tracks_path: {zarr_path}"
            )
            continue

        tracked = TrackEddiesObservations.load_file(str(zarr_path))
        n_tracks = len(np.unique(tracked.track))

        polarity_date_index = build_date_eddy_index(
            tracked, polarity, track_ids, region, date_range
        )
        for day, observations in polarity_date_index.items():
            date_index[day].extend(observations)

        n_observations = sum(
            len(observations) for observations in polarity_date_index.values()
        )
        n_indexed_tracks = len(
            {
                observation.track_id
                for observations in polarity_date_index.values()
                for observation in observations
            }
        )
        print(
            f"polarity: {polarity}\n"
            f"indexed_tracks: {n_indexed_tracks}\n"
            f"total_tracks: {n_tracks}\n"
            f"indexed_observations: {n_observations}"
        )

    if not date_index:
        print(
            "status: skipped\n"
            "reason: no_eddy_observations"
        )
        return

    pace_files = sorted(pace_dir.glob("*.nc"))
    if not pace_files:
        print(
            "status: skipped\n"
            "reason: no_pace_files\n"
            f"pace_dir: {pace_dir}"
        )
        return

    print(
        f"pace_files: {len(pace_files)}\n"
        f"unique_eddy_dates: {len(date_index)}"
    )

    with xr.open_dataset(pace_files[0]) as sample:
        wavelengths = sample.coords["wavelength"].values.astype(int)
    rrs_columns = [f"Rrs_{wavelength}" for wavelength in wavelengths]

    rows_by_eddy: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    n_matched_files = 0

    for pace_path in pace_files:
        if temporal_resolution == "8D":
            window_start, window_end = parse_pace_date_range(pace_path.name)
            if window_start is None:
                continue
            matched_eddies = collect_eddies_for_window(
                date_index, window_start, window_end
            )
            representative_date = window_start + (window_end - window_start) / 2
            date_label = f"{window_start}..{window_end}"
        else:
            representative_date = parse_pace_date(pace_path.name)
            if representative_date is None:
                continue
            matched_eddies = date_index.get(representative_date, [])
            date_label = str(representative_date)

        if not matched_eddies:
            continue

        n_matched_files += 1

        try:
            with xr.open_dataset(pace_path) as dataset:
                longitudes = dataset["lon"].values
                latitudes = dataset["lat"].values
                rrs = dataset["Rrs"].values
        except OSError as exc:
            print(
                f"input_file: {pace_path.name}\n"
                "status: skipped\n"
                f"error: {exc}"
            )
            continue

        for eddy in matched_eddies:
            result = collocate_one_observation(
                longitudes,
                latitudes,
                rrs,
                eddy.contour_lon,
                eddy.contour_lat,
                min_coverage=min_coverage,
            )
            if result is None:
                continue

            n_pixels = len(result["lon"])
            days_since_pet_epoch = (representative_date - PET_EPOCH).days
            rows = np.column_stack(
                [
                    np.full(n_pixels, eddy.track_id),
                    np.full(n_pixels, days_since_pet_epoch),
                    result["lon"],
                    result["lat"],
                    np.full(n_pixels, eddy.center_lon),
                    np.full(n_pixels, eddy.center_lat),
                    np.full(n_pixels, result["coverage"]),
                    result["rrs"],
                ]
            )
            rows_by_eddy[(eddy.track_id, eddy.polarity)].append(rows)

            print(
                f"date_window: {date_label}\n"
                f"polarity: {eddy.polarity}\n"
                f"track_id: {eddy.track_id}\n"
                f"pixels: {n_pixels}\n"
                f"coverage: {result['coverage']:.2f}"
            )

    print(
        f"matched_pace_files: {n_matched_files}\n"
        f"total_pace_files: {len(pace_files)}"
    )

    columns = METADATA_COLS + rrs_columns

    n_written = 0
    for (track_id, polarity), row_chunks in sorted(rows_by_eddy.items()):
        observations = pd.DataFrame(np.vstack(row_chunks), columns=columns)
        observations["track_id"] = observations["track_id"].astype(int)
        observations["date"] = (
            pd.Timestamp("1950-01-01")
            + pd.to_timedelta(observations["date"], unit="D")
        )

        out_dir = output_dirs[polarity]
        out_path = out_dir / f"eddy_{track_id}_rrs.parquet"
        observations.to_parquet(out_path, index=False)
        n_written += 1

        n_dates = observations["date"].nunique()
        print(
            f"output_file: {out_path.name}\n"
            f"pixels_written: {len(observations)}\n"
            f"dates: {n_dates}"
        )

    print(
        "status: complete\n"
        f"eddy_files_written: {n_written}"
    )


if __name__ == "__main__":
    main()
