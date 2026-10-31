"""
Collocate PACE L3 Rrs observations with tracked eddy contours.

For each PACE file (daily or 8-day composite), finds all eddies that were detected during the observation period, extracts valid Rrs pixels within max_radius speed radii of each eddy's center or inside its contour, and writes per-eddy Parquet files.

Temporal resolution is set by --temporal-resolution:
  - "DAY": exact date match between PACE file and eddy detection
  - "8D": for each 8-day composite, picks the eddy contour from the day closest to the date range midpoint (since the Rrs is a temporal average)
"""

import argparse
import datetime as dt
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from eddy_tracking.config import METADATA_COLS
from eddy_tracking.preprocess.pace import parse_fn_for_date_range
from eddy_tracking.preprocess.tracks import (
    PET_EPOCH,
    EddyObs,
    build_date_eddy_index,
    collect_eddies_for_date_range,
    load_tracks,
    mask_pixels_inside_contour,
)
from eddy_tracking.utils.cli import write_settings
from eddy_tracking.utils.geography import calculate_dist_to_point
from eddy_tracking.utils.subset import parse_date_range


def collocate_one_observation(
    lon: np.ndarray,
    lat: np.ndarray,
    rrs: np.ndarray,
    eddy: EddyObs,
    min_coverage: float,
    max_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None:
    """
    Return valid Rrs pixels within max_radius speed radii of the eddy center or inside its contour when contour coverage meets the threshold, as (rrs, lon, lat, inside_contour, coverage).

    NASA L3 quality flags are already represented as NaN values in ``rrs``.
    """
    longitude_grid, latitude_grid = np.meshgrid(lon, lat)  # (n_lon,) + (n_lat,) -> (n_lat, n_lon) each
    n_grid_cells = longitude_grid.size
    grid_longitudes = longitude_grid.ravel()  # (n_lat, n_lon) -> (n_lat*n_lon,)
    grid_latitudes = latitude_grid.ravel()  # (n_lat, n_lon) -> (n_lat*n_lon,)
    flattened_rrs = rrs.reshape(n_grid_cells, -1)  # (n_lat, n_lon, n_wavelength) -> (n_lat*n_lon, n_wavelength)

    inside_contour = mask_pixels_inside_contour(lon, lat, eddy.contour_lon, eddy.contour_lat).ravel()
    distance_km = calculate_dist_to_point(
        pd.Series(grid_longitudes), pd.Series(grid_latitudes), eddy.center_lon, eddy.center_lat,
    ).to_numpy()
    inside_disc = distance_km <= max_radius * eddy.radius_km

    finite_spectra = np.all(np.isfinite(flattened_rrs), axis=1)  # (n_lat*n_lon, n_wavelength) -> (n_lat*n_lon,)
    valid_pixels = (inside_contour | inside_disc) & finite_spectra

    n_inside = int(np.sum(inside_contour))
    if n_inside == 0:
        return None

    coverage = float(np.sum(inside_contour & finite_spectra)) / n_inside
    if coverage < min_coverage:
        return None

    return (
        flattened_rrs[valid_pixels],  # (n_lat*n_lon, n_wavelength) -> (n_valid, n_wavelength)
        grid_longitudes[valid_pixels],  # (n_lat*n_lon,) -> (n_valid,)
        grid_latitudes[valid_pixels],  # (n_lat*n_lon,) -> (n_valid,)
        inside_contour[valid_pixels],
        coverage,
    )


def main(argv: list[str]) -> None:
    """Collocate PACE observations and write one Parquet file per tracked eddy."""
    parser = argparse.ArgumentParser(prog="eddy-tracking collocate-pace")
    parser.add_argument("--tracks", type=Path, required=True, help="folder track-eddies wrote")
    parser.add_argument("--pace", type=Path, required=True, help="folder of PACE L3 Rrs files")
    parser.add_argument("--dates", nargs=2, required=True, metavar=("START", "END"), help="ISO dates, both inclusive; eddy-days outside them are left out")
    parser.add_argument("--temporal-resolution", choices=["DAY", "8D"], required=True, help="whether the PACE files are daily maps or 8-day composites")
    parser.add_argument("--min-coverage", type=float, required=True, help="fraction of interior pixels that must hold a finite spectrum")
    parser.add_argument("--max-radius", type=float, required=True, metavar="SPEED_RADII", help="pixels within this many speed radii of the center are kept, besides the interior")
    parser.add_argument("--track-ids", nargs="+", required=True, help="track ids to collocate in both polarities, or all")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    write_settings(args.out, "collocate-pace", args, operational=set())

    pace_dir = args.pace
    track_ids = None if args.track_ids == ["all"] else {int(track_id) for track_id in args.track_ids}
    date_range = parse_date_range(args.dates)
    output_dirs = {
        polarity: args.out / polarity
        for polarity in ("cyclone", "anticyclone")
    }
    for out_dir in output_dirs.values():
        out_dir.mkdir(parents=True, exist_ok=True)

    date_index: dict[dt.date, list[EddyObs]] = defaultdict(list)

    for polarity in ("cyclone", "anticyclone"):
        tracked = load_tracks(args.tracks, polarity)
        n_tracks = len(np.unique(tracked.track))

        polarity_date_index = build_date_eddy_index(
            tracked, polarity, track_ids, date_range=date_range
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
        wavelengths = sample.coords["wavelength"].values
    # SDP reads each band center back from the column name, and OCI centers sit up to 1 nm off whole numbers
    rrs_columns = [f"Rrs_{wavelength:.3f}" for wavelength in wavelengths]

    rows_by_eddy: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    n_matched_files = 0

    for pace_path in pace_files:
        composite_date_range = parse_fn_for_date_range(pace_path.name, args.temporal_resolution)
        if composite_date_range is None:
            continue
        representative_date, date_range_start, date_range_end = composite_date_range
        matched_eddies = collect_eddies_for_date_range(date_index, date_range_start, date_range_end)
        date_label = f"{date_range_start}..{date_range_end}" if date_range_start != date_range_end else str(date_range_start)

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
                eddy,
                min_coverage=args.min_coverage,
                max_radius=args.max_radius,
            )
            if result is None:
                continue
            valid_rrs, valid_lon, valid_lat, valid_inside, coverage = result

            n_pixels = len(valid_lon)
            days_since_pet_epoch = (representative_date - PET_EPOCH).days
            # 9x (n_pixels,) + (n_pixels, n_wavelength) -> (n_pixels, 9 + n_wavelength), matching METADATA_COLS + rrs_columns.
            rows = np.column_stack(
                [
                    np.full(n_pixels, eddy.track_id),
                    np.full(n_pixels, days_since_pet_epoch),
                    valid_lon,
                    valid_lat,
                    np.full(n_pixels, eddy.center_lon),
                    np.full(n_pixels, eddy.center_lat),
                    np.full(n_pixels, eddy.radius_km),
                    valid_inside,
                    np.full(n_pixels, coverage),
                    valid_rrs,
                ]
            )
            rows_by_eddy[(eddy.track_id, eddy.polarity)].append(rows)

            print(
                f"date_range: {date_label}\n"
                f"polarity: {eddy.polarity}\n"
                f"track_id: {eddy.track_id}\n"
                f"pixels: {n_pixels}\n"
                f"interior_pixels: {int(valid_inside.sum())}\n"
                f"coverage: {coverage:.2f}"
            )

    print(
        f"matched_pace_files: {n_matched_files}\n"
        f"total_pace_files: {len(pace_files)}"
    )

    columns = METADATA_COLS + rrs_columns

    n_written = 0
    for (track_id, polarity), row_chunks in sorted(rows_by_eddy.items()):
        # list of (n_pixels_i, 9 + n_wavelength) -> (sum_i n_pixels_i, 9 + n_wavelength)
        observations = pd.DataFrame(np.vstack(row_chunks), columns=columns)  # pyright: ignore[reportArgumentType]
        observations["track_id"] = observations["track_id"].astype(int)
        observations["inside_contour"] = observations["inside_contour"].astype(bool)
        observations["date"] = (
            pd.Timestamp(PET_EPOCH).as_unit("ns")
            + pd.to_timedelta(observations["date"], unit="D")  # pyright: ignore[reportArgumentType, reportCallIssue]
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
