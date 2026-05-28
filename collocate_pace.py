"""
Collocate PACE L3 Rrs observations with tracked eddy contours.

For each PACE file (daily or 8-day composite), finds all eddies that
were detected during the observation period, extracts valid Rrs pixels
within each eddy's contour, and writes per-eddy Parquet files.

Temporal resolution is set via collocate_pace.yaml:
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
from py_eddy_tracker.observations.tracking import TrackEddiesObservations

from utils.config import load_config, resolve_data_dir, resolve_output_dir, METADATA_COLS
from utils.subset import parse_date_range, in_subset

parser = argparse.ArgumentParser()
parser.add_argument("experiment")
args = parser.parse_args()

cfg = load_config(args.experiment, "base.yaml", "collocate_pace.yaml")

PACE_DIR = resolve_data_dir(cfg, "pace_dir")
MIN_COVERAGE = cfg["collocate_pace"]["min_coverage"]
TRACK_IDS = cfg["collocate_pace"].get("track_ids")
TEMPORAL_RES = cfg["collocate_pace"].get("temporal_resolution", "DAY")
REGION = cfg["collocate_pace"].get("region")
DATE_RANGE = parse_date_range(cfg["collocate_pace"].get("date_range"))

CYCLONE_TRACK_DIR = resolve_output_dir(args.experiment, "eddy_track", "cyclone")
ANTICYCLONE_TRACK_DIR = resolve_output_dir(args.experiment, "eddy_track", "anticyclone")

OUT_CYCLONE_DIR = resolve_output_dir(args.experiment, "collocate_pace", "cyclone")
OUT_ANTICYCLONE_DIR = resolve_output_dir(args.experiment, "collocate_pace", "anticyclone")

PET_EPOCH = dt.date(1950, 1, 1)
PACE_DAILY_RE = re.compile(r"PACE_OCI\.(\d{8})\.L3m\.DAY\.RRS\.")
PACE_8DAY_RE = re.compile(r"PACE_OCI\.(\d{8})_(\d{8})\.L3m\.8D\.RRS\.")


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
    Extract QC-filtered Rrs pixels inside one eddy contour from one L3 grid.

    Builds a polygon from the ~50 eddy contour points and tests which grid
    cells fall inside using matplotlib's ray-casting algorithm. L3 QC is
    pre-applied during NASA processing — flagged pixels are NaN.

    Args:
        lon: 1D longitude coordinates from the L3 grid (600,).
        lat: 1D latitude coordinates from the L3 grid (360,).
        rrs: Remote-sensing reflectance array (lat, lon, wavelength) —
            (360, 600, 172) for PACE OCI's 172 hyperspectral bands.
        contour_lon: Eddy contour longitudes (~50 vertices).
        contour_lat: Eddy contour latitudes (~50 vertices).
        min_coverage: Fraction of in-contour grid cells that must have
            valid (non-NaN) Rrs to keep this eddy-date instance.

    Returns a mapping with valid pixels and coverage on success, otherwise
    None if the eddy has no overlapping grid cells or insufficient valid
    coverage.
    """
    lon2d, lat2d = np.meshgrid(lon, lat)
    n_gridcells = lon2d.size # lon * lat
    lon_flat = lon2d.ravel() # lon^2
    lat_flat = lat2d.ravel() # lat^2
    rrs_flat = rrs.reshape(n_gridcells, -1) # (lon*lat, 172)

    polygon = MplPath(np.column_stack([contour_lon, contour_lat]))
    inside = polygon.contains_points(np.column_stack([lon_flat, lat_flat]))

    all_finite = np.all(np.isfinite(rrs_flat), axis=1)
    valid = inside & all_finite

    n_inside = int(np.sum(inside))
    if n_inside == 0:
        return None

    n_valid = int(np.sum(valid))
    coverage = float(n_valid) / n_inside
    if coverage < min_coverage:
        return None

    return {
        "rrs": rrs_flat[valid],
        "lon": lon_flat[valid],
        "lat": lat_flat[valid],
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
    Build an inverted index mapping date → list of EddyObs.

    Scans all non-virtual observations in the tracks zarr. Virtual
    observations (gap-filled by interpolation during tracking) are skipped
    because their contours are estimated, not detected.

    Contour longitudes are converted from PET's 0–360 convention to
    PACE L3's -180/180.
    """
    index: dict[dt.date, list[EddyObs]] = defaultdict(list)

    unique_ids = np.unique(tracked.track)
    for tid in unique_ids:
        if track_ids is not None and tid not in track_ids:
            continue

        mask = tracked.track == tid
        times = tracked.time[mask]
        virtuals = tracked.virtual[mask]
        contour_lons = tracked.contour_lon_e[mask]
        contour_lats = tracked.contour_lat_e[mask]
        center_lons = tracked.longitude[mask]
        center_lats = tracked.latitude[mask]

        for j in range(len(times)):
            if virtuals[j]:
                continue

            day = PET_EPOCH + dt.timedelta(days=int(times[j]))
            center_lon = float((center_lons[j] + 180) % 360 - 180)
            center_lat = float(center_lats[j])
            if not in_subset(center_lon, center_lat, day, region, date_range):
                continue

            obs = EddyObs(
                track_id=int(tid),
                polarity=polarity,
                contour_lon=(contour_lons[j] + 180) % 360 - 180,
                contour_lat=contour_lats[j],
                center_lon=center_lon,
                center_lat=center_lat,
            )
            index[day].append(obs)

    return index


def parse_pace_date(filename: str) -> dt.date | None:
    """Extract date from a daily PACE L3 filename."""
    m = PACE_DAILY_RE.search(filename)
    if m is None:
        return None
    return dt.datetime.strptime(m.group(1), "%Y%m%d").date()


def parse_pace_date_range(filename: str) -> tuple[dt.date, dt.date] | tuple[None, None]:
    """Extract start/end dates from an 8-day composite filename."""
    m = PACE_8DAY_RE.search(filename)
    if m is None:
        return None, None
    start = dt.datetime.strptime(m.group(1), "%Y%m%d").date()
    end = dt.datetime.strptime(m.group(2), "%Y%m%d").date()
    return start, end


def collect_eddies_for_window(
    date_index: dict[dt.date, list["EddyObs"]],
    start: dt.date,
    end: dt.date,
) -> list["EddyObs"]:
    """
    Collect one EddyObs per (track_id, polarity) for an 8-day window.

    An eddy may be detected on multiple days within the window. We keep
    the observation whose date is closest to the window midpoint, since
    the composite Rrs is a temporal average and the midpoint contour
    best represents the eddy's position during the observation.
    """
    midpoint = start + (end - start) / 2
    # (track_id, polarity) → (obs, distance_to_midpoint)
    best: dict[tuple[int, str], tuple[EddyObs, float]] = {}

    day = start
    while day <= end:
        for obs in date_index.get(day, []):
            key = (obs.track_id, obs.polarity)
            dist = abs((day - midpoint).days)
            if key not in best or dist < best[key][1]:
                best[key] = (obs, dist)
        day += dt.timedelta(days=1)

    return [obs for obs, _ in best.values()]


def main():
    track_ids_set = set(TRACK_IDS) if TRACK_IDS else None

    # Load tracks and build combined date index across both polarities
    date_index: dict[dt.date, list[EddyObs]] = defaultdict(list)

    for polarity, track_dir in [
        ("cyclone", CYCLONE_TRACK_DIR),
        ("anticyclone", ANTICYCLONE_TRACK_DIR),
    ]:
        zarr_path = track_dir / f"{track_dir.name}_tracks.zarr"
        if not zarr_path.exists():
            print(f"[{polarity}] No tracks zarr at {zarr_path}, skipping")
            continue

        tracked = TrackEddiesObservations.load_file(str(zarr_path))
        n_tracks = len(np.unique(tracked.track))

        pol_index = build_date_eddy_index(
            tracked, polarity, track_ids_set, REGION, DATE_RANGE
        )
        for day, obs_list in pol_index.items():
            date_index[day].extend(obs_list)

        n_obs = sum(len(v) for v in pol_index.values())
        n_indexed = len({
            obs.track_id for obs_list in pol_index.values() for obs in obs_list
        })
        print(f"[{polarity}] {n_indexed}/{n_tracks} tracks, {n_obs} observations indexed")

    if not date_index:
        print("No eddy observations found. Nothing to do.")
        return

    # Discover PACE files and read wavelength grid from the first one
    pace_files = sorted(PACE_DIR.glob("*.nc"))
    if not pace_files:
        print(f"No PACE files found in {PACE_DIR}")
        return

    print(f"{len(pace_files)} PACE files, {len(date_index)} unique eddy-dates")

    with xr.open_dataset(pace_files[0]) as sample:
        wavelengths = sample.coords["wavelength"].values.astype(int)
    rrs_cols = [f"Rrs_{w}" for w in wavelengths]

    # Per-eddy accumulator: (track_id, polarity) → list of row arrays
    accum: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    n_matched = 0

    for fp in pace_files:
        # Parse date(s) from filename based on temporal resolution
        if TEMPORAL_RES == "8D":
            win_start, win_end = parse_pace_date_range(fp.name)
            if win_start is None:
                continue
            eddies_matched = collect_eddies_for_window(date_index, win_start, win_end)
            # Use window midpoint as the representative date for output
            repr_date = win_start + (win_end - win_start) / 2
            date_label = f"{win_start}..{win_end}"
        else:
            repr_date = parse_pace_date(fp.name)
            if repr_date is None:
                continue
            eddies_matched = date_index.get(repr_date, [])
            date_label = str(repr_date)

        if not eddies_matched:
            continue

        n_matched += 1

        try:
            with xr.open_dataset(fp) as ds:
                lon = ds["lon"].values
                lat = ds["lat"].values
                rrs = ds["Rrs"].values
        except OSError as e:
            print(f"Skipping {fp.name}: {e}")
            continue

        for eddy in eddies_matched:
            result = collocate_one_observation(
                lon, lat, rrs,
                eddy.contour_lon, eddy.contour_lat,
                min_coverage=MIN_COVERAGE,
            )
            if result is None:
                continue

            n_pix = len(result["lon"])
            pet_day = (repr_date - PET_EPOCH).days
            chunk = np.column_stack([
                np.full(n_pix, eddy.track_id),
                np.full(n_pix, pet_day),
                result["lon"],
                result["lat"],
                np.full(n_pix, eddy.center_lon),
                np.full(n_pix, eddy.center_lat),
                np.full(n_pix, result["coverage"]),
                result["rrs"],
            ])
            accum[(eddy.track_id, eddy.polarity)].append(chunk)

            print(
                f"{date_label} | {eddy.polarity} #{eddy.track_id}: "
                f"{n_pix} pixels, coverage={result['coverage']:.2f}"
            )

    print(f"Matched {n_matched}/{len(pace_files)} PACE files to eddy dates")

    columns = METADATA_COLS + rrs_cols

    n_written = 0
    for (track_id, polarity), chunks in sorted(accum.items()):
        all_data = np.vstack(chunks)
        df = pd.DataFrame(all_data, columns=columns)
        df["track_id"] = df["track_id"].astype(int)
        df["date"] = (
            pd.Timestamp("1950-01-01") + pd.to_timedelta(df["date"], unit="D")
        )

        out_dir = OUT_CYCLONE_DIR if polarity == "cyclone" else OUT_ANTICYCLONE_DIR
        out_path = out_dir / f"eddy_{track_id}_rrs.parquet"
        df.to_parquet(out_path, index=False)
        n_written += 1

        n_dates = df["date"].nunique()
        print(f"Wrote {out_path.name}: {len(df)} pixels across {n_dates} dates")

    print(f"Done. {n_written} eddy files written.")


if __name__ == "__main__":
    main()
