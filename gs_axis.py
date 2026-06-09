"""
Find the Gulf Stream jet-core axis per date from SWOT SSH, and classify each
eddy track's movement relative to it.

The axis is the line of maximum surface-current speed across the Gulf Stream
latitude band, smoothed along longitude. Movement (NN/NS/SN/SS) compares an
eddy's side of the axis (north/south) at birth vs death.

Outputs to silver/gs_axis/:
  - axis.parquet     one row per (date, longitude): the core latitude
  - movement.parquet one row per (polarity, track_id): movement class + sides
"""

import argparse
import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from py_eddy_tracker.observations.tracking import TrackEddiesObservations

from utils.config import load_config, resolve_data_dir, resolve_output_dir

parser = argparse.ArgumentParser()
parser.add_argument("experiment")
args = parser.parse_args()

cfg = load_config(args.experiment)

SWOT_DIR = resolve_data_dir(cfg, "swot_dir")
OUT_DIR = resolve_output_dir(args.experiment, "gs_axis")
CYCLONE_TRACK_DIR = resolve_output_dir(args.experiment, "eddy_track", "cyclone")
ANTICYCLONE_TRACK_DIR = resolve_output_dir(args.experiment, "eddy_track", "anticyclone")

PET_EPOCH = dt.date(1950, 1, 1)
# The Gulf Stream core stays well inside this latitude band within the ROI;
# restricting the search keeps the line off coastal/subpolar currents.
GS_LAT_BAND = (32.0, 43.0)
MIN_CORE_SPEED = 0.3   # m/s; a column with a slower max has no clear jet
SMOOTH_WINDOW = 9      # longitudes; along-axis median+mean smoothing
KM_PER_DEG_LAT = 111.0


def parse_file_date(fp: Path) -> dt.date:
    return dt.datetime.strptime(re.search(r"\d{8}", fp.name).group(), "%Y%m%d").date()


def axis_for_file(fp: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Jet-core latitude at each longitude for one SWOT day.

    Returns (lon, core_lat), both 1D over longitude; core_lat is NaN where no
    jet is found (max speed below threshold or all-NaN column).
    """
    with xr.open_dataset(fp) as ds:
        if "time" in ds.ugos.dims:
            ds = ds.isel(time=0)
        lon = ds.longitude.to_numpy()
        lat = ds.latitude.to_numpy()
        speed = np.hypot(ds.ugos.to_numpy(), ds.vgos.to_numpy())  # (lat, lon)

    in_band = (lat >= GS_LAT_BAND[0]) & (lat <= GS_LAT_BAND[1])
    lat_band = lat[in_band]
    speed_band = speed[in_band, :]

    core_lat = np.full(lon.shape, np.nan)
    for k in range(lon.size):
        col = speed_band[:, k]
        if not np.any(np.isfinite(col)):
            continue
        i_max = np.nanargmax(col)
        if col[i_max] >= MIN_CORE_SPEED:
            core_lat[k] = lat_band[i_max]

    return lon, _smooth_along(core_lat)


def _smooth_along(core_lat: np.ndarray) -> np.ndarray:
    """Fill short gaps, then median+mean smooth the core latitude vs longitude."""
    s = pd.Series(core_lat).interpolate(limit=5, limit_area="inside")
    s = s.rolling(SMOOTH_WINDOW, center=True, min_periods=1).median()
    s = s.rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()
    return s.to_numpy()


def signed_distance_km(
    axis_lon: np.ndarray, axis_lat: np.ndarray, center_lon: float, center_lat: float
) -> tuple[float, str]:
    """
    Meridional distance from an eddy center to the axis at the eddy's longitude.

    Positive (side 'N') means north of the jet, negative ('S') south. Returns
    (nan, '') if the center sits outside the axis's longitude span or the axis
    has too few points that day.
    """
    finite = np.isfinite(axis_lat)
    if finite.sum() < 2:
        return np.nan, ""
    axis_lat_here = np.interp(
        center_lon, axis_lon[finite], axis_lat[finite], left=np.nan, right=np.nan
    )
    if not np.isfinite(axis_lat_here):
        return np.nan, ""
    dist = (center_lat - axis_lat_here) * KM_PER_DEG_LAT
    return float(dist), ("N" if dist >= 0 else "S")


def load_track_observations() -> pd.DataFrame:
    """
    Non-virtual eddy observations across both polarities, one row each.

    Columns: polarity, track_id, date, center_lon (-180/180), center_lat.
    """
    frames = []
    for polarity, track_dir in [
        ("cyclone", CYCLONE_TRACK_DIR),
        ("anticyclone", ANTICYCLONE_TRACK_DIR),
    ]:
        zarr_path = track_dir / f"{track_dir.name}_tracks.zarr"
        tracked = TrackEddiesObservations.load_file(str(zarr_path))
        keep = ~tracked.virtual.astype(bool)
        days = [PET_EPOCH + dt.timedelta(days=int(t)) for t in tracked.time[keep]]
        frames.append(pd.DataFrame({
            "polarity": polarity,
            "track_id": tracked.track[keep].astype(int),
            "date": pd.to_datetime(days),
            "center_lon": (tracked.longitude[keep] + 180) % 360 - 180,
            "center_lat": tracked.latitude[keep],
        }))
    return pd.concat(frames, ignore_index=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Per-date axis from every SWOT day
    swot_files = sorted(SWOT_DIR.glob("*.nc"))
    print(f"Computing jet-core axis for {len(swot_files)} SWOT days...")
    axis_rows = []
    axis_by_date: dict[dt.date, tuple[np.ndarray, np.ndarray]] = {}
    for fp in swot_files:
        date = parse_file_date(fp)
        lon, core_lat = axis_for_file(fp)
        axis_by_date[date] = (lon, core_lat)
        axis_rows.append(pd.DataFrame({"date": date, "lon": lon, "core_lat": core_lat}))
    axis_df = pd.concat(axis_rows, ignore_index=True)
    axis_df["date"] = pd.to_datetime(axis_df["date"])
    axis_df.to_parquet(OUT_DIR / "axis.parquet", index=False)
    found = axis_df.dropna(subset=["core_lat"])
    print(f"Wrote axis.parquet: median core latitude {found['core_lat'].median():.2f} N")

    # Movement per track: side of the axis at birth vs death
    obs = load_track_observations()
    movement_rows = []
    for (polarity, track_id), grp in obs.groupby(["polarity", "track_id"]):
        grp = grp.sort_values("date")
        birth, death = grp.iloc[0], grp.iloc[-1]
        _, birth_side = _axis_side(axis_by_date, birth)
        _, death_side = _axis_side(axis_by_date, death)
        movement_rows.append({
            "polarity": polarity,
            "track_id": track_id,
            "birth_date": birth["date"],
            "death_date": death["date"],
            "birth_side": birth_side,
            "death_side": death_side,
            "movement": (birth_side + death_side) if birth_side and death_side else "",
        })
    movement_df = pd.DataFrame(movement_rows)
    movement_df.to_parquet(OUT_DIR / "movement.parquet", index=False)
    counts = movement_df["movement"].replace("", "unknown").value_counts().to_dict()
    print(f"Wrote movement.parquet: {len(movement_df)} tracks, classes {counts}")


def _axis_side(axis_by_date, row) -> tuple[float, str]:
    """Signed distance + side for one observation, using its date's axis."""
    axis = axis_by_date.get(row["date"].date())
    if axis is None:
        return np.nan, ""
    return signed_distance_km(axis[0], axis[1], row["center_lon"], row["center_lat"])


if __name__ == "__main__":
    main()
