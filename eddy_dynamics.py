"""
Compute per-eddy dynamical diagnostics from SWOT.

The SWOT relative_vorticity field is normalized by Coriolis in the DUACS/MIOST
files used here, so it is written as Rossby number (zeta/f). Outputs one
dynamics.parquet per polarity under silver/eddy_dynamics/.
"""

import argparse
import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path as MplPath
from scipy.interpolate import RegularGridInterpolator

from utils.config import load_config, resolve_data_dir, resolve_output_dir

PET_EPOCH = dt.date(1950, 1, 1)
SWOT_DATE_RE = re.compile(r"\d{8}")
SWOT_SEARCH_DAYS = 4
DYNAMICS_COLUMNS = [
    "polarity",
    "track_id",
    "date",
    "center_lon",
    "center_lat",
    "rossby_center",
    "rossby_mean",
    "rossby_abs_mean",
    "rossby_min",
    "rossby_max",
    "n_rossby_pixels",
]


def parse_file_date(fp: Path) -> dt.date:
    return dt.datetime.strptime(SWOT_DATE_RE.search(fp.name).group(), "%Y%m%d").date()


def swot_files_by_date(swot_dir: Path) -> dict[dt.date, Path]:
    return {parse_file_date(fp): fp for fp in sorted(swot_dir.glob("*.nc"))}


def nearest_swot_file(files: dict[dt.date, Path], target: dt.date) -> Path | None:
    """SWOT file on target, else the closest within SWOT_SEARCH_DAYS."""
    for delta in range(SWOT_SEARCH_DAYS + 1):
        for day in (target - dt.timedelta(delta), target + dt.timedelta(delta)):
            if day in files:
                return files[day]
    return None


def track_observations_to_frame(tracked, polarity: str) -> pd.DataFrame:
    """Non-virtual observations from one py-eddy-tracker object."""
    keep = ~tracked.virtual.astype(bool)
    days = [PET_EPOCH + dt.timedelta(days=int(t)) for t in tracked.time[keep]]
    contour_lon = (tracked.contour_lon_e[keep] + 180) % 360 - 180
    contour_lat = tracked.contour_lat_e[keep]
    return pd.DataFrame({
        "polarity": polarity,
        "track_id": tracked.track[keep].astype(int),
        "date": pd.to_datetime(days),
        "center_lon": (tracked.longitude[keep] + 180) % 360 - 180,
        "center_lat": tracked.latitude[keep],
        "contour_lon": list(contour_lon),
        "contour_lat": list(contour_lat),
    })


def load_track_observations(experiment: str) -> pd.DataFrame:
    from py_eddy_tracker.observations.tracking import TrackEddiesObservations

    frames = []
    for polarity in ("cyclone", "anticyclone"):
        track_dir = resolve_output_dir(experiment, "eddy_track", polarity)
        tracked = TrackEddiesObservations.load_file(str(track_dir / f"{polarity}_tracks.zarr"))
        frames.append(track_observations_to_frame(tracked, polarity))
    return pd.concat(frames, ignore_index=True)


def load_rossby_field(swot_fp: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lon, lat, and normalized relative vorticity (zeta/f) for one SWOT file."""
    with xr.open_dataset(swot_fp) as ds:
        if "time" in ds["relative_vorticity"].dims:
            ds = ds.isel(time=0)
        lon = ds["longitude"].to_numpy()
        lat = ds["latitude"].to_numpy()
        rossby = ds["relative_vorticity"].to_numpy()
    return lon, lat, rossby


def rossby_stats(
    rossby: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    contour_lon: np.ndarray,
    contour_lat: np.ndarray,
    center_lon: float,
    center_lat: float,
) -> dict:
    """Rossby number at the eddy center plus summary stats inside its contour."""
    interp = RegularGridInterpolator((lat, lon), rossby, bounds_error=False, fill_value=np.nan)
    center = float(interp([[center_lat, center_lon]])[0])

    lon2d, lat2d = np.meshgrid(lon, lat)
    points = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    inside = MplPath(np.column_stack([contour_lon, contour_lat])).contains_points(points)
    values = rossby.ravel()[inside]
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {
            "rossby_center": center,
            "rossby_mean": np.nan,
            "rossby_abs_mean": np.nan,
            "rossby_min": np.nan,
            "rossby_max": np.nan,
            "n_rossby_pixels": 0,
        }

    return {
        "rossby_center": center,
        "rossby_mean": float(values.mean()),
        "rossby_abs_mean": float(np.abs(values).mean()),
        "rossby_min": float(values.min()),
        "rossby_max": float(values.max()),
        "n_rossby_pixels": int(values.size),
    }


def build_dynamics(obs: pd.DataFrame, swot_files: dict[dt.date, Path]) -> pd.DataFrame:
    rows = []
    for date, grp in obs.groupby("date"):
        swot_fp = nearest_swot_file(swot_files, pd.Timestamp(date).date())
        if swot_fp is None:
            continue
        lon, lat, rossby = load_rossby_field(swot_fp)
        for row in grp.itertuples(index=False):
            stats = rossby_stats(
                rossby,
                lon,
                lat,
                row.contour_lon,
                row.contour_lat,
                row.center_lon,
                row.center_lat,
            )
            rows.append({
                "polarity": row.polarity,
                "track_id": int(row.track_id),
                "date": row.date,
                "center_lon": float(row.center_lon),
                "center_lat": float(row.center_lat),
                **stats,
            })
    return pd.DataFrame(rows, columns=DYNAMICS_COLUMNS)


def write_dynamics(experiment: str, dynamics: pd.DataFrame) -> None:
    for polarity in ("cyclone", "anticyclone"):
        out_dir = resolve_output_dir(experiment, "eddy_dynamics", polarity)
        out = dynamics[dynamics["polarity"] == polarity].copy()
        out.to_parquet(out_dir / "dynamics.parquet", index=False)
        print(f"Wrote {out_dir / 'dynamics.parquet'}  ({len(out)} rows)")


def main(experiment: str | None = None):
    if experiment is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("experiment")
        args = parser.parse_args()
        experiment = args.experiment

    cfg = load_config(experiment)
    swot_files = swot_files_by_date(resolve_data_dir(cfg, "swot_dir"))
    obs = load_track_observations(experiment)
    print(f"Computing Rossby diagnostics for {len(obs)} eddy observations...")
    dynamics = build_dynamics(obs, swot_files)
    write_dynamics(experiment, dynamics)


if __name__ == "__main__":
    main()
