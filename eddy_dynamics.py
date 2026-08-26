"""
Compute per-eddy dynamical diagnostics from SWOT.

The DUACS/MIOST source variable is named relative_vorticity, but in the files used here it is not raw relative vorticity in s^-1. It is already normalized by the Coriolis parameter, so the stored quantity is Rossby number (Ro = zeta/f). Outputs one dynamics.parquet per polarity under silver/eddy_dynamics/.
"""

import argparse
import datetime as dt
import re
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy.interpolate import RegularGridInterpolator

from utils.config import load_config, resolve_data_dir, resolve_output_dir
from eddy_tracking.utils.subset import load_rossby_field


def find_nearest_swot_file(files: dict[dt.date, Path], target: dt.date) -> Path | None:
    """SWOT file on target, else the closest within 4 days, else None."""
    swot_search_days = 4
    for delta in range(swot_search_days + 1):
        for day in (target - dt.timedelta(delta), target + dt.timedelta(delta)):
            if day in files:
                return files[day]
    return None


def track_observations_to_frame(tracked, polarity: str) -> pd.DataFrame:
    """Non-virtual observations from one py-eddy-tracker object."""
    pet_epoch = dt.date(1950, 1, 1)
    keep = ~tracked.virtual.astype(bool)
    days = [pet_epoch + dt.timedelta(days=int(t)) for t in tracked.time[keep]]
    contour_lon = (tracked.contour_lon_s[keep] + 180) % 360 - 180
    contour_lat = tracked.contour_lat_s[keep]
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
    """Load non-virtual track observations for both polarities."""
    from eddy_tracking.packages.py_eddy_tracker.observations.tracking import (
        TrackEddiesObservations,
    )

    frames = []
    for polarity in ("cyclone", "anticyclone"):
        track_dir = resolve_output_dir(experiment, "eddy_track", polarity)
        tracked = TrackEddiesObservations.load_file(str(track_dir / f"{polarity}_tracks.zarr"))
        frames.append(track_observations_to_frame(tracked, polarity))
    return pd.concat(frames, ignore_index=True)


def compute_rossby_stats(
    rossby_number: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    contour_lon: np.ndarray,
    contour_lat: np.ndarray,
    center_lon: float,
    center_lat: float,
) -> dict:
    """Rossby number at the eddy center plus summary stats inside its contour."""
    interp = RegularGridInterpolator((lat, lon), rossby_number, bounds_error=False, fill_value=np.nan)
    center = float(interp([[center_lat, center_lon]])[0])

    lon2d, lat2d = np.meshgrid(lon, lat)  # (n_lon,) + (n_lat,) -> (n_lat, n_lon) each
    points = np.column_stack([lon2d.ravel(), lat2d.ravel()])  # (n_lat, n_lon) each -> (n_lat*n_lon,) each -> (n_lat*n_lon, 2)
    inside = MplPath(np.column_stack([contour_lon, contour_lat])).contains_points(points)  # contour (n_vertices,) + (n_vertices,) -> (n_vertices, 2)
    values = rossby_number.ravel()[inside]  # (n_lat, n_lon) -> (n_lat*n_lon,) -> (n_inside,)
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
    """Compute Rossby diagnostics for each eddy observation with SWOT coverage."""
    dynamics_columns = [
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
    rows = []
    for date, grp in obs.groupby("date"):
        swot_fp = find_nearest_swot_file(swot_files, pd.Timestamp(date).date())  # pyright: ignore[reportArgumentType]
        if swot_fp is None:
            continue
        lon, lat, rossby_number = load_rossby_field(swot_fp)
        for row in grp.itertuples(index=False):
            stats = compute_rossby_stats(
                rossby_number,
                lon,
                lat,
                row.contour_lon,  # pyright: ignore[reportAttributeAccessIssue]
                row.contour_lat,  # pyright: ignore[reportAttributeAccessIssue]
                row.center_lon,  # pyright: ignore[reportAttributeAccessIssue]
                row.center_lat,  # pyright: ignore[reportAttributeAccessIssue]
            )
            rows.append({
                "polarity": row.polarity,  # pyright: ignore[reportAttributeAccessIssue]
                "track_id": int(row.track_id),  # pyright: ignore[reportAttributeAccessIssue]
                "date": row.date,  # pyright: ignore[reportAttributeAccessIssue]
                "center_lon": float(row.center_lon),  # pyright: ignore[reportAttributeAccessIssue]
                "center_lat": float(row.center_lat),  # pyright: ignore[reportAttributeAccessIssue]
                **stats,
            })
    return pd.DataFrame(rows, columns=dynamics_columns)  # pyright: ignore[reportArgumentType]


def write_dynamics(experiment: str, dynamics: pd.DataFrame) -> None:
    """Write one dynamics Parquet file per polarity."""
    for polarity in ("cyclone", "anticyclone"):
        out_dir = resolve_output_dir(experiment, "eddy_dynamics", polarity)
        out = dynamics[dynamics["polarity"] == polarity].copy()
        out.to_parquet(out_dir / "dynamics.parquet", index=False)
        print(
            f"output_path: {out_dir / 'dynamics.parquet'}\n"
            f"rows_written: {len(out)}"
        )


def main(experiment: str | None = None) -> None:
    """Compute and write Rossby diagnostics for one experiment."""
    if experiment is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("experiment")
        args = parser.parse_args()
        experiment = cast(str, args.experiment)

    cfg = load_config(experiment)
    swot_date_re = re.compile(r"\d{8}")
    swot_files = {
        dt.datetime.strptime(swot_date_re.search(fp.name).group(), "%Y%m%d").date(): fp  # pyright: ignore[reportOptionalMemberAccess]
        for fp in sorted(resolve_data_dir(cfg, "swot_dir").glob("*.nc"))
    }
    obs = load_track_observations(experiment)
    print(
        "status: computing_rossby_diagnostics\n"
        f"eddy_observations: {len(obs)}"
    )
    dynamics = build_dynamics(obs, swot_files)
    write_dynamics(experiment, dynamics)


if __name__ == "__main__":
    main()
