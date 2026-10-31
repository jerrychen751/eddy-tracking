"""
Compute per-eddy dynamical diagnostics from SWOT.

The DUACS/MIOST source variable is named relative_vorticity, but in the files used here it is not raw relative vorticity in s^-1. It is already normalized by the Coriolis parameter, so the stored quantity is Rossby number (Ro = zeta/f). Outputs one dynamics.parquet per polarity under --out.
"""

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from eddy_tracking.preprocess.swot import find_nearest_swot_file, index_swot_files_by_date, load_rossby_field
from eddy_tracking.preprocess.tracks import load_track_observations, mask_pixels_inside_contour
from eddy_tracking.utils.cli import write_settings


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

    inside = mask_pixels_inside_contour(lon, lat, contour_lon, contour_lat)
    values = rossby_number[inside]  # (n_lat, n_lon) -> (n_inside,)
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
        for polarity, track_id, contour_lon, contour_lat, center_lon, center_lat in zip(
            grp["polarity"], grp["track_id"], grp["contour_lon"], grp["contour_lat"], grp["center_lon"], grp["center_lat"],
        ):
            stats = compute_rossby_stats(rossby_number, lon, lat, contour_lon, contour_lat, center_lon, center_lat)
            rows.append({
                "polarity": polarity,
                "track_id": int(track_id),
                "date": date,
                "center_lon": float(center_lon),
                "center_lat": float(center_lat),
                **stats,
            })
    return pd.DataFrame(rows, columns=dynamics_columns)  # pyright: ignore[reportArgumentType]


def write_dynamics(dynamics_dir: Path, dynamics: pd.DataFrame) -> None:
    """Write one dynamics Parquet file per polarity."""
    for polarity in ("cyclone", "anticyclone"):
        out_dir = dynamics_dir / polarity
        out_dir.mkdir(parents=True, exist_ok=True)
        out = dynamics[dynamics["polarity"] == polarity].copy()
        out.to_parquet(out_dir / "dynamics.parquet", index=False)
        print(
            f"output_path: {out_dir / 'dynamics.parquet'}\n"
            f"rows_written: {len(out)}"
        )


def main(argv: list[str]) -> None:
    """Compute and write Rossby diagnostics for every tracked eddy observation."""
    parser = argparse.ArgumentParser(prog="eddy-tracking measure-dynamics")
    parser.add_argument("--ssh", type=Path, required=True, help="folder of daily SSH maps in the AVISO layout")
    parser.add_argument("--tracks", type=Path, required=True, help="folder track-eddies wrote")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    write_settings(args.out, "measure-dynamics", args, operational=set())

    swot_files = index_swot_files_by_date(args.ssh)
    obs = load_track_observations(args.tracks)
    print(
        "status: computing_rossby_diagnostics\n"
        f"eddy_observations: {len(obs)}"
    )
    dynamics = build_dynamics(obs, swot_files)
    write_dynamics(args.out, dynamics)
