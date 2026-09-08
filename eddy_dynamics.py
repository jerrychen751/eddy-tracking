"""
Compute per-eddy dynamical diagnostics from SWOT.

The DUACS/MIOST source variable is named relative_vorticity, but in the files used here it is not raw relative vorticity in s^-1. It is already normalized by the Coriolis parameter, so the stored quantity is Rossby number (Ro = zeta/f). Outputs one dynamics.parquet per polarity under silver/eddy_dynamics/.
"""

import argparse
import datetime as dt
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from eddy_tracking.config import load_config, resolve_data_dir, resolve_output_dir
from eddy_tracking.preprocess.swot import find_nearest_swot_file, index_swot_files_by_date, load_rossby_field
from eddy_tracking.preprocess.tracks import load_track_observations, mask_pixels_inside_contour


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
    swot_files = index_swot_files_by_date(resolve_data_dir(cfg, "swot_dir"))
    obs = load_track_observations(experiment)
    print(
        "status: computing_rossby_diagnostics\n"
        f"eddy_observations: {len(obs)}"
    )
    dynamics = build_dynamics(obs, swot_files)
    write_dynamics(experiment, dynamics)


if __name__ == "__main__":
    main()
