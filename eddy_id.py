"""
Identify eddies in daily SWOT L4 SSH files in parallel.

For each daily NetCDF, subsets to the configured lon/lat region, applies a
Bessel high-pass filter on ADT, and runs PET contour-based identification.
Writes per-day cyclonic and anticyclonic eddy observation files to eddy_id/.
"""

import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from utils.config import load_config, resolve_data_dir, resolve_output_dir

# Correspondances takes a sorted list of files, so naming should be in YYYY-MM-DD format.
# Use separate naming prefixes for cyclonic/anticyclonic


def parse_file_datetime(input_path: Path) -> datetime:
    """Parse the first YYYYMMDD token in a SWOT filename."""
    match = re.search(r"(\d{8})", input_path.name)
    if not match:
        raise ValueError(f"No 8-digit date in filename: {input_path.name}")
    return datetime.strptime(match.group(1), "%Y%m%d")


def resolve_eddy_output_paths(
    anticyclone_dir: Path, cyclone_dir: Path, date: datetime
) -> tuple[Path, Path]:
    """
    Explicit PET output paths for one identification date.

    This intentionally bypasses py-eddy-tracker's CLI filename template path,
    whose upstream ``date.strftime("%(path)s/%(sign_type)s_...")`` handling
    consumes the placeholders before ``write_file()`` can substitute them.
    """
    return (
        anticyclone_dir / f"Anticyclonic_{date:%Y-%m-%d}.nc",
        cyclone_dir / f"Cyclonic_{date:%Y-%m-%d}.nc",
    )


def identify_one(
    input_path: Path,
    date: datetime,
    anticyclone_output_path: Path,
    cyclone_output_path: Path,
    longitude_range: tuple[float, float],
    latitude_range: tuple[float, float],
    bessel_wavelength: float,
    contour_step: float,
    shape_error: int,
) -> tuple[Path, Path]:
    """
    Identify eddies in a single SWOT L4 SSH file.

    Subsets to region, applies Bessel high-pass filter, then runs
    PET's contour-based identification on ADT.
    """
    if anticyclone_output_path.exists() and cyclone_output_path.exists():
        return anticyclone_output_path, cyclone_output_path

    from eddy_tracking.packages.py_eddy_tracker.dataset.grid import (
        RegularGridDataset,
    )
    from eddy_tracking.packages.py_eddy_tracker.observations.observation import (
        EddiesObservations,
    )

    # Read coordinates first to compute index slices, then re-open via
    # RegularGridDataset. The double open is necessary because PET's
    # RegularGridDataset takes a filename + index dict, not an open dataset.
    with xr.open_dataset(input_path) as dataset:
        longitude_min, longitude_max = longitude_range
        latitude_min, latitude_max = latitude_range

        longitude = dataset.coords["longitude"].to_numpy()
        latitude = dataset.coords["latitude"].to_numpy()

        longitude_min_idx = np.searchsorted(longitude, longitude_min, "left")
        longitude_max_idx = np.searchsorted(longitude, longitude_max, "right")
        latitude_min_idx = np.searchsorted(latitude, latitude_min, "left")
        latitude_max_idx = np.searchsorted(latitude, latitude_max, "right")

    grid = RegularGridDataset(
        filename=input_path,
        x_name="longitude",
        y_name="latitude",
        indexs={
            "time": 0,
            "longitude": slice(longitude_min_idx, longitude_max_idx),
            "latitude": slice(latitude_min_idx, latitude_max_idx),
        },
    )

    grid.bessel_high_filter("adt", bessel_wavelength)

    anticyclones, cyclones = grid.eddy_identification(
        grid_height="adt",
        uname="ugos",
        vname="vgos",
        date=date,
        step=contour_step,
        shape_error=shape_error,
    )
    if not isinstance(anticyclones, EddiesObservations) or not isinstance(
        cyclones, EddiesObservations
    ):
        raise ValueError("Some step in eddy identification went wrong")

    anticyclones.write_file(filename=str(anticyclone_output_path))
    cyclones.write_file(filename=str(cyclone_output_path))
    return anticyclone_output_path, cyclone_output_path


def main(experiment: str | None = None) -> None:
    """Identify daily eddies in worker processes and write PET NetCDF files."""
    if experiment is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("experiment")
        args = parser.parse_args()
        experiment = args.experiment
    cfg = load_config(experiment)
    longitude_range = tuple(cfg["base"]["region"]["lon_range"])
    latitude_range = tuple(cfg["base"]["region"]["lat_range"])
    swot_dir = resolve_data_dir(cfg, "swot_dir")
    anticyclone_dir = resolve_output_dir(experiment, "eddy_id", "anticyclone")
    cyclone_dir = resolve_output_dir(experiment, "eddy_id", "cyclone")
    max_workers = cfg["eddy_id"]["max_workers"]
    bessel_wavelength = cfg["eddy_id"]["bessel_high_filter_wavelength"]
    contour_step = cfg["eddy_id"]["step"]
    shape_error = cfg["eddy_id"]["shape_error"]

    input_paths = list(swot_dir.glob("*.nc"))
    if not input_paths:
        print(
            "status: skipped\n"
            "reason: no_nc_files\n"
            f"input_dir: {swot_dir}"
        )
        return

    # CPU-bound (Bessel filter + contour detection) - ThreadPoolExecutor would
    # serialize due to the GIL; must use ProcessPoolExecutor here
    n_failed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for input_path in input_paths:
            date = parse_file_datetime(input_path)
            anticyclone_path, cyclone_path = resolve_eddy_output_paths(
                anticyclone_dir, cyclone_dir, date
            )
            future = executor.submit(
                identify_one,
                input_path,
                date,
                anticyclone_path,
                cyclone_path,
                longitude_range,
                latitude_range,
                bessel_wavelength,
                contour_step,
                shape_error,
            )
            futures[future] = input_path

        for future in as_completed(futures):
            input_path = futures[future]
            try:
                anticyclone_output_path, cyclone_output_path = future.result()
                print(
                    f"input_file: {input_path.name}\n"
                    "status: complete\n"
                    f"anticyclone_output_path: {anticyclone_output_path}\n"
                    f"cyclone_output_path: {cyclone_output_path}"
                )
            except Exception as exc:
                n_failed += 1
                print(
                    "status: failed\n"
                    f"input_file: {input_path.name}\n"
                    f"error: {exc}"
                )

    if n_failed:
        raise RuntimeError(f"{n_failed}/{len(input_paths)} files failed in eddy_id")


if __name__ == "__main__":
    main()
