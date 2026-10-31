"""
Identify eddies in daily SWOT L4 SSH files in parallel.

For each daily NetCDF, subsets to the --lon/--lat region, applies a Bessel high-pass filter on ADT, and runs PET contour-based identification. Writes per-day cyclonic and anticyclonic eddy observation files to cyclone/ and anticyclone/ under --out.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from eddy_tracking.preprocess.swot import index_swot_files_by_date
from eddy_tracking.utils.cli import add_region_arguments, write_settings


def resolve_eddy_output_paths(
    anticyclone_dir: Path, cyclone_dir: Path, date: datetime
) -> tuple[Path, Path]:
    """
    Explicit PET output paths for one identification date.

    This intentionally bypasses py-eddy-tracker's CLI filename template path, whose upstream ``date.strftime("%(path)s/%(sign_type)s_...")`` handling consumes the placeholders before ``write_file()`` can substitute them.
    """
    # eddy_track.py feeds these files to PET Correspondances in sorted order, so the date must sort chronologically as a string: Anticyclonic_2024-01-15.nc.
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
    min_amplitude: float,
) -> tuple[Path, Path]:
    """
    Identify eddies in a single SWOT L4 SSH file.

    Subsets to region, applies Bessel high-pass filter, then runs PET's contour-based identification on ADT.
    """
    if anticyclone_output_path.exists() and cyclone_output_path.exists():
        return anticyclone_output_path, cyclone_output_path

    from eddy_tracking.packages.py_eddy_tracker.dataset.grid import (
        RegularGridDataset,
    )
    from eddy_tracking.packages.py_eddy_tracker.observations.observation import (
        EddiesObservations,
    )

    # Read coordinates first to compute index slices, then re-open via RegularGridDataset.
    # The double open is necessary because PET's RegularGridDataset takes a filename + index dict, not an open dataset.
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
        nb_step_min=min_amplitude / contour_step,
    )
    if not isinstance(anticyclones, EddiesObservations) or not isinstance(
        cyclones, EddiesObservations
    ):
        raise ValueError("Some step in eddy identification went wrong")

    anticyclones.write_file(filename=str(anticyclone_output_path))
    cyclones.write_file(filename=str(cyclone_output_path))
    return anticyclone_output_path, cyclone_output_path


def main(argv: list[str]) -> None:
    """Identify daily eddies in worker processes and write PET NetCDF files."""
    parser = argparse.ArgumentParser(prog="eddy-tracking identify-eddies")
    parser.add_argument("--ssh", type=Path, required=True, help="folder of daily SSH maps in the AVISO layout")
    add_region_arguments(parser)
    parser.add_argument("--bessel-km", type=float, required=True, help="high-pass cutoff wavelength of the Bessel filter on ADT")
    parser.add_argument("--step-m", type=float, required=True, help="SSH contour step")
    parser.add_argument("--shape-error", type=int, required=True, help="maximum percent deviation of a contour from a circle")
    parser.add_argument("--min-amplitude-m", type=float, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    write_settings(args.out, "identify-eddies", args, operational={"workers"})

    longitude_range = tuple(args.lon)
    latitude_range = tuple(args.lat)
    anticyclone_dir = args.out / "anticyclone"
    cyclone_dir = args.out / "cyclone"
    anticyclone_dir.mkdir(parents=True, exist_ok=True)
    cyclone_dir.mkdir(parents=True, exist_ok=True)
    start, end = (datetime.fromisoformat(day).date() for day in args.dates)

    swot_files = {day: path for day, path in index_swot_files_by_date(args.ssh).items() if start <= day <= end}
    if not swot_files:
        print(
            "status: skipped\n"
            "reason: no_nc_files\n"
            f"input_dir: {args.ssh}"
        )
        return

    # CPU-bound (Bessel filter + contour detection) - ThreadPoolExecutor would serialize due to the GIL; must use ProcessPoolExecutor here.
    n_failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for day, input_path in swot_files.items():
            date = datetime(day.year, day.month, day.day)
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
                args.bessel_km,
                args.step_m,
                args.shape_error,
                args.min_amplitude_m,
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
        raise RuntimeError(f"{n_failed}/{len(swot_files)} files failed in eddy_id")

