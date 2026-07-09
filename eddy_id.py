"""
Identify eddies in daily SWOT L4 SSH files in parallel.

For each daily NetCDF, subsets to the configured lon/lat region, applies a
Bessel high-pass filter on ADT, and runs PET contour-based identification.
Writes per-day cyclonic and anticyclonic eddy observation files to eddy_id/.
"""

import argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import re
import xarray as xr
import numpy as np

from utils.config import load_config, resolve_data_dir, resolve_output_dir

# Correspondances takes a sorted list of files, so naming should be in YYYY-MM-DD format.
# Use separate naming prefixes for cyclonic/anticyclonic

def parse_file_datetime(local_fp: Path) -> datetime:
    match = re.search(r'(\d{8})', local_fp.name)
    if not match:
        raise ValueError(f"No 8-digit date in filename: {local_fp.name}")
    return datetime.strptime(match.group(1), '%Y%m%d')


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
    local_fp: Path,
    date: datetime,
    out_anticyclone_path: Path,
    out_cyclone_path: Path,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    bessel_wavelength: float,
    id_step: float,
    id_shape_error: float,
) -> tuple[Path, Path]:
    """
    Identify eddies in a single SWOT L4 SSH file.

    Subsets to region, applies Bessel high-pass filter, then runs
    PET's contour-based identification on ADT.
    """
    if out_anticyclone_path.exists() and out_cyclone_path.exists():
        return out_anticyclone_path, out_cyclone_path

    from utils.py_eddy_tracker.dataset.grid import RegularGridDataset
    from utils.py_eddy_tracker.observations.observation import EddiesObservations

    # Read coordinates first to compute index slices, then re-open via
    # RegularGridDataset. The double open is necessary because PET's
    # RegularGridDataset takes a filename + index dict, not an open dataset.
    with xr.open_dataset(local_fp) as ds:
        lon_min, lon_max = lon_range
        lat_min, lat_max = lat_range

        lon = ds.coords['longitude'].to_numpy()
        lat = ds.coords['latitude'].to_numpy()

    i_lon_min = np.searchsorted(lon, lon_min, 'left')
    i_lon_max = np.searchsorted(lon, lon_max, 'right')
    i_lat_min = np.searchsorted(lat, lat_min, 'left')
    i_lat_max = np.searchsorted(lat, lat_max, 'right')

    grid = RegularGridDataset(
        filename=local_fp,
        x_name='longitude',
        y_name='latitude',
        indexs={
            'time': 0,
            'longitude': slice(i_lon_min, i_lon_max),
            'latitude': slice(i_lat_min, i_lat_max)
        }
    )

    # Apply bessel high pass filter
    grid.bessel_high_filter('adt', bessel_wavelength)

    # Perform eddy id
    a, c = grid.eddy_identification(
        grid_height='adt',
        uname='ugos',
        vname='vgos',
        date=date,
        step=id_step,
        shape_error=id_shape_error,
    )
    if not isinstance(a, EddiesObservations) or not isinstance(c, EddiesObservations):
        raise ValueError("Some step in eddy identification went wrong")

    a.write_file(filename=str(out_anticyclone_path))
    c.write_file(filename=str(out_cyclone_path))
    return out_anticyclone_path, out_cyclone_path


def main(experiment: str | None = None):
    if experiment is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("experiment")
        args = parser.parse_args()
        experiment = args.experiment

    cfg = load_config(experiment)
    lon_range = tuple(cfg["base"]["region"]["lon_range"])
    lat_range = tuple(cfg["base"]["region"]["lat_range"])
    data_dir = resolve_data_dir(cfg, "swot_dir")
    anticyclone_dir = resolve_output_dir(experiment, "eddy_id", "anticyclone")
    cyclone_dir = resolve_output_dir(experiment, "eddy_id", "cyclone")
    max_workers = cfg["eddy_id"]["max_workers"]
    bessel_wavelength = cfg["eddy_id"]["bessel_high_filter_wavelength"]
    id_step = cfg["eddy_id"]["step"]
    id_shape_error = cfg["eddy_id"]["shape_error"]

    # Ensure output paths
    anticyclone_dir.mkdir(parents=True, exist_ok=True)
    cyclone_dir.mkdir(parents=True, exist_ok=True)

    # Obtain a list of local filepaths
    local_filepaths = list(data_dir.glob("*.nc"))
    if not local_filepaths:
        print(f"No .nc files found in {data_dir}")
        return

    # CPU-bound (Bessel filter + contour detection) — ThreadPoolExecutor would
    # serialize due to the GIL; must use ProcessPoolExecutor here
    n_failed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for fp in local_filepaths:
            date = parse_file_datetime(fp)
            a_path, c_path = resolve_eddy_output_paths(anticyclone_dir, cyclone_dir, date)
            futures[executor.submit(
                identify_one,
                fp,
                date,
                a_path,
                c_path,
                lon_range,
                lat_range,
                bessel_wavelength,
                id_step,
                id_shape_error,
            )] = fp

        for future in as_completed(futures):
            fp = futures[future]
            try:
                print(future.result())
            except Exception as e:
                n_failed += 1
                print(f"FAILED {fp.name}: {e}")

    if n_failed > 0:
        raise RuntimeError(f"{n_failed}/{len(local_filepaths)} files failed in eddy_id")

if __name__ == '__main__':
    main()
