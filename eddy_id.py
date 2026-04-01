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
from py_eddy_tracker.dataset.grid import RegularGridDataset
from py_eddy_tracker.observations.observation import EddiesObservations

from utils.config import load_config, resolve_data_dir, resolve_output_dir

# Correspondances takes a sorted list of files, so naming should be in YYYY-MM-DD format.
# Use separate naming prefixes for cyclonic/anticyclonic

parser = argparse.ArgumentParser()
parser.add_argument("experiment")
args = parser.parse_args()

# Config from YAML (base + eddy_id merged)
cfg = load_config(args.experiment, "base.yaml", "eddy_id.yaml")

LON_RANGE = tuple(cfg["base"]["region"]["lon_range"])
LAT_RANGE = tuple(cfg["base"]["region"]["lat_range"])
DATA_DIR = resolve_data_dir(cfg, "swot_dir")
ANTICYCLONE_DIR = resolve_output_dir(args.experiment, "eddy_id", "anticyclone")
CYCLONE_DIR = resolve_output_dir(args.experiment, "eddy_id", "cyclone")

MAX_WORKERS = cfg["eddy_id"]["max_workers"]
BESSEL_WAVELENGTH = cfg["eddy_id"]["bessel_high_filter_wavelength"]
ID_STEP = cfg["eddy_id"]["step"]
ID_SHAPE_ERROR = cfg["eddy_id"]["shape_error"]

def parse_file_datetime(local_fp: str) -> datetime:
    filename = Path(local_fp).name
    pattern = r'(\d{8})'
    matches = re.search(pattern, filename)
    
    if not matches:
        raise ValueError("Check that the filepath contains an 8-digit date")
    
    return datetime.strptime(matches.group(1), '%Y%m%d')

def id_one(
    local_fp: str,
    out_anticyclone_dir: str,
    out_cyclone_dir: str,
) -> tuple[str, str]:
    """
    Identify eddies in a single SWOT L4 SSH file.

    Subsets to region, applies Bessel high-pass filter, then runs
    PET's contour-based identification on ADT.
    """
    date = parse_file_datetime(local_fp)
    a_fn = f"{out_anticyclone_dir}/Anticyclonic_{date.strftime('%Y-%m-%d')}.nc"
    c_fn = f"{out_cyclone_dir}/Cyclonic_{date.strftime('%Y-%m-%d')}.nc"
    if Path(a_fn).exists() and Path(c_fn).exists():
        return a_fn, c_fn

    # Read coordinates first to compute index slices, then re-open via
    # RegularGridDataset. The double open is necessary because PET's
    # RegularGridDataset takes a filename + index dict, not an open dataset.
    with xr.open_dataset(local_fp) as ds:
        lon_min, lon_max = LON_RANGE
        lat_min, lat_max = LAT_RANGE

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
    grid.bessel_high_filter('adt', BESSEL_WAVELENGTH)

    # Perform eddy id
    a, c = grid.eddy_identification(
        grid_height='adt',
        uname='ugos',
        vname='vgos',
        date=date,
        step=ID_STEP,
        shape_error=ID_SHAPE_ERROR,
    )
    if not isinstance(a, EddiesObservations) or not isinstance(c, EddiesObservations):
        raise ValueError("Some step in eddy identification went wrong")

    a.write_file(filename=a_fn)
    c.write_file(filename=c_fn)
    return a_fn, c_fn


def main():
    # Ensure output paths
    ANTICYCLONE_DIR.mkdir(parents=True, exist_ok=True)
    CYCLONE_DIR.mkdir(parents=True, exist_ok=True)

    # Obtain a list of local filepaths
    local_filepaths = list(DATA_DIR.glob("*.nc"))
    if not local_filepaths:
        print(f"No .nc files found in {DATA_DIR}")
        return

    a_dir = str(ANTICYCLONE_DIR)
    c_dir = str(CYCLONE_DIR)

    # CPU-bound (Bessel filter + contour detection) — ThreadPoolExecutor would
    # serialize due to the GIL; must use ProcessPoolExecutor here
    n_failed = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(id_one, f, a_dir, c_dir): f for f in local_filepaths}

        for future in as_completed(futures):
            fp = futures[future]
            try:
                print(future.result())
            except Exception as e:
                n_failed += 1
                print(f"FAILED {Path(fp).name}: {e}")

    if n_failed > 0:
        raise RuntimeError(f"{n_failed}/{len(local_filepaths)} files failed in eddy_id")

if __name__ == '__main__':
    main()