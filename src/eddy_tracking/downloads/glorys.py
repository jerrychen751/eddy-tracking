# Download the GLORYS12 reanalysis (Copernicus Marine GLOBAL_MULTIYEAR_PHY_001_030) over the study region for the analysis period

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from eddy_tracking.downloads.cmems import download_cmems_dataset
from eddy_tracking.utils.cli import add_region_arguments, write_settings


# One NetCDF per calendar month into three subdirectories of --out: surface/ holds the 2D fields (mixed layer depth, SSH, bottom temperature), currents/ holds the velocity at the top model level, and profiles/ holds temperature and salinity from the surface down to --max-depth-m metres. The 3D profiles are by far the largest part, so the depth cap keeps the download to the upper ocean that mixed layer and isopycnal analyses need.
def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="eddy-tracking download-glorys")
    add_region_arguments(parser)
    parser.add_argument("--dataset-id", required=True, help="Copernicus Marine dataset id, such as cmems_mod_glo_phy_my_0.083deg_P1D-m")
    parser.add_argument("--dataset-version", required=True, help="such as 202311")
    parser.add_argument("--surface-variables", nargs="+", required=True, help="2D fields for surface/, such as mlotst zos bottomT")
    parser.add_argument("--current-variables", nargs="+", required=True, help="velocity fields for currents/, such as uo vo")
    parser.add_argument("--profile-variables", nargs="+", required=True, help="3D fields for profiles/, such as thetao so")
    parser.add_argument("--max-depth-m", type=float, required=True, help="deepest level of the profiles")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    write_settings(args.out, "download-glorys", args, operational=set())
    args.out.mkdir(parents=True, exist_ok=True)

    common: dict[str, Any] = dict(
        date_range=tuple(args.dates),
        lon_range=tuple(args.lon),
        lat_range=tuple(args.lat),
        dataset_id=args.dataset_id,
        dataset_version=args.dataset_version,
    )
    n_saved = 0
    n_saved += download_cmems_dataset(
        out_dir=args.out / "surface", variables=args.surface_variables, **common
    )
    n_saved += download_cmems_dataset(
        out_dir=args.out / "currents",
        variables=args.current_variables,
        depth_range=(0.0, 1.0),
        **common,
    )
    n_saved += download_cmems_dataset(
        out_dir=args.out / "profiles",
        variables=args.profile_variables,
        depth_range=(0.0, args.max_depth_m),
        **common,
    )
    print(
        "status: complete\n"
        f"glorys_files_saved: {n_saved}"
    )
