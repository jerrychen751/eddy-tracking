# Download the GLORYS12 reanalysis (Copernicus Marine GLOBAL_MULTIYEAR_PHY_001_030) over the study region for the analysis period

from __future__ import annotations

import sys
from typing import Any

from eddy_tracking.config import load_config, resolve_data_dir
from eddy_tracking.downloads.cmems import download_cmems_dataset


# One NetCDF per calendar month into three subdirectories of glorys_dir: surface/ holds the 2D fields (mixed layer depth, SSH, bottom temperature), currents/ holds the velocity at the top model level, and profiles/ holds temperature and salinity from the surface down to max_depth metres. The 3D profiles are by far the largest part, so the depth cap keeps the download to the upper ocean that mixed layer and isopycnal analyses need.
def main(experiment: str) -> None:
    cfg = load_config(experiment)
    base = cfg["base"]
    glorys = base["download"]["glorys"]
    common: dict[str, Any] = dict(
        date_range=tuple(base["time"]["analysis_date_range"]),
        lon_range=tuple(base["region"]["lon_range"]),
        lat_range=tuple(base["region"]["lat_range"]),
        dataset_id=glorys["dataset_id"],
        dataset_version=glorys["dataset_version"],
    )
    root = resolve_data_dir(cfg, "glorys_dir")
    n_saved = 0
    n_saved += download_cmems_dataset(
        out_dir=root / "surface", variables=list(glorys["surface_variables"]), **common
    )
    n_saved += download_cmems_dataset(
        out_dir=root / "currents",
        variables=list(glorys["current_variables"]),
        depth_range=(0.0, 1.0),
        **common,
    )
    n_saved += download_cmems_dataset(
        out_dir=root / "profiles",
        variables=list(glorys["profile_variables"]),
        depth_range=(0.0, float(glorys["max_depth"])),
        **common,
    )
    print(
        "status: complete\n"
        f"glorys_files_saved: {n_saved}"
    )


if __name__ == "__main__":
    main(sys.argv[1])
