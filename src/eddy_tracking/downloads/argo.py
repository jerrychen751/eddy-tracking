# Download the Argo profiles inside the study region over the analysis period from the Ifremer ERDDAP server, through argopy

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import argopy
from argopy import ArgoIndex, DataFetcher

from eddy_tracking.config import load_config, resolve_data_dir


def _build_box(cfg: dict) -> list:
    base = cfg["base"]
    lon0, lon1 = base["region"]["lon_range"]
    lat0, lat1 = base["region"]["lat_range"]
    start, end = base["time"]["analysis_date_range"]
    return [lon0, lon1, lat0, lat1, start, end]


# GDAC index rows (one per profile file) inside the box, for the core and the synthetic BGC index
def download_argo_index(cfg: dict, out_dir: Path) -> int:
    box = _build_box(cfg)
    n_total = 0
    for name, index_file in (("core", "core"), ("bgc", "bgc-s")):
        path = out_dir / f"{name}_index.parquet"
        if path.exists():
            continue
        index = ArgoIndex(index_file=index_file).load()
        index.query.box(box)
        frame = index.to_dataframe()
        frame.to_parquet(path, index=False)
        print(f"output_file: {path.name}\nprofiles: {len(frame)}\nfloats: {frame['wmo'].nunique()}")
        n_total += len(frame)
    return n_total


# core_profiles.nc uses argopy "standard" mode, which keeps only good or probably good QC and prefers adjusted values. bgc_profiles.nc uses "expert" mode, which keeps real-time data and every QC flag, because most BGC profiles in the region are still real-time; filter on the *_QC variables when using them
def download_argo_profiles(cfg: dict, out_dir: Path) -> int:
    lon0, lon1, lat0, lat1, start, end = _build_box(cfg)
    # The data fetcher's box also takes a pressure range (dbar); 0 to 6000 keeps every level, deep floats included.
    box = [lon0, lon1, lat0, lat1, 0.0, 6000.0, start, end]
    argo_cfg = cfg["base"]["download"]["argo"]
    saved = 0
    jobs: tuple[tuple[str, dict[str, Any]], ...] = (
        ("core_profiles.nc", dict(ds="phy", mode="standard")),
        ("bgc_profiles.nc", dict(ds="bgc", mode="expert", params="all", measured=None)),
    )
    for filename, options in jobs:
        path = out_dir / filename
        if path.exists():
            continue
        fetcher = DataFetcher(
            src="erddap",
            parallel=True,
            chunks_maxsize={"time": argo_cfg["chunk_days"]},
            progress=False,
            **options,
        )
        points = fetcher.region(box).to_xarray()
        profiles = points.argo.point2profile()
        tmp = path.with_suffix(".nc.part")
        profiles.to_netcdf(tmp)
        tmp.rename(path)
        print(f"output_file: {filename}\nprofiles: {profiles.sizes['N_PROF']}\nlevels: {profiles.sizes['N_LEVELS']}")
        saved += 1
    return saved


def main(experiment: str) -> None:
    cfg = load_config(experiment)
    argopy.set_options(cachedir=str(Path.home() / ".cache" / "argopy"))
    out_dir = resolve_data_dir(cfg, "argo_dir")
    n_listed = download_argo_index(cfg, out_dir)
    n_saved = download_argo_profiles(cfg, out_dir)
    print(
        "status: complete\n"
        f"argo_index_profiles: {n_listed}\n"
        f"argo_files_saved: {n_saved}"
    )


if __name__ == "__main__":
    main(sys.argv[1])
