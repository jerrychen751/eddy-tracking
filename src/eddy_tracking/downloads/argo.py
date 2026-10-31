# Download the Argo profiles inside the --lon/--lat box over --dates from the Ifremer ERDDAP server, through argopy

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import argopy
from argopy import ArgoIndex, DataFetcher
from argopy.utils import list_standard_variables

from eddy_tracking.utils.cli import add_region_arguments, write_settings


# GDAC index rows (one per profile file) inside the box [lon0, lon1, lat0, lat1, start, end], for the core and the synthetic BGC index
def download_argo_index(box: list, profile_sets: list[str], out_dir: Path) -> int:
    n_total = 0
    for name, index_file in (("core", "core"), ("bgc", "bgc-s")):
        if name not in profile_sets:
            continue
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


# core_profiles.nc keeps only good or probably good position, time, pressure and temperature and prefers adjusted values, like argopy "standard" mode, but blanks salinity that failed QC instead of dropping the level, so floats with a failed conductivity sensor still give temperature profiles. bgc_profiles.nc uses "expert" mode, which keeps real-time data and every QC flag, because most BGC profiles in the region are still real-time; filter on the *_QC variables when using them
def download_argo_profiles(index_box: list, profile_sets: list[str], chunk_days: int, out_dir: Path) -> int:
    lon0, lon1, lat0, lat1, start, end = index_box
    # The data fetcher's box also takes a pressure range (dbar); 0 to 6000 keeps every level, deep floats included.
    box = [lon0, lon1, lat0, lat1, 0.0, 6000.0, start, end]
    saved = 0
    jobs: tuple[tuple[str, str, dict[str, Any]], ...] = (
        ("core", "core_profiles.nc", dict(ds="phy", mode="expert")),
        ("bgc", "bgc_profiles.nc", dict(ds="bgc", mode="expert", params="all", measured=None)),
    )
    for name, filename, options in jobs:
        if name not in profile_sets:
            continue
        path = out_dir / filename
        if path.exists():
            continue
        fetcher = DataFetcher(
            src="erddap",
            parallel=True,
            chunks_maxsize={"time": chunk_days},
            progress=False,
            **options,
        )
        points = fetcher.region(box).to_xarray()
        # Standard mode filters every QC field at once, so a level whose salinity failed QC lost its good temperature too (about 1,100 profiles in the Gulf Stream box)
        if filename == "core_profiles.nc":
            points = points.argo.datamode.merge()
            points = points.argo.filter_qc(QC_list=[1, 2], QC_fields=["POSITION_QC", "TIME_QC", "PRES_QC", "TEMP_QC"])
            points["PSAL"] = points["PSAL"].where(points["PSAL_QC"].isin([1, 2]))
            points = points.drop_vars(sorted(set(points.data_vars) - set(list_standard_variables())))
        profiles = points.argo.point2profile()
        tmp = path.with_suffix(".nc.part")
        profiles.to_netcdf(tmp)
        tmp.rename(path)
        print(f"output_file: {filename}\nprofiles: {profiles.sizes['N_PROF']}\nlevels: {profiles.sizes['N_LEVELS']}")
        saved += 1
    return saved


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="eddy-tracking download-argo")
    add_region_arguments(parser)
    parser.add_argument("--profiles", nargs="+", choices=["core", "bgc"], required=True, help="core T/S profiles, synthetic BGC profiles, or both")
    parser.add_argument("--chunk-days", type=int, default=30, help="days per ERDDAP request")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    write_settings(args.out, "download-argo", args, operational={"chunk_days"})
    args.out.mkdir(parents=True, exist_ok=True)

    argopy.set_options(cachedir=str(Path.home() / ".cache" / "argopy"))
    box = [*args.lon, *args.lat, *args.dates]
    n_listed = download_argo_index(box, args.profiles, args.out)
    n_saved = download_argo_profiles(box, args.profiles, args.chunk_days, args.out)
    print(
        "status: complete\n"
        f"argo_index_profiles: {n_listed}\n"
        f"argo_files_saved: {n_saved}"
    )
