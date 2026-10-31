# Download the Argo profiles inside the --lon/--lat box over --dates from the Ifremer ERDDAP server, through argopy

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import argopy
import numpy as np
import pandas as pd
import xarray as xr
from argopy import ArgoIndex, DataFetcher
from argopy.errors import DataNotFound
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
def download_argo_profiles(index_box: list, profile_sets: list[str], chunk_days: int, window_days: int, out_dir: Path) -> int:
    lon0, lon1, lat0, lat1, start, end = index_box
    # --dates includes its end day, so the last window runs to the midnight after it
    edges = sorted({*pd.date_range(start, end, freq=f"{window_days}D"), pd.Timestamp(end) + pd.Timedelta(days=1)})
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

        # Each window of --window-days is fetched and saved on its own, so a long period never sits in memory at once (33 years of core points ran past 64 GB) and a rerun resumes at the first missing window
        window_dir = out_dir / f"{name}_windows"
        window_dir.mkdir(exist_ok=True)
        for window_start, window_end in zip(edges[:-1], edges[1:]):
            window_path = window_dir / f"{window_start:%Y%m%d}.nc"
            if window_path.exists():
                continue
            fetcher = DataFetcher(
                src="erddap",
                parallel=True,
                chunks_maxsize={"time": chunk_days},
                progress=False,
                **options,
            )
            # The box's end time is inclusive, so the window stops one second before the next one starts. The pressure range (dbar) 0 to 6000 keeps every level, deep floats included.
            box = [lon0, lon1, lat0, lat1, 0.0, 6000.0, f"{window_start:%Y-%m-%dT%H:%M:%S}", f"{window_end - pd.Timedelta(seconds=1):%Y-%m-%dT%H:%M:%S}"]
            # argopy's default skips an ERDDAP request that keeps failing, which leaves a lon, lat, time and 500 dbar block out of the profiles without a word, so a failure stops the run instead
            try:
                points = fetcher.region(box).to_xarray(errors="raise")
            except DataNotFound:
                print(f"window: {window_start:%Y-%m-%d}\nprofiles: 0")
                continue
            # Standard mode filters every QC field at once, so a level whose salinity failed QC lost its good temperature too (about 1,100 profiles in the Gulf Stream box)
            if filename == "core_profiles.nc":
                points = points.argo.datamode.merge()
                points = points.argo.filter_qc(QC_list=[1, 2], QC_fields=["POSITION_QC", "TIME_QC", "PRES_QC", "TEMP_QC"])
                points["PSAL"] = points["PSAL"].where(points["PSAL_QC"].isin([1, 2]))
                points = points.drop_vars(sorted(set(points.data_vars) - set(list_standard_variables())))
            window_profiles = points.argo.point2profile()
            window_profiles.to_netcdf(window_path.with_suffix(".nc.part"))
            window_path.with_suffix(".nc.part").rename(window_path)
            print(f"window: {window_start:%Y-%m-%d}\nprofiles: {window_profiles.sizes['N_PROF']}")

        # point2profile stores a variable that is constant within every profile of its window per profile only, so TEMP_QC can be per level in one window and per profile in the next; such a variable goes back onto the levels of its profiles. Windows also differ in their deepest level count, and the outer join pads the shorter ones. Both fill the way point2profile pads a short profile, NaN values and QC 99999.
        windows = [xr.open_dataset(p, chunks={}) for p in sorted(window_dir.glob("*.nc"))]
        level_vars = {str(v) for window in windows for v in window.data_vars if "N_LEVELS" in window[v].dims}
        for window in windows:
            for v in level_vars & {str(v) for v in window.data_vars if "N_LEVELS" not in window[v].dims}:
                window[v] = window[v].broadcast_like(window["PRES"]).where(window["PRES"].notnull(), 99999 if v.endswith("_QC") else np.nan)
        profiles = xr.concat(
            windows,
            dim="N_PROF",
            join="outer",
            fill_value={str(v): 99999 for v in windows[0].data_vars if str(v).endswith("_QC")},
        )
        profiles = profiles.assign_coords(N_PROF=np.arange(profiles.sizes["N_PROF"]))
        tmp = path.with_suffix(".nc.part")
        profiles.to_netcdf(tmp)
        for window in windows:
            window.close()
        tmp.rename(path)
        shutil.rmtree(window_dir)
        print(f"output_file: {filename}\nprofiles: {profiles.sizes['N_PROF']}\nlevels: {profiles.sizes['N_LEVELS']}")
        saved += 1
    return saved


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="eddy-tracking download-argo")
    add_region_arguments(parser)
    parser.add_argument("--profiles", nargs="+", choices=["core", "bgc"], required=True, help="core T/S profiles, synthetic BGC profiles, or both")
    parser.add_argument("--chunk-days", type=int, default=30, help="days per ERDDAP request")
    parser.add_argument("--window-days", type=int, default=365, help="days fetched and saved together before the windows join into one file")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    write_settings(args.out, "download-argo", args, operational={"chunk_days", "window_days"})
    args.out.mkdir(parents=True, exist_ok=True)

    argopy.set_options(cachedir=str(Path.home() / ".cache" / "argopy"))
    box = [*args.lon, *args.lat, *args.dates]
    n_listed = download_argo_index(box, args.profiles, args.out)
    n_saved = download_argo_profiles(box, args.profiles, args.chunk_days, args.window_days, args.out)
    print(
        "status: complete\n"
        f"argo_index_profiles: {n_listed}\n"
        f"argo_files_saved: {n_saved}"
    )
