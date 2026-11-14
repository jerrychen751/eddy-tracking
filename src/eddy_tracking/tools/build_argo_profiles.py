# The Argo profiles of a study in the form notebooks read: levels down to --max-pressure only, packed to the front of each profile, and per profile the |Rossby number| of the SSH map at its position and its signed distance from the daily Gulf Stream axis, so a laptop without the SSH maps can still pick calm background water

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from eddy_tracking.preprocess.streamline import compute_signed_distance_km, index_centerlines_by_date
from eddy_tracking.preprocess.swot import find_nearest_swot_file, index_swot_files_by_date, load_rossby_field
from eddy_tracking.utils.cli import write_settings
from eddy_tracking.utils.files import write_netcdf_atomically


def sample_rossby_on_days(days: list[tuple[Path, np.ndarray, np.ndarray, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray]]:
    # Each day: |Ro| interpolated linearly from its SSH map at its profile positions; NaN where the map is masked
    sampled = []
    for ssh_path, profile_rows, lon, lat in days:
        ssh_lon, ssh_lat, rossby_number = load_rossby_field(ssh_path)
        field = xr.DataArray(np.abs(rossby_number), coords={"latitude": ssh_lat, "longitude": ssh_lon}, dims=["latitude", "longitude"])
        sampled.append((profile_rows, field.interp(latitude=xr.DataArray(lat, dims="point"), longitude=xr.DataArray(lon, dims="point")).to_numpy()))
    return sampled


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="eddy-tracking build-argo-profiles")
    parser.add_argument("--argo", type=Path, required=True, help="core_profiles.nc that download-argo wrote")
    parser.add_argument("--ssh", type=Path, required=True, help="folder of daily SSH files with relative_vorticity (the Rossby number)")
    parser.add_argument("--axis", type=Path, required=True, help="folder trace-axis wrote")
    parser.add_argument("--max-pressure", type=float, required=True, help="keep levels at or above this pressure (dbar)")
    parser.add_argument("--out", type=Path, required=True, help="netCDF file")
    parser.add_argument("--workers", type=int, default=1, help="processes that read the SSH maps; the reads wait on the filesystem, so more workers than cores still help")
    args = parser.parse_args(argv)
    write_settings(args.out, "build-argo-profiles", args, operational={"workers"})
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(args.argo) as argo:
        pressure = argo["PRES"].to_numpy()
        levels = {name: argo[name].to_numpy() for name in ("PRES", "TEMP", "PSAL")}
        profile_variables = {name: argo[name].to_numpy() for name in ("PLATFORM_NUMBER", "CYCLE_NUMBER", "DIRECTION", "DATA_MODE", "LONGITUDE", "LATITUDE", "TIME")}
        attributes = dict(argo.attrs)

    # Kept levels move to the front of each profile in their original order, so the level dimension shrinks to the longest kept profile
    keep = np.isfinite(pressure) & (pressure <= args.max_pressure)
    order = np.argsort(~keep, axis=1, kind="stable")
    n_levels = int(keep.sum(axis=1).max())
    packed_keep = np.take_along_axis(keep, order, axis=1)[:, :n_levels]
    packed = {name: np.where(packed_keep, np.take_along_axis(values, order, axis=1)[:, :n_levels], np.nan).astype(np.float32) for name, values in levels.items()}

    profiles = pd.DataFrame({"lon": profile_variables["LONGITUDE"], "lat": profile_variables["LATITUDE"], "date": pd.Series(profile_variables["TIME"]).dt.normalize()})

    # |Ro| interpolated linearly from the SSH map of the profile day, or the nearest day within SWOT_SEARCH_DAYS; NaN where the map is masked or missing
    ssh_files = index_swot_files_by_date(args.ssh)
    days = []
    for day, day_profiles in profiles.groupby("date"):
        ssh_path = find_nearest_swot_file(ssh_files, pd.Timestamp(day).date())  # pyright: ignore[reportArgumentType]
        if ssh_path is not None:
            days.append((ssh_path, day_profiles.index.to_numpy(), day_profiles["lon"].to_numpy(), day_profiles["lat"].to_numpy()))
    abs_rossby = np.full(len(profiles), np.nan)
    # Interleaved chunks, so each worker gets early and late years alike
    with ProcessPoolExecutor(args.workers) as executor:
        for sampled in executor.map(sample_rossby_on_days, [days[start::args.workers] for start in range(args.workers)]):
            for profile_rows, values in sampled:
                abs_rossby[profile_rows] = values

    # Positive (N) left of the flow at the nearest point on that day's axis, which is north where the jet flows east
    centerlines = index_centerlines_by_date(pd.read_parquet(args.axis / "streamline.parquet"))
    axis_km = np.array([
        compute_signed_distance_km(centerlines[day.date()].lon, centerlines[day.date()].lat, lon, lat)[0] if day.date() in centerlines else np.nan
        for day, lon, lat in zip(profiles["date"], profiles["lon"], profiles["lat"])
    ])

    dataset = xr.Dataset(
        {
            **{name: (("N_PROF", "N_LEVELS"), values) for name, values in packed.items()},
            **{name: ("N_PROF", values) for name, values in profile_variables.items() if name not in ("LONGITUDE", "LATITUDE", "TIME")},
            "abs_rossby": ("N_PROF", abs_rossby.astype(np.float32), {"long_name": "|relative vorticity / f| of the SSH map at the profile position"}),
            "axis_km": ("N_PROF", axis_km.astype(np.float32), {"long_name": "signed distance from the daily Gulf Stream axis, positive north", "units": "km"}),
        },
        coords={name: ("N_PROF", profile_variables[name]) for name in ("LONGITUDE", "LATITUDE", "TIME")},
        attrs={**attributes, "comment": f"Levels at or above {args.max_pressure:g} dbar from {args.argo.name}"},
    )
    write_netcdf_atomically(dataset, args.out, encoding={name: {"zlib": True, "complevel": 4} for name in packed})

    print(
        "status: complete\n"
        f"output_file: {args.out}\n"
        f"profiles: {len(profiles)}\n"
        f"levels: {n_levels}\n"
        f"profiles_with_rossby: {int(np.isfinite(abs_rossby).sum())}\n"
        f"profiles_with_axis: {int(np.isfinite(axis_km).sum())}\n"
        f"duplicate_profile_keys: {int(pd.DataFrame({key: profile_variables[key] for key in ('PLATFORM_NUMBER', 'CYCLE_NUMBER', 'DIRECTION')}).duplicated().sum())}"
    )
