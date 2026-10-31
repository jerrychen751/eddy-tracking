# Download the Copernicus Marine in situ profiles (INSITU_GLO_PHYBGCWAV_DISCRETE_MYNRT_013_030: gliders, ship CTD, XBT) inside the study region over the analysis period, and build one profile file in the layout of the Argo download.
# The product stores one NetCDF per platform per month (monthly/<type>/) or per platform lifetime (history/<type>/), listed in index files with each file's bounding box and time span. The toolbox's `get` lists the whole multi-million-file store before matching a file list, which takes longer than the download, so this module reads the index files through the toolbox and fetches the selected files straight from the public native bucket. Argo (PF) is excluded here because downloads/argo.py already provides it.

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import warnings
from pathlib import Path
from typing import cast

import copernicusmarine
import gsw
import numpy as np
import pandas as pd
import requests
import xarray as xr

from eddy_tracking.downloads.auth import load_cmems_credentials
from eddy_tracking.utils.cli import add_region_arguments, write_settings

PLATFORM_NAMES = {"GL": "glider", "CT": "ctd", "XB": "xbt"}


# The history index serves only platform types that have no monthly files, such as XBT. A history file spans a platform's lifetime, so this over-selects; the profiles themselves are filtered after download.
def select_index_files(index_dir: Path, lon_range, lat_range, date_range, platform_types: list[str]) -> pd.DataFrame:
    frames = []
    for index_name in ("index_monthly.txt", "index_history.txt"):
        table = pd.read_csv(index_dir / index_name, comment="#", names=[
            "product_id", "file_name", "lat_min", "lat_max", "lon_min", "lon_max",
            "time_start", "time_end", "institution", "date_update", "data_mode", "parameters",
        ])
        for column in ("time_start", "time_end"):
            table[column] = pd.to_datetime(table[column], errors="coerce", utc=True).dt.tz_localize(None)
        table["platform_type"] = table["file_name"].str.extract(r"/(?:history|monthly|latest)/([A-Z]{2})/")[0]
        keep = (
            table["platform_type"].isin(platform_types)
            & table["lat_max"].ge(lat_range[0]) & table["lat_min"].le(lat_range[1])
            & table["lon_max"].ge(lon_range[0]) & table["lon_min"].le(lon_range[1])
            & table["time_end"].ge(date_range[0]) & table["time_start"].le(date_range[1])
            & table["parameters"].fillna("").str.contains("TEMP")
        )
        frames.append(table.loc[keep])
    monthly, history = frames
    history = history.loc[~history["platform_type"].isin(monthly["platform_type"].unique())]
    return pd.concat([monthly, history], ignore_index=True)


def download_insitu_files(out_dir: Path, lon_range, lat_range, date_range, platform_types: list[str]) -> pd.DataFrame:
    index_dir = out_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    username, password = load_cmems_credentials()
    copernicusmarine.get(
        dataset_id="cmems_obs-ins_glo_phybgcwav_mynrt_na_irr", index_parts=True, output_directory=index_dir, no_directories=True,
        username=username, password=password, disable_progress_bar=True, overwrite=True,
    )
    selection = select_index_files(index_dir, lon_range, lat_range, date_range, platform_types)
    selection["local_name"] = selection["file_name"].str.rsplit("/", n=1).str[1]
    selection.to_parquet(out_dir / "index_selection.parquet", index=False)
    print(
        "status: downloading_insitu\n"
        f"files_selected: {len(selection)}\n"
        + "\n".join(f"files_{PLATFORM_NAMES.get(str(t), t)}: {n}" for t, n in selection["platform_type"].value_counts().items())
    )

    def fetch(row) -> int:
        target = out_dir / row.local_name
        if target.exists():
            return 0
        response = requests.get("https://s3.waw3-1.cloudferro.com/mdl-native-01/native/" + row.file_name, timeout=600)
        response.raise_for_status()
        partial = target.with_suffix(".nc.part")
        partial.write_bytes(response.content)
        partial.rename(target)
        return 1

    with cf.ThreadPoolExecutor(8) as pool:
        saved = sum(pool.map(fetch, selection.itertuples(index=False)))
    print(f"files_saved: {saved}")
    return selection


# A few gliders report every 0.3 m, which would multiply the file size for no gain at the 10 dbar grid of the analyses, so a profile with more than max_levels valid levels is averaged into 2 dbar bins; other profiles pass through unchanged
def _bin_profile(pressure: np.ndarray, temp: np.ndarray, psal: np.ndarray, max_levels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(pressure)
    if valid.sum() <= max_levels:
        return pressure, temp, psal
    bins = np.floor(pressure[valid] / 2.0).astype(int)
    frame = cast(pd.DataFrame, pd.DataFrame({"bin": bins, "pressure": pressure[valid], "temp": temp[valid], "psal": psal[valid]}).groupby("bin").mean())
    out = [np.full(len(pressure), np.nan) for _ in range(3)]
    for column, target in zip(("pressure", "temp", "psal"), out):
        target[: len(frame)] = frame[column].to_numpy()
    return out[0], out[1], out[2]


# Keeps levels whose TEMP (and PSAL, when present) QC is good or probably good (1 or 2), and at most one profile per platform per thin_hours, because gliders profile every few minutes to hours and the analyses need daily resolution at most. Depth (DEPH) is converted to pressure with gsw at the profile latitude when the file has no PRES.
def _read_platform_file(path: Path, lon_range, lat_range, date_range, min_depth_m: float, thin_hours: float, max_levels: int) -> pd.DataFrame | None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xr.open_dataset(path)
    try:
        time = pd.to_datetime(ds["TIME"].to_numpy())
        lat = np.broadcast_to(ds["LATITUDE"].to_numpy(), time.shape).astype(float)
        lon = np.broadcast_to(ds["LONGITUDE"].to_numpy(), time.shape).astype(float)
        if "PRES" in ds:
            vertical = ds["PRES"].to_numpy().astype(float)
            pressure = np.broadcast_to(vertical, (len(time), vertical.shape[-1])).copy()
        else:
            depth = np.broadcast_to(ds["DEPH"].to_numpy().astype(float), (len(time), ds.sizes["DEPTH"]))
            pressure = gsw.p_from_z(-depth, lat[:, None])
        temp = ds["TEMP"].to_numpy().astype(float)
        temp_qc = ds["TEMP_QC"].to_numpy()
        good = np.isin(temp_qc, (1, 2)) & np.isfinite(temp)
        if "PSAL" in ds:
            psal = ds["PSAL"].to_numpy().astype(float)
            good &= np.isin(ds["PSAL_QC"].to_numpy(), (1, 2)) & np.isfinite(psal)
        else:
            psal = np.full_like(temp, np.nan)
        platform = str(ds.attrs.get("platform_code", path.stem.split("_")[3]))
        data_mode = str(ds.attrs.get("data_mode", ""))
    finally:
        ds.close()
    temp = np.where(good, temp, np.nan)
    psal = np.where(good, psal, np.nan)
    pressure = np.where(good, pressure, np.nan)
    deepest = np.nanmax(np.where(good, pressure, -np.inf), axis=1)
    keep = (
        (time >= pd.Timestamp(date_range[0])) & (time <= pd.Timestamp(date_range[1]) + pd.Timedelta(days=1))
        & (lon >= lon_range[0]) & (lon <= lon_range[1]) & (lat >= lat_range[0]) & (lat <= lat_range[1])
        & (deepest >= min_depth_m)
    )
    if not keep.any():
        return None
    order = np.flatnonzero(keep)[np.argsort(time[keep])]
    chosen = []
    last = None
    for index in order:
        if last is None or (time[index] - last) >= pd.Timedelta(hours=thin_hours):
            chosen.append(index)
            last = time[index]
    binned = [_bin_profile(pressure[i], temp[i], psal[i], max_levels) for i in chosen]
    return pd.DataFrame({
        "platform": platform, "platform_type": PLATFORM_NAMES.get(path.stem.split("_")[2], path.stem.split("_")[2]),
        "data_mode": data_mode, "time": time[chosen], "lon": lon[chosen], "lat": lat[chosen],
        "pressure": [b[0] for b in binned], "temp": [b[1] for b in binned], "psal": [b[2] for b in binned],
    })


# profiles.nc is laid out like the Argo download (N_PROF x N_LEVELS), with the levels of each profile left-packed and padded with NaN. Only good or probably good levels are kept, so no QC variables are written.
def build_profile_file(out_dir: Path, files: list[Path], lon_range, lat_range, date_range, min_depth_m: float, thin_hours: float, max_levels: int) -> xr.Dataset:
    args = (lon_range, lat_range, date_range, min_depth_m, thin_hours, max_levels)
    frames = [frame for frame in (_read_platform_file(path, *args) for path in files) if frame is not None]
    table = pd.concat(frames, ignore_index=True).sort_values("time").reset_index(drop=True)
    n_levels = max(int(np.isfinite(row).sum()) for row in table["pressure"])
    packed = {name: np.full((len(table), n_levels), np.nan) for name in ("pressure", "temp", "psal")}
    for index, (pressure, temp, psal) in enumerate(zip(table["pressure"], table["temp"], table["psal"])):
        valid = np.isfinite(pressure)
        order = np.argsort(pressure[valid])
        count = int(valid.sum())
        for name, values in zip(packed, (pressure, temp, psal)):
            packed[name][index, :count] = values[valid][order]
    profiles = xr.Dataset(
        {
            "PLATFORM_NUMBER": ("N_PROF", table["platform"].to_numpy().astype(str)),
            "PLATFORM_TYPE": ("N_PROF", table["platform_type"].to_numpy().astype(str)),
            "DATA_MODE": ("N_PROF", table["data_mode"].to_numpy().astype(str)),
            "TIME": ("N_PROF", table["time"].to_numpy().astype("datetime64[ns]")),
            "LONGITUDE": ("N_PROF", table["lon"].to_numpy()),
            "LATITUDE": ("N_PROF", table["lat"].to_numpy()),
            "PRES": (("N_PROF", "N_LEVELS"), packed["pressure"].astype("float32")),
            "TEMP": (("N_PROF", "N_LEVELS"), packed["temp"].astype("float32")),
            "PSAL": (("N_PROF", "N_LEVELS"), packed["psal"].astype("float32")),
        },
        attrs={
            "source": "Copernicus Marine INSITU_GLO_PHYBGCWAV_DISCRETE_MYNRT_013_030, gliders, ship CTD and XBT",
            "quality_control": "levels with TEMP_QC and PSAL_QC in (1, 2) only",
            "thinning": f"at most one profile per platform per {thin_hours:g} hours, profiles reaching {min_depth_m:g} m",
            "pressure": "PRES where the file has it, else gsw.p_from_z of DEPH at the profile latitude",
        },
    )
    encoding = {name: {"zlib": True, "complevel": 4} for name in ("PRES", "TEMP", "PSAL")}
    target = out_dir / "profiles.nc"
    partial = target.with_suffix(".nc.part")
    profiles.to_netcdf(partial, encoding=encoding)
    partial.rename(target)
    print(
        "output_file: profiles.nc\n"
        f"profiles: {profiles.sizes['N_PROF']}\n"
        f"levels: {profiles.sizes['N_LEVELS']}\n"
        + "\n".join(f"profiles_{t}: {n}" for t, n in table["platform_type"].value_counts().items())
    )
    return profiles


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="eddy-tracking download-insitu")
    add_region_arguments(parser)
    parser.add_argument("--platform-types", nargs="+", required=True, help="two-letter Copernicus platform codes, such as GL (glider), CT (ship CTD) and XB (XBT)")
    parser.add_argument("--min-depth-m", type=float, required=True, help="profiles must reach this depth")
    parser.add_argument("--thin-hours", type=float, required=True, help="keep at most one profile per platform per this many hours")
    parser.add_argument("--max-levels", type=int, required=True, help="profiles with more valid levels are averaged into 2 dbar bins")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    write_settings(args.out, "download-insitu", args, operational=set())
    args.out.mkdir(parents=True, exist_ok=True)

    lon_range, lat_range, date_range = tuple(args.lon), tuple(args.lat), tuple(args.dates)
    selection = download_insitu_files(args.out, lon_range, lat_range, date_range, args.platform_types)
    build_profile_file(
        args.out, [args.out / name for name in selection["local_name"]],
        lon_range, lat_range, date_range, args.min_depth_m, args.thin_hours, args.max_levels,
    )
    print("status: complete")
