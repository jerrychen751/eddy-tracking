"""Download, trim, and optionally mask daily AVISO L4 SSH files."""

import argparse
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import cast

import numpy as np
from scipy.ndimage import distance_transform_edt
import xarray as xr

from eddy_tracking.downloads.auth import login_aviso, load_aviso_credentials


# netCDF4/HDF5 serialization is not safe across downloader threads.
_NETCDF_IO_LOCK = Lock()


@dataclass(frozen=True)
class DownloadSettings:
    """Configuration and credentials for one SWOT download run."""

    host: str
    user: str
    password: str = field(repr=False)
    remote_dir: str
    local_dir: Path
    max_workers: int
    date_range: tuple[str | None, str | None]
    longitude_range: tuple[float, float]
    latitude_range: tuple[float, float]
    filter_open_ocean: bool


def load_settings(experiment: str) -> DownloadSettings:
    from utils.config import load_config, resolve_data_dir

    credentials = load_aviso_credentials()
    cfg = load_config(experiment)
    return DownloadSettings(
        host=credentials.host,
        user=credentials.user,
        password=credentials.password,
        remote_dir=cfg["base"]["download"]["swot"]["ftp_dir"],
        local_dir=resolve_data_dir(cfg, "swot_dir"),
        max_workers=cfg["base"]["download"]["swot"]["max_workers"],
        date_range=tuple(cfg["base"]["time"]["eddy_date_range"]),
        longitude_range=tuple(cfg["base"]["region"]["lon_range"]),
        latitude_range=tuple(cfg["base"]["region"]["lat_range"]),
        filter_open_ocean=cfg["base"]["download"]["swot"].get(
            "filter_open_ocean", False
        ),
    )


def filter_by_date_range(
    files: list[tuple[str, int | None]],
    date_range: tuple[str | None, str | None],
) -> list[tuple[str, int | None]]:
    """Keep files whose filename date falls within the inclusive range."""
    start = datetime.strptime(date_range[0], "%Y-%m-%d") if date_range[0] else None
    end = datetime.strptime(date_range[1], "%Y-%m-%d") if date_range[1] else None
    filtered_files = []
    for path, size in files:
        match = re.search(r"\d{8}", Path(path).name)
        if not match:
            continue
        observation_date = datetime.strptime(match.group(), "%Y%m%d")
        if (start is not None and observation_date < start) or (
            end is not None and observation_date > end
        ):
            continue
        filtered_files.append((path, size))
    return filtered_files


def mask_open_ocean(dataset: xr.Dataset) -> xr.Dataset:
    """Mask invalid cells, their eight-cell coast buffer, and the Great Lakes."""
    swot_validity_fields = ("adt", "ugos", "vgos", "relative_vorticity")
    coast_min_distance_pixels = 8
    great_lakes_lon_range = (-81, -75)
    great_lakes_lat_range = (40, 44)

    surface = dataset
    if "time" in surface["adt"].dims:
        # (1, n_lat, n_lon) -> (n_lat, n_lon)
        surface = surface.isel(time=0)

    # 4 field masks of (n_lat, n_lon) -> (n_lat, n_lon)
    valid = np.logical_and.reduce(
        [np.isfinite(surface[field].to_numpy()) for field in swot_validity_fields]
    )
    coast_ok = distance_transform_edt(valid) >= coast_min_distance_pixels  # pyright: ignore[reportOperatorIssue]

    longitude = surface["longitude"].to_numpy()
    latitude = surface["latitude"].to_numpy()
    # longitude (n_lon,) -> (1, n_lon) and latitude (n_lat,) -> (n_lat, 1) broadcast to (n_lat, n_lon)
    in_great_lakes = (
        (longitude[np.newaxis, :] >= great_lakes_lon_range[0])
        & (longitude[np.newaxis, :] <= great_lakes_lon_range[1])
        & (latitude[:, np.newaxis] >= great_lakes_lat_range[0])
        & (latitude[:, np.newaxis] <= great_lakes_lat_range[1])
    )
    keep = xr.DataArray(
        coast_ok & ~in_great_lakes,
        coords={"latitude": surface["latitude"], "longitude": surface["longitude"]},
        dims=("latitude", "longitude"),
    )

    filtered = dataset.copy()
    for field in swot_validity_fields:
        filtered[field] = filtered[field].where(keep)
    return filtered


def _download_one(remote_path: str, settings: DownloadSettings) -> str:
    """Download and trim one file, or report that its output already exists."""
    filename = Path(remote_path).name
    local_path = settings.local_dir / filename

    if local_path.exists():
        return (
            "status: skipped\n"
            f"file: {filename}\n"
            "reason: already_exists"
        )

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with login_aviso(settings.host, settings.user, settings.password) as ftp:
            ftp.cwd(settings.remote_dir)
            with tmp_path.open("wb") as temp_file:
                ftp.retrbinary(f"RETR {filename}", temp_file.write)
        with _NETCDF_IO_LOCK:
            _trim_file(
                tmp_path,
                local_path,
                settings.longitude_range,
                settings.latitude_range,
                settings.filter_open_ocean,
            )
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        return (
            "status: error\n"
            f"file: {filename}\n"
            f"error: {exc}"
        )

    return (
        "status: downloaded\n"
        f"file: {filename}"
    )


def _trim_file(
    raw_path: Path,
    out_path: Path,
    longitude_range: tuple[float, float],
    latitude_range: tuple[float, float],
    filter_open_ocean: bool,
) -> None:
    """Install a trimmed output through a temporary file, then delete the raw input."""
    temporary_path = out_path.with_suffix(".tmp.nc")
    try:
        with xr.open_dataset(raw_path) as dataset:
            trimmed = dataset.sel(
                longitude=slice(*longitude_range),
                latitude=slice(*latitude_range),
            )
            if filter_open_ocean:
                trimmed = mask_open_ocean(trimmed)
            trimmed.to_netcdf(temporary_path)
        temporary_path.replace(out_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    raw_path.unlink()


def download_files(settings: DownloadSettings) -> list[str]:
    print("status: listing_remote_files")
    with login_aviso(settings.host, settings.user, settings.password) as ftp:
        remote_paths = [
            path for path in ftp.nlst(settings.remote_dir) if path.endswith(".nc")
        ]
        # ftp.size fails in the default ASCII mode, so switch to binary type first.
        ftp.voidcmd("TYPE I")
        # Each entry is (remote path, size in bytes), and ftp.size gives None when the server reports no size.
        remote_files = [(path, ftp.size(path)) for path in remote_paths]
    remote_files.sort(key=lambda item: Path(item[0]).name)
    print(f"remote_files: {len(remote_files)}")

    selected_files = filter_by_date_range(remote_files, settings.date_range)
    print(
        f"selected_files: {len(selected_files)}\n"
        f"date_range: {settings.date_range}"
    )

    paths_to_download = [path for path, _ in selected_files]
    total_bytes = sum(size for _, size in selected_files if size is not None)
    print(f"download_size_gb: {(total_bytes / 1024**3):.2f}")
    print(
        "status: downloading\n"
        f"files: {len(paths_to_download)}"
    )

    failures = []
    with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
        futures = [
            executor.submit(_download_one, path, settings)
            for path in paths_to_download
        ]
        for future in as_completed(futures):
            result = future.result()
            print(result)
            if result.startswith("status: error\n"):
                failures.append(result)
    return failures


def main(experiment: str | None = None) -> None:
    """Download configured SWOT files in parallel and trim them to the region."""
    if experiment is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("experiment")
        experiment = cast(str, parser.parse_args().experiment)

    failures = download_files(load_settings(experiment))
    if failures:
        print(
            "\nstatus: failed\n"
            f"files_failed: {len(failures)}"
        )
        for message in failures:
            print(message)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
