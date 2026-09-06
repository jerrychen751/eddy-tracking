"""Download one Copernicus Marine dataset over the study region, one NetCDF per calendar month."""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
from pathlib import Path
from typing import cast

import copernicusmarine

from eddy_tracking.downloads.auth import load_cmems_credentials


def download_cmems_dataset(
    date_range: tuple[str, str],
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    out_dir: Path,
    dataset_id: str,
    dataset_version: str,
) -> int:
    """
    Download every field of the dataset over the region and dates, one zlib-compressed NetCDF per calendar month named <out_dir name>_<first day>_<last day>.nc, where the first and last month are clipped to date_range. Creates out_dir and skips a month whose file exists, so a changed range start leaves an earlier partial month in place. The toolbox writes through a temporary path, so an interrupted transfer leaves no partial file. lon_range and lat_range are (low, high) in degrees east and degrees north. Returns the number of files written.
    """
    print(
        "status: downloading_cmems\n"
        f"dataset_id: {dataset_id}\n"
        f"dataset_version: {dataset_version}"
    )
    username, password = load_cmems_credentials()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    start = dt.date.fromisoformat(date_range[0])
    stop = dt.date.fromisoformat(date_range[1])
    while start <= stop:
        end = min(stop, start.replace(day=calendar.monthrange(start.year, start.month)[1]))
        filename = f"{out_dir.name}_{start}_{end}.nc"
        if not (out_dir / filename).exists():
            copernicusmarine.subset(
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                username=username,
                password=password,
                minimum_longitude=lon_range[0],
                maximum_longitude=lon_range[1],
                minimum_latitude=lat_range[0],
                maximum_latitude=lat_range[1],
                start_datetime=f"{start}T00:00:00",
                end_datetime=f"{end}T23:59:59",
                output_directory=out_dir,
                output_filename=filename,
                netcdf_compression_level=4,
                disable_progress_bar=True,
            )
            print(f"output_file: {filename}")
            saved += 1
        start = end + dt.timedelta(days=1)
    return saved


def main(experiment: str | None = None) -> None:
    """Download the months of the plankton dataset that cover the eddy tracking window of the experiment."""
    if experiment is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("experiment")
        experiment = cast(str, parser.parse_args().experiment)

    from utils.config import load_config, resolve_data_dir

    cfg = load_config(experiment)
    n_saved = download_cmems_dataset(
        date_range=tuple(cfg["base"]["time"]["eddy_date_range"]),
        lon_range=tuple(cfg["base"]["region"]["lon_range"]),
        lat_range=tuple(cfg["base"]["region"]["lat_range"]),
        out_dir=resolve_data_dir(cfg, "plankton_dir"),
        dataset_id=cfg["base"]["download"]["cmems"]["dataset_id"],
        dataset_version=cfg["base"]["download"]["cmems"]["dataset_version"],
    )
    print(
        "status: complete\n"
        f"cmems_files_saved: {n_saved}"
    )


if __name__ == "__main__":
    main()
