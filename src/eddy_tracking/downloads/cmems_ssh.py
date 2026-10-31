# Read a Copernicus Marine gridded SSH dataset from its public store and write one file per day in the layout of the AVISO files, so every tool reads both sources alike.
# With --grid-from the maps are interpolated onto the grid of an AVISO folder and take its open_ocean label (the canonical study continues AVISO this way); with --exclude-box they keep the product's own grid and get their own open-ocean mask.

from __future__ import annotations

import argparse
from pathlib import Path

import copernicusmarine
import numpy as np
import pandas as pd
import xarray as xr
from zarr.storage import FsspecStore

from eddy_tracking.downloads.swot import mask_open_ocean
from eddy_tracking.preprocess.streamline import KM_PER_DEG_LAT
from eddy_tracking.utils.cli import add_region_arguments, write_settings
from eddy_tracking.utils.files import write_netcdf_atomically


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="eddy-tracking download-cmems-ssh")
    parser.add_argument("--dataset", required=True, help="Copernicus Marine dataset id")
    parser.add_argument("--version", required=True, help="dataset version label, such as 202411")
    add_region_arguments(parser)
    grid = parser.add_mutually_exclusive_group(required=True)
    grid.add_argument("--grid-from", type=Path, help="AVISO SSH folder whose newest file gives the grid and the open_ocean label")
    grid.add_argument("--exclude-box", nargs=4, type=float, metavar=("WEST", "EAST", "SOUTH", "NORTH"), help="keep the product's grid; mask this box, invalid cells and their 8-cell coast buffer")
    # Despite the names, arco-geo-series is chunked in time (a few days of a large map per block) and arco-time-series by area (a small tile over many days); arco-geo-series read this region at about 0.7 s per day on 2026-10-30
    parser.add_argument("--store", choices=("arco-geo-series", "arco-time-series"), default="arco-geo-series")
    parser.add_argument("--block-days", type=int, default=31, help="days loaded from the store at once")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    write_settings(args.out, "download-cmems-ssh", args, operational={"store", "block_days"})
    args.out.mkdir(parents=True, exist_ok=True)

    catalogue = copernicusmarine.describe(dataset_id=args.dataset, disable_progress_bar=True)
    [version] = [version for version in catalogue.products[0].datasets[0].versions if version.label == args.version]
    # Each store sits at https://s3.waw3-1.cloudferro.com/<bucket>/arco/<product>/<dataset>_<version>/<layout>.zarr
    [store_url] = [str(service.uri) for service in version.parts[0].services if service.service_name == args.store]
    endpoint, _, store_path = store_url.removeprefix("https://").partition("/")
    # The bucket is public. It answers 403 for a day it does not hold; s3fs raises that as PermissionError, which allowed_exceptions makes zarr treat as a missing chunk and fill with NaN
    store = FsspecStore.from_url(
        f"s3://{store_path}", storage_options={"anon": True, "client_kwargs": {"endpoint_url": f"https://{endpoint}"}}, read_only=True,
        allowed_exceptions=(FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError),
    )
    maps = xr.open_zarr(store, consolidated=True)[["sla", "adt", "ugos", "vgos", "ugosa", "vgosa"]]

    if args.grid_from:
        # The newest AVISO file gives the grid, the file layout, and the open_ocean label, so the Copernicus days read exactly like the AVISO days they follow
        with xr.open_dataset(sorted(args.grid_from.glob("dt_global_allsat_phy_l4_*.nc"))[-1]) as dataset:
            aviso = dataset.sel(longitude=slice(*args.lon), latitude=slice(*args.lat)).load()
        longitude = aviso["longitude"]
        latitude = aviso["latitude"]
        # Half a degree past the region on each side, so every AVISO grid point has Copernicus cells around it
        maps = maps.sel(
            longitude=slice(float(longitude[0]) - 0.5, float(longitude[-1]) + 0.5),
            latitude=slice(float(latitude[0]) - 0.5, float(latitude[-1]) + 0.5),
        )
    else:
        aviso = None
        maps = maps.sel(longitude=slice(*args.lon), latitude=slice(*args.lat))
        longitude = maps["longitude"]
        latitude = maps["latitude"]
    # Meters per grid step north and east, and the Coriolis parameter, each (n_lat, 1) or a scalar
    latitude_rad = np.radians(latitude.to_numpy())[:, np.newaxis]
    step_north_m = KM_PER_DEG_LAT * 1000 * float(latitude[1] - latitude[0])
    step_east_m = KM_PER_DEG_LAT * 1000 * float(longitude[1] - longitude[0]) * np.cos(latitude_rad)
    coriolis = 2 * 7.2921e-5 * np.sin(latitude_rad)
    print(
        "status: downloading_cmems_ssh\n"
        f"dataset_id: {args.dataset}\n"
        f"dataset_version: {args.version}\n"
        f"date_range: {args.dates}\n"
        f"grid: {'from ' + str(args.grid_from) if args.grid_from else 'product'}"
    )

    days = pd.date_range(*args.dates)
    n_saved = n_skipped = n_missing = 0
    for block_start in range(0, len(days), args.block_days):
        block = days[block_start:block_start + args.block_days]
        out_paths = {day: args.out / f"{args.dataset}_{day:%Y%m%d}.nc" for day in block}
        todo = [day for day, out_path in out_paths.items() if not out_path.exists()]
        n_skipped += len(block) - len(todo)
        if not todo:
            continue
        block_maps = maps.sel(time=slice(todo[0], todo[-1])).load()

        for day in todo:
            # A day past the end of the store's time axis has no map, like a day the store fills with NaN
            fields = block_maps.sel(time=day) if day in block_maps.indexes["time"] else None
            if fields is None or not fields["adt"].notnull().any():
                print(
                    "status: no_map\n"
                    f"date: {day.date()}"
                )
                n_missing += 1
                continue
            if aviso is not None:
                # Copernicus cell centers sit half a cell off the AVISO grid points (-80.9375 against -81 and -80.875), so interpolate linearly onto the AVISO grid
                fields = fields.interp(longitude=longitude, latitude=latitude)
            # AVISO's relative_vorticity is the Rossby number of the anomaly velocities, vorticity over f; centered differences of AVISO's own ugosa and vgosa reproduce it with a correlation of 0.999 and 4% less amplitude
            rossby_number = (
                np.gradient(fields["vgosa"].to_numpy(), axis=1) / step_east_m
                - np.gradient(fields["ugosa"].to_numpy(), axis=0) / step_north_m
            ) / coriolis

            if aviso is not None:
                open_ocean = aviso["open_ocean"].to_numpy().astype(bool)
                daily = aviso.assign_coords(time=[day])
                for name in ("sla", "ugosa", "vgosa"):
                    # (n_lat, n_lon) -> (1, n_lat, n_lon)
                    daily[name] = daily[name].copy(data=fields[name].to_numpy()[np.newaxis])
                for name, values in (
                    ("adt", fields["adt"].to_numpy()),
                    ("ugos", fields["ugos"].to_numpy()),
                    ("vgos", fields["vgos"].to_numpy()),
                    ("relative_vorticity", rossby_number),
                ):
                    daily[f"{name}_full"] = daily[f"{name}_full"].copy(data=values[np.newaxis])
                    daily[name] = daily[name].copy(data=np.where(open_ocean, values, np.nan)[np.newaxis])
                comment = "Interpolated linearly onto the grid of the AVISO files in this directory; relative_vorticity is derived from ugosa and vgosa; open_ocean is the AVISO label"
            else:
                daily = fields.expand_dims("time").assign(relative_vorticity=(("time", "latitude", "longitude"), rossby_number[np.newaxis]))
                west, east, south, north = args.exclude_box
                daily = mask_open_ocean(daily, (west, east), (south, north))
                comment = "The product's own grid; relative_vorticity is derived from ugosa and vgosa; open_ocean keeps cells 8 or more cells from invalid data and outside the exclusion box"
            daily.attrs = {
                "title": maps.attrs["title"],
                "platform": maps.attrs["platform"],
                "product_version": maps.attrs["product_version"],
                "dataset_id": args.dataset,
                "comment": comment,
            }
            write_netcdf_atomically(daily, out_paths[day])
            n_saved += 1

    status = "incomplete" if n_missing else "complete"
    print(
        f"status: {status}\n"
        f"files_saved: {n_saved}\n"
        f"files_skipped: {n_skipped}\n"
        f"days_without_map: {n_missing}"
    )
    # Tracking would run straight across the gap, so exit non-zero and stop a run.sh here; a rerun keeps the saved days and tries the missing ones again
    if n_missing:
        raise SystemExit(f"{n_missing} day(s) had no Copernicus map")
