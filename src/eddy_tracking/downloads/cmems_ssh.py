from __future__ import annotations

import sys

import copernicusmarine
import numpy as np
import pandas as pd
import xarray as xr
from zarr.storage import FSStore

from eddy_tracking.config import load_config, resolve_data_dir
from eddy_tracking.preprocess.streamline import KM_PER_DEG_LAT
from eddy_tracking.utils.files import write_netcdf_atomically


def main(experiment: str) -> None:
    cfg = load_config(experiment)
    # The stage runs by default, and an experiment without this block keeps only the AVISO record
    if "cmems_ssh" not in cfg["base"]["download"]:
        print(
            "status: skipped\n"
            "reason: no_cmems_ssh_config"
        )
        return
    ssh_cfg = cfg["base"]["download"]["cmems_ssh"]
    swot_dir = resolve_data_dir(cfg, "swot_dir")

    # The newest AVISO file in the directory gives the grid, the file layout, and the open_ocean label, so the Copernicus days read exactly like the AVISO days they follow
    with xr.open_dataset(sorted(swot_dir.glob("dt_global_allsat_phy_l4_*.nc"))[-1]) as dataset:
        aviso = dataset.load()
    longitude = aviso["longitude"]
    latitude = aviso["latitude"]
    open_ocean = aviso["open_ocean"].to_numpy().astype(bool)
    # Meters per grid step north and east, and the Coriolis parameter, each (n_lat, 1) or a scalar
    latitude_rad = np.radians(latitude.to_numpy())[:, np.newaxis]
    step_north_m = KM_PER_DEG_LAT * 1000 * float(latitude[1] - latitude[0])
    step_east_m = KM_PER_DEG_LAT * 1000 * float(longitude[1] - longitude[0]) * np.cos(latitude_rad)
    coriolis = 2 * 7.2921e-5 * np.sin(latitude_rad)

    catalogue = copernicusmarine.describe(dataset_id=ssh_cfg["dataset_id"], disable_progress_bar=True)
    [version] = [version for version in catalogue.products[0].datasets[0].versions if version.label == ssh_cfg["dataset_version"]]
    # "arco-geo-series" is the store chunked one day at a time: https://s3.waw3-1.cloudferro.com/<bucket>/arco/<product>/<dataset>_<version>/timeChunked.zarr
    [store_url] = [str(service.uri) for service in version.parts[0].services if service.service_name == "arco-geo-series"]
    endpoint, _, store_path = store_url.removeprefix("https://").partition("/")
    # The bucket is public. It answers 403 for a day it does not hold; s3fs raises that as PermissionError, which zarr treats as a missing chunk and fills with NaN
    store = FSStore(f"s3://{store_path}", anon=True, client_kwargs={"endpoint_url": f"https://{endpoint}"})
    maps = xr.open_zarr(store, consolidated=True)[["sla", "adt", "ugos", "vgos", "ugosa", "vgosa"]]
    # Half a degree past the region on each side, so every AVISO grid point has Copernicus cells around it
    maps = maps.sel(
        longitude=slice(float(longitude[0]) - 0.5, float(longitude[-1]) + 0.5),
        latitude=slice(float(latitude[0]) - 0.5, float(latitude[-1]) + 0.5),
    )
    print(
        "status: downloading_cmems_ssh\n"
        f"dataset_id: {ssh_cfg['dataset_id']}\n"
        f"dataset_version: {ssh_cfg['dataset_version']}\n"
        f"date_range: {ssh_cfg['date_range']}"
    )

    n_saved = n_skipped = n_missing = 0
    for day in pd.date_range(*ssh_cfg["date_range"]):
        out_path = swot_dir / f"nrt_global_allsat_swos_phy_l4_{day:%Y%m%d}.nc"
        if out_path.exists():
            n_skipped += 1
            continue
        fields = maps.sel(time=day).load()
        if not fields["adt"].notnull().any():
            print(
                "status: no_map\n"
                f"date: {day.date()}"
            )
            n_missing += 1
            continue
        # Copernicus cell centers sit half a cell off the AVISO grid points (-80.9375 against -81 and -80.875), so interpolate linearly onto the AVISO grid
        fields = fields.interp(longitude=longitude, latitude=latitude)
        # AVISO's relative_vorticity is the Rossby number of the anomaly velocities, vorticity over f; centered differences of AVISO's own ugosa and vgosa reproduce it with a correlation of 0.999 and 4% less amplitude
        rossby_number = (
            np.gradient(fields["vgosa"].to_numpy(), axis=1) / step_east_m
            - np.gradient(fields["ugosa"].to_numpy(), axis=0) / step_north_m
        ) / coriolis

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
        daily.attrs = {
            "title": maps.attrs["title"],
            "platform": maps.attrs["platform"],
            "product_version": maps.attrs["product_version"],
            "dataset_id": ssh_cfg["dataset_id"],
            "comment": "Interpolated linearly onto the grid of the AVISO files in this directory; relative_vorticity is derived from ugosa and vgosa; open_ocean is the AVISO label",
        }
        write_netcdf_atomically(daily, out_path)
        n_saved += 1

    print(
        "status: complete\n"
        f"files_saved: {n_saved}\n"
        f"files_skipped: {n_skipped}\n"
        f"days_without_map: {n_missing}"
    )


if __name__ == "__main__":
    main(sys.argv[1])
