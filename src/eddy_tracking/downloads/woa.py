# Download the World Ocean Atlas 2023 monthly temperature and salinity climatology over the study region, the seasonal baseline for temperature anomalies

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr

from eddy_tracking.utils.cli import write_settings


# woa23_monthly.nc holds t_an (in situ temperature, °C) and s_an (practical salinity) on (month, depth, lat, lon); one OPeNDAP request per month and variable reads only the region
def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="eddy-tracking download-woa")
    # A climatology has no dates, so this takes the region flags of add_region_arguments without --dates
    parser.add_argument("--lon", nargs=2, type=float, required=True, metavar=("WEST", "EAST"), help="degrees east")
    parser.add_argument("--lat", nargs=2, type=float, required=True, metavar=("SOUTH", "NORTH"), help="degrees north")
    parser.add_argument("--opendap-root", required=True, help="such as https://www.ncei.noaa.gov/thredds-ocean/dodsC/woa23/DATA")
    parser.add_argument("--period", required=True, help="WOA averaging period, such as decav91C0 for 1991 to 2020")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    write_settings(args.out, "download-woa", args, operational=set())
    args.out.mkdir(parents=True, exist_ok=True)

    lon0, lon1 = args.lon
    lat0, lat1 = args.lat
    path = args.out / "woa23_monthly.nc"
    if path.exists():
        print(f"status: skipped\noutput_file: {path.name}")
        return
    months = []
    for month in range(1, 13):
        fields = []
        for variable, folder in (("t", "temperature"), ("s", "salinity")):
            url = f"{args.opendap_root}/{folder}/netcdf/{args.period}/1.00/woa23_{args.period}_{variable}{month:02d}_01.nc"
            # The time axis counts months since 1991, which the climatology does not need
            with xr.open_dataset(url, decode_times=False) as dataset:
                fields.append(dataset[f"{variable}_an"].sel(lon=slice(lon0, lon1), lat=slice(lat0, lat1)).squeeze("time", drop=True).load())
        months.append(xr.merge(fields))
    climatology = xr.concat(months, dim=pd.Index(range(1, 13), name="month"))
    climatology.attrs = {"source": f"World Ocean Atlas 2023, {args.period}, 1.00 degree, objectively analyzed monthly means"}
    tmp = path.with_suffix(".nc.part")
    climatology.to_netcdf(tmp)
    tmp.rename(path)
    print(f"status: complete\noutput_file: {path.name}\ndepth_levels: {climatology.sizes['depth']}")
