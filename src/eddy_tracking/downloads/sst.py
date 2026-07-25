"""Download AQUA MODIS SST subsets through OPeNDAP."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import earthaccess
import xarray as xr
from earthaccess import DataGranule

from eddy_tracking.authentication import (
    configure_obdaac_opendap_auth,
    login_earthdata,
)


_OPENDAP_BASE_URL = "https://oceandata.sci.gsfc.nasa.gov/opendap/MODISA/L3SMI"

SST_FILENAME_RE = re.compile(
    r"^AQUA_MODIS\.\d{8}_\d{8}\.L3m\.8D\.SST\.sst\.4km\.nc$"
)


def infer_lat_lon_names(ds: xr.Dataset) -> tuple[str, str]:
    """
    SST uses 'lat'/'lon', SSS uses 'latitude'/'longitude' - this normalizes
    both so downstream code does not need to know which product is loaded.
    """
    lat_name = next((n for n in ("lat", "latitude") if n in ds.coords), None)
    lon_name = next((n for n in ("lon", "longitude") if n in ds.coords), None)
    if lat_name is None or lon_name is None:
        raise KeyError(f"Cannot infer lat/lon coords from {list(ds.coords)}")
    return lat_name, lon_name


def subset_to_bbox_ds(
    ds: xr.Dataset,
    output_path: Path,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
) -> None:
    """
    Handles both ascending and descending latitude coordinates. Calls
    ``.load()`` before writing because remote xarray datasets need
    to be materialized before netcdf serialization.
    """
    lat_name, lon_name = infer_lat_lon_names(ds)
    lat_vals = ds[lat_name].values
    lat_ascending = lat_vals[0] < lat_vals[-1]

    lat_slice = (
        slice(lat_range[0], lat_range[1]) if lat_ascending
        else slice(lat_range[1], lat_range[0])
    )
    lon_slice = slice(lon_range[0], lon_range[1])

    subset = ds.sel({lat_name: lat_slice, lon_name: lon_slice})
    subset.load()
    tmp_path = output_path.with_suffix(".tmp.nc")
    subset.to_netcdf(tmp_path)
    tmp_path.rename(output_path)


def download_aqua_sst_8d_4km(
    date_range: tuple[str, str],
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    out_dir: Path,
    collection_id: str,
) -> int:
    """
    Download AQUA MODIS 8-day 4km SST subsets through OPeNDAP.

    Returns the number of new files saved. Skips files already in out_dir.
    """
    print("Downloading SST (AQUA MODIS L3m 8D 4km)")
    login_earthdata()
    configure_obdaac_opendap_auth()

    granules = earthaccess.search_data(
        concept_id=collection_id,
        temporal=(date_range[0], date_range[1]),
        count=-1,
    )
    print(f"CMR returned {len(granules)} granules")

    if not granules:
        return 0

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matches: list[tuple[str, DataGranule]] = []
    for granule in granules:
        for url in granule.data_links():
            filename = url.split("?", 1)[0].rsplit("/", 1)[-1]
            if SST_FILENAME_RE.match(filename):
                if not (out_dir / filename).exists():
                    matches.append((filename, granule))
                break

    if not matches:
        return 0

    saved = 0
    for filename, granule in sorted(matches, key=lambda match: match[0]):
        out_path = out_dir / filename
        opendap_url = _opendap_url(granule, filename)
        with xr.open_dataset(opendap_url, engine="netcdf4") as ds:
            subset_to_bbox_ds(ds, out_path, lon_range, lat_range)
        saved += 1

    return saved


def _opendap_url(granule: DataGranule, filename: str) -> str:
    related_urls = granule.get("umm", {}).get("RelatedUrls", [])
    for related_url in related_urls:
        if related_url.get("Subtype") == "OPENDAP DATA":
            return related_url["URL"]

    start_date = filename.split(".")[1].split("_")[0]
    return (
        f"{_OPENDAP_BASE_URL}/{start_date[:4]}/{start_date[4:8]}/{filename}"
    )


def main(experiment: str | None = None) -> None:
    """Download configured SST files."""
    if experiment is None:
        experiment = _parse_args().experiment

    from utils.config import load_config, resolve_data_dir

    cfg = load_config(experiment)
    n_saved = download_aqua_sst_8d_4km(
        date_range=tuple(cfg["base"]["time"]["rrs_date_range"]),
        lon_range=tuple(cfg["base"]["region"]["lon_range"]),
        lat_range=tuple(cfg["base"]["region"]["lat_range"]),
        out_dir=resolve_data_dir(cfg, "sst_dir"),
        collection_id=cfg["base"]["download"]["sst"]["collection_id"],
    )
    print(f"Done. {n_saved} SST files saved.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    return parser.parse_args()


if __name__ == "__main__":
    main()
