"""
Download PACE OCI L3 mapped files.

OB.DAAC OPenDAP provides server-side subsetting.

The product code selects the suite: "AOP" holds Rrs, "BGC" holds chlor_a, poc, pic, and carbon_phyto.
Both use the same granule naming and the same OB.DAAC collection pattern, so one download path serves them.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import earthaccess
import xarray as xr

from eddy_tracking.downloads.auth import (
    configure_obdaac_opendap_auth,
    login_earthdata,
    open_obdaac_dataset,
)


def search_pace_l3_granules(
    date_range: tuple[str, str],
    temporal_res: str = "DAY",
    product: str = "AOP",
) -> dict[str, tuple[str, object]]:
    """
    Search CMR for PACE L3 granules of one product in a date range and filter to matching temporal resolution and 4km.

    Returns {date_key: (filename, granule)}, where date_key is the granule start date "20240929" and filename is "PACE_OCI.20240929_20241006.L3m.8D.BGC.V3_2.4km.nc".
    """
    # The version segment stays open so a new reprocessing does not break the match.
    granule_pattern = {
        "DAY": r"PACE_OCI\.(\d{{8}})\.L3m\.DAY\.{product}\..*\.4km\.nc",
        "8D": r"PACE_OCI\.(\d{{8}})_(\d{{8}})\.L3m\.8D\.{product}\..*\.4km\.nc",
    }[temporal_res]
    regex = re.compile(granule_pattern.format(product=product))
    results = earthaccess.search_data(
        short_name=f"PACE_OCI_L3M_{product}",
        temporal=(date_range[0], date_range[1]),
        count=5000,
    )
    matches: dict[str, tuple[str, object]] = {}
    for granule in results:
        for link in granule.data_links():
            filename = link.split("/")[-1]
            m = regex.match(filename)
            if m:
                matches[m.group(1)] = (filename, granule)
                break
    return matches


def _subset_and_save(
    ds: xr.Dataset,
    out_path: Path,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
) -> None:
    """
    L3 latitude runs 90 to -90 (descending), so the lat slice goes high-to-low.

    Drops 'palette' (visualization artifact).
    Writes with zlib level 4 and atomic rename.
    """
    subset = ds.sel(
        lat=slice(lat_range[1], lat_range[0]),
        lon=slice(lon_range[0], lon_range[1]),
    )
    if "palette" in subset:
        subset = subset.drop_vars("palette")
    subset.load()
    tmp_path = out_path.with_suffix(".tmp.nc")
    encoding = {v: {"zlib": True, "complevel": 4} for v in subset.data_vars}
    try:
        subset.to_netcdf(tmp_path, encoding=encoding)
        tmp_path.rename(out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def download_pace_l3(
    date_range: tuple[str, str],
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    out_dir: Path,
    temporal_res: str = "DAY",
    product: str = "AOP",
) -> tuple[int, int, int]:
    """
    Search + download PACE L3 granules matching date_range, temporal_res, and product.

    Creates out_dir and writes one region subset NetCDF per granule into it, named after the source granule.
    Skips files already in out_dir.
    lon_range and lat_range are (low, high) in degrees east and degrees north.
    Returns (saved, skipped, errors).
    """
    login_earthdata()
    configure_obdaac_opendap_auth()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    download_base_url = "https://oceandata.sci.gsfc.nasa.gov/opendap/PACE_OCI/L3SMI"

    matches = search_pace_l3_granules(date_range, temporal_res, product)
    print(
        f"matched_granules: {len(matches)}\n"
        f"temporal_resolution: {temporal_res}\n"
        f"product: {product}\n"
        "spatial_resolution_km: 4"
    )

    saved = skipped = errors = 0
    for date_key, (filename, granule) in sorted(matches.items()):
        out_path = out_dir / filename
        if out_path.exists():
            skipped += 1
            continue
        try:
            opendap_url = next(
                (
                    ru["URL"]
                    for ru in granule.get("umm", {}).get("RelatedUrls", [])
                    if ru.get("Subtype") == "OPENDAP DATA"
                ),
                None,
            )
            if opendap_url is None:
                # Recent granules sit on Hyrax but omit the OPENDAP DATA link, so build the stable {base}/YYYY/MMDD/filename URL instead.
                # Field 1 of the filename is "20240929_20241006" for 8D and "20240929" for DAY, so its first 8 characters give the start date either way.
                start = filename.split(".")[1]
                opendap_url = f"{download_base_url}/{start[:4]}/{start[4:8]}/{filename}"
            # Subsetting over OPeNDAP transfers only the ROI hyperslab (~42 MB gzipped) instead of the full global granule (~2.3 GB).
            with open_obdaac_dataset(opendap_url) as ds:
                _subset_and_save(ds, out_path, lon_range, lat_range)
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(
                "status: saved\n"
                f"file: {filename}\n"
                f"size_mb: {size_mb:.1f}"
            )
            saved += 1
        except Exception as exc:
            print(
                "status: error\n"
                f"date: {date_key}\n"
                f"error: {exc}",
                file=sys.stderr,
            )
            errors += 1

    return saved, skipped, errors


def main(experiment: str | None = None) -> None:
    """Download and save PACE files, exiting if any date fails."""
    if experiment is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("experiment")
        experiment = parser.parse_args().experiment

    from utils.config import load_config, resolve_data_dir

    cfg = load_config(experiment)
    longitude_range = tuple(cfg["base"]["region"]["lon_range"])
    latitude_range = tuple(cfg["base"]["region"]["lat_range"])
    date_range = tuple(cfg["base"]["time"]["rrs_date_range"])
    temporal_resolution = cfg["base"]["download"]["pace"].get(
        "temporal_resolution", "DAY"
    )

    n_saved, n_skipped, n_errors = download_pace_l3(
        date_range=date_range,
        lon_range=longitude_range,
        lat_range=latitude_range,
        out_dir=resolve_data_dir(cfg, "pace_dir"),
        temporal_res=temporal_resolution,
    )

    print(
        "status: download_finished\n"
        f"files_saved: {n_saved}\n"
        f"files_skipped: {n_skipped}\n"
        f"errors: {n_errors}"
    )
    if n_errors:
        raise SystemExit(f"{n_errors} date(s) failed to download")


if __name__ == "__main__":
    main()
