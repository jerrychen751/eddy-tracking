"""
Download PACE OCI L3 mapped files.

OB.DAAC OPenDAP provides server-side subsetting.

The product code selects the suite: "AOP" holds Rrs, "BGC" holds chlor_a, poc, pic, and carbon_phyto. Both use the same granule naming, so one download path serves them, and each run names the CMR collection id of its product with --collection-id.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import earthaccess
import xarray as xr
from earthaccess import DataGranule

from eddy_tracking.downloads.auth import (
    configure_obdaac_opendap_auth,
    login_earthdata,
    open_obdaac_dataset,
)
from eddy_tracking.utils.cli import add_region_arguments, write_settings
from eddy_tracking.utils.files import write_netcdf_atomically


def search_pace_l3_granules(
    date_range: tuple[str, str],
    collection_id: str,
    version: str,
    temporal_res: str = "DAY",
    product: str = "AOP",
) -> dict[str, tuple[str, DataGranule]]:
    """
    Search CMR for the granules of one PACE L3 collection in a date range and keep those whose filename carries the requested temporal resolution, product, version, and the 4km grid.

    Returns {date_key: (filename, granule)}, where date_key is the granule start date "20240929" and filename is "PACE_OCI.20240929_20241006.L3m.8D.BGC.V3_2.4km.nc". A granule from another reprocessing fails the match, so one directory never holds two versions of the same date.
    """
    # The version segment is pinned, so a new reprocessing fails the match instead of mixing two versions in one directory.
    granule_pattern = {
        "DAY": r"PACE_OCI\.(\d{{8}})\.L3m\.DAY\.{product}\.{version}\.4km\.nc",
        "8D": r"PACE_OCI\.(\d{{8}})_(\d{{8}})\.L3m\.8D\.{product}\.{version}\.4km\.nc",
    }[temporal_res]
    regex = re.compile(granule_pattern.format(product=product, version=re.escape(version)))
    results = earthaccess.search_data(
        concept_id=collection_id,
        temporal=(date_range[0], date_range[1]),
        count=5000,
    )
    matches: dict[str, tuple[str, DataGranule]] = {}
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

    Drops 'palette' (visualization artifact). Writes with zlib level 4 and atomic rename.
    """
    subset = ds.sel(
        lat=slice(lat_range[1], lat_range[0]),
        lon=slice(lon_range[0], lon_range[1]),
    )
    if "palette" in subset:
        subset = subset.drop_vars("palette")
    subset.load()
    write_netcdf_atomically(subset, out_path, encoding={v: {"zlib": True, "complevel": 4} for v in subset.data_vars})


def download_pace_l3(
    date_range: tuple[str, str],
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    out_dir: Path,
    collection_id: str,
    version: str,
    temporal_res: str = "DAY",
    product: str = "AOP",
) -> tuple[int, int, int]:
    """
    Search + download the granules of one PACE L3 collection matching date_range, temporal_res, product, and version.

    Creates out_dir and writes one region subset NetCDF per granule into it, named after the source granule. Skips files already in out_dir. lon_range and lat_range are (low, high) in degrees east and degrees north. Returns (saved, skipped, errors).
    """
    login_earthdata()
    configure_obdaac_opendap_auth()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    download_base_url = "https://oceandata.sci.gsfc.nasa.gov/opendap/PACE_OCI/L3SMI"

    matches = search_pace_l3_granules(date_range, collection_id, version, temporal_res, product)
    print(
        f"matched_granules: {len(matches)}\n"
        f"collection_id: {collection_id}\n"
        f"temporal_resolution: {temporal_res}\n"
        f"product: {product}\n"
        f"version: {version}\n"
        "spatial_resolution_km: 4"
    )

    saved = skipped = errors = 0
    use_remote_subset = True
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
            if use_remote_subset:
                try:
                    with open_obdaac_dataset(opendap_url) as ds:
                        _subset_and_save(ds, out_path, lon_range, lat_range)
                except OSError:
                    use_remote_subset = False
            if not use_remote_subset:
                print(
                    "status: downloading_global_fallback\n"
                    f"file: {filename}"
                )
                with TemporaryDirectory(prefix="pace-global-") as temporary_dir:
                    [downloaded_path] = earthaccess.download(
                        [granule],
                        local_path=temporary_dir,
                        threads=1,
                    )
                    with xr.open_dataset(downloaded_path) as ds:
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


def main(argv: list[str]) -> None:
    """Download and save the PACE files of one --product, exiting if any date fails."""
    parser = argparse.ArgumentParser(prog="eddy-tracking download-pace")
    add_region_arguments(parser)
    parser.add_argument("--product", choices=["AOP", "BGC"], required=True, help="AOP holds Rrs, BGC holds chlor_a, poc, pic and carbon_phyto")
    parser.add_argument("--collection-id", required=True, help="CMR collection id of the product, such as C4184125822-OB_CLOUD")
    parser.add_argument("--version", required=True, help="reprocessing version in the granule names, such as V3_2")
    parser.add_argument("--temporal-resolution", choices=["DAY", "8D"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    write_settings(args.out, "download-pace", args, operational=set())
    args.out.mkdir(parents=True, exist_ok=True)

    n_saved, n_skipped, n_errors = download_pace_l3(
        date_range=tuple(args.dates),
        lon_range=tuple(args.lon),
        lat_range=tuple(args.lat),
        out_dir=args.out,
        collection_id=args.collection_id,
        version=args.version,
        temporal_res=args.temporal_resolution,
        product=args.product,
    )

    print(
        "status: download_finished\n"
        f"files_saved: {n_saved}\n"
        f"files_skipped: {n_skipped}\n"
        f"errors: {n_errors}"
    )
    if n_errors:
        raise SystemExit(f"{n_errors} date(s) failed to download")
