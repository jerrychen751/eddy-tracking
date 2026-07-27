"""
Download PACE OCI L3 mapped Rrs files.

OB.DAAC OPenDAP provides server-side subsetting.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import earthaccess
import xarray as xr

from eddy_tracking.utils.authentication import (
    configure_obdaac_opendap_auth,
    login_earthdata,
)


GRANULE_RE = {
    "DAY": re.compile(r"PACE_OCI\.(\d{8})\.L3m\.DAY\.RRS\..*\.Rrs\.4km\.nc"),
    "8D": re.compile(r"PACE_OCI\.(\d{8})_(\d{8})\.L3m\.8D\.RRS\..*\.Rrs\.4km\.nc"),
}

# OB.DAAC Hyrax OPeNDAP root for PACE L3 mapped products, used to build a URL
# when a granule's CMR metadata omits its OPENDAP DATA link (see download loop).
_DOWNLOAD_BASE_URL = "https://oceandata.sci.gsfc.nasa.gov/opendap/PACE_OCI/L3SMI"


def search_pace_l3_granules(
    date_range: tuple[str, str],
    temporal_res: str = "DAY",
) -> dict[str, tuple[str, object]]:
    """
    Search CMR for PACE L3 Rrs granules in a date range and filter to
    matching temporal resolution and 4km. Returns {date_key: (filename, granule)}.
    """
    regex = GRANULE_RE[temporal_res]
    results = earthaccess.search_data(
        short_name="PACE_OCI_L3M_RRS",
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
    L3 latitude runs 90→-90 (descending), so lat slice goes high-to-low.
    Drops 'palette' (visualization artifact). Writes with zlib level 4
    and atomic rename.
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
    subset.to_netcdf(tmp_path, encoding=encoding)
    tmp_path.rename(out_path)


def download_pace_l3(
    date_range: tuple[str, str],
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    out_dir: Path,
    temporal_res: str = "DAY",
) -> tuple[int, int, int]:
    """
    Search + download PACE L3 Rrs granules matching date_range and temporal_res.
    Skips files already in out_dir. Returns (saved, skipped, errors).
    """
    login_earthdata()
    configure_obdaac_opendap_auth()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matches = search_pace_l3_granules(date_range, temporal_res)
    print(f"Matched {len(matches)} {temporal_res} 4km granules")

    saved = skipped = errors = 0
    for date_key, (filename, granule) in sorted(matches.items()):
        out_path = out_dir / filename
        if out_path.exists():
            skipped += 1
            continue
        try:
            # OB.DAAC serves every granule via OPeNDAP; opening that URL and
            # subsetting transfers only the ROI hyperslab (~42 MB gzipped)
            # instead of the full global granule (~2.3 GB). Prefer the OPENDAP
            # DATA link in the granule's CMR metadata; recent granules are on
            # the Hyrax server but sometimes omit that link, so fall back to the
            # stable {base}/YYYY/MMDD/filename pattern.
            opendap_url = next(
                (ru["URL"] for ru in granule.get("umm", {}).get("RelatedUrls", [])
                 if ru.get("Subtype") == "OPENDAP DATA"),
                None,
            )
            if opendap_url is None:
                start = filename.split(".")[1]  # PACE_OCI.<start>_<end>.L3m...
                opendap_url = f"{_DOWNLOAD_BASE_URL}/{start[:4]}/{start[4:8]}/{filename}"
            with xr.open_dataset(opendap_url, engine="netcdf4") as ds:
                _subset_and_save(ds, out_path, lon_range, lat_range)
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"Saved: {filename} ({size_mb:.1f} MB)")
            saved += 1
        except Exception as exc:
            print(f"Error on {date_key}: {exc}", file=sys.stderr)
            errors += 1

    return saved, skipped, errors


def main(experiment: str | None = None) -> None:
    """Download and save PACE files, exiting if any date fails."""
    if experiment is None:
        experiment = _parse_args().experiment

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

    print(f"Done. {n_saved} saved, {n_skipped} skipped, {n_errors} errors.")
    if n_errors:
        raise SystemExit(f"{n_errors} date(s) failed to download")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    return parser.parse_args()


if __name__ == "__main__":
    main()
