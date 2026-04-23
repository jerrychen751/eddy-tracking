"""
PACE OCI L3 mapped Rrs download helpers. Pure functions — no module globals.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import earthaccess
import xarray as xr


GRANULE_RE = {
    "DAY": re.compile(r"PACE_OCI\.(\d{8})\.L3m\.DAY\.RRS\..*\.Rrs\.4km\.nc"),
    "8D": re.compile(r"PACE_OCI\.(\d{8})_(\d{8})\.L3m\.8D\.RRS\..*\.Rrs\.4km\.nc"),
}


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


def download_pace_l3_granule(
    granule: object,
    out_path: Path,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
) -> None:
    """Download a single granule to a temp dir, subset, save to out_path."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = earthaccess.download([granule], local_path=tmp_dir)
        if not paths:
            raise RuntimeError("earthaccess returned no file")
        raw_path = Path(paths[0])
        with xr.open_dataset(raw_path) as ds:
            _subset_and_save(ds, out_path, lon_range, lat_range)


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
    earthaccess.login()
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
            download_pace_l3_granule(granule, out_path, lon_range, lat_range)
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"Saved: {filename} ({size_mb:.1f} MB)")
            saved += 1
        except Exception as exc:
            print(f"Error on {date_key}: {exc}", file=sys.stderr)
            errors += 1

    return saved, skipped, errors
