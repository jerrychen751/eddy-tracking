"""
Sample PACE L3 chlor_a on the same eddy and background pixels the SDP stage used.

This is the chlorophyll counterpart of `collocate_pace.py` plus `background.py`.
It supports the empirical-fraction baseline, where a pigment is a fraction of total chlorophyll a instead of an SDP retrieval.
Reading chlor_a on the SDP pixel sets keeps every eddy-day, mask, and background definition identical, so the two pigment fields differ only in how they turn light into pigment.

Writes `silver/pace_chl/eddy_bg_chlor_a.parquet`: one row per eddy-day with the eddy-interior and background means of chlor_a, and the same means of chlor_a**b for one exponent b per pigment.
A pigment written as a * chlor_a**b needs the mean of chlor_a**b over the pixel set, because the mean of a power is not the power of the mean.

Usage:
    python scripts/collocate_pace_chl.py <experiment> --exponents exponents.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from background import (  # noqa: E402
    SWOT_SEARCH_DAYS,
    compute_calm_mask_on_pace,
    find_nearest_swot_file,
    in_any_contour,
    index_swot_files_by_date,
    load_eddy_contours,
)
from utils.config import resolve_output_dir  # noqa: E402


def read_eddy_pixels(experiment: str) -> pd.DataFrame:
    """Every SDP eddy-interior pixel, with its track, date, polarity, and position."""
    frames = []
    for polarity, polarity_value in [("anticyclone", 0), ("cyclone", 1)]:
        pigment_dir = resolve_output_dir(experiment, "pigments", polarity)
        for path in sorted(pigment_dir.glob("*_pigments.parquet")):
            frame = pd.read_parquet(
                path, columns=["track_id", "date", "pixel_lon", "pixel_lat"]
            )
            frame["polarity"] = polarity_value
            frames.append(frame)
    pixels = pd.concat(frames, ignore_index=True)
    pixels["date"] = pd.to_datetime(pixels["date"])
    return pixels


def resolve_chl_path(experiment: str) -> Path:
    """Silver path for the collocated chlorophyll table."""
    return (
        PROJECT_ROOT / "data" / experiment / "silver" / "pace_chl" / "eddy_bg_chlor_a.parquet"
    )


def collocate_chlor_a(
    experiment: str,
    exponents: dict[str, float],
    out_path: Path | None = None,
) -> pd.DataFrame:
    """
    Average PACE L3 chlor_a over the SDP eddy and background pixel sets.

    exponents maps a pigment name to its power-law exponent b.
    The result carries eddy_chl_mean, bg_chl_mean, and one eddy_chl_pow_<pigment> and bg_chl_pow_<pigment> column per entry, plus the pixel count behind each mean.
    """
    out_path = out_path or resolve_chl_path(experiment)
    data_dir = PROJECT_ROOT / "data" / experiment
    bgc_dir = data_dir / "bronze" / "pace_l3_8d_bgc"
    rrs_dir = data_dir / "bronze" / "pace_l3_8d"
    # Group 1 is the window start and group 2 the window end: PACE_OCI.20240929_20241006.L3m.8D.BGC.V3_2.4km.nc
    bgc_re = re.compile(r"PACE_OCI\.(\d{8})_(\d{8})\.L3m\.8D\.BGC\.")
    rrs_re = re.compile(r"PACE_OCI\.(\d{8})_(\d{8})\.L3m\.8D\.RRS\.")
    swot_files = index_swot_files_by_date(data_dir / "bronze" / "swot_l4_open_ocean")
    contours = load_eddy_contours(
        resolve_output_dir(experiment, "eddy_track", "cyclone"),
        resolve_output_dir(experiment, "eddy_track", "anticyclone"),
    )
    pixels = read_eddy_pixels(experiment)
    print(
        f"eddy_pixels: {len(pixels):,}\n"
        f"dates: {pixels['date'].nunique()}"
    )

    rrs_by_window = {}
    for path in sorted(rrs_dir.glob("*.nc")):
        match = rrs_re.search(path.name)
        if match:
            rrs_by_window[f"{match.group(1)}_{match.group(2)}"] = path

    def compute_means(values: np.ndarray, prefix: str) -> dict[str, float]:
        summary = {f"{prefix}_chl_mean": float(values.mean())}
        for pigment, exponent in exponents.items():
            summary[f"{prefix}_chl_pow_{pigment}"] = float((values**exponent).mean())
        return summary

    eddy_rows, bg_rows = [], []
    for bgc_path in sorted(bgc_dir.glob("*.nc")):
        match = bgc_re.search(bgc_path.name)
        if match is None:
            continue
        win_start = dt.datetime.strptime(match.group(1), "%Y%m%d").date()
        win_end = dt.datetime.strptime(match.group(2), "%Y%m%d").date()
        # The window midpoint is the same join key collocate_pace and background write, so chlorophyll rows land on the gold table's dates.
        repr_date = win_start + (win_end - win_start) / 2
        window_key = f"{win_start:%Y%m%d}_{win_end:%Y%m%d}"

        with xr.open_dataset(bgc_path) as bgc:
            chlor_a = bgc["chlor_a"].load()
        pace_lon = chlor_a["lon"].values
        pace_lat = chlor_a["lat"].values

        # SDP kept only pixels with a finite spectrum in every band, so apply the same rule here and the two pixel sets stay comparable.
        with xr.open_dataset(rrs_by_window[window_key]) as rrs_ds:
            rrs_finite = np.all(np.isfinite(rrs_ds["Rrs"].values), axis=-1) # (n_lat, n_lon, n_wavelengths) -> (n_lat, n_lon)
        usable = xr.DataArray(
            np.isfinite(chlor_a.values) & rrs_finite,
            coords=chlor_a.coords,
            dims=chlor_a.dims,
        )

        day_pixels = pixels[pixels["date"] == pd.Timestamp(repr_date)]
        if len(day_pixels):
            point = {
                "lon": xr.DataArray(day_pixels["pixel_lon"].to_numpy(), dims="pixel"),
                "lat": xr.DataArray(day_pixels["pixel_lat"].to_numpy(), dims="pixel"),
            }
            found = day_pixels.assign(
                chlor_a=chlor_a.sel(**point, method="nearest").values,
                usable=usable.sel(**point, method="nearest").values,
            )
            for (track_id, polarity), group in found.groupby(["track_id", "polarity"]):
                good = group.loc[group["usable"], "chlor_a"].to_numpy()
                if good.size == 0:
                    continue
                eddy_rows.append(
                    {
                        "track_id": int(track_id),
                        "polarity": int(polarity),
                        "date": pd.Timestamp(repr_date),
                        "n_eddy_chl_pixels": int(good.size),
                        "n_eddy_sdp_pixels": int(len(group)),
                        **compute_means(good, "eddy"),
                    }
                )

        swot_path = find_nearest_swot_file(swot_files, repr_date)
        if swot_path is None:
            print(
                f"date: {repr_date}\n"
                "status: no_background\n"
                "reason: no_swot_day\n"
                f"swot_search_days: {SWOT_SEARCH_DAYS}"
            )
            continue
        calm = compute_calm_mask_on_pace(swot_path, pace_lon, pace_lat)
        lon_grid, lat_grid = np.meshgrid(pace_lon, pace_lat) # (n_lon,), (n_lat,) -> two (n_lat, n_lon)
        candidate = np.flatnonzero(calm & usable.values) # (n_lat, n_lon) -> (n_candidate,) flat indices
        if candidate.size == 0:
            print(
                f"date: {repr_date}\n"
                "status: no_background\n"
                "reason: no_calm_background_pixel"
            )
            continue

        window_contours = []
        day = win_start
        while day <= win_end:
            window_contours.extend(contours.get(day, []))
            day += dt.timedelta(days=1)
        if window_contours:
            # ravel()[candidate]: (n_lat, n_lon) -> (n_lat * n_lon,) -> (n_candidate,)
            inside = in_any_contour(
                window_contours, lon_grid.ravel()[candidate], lat_grid.ravel()[candidate]
            )
            candidate = candidate[~inside]
        if candidate.size == 0:
            print(
                f"date: {repr_date}\n"
                "status: no_background\n"
                "reason: all_calm_pixels_inside_eddies"
            )
            continue

        # background.py subsamples to keep SDP fast; chlor_a is already computed, so every calm pixel goes into the mean here.
        bg_values = chlor_a.values.ravel()[candidate] # (n_lat, n_lon) -> (n_lat * n_lon,) -> (n_candidate,)
        bg_rows.append(
            {
                "date": pd.Timestamp(repr_date),
                "n_bg_chl_pixels": int(bg_values.size),
                **compute_means(bg_values, "bg"),
            }
        )
        print(
            f"date: {repr_date}\n"
            f"eddy_days: {len(eddy_rows)}\n"
            f"background_pixels: {bg_values.size}",
            flush=True,
        )

    table = pd.DataFrame(eddy_rows).merge(pd.DataFrame(bg_rows), on="date", how="inner")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_path, index=False)
    # The chlor_a**b columns only mean anything next to the b that made them, so the exponents travel with the table.
    resolve_exponents_path(out_path).write_text(json.dumps(exponents, indent=1, sort_keys=True))
    print(
        f"\noutput_path: {out_path}\n"
        f"eddy_days_written: {len(table)}\n"
        f"columns_written: {len(table.columns)}"
    )
    return table


def resolve_exponents_path(table_path: Path) -> Path:
    """Sidecar holding the exponents a cached table was built with."""
    return table_path.with_name(table_path.stem + "_exponents.json")


def load_or_collocate(experiment: str, exponents: dict[str, float]) -> pd.DataFrame:
    """
    Read the cached chlorophyll table, recomputing when it is missing or stale.

    Stale means built with different exponents.
    Reusing it then would silently pair one pigment's power-law scale with another's exponent.
    """
    path = resolve_chl_path(experiment)
    sidecar = resolve_exponents_path(path)
    if path.exists() and sidecar.exists():
        cached = json.loads(sidecar.read_text())
        if cached == {k: float(v) for k, v in exponents.items()}:
            return pd.read_parquet(path)
        print(
            "status: recomputing\n"
            "reason: cached_chlorophyll_exponents_changed"
        )
    return collocate_chlor_a(experiment, exponents, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument(
        "--exponents",
        type=Path,
        required=True,
        help="JSON file mapping each pigment to its chlor_a power-law exponent.",
    )
    args = parser.parse_args()
    collocate_chlor_a(args.experiment, json.loads(args.exponents.read_text()))
