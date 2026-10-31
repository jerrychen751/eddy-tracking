"""
Compute the per-date background pigment means: the denominator of the eddy log-ratio targets.

For each PACE composite, background pixels are open-water pixels that are both calm (|normalized relative vorticity| < 0.1, from the matching SWOT day) and outside every tracked eddy contour active during the date range. Their Rrs spectra run through the same SDP model as the eddy pixels, and the per-pigment mean over those pixels is the background for that date.

Writes bg_mean.parquet under --out: one row per composite date with columns date, bg_mean_<pigment> (13), and n_bg_pixels.
"""

import argparse
import datetime as dt
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import xarray as xr

from eddy_tracking.packages.sdp import PIGMENTS, run_sdp_on_pace_l3
from eddy_tracking.preprocess.ancillary import read_ancillary_grids
from eddy_tracking.preprocess.pace import parse_fn_for_date_range
from eddy_tracking.preprocess.swot import (
    SWOT_SEARCH_DAYS,
    compute_calm_mask_on_pace,
    find_nearest_swot_file,
    index_swot_files_by_date,
)
from eddy_tracking.preprocess.tracks import is_in_any_contour, load_date_eddy_index
from eddy_tracking.utils.cli import write_settings


def compute_background_means(
    df: pd.DataFrame,
    sst_df: pd.DataFrame,
    sss_df: pd.DataFrame,
) -> dict | None:
    """
    Run the SDP model on background pixels and average each pigment.

    Shares run_sdp_on_pace_l3 with run_sdp.process_eddy: preprocess Rrs to 1 nm, sample nearest SST/SSS, drop pixels missing either, run SDP. Returns bg_mean_<pigment> for the 13 pigments plus n_bg_pixels, or None if no pixel survives the SST/SSS filter and the SDP inversion.
    """
    pigments_df, _ = run_sdp_on_pace_l3(df, sst_df, sss_df)
    if pigments_df.empty:
        return None
    means = {f"bg_mean_{canon}": max(float(cast(pd.Series, pigments_df[raw]).mean()), 0.0) for raw, canon in PIGMENTS.items()}
    means["n_bg_pixels"] = len(pigments_df)
    return means


def main(argv: list[str]) -> None:
    """Compute and write per-date background pigment means."""
    parser = argparse.ArgumentParser(prog="eddy-tracking sample-background")
    parser.add_argument("--ssh", type=Path, required=True, help="folder of daily SSH maps in the AVISO layout")
    parser.add_argument("--pace", type=Path, required=True, help="folder of PACE L3 Rrs files")
    parser.add_argument("--sst", type=Path, required=True, help="folder of SST files")
    parser.add_argument("--sss", type=Path, required=True, help="folder of SSS files")
    parser.add_argument("--tracks", type=Path, required=True, help="folder track-eddies wrote")
    parser.add_argument("--temporal-resolution", choices=["DAY", "8D"], required=True, help="whether the PACE files are daily maps or 8-day composites")
    parser.add_argument(
        "--subsample", type=int, required=True,
        help="Max background pixels per composite to push through SDP (0 = use all). "
            "The regional mean is stable well below the full count, so subsampling "
            "keeps a local run fast; raise it or set 0 for a faithful all-pixel mean.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process at most this many PACE files (0 = all). For quick smoke tests.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    write_settings(args.out, "sample-background", args, operational={"limit"})
    args.out.mkdir(parents=True, exist_ok=True)

    swot_files = index_swot_files_by_date(args.ssh)
    date_index = load_date_eddy_index(args.tracks)
    print("status: loading_sst_sss_grids")
    sst_df, sss_df = read_ancillary_grids(args.sst, args.sss)

    pace_files = sorted(args.pace.glob("*.nc"))
    if args.limit:
        pace_files = pace_files[:args.limit]
    print(
        "status: computing_background_means\n"
        f"pace_composites: {len(pace_files)}"
    )

    rng = np.random.default_rng(0)
    rows = []
    for fp in pace_files:
        composite_date_range = parse_fn_for_date_range(fp.name, args.temporal_resolution)
        if composite_date_range is None:
            continue
        repr_date, date_range_start, date_range_end = composite_date_range

        swot_fp = find_nearest_swot_file(swot_files, repr_date)
        if swot_fp is None:
            print(
                f"date: {repr_date}\n"
                "status: skipped\n"
                "reason: no_swot_day\n"
                f"swot_search_days: {SWOT_SEARCH_DAYS}"
            )
            continue

        with xr.open_dataset(fp) as ds:
            pace_lon = ds["lon"].values
            pace_lat = ds["lat"].values
            wavelengths = ds.coords["wavelength"].values
            rrs = ds["Rrs"].values  # (lat, lon, wavelength)

        lon2d, lat2d = np.meshgrid(pace_lon, pace_lat)  # (n_lon,) + (n_lat,) -> (n_lat, n_lon) each
        calm = compute_calm_mask_on_pace(swot_fp, pace_lon, pace_lat)
        rrs_flat = rrs.reshape(-1, rrs.shape[-1])  # (lat, lon, wavelength) -> (lat*lon, wavelength)
        all_finite = np.all(np.isfinite(rrs_flat), axis=1)  # (lat*lon, wavelength) -> (lat*lon,)

        candidate = np.flatnonzero(calm.ravel() & all_finite)  # calm (lat, lon) -> (lat*lon,), candidate (n_candidate,) of flat indices
        if candidate.size == 0:
            print(
                f"date: {repr_date}\n"
                "status: skipped\n"
                "reason: no_calm_observed_pixels"
            )
            continue

        date_range_contours = []
        day = date_range_start
        while day <= date_range_end:
            date_range_contours.extend((eddy.contour_lon, eddy.contour_lat) for eddy in date_index.get(day, []))
            day += dt.timedelta(days=1)
        if date_range_contours:
            inside = is_in_any_contour(
                date_range_contours, lon2d.ravel()[candidate], lat2d.ravel()[candidate]  # (lat, lon) -> (lat*lon,) -> (n_candidate,) each
            )
            candidate = candidate[~inside]
        n_candidate = candidate.size
        if n_candidate == 0:
            print(
                f"date: {repr_date}\n"
                "status: skipped\n"
                "reason: no_background_pixels_after_eddy_exclusion"
            )
            continue

        if args.subsample and n_candidate > args.subsample:
            candidate = np.sort(rng.choice(candidate, size=args.subsample, replace=False))

        date_value = pd.Timestamp(repr_date).as_unit("ns")
        df = pd.DataFrame({
            "date": date_value,
            "pixel_lon": lon2d.ravel()[candidate],  # (lat, lon) -> (lat*lon,) -> (n_candidate,)
            "pixel_lat": lat2d.ravel()[candidate],  # (lat, lon) -> (lat*lon,) -> (n_candidate,)
        })
        rrs_df = pd.DataFrame(rrs_flat[candidate], columns=[f"Rrs_{w:.3f}" for w in wavelengths])  # pyright: ignore[reportArgumentType]  # (lat*lon, wavelength) -> (n_candidate, wavelength)
        df = pd.concat([df, rrs_df], axis=1)

        means = compute_background_means(df, sst_df, sss_df)
        if means is None:
            print(
                f"date: {repr_date}\n"
                "status: skipped\n"
                "reason: no_valid_pixels"
            )
            continue
        means["date"] = date_value
        rows.append(means)
        print(
            f"date: {repr_date}\n"
            f"background_pixels: {means['n_bg_pixels']}\n"
            f"candidate_pixels: {n_candidate}"
        )

    if not rows:
        print("status: no_background_rows_produced")
        return

    out = pd.DataFrame(rows)
    out = out[["date"] + [f"bg_mean_{c}" for c in PIGMENTS.values()] + ["n_bg_pixels"]]
    out_path = args.out / "bg_mean.parquet"
    out.to_parquet(out_path, index=False)
    print(
        f"output_path: {out_path}\n"
        f"dates_written: {len(out)}"
    )
