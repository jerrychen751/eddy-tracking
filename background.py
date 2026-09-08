"""
Compute the per-date background pigment means: the denominator of the eddy log-ratio targets.

For each PACE composite, background pixels are open-water pixels that are both calm (|normalized relative vorticity| < 0.1, from the matching SWOT day) and outside every tracked eddy contour active during the window. Their Rrs spectra run through the same SDP model as the eddy pixels, and the per-pigment mean over those pixels is the background for that date.

Writes silver/pigments/background/bg_mean.parquet: one row per composite date with columns date, bg_mean_<pigment> (13), and n_bg_pixels.
"""

import argparse
import datetime as dt
from collections import defaultdict
from typing import cast

import numpy as np
import pandas as pd
import xarray as xr

from eddy_tracking.config import load_config, resolve_data_dir, resolve_output_dir
from eddy_tracking.packages.sdp import PIGMENTS, run_sdp_on_pace_l3
from eddy_tracking.preprocess.ancillary import read_ancillary_grids
from eddy_tracking.preprocess.pace import parse_pace_window
from eddy_tracking.preprocess.swot import (
    SWOT_SEARCH_DAYS,
    compute_calm_mask_on_pace,
    find_nearest_swot_file,
    index_swot_files_by_date,
)
from eddy_tracking.preprocess.tracks import EddyObs, build_date_eddy_index, is_in_any_contour, load_tracks


def parse_args() -> argparse.Namespace:
    """Parse background-stage CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument(
        "--subsample", type=int, default=2000,
        help="Max background pixels per composite to push through SDP (0 = use all). "
            "The regional mean is stable well below the full count, so subsampling "
            "keeps a local run fast; raise it or set 0 for a faithful all-pixel mean.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process at most this many PACE files (0 = all). For quick smoke tests.",
    )
    return parser.parse_args()


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
    means = {f"bg_mean_{canon}": float(pigments_df[raw].mean()) for raw, canon in PIGMENTS.items()}
    means["n_bg_pixels"] = len(pigments_df)
    return means


def main(
    experiment: str | None = None,
    subsample: int = 2000,
    limit: int = 0,
) -> None:
    """Compute and write per-date background pigment means."""
    if experiment is None:
        args = parse_args()
        experiment = cast(str, args.experiment)
        subsample = args.subsample
        limit = args.limit

    cfg = load_config(experiment)
    swot_dir = resolve_data_dir(cfg, "swot_dir")
    pace_dir = resolve_data_dir(cfg, "pace_dir")
    sst_dir = resolve_data_dir(cfg, "sst_dir")
    sss_dir = resolve_data_dir(cfg, "sss_dir")
    out_dir = resolve_output_dir(experiment, "pigments", "background")
    temporal_res = cfg["collocate_pace"].get("temporal_resolution", "DAY")

    swot_files = index_swot_files_by_date(swot_dir)
    date_index: dict[dt.date, list[EddyObs]] = defaultdict(list)
    for polarity in ("cyclone", "anticyclone"):
        for day, eddies in build_date_eddy_index(load_tracks(experiment, polarity), polarity).items():
            date_index[day].extend(eddies)
    print("status: loading_sst_sss_grids")
    sst_df, sss_df = read_ancillary_grids(sst_dir, sss_dir)

    pace_files = sorted(pace_dir.glob("*.nc"))
    if limit:
        pace_files = pace_files[:limit]
    print(
        "status: computing_background_means\n"
        f"pace_composites: {len(pace_files)}"
    )

    rng = np.random.default_rng(0)
    rows = []
    for fp in pace_files:
        window = parse_pace_window(fp.name, temporal_res)
        if window is None:
            continue
        repr_date, win_start, win_end = window

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
            wavelengths = ds.coords["wavelength"].values.astype(int)
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

        window_contours = []
        day = win_start
        while day <= win_end:
            window_contours.extend((eddy.contour_lon, eddy.contour_lat) for eddy in date_index.get(day, []))
            day += dt.timedelta(days=1)
        if window_contours:
            inside = is_in_any_contour(
                window_contours, lon2d.ravel()[candidate], lat2d.ravel()[candidate]  # (lat, lon) -> (lat*lon,) -> (n_candidate,) each
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

        if subsample and n_candidate > subsample:
            candidate = np.sort(rng.choice(candidate, size=subsample, replace=False))

        date_value = pd.Timestamp(repr_date)
        df = pd.DataFrame({
            "date": date_value,
            "pixel_lon": lon2d.ravel()[candidate],  # (lat, lon) -> (lat*lon,) -> (n_candidate,)
            "pixel_lat": lat2d.ravel()[candidate],  # (lat, lon) -> (lat*lon,) -> (n_candidate,)
        })
        rrs_df = pd.DataFrame(rrs_flat[candidate], columns=[f"Rrs_{w}" for w in wavelengths])  # pyright: ignore[reportArgumentType]  # (lat*lon, wavelength) -> (n_candidate, wavelength)
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
    out_path = out_dir / "bg_mean.parquet"
    out.to_parquet(out_path, index=False)
    print(
        f"output_path: {out_path}\n"
        f"dates_written: {len(out)}"
    )


if __name__ == "__main__":
    main()
