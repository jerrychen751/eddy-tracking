"""
Compute the per-date background pigment means: the denominator of the eddy log-ratio targets.

For each PACE composite, background pixels are open-water pixels that are both calm (|normalized relative vorticity| < 0.1, from the matching SWOT day) and outside every tracked eddy contour active during the window. Their Rrs spectra run through the same SDP model as the eddy pixels, and the per-pigment mean over those pixels is the background for that date.

Writes silver/pigments/background/bg_mean.parquet: one row per composite date with columns date, bg_mean_<pigment> (13), and n_bg_pixels.
"""

import argparse
import datetime as dt
import re
from collections import defaultdict
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path as MplPath
from eddy_tracking.packages.py_eddy_tracker.observations.tracking import (
    TrackEddiesObservations,
)

from utils.config import load_config, resolve_data_dir, resolve_output_dir
from eddy_tracking.utils.subset import load_rossby_field
from eddy_tracking.packages.sdp import run_sdp
from eddy_tracking.packages.sdp.ancillary import sample_ancillary
from eddy_tracking.packages.sdp.physics import GSMInversionError
from eddy_tracking.packages.sdp.preprocessing import preprocess_rrs_batch
from eddy_tracking.preprocess.sss import read_multiple_sss
from eddy_tracking.preprocess.sst import read_multiple_sst

SWOT_SEARCH_DAYS = 4

# on-disk SDP pigment name -> canonical suffix (must match build_gold_table.PIGMENTS)
PIGMENTS = {
    "T chla": "Tchla", "Zea": "Zea", "DV chla": "DV_chla", "ButFuco": "ButFuco",
    "HexFuco": "HexFuco", "Allo": "Allo", "MV chlb": "MV_chlb", "Neo": "Neo",
    "Viola": "Viola", "Fuco": "Fuco", "chl c1+c2": "Chlc12", "chl c3": "Chlc3",
    "Perid": "Perid",
}


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


def parse_pace_window(filename: str, temporal_res: str) -> tuple[dt.date, dt.date, dt.date] | None:
    """
    (repr_date, win_start, win_end) for a PACE file, or None if it doesn't parse.

    repr_date is the join key written to the table; it is computed exactly as in collocate_pace so background and eddy rows share a date. For 8-day composites it is the window midpoint; for daily files all three dates are the same day.
    """
    if temporal_res == "8D":
        m = re.search(r"PACE_OCI\.(\d{8})_(\d{8})\.L3m\.8D\.AOP\.", filename)
        if m is None:
            return None
        start = dt.datetime.strptime(m.group(1), "%Y%m%d").date()
        end = dt.datetime.strptime(m.group(2), "%Y%m%d").date()
        return start + (end - start) / 2, start, end
    m = re.search(r"PACE_OCI\.(\d{8})\.L3m\.DAY\.AOP\.", filename)
    if m is None:
        return None
    day = dt.datetime.strptime(m.group(1), "%Y%m%d").date()
    return day, day, day


def index_swot_files_by_date(swot_dir: Path) -> dict[dt.date, Path]:
    """Map measurement date (first 8-digit token in the name) to SWOT file path."""
    swot_date_re = re.compile(r"\d{8}")
    files = {}
    for fp in sorted(swot_dir.glob("*.nc")):
        m = swot_date_re.search(fp.name)
        if m:
            files[dt.datetime.strptime(m.group(), "%Y%m%d").date()] = fp
    return files


def find_nearest_swot_file(files: dict[dt.date, Path], target: dt.date) -> Path | None:
    """SWOT file on target, else the closest within SWOT_SEARCH_DAYS, else None."""
    for delta in range(SWOT_SEARCH_DAYS + 1):
        for day in (target - dt.timedelta(delta), target + dt.timedelta(delta)):
            if day in files:
                return files[day]
    return None


def load_eddy_contours(
    cyclone_track_dir: Path, anticyclone_track_dir: Path
) -> dict[dt.date, list[tuple[np.ndarray, np.ndarray]]]:
    """
    Map date -> list of (contour_lon, contour_lat) for every non-virtual eddy.

    Contours come from both polarities, longitudes converted from py-eddy-tracker's 0-360 convention to PACE's -180/180.
    """
    pet_epoch = dt.date(1950, 1, 1)
    contours: dict[dt.date, list] = defaultdict(list)
    for track_dir in (cyclone_track_dir, anticyclone_track_dir):
        zarr_path = track_dir / f"{track_dir.name}_tracks.zarr"
        if not zarr_path.exists():
            continue
        tracked = TrackEddiesObservations.load_file(str(zarr_path))
        keep = ~tracked.virtual.astype(bool)
        days = [pet_epoch + dt.timedelta(days=int(t)) for t in tracked.time[keep]]
        contour_lon = (tracked.contour_lon_s[keep] + 180) % 360 - 180
        contour_lat = tracked.contour_lat_s[keep]
        for i, day in enumerate(days):
            contours[day].append((contour_lon[i], contour_lat[i]))
    return contours


def compute_calm_mask_on_pace(swot_fp, pace_lon: np.ndarray, pace_lat: np.ndarray) -> np.ndarray:
    """
    Boolean (lat, lon) PACE-grid mask of calm water for one SWOT day.

    Interpolates |Ro| from the coarser SWOT grid onto the PACE pixels and thresholds it. Pixels saved as NaN in the SWOT bronze file interpolate to NaN and fail the comparison, so they are treated as not-calm.
    """
    # The DUACS/MIOST source variable is named relative_vorticity, but these files store normalized relative vorticity, not raw zeta in s^-1.
    # The values are Rossby number (Ro = zeta/f), so "calm" water is a direct threshold on |Ro|.
    bg_threshold_rossby = 0.1
    swot_lon, swot_lat, rossby_number = load_rossby_field(swot_fp)
    abs_rossby_number = xr.DataArray(
        np.abs(rossby_number),
        coords={"latitude": swot_lat, "longitude": swot_lon},
        dims=["latitude", "longitude"],
    )
    on_pace = abs_rossby_number.interp(
        latitude=pace_lat, longitude=pace_lon, method="linear"
    ).values
    return on_pace < bg_threshold_rossby


def is_in_any_contour(
    contours: list[tuple[np.ndarray, np.ndarray]], lons: np.ndarray, lats: np.ndarray
) -> np.ndarray:
    """Boolean over the given points: inside at least one eddy contour polygon."""
    points = np.column_stack([lons, lats])  # (n_points,) + (n_points,) -> (n_points, 2)
    inside = np.zeros(points.shape[0], dtype=bool)
    for contour_lon, contour_lat in contours:
        polygon = MplPath(np.column_stack([contour_lon, contour_lat]))  # (n_vertices,) + (n_vertices,) -> (n_vertices, 2)
        inside |= polygon.contains_points(points)
    return inside


def run_sdp_filtering_nonconvergent(
    rrs: pd.DataFrame,
    wavelengths: np.ndarray,
    sst: np.ndarray,
    sss: np.ndarray,
) -> tuple[pd.DataFrame, int]:
    try:
        return run_sdp(rrs=rrs, wl=wavelengths, sst=sst, sss=sss), 0
    except GSMInversionError:
        if len(rrs) == 1:
            return pd.DataFrame(columns=list(PIGMENTS)), 1  # pyright: ignore[reportArgumentType]

    midpoint = len(rrs) // 2
    left, left_dropped = run_sdp_filtering_nonconvergent(
        rrs.iloc[:midpoint].reset_index(drop=True),
        wavelengths,
        sst[:midpoint],
        sss[:midpoint],
    )
    right, right_dropped = run_sdp_filtering_nonconvergent(
        rrs.iloc[midpoint:].reset_index(drop=True),
        wavelengths,
        sst[midpoint:],
        sss[midpoint:],
    )
    predictions = [frame for frame in (left, right) if not frame.empty]
    return (
        pd.concat(predictions, ignore_index=True)
        if predictions
        else pd.DataFrame(columns=list(PIGMENTS)),  # pyright: ignore[reportArgumentType]
        left_dropped + right_dropped,
    )


def compute_background_means(
    df: pd.DataFrame,
    sst_df: pd.DataFrame,
    sss_df: pd.DataFrame,
) -> dict | None:
    """
    Run the SDP model on background pixels and average each pigment.

    Mirrors run_sdp.process_eddy: preprocess Rrs to 1 nm, sample nearest SST/SSS, drop pixels missing either, run SDP. Returns bg_mean_<pigment> for the 13 pigments plus n_bg_pixels, or None if no pixel survives the SST/SSS filter.
    """
    rrs_cols = [c for c in df.columns if c.startswith("Rrs_")]
    wavelengths = np.array([float(c.split("_")[1]) for c in rrs_cols])
    wl_processed, rrs_processed = preprocess_rrs_batch(wavelengths, df[rrs_cols].to_numpy())

    sst_vals, sss_vals = sample_ancillary(
        sst_df,
        sss_df,
        lons=df["pixel_lon"].to_numpy(),
        lats=df["pixel_lat"].to_numpy(),
        times=pd.to_datetime(df["date"]).to_numpy(),
    )
    valid = np.isfinite(sst_vals) & np.isfinite(sss_vals)
    if valid.sum() == 0:
        return None

    # rrs_processed[valid]: (n_pixels, n_wl_1nm) -> (n_valid, n_wl_1nm)
    rrs_frame = pd.DataFrame(
        rrs_processed[valid], columns=wl_processed.astype(int)
    )
    pigments_df, n_nonconvergent = run_sdp_filtering_nonconvergent(
        rrs_frame,
        wl_processed,
        sst_vals[valid],
        sss_vals[valid],
    )
    if n_nonconvergent:
        print(
            f"background_pixels_dropped: {n_nonconvergent}\n"
            f"total_background_pixels: {len(rrs_frame)}\n"
            "reason: gsm_inversion_nonconvergence"
        )
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
    cyclone_track_dir = resolve_output_dir(experiment, "eddy_track", "cyclone")
    anticyclone_track_dir = resolve_output_dir(experiment, "eddy_track", "anticyclone")
    temporal_res = cfg["collocate_pace"].get("temporal_resolution", "DAY")

    swot_files = index_swot_files_by_date(swot_dir)
    eddy_contours = load_eddy_contours(cyclone_track_dir, anticyclone_track_dir)
    print("status: loading_sst_sss_grids")
    sst_df = read_multiple_sst(sorted(sst_dir.glob("*.nc")))
    sss_df = read_multiple_sss(sorted(sss_dir.glob("*.nc4")))

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
            window_contours.extend(eddy_contours.get(day, []))
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
                "reason: no_valid_pixels_after_sst_sss_filter"
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
