"""
Assemble the gold eddy-pigment table: one analysis-ready row per eddy-day.

Aggregates the per-pixel pigments from run_sdp (silver/pigments) to eddy-interior means, attaches eddy size/strength/age (tracks), movement and signed Gulf-Stream distance (gs_axis), season, and the log-ratio targets once the background means exist. Writes gold/eddy_pigment_table.parquet.
"""

import argparse
import datetime as dt

import numpy as np
import pandas as pd
from py_eddy_tracker.observations.tracking import TrackEddiesObservations

from utils.config import load_config, resolve_output_dir, resolve_gold_dir

parser = argparse.ArgumentParser()
parser.add_argument("experiment")
args = parser.parse_args()
cfg = load_config(args.experiment)

EXPERIMENT = args.experiment
GS_AXIS_DIR = resolve_output_dir(EXPERIMENT, "gs_axis")
BACKGROUND_DIR = resolve_output_dir(EXPERIMENT, "background")
EDDY_START = pd.Timestamp(cfg["base"]["time"]["eddy_date_range"][0])
EDDY_END = pd.Timestamp(cfg["base"]["time"]["eddy_date_range"][1])

# on-disk pigment name -> canonical (space-free) suffix used in the table
PIGMENTS = {
    "T chla": "Tchla", "Zea": "Zea", "DV chla": "DV_chla", "ButFuco": "ButFuco",
    "HexFuco": "HexFuco", "Allo": "Allo", "MV chlb": "MV_chlb", "Neo": "Neo",
    "Viola": "Viola", "Fuco": "Fuco", "chl c1+c2": "Chlc12", "chl c3": "Chlc3",
    "Perid": "Perid",
}
PET_EPOCH = dt.date(1950, 1, 1)
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
          6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
KM_PER_DEG_LAT = 111.0
# Eddy-days with fewer interior pixels than this give unreliable means and are dropped.
MIN_EDDY_PIXELS = 10


def aggregate_eddy_days() -> pd.DataFrame:
    """One row per (polarity, track_id, date): eddy-interior pigment means + QC."""
    rows = []
    for polarity, pol_val in [("anticyclone", 0), ("cyclone", 1)]:
        pig_dir = resolve_output_dir(EXPERIMENT, "pigments", polarity)
        for fp in sorted(pig_dir.glob("*_pigments.parquet")):
            df = pd.read_parquet(fp)
            for date, grp in df.groupby("date"):
                row = {
                    "track_id": int(grp["track_id"].iloc[0]),
                    "date": pd.Timestamp(date),
                    "polarity": pol_val,
                    "center_lon": float(grp["center_lon"].mean()),
                    "center_lat": float(grp["center_lat"].mean()),
                    "valid_frac": float(grp["coverage"].iloc[0]),
                    "n_eddy_pixels": int(len(grp)),
                }
                for raw, canon in PIGMENTS.items():
                    row[f"eddy_mean_{canon}"] = float(grp[raw].mean())
                rows.append(row)
    return pd.DataFrame(rows)


def track_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-obs (radius_km, amplitude_cm) and per-track (birth/death) for both polarities."""
    obs_frames, life_rows = [], []
    for polarity, pol_val in [("anticyclone", 0), ("cyclone", 1)]:
        tdir = resolve_output_dir(EXPERIMENT, "eddy_track", polarity)
        tr = TrackEddiesObservations.load_file(str(tdir / f"{tdir.name}_tracks.zarr"))
        keep = ~tr.virtual.astype(bool)
        days = pd.to_datetime([PET_EPOCH + dt.timedelta(days=int(t)) for t in tr.time[keep]])
        df = pd.DataFrame({
            "polarity": pol_val,
            "track_id": tr.track[keep].astype(int),
            "date": days,
            "radius_km": tr.radius_e[keep] / 1000.0,
            "amplitude_cm": tr.amplitude[keep] * 100.0,
        }).sort_values("date")
        obs_frames.append(df)
        for tid, grp in df.groupby("track_id"):
            life_rows.append({
                "polarity": pol_val, "track_id": tid,
                "birth_date": grp["date"].min(), "death_date": grp["date"].max(),
            })
    return pd.concat(obs_frames, ignore_index=True), pd.DataFrame(life_rows)


def gs_distance(axis_by_date, date, center_lon, center_lat) -> float:
    """Signed meridional distance (km, + = north of the jet) to the nearest-day axis."""
    base = pd.Timestamp(date).date()
    axis = next(
        (axis_by_date[d] for delta in range(5)
         for d in (base - dt.timedelta(delta), base + dt.timedelta(delta))
         if d in axis_by_date),
        None,
    )
    if axis is None:
        return np.nan
    lon, core_lat = axis
    finite = np.isfinite(core_lat)
    if finite.sum() < 2:
        return np.nan
    here = np.interp(center_lon, lon[finite], core_lat[finite], left=np.nan, right=np.nan)
    return float((center_lat - here) * KM_PER_DEG_LAT) if np.isfinite(here) else np.nan


def main():
    eddy = aggregate_eddy_days()
    if eddy.empty:
        print("No eddy pigment files found; nothing to assemble.")
        return
    n_eddies = len(eddy[["polarity", "track_id"]].drop_duplicates())
    print(f"Aggregated {len(eddy)} eddy-days from {n_eddies} eddies")

    # Drop eddy-days whose interior is too sparsely sampled for a reliable mean.
    n_before = len(eddy)
    eddy = eddy[eddy["n_eddy_pixels"] >= MIN_EDDY_PIXELS].reset_index(drop=True)
    if n_before - len(eddy):
        print(f"Dropped {n_before - len(eddy)} eddy-days with < {MIN_EDDY_PIXELS} interior pixels")

    # Eddy size/strength (nearest track obs by date) + lifetime
    obs, life = track_features()
    eddy = pd.merge_asof(
        eddy.sort_values("date"), obs.sort_values("date"),
        on="date", by=["polarity", "track_id"], direction="nearest",
    ).merge(life, on=["polarity", "track_id"], how="left")

    lifetime = (eddy["death_date"] - eddy["birth_date"]).dt.days
    eddy["age_days"] = (eddy["date"] - eddy["birth_date"]).dt.days
    eddy["age_frac"] = np.where(lifetime > 0, eddy["age_days"] / lifetime, np.nan)
    # A composite date can fall just outside a track's daily detections; keep the
    # life fraction within [0, 1].
    eddy["age_frac"] = eddy["age_frac"].clip(0, 1)
    eddy["birth_observed"] = eddy["birth_date"] > EDDY_START
    eddy["death_observed"] = eddy["death_date"] < EDDY_END
    eddy["season"] = eddy["date"].dt.month.map(SEASON)

    # Movement + signed Gulf-Stream distance
    movement = pd.read_parquet(GS_AXIS_DIR / "movement.parquet")
    movement["polarity"] = movement["polarity"].map({"anticyclone": 0, "cyclone": 1})
    eddy = eddy.merge(movement[["polarity", "track_id", "movement"]], on=["polarity", "track_id"], how="left")
    axis = pd.read_parquet(GS_AXIS_DIR / "axis.parquet")
    axis_by_date = {
        pd.Timestamp(d).date(): (g["lon"].to_numpy(), g["core_lat"].to_numpy())
        for d, g in axis.groupby("date")
    }
    eddy["gs_dist_km"] = [
        gs_distance(axis_by_date, d, lo, la)
        for d, lo, la in zip(eddy["date"], eddy["center_lon"], eddy["center_lat"])
    ]

    # Background means + log-ratio targets (only once the background stage has run)
    bg_path = BACKGROUND_DIR / "bg_mean.parquet"
    if bg_path.exists():
        eddy = eddy.merge(pd.read_parquet(bg_path), on="date", how="left")
        log_ratio_cols = []
        for canon in PIGMENTS.values():
            if f"bg_mean_{canon}" in eddy:
                col = f"log_ratio_{canon}"
                with np.errstate(divide="ignore"):
                    eddy[col] = np.log(eddy[f"eddy_mean_{canon}"] / eddy[f"bg_mean_{canon}"])
                log_ratio_cols.append(col)
        # A zero eddy mean (pigment below detection) gives -inf; treat it as missing.
        n_nonfinite = int(np.isinf(eddy[log_ratio_cols].to_numpy()).sum())
        eddy[log_ratio_cols] = eddy[log_ratio_cols].replace([np.inf, -np.inf], np.nan)
        print(f"Joined background means; computed log-ratio targets "
              f"({n_nonfinite} non-finite set to NaN)")
    else:
        print(f"(no background means at {bg_path} yet — log-ratio targets skipped)")

    out_path = resolve_gold_dir(EXPERIMENT, "eddy_pigment_table.parquet")
    eddy.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}  ({len(eddy)} rows x {len(eddy.columns)} cols)")


if __name__ == "__main__":
    main()
