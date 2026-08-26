"""
Assemble the gold eddy-pigment table: one analysis-ready row per eddy-day.

Aggregates per-pixel pigments to eddy-interior means, joins track and environmental features, and writes ``gold/eddy_pigment_table.parquet``.
"""

import argparse
import datetime as dt
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from gulf_stream import compute_signed_distance_km, index_centerlines_by_date
from utils.config import load_config, resolve_gold_dir, resolve_output_dir
from eddy_tracking.packages.py_eddy_tracker.observations.tracking import (
    TrackEddiesObservations,
)

# On-disk pigment name to the space-free suffix used in table columns.
PIGMENTS = {
    "T chla": "Tchla",
    "Zea": "Zea",
    "DV chla": "DV_chla",
    "ButFuco": "ButFuco",
    "HexFuco": "HexFuco",
    "Allo": "Allo",
    "MV chlb": "MV_chlb",
    "Neo": "Neo",
    "Viola": "Viola",
    "Fuco": "Fuco",
    "chl c1+c2": "Chlc12",
    "chl c3": "Chlc3",
    "Perid": "Perid",
}


def aggregate_eddy_days(experiment: str) -> pd.DataFrame:
    """Aggregate pigment files to one quality-control row per eddy-day."""
    rows = []
    for polarity, polarity_value in [("anticyclone", 0), ("cyclone", 1)]:
        pigment_dir = resolve_output_dir(experiment, "pigments", polarity)
        for pigment_path in sorted(pigment_dir.glob("*_pigments.parquet")):
            pigments = pd.read_parquet(pigment_path)
            for date, group in pigments.groupby("date"):
                row = {
                    "track_id": int(group["track_id"].iloc[0]),
                    "date": pd.Timestamp(date),  # pyright: ignore[reportArgumentType]
                    "polarity": polarity_value,
                    "center_lon": float(group["center_lon"].mean()),
                    "center_lat": float(group["center_lat"].mean()),
                    "valid_frac": float(group["coverage"].iloc[0]),
                    "n_eddy_pixels": int(len(group)),
                }
                for source_name, column_suffix in PIGMENTS.items():
                    row[f"eddy_mean_{column_suffix}"] = float(
                        group[source_name].mean()
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def build_track_features(experiment: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build observation features and birth/death dates for both polarities."""
    pet_epoch = dt.date(1950, 1, 1)
    observation_frames = []
    lifetime_rows = []
    for polarity, polarity_value in [("anticyclone", 0), ("cyclone", 1)]:
        track_dir = resolve_output_dir(experiment, "eddy_track", polarity)
        tracks = TrackEddiesObservations.load_file(
            str(track_dir / f"{track_dir.name}_tracks.zarr")
        )
        observed = ~tracks.virtual.astype(bool)
        dates = pd.to_datetime(
            [
                pet_epoch + dt.timedelta(days=int(time))
                for time in tracks.time[observed]
            ]
        )
        track_observations = pd.DataFrame(
            {
                "polarity": polarity_value,
                "track_id": tracks.track[observed].astype(int),
                "date": dates,
                "radius_km": tracks.radius_s[observed] / 1000.0,
                "amplitude_cm": tracks.amplitude[observed] * 100.0,
            }
        ).sort_values("date")
        observation_frames.append(track_observations)
        for track_id, group in track_observations.groupby("track_id"):
            lifetime_rows.append(
                {
                    "polarity": polarity_value,
                    "track_id": track_id,
                    "birth_date": group["date"].min(),
                    "death_date": group["date"].max(),
                }
            )
    return (
        pd.concat(observation_frames, ignore_index=True),
        pd.DataFrame(lifetime_rows),
    )


def load_eddy_dynamics(dynamics_dir: Path) -> pd.DataFrame:
    """Load available per-observation Rossby diagnostics for both polarities."""
    frames = []
    for polarity, polarity_value in [("anticyclone", 0), ("cyclone", 1)]:
        dynamics_path = dynamics_dir / polarity / "dynamics.parquet"
        if not dynamics_path.exists():
            continue
        dynamics = pd.read_parquet(dynamics_path)
        dynamics["polarity"] = polarity_value
        frames.append(
            dynamics[
                [
                    "polarity",
                    "track_id",
                    "date",
                    "rossby_center",
                    "rossby_mean",
                    "rossby_abs_mean",
                    "rossby_min",
                    "rossby_max",
                    "n_rossby_pixels",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_gs_distance(centerline_by_date, date, center_lon, center_lat) -> float:
    """Return signed distance in km to the nearest-date Gulf Stream line."""
    base = pd.Timestamp(date).date()
    centerline = next(
        (
            centerline_by_date[candidate_date]
            for delta in range(5)
            for candidate_date in (
                base - dt.timedelta(delta),
                base + dt.timedelta(delta),
            )
            if candidate_date in centerline_by_date
        ),
        None,
    )
    if centerline is None:
        return np.nan
    dist, _ = compute_signed_distance_km(
        centerline.lon, centerline.lat, center_lon, center_lat
    )
    return dist


def main(experiment: str | None = None) -> None:
    """Assemble and write the experiment's gold eddy-pigment table."""
    if experiment is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("experiment")
        experiment = cast(str, parser.parse_args().experiment)

    cfg = load_config(experiment)
    gulf_stream_dir = resolve_output_dir(experiment, "gulf_stream")
    background_dir = resolve_output_dir(experiment, "pigments", "background")
    dynamics_dir = resolve_output_dir(experiment, "eddy_dynamics")
    eddy_start = pd.Timestamp(cfg["base"]["time"]["eddy_date_range"][0])
    eddy_end = pd.Timestamp(cfg["base"]["time"]["eddy_date_range"][1])

    eddy_days = aggregate_eddy_days(experiment)
    if eddy_days.empty:
        print(
            "status: skipped\n"
            "reason: no_eddy_pigment_files"
        )
        return
    n_eddies = len(eddy_days[["polarity", "track_id"]].drop_duplicates())
    print(
        f"eddy_days_aggregated: {len(eddy_days)}\n"
        f"eddies: {n_eddies}"
    )

    min_eddy_pixels = 10
    n_before = len(eddy_days)
    eddy_days = eddy_days[
        eddy_days["n_eddy_pixels"] >= min_eddy_pixels
    ].reset_index(drop=True)
    n_dropped = n_before - len(eddy_days)
    if n_dropped:
        print(
            f"eddy_days_dropped: {n_dropped}\n"
            f"minimum_interior_pixels: {min_eddy_pixels}"
        )

    track_observations, track_lifetimes = build_track_features(experiment)
    eddy_days = pd.merge_asof(
        eddy_days.sort_values("date"),  # pyright: ignore[reportCallIssue]
        track_observations.sort_values("date"),
        on="date",
        by=["polarity", "track_id"],
        direction="nearest",
    ).merge(track_lifetimes, on=["polarity", "track_id"], how="left")

    lifetime_days = (eddy_days["death_date"] - eddy_days["birth_date"]).dt.days
    eddy_days["age_days"] = (
        eddy_days["date"] - eddy_days["birth_date"]
    ).dt.days
    eddy_days["age_frac"] = np.where(
        lifetime_days > 0,
        eddy_days["age_days"] / lifetime_days,
        np.nan,
    )
    # A composite date can fall just outside a track's daily detections; keep the life fraction within [0, 1].
    eddy_days["age_frac"] = eddy_days["age_frac"].clip(0, 1)
    eddy_days["birth_observed"] = eddy_days["birth_date"] > eddy_start
    eddy_days["death_observed"] = eddy_days["death_date"] < eddy_end
    # Cyclical yd/365 encoding; convention from Gregor et al. (2018)
    year_day = eddy_days["date"].dt.dayofyear - 1
    angle = 2 * np.pi * year_day / 365
    eddy_days["time_of_year_cos"] = np.cos(angle)
    eddy_days["time_of_year_sin"] = np.sin(angle)

    dynamics = load_eddy_dynamics(dynamics_dir)
    if not dynamics.empty:
        eddy_days = pd.merge_asof(
            eddy_days.sort_values("date"),
            dynamics.sort_values("date"),
            on="date",
            by=["polarity", "track_id"],
            direction="nearest",
        )
        print("rossby_diagnostics: joined")
    else:
        print(
            "rossby_diagnostics: skipped\n"
            "reason: no_eddy_dynamics_files\n"
            f"dynamics_dir: {dynamics_dir}"
        )

    movement = pd.read_parquet(gulf_stream_dir / "eddy_movement.parquet")
    movement["polarity"] = movement["polarity"].map({"anticyclone": 0, "cyclone": 1})  # pyright: ignore[reportArgumentType]
    eddy_days = eddy_days.merge(
        movement[["polarity", "track_id", "movement"]],
        on=["polarity", "track_id"],
        how="left",
    )
    streamline = pd.read_parquet(gulf_stream_dir / "streamline.parquet")
    centerline_by_date = index_centerlines_by_date(streamline)
    eddy_days["gs_dist_km"] = [
        compute_gs_distance(centerline_by_date, date, lon, lat)
        for date, lon, lat in zip(
            eddy_days["date"],
            eddy_days["center_lon"],
            eddy_days["center_lat"],
        )
    ]

    background_path = background_dir / "bg_mean.parquet"
    if background_path.exists():
        eddy_days = eddy_days.merge(
            pd.read_parquet(background_path), on="date", how="left"
        )
        log_ratio_columns = []
        for column_suffix in PIGMENTS.values():
            if f"bg_mean_{column_suffix}" in eddy_days:
                log_ratio_column = f"log_ratio_{column_suffix}"
                with np.errstate(divide="ignore"):
                    eddy_days[log_ratio_column] = np.log(
                        eddy_days[f"eddy_mean_{column_suffix}"]
                        / eddy_days[f"bg_mean_{column_suffix}"]
                    )
                log_ratio_columns.append(log_ratio_column)
        # A zero eddy mean (pigment below detection) gives -inf; treat it as missing.
        n_nonfinite = int(
            np.isinf(eddy_days[log_ratio_columns].to_numpy()).sum()
        )
        eddy_days[log_ratio_columns] = eddy_days[log_ratio_columns].replace(
            [np.inf, -np.inf], np.nan
        )
        print(
            "background_means: joined\n"
            "log_ratio_targets: computed\n"
            f"nonfinite_values_set_to_nan: {n_nonfinite}"
        )
    else:
        print(
            "log_ratio_targets: skipped\n"
            "reason: no_background_means\n"
            f"background_path: {background_path}"
        )

    out_path = resolve_gold_dir(experiment, "eddy_pigment_table.parquet")
    eddy_days.to_parquet(out_path, index=False)
    print(
        f"output_path: {out_path}\n"
        f"rows_written: {len(eddy_days)}\n"
        f"columns_written: {len(eddy_days.columns)}"
    )


if __name__ == "__main__":
    main()
