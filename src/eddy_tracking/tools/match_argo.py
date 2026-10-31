# Match each Argo profile to the eddies of its own day: one row per profile and eddy whose speed contour holds the profile or whose center lies within --max-radius-fraction speed radii of it

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path as MplPath

from eddy_tracking.preprocess.streamline import compute_signed_distance_km, index_centerlines_by_date
from eddy_tracking.preprocess.tracks import load_track_observations
from eddy_tracking.utils.cli import write_settings
from eddy_tracking.utils.files import write_parquet_atomically


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="eddy-tracking match-argo")
    parser.add_argument("--tracks", type=Path, required=True, help="folder track-eddies wrote")
    parser.add_argument("--argo", type=Path, required=True, help="core_profiles.nc that download-argo wrote")
    parser.add_argument("--axis", type=Path, required=True, help="folder trace-axis wrote")
    parser.add_argument("--max-radius-fraction", type=float, required=True, help="keep an eddy whose center is this many speed radii or fewer from the profile")
    parser.add_argument("--out", type=Path, required=True, help="Parquet file")
    args = parser.parse_args(argv)
    write_settings(args.out, "match-argo", args, operational=set())
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(args.argo) as argo:
        profiles = pd.DataFrame({
            "platform_number": argo["PLATFORM_NUMBER"].to_numpy(),
            "cycle_number": argo["CYCLE_NUMBER"].to_numpy(),
            "direction": argo["DIRECTION"].to_numpy(),
            "profile_time": argo["TIME"].to_numpy(),
            "profile_lon": argo["LONGITUDE"].to_numpy(),
            "profile_lat": argo["LATITUDE"].to_numpy(),
        })
    profiles["date"] = profiles["profile_time"].dt.normalize()
    observations = load_track_observations(args.tracks)
    observations_by_date = dict(tuple(observations.groupby("date")))  # pyright: ignore[reportArgumentType]

    matches = []
    for profile, lon, lat, day in zip(profiles.index, profiles["profile_lon"], profiles["profile_lat"], profiles["date"]):
        if day not in observations_by_date:
            continue
        day_observations = observations_by_date[day]
        # A 4 degree box first, because the contour test is slow and 1.5 speed radii of the largest ring stay inside it
        nearby = day_observations.loc[day_observations["center_lon"].sub(lon).abs().lt(4) & day_observations["center_lat"].sub(lat).abs().lt(4)]
        for polarity, track_id, contour_lon, contour_lat, center_lon, center_lat, radius_km in zip(
            nearby["polarity"], nearby["track_id"], nearby["contour_lon"], nearby["contour_lat"], nearby["center_lon"], nearby["center_lat"], nearby["radius_km"]
        ):
            distance_km = np.hypot((lon - center_lon) * 111.32 * np.cos(np.radians(lat)), (lat - center_lat) * 111.32)
            inside_contour = MplPath(np.column_stack([contour_lon, contour_lat])).contains_point((lon, lat))
            if inside_contour or distance_km / radius_km <= args.max_radius_fraction:
                matches.append({
                    "profile": profile,
                    "polarity": polarity,
                    "track_id": track_id,
                    "radius_fraction": distance_km / radius_km,
                    "inside_contour": inside_contour,
                })
    matches = pd.DataFrame(matches)

    # The crossing date is the first day an eddy that ends on the other side of the axis is observed on that side; an eddy that stays on one side has none
    movement = pd.read_parquet(args.axis / "eddy_movement.parquet")
    centerlines = index_centerlines_by_date(pd.read_parquet(args.axis / "streamline.parquet"))
    crossing = movement.loc[movement["birth_side"].ne(movement["death_side"]) & movement["birth_side"].ne("") & movement["death_side"].ne("")]
    crossing = crossing.merge(matches[["polarity", "track_id"]].drop_duplicates(), on=["polarity", "track_id"])
    crossing_observations = observations.merge(crossing[["polarity", "track_id", "death_side"]], on=["polarity", "track_id"])
    crossing_observations["side"] = [
        compute_signed_distance_km(centerlines[day.date()].lon, centerlines[day.date()].lat, lon, lat)[1] if day.date() in centerlines else ""
        for day, lon, lat in zip(crossing_observations["date"], crossing_observations["center_lon"], crossing_observations["center_lat"])
    ]
    crossing_dates = (
        crossing_observations.loc[crossing_observations["side"].eq(crossing_observations["death_side"])]
        .groupby(["polarity", "track_id"], as_index=False)["date"].min()
        .rename(columns={"date": "crossing_date"})
    )

    table = (
        matches.merge(profiles, left_on="profile", right_index=True)
        .merge(movement[["polarity", "track_id", "movement", "birth_date", "death_date"]], on=["polarity", "track_id"], how="left")
        .merge(crossing_dates, on=["polarity", "track_id"], how="left")
        .drop(columns="profile")
    )
    table["days_from_crossing"] = (table["date"] - table["crossing_date"]).dt.days
    # Positive (N) left of the flow at the nearest point on that day's axis, which is north where the jet flows east
    table["profile_axis_km"] = [
        compute_signed_distance_km(centerlines[day.date()].lon, centerlines[day.date()].lat, lon, lat)[0] if day.date() in centerlines else np.nan
        for day, lon, lat in zip(table["date"], table["profile_lon"], table["profile_lat"])
    ]
    write_parquet_atomically(table, args.out)

    print(
        "status: complete\n"
        f"output_file: {args.out}\n"
        f"profiles: {len(profiles)}\n"
        f"rows: {len(table)}\n"
        f"rows_inside_contour: {int(table['inside_contour'].to_numpy().sum())}\n"
        f"eddies: {len(table[['polarity', 'track_id']].drop_duplicates())}"
    )
