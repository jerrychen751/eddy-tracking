"""
Find the Gulf Stream jet-core axis per date from SWOT SSH, and classify each eddy track's movement relative to it.

The axis is an ordered streamline traced through the fastest Gulf Stream core flow in the Gulf Stream latitude band. Movement (NN/NS/SN/SS) compares an eddy's geographic side of the axis (north/south) at birth vs death.

Outputs to silver/gulf_stream/:
  - streamline.parquet holds one row per ordered centerline point: date, point_idx, lon, lat
  - eddy_movement.parquet holds one row per (polarity, track_id): movement class, sides, and signed axis distances in km
"""

import datetime as dt
import sys

import numpy as np
import pandas as pd

from eddy_tracking.config import load_config, resolve_data_dir, resolve_output_dir
from eddy_tracking.preprocess.streamline import GulfStreamCenterline, compute_signed_distance_km, trace_streamline_for_file
from eddy_tracking.preprocess.swot import index_swot_files_by_date
from eddy_tracking.preprocess.tracks import load_track_observations


def main(experiment: str) -> None:
    """Trace daily streamlines and write streamline and movement Parquet files."""
    cfg = load_config(experiment)
    swot_dir = resolve_data_dir(cfg, "swot_dir")
    out_dir = resolve_output_dir(experiment, "gulf_stream")
    out_dir.mkdir(parents=True, exist_ok=True)

    swot_files = index_swot_files_by_date(swot_dir)
    print(
        "status: computing_gulf_stream_streamline\n"
        f"swot_days: {len(swot_files)}"
    )
    streamline_rows = []
    centerline_by_date: dict[dt.date, GulfStreamCenterline] = {}
    for date, fp in sorted(swot_files.items()):
        centerline = trace_streamline_for_file(fp)
        centerline_by_date[date] = centerline
        streamline_rows.append(pd.DataFrame({
            "date": pd.Timestamp(date),
            "point_idx": np.arange(centerline.lon.size, dtype=int),
            "lon": centerline.lon,
            "lat": centerline.lat,
        }))
    streamline_df = pd.concat(streamline_rows, ignore_index=True)
    streamline_df["date"] = pd.to_datetime(streamline_df["date"])
    streamline_df.to_parquet(out_dir / "streamline.parquet", index=False)
    print(
        "output_file: streamline.parquet\n"
        "median_centerline_latitude_degrees_north: "
        f"{streamline_df['lat'].median():.2f}"
    )

    obs = load_track_observations(experiment)
    movement_rows = []
    for (polarity, track_id), grp in obs.groupby(["polarity", "track_id"]):  # pyright: ignore[reportGeneralTypeIssues]
        grp = grp.sort_values("date")
        birth, death = grp.iloc[0], grp.iloc[-1]
        birth_distance_km, birth_side = _classify_streamline_side(centerline_by_date, birth)
        death_distance_km, death_side = _classify_streamline_side(centerline_by_date, death)
        movement_rows.append({
            "polarity": polarity,
            "track_id": track_id,
            "birth_date": birth["date"],
            "death_date": death["date"],
            "birth_side": birth_side,
            "death_side": death_side,
            "birth_distance_km": birth_distance_km,
            "death_distance_km": death_distance_km,
            "movement": (birth_side + death_side) if birth_side and death_side else "",
        })
    movement_df = pd.DataFrame(movement_rows)
    movement_df.to_parquet(out_dir / "eddy_movement.parquet", index=False)
    counts = movement_df["movement"].replace("", "unknown").value_counts().to_dict()
    print(
        "output_file: eddy_movement.parquet\n"
        f"tracks_written: {len(movement_df)}\n"
        f"movement_classes: {counts}"
    )


def _classify_streamline_side(centerline_by_date, row) -> tuple[float, str]:
    """Signed distance + side for one observation, using its date's streamline."""
    centerline = centerline_by_date.get(row["date"].date())
    if centerline is None:
        return np.nan, ""
    return compute_signed_distance_km(centerline.lon, centerline.lat, row["center_lon"], row["center_lat"])


if __name__ == "__main__":
    main(sys.argv[1])
