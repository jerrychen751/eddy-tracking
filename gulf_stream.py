"""
Find the Gulf Stream jet-core axis per date from SWOT SSH, and classify each
eddy track's movement relative to it.

The axis is an ordered streamline traced through the fastest Gulf Stream core
flow in the Gulf Stream latitude band. Movement (NN/NS/SN/SS) compares an
eddy's geographic side of the axis (north/south) at birth vs death.

Outputs to silver/gulf_stream/:
  - streamline.parquet     one row per ordered centerline point: date, point_idx, lon, lat
  - eddy_movement.parquet  one row per (polarity, track_id): movement class + sides
"""

import argparse
import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import uniform_filter

from utils.config import load_config, resolve_data_dir, resolve_output_dir

PET_EPOCH = dt.date(1950, 1, 1)
# The Gulf Stream core stays well inside this latitude band within the ROI;
# restricting the search keeps the line off coastal/subpolar currents.
GS_LAT_BAND = (32.0, 43.0)
SPEED_THRESHOLD_PERCENTILE = 70
SEED_WINDOW = 5  # box width (cells) for the coherent-flow seed; ~70 km at 1/8 deg
STREAMLINE_STEP_KM = 5.0
MAX_GAP_STEPS = 10
MAX_TRACE_STEPS = 2000
LOOP_MIN_POINTS = 40
LOOP_SKIP_RECENT_POINTS = 30
KM_PER_DEG_LAT = 111.0


def parse_file_date(fp: Path) -> dt.date:
    """Parse the first YYYYMMDD token in a SWOT filename."""
    return dt.datetime.strptime(re.search(r"\d{8}", fp.name).group(), "%Y%m%d").date()


class GulfStreamCenterline:
    """
    Ordered Gulf Stream centerline points.

    The points are stored in streamline order, not sorted by longitude. This
    allows meanders and repeated longitude values.
    """

    def __init__(self, lon: np.ndarray, lat: np.ndarray) -> None:
        lon_arr = np.asarray(lon, dtype=float)
        lat_arr = np.asarray(lat, dtype=float)
        if lon_arr.shape != lat_arr.shape:
            raise ValueError("lon and lat must have the same shape")
        self.lon = lon_arr
        self.lat = lat_arr

    @classmethod
    def from_streamline_field(
        cls,
        ugos: np.ndarray,
        vgos: np.ndarray,
        lon: np.ndarray,
        lat: np.ndarray,
        speed_threshold_percentile: int = SPEED_THRESHOLD_PERCENTILE,
    ) -> "GulfStreamCenterline":
        """
        Trace the Gulf Stream core by following the local surface-current direction.

        Starts at the cell with the strongest local-average flow, traces downstream and upstream, allows short slow gaps, trims weak tails, and stops at the grid edge, land, or a curl back onto an earlier part of the path.
        """
        speed = np.hypot(ugos, vgos)
        if not np.any(np.isfinite(speed)):
            return cls(np.array([]), np.array([]))

        threshold = np.nanpercentile(speed, speed_threshold_percentile)
        u_at = RegularGridInterpolator((lat, lon), ugos, bounds_error=False, fill_value=np.nan)
        v_at = RegularGridInterpolator((lat, lon), vgos, bounds_error=False, fill_value=np.nan)

        # Seed from the strongest *coherent* flow rather than the single fastest pixel: the cell with the highest mean speed over a fully-finite SEED_WINDOW box.
        # This avoids lone coastal spikes whose NaN neighbours would make the interpolator return NaN at the seed and end the trace on its first step.
        finite = np.isfinite(speed)
        box = SEED_WINDOW * SEED_WINDOW
        finite_in_box = uniform_filter(finite.astype(float), SEED_WINDOW, mode="constant") * box
        local_mean = uniform_filter(np.where(finite, speed, 0.0), SEED_WINDOW, mode="constant") * box / np.maximum(finite_in_box, 1.0)
        seed_score = np.where(finite_in_box >= box - 0.5, local_mean, -np.inf)
        if seed_score.max() <= 0:
            return cls(np.array([]), np.array([]))
        origin_lat_idx, origin_lon_idx = np.unravel_index(
            np.argmax(seed_score), speed.shape
        )
        origin = np.array(
            [lat[origin_lat_idx], lon[origin_lon_idx]], dtype=float
        )

        def trace(direction: int) -> list[np.ndarray]:
            point = origin.copy()
            path: list[np.ndarray] = []
            gap = 0
            last_strong = -1

            for _ in range(MAX_TRACE_STEPS):
                u = float(u_at([point])[0])
                v = float(v_at([point])[0])
                spd = float(np.hypot(u, v))
                if not np.isfinite(spd) or spd == 0:
                    break

                path.append(point.copy())
                if spd >= threshold:
                    gap = 0
                    last_strong = len(path) - 1
                else:
                    gap += 1
                    if gap > MAX_GAP_STEPS:
                        break

                point = point + np.array([
                    STREAMLINE_STEP_KM / KM_PER_DEG_LAT * direction * v / spd,
                    STREAMLINE_STEP_KM
                    / (KM_PER_DEG_LAT * np.cos(np.radians(point[0])))
                    * direction * u / spd,
                ])

                if not (lat.min() <= point[0] <= lat.max() and lon.min() <= point[1] <= lon.max()):
                    break
                if len(path) > LOOP_MIN_POINTS:
                    earlier = np.array(path[:-LOOP_SKIP_RECENT_POINTS])
                    if (
                        np.hypot(earlier[:, 0] - point[0], earlier[:, 1] - point[1]).min()
                        < STREAMLINE_STEP_KM / KM_PER_DEG_LAT
                    ):
                        break

            return path[:last_strong + 1]

        upstream = trace(-1)
        downstream = trace(+1)
        path = np.array(upstream[::-1] + downstream[1:])
        if path.size == 0:
            return cls(np.array([]), np.array([]))
        return cls(path[:, 1], path[:, 0])


def trace_streamline_for_file(fp: Path) -> GulfStreamCenterline:
    """Ordered Gulf Stream streamline for one SWOT day."""
    with xr.open_dataset(fp) as ds:
        if "time" in ds.ugos.dims:
            ds = ds.isel(time=0)
        lon = ds.longitude.to_numpy()
        lat = ds.latitude.to_numpy()
        ugos = ds.ugos.to_numpy()
        vgos = ds.vgos.to_numpy()

    in_band = (lat >= GS_LAT_BAND[0]) & (lat <= GS_LAT_BAND[1])
    ugos = np.where(in_band[:, np.newaxis], ugos, np.nan)
    vgos = np.where(in_band[:, np.newaxis], vgos, np.nan)
    return GulfStreamCenterline.from_streamline_field(ugos, vgos, lon, lat)


def centerline_to_frame(date: dt.date, centerline: GulfStreamCenterline) -> pd.DataFrame:
    """Silver streamline rows for one date: an ordered polyline, not lat=f(lon)."""
    return pd.DataFrame({
        "date": pd.Timestamp(date),
        "point_idx": np.arange(centerline.lon.size, dtype=int),
        "lon": centerline.lon,
        "lat": centerline.lat,
    })


def index_centerlines_by_date(streamline_df: pd.DataFrame) -> dict[dt.date, GulfStreamCenterline]:
    """Read silver streamline rows into ordered centerlines keyed by date."""
    centerlines: dict[dt.date, GulfStreamCenterline] = {}
    for date, grp in streamline_df.groupby("date"):
        ordered = grp.sort_values("point_idx")
        centerlines[pd.Timestamp(date).date()] = GulfStreamCenterline(
            ordered["lon"].to_numpy(),
            ordered["lat"].to_numpy(),
        )
    return centerlines


def compute_signed_distance_km(
    streamline_lon: np.ndarray, streamline_lat: np.ndarray, center_lon: float, center_lat: float
) -> tuple[float, str]:
    """
    Signed shortest distance from an eddy center to an ordered streamline.

    Positive (side 'N') means the eddy is geographically north of the nearest
    point on the jet, negative ('S') south. Returns (nan, '') if the streamline
    has too few finite points.
    """
    streamline_lon = np.asarray(streamline_lon, dtype=float)
    streamline_lat = np.asarray(streamline_lat, dtype=float)
    finite = np.isfinite(streamline_lon) & np.isfinite(streamline_lat)
    if finite.sum() < 2:
        return np.nan, ""

    lon = streamline_lon[finite]
    lat = streamline_lat[finite]
    scale_x = KM_PER_DEG_LAT * np.cos(np.radians(center_lat))
    x = (lon - center_lon) * scale_x
    y = (lat - center_lat) * KM_PER_DEG_LAT

    start = np.column_stack([x[:-1], y[:-1]])
    end = np.column_stack([x[1:], y[1:]])
    seg = end - start
    seg_len2 = np.einsum("ij,ij->i", seg, seg)
    nonzero = seg_len2 > 0
    if not np.any(nonzero):
        return np.nan, ""

    start = start[nonzero]
    seg = seg[nonzero]
    seg_len2 = seg_len2[nonzero]
    start_lat = lat[:-1][nonzero]
    end_lat = lat[1:][nonzero]

    # The eddy center is the origin in this local km coordinate frame.
    t = np.clip(-np.einsum("ij,ij->i", start, seg) / seg_len2, 0.0, 1.0)
    closest = start + t[:, np.newaxis] * seg
    dist = np.hypot(closest[:, 0], closest[:, 1])
    nearest_idx = int(np.argmin(dist))
    closest_lat = start_lat[nearest_idx] + t[nearest_idx] * (
        end_lat[nearest_idx] - start_lat[nearest_idx]
    )
    side = "N" if center_lat >= closest_lat else "S"
    signed = dist[nearest_idx] if side == "N" else -dist[nearest_idx]
    return float(signed), side


def load_track_observations(
    cyclone_track_dir: Path, anticyclone_track_dir: Path
) -> pd.DataFrame:
    """
    Non-virtual eddy observations across both polarities, one row each.

    Columns: polarity, track_id, date, center_lon (-180/180), center_lat.
    """
    from utils.py_eddy_tracker.observations.tracking import TrackEddiesObservations

    frames = []
    for polarity, track_dir in [
        ("cyclone", cyclone_track_dir),
        ("anticyclone", anticyclone_track_dir),
    ]:
        zarr_path = track_dir / f"{track_dir.name}_tracks.zarr"
        tracked = TrackEddiesObservations.load_file(str(zarr_path))
        keep = ~tracked.virtual.astype(bool)
        days = [PET_EPOCH + dt.timedelta(days=int(t)) for t in tracked.time[keep]]
        frames.append(pd.DataFrame({
            "polarity": polarity,
            "track_id": tracked.track[keep].astype(int),
            "date": pd.to_datetime(days),
            "center_lon": (tracked.longitude[keep] + 180) % 360 - 180,
            "center_lat": tracked.latitude[keep],
        }))
    return pd.concat(frames, ignore_index=True)


def main(experiment: str | None = None) -> None:
    """Trace daily streamlines and write streamline and movement Parquet files."""
    if experiment is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("experiment")
        args = parser.parse_args()
        experiment = args.experiment

    cfg = load_config(experiment)
    swot_dir = resolve_data_dir(cfg, "swot_dir")
    out_dir = resolve_output_dir(experiment, "gulf_stream")
    cyclone_track_dir = resolve_output_dir(experiment, "eddy_track", "cyclone")
    anticyclone_track_dir = resolve_output_dir(experiment, "eddy_track", "anticyclone")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-date streamline from every SWOT day
    swot_files = sorted(swot_dir.glob("*.nc"))
    print(f"Computing Gulf Stream streamline for {len(swot_files)} SWOT days...")
    streamline_rows = []
    centerline_by_date: dict[dt.date, GulfStreamCenterline] = {}
    for fp in swot_files:
        date = parse_file_date(fp)
        centerline = trace_streamline_for_file(fp)
        centerline_by_date[date] = centerline
        streamline_rows.append(centerline_to_frame(date, centerline))
    streamline_df = pd.concat(streamline_rows, ignore_index=True)
    streamline_df["date"] = pd.to_datetime(streamline_df["date"])
    streamline_df.to_parquet(out_dir / "streamline.parquet", index=False)
    print(f"Wrote streamline.parquet: median centerline latitude {streamline_df['lat'].median():.2f} N")

    # Movement per track: side of the streamline at birth vs death
    obs = load_track_observations(cyclone_track_dir, anticyclone_track_dir)
    movement_rows = []
    for (polarity, track_id), grp in obs.groupby(["polarity", "track_id"]):
        grp = grp.sort_values("date")
        birth, death = grp.iloc[0], grp.iloc[-1]
        _, birth_side = _classify_streamline_side(centerline_by_date, birth)
        _, death_side = _classify_streamline_side(centerline_by_date, death)
        movement_rows.append({
            "polarity": polarity,
            "track_id": track_id,
            "birth_date": birth["date"],
            "death_date": death["date"],
            "birth_side": birth_side,
            "death_side": death_side,
            "movement": (birth_side + death_side) if birth_side and death_side else "",
        })
    movement_df = pd.DataFrame(movement_rows)
    movement_df.to_parquet(out_dir / "eddy_movement.parquet", index=False)
    counts = movement_df["movement"].replace("", "unknown").value_counts().to_dict()
    print(f"Wrote eddy_movement.parquet: {len(movement_df)} tracks, classes {counts}")


def _classify_streamline_side(centerline_by_date, row) -> tuple[float, str]:
    """Signed distance + side for one observation, using its date's streamline."""
    centerline = centerline_by_date.get(row["date"].date())
    if centerline is None:
        return np.nan, ""
    return compute_signed_distance_km(centerline.lon, centerline.lat, row["center_lon"], row["center_lat"])


if __name__ == "__main__":
    main()
