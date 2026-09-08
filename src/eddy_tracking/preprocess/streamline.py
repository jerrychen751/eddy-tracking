import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import uniform_filter

KM_PER_DEG_LAT = 111.0


class GulfStreamCenterline:
    """
    Ordered Gulf Stream centerline points.

    The points are stored in streamline order, not sorted by longitude. This allows meanders and repeated longitude values.
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
        speed_threshold_percentile: int = 70,
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

        # Seed from the strongest *coherent* flow rather than the single fastest pixel: the cell with the highest mean speed over a fully-finite seed_window box.
        # This avoids lone coastal spikes whose NaN neighbours would make the interpolator return NaN at the seed and end the trace on its first step.
        seed_window = 5  # box width (cells) for the coherent-flow seed; ~70 km at 1/8 deg
        finite = np.isfinite(speed)
        box = seed_window * seed_window
        finite_in_box = uniform_filter(finite.astype(float), seed_window, mode="constant") * box
        local_mean = uniform_filter(np.where(finite, speed, 0.0), seed_window, mode="constant") * box / np.maximum(finite_in_box, 1.0)
        seed_score = np.where(finite_in_box >= box - 0.5, local_mean, -np.inf)
        if seed_score.max() <= 0:
            return cls(np.array([]), np.array([]))
        # flat argmax over seed_score (n_lat, n_lon) -> (lat_idx, lon_idx)
        origin_lat_idx, origin_lon_idx = np.unravel_index(
            np.argmax(seed_score), speed.shape
        )
        # origin, point, and every path row are (2,) holding (lat, lon) in degrees.
        origin = np.array(
            [lat[origin_lat_idx], lon[origin_lon_idx]], dtype=float
        )

        streamline_step_km = 5.0
        max_gap_steps = 10
        max_trace_steps = 2000
        loop_min_points = 40
        loop_skip_recent_points = 30

        def trace(direction: int) -> list[np.ndarray]:
            point = origin.copy()
            path: list[np.ndarray] = []
            gap = 0
            last_strong = -1

            for _ in range(max_trace_steps):
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
                    if gap > max_gap_steps:
                        break

                point = point + np.array([
                    streamline_step_km / KM_PER_DEG_LAT * direction * v / spd,
                    streamline_step_km
                    / (KM_PER_DEG_LAT * np.cos(np.radians(point[0])))
                    * direction * u / spd,
                ])

                if not (lat.min() <= point[0] <= lat.max() and lon.min() <= point[1] <= lon.max()):
                    break
                if len(path) > loop_min_points:
                    # list of (2,) -> earlier (n_earlier, 2)
                    earlier = np.array(path[:-loop_skip_recent_points])
                    if (
                        np.hypot(earlier[:, 0] - point[0], earlier[:, 1] - point[1]).min()
                        < streamline_step_km / KM_PER_DEG_LAT
                    ):
                        break

            return path[:last_strong + 1]

        upstream = trace(-1)
        downstream = trace(+1)
        # two lists of (2,) -> path (n_points, 2)
        path = np.array(upstream[::-1] + downstream[1:])
        if path.size == 0:
            return cls(np.array([]), np.array([]))
        # path (n_points, 2) -> lon (n_points,), lat (n_points,)
        return cls(path[:, 1], path[:, 0])


def trace_streamline_for_file(fp: Path) -> GulfStreamCenterline:
    """Ordered Gulf Stream streamline for one SWOT day."""
    with xr.open_dataset(fp) as ds:
        if "time" in ds.ugos.dims:
            # ugos and vgos (1, n_lat, n_lon) -> (n_lat, n_lon)
            ds = ds.isel(time=0)
        lon = ds.longitude.to_numpy()
        lat = ds.latitude.to_numpy()
        ugos = ds.ugos.to_numpy()
        vgos = ds.vgos.to_numpy()

    # The Gulf Stream core stays well inside this latitude band within the ROI; restricting the search keeps the line off coastal/subpolar currents.
    gs_lat_band = (32.0, 43.0)
    in_band = (lat >= gs_lat_band[0]) & (lat <= gs_lat_band[1])
    # in_band (n_lat,) -> (n_lat, 1) to broadcast down each column of ugos and vgos (n_lat, n_lon)
    ugos = np.where(in_band[:, np.newaxis], ugos, np.nan)
    vgos = np.where(in_band[:, np.newaxis], vgos, np.nan)
    return GulfStreamCenterline.from_streamline_field(ugos, vgos, lon, lat)


def index_centerlines_by_date(streamline_df: pd.DataFrame) -> dict[dt.date, GulfStreamCenterline]:
    """Read silver streamline rows into ordered centerlines keyed by date."""
    centerlines: dict[dt.date, GulfStreamCenterline] = {}
    for date, grp in streamline_df.groupby("date"):
        ordered = grp.sort_values("point_idx")
        centerlines[pd.Timestamp(date).date()] = GulfStreamCenterline(  # pyright: ignore[reportArgumentType]
            ordered["lon"].to_numpy(),
            ordered["lat"].to_numpy(),
        )
    return centerlines


def compute_signed_distance_km(
    streamline_lon: np.ndarray, streamline_lat: np.ndarray, center_lon: float, center_lat: float
) -> tuple[float, str]:
    """
    Signed shortest distance from an eddy center to an ordered streamline.

    Positive (side 'N') means the eddy is geographically north of the nearest point on the jet, negative ('S') south. Returns (nan, '') if the streamline has too few finite points.
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

    # x and y (n_finite,) -> start and end (n_finite - 1, 2) holding (x_km, y_km) per segment
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
    # t (n_segments,) -> (n_segments, 1) to scale seg (n_segments, 2), then closest (n_segments, 2) -> dist (n_segments,)
    closest = start + t[:, np.newaxis] * seg
    dist = np.hypot(closest[:, 0], closest[:, 1])
    nearest_idx = int(np.argmin(dist))
    closest_lat = start_lat[nearest_idx] + t[nearest_idx] * (
        end_lat[nearest_idx] - start_lat[nearest_idx]
    )
    side = "N" if center_lat >= closest_lat else "S"
    signed = dist[nearest_idx] if side == "N" else -dist[nearest_idx]
    return float(signed), side
