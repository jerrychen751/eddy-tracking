import datetime as dt

import numpy as np
import pandas as pd
import pytest

from gulf_stream import (
    GulfStreamCenterline,
    centerlines_by_date,
    centerline_to_frame,
    signed_distance_km,
)


def test_centerline_to_frame_preserves_ordered_polyline_points():
    centerline = GulfStreamCenterline(
        lon=np.array([-70.0, -69.5, -70.0]),
        lat=np.array([36.0, 37.0, 38.0]),
    )

    df = centerline_to_frame(dt.date(2025, 1, 15), centerline)

    assert list(df.columns) == ["date", "point_idx", "lon", "lat"]
    assert df["date"].tolist() == [pd.Timestamp("2025-01-15")] * 3
    assert df["point_idx"].tolist() == [0, 1, 2]
    assert df["lon"].tolist() == [-70.0, -69.5, -70.0]
    assert df["lat"].tolist() == [36.0, 37.0, 38.0]


def test_centerlines_by_date_reads_streamline_frame_in_point_order():
    df = pd.DataFrame({
        "date": [pd.Timestamp("2025-01-15"), pd.Timestamp("2025-01-15")],
        "point_idx": [1, 0],
        "lon": [-69.5, -70.0],
        "lat": [37.0, 36.0],
    })

    centerlines = centerlines_by_date(df)

    centerline = centerlines[dt.date(2025, 1, 15)]
    assert centerline.lon.tolist() == [-70.0, -69.5]
    assert centerline.lat.tolist() == [36.0, 37.0]


def test_signed_distance_uses_nearest_polyline_segment_for_repeated_longitudes():
    streamline_lon = np.array([0.0, 1.0, 1.0, 0.0])
    streamline_lat = np.array([0.0, 0.0, 1.0, 1.0])

    north_dist, north_side = signed_distance_km(streamline_lon, streamline_lat, 0.5, 1.2)
    south_dist, south_side = signed_distance_km(streamline_lon, streamline_lat, 0.5, 0.8)

    assert north_side == "N"
    assert north_dist == pytest.approx(22.2, rel=0.02)
    assert south_side == "S"
    assert south_dist == pytest.approx(-22.2, rel=0.02)


def test_streamline_field_traces_flow_direction_in_path_order():
    lon = np.linspace(-80.0, -79.0, 5)
    lat = np.linspace(34.0, 35.0, 5)
    ugos = np.ones((lat.size, lon.size))
    vgos = np.zeros((lat.size, lon.size))

    centerline = GulfStreamCenterline.from_streamline_field(
        ugos,
        vgos,
        lon,
        lat,
        speed_threshold_percentile=0,
    )

    assert centerline.lon.size > lon.size
    assert np.all(np.diff(centerline.lon) > 0)
    assert np.allclose(centerline.lat, lat[0])
