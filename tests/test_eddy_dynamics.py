import datetime as dt

import numpy as np
import pandas as pd
import pytest

from eddy_dynamics import rossby_stats, track_observations_to_frame


class DummyTracks:
    virtual = np.array([False, True, False])
    time = np.array([27394, 27395, 27396])
    track = np.array([10, 11, 12])
    longitude = np.array([281.0, 282.0, 283.0])
    latitude = np.array([35.0, 36.0, 37.0])
    contour_lon_e = np.array([
        [280.9, 281.1, 281.1, 280.9],
        [281.9, 282.1, 282.1, 281.9],
        [282.9, 283.1, 283.1, 282.9],
    ])
    contour_lat_e = np.array([
        [34.9, 34.9, 35.1, 35.1],
        [35.9, 35.9, 36.1, 36.1],
        [36.9, 36.9, 37.1, 37.1],
    ])


def test_track_observations_to_frame_drops_virtual_and_wraps_longitudes():
    df = track_observations_to_frame(DummyTracks(), "cyclone")

    assert df["polarity"].tolist() == ["cyclone", "cyclone"]
    assert df["track_id"].tolist() == [10, 12]
    assert df["date"].tolist() == [
        pd.Timestamp(dt.date(2025, 1, 1)),
        pd.Timestamp(dt.date(2025, 1, 3)),
    ]
    assert df["center_lon"].tolist() == [-79.0, -77.0]
    assert df["center_lat"].tolist() == [35.0, 37.0]
    assert df.iloc[0]["contour_lon"].tolist() == pytest.approx([-79.1, -78.9, -78.9, -79.1])


def test_rossby_stats_samples_center_and_contour_interior():
    lon = np.array([0.0, 1.0, 2.0])
    lat = np.array([0.0, 1.0, 2.0])
    rossby = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ])
    contour_lon = np.array([0.4, 1.6, 1.6, 0.4])
    contour_lat = np.array([0.4, 0.4, 1.6, 1.6])

    stats = rossby_stats(rossby, lon, lat, contour_lon, contour_lat, 1.0, 1.0)

    assert stats["rossby_center"] == pytest.approx(0.5)
    assert stats["rossby_mean"] == pytest.approx(0.5)
    assert stats["rossby_abs_mean"] == pytest.approx(0.5)
    assert stats["rossby_min"] == pytest.approx(0.5)
    assert stats["rossby_max"] == pytest.approx(0.5)
    assert stats["n_rossby_pixels"] == 1
