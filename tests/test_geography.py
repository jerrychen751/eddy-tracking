import numpy as np
import pandas as pd
import pytest

from eddy_tracking.utils.geography import (
    calculate_dist_to_point,
    get_5x5_pace_l2_matchups,
)


def test_calculate_dist_to_point_preserves_dataframe_labels() -> None:
    lon = pd.DataFrame(
        [[0.0, 90.0], [0.0, 90.0]],
        index=["north", "equator"],
        columns=["west", "east"],
    )
    lat = pd.DataFrame(
        [[45.0, 45.0], [0.0, 0.0]],
        index=lon.index,
        columns=lon.columns,
    )

    distance = calculate_dist_to_point(lon, lat, target_lon=0.0, target_lat=0.0)

    assert distance.index.equals(lon.index)
    assert distance.columns.equals(lon.columns)
    assert distance.loc["equator", "west"] == pytest.approx(0.0)
    assert distance.loc["equator", "east"] == pytest.approx(
        6371.0088 * np.pi / 2
    )


@pytest.mark.parametrize(
    ("lat_index", "lat_columns"),
    [
        (["different-row"], ["column"]),
        (["row"], ["different-column"]),
    ],
)
def test_calculate_dist_to_point_requires_matching_dataframe_labels(
    lat_index: list[str],
    lat_columns: list[str],
) -> None:
    lon = pd.DataFrame([[0.0]], index=["row"], columns=["column"])
    lat = pd.DataFrame([[0.0]], index=lat_index, columns=lat_columns)

    with pytest.raises(
        ValueError,
        match="lon and lat must have matching indexes and columns",
    ):
        calculate_dist_to_point(lon, lat, target_lon=0.0, target_lat=0.0)


def test_calculate_dist_to_point_preserves_series_index() -> None:
    index = pd.Index(["origin", "quarter-circle"], name="pixel")
    lon = pd.Series([0.0, 90.0], index=index, name="longitude")
    lat = pd.Series([0.0, 0.0], index=index, name="latitude")

    distance = calculate_dist_to_point(lon, lat, target_lon=0.0, target_lat=0.0)

    expected = pd.Series(
        [0.0, 6371.0088 * np.pi / 2],
        index=index,
        name="distance_km",
    )
    pd.testing.assert_series_equal(distance, expected)


def test_calculate_dist_to_point_requires_matching_series_indexes() -> None:
    lon = pd.Series([0.0], index=["lon-row"])
    lat = pd.Series([0.0], index=["lat-row"])

    with pytest.raises(ValueError, match="lon and lat must have matching indexes"):
        calculate_dist_to_point(lon, lat, target_lon=0.0, target_lat=0.0)


def test_calculate_dist_to_point_rejects_mixed_pandas_types() -> None:
    lon = pd.Series([0.0], index=["row"])
    lat = pd.DataFrame([[0.0]], index=["row"], columns=["column"])

    with pytest.raises(
        TypeError,
        match="lon and lat must both be pandas Series or both be pandas DataFrames",
    ):
        calculate_dist_to_point(lon, lat, target_lon=0.0, target_lat=0.0)


def test_get_5x5_pace_l2_matchups_selects_25_closest_rows() -> None:
    df = pd.DataFrame(
        {
            "scan_line": np.arange(30)[::-1],
            "pixel": np.arange(30) * 10,
            "longitude": np.arange(30, dtype=float),
            "latitude": np.zeros(30),
            "Rrs_400": np.arange(30, dtype=float) / 100,
        }
    )
    df.attrs["units"] = {"Rrs_400": "sr^-1"}

    matchups = get_5x5_pace_l2_matchups(
        df,
        target_lon=29.0,
        target_lat=0.0,
    )

    assert len(matchups) == 25
    assert matchups["longitude"].tolist() == list(
        np.arange(29, 4, -1, dtype=float)
    )
    assert matchups.index.tolist() == list(range(25))
    assert list(matchups.columns) == list(df.columns)
    assert matchups.attrs == df.attrs


def test_get_5x5_pace_l2_matchups_excludes_missing_coordinates(
) -> None:
    df = pd.DataFrame(
        {
            "longitude": [np.nan, *np.arange(25, dtype=float)],
            "latitude": np.zeros(26),
        }
    )

    matchups = get_5x5_pace_l2_matchups(
        df,
        target_lon=0.0,
        target_lat=0.0,
    )

    assert len(matchups) == 25
    assert matchups["longitude"].notna().all()
