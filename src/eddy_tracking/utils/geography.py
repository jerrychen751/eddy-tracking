import numpy as np
import pandas as pd


def calculate_dist_to_point(
    lon: pd.Series | pd.DataFrame,
    lat: pd.Series | pd.DataFrame,
    target_lon: float,
    target_lat: float,
) -> pd.Series | pd.DataFrame:
    """Return distances in kilometers with the input pandas type and labels."""
    if isinstance(lon, pd.Series) and isinstance(lat, pd.Series):
        if not lon.index.equals(lat.index):
            raise ValueError("lon and lat must have matching indexes")
    elif isinstance(lon, pd.DataFrame) and isinstance(lat, pd.DataFrame):
        if not lon.index.equals(lat.index) or not lon.columns.equals(lat.columns):
            raise ValueError("lon and lat must have matching indexes and columns")
    else:
        raise TypeError(
            "lon and lat must both be pandas Series or both be pandas DataFrames"
        )

    lon_rad = np.radians(lon.to_numpy(dtype=float))
    lat_rad = np.radians(lat.to_numpy(dtype=float))
    target_lon_rad = np.radians(target_lon)
    target_lat_rad = np.radians(target_lat)

    delta_lon = lon_rad - target_lon_rad
    delta_lat = lat_rad - target_lat_rad

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat_rad)
        * np.cos(target_lat_rad)
        * np.sin(delta_lon / 2) ** 2
    )

    # Clipping prevents small floating-point errors outside the valid range.
    central_angle = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    distance_km = 6371.0088 * central_angle

    if isinstance(lon, pd.Series):
        return pd.Series(distance_km, index=lon.index, name="distance_km")
    return pd.DataFrame(distance_km, index=lon.index, columns=lon.columns)


def get_5x5_pace_l2_matchups(
    df: pd.DataFrame,
    target_lon: float,
    target_lat: float,
) -> pd.DataFrame:
    """
    Return rows of the dataframe which are the closest 5x5 points geographically to the target lon & lat. Assumes columns "longitude" and "latitude" exist in the dataframe.
    """
    distances = calculate_dist_to_point(
        df["longitude"],
        df["latitude"],
        target_lon,
        target_lat,
    )
    closest_indices = distances.nsmallest(25).index
    matchups = df.loc[closest_indices].copy().reset_index(drop=True)
    matchups.attrs = df.attrs.copy()
    return matchups
