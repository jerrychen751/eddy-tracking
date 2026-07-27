"""Sample SST and SSS grids for the SDP model."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sample_ancillary(
    sst_df: pd.DataFrame,
    sss_df: pd.DataFrame,
    lons: np.ndarray,
    lats: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample SST and SSS at given (lon, lat, time) points via nearest-neighbor.

    Uses nearest-neighbor indexing in all three dimensions. Returns NaN where
    no data is available, such as land masks or gaps.

    Note:
        SST grid uses coord names 'lat'/'lon'.
        SSS grid uses 'latitude'/'longitude'.
    """
    sst_values = _sample_nearest(
        sst_df,
        value_column="sst",
        latitude_name="lat",
        longitude_name="lon",
        lons=lons,
        lats=lats,
        times=times,
    )
    sss_values = _sample_nearest(
        sss_df,
        value_column="smap_sss",
        latitude_name="latitude",
        longitude_name="longitude",
        lons=lons,
        lats=lats,
        times=times,
    )
    return sst_values, sss_values


def _sample_nearest(
    grid_df: pd.DataFrame,
    value_column: str,
    latitude_name: str,
    longitude_name: str,
    lons: np.ndarray,
    lats: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    nearest_index = pd.MultiIndex.from_arrays(
        [
            _nearest_level_values(grid_df, "time", pd.DatetimeIndex(times)),
            _nearest_level_values(grid_df, latitude_name, lats),
            _nearest_level_values(grid_df, longitude_name, lons),
        ],
        names=("time", latitude_name, longitude_name),
    )
    return grid_df[value_column].reindex(nearest_index).to_numpy(dtype=float)


def _nearest_level_values(
    grid_df: pd.DataFrame,
    level_name: str,
    targets: np.ndarray | pd.Index,
) -> pd.Index:
    level_values = (
        pd.Index(grid_df.index.get_level_values(level_name).unique())
        .sort_values()
    )
    positions = level_values.get_indexer(targets, method="nearest")
    return level_values.take(positions)
