"""
Module to find and download PACE L2 granules.
"""

from datetime import datetime, timedelta, timezone

import earthaccess
from earthaccess import DataGranule

PACE_COLLECTION_SHORT_NAME = "PACE_OCI_L2_AOP"
PACE_VERSION = "3.2"
MATCHUP_TIME_WINDOW = timedelta(hours=3)


def list_matchup_granules(
    lon: float,
    lat: float,
    measurement_dttm: datetime,
    matchup_window: timedelta,
    count: int | None = None,
) -> list[DataGranule]:
    """
    Search NASA Earthdata for granules in the PACE_OCI_L2_AOP V3.2 collection that match with the provided coordinates and datetime, with an optional limit on granules returned.

    Requires `measurement_dttm` to have a timezone.
    """
    if measurement_dttm.tzinfo is None:
        raise ValueError("measurement_dttm must have timezone information")

    measurement_dttm = measurement_dttm.astimezone(timezone.utc)
    start_time = (measurement_dttm - matchup_window).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_time = (measurement_dttm + matchup_window).strftime("%Y-%m-%dT%H:%M:%SZ")

    max_returned = count if isinstance(count, int) else -1

    return earthaccess.search_data(
        short_name=PACE_COLLECTION_SHORT_NAME,
        version=PACE_VERSION,
        point=(lon, lat),
        temporal=(start_time, end_time),
        count=max_returned,
    )
