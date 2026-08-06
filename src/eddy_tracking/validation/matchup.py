"""List PACE Level-2, SST, and SSS matchup granules."""

from datetime import datetime, timedelta, timezone

import earthaccess
from earthaccess import DataGranule

MATCHUP_TIME_WINDOW = timedelta(hours=3)


def list_pace_l2_matchups(
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
    return _list_matchups(
        lon,
        lat,
        measurement_dttm,
        matchup_window,
        count,
        short_name="PACE_OCI_L2_AOP",
        version="3.2",
    )


def list_sss_matchups(
    lon: float,
    lat: float,
    measurement_dttm: datetime,
    matchup_window: timedelta,
    count: int | None = None,
) -> list[DataGranule]:
    return _list_matchups(
        lon,
        lat,
        measurement_dttm,
        matchup_window,
        count,
        concept_id="C2208422957-POCLOUD",
        granule_name="SMAP_L3_SSS_*_8DAYS_V5.0",
    )


def list_sst_matchups(
    lon: float,
    lat: float,
    measurement_dttm: datetime,
    matchup_window: timedelta,
    count: int | None = None,
) -> list[DataGranule]:
    return _list_matchups(
        lon,
        lat,
        measurement_dttm,
        matchup_window,
        count,
        concept_id="C1615905770-OB_DAAC",
        granule_name="AQUA_MODIS.*.L3m.8D.SST.sst.4km.nc",
    )


def _list_matchups(
    lon: float,
    lat: float,
    measurement_dttm: datetime,
    matchup_window: timedelta,
    count: int | None,
    **collection_query: str,
) -> list[DataGranule]:
    if measurement_dttm.tzinfo is None:
        raise ValueError("measurement_dttm must have timezone information")

    measurement_dttm = measurement_dttm.astimezone(timezone.utc)
    start_time = (measurement_dttm - matchup_window).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_time = (measurement_dttm + matchup_window).strftime("%Y-%m-%dT%H:%M:%SZ")

    max_returned = count if isinstance(count, int) else -1

    return earthaccess.search_data(
        point=(lon, lat),
        temporal=(start_time, end_time),
        count=max_returned,
        **collection_query,
    )
