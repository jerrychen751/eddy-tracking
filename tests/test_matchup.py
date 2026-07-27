from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from eddy_tracking.validation import matchup


@pytest.mark.parametrize(
    ("list_matchups", "collection_query"),
    [
        (
            matchup.list_pace_l2_matchups,
            {
                "short_name": "PACE_OCI_L2_AOP",
                "version": "3.2",
            },
        ),
        (
            matchup.list_sss_matchups,
            {
                "concept_id": "C2208422957-POCLOUD",
                "granule_name": "SMAP_L3_SSS_*_8DAYS_V5.0",
            },
        ),
        (
            matchup.list_sst_matchups,
            {
                "concept_id": "C1615905770-OB_DAAC",
                "granule_name": (
                    "AQUA_MODIS.*.L3m.8D.SST.sst.4km.nc"
                ),
            },
        ),
    ],
)
def test_list_matchups_searches_product_at_point_and_utc_window(
    list_matchups: Callable[..., object],
    collection_query: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    granules = [Mock(), Mock()]
    search_data = Mock(return_value=granules)
    monkeypatch.setattr(matchup.earthaccess, "search_data", search_data)

    result = list_matchups(
        lon=-70.0,
        lat=35.0,
        measurement_dttm=datetime(
            2025,
            1,
            4,
            7,
            30,
            tzinfo=timezone(timedelta(hours=-5)),
        ),
        matchup_window=timedelta(hours=3),
        count=5,
    )

    assert result == granules
    search_data.assert_called_once_with(
        point=(-70.0, 35.0),
        temporal=(
            "2025-01-04T09:30:00Z",
            "2025-01-04T15:30:00Z",
        ),
        count=5,
        **collection_query,
    )


def test_list_matchups_searches_all_results_when_count_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_data = Mock(return_value=[])
    monkeypatch.setattr(matchup.earthaccess, "search_data", search_data)

    matchup.list_pace_l2_matchups(
        lon=-70.0,
        lat=35.0,
        measurement_dttm=datetime(2025, 1, 4, tzinfo=timezone.utc),
        matchup_window=timedelta(hours=3),
    )

    assert search_data.call_args.kwargs["count"] == -1


def test_list_matchups_requires_timezone() -> None:
    with pytest.raises(
        ValueError,
        match="measurement_dttm must have timezone information",
    ):
        matchup.list_pace_l2_matchups(
            lon=-70.0,
            lat=35.0,
            measurement_dttm=datetime(2025, 1, 4),
            matchup_window=timedelta(hours=3),
        )
