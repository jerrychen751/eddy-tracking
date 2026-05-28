import datetime as dt

from utils.subset import parse_date_range, in_subset

REGION = {"lon_range": [-75, -58], "lat_range": [32, 42]}
WINDOW = (dt.date(2024, 12, 1), dt.date(2025, 3, 31))


def test_parse_date_range_none():
    assert parse_date_range(None) is None
    assert parse_date_range([]) is None


def test_parse_date_range_values():
    assert parse_date_range(["2024-12-01", "2025-03-31"]) == (
        dt.date(2024, 12, 1), dt.date(2025, 3, 31)
    )


def test_no_filters_always_passes():
    assert in_subset(0.0, 0.0, dt.date(2030, 1, 1), None, None) is True


def test_date_window():
    assert in_subset(-65, 37, dt.date(2025, 1, 15), None, WINDOW) is True
    assert in_subset(-65, 37, dt.date(2024, 11, 30), None, WINDOW) is False
    assert in_subset(-65, 37, dt.date(2025, 4, 1), None, WINDOW) is False


def test_region_box():
    assert in_subset(-65, 37, dt.date(2025, 1, 15), REGION, None) is True
    assert in_subset(-80, 37, dt.date(2025, 1, 15), REGION, None) is False  # lon too west
    assert in_subset(-65, 45, dt.date(2025, 1, 15), REGION, None) is False  # lat too north


def test_combined_and_boundaries():
    assert in_subset(-65, 37, dt.date(2025, 1, 15), REGION, WINDOW) is True
    assert in_subset(-65, 37, dt.date(2025, 7, 1), REGION, WINDOW) is False   # out of window
    assert in_subset(-50, 37, dt.date(2025, 1, 15), REGION, WINDOW) is False  # out of box
    assert in_subset(-75, 32, dt.date(2024, 12, 1), REGION, WINDOW) is True   # inclusive edges
    assert in_subset(-58, 42, dt.date(2025, 3, 31), REGION, WINDOW) is True
