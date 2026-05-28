"""Spatial and temporal subsetting predicates for collocation.

Extracted from collocate_pace.py so the logic is unit-testable — that script
parses CLI args at import time and cannot be imported directly.
"""
import datetime as dt


def parse_date_range(date_range: list[str] | None) -> tuple[dt.date, dt.date] | None:
    """Parse a ["YYYY-MM-DD", "YYYY-MM-DD"] config value into (start, end) dates.

    Returns None when no window is configured, which disables temporal filtering.
    """
    if not date_range:
        return None
    return dt.date.fromisoformat(date_range[0]), dt.date.fromisoformat(date_range[1])


def in_subset(
    center_lon: float,
    center_lat: float,
    day: dt.date,
    region: dict | None,
    date_range: tuple[dt.date, dt.date] | None,
) -> bool:
    """Whether an eddy observation falls within the optional box and date window.

    region is {"lon_range": [lo, hi], "lat_range": [lo, hi]} in -180/180 longitude,
    or None for no spatial filter; date_range is an inclusive (start, end) or None.
    Both bounds inclusive; a None filter always passes, so omitting both reproduces
    the original unfiltered behavior.
    """
    if date_range is not None:
        start, end = date_range
        if not (start <= day <= end):
            return False
    if region is not None:
        lon_lo, lon_hi = region["lon_range"]
        lat_lo, lat_hi = region["lat_range"]
        if not (lon_lo <= center_lon <= lon_hi and lat_lo <= center_lat <= lat_hi):
            return False
    return True
