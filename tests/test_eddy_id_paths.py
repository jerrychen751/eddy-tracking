from datetime import datetime
from pathlib import Path

from eddy_id import eddy_output_paths, parse_file_datetime


def test_parse_file_datetime_reads_first_yyyymmdd_token():
    assert parse_file_datetime(Path("dt_global_allsat_phy_l4_20250102_20250103.nc")) == (
        datetime(2025, 1, 2)
    )


def test_eddy_output_paths_are_explicit_write_file_targets():
    anti, cycl = eddy_output_paths(
        Path("/tmp/anticyclone"),
        Path("/tmp/cyclone"),
        datetime(2025, 1, 2),
    )

    assert anti == Path("/tmp/anticyclone/Anticyclonic_2025-01-02.nc")
    assert cycl == Path("/tmp/cyclone/Cyclonic_2025-01-02.nc")
    assert "%(" not in str(anti)
    assert "%(" not in str(cycl)
