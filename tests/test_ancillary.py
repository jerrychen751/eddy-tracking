from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from eddy_tracking.packages.sdp.ancillary import sample_ancillary
from eddy_tracking.preprocess.sss import read_multiple_sss
from eddy_tracking.preprocess.sst import read_multiple_sst


def test_read_multiple_sst_returns_time_indexed_dataframe(
    tmp_path: Path,
) -> None:
    _write_sst(
        tmp_path / "AQUA_MODIS.20250109_20250116.L3m.8D.SST.sst.4km.nc",
        2.0,
    )
    _write_sst(
        tmp_path / "AQUA_MODIS.20250101_20250108.L3m.8D.SST.sst.4km.nc",
        1.0,
    )

    df = read_multiple_sst(tmp_path)

    assert isinstance(df, pd.DataFrame)
    assert df.index.names == ["time", "lat", "lon"]
    assert df.columns.tolist() == ["sst"]
    assert df["sst"].tolist() == [1.0, 2.0]
    assert df.index.get_level_values("time").unique().tolist() == [
        pd.Timestamp("2025-01-04"),
        pd.Timestamp("2025-01-12"),
    ]


def test_read_multiple_sss_returns_time_indexed_dataframe(
    tmp_path: Path,
) -> None:
    _write_sss(
        tmp_path / "SMAP_L3_SSS_20250109_8DAYS_V5.0.nc4",
        32.0,
    )
    _write_sss(
        tmp_path / "SMAP_L3_SSS_20250101_8DAYS_V5.0.nc4",
        31.0,
    )

    df = read_multiple_sss(tmp_path)

    assert isinstance(df, pd.DataFrame)
    assert df.index.names == ["time", "latitude", "longitude"]
    assert df.columns.tolist() == ["smap_sss"]
    assert df["smap_sss"].tolist() == [31.0, 32.0]
    assert df.index.get_level_values("time").unique().tolist() == [
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-09"),
    ]


def test_read_multiple_ancillary_rejects_empty_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match=r"No SST \.nc files"):
        read_multiple_sst(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"No SSS \.nc4 files"):
        read_multiple_sss(tmp_path)


def test_sample_ancillary_selects_nearest_grid_points(tmp_path: Path) -> None:
    _write_sst_grid(
        tmp_path / "AQUA_MODIS.20250101_20250108.L3m.8D.SST.sst.4km.nc"
    )
    _write_sss_grid(
        tmp_path / "SMAP_L3_SSS_20250101_8DAYS_V5.0.nc4"
    )
    sst_df = read_multiple_sst(tmp_path)
    sss_df = read_multiple_sss(tmp_path)

    sst, sss = sample_ancillary(
        sst_df,
        sss_df,
        lons=np.array([-69.8, -60.1]),
        lats=np.array([30.2, 39.8]),
        times=np.array(["2025-01-04", "2025-01-04"], dtype="datetime64[ns]"),
    )

    np.testing.assert_array_equal(sst, [1.0, 4.0])
    np.testing.assert_array_equal(sss, [31.0, 34.0])


def _write_sst(path: Path, value: float) -> None:
    xr.Dataset(
        {"sst": (("lat", "lon"), [[value]])},
        coords={"lat": [30.0], "lon": [-70.0]},
    ).to_netcdf(path)


def _write_sss(path: Path, value: float) -> None:
    xr.Dataset(
        {"smap_sss": (("latitude", "longitude"), [[value]])},
        coords={"latitude": [30.0], "longitude": [-70.0]},
    ).to_netcdf(path)


def _write_sst_grid(path: Path) -> None:
    xr.Dataset(
        {
            "sst": (
                ("lat", "lon"),
                np.array([[1.0, 2.0], [3.0, 4.0]]),
            )
        },
        coords={"lat": [30.0, 40.0], "lon": [-70.0, -60.0]},
    ).to_netcdf(path)


def _write_sss_grid(path: Path) -> None:
    xr.Dataset(
        {
            "smap_sss": (
                ("latitude", "longitude"),
                np.array([[31.0, 32.0], [33.0, 34.0]]),
            )
        },
        coords={
            "latitude": [30.0, 40.0],
            "longitude": [-70.0, -60.0],
        },
    ).to_netcdf(path)
