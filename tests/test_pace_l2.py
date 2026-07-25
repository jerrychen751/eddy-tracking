from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from eddy_tracking.preprocess.pace import read_pace_l2
from eddy_tracking.validation.quality import (
    BAILEY_WERDELL_2006_EXCLUDED_L2_FLAGS,
    apply_l2_quality_flags,
)


def test_read_pace_l2_joins_selected_groups(tmp_path: Path) -> None:
    path = tmp_path / "pace_l2.nc"
    _write_pace_l2(path)

    df = read_pace_l2(
        path,
        line_indexer=slice(0, 1),
        pixel_indexer=(1, 2),
    )

    assert df.shape == (2, 11)
    assert list(df.columns[:7]) == [
        "scan_line",
        "pixel",
        "datetime",
        "latitude",
        "longitude",
        "aot_865",
        "l2_flags",
    ]
    assert list(df.attrs["rrs_columns"]) == ["Rrs_400.25", "Rrs_500.5"]
    assert list(df.attrs["rrs_unc_columns"]) == [
        "Rrs_unc_400.25",
        "Rrs_unc_500.5",
    ]
    assert df["pixel"].tolist() == [1, 2]
    np.testing.assert_allclose(df["latitude"], [30.1, 30.2])
    assert df["Rrs_400.25"].tolist() == [2.0, 4.0]
    assert str(df["datetime"].dtype) == "datetime64[ns, UTC]"
    assert df["l2_flags"].dtype == np.dtype("uint32")
    assert df.attrs["l2_flag_masks"]["LAND"] == 2
    assert df.attrs["product_name"] == "synthetic_pace_l2.nc"
    assert df.attrs["processing_version"] == "3.2"
    assert df.attrs["units"]["Rrs_400.25"] == "sr^-1"


def test_apply_l2_quality_flags_excludes_bailey_werdell_flags(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pace_l2.nc"
    _write_pace_l2(path)
    df = read_pace_l2(path)

    filtered_df = apply_l2_quality_flags(df)

    assert BAILEY_WERDELL_2006_EXCLUDED_L2_FLAGS == (
        "ATMFAIL",
        "LAND",
        "HIGLINT",
        "HILT",
        "STRAYLIGHT",
        "CLDICE",
        "LOWLW",
    )
    assert list(
        zip(filtered_df["scan_line"], filtered_df["pixel"], strict=True)
    ) == [
        (0, 0),
        (0, 2),
    ]
    assert filtered_df.attrs["excluded_l2_flags"] == (
        BAILEY_WERDELL_2006_EXCLUDED_L2_FLAGS
    )
    assert len(df) == 6


def test_apply_l2_quality_flags_accepts_custom_flags(tmp_path: Path) -> None:
    path = tmp_path / "pace_l2.nc"
    _write_pace_l2(path)
    df = read_pace_l2(path)

    filtered_df = apply_l2_quality_flags(df, excluded_flags=("HISATZEN",))

    assert list(
        zip(filtered_df["scan_line"], filtered_df["pixel"], strict=True)
    ) == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (1, 2),
    ]


def _write_pace_l2(path: Path) -> None:
    xr.Dataset(
        attrs={
            "product_name": "synthetic_pace_l2.nc",
            "processing_version": "3.2",
        }
    ).to_netcdf(path, engine="netcdf4")

    wavelengths = np.array([400.25, 500.5], dtype=np.float32)
    rrs = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    geophysical = xr.Dataset(
        data_vars={
            "Rrs": (
                ("number_of_lines", "pixels_per_line", "wavelength"),
                rrs,
                {"units": "sr^-1"},
            ),
            "Rrs_unc": (
                ("number_of_lines", "pixels_per_line", "wavelength"),
                rrs / 10,
                {"units": "sr^-1"},
            ),
            "aot_865": (
                ("number_of_lines", "pixels_per_line"),
                np.full((2, 3), 0.1, dtype=np.float32),
            ),
            "l2_flags": (
                ("number_of_lines", "pixels_per_line"),
                np.array(
                    [
                        [0, 2, 32],
                        [8, 16, 512],
                    ],
                    dtype=np.int32,
                ),
                {
                    "flag_masks": np.array(
                        [1, 2, 8, 16, 32, 256, 512, 16384],
                        dtype=np.int32,
                    ),
                    "flag_meanings": (
                        "ATMFAIL LAND HIGLINT HILT HISATZEN "
                        "STRAYLIGHT CLDICE LOWLW"
                    ),
                },
            ),
        },
        coords={"wavelength": wavelengths},
    )
    geophysical.to_netcdf(
        path,
        group="geophysical_data",
        mode="a",
        engine="netcdf4",
    )

    navigation = xr.Dataset(
        data_vars={
            "latitude": (
                ("number_of_lines", "pixels_per_line"),
                np.array(
                    [
                        [30.0, 30.1, 30.2],
                        [31.0, 31.1, 31.2],
                    ],
                    dtype=np.float32,
                ),
                {"units": "degrees_north"},
            ),
            "longitude": (
                ("number_of_lines", "pixels_per_line"),
                np.array(
                    [
                        [-70.0, -69.9, -69.8],
                        [-69.0, -68.9, -68.8],
                    ],
                    dtype=np.float32,
                ),
                {"units": "degrees_east"},
            ),
        }
    )
    navigation.to_netcdf(
        path,
        group="navigation_data",
        mode="a",
        engine="netcdf4",
    )

    scan_lines = xr.Dataset(
        data_vars={
            "time": (
                "number_of_lines",
                pd.to_datetime(
                    [
                        "2025-06-15T15:28:08Z",
                        "2025-06-15T15:28:09Z",
                    ]
                ).tz_convert(None),
            )
        }
    )
    scan_lines["time"].encoding["units"] = "seconds since 1970-01-01"
    scan_lines.to_netcdf(
        path,
        group="scan_line_attributes",
        mode="a",
        engine="netcdf4",
    )
