from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from eddy_tracking.preprocess.pace import read_multiple_pace_l2, read_pace_l2
from eddy_tracking.validation.quality import (
    EXCLUDED_L2_FLAGS,
    GEOMETRY_L2_FLAGS,
    QUALITY_L2_FLAGS,
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


def test_read_multiple_pace_l2_combines_files_in_input_order(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first_pace_l2.nc"
    second_path = tmp_path / "second_pace_l2.nc"
    _write_pace_l2(first_path)
    _write_pace_l2(second_path)

    df = read_multiple_pace_l2(
        [first_path, second_path],
        line_indexer=slice(0, 1),
        pixel_indexer=(2,),
    )

    assert df.shape == (2, 12)
    assert df["source_file"].tolist() == [
        str(first_path),
        str(second_path),
    ]
    assert df["scan_line"].tolist() == [0, 0]
    assert df["pixel"].tolist() == [2, 2]
    assert df["Rrs_400.25"].tolist() == [4.0, 4.0]
    assert df.index.tolist() == [0, 1]
    assert df.attrs["source_files"] == (
        str(first_path),
        str(second_path),
    )
    assert "source_file" not in df.attrs
    assert df.attrs["l2_flag_masks"]["LAND"] == 2


def test_read_multiple_pace_l2_rejects_empty_paths() -> None:
    with pytest.raises(
        ValueError,
        match="paths must contain at least one PACE Level-2 file",
    ):
        read_multiple_pace_l2([])


def test_apply_l2_quality_flags_masks_facts_and_preserves_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "pace_l2.nc"
    _write_pace_l2(path)
    df = read_multiple_pace_l2([path])

    filtered_df = apply_l2_quality_flags(df)

    assert QUALITY_L2_FLAGS == (
        "ATMFAIL",
        "LAND",
        "HIGLINT",
        "HILT",
        "STRAYLIGHT",
        "CLDICE",
        "LOWLW",
    )
    assert GEOMETRY_L2_FLAGS == (
        "HISATZEN",
        "HISOLZEN",
    )
    assert EXCLUDED_L2_FLAGS == (
        *QUALITY_L2_FLAGS,
        *GEOMETRY_L2_FLAGS,
    )
    fact_columns = [
        "aot_865",
        *df.attrs["rrs_columns"],
        *df.attrs["rrs_unc_columns"],
    ]
    excluded_rows = [False, True, True, True, False, True]
    assert filtered_df.loc[excluded_rows, fact_columns].isna().all().all()
    pd.testing.assert_frame_equal(
        filtered_df.loc[[0, 4], fact_columns],
        df.loc[[0, 4], fact_columns],
    )
    pd.testing.assert_frame_equal(
        filtered_df.drop(columns=fact_columns),
        df.drop(columns=fact_columns),
    )
    assert filtered_df.attrs["excluded_l2_flags"] == (
        EXCLUDED_L2_FLAGS
    )
    assert filtered_df.index.equals(df.index)
    assert len(filtered_df) == len(df) == 6
    assert not df[fact_columns].isna().any().any()
    assert capsys.readouterr().out == (
        "L2 quality flags: masked fact columns in 4 of 6 rows (66.7%).\n"
    )


def test_apply_l2_quality_flags_drops_rows_when_to_nan_is_false(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "pace_l2.nc"
    _write_pace_l2(path)
    df = read_multiple_pace_l2([path])

    filtered_df = apply_l2_quality_flags(
        df,
        excluded_flags=("HISATZEN",),
        to_nan=False,
    )

    assert list(
        zip(filtered_df["scan_line"], filtered_df["pixel"], strict=True)
    ) == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    assert filtered_df.index.tolist() == [0, 1, 2, 3, 4]
    assert capsys.readouterr().out == (
        "L2 quality flags: 6 rows before, 5 rows after; "
        "filtered 1 of 6 rows (16.7%).\n"
    )


def test_apply_l2_quality_flags_accepts_empty_dataframe(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "pace_l2.nc"
    _write_pace_l2(path)
    df = read_multiple_pace_l2([path]).iloc[0:0]

    filtered_df = apply_l2_quality_flags(df)

    assert filtered_df.empty
    assert capsys.readouterr().out == (
        "L2 quality flags: masked fact columns in 0 of 0 rows (0.0%).\n"
    )


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
                        [4096, 0, 512],
                    ],
                    dtype=np.int32,
                ),
                {
                    "flag_masks": np.array(
                        [1, 2, 8, 16, 32, 256, 512, 4096, 16384],
                        dtype=np.int32,
                    ),
                    "flag_meanings": (
                        "ATMFAIL LAND HIGLINT HILT HISATZEN "
                        "STRAYLIGHT CLDICE HISOLZEN LOWLW"
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
