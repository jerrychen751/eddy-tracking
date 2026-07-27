from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pandas as pd
import xarray as xr


_DimensionIndexer: TypeAlias = slice | Sequence[int] | np.ndarray


def read_pace_l2(
    path: Path | str,
    *,
    line_indexer: _DimensionIndexer | None = None,
    pixel_indexer: _DimensionIndexer | None = None,
) -> pd.DataFrame:
    """
    Read selected PACE OCI Level-2 pixels into one row per pixel.

    Returned columns:
        scan_line (int64): Zero-based row index along the travel direction.
        pixel (int64): Zero-based column index across the satellite swath.
        datetime (datetime64[ns, UTC]): Scan-line observation time.
        latitude (float32): Pixel latitude in degrees north.
        longitude (float32): Pixel longitude in degrees east.
        aot_865 (float32): Dimensionless aerosol optical thickness at 865 nm.
        l2_flags (uint32): Level-2 quality bit field.
        Rrs_<wavelength_nm> (float32): Remote sensing reflectance in sr^-1.
        Rrs_unc_<wavelength_nm> (float32): Rrs uncertainty in sr^-1.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    with xr.open_dataset(
        path,
        group="geophysical_data",
        engine="netcdf4",
    ) as geophysical:
        line_positions = _resolve_positions(
            geophysical.sizes["number_of_lines"],
            line_indexer,
            "line_indexer",
        )
        pixel_positions = _resolve_positions(
            geophysical.sizes["pixels_per_line"],
            pixel_indexer,
            "pixel_indexer",
        )
        # Load selected pixels because both full spectral arrays need several gigabytes.
        selected_geophysical = geophysical[
            ["Rrs", "Rrs_unc", "aot_865", "l2_flags"]
        ].isel(
            number_of_lines=line_positions,
            pixels_per_line=pixel_positions,
        ).load()
        wavelengths_nm = geophysical["wavelength"].to_numpy().astype(
            float,
            copy=False,
        )
        rrs_units = geophysical["Rrs"].attrs["units"]
        rrs_unc_units = geophysical["Rrs_unc"].attrs["units"]
        flag_masks = _read_l2_flag_masks(geophysical["l2_flags"])

    with xr.open_dataset(
        path,
        group="navigation_data",
        engine="netcdf4",
    ) as navigation:
        selected_navigation = navigation[["latitude", "longitude"]].isel(
            number_of_lines=line_positions,
            pixels_per_line=pixel_positions,
        ).load()
        latitude_units = navigation["latitude"].attrs["units"]
        longitude_units = navigation["longitude"].attrs["units"]

    with xr.open_dataset(
        path,
        group="scan_line_attributes",
        engine="netcdf4",
        decode_timedelta=False,
    ) as scan_lines:
        scan_times = scan_lines["time"].isel(
            number_of_lines=line_positions
        ).load()

    with xr.open_dataset(path, engine="netcdf4") as root:
        product_name = root.attrs.get("product_name", path.name)
        processing_version = root.attrs.get("processing_version")

    line_count = len(line_positions)
    pixel_count = len(pixel_positions)
    row_count = line_count * pixel_count
    rrs_columns = _spectral_column_names("Rrs", wavelengths_nm)
    rrs_unc_columns = _spectral_column_names("Rrs_unc", wavelengths_nm)

    # Keep one pixel per row. Spectral values stay wide for SDP.
    df = pd.DataFrame(
        {
            "scan_line": np.repeat(line_positions, pixel_count),
            "pixel": np.tile(pixel_positions, line_count),
            "datetime": pd.to_datetime(
                np.repeat(scan_times.to_numpy(), pixel_count),
                utc=True,
            ),
            "latitude": selected_navigation["latitude"].to_numpy().reshape(-1),
            "longitude": selected_navigation["longitude"].to_numpy().reshape(-1),
            "aot_865": selected_geophysical["aot_865"].to_numpy().reshape(-1),
            "l2_flags": selected_geophysical["l2_flags"]
            .to_numpy()
            .astype(np.uint32, copy=False)
            .reshape(-1),
        }
    )
    rrs = selected_geophysical["Rrs"].to_numpy().reshape(
        row_count,
        len(wavelengths_nm),
    )
    rrs_unc = selected_geophysical["Rrs_unc"].to_numpy().reshape(
        row_count,
        len(wavelengths_nm),
    )
    df = pd.concat(
        [
            df,
            pd.DataFrame(rrs, columns=rrs_columns),
            pd.DataFrame(rrs_unc, columns=rrs_unc_columns),
        ],
        axis="columns",
    )

    units = {
        "latitude": latitude_units,
        "longitude": longitude_units,
        "aot_865": "1",
        "l2_flags": "bit field",
    }
    units.update(dict.fromkeys(rrs_columns, rrs_units))
    units.update(dict.fromkeys(rrs_unc_columns, rrs_unc_units))
    df.attrs["units"] = units
    df.attrs["product_name"] = product_name
    df.attrs["processing_version"] = processing_version
    df.attrs["source_file"] = str(path)
    df.attrs["wavelengths_nm"] = tuple(wavelengths_nm)
    df.attrs["rrs_columns"] = tuple(rrs_columns)
    df.attrs["rrs_unc_columns"] = tuple(rrs_unc_columns)
    df.attrs["l2_flag_masks"] = flag_masks
    return df


def _resolve_positions(
    size: int,
    indexer: _DimensionIndexer | None,
    indexer_name: str,
) -> np.ndarray:
    positions = np.arange(size, dtype=np.int64)
    if indexer is None:
        return positions

    normalized_indexer = (
        np.asarray(indexer) if isinstance(indexer, Sequence) else indexer
    )
    try:
        selected = np.atleast_1d(positions[normalized_indexer])
    except (IndexError, TypeError) as exc:
        raise IndexError(f"Invalid {indexer_name}: {indexer!r}") from exc
    if selected.ndim != 1:
        raise IndexError(f"{indexer_name} must select one dimension")
    return selected


def _spectral_column_names(
    product_name: str,
    wavelengths_nm: np.ndarray,
) -> list[str]:
    columns = [
        f"{product_name}_{float(wavelength):.3f}".rstrip("0").rstrip(".")
        for wavelength in wavelengths_nm
    ]
    if len(columns) != len(set(columns)):
        raise ValueError(f"{product_name} wavelengths do not make unique columns")
    return columns


def _read_l2_flag_masks(flags: xr.DataArray) -> dict[str, int]:
    names = str(flags.attrs["flag_meanings"]).upper().split()
    signed_masks = np.asarray(flags.attrs["flag_masks"], dtype=np.int32)
    if len(names) != len(signed_masks):
        raise ValueError("l2_flags flag names and masks have different lengths")

    unsigned_masks = signed_masks.view(np.uint32)
    return {
        name: int(mask)
        for name, mask in zip(names, unsigned_masks, strict=True)
        if name != "SPARE"
    }


def read_multiple_pace_l2(
    paths: Sequence[Path | str],
    *,
    line_indexer: _DimensionIndexer | None = None,
    pixel_indexer: _DimensionIndexer | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    source_files: list[str] = []
    product_names: list[str] = []
    processing_versions: list[str | None] = []
    expected_columns: pd.Index | None = None
    shared_attrs: dict[str, object] = {}
    shared_attr_names = (
        "units",
        "wavelengths_nm",
        "rrs_columns",
        "rrs_unc_columns",
        "l2_flag_masks",
    )

    for path in paths:
        frame = read_pace_l2(
            path,
            line_indexer=line_indexer,
            pixel_indexer=pixel_indexer,
        )
        if expected_columns is None:
            expected_columns = frame.columns
            shared_attrs = {
                name: frame.attrs[name] for name in shared_attr_names
            }
        else:
            if not frame.columns.equals(expected_columns):
                raise ValueError(
                    f"{path} has data columns that do not match the first file"
                )
            for name in shared_attr_names:
                if frame.attrs[name] != shared_attrs[name]:
                    raise ValueError(
                        f"{path} has {name} metadata that does not match "
                        "the first file"
                    )

        source_file = frame.attrs["source_file"]
        source_files.append(source_file)
        product_names.append(frame.attrs["product_name"])
        processing_versions.append(frame.attrs["processing_version"])
        frame.insert(0, "source_file", source_file)
        frames.append(frame)

    if not frames:
        raise ValueError("paths must contain at least one PACE Level-2 file")

    combined = pd.concat(frames, ignore_index=True)
    combined.attrs = shared_attrs
    combined.attrs["source_files"] = tuple(source_files)
    combined.attrs["product_names"] = tuple(product_names)
    combined.attrs["processing_versions"] = tuple(processing_versions)
    return combined
