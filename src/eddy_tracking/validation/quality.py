from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


# Bailey and Werdell (2006), section 2.2.4, excludes these conditions from radiometric validation pixels.
QUALITY_L2_FLAGS: tuple[str, ...] = (
    "ATMFAIL",
    "LAND",
    "HIGLINT",
    "HILT",
    "STRAYLIGHT",
    "CLDICE",
    "LOWLW",
)
GEOMETRY_L2_FLAGS: tuple[str, ...] = (
    "HISATZEN",
    "HISOLZEN",
)
EXCLUDED_L2_FLAGS: tuple[str, ...] = (
    *QUALITY_L2_FLAGS,
    *GEOMETRY_L2_FLAGS,
)


def apply_l2_quality_flags(
    df: pd.DataFrame,
    excluded_flags: Sequence[str] = EXCLUDED_L2_FLAGS,
    to_nan: bool = True,
) -> pd.DataFrame:
    """
    Apply Level-2 quality flags to a PACE observation table.

    The input must have the schema from `read_multiple_pace_l2`.
    It must contain `source_file`, `scan_line`, `pixel`, `datetime`, `latitude`, `longitude`, `aot_865`, `l2_flags`, `Rrs_*`, and `Rrs_unc_*` columns.
    It must also contain `l2_flag_masks`, `rrs_columns`, and `rrs_unc_columns` in `DataFrame.attrs`.

    If `to_nan` is true, this function keeps all rows.
    It replaces `aot_865`, `Rrs_*`, and `Rrs_unc_*` values with NaN in rows with excluded flags.
    It preserves all metadata columns.

    If `to_nan` is false, this function removes rows with excluded flags and resets the result index.

    This function does not change the input.
    It prints a row count summary and records the applied flags in `attrs["excluded_l2_flags"]`.
    """
    if "l2_flags" not in df.columns:
        raise KeyError("DataFrame must contain an 'l2_flags' column")
    if df["l2_flags"].isna().any():
        raise ValueError("l2_flags cannot contain missing values")

    flag_masks = df.attrs.get("l2_flag_masks")
    if not isinstance(flag_masks, Mapping):
        raise ValueError("DataFrame attrs must contain an l2_flag_masks mapping")

    normalized_flags = tuple(flag.upper() for flag in excluded_flags)
    missing_flags = [
        flag for flag in normalized_flags if flag not in flag_masks
    ]
    if missing_flags:
        raise ValueError(
            f"l2_flag_masks does not define these flags: {missing_flags}"
        )

    combined_mask = 0
    for flag in normalized_flags:
        combined_mask |= int(flag_masks[flag])

    flag_values = df["l2_flags"].to_numpy(dtype=np.uint32, copy=False)
    keep = (flag_values & np.uint32(combined_mask)) == 0
    total_count = len(df)
    retained_count = int(keep.sum())
    filtered_count = total_count - retained_count
    filtered_percent = (
        100 * filtered_count / total_count if total_count else 0.0
    )
    if to_nan:
        fact_columns = (
            "aot_865",
            *df.attrs["rrs_columns"],
            *df.attrs["rrs_unc_columns"],
        )
        filtered_df = df.copy()
        filtered_df.loc[~keep, list(fact_columns)] = np.nan
        print(
            "quality_flag_action: mask\n"
            f"rows_masked: {filtered_count:,}\n"
            f"total_rows: {total_count:,}\n"
            f"filtered_percent: {filtered_percent:.1f}"
        )
    else:
        filtered_df = df.loc[keep].copy().reset_index(drop=True)
        print(
            "quality_flag_action: remove\n"
            f"rows_before: {total_count:,}\n"
            f"rows_after: {retained_count:,}\n"
            f"rows_filtered: {filtered_count:,}\n"
            f"filtered_percent: {filtered_percent:.1f}"
        )

    filtered_df.attrs = df.attrs.copy()
    filtered_df.attrs["excluded_l2_flags"] = normalized_flags
    return filtered_df
