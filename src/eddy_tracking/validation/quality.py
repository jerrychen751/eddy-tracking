from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from eddy_tracking.packages.sdp.prediction import SDP_PIGMENT_COLUMNS


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

    The input must have the schema from `read_multiple_pace_l2`. It must contain `source_file`, `scan_line`, `pixel`, `datetime`, `latitude`, `longitude`, `aot_865`, `l2_flags`, `Rrs_*`, and `Rrs_unc_*` columns. It must also contain `l2_flag_masks`, `rrs_columns`, and `rrs_unc_columns` in `DataFrame.attrs`.

    If `to_nan` is true, this function keeps all rows. It replaces `aot_865`, `Rrs_*`, and `Rrs_unc_*` values with NaN in rows with excluded flags. It preserves all metadata columns.

    If `to_nan` is false, this function removes rows with excluded flags and resets the result index.

    This function does not change the input. It prints a row count summary and records the applied flags in `attrs["excluded_l2_flags"]`.
    """
    if "l2_flags" not in df.columns:
        raise KeyError("DataFrame must contain an 'l2_flags' column")
    if df["l2_flags"].isna().any():  # pyright: ignore[reportGeneralTypeIssues]
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


def keep_valid_sdp_rows(
    pigment_table: pd.DataFrame,
    max_tchla_zero_frac: float = 0.5,
) -> pd.DataFrame:
    # A gold row already has collocate_pace.min_coverage of its interior and at least 10 interior pixels, so only the retrieval itself is judged here
    mostly_zero_tchla = pigment_table["tchla_zero_frac"].ge(max_tchla_zero_frac)
    # Total chlorophyll-a contains divinyl chlorophyll-a, so an interior mean of DV_chla at or above that of Tchla cannot be real. DV_chla is the only SDP pigment that is part of Tchla: the other 11 are separate accessory pigments whose sum normally sits just above Tchla, so no sum of pigments has Tchla as a ceiling
    dv_chla_not_below_tchla = pigment_table["eddy_mean_DV_chla"].ge(
        pigment_table["eddy_mean_Tchla"]
    )
    valid = ~mostly_zero_tchla & ~dv_chla_not_below_tchla
    print(
        "sdp_quality_action: remove\n"
        f"rows_before: {len(pigment_table):,}\n"
        f"rows_after: {int(valid.sum()):,}\n"
        f"rows_mostly_zero_tchla: {int(mostly_zero_tchla.sum()):,}\n"
        f"rows_dv_chla_not_below_tchla: {int(dv_chla_not_below_tchla.sum()):,}"
    )
    return pigment_table.loc[valid].reset_index(drop=True)


def clip_sdp_pixels(pixels: pd.DataFrame) -> pd.DataFrame:
    # Pixel files keep negative SDP predictions so that a mean over pixels can cancel the noise; clip here only where a single pixel is read as a concentration, and clip a mean after averaging instead
    pigment_columns = [column for column in SDP_PIGMENT_COLUMNS if column in pixels]
    return pixels.assign(**{column: pixels[column].clip(lower=0) for column in pigment_columns})


def mask_plankton_groups_above_chl(
    plankton_table: pd.DataFrame,
    groups: Sequence[str],
) -> pd.DataFrame:
    # Each Copernicus group is the chlorophyll-a of one part of the community, so an interior mean above that of total CHL cannot be real
    masked = plankton_table.copy()
    above_chl = masked[list(groups)].gt(masked["CHL"], axis=0)
    masked[list(groups)] = masked[list(groups)].mask(above_chl)
    print(
        "plankton_quality_action: mask\n"
        f"rows: {len(masked):,}\n"
        + "\n".join(
            f"{group}_above_chl: {int(above_chl[group].sum()):,}" for group in groups
        )
    )
    return masked
