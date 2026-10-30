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
    max_accessory_ratio: float = 1.64,
) -> pd.DataFrame:
    # A gold row already has collocate_pace.min_coverage of its interior and at least 10 interior pixels, so only the retrieval itself is judged here
    mostly_zero_tchla = pigment_table["tchla_zero_frac"].ge(max_tchla_zero_frac)
    # Total chlorophyll-a contains divinyl chlorophyll-a, so an interior mean of DV_chla at or above that of Tchla cannot be real
    dv_chla_not_below_tchla = pigment_table["eddy_mean_DV_chla"].ge(pigment_table["eddy_mean_Tchla"])
    # The 11 SDP pigments that are not part of Tchla. 1.64 is the largest accessory sum / Tchla among the 145 HPLC samples SDP was trained on (Kramer et al. 2021), so a larger value lies outside anything the model learned
    accessory_pigments = ("Zea", "ButFuco", "HexFuco", "Allo", "MV_chlb", "Neo", "Viola", "Fuco", "Chlc12", "Chlc3", "Perid")
    accessory_ratio = pigment_table[[f"eddy_mean_{pigment}" for pigment in accessory_pigments]].sum(axis=1).div(pigment_table["eddy_mean_Tchla"])
    accessory_above_training = accessory_ratio.gt(max_accessory_ratio)
    valid = ~mostly_zero_tchla & ~dv_chla_not_below_tchla & ~accessory_above_training
    print(
        "sdp_quality_action: remove\n"
        f"rows_before: {len(pigment_table):,}\n"
        f"rows_after: {int(valid.to_numpy().sum()):,}\n"
        f"rows_mostly_zero_tchla: {int(mostly_zero_tchla.to_numpy().sum()):,}\n"
        f"rows_dv_chla_not_below_tchla: {int(dv_chla_not_below_tchla.to_numpy().sum()):,}\n"
        f"rows_accessory_above_{max_accessory_ratio:g}x_tchla: {int(accessory_above_training.to_numpy().sum()):,}"
    )
    return pigment_table.loc[valid].reset_index(drop=True)


def clip_sdp_pixels(pixels: pd.DataFrame) -> pd.DataFrame:
    # Pixel files keep negative SDP predictions so that a mean over pixels can cancel the noise; clip here only where a single pixel is read as a concentration, and clip a mean after averaging instead
    pigment_columns = [column for column in SDP_PIGMENT_COLUMNS if column in pixels]
    return pixels.assign(**{column: pixels[column].clip(lower=0) for column in pigment_columns})


# settings is the collocate_plankton block of the experiment config
def mask_invalid_plankton_groups(
    plankton_table: pd.DataFrame,
    settings: Mapping[str, float],
    max_partition_factor: float = 2.0,
) -> pd.DataFrame:
    table = plankton_table.copy()
    # The five Copernicus groups that partition CHL; the size classes are sums of them and Prochlorococcus is part of the prokaryotes
    partition_groups = ["DIATO", "DINO", "HAPTO", "GREEN", "PROKAR"]
    groups = [field for field in (*partition_groups, "PROCHLO", "MICRO", "NANO", "PICO") if field in table]
    # The groups are retrieved independently with no closure constraint (Xi et al. 2021), so their sum scatters around CHL; over the 10,413 Gulf Stream composites the robust two-sigma band of sum / CHL is 0.61 to 1.96, so a factor of 2 either way catches only composites whose partition has clearly failed. A failed composite loses every group value but keeps its CHL, and a composite missing one of the five groups is kept, since the sum cannot be checked
    partition_ratio = table[partition_groups].sum(axis=1, min_count=len(partition_groups)).div(table["CHL"])
    partition_failed = partition_ratio.gt(max_partition_factor) | partition_ratio.lt(1 / max_partition_factor)
    table.loc[partition_failed, groups] = np.nan
    # The rule the gold table applies to CHL, applied to each group's own valid pixels
    below_coverage = pd.DataFrame({
        group: table[f"{group}_n_pixels"].lt(settings["min_pixels"])
        | table[f"{group}_n_pixels"].div(table["n_pixels"]).lt(settings["min_coverage"])
        for group in groups
    })
    table[groups] = table[groups].mask(below_coverage)
    # Each group is the chlorophyll-a of one part of the community, so an interior mean above that of total CHL cannot be real
    above_chl = table[groups].gt(table["CHL"], axis=0)
    table[groups] = table[groups].mask(above_chl)
    # Prochlorococcus is part of the prokaryotes. Only the PROCHLO value goes; it fails this in over half the composites, so removing rows would empty the table
    prochlo_above_prokar = table["PROCHLO"].gt(table["PROKAR"]) if {"PROCHLO", "PROKAR"} <= set(table) else pd.Series(False, index=table.index)
    if "PROCHLO" in table:
        table["PROCHLO"] = table["PROCHLO"].mask(prochlo_above_prokar)
    print(
        "plankton_quality_action: mask\n"
        f"rows: {len(table):,}\n"
        f"rows_partition_outside_{max_partition_factor:g}x_chl: {int(partition_failed.to_numpy().sum()):,}\n"
        f"values_below_coverage: {int(below_coverage.to_numpy().sum()):,}\n"
        f"values_above_chl: {int(above_chl.to_numpy().sum()):,}\n"
        f"prochlo_values_above_prokar: {int(prochlo_above_prokar.to_numpy().sum()):,}"
    )
    return table
