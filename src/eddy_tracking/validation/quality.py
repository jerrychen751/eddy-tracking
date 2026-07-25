from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


# Bailey and Werdell (2006), section 2.2.4, excludes these conditions
# from radiometric validation pixels.
BAILEY_WERDELL_2006_EXCLUDED_L2_FLAGS: tuple[str, ...] = (
    "ATMFAIL",
    "LAND",
    "HIGLINT",
    "HILT",
    "STRAYLIGHT",
    "CLDICE",
    "LOWLW",
)


def apply_l2_quality_flags(
    df: pd.DataFrame,
    excluded_flags: Sequence[str] = BAILEY_WERDELL_2006_EXCLUDED_L2_FLAGS,
) -> pd.DataFrame:
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
    filtered_df = df.loc[keep].copy()
    filtered_df.attrs = df.attrs.copy()
    filtered_df.attrs["excluded_l2_flags"] = normalized_flags
    return filtered_df
