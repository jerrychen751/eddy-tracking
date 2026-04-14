"""
Pre-flight sanity filter for S and F. Ports phytoclass::Matrix_checks.

Drops pigment columns that are sparse or zero in S, phytoplankton groups with
only one pigment, and phytoplankton groups whose diagnostic pigment has been
stripped from S. Should be called before simulated_annealing when using
standard class/pigment names.
"""

import numpy as np
import pandas as pd


DIAGNOSTIC_PIGMENTS: list[tuple[str, str]] = [
    ("Chlorophytes", "Chl_b"),
    ("Prasinophytes", "Chl_b"),
    ("Prasinophytes", "Pra"),
    ("Dinoflagellates-1", "Per"),
    ("Diatoms-1", "Chl_c1"),
    ("Diatoms-2", "Fuco"),
    ("Syn", "Zea"),
    ("Cryptophytes", "Allo"),
    ("Haptophytes-H", "X19hex"),
    ("Haptophytes-L", "X19hex"),
    ("Diatoms-1", "Fuco"),
    ("Pelagophytes", "X19but"),
    ("Prasinophytes", "Chl_b"),
]


def matrix_checks(
    S: pd.DataFrame,
    Fmat: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter (S, Fmat) to drop sparse pigments and unfittable groups.

    Mirrors R's Matrix_checks:
        1. Binarize Fmat (set all > 0 to 1) for the pigment-presence check.
        2. Drop F rows with row sum <= 1 (groups with only Tchla).
        3. Drop F columns with column sum == 0 (pigments no group uses).
        4. Intersect S and F column names, keeping only the overlap.
        5. Drop pigment columns that appear in < 1% of samples (colSum(S > 0) / n_samples).
        6. Drop F rows whose diagnostic pigment (per DIAGNOSTIC_PIGMENTS) is
           no longer in S.
        7. After the group drop, drop any F columns that became all-zero.

    Returns:
        (S_new, F_new) — new DataFrames with matched columns.
    """
    if not isinstance(S, pd.DataFrame):
        raise TypeError("Matrix_checks expects DataFrames so column names drive the filter.")
    if not isinstance(Fmat, pd.DataFrame):
        raise TypeError("Matrix_checks expects DataFrames so column names drive the filter.")

    f_bin = (Fmat > 0).astype(int)

    row_sums = f_bin.sum(axis=1)
    f_bin = f_bin.loc[row_sums > 1]

    col_sums = f_bin.sum(axis=0)
    f_bin = f_bin.loc[:, col_sums > 0]

    common_cols = [c for c in f_bin.columns if c in S.columns]
    S_new = S[common_cols].copy()
    f_bin = f_bin[common_cols]

    n_samples = len(S_new)
    if n_samples > 0:
        present_fraction = (S_new != 0).sum(axis=0) / n_samples
        keep_cols = present_fraction[present_fraction > 0.01].index.tolist()
        S_new = S_new[keep_cols]
        f_bin = f_bin[keep_cols]

    rows_to_drop = []
    for cls, pig in DIAGNOSTIC_PIGMENTS:
        if cls in f_bin.index and pig not in S_new.columns:
            rows_to_drop.append(cls)
    if rows_to_drop:
        f_bin = f_bin.drop(index=rows_to_drop)

    zero_cols = f_bin.columns[f_bin.sum(axis=0) == 0].tolist()
    if zero_cols:
        f_bin = f_bin.drop(columns=zero_cols)
        S_new = S_new.drop(columns=zero_cols)

    F_new = Fmat.loc[f_bin.index, f_bin.columns].copy()

    return S_new, F_new
