"""
Pre-flight sanity filter for S and F. Ports phytoclass::Matrix_checks.

Call this before simulated_annealing when S and Fmat carry the standard phytoclass names, for example class "Syn" and pigment "Zea". The filter is driven by those names.
"""

import pandas as pd


def matrix_checks(
    S: pd.DataFrame,
    Fmat: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter (S, Fmat) to drop sparse pigments and unfittable groups. Mirrors R's Matrix_checks.

    The steps run in this order, and step 6 depends on what step 5 removed:
        1. Binarize Fmat (every entry above 0 becomes 1) for the presence checks.
        2. Drop F rows with row sum <= 1, which are groups holding Tchla alone.
        3. Drop F columns with column sum 0, which are pigments no group uses.
        4. Keep only the column names that S and F share.
        5. Drop pigment columns present in 1% or fewer of the samples.
        6. Drop each F row whose diagnostic pigment left S, for example row "Syn" once column "Zea" is gone.
        7. Drop the F columns that step 6 left all-zero, and the same columns of S.

    Returns:
        (S_new, F_new) with matched column names. F_new carries Fmat's original values, not the binarized ones.

    Raises:
        TypeError: S or Fmat is not a DataFrame.
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

    # (Fmat row label, S column label) pairs, for example ("Syn", "Zea"): the group cannot be fitted once its diagnostic pigment leaves S.
    diagnostic_pigments = [
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

    rows_to_drop = []
    for cls, pig in diagnostic_pigments:
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
