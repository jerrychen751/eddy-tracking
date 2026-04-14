"""
Low-level helpers for the phytoclass R-port.

Each function here mirrors a function in phytoclass::util.R. Names and
semantics are preserved for traceability against the R source.
"""

import numpy as np
import pandas as pd


def normalise_S(S: np.ndarray) -> np.ndarray:
    """
    Row-normalize S to unit row sums. Mirrors R's Normalise_S.

    Every sample (row) is rescaled so that its pigment concentrations sum to 1.
    The caller is expected to save S[:, -1] (Tchla) before normalization if
    absolute-scale class abundances are needed later.
    """
    row_sums = S.sum(axis=1, keepdims=True)
    return S / row_sums


def normalise_F(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize F so each row is a fraction-of-pigment-mix. Mirrors R's Normalise_F.

    Divides each entry by its row's Tchla value, then divides each row by its
    new row sum. Returns (normalized F, row-sum vector) so callers can undo
    the scaling.
    """
    F = np.asarray(F, dtype=float)
    tchla = F[:, -1:]
    F_scaled = F / tchla
    row_sums = F_scaled.sum(axis=1)
    F_norm = F_scaled / row_sums[:, np.newaxis]
    return F_norm, row_sums


def bounded_weights(S: np.ndarray, upper_bound: float = 30.0) -> np.ndarray:
    """
    Per-column weights: 1/col_mean, capped at upper_bound, with Tchla forced to 1.

    Mirrors R's Bounded_weights. The last column (Tchla) always gets weight 1
    regardless of its mean. Zero-mean columns would produce infinite weights,
    so they get upper_bound instead.
    """
    col_means = np.mean(S, axis=0)
    with np.errstate(divide="ignore"):
        weights = 1.0 / col_means
    weights = np.minimum(weights, upper_bound)
    weights[-1] = 1.0
    return weights


def vectorise(F: np.ndarray) -> np.ndarray:
    """
    Extract non-zero elements of F in column-major order. Mirrors R's vectorise.

    R's `Fmat[Fmat > 0]` uses column-major storage, so iteration order is
    "column 0 rows, column 1 rows, ...". numpy stores row-major by default,
    so we flatten with order='F' to match R.
    """
    flat = F.flatten(order="F")
    return flat[flat > 0]


def apply_weights(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Column-scale X by weights. Mirrors R's Weight_error which does S %*% diag(cm).

    Equivalent to `X * weights[np.newaxis, :]` but named for parity with R.
    """
    return X * weights[np.newaxis, :]


def wrangling(
    F: np.ndarray,
    min_vals: np.ndarray,
    max_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build flat min/max/current-value vectors aligned to F's non-zero entries.

    Mirrors R's Wrangling. Given F (classes x pigments, last column = Tchla),
    and vectors of per-entry min/max values ordered in column-major over F's
    non-Tchla columns:

    1. Drop the Tchla column from F to get Fd (classes x pigments-1).
    2. Build Fmin, Fmax matrices with the same shape as Fd: each non-zero
       entry gets its corresponding min_vals / max_vals value, zero entries
       stay zero.
    3. Multiply Fmin and Fmax column-wise by the Tchla (chlv) column of F,
       so bounds are expressed in absolute ratios rather than per-unit-Tchla.
    4. Vectorise Fmin, Fmax, and Fd (column-major, non-zero only).

    Returns (minF_vec, maxF_vec, SE_vec, chlv) where chlv is the Tchla column
    of F.
    """
    F = np.asarray(F, dtype=float)
    chlv = F[:, -1].copy()
    Fd = F[:, :-1].copy()

    Fmin = np.zeros_like(Fd)
    Fmax = np.zeros_like(Fd)
    mask_column_major = Fd.flatten(order="F") > 0
    Fmin_flat = Fmin.flatten(order="F")
    Fmax_flat = Fmax.flatten(order="F")
    Fmin_flat[mask_column_major] = min_vals
    Fmax_flat[mask_column_major] = max_vals
    Fmin = Fmin_flat.reshape(Fd.shape, order="F")
    Fmax = Fmax_flat.reshape(Fd.shape, order="F")

    Fmin = Fmin * chlv[:, np.newaxis]
    Fmax = Fmax * chlv[:, np.newaxis]

    minF_vec = vectorise(Fmin)
    maxF_vec = vectorise(Fmax)
    SE_vec = vectorise(Fd)

    return minF_vec, maxF_vec, SE_vec, chlv


def default_min_max(
    min_max_df: pd.DataFrame,
    F: np.ndarray,
    class_names: list[str],
    pigment_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Look up (class, pigment) pairs in min_max_df for every non-zero entry of F.

    Mirrors R's Default_min_max. Iterates F in column-major order (to match
    R's which(F > 0, arr.ind=TRUE)), finds each entry's class/pigment names,
    and matches them against min_max_df's Class and Pig_Abbrev columns.

    Args:
        min_max_df: DataFrame with columns "Class", "Pig_Abbrev", "min", "max".
        F: Binary or scored matrix excluding the Tchla column.
        class_names: Row names of F.
        pigment_names: Column names of F.

    Returns:
        (min_vec, max_vec) — flat vectors ordered in column-major over F.

    Raises:
        ValueError if any (class, pigment) pair is missing from min_max_df.
    """
    F = np.asarray(F)
    min_vec = []
    max_vec = []
    missing = []

    for col_idx in range(F.shape[1]):
        for row_idx in range(F.shape[0]):
            if F[row_idx, col_idx] <= 0:
                continue
            cls = class_names[row_idx]
            pig = pigment_names[col_idx]
            match = min_max_df[
                (min_max_df["Class"] == cls) & (min_max_df["Pig_Abbrev"] == pig)
            ]
            if len(match) == 0:
                missing.append((cls, pig))
                min_vec.append(np.nan)
                max_vec.append(np.nan)
            else:
                min_vec.append(float(match["min"].iloc[0]))
                max_vec.append(float(match["max"].iloc[0]))

    if missing:
        raise ValueError(
            "min_max table is missing (class, pigment) pairs: "
            + ", ".join(f"{c}/{p}" for c, p in missing)
        )

    return np.array(min_vec), np.array(max_vec)
