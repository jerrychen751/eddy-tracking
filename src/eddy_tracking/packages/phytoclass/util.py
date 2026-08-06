"""
Low-level helpers for the phytoclass R-port.

Each function here mirrors a function in phytoclass::util.R. Names and semantics are preserved for traceability against the R source.
"""

import numpy as np
import pandas as pd


def normalise_S(S: np.ndarray) -> np.ndarray:
    """
    Row-normalize S to unit row sums. Mirrors R's Normalise_S.

    Save S[:, -1] (Tchla) before calling when absolute-scale class abundances are needed later.
    """
    row_sums = S.sum(axis=1, keepdims=True)
    return S / row_sums


def normalise_F(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize F so each row is a fraction-of-pigment-mix. Mirrors R's Normalise_F.

    Returns (F_norm, row_sums). F_norm rows sum to 1. row_sums holds the row sums of F / Tchla, so F_norm * row_sums restores F / Tchla, not F.
    """
    F = np.asarray(F, dtype=float)
    tchla = F[:, -1:]  # (n_classes, n_pigments) -> (n_classes, 1)
    F_scaled = F / tchla
    row_sums = F_scaled.sum(axis=1)  # (n_classes, n_pigments) -> (n_classes,)
    F_norm = F_scaled / row_sums[:, np.newaxis]  # row_sums: (n_classes,) -> (n_classes, 1)
    return F_norm, row_sums


def bounded_weights(S: np.ndarray, upper_bound: float = 30.0) -> np.ndarray:
    """
    Per-column weights: 1/col_mean, capped at upper_bound, with Tchla forced to 1. Mirrors R's Bounded_weights.

    A zero-mean column gives an infinite weight, so it gets upper_bound instead.
    """
    col_means = np.mean(S, axis=0)  # (n_samples, n_pigments) -> (n_pigments,)
    with np.errstate(divide="ignore"):
        weights = 1.0 / col_means
    weights = np.minimum(weights, upper_bound)
    weights[-1] = 1.0
    return weights


def vectorise(F: np.ndarray) -> np.ndarray:
    """
    Extract non-zero elements of F in column-major order. Mirrors R's vectorise.

    R's `Fmat[Fmat > 0]` reads column-major storage, so the flatten uses order="F" to give the same element order.
    """
    flat = F.flatten(order="F")  # (n_classes, n_pigments - 1) -> (n_classes * (n_pigments - 1),)
    return flat[flat > 0]


def apply_weights(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Column-scale X by weights. Mirrors R's Weight_error, which does S %*% diag(cm)."""
    return X * weights[np.newaxis, :]  # weights: (n_pigments,) -> (1, n_pigments)


def wrangling(
    F: np.ndarray,
    min_vals: np.ndarray,
    max_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build flat min/max/current-value vectors aligned to F's non-zero entries. Mirrors R's Wrangling.

    Args:
        F: (n_classes, n_pigments) with Tchla as the last column.
        min_vals: lower bounds per unit Tchla, ordered column-major over the non-zero entries of F's non-Tchla columns.
        max_vals: upper bounds in the same order as min_vals.

    Returns:
        (minF_vec, maxF_vec, SE_vec, chlv). The first three are (n_vary,) column-major non-zero vectors over F's non-Tchla columns, and SE_vec holds F's own values. minF_vec and maxF_vec are multiplied by chlv, so they carry absolute pigment ratios instead of ratios per unit Tchla. chlv is F's Tchla column, shape (n_classes,).
    """
    F = np.asarray(F, dtype=float)
    chlv = F[:, -1].copy()  # (n_classes, n_pigments) -> (n_classes,)
    Fd = F[:, :-1].copy()  # (n_classes, n_pigments) -> (n_classes, n_pigments - 1)

    Fmin = np.zeros_like(Fd)
    Fmax = np.zeros_like(Fd)
    mask_column_major = Fd.flatten(order="F") > 0  # (n_classes, n_pigments - 1) -> (n_classes * (n_pigments - 1),)
    Fmin_flat = Fmin.flatten(order="F")  # (n_classes, n_pigments - 1) -> (n_classes * (n_pigments - 1),)
    Fmax_flat = Fmax.flatten(order="F")  # (n_classes, n_pigments - 1) -> (n_classes * (n_pigments - 1),)
    Fmin_flat[mask_column_major] = min_vals
    Fmax_flat[mask_column_major] = max_vals
    Fmin = Fmin_flat.reshape(Fd.shape, order="F")  # (n_classes * (n_pigments - 1),) -> (n_classes, n_pigments - 1)
    Fmax = Fmax_flat.reshape(Fd.shape, order="F")  # (n_classes * (n_pigments - 1),) -> (n_classes, n_pigments - 1)

    Fmin = Fmin * chlv[:, np.newaxis]  # chlv: (n_classes,) -> (n_classes, 1)
    Fmax = Fmax * chlv[:, np.newaxis]  # chlv: (n_classes,) -> (n_classes, 1)

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
    Look up (class, pigment) pairs in min_max_df for every non-zero entry of F. Mirrors R's Default_min_max.

    Args:
        min_max_df: DataFrame with columns "Class", "Pig_Abbrev", "min", "max". One row per pair, for example Class "Syn" with Pig_Abbrev "Zea".
        F: Binary or scored matrix excluding the Tchla column.
        class_names: Row names of F.
        pigment_names: Column names of F.

    Returns:
        (min_vec, max_vec), each ordered column-major over F's non-zero entries to match R's which(F > 0, arr.ind=TRUE).

    Raises:
        ValueError: at least one (class, pigment) pair is missing from min_max_df.
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
