"""
Weighted NNLS matrix factorization. Ports phytoclass::NNLS_MF and ::NNLS_MF_Final.

Solves S ~ C @ F subject to C >= 0, column-weighted. The returned "C matrix" is row-normalized, but the raw C drives the RMSE.

scipy.optimize.nnls takes one right-hand side per call, so this module runs coordinate descent on the normal equations instead, vectorized over every sample at once, as R's RcppML::nnls does.
"""

import numpy as np

from eddy_tracking.packages.phytoclass.util import (
    apply_weights,
    normalise_F,
)


def _solve_nnls_batched(
    A: np.ndarray,
    B: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Solve min ||A x_i - b_i||^2 s.t. x_i >= 0 for every row b_i of B.

    Args:
        A: (n_pigments, n_classes) - weighted F, transposed.
        B: (n_samples, n_pigments) - weighted S.
        max_iter, tol: defaults mirror R's RcppML::nnls(cd_maxit=1000, cd_tol=1e-8).

    Returns:
        X: (n_samples, n_classes), every entry non-negative.
    """
    AtA = A.T @ A  # (n_pigments, n_classes) -> (n_classes, n_classes)
    AtB = A.T @ B.T  # (n_pigments, n_classes), (n_samples, n_pigments) -> (n_classes, n_samples)
    k, n_samples = AtB.shape
    X = np.zeros((k, n_samples))
    diag_AtA = np.diag(AtA).copy()  # (n_classes, n_classes) -> (n_classes,)
    diag_AtA[diag_AtA == 0] = 1.0

    for _ in range(max_iter):
        X_prev = X.copy()
        for j in range(k):
            residual_j = AtB[j] - AtA[j] @ X + AtA[j, j] * X[j]
            X[j] = np.maximum(0.0, residual_j / diag_AtA[j])
        if np.max(np.abs(X - X_prev)) < tol:
            break

    return X.T  # (n_classes, n_samples) -> (n_samples, n_classes)


def nnls_mf(F: np.ndarray, S: np.ndarray, S_weights: np.ndarray | None = None) -> dict:
    """
    Solve S ~ C @ F column-weighted, with NNLS. Mirrors R's NNLS_MF.

    Returns dict with keys:
        F: the input F, unchanged.
        RMSE: sqrt(mean((S - C_raw @ F)^2)), from the raw C, not the row-normalized one.
        C: (n_samples, n_classes) with every row summing to 1.
        C_raw: the NNLS output before row-normalization, for internal use.
    """
    F = np.asarray(F, dtype=float)
    S = np.asarray(S, dtype=float)
    n_pigments = S.shape[1]

    if S_weights is None:
        S_weights = np.ones(n_pigments)

    F_weighted = apply_weights(F, S_weights)
    S_weighted = apply_weights(S, S_weights)

    C_raw = _solve_nnls_batched(F_weighted.T, S_weighted)  # F_weighted: (n_classes, n_pigments) -> (n_pigments, n_classes); C_raw: (n_samples, n_classes)

    row_sums = C_raw.sum(axis=1, keepdims=True)
    row_sums_safe = np.where(row_sums == 0, 1.0, row_sums)
    C_norm = C_raw / row_sums_safe

    reconstruction = C_raw @ F
    rmse = float(np.sqrt(np.mean((S - reconstruction) ** 2)))

    return {"F": F, "RMSE": rmse, "C": C_norm, "C_raw": C_raw}


def nnls_mf_final(
    F: np.ndarray,
    S: np.ndarray,
    S_Chl: np.ndarray,
    S_weights: np.ndarray,
) -> dict:
    """
    Final-step NNLS. Mirrors R's NNLS_MF_Final.

    Pass the row-normalized S here, not raw S. S_Chl holds each sample's Tchla, taken before that normalization, in mg/m^3.

    Returns dict with keys:
        F: the input F divided by its Tchla column, so the last column is all 1 and the rows do not sum to 1.
        RMSE: sqrt(mean(residuals^2)), in the units of the S that was passed in.
        condition_number: kappa of F @ S.T.
        class_abundances: (n_samples, n_classes) in mg/m^3 Chla, each row summing to that sample's S_Chl.
        MAE: (n_pigments,) mean absolute residual per pigment.
        residuals: S - C_raw @ F.
    """
    F = np.asarray(F, dtype=float)
    S = np.asarray(S, dtype=float)

    F_norm, row_sums = normalise_F(F)
    F_scaled = F_norm * row_sums[:, np.newaxis]  # row_sums: (n_classes,) -> (n_classes, 1)

    F_weighted = apply_weights(F_scaled, S_weights)
    S_weighted = apply_weights(S, S_weights)

    C_raw = _solve_nnls_batched(F_weighted.T, S_weighted)  # F_weighted: (n_classes, n_pigments) -> (n_pigments, n_classes); C_raw: (n_samples, n_classes)

    Cn_row_sums = C_raw.sum(axis=1, keepdims=True)
    Cn_row_sums_safe = np.where(Cn_row_sums == 0, 1.0, Cn_row_sums)
    C_normalized = C_raw / Cn_row_sums_safe
    class_abundances = C_normalized * S_Chl[:, np.newaxis]  # S_Chl: (n_samples,) -> (n_samples, 1)

    residuals = S - C_raw @ F_scaled
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = np.mean(np.abs(residuals), axis=0)  # (n_samples, n_pigments) -> (n_pigments,)

    condition_number = float(np.linalg.cond(F_scaled @ S.T))  # (n_classes, n_pigments), (n_pigments, n_samples) -> (n_classes, n_samples)

    return {
        "F": F_scaled,
        "RMSE": rmse,
        "condition_number": condition_number,
        "class_abundances": class_abundances,
        "MAE": mae,
        "residuals": residuals,
    }
