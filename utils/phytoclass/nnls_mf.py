"""
Weighted NNLS matrix factorization. Ports phytoclass::NNLS_MF and ::NNLS_MF_Final.

Solves S ~ C @ F subject to C >= 0, column-weighted, with C row-normalized
in the returned "C matrix" but the raw C used for RMSE.

Performance: R's RcppML::nnls solves the normal equations A^T A x = A^T b
via coordinate descent, reusing A^T A across all samples. scipy.optimize.nnls
only takes one b at a time and has significant per-call Python overhead, so
a pure scipy implementation is ~4x slower than necessary. We implement the
same coordinate-descent-on-normal-equations approach here, vectorized across
all samples simultaneously.
"""

import numpy as np

from utils.phytoclass.util import (
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

    Coordinate descent on the normal equations A^T A x = A^T b with shared
    A^T A across all samples. Mirrors R's RcppML::nnls(cd_maxit=1000, cd_tol=1e-8).

    Args:
        A: (m, k) - in phytoclass: weighted F transpose (pigments, classes).
        B: (n, m) - in phytoclass: weighted S (samples, pigments).

    Returns:
        X: (n, k) - non-negative solutions.
    """
    AtA = A.T @ A
    AtB = A.T @ B.T
    k, n_samples = AtB.shape
    X = np.zeros((k, n_samples))
    diag_AtA = np.diag(AtA).copy()
    diag_AtA[diag_AtA == 0] = 1.0

    for _ in range(max_iter):
        X_prev = X.copy()
        for j in range(k):
            residual_j = AtB[j] - AtA[j] @ X + AtA[j, j] * X[j]
            X[j] = np.maximum(0.0, residual_j / diag_AtA[j])
        if np.max(np.abs(X - X_prev)) < tol:
            break

    return X.T


def nnls_mf(F: np.ndarray, S: np.ndarray, S_weights: np.ndarray | None = None) -> dict:
    """
    Solve S ~ C @ F column-weighted, with NNLS. Mirrors R's NNLS_MF.

    Returns dict with keys:
        F: input F unchanged
        RMSE: sqrt(mean((S - C_raw @ F)^2)) - raw (not row-normalized) C
        C: row-normalized C (each row sums to 1), shape (n_samples, n_classes)
        C_raw: the raw NNLS output before row-normalization (for internal use)
    """
    F = np.asarray(F, dtype=float)
    S = np.asarray(S, dtype=float)
    n_pigments = S.shape[1]

    if S_weights is None:
        S_weights = np.ones(n_pigments)

    F_weighted = apply_weights(F, S_weights)
    S_weighted = apply_weights(S, S_weights)

    C_raw = _solve_nnls_batched(F_weighted.T, S_weighted)

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

    Normalizes F rows (via normalise_F), then solves NNLS on the row-normalized
    S, then rescales the resulting C by S_Chl to get per-sample class abundances
    in absolute units (mg/m^3 Chla).

    Returns dict with keys F (row-normalized, rescaled to absolute ratios via
    row_sums), RMSE, condition_number, class_abundances (n_samples, n_classes),
    MAE, residuals. Callers pass the row-normalized S here, not raw S.
    """
    F = np.asarray(F, dtype=float)
    S = np.asarray(S, dtype=float)

    F_norm, row_sums = normalise_F(F)
    F_scaled = F_norm * row_sums[:, np.newaxis]

    F_weighted = apply_weights(F_scaled, S_weights)
    S_weighted = apply_weights(S, S_weights)

    C_raw = _solve_nnls_batched(F_weighted.T, S_weighted)

    Cn_row_sums = C_raw.sum(axis=1, keepdims=True)
    Cn_row_sums_safe = np.where(Cn_row_sums == 0, 1.0, Cn_row_sums)
    C_normalized = C_raw / Cn_row_sums_safe
    class_abundances = C_normalized * S_Chl[:, np.newaxis]

    residuals = S - C_raw @ F_scaled
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = np.mean(np.abs(residuals), axis=0)

    condition_number = float(np.linalg.cond(F_scaled @ S.T))

    return {
        "F": F_scaled,
        "RMSE": rmse,
        "condition_number": condition_number,
        "class_abundances": class_abundances,
        "MAE": mae,
        "residuals": residuals,
    }
