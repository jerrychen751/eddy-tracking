"""NNLS-based matrix factorization: solve S ~ C @ F subject to C >= 0."""

import numpy as np
from scipy.optimize import nnls


def _compute_weights(S: np.ndarray, upper_bound: float = 30.0) -> np.ndarray:
    """
    Compute column weights as inverse of column means, capped at upper_bound.

    Prevents rare pigments from dominating the fit while avoiding
    extreme upweighting of near-zero columns.

    Args:
        S: Sample matrix (n_samples, n_pigments).
        upper_bound: Maximum allowed weight value.

    Returns:
        1D weight array of length n_pigments.
    """
    col_means = np.mean(S, axis=0)
    col_means = np.where(col_means > 0, col_means, 1e-10)
    weights = 1.0 / col_means
    return np.minimum(weights, upper_bound)


def nnls_factorize(
    F: np.ndarray,
    S: np.ndarray,
    weight_upper_bound: float = 30.0,
) -> dict:
    """
    Solve S ~ C @ F via vectorized least-squares with NNLS fallback.

    First solves the unconstrained weighted least-squares for all samples
    in a single LAPACK call. Samples whose solutions are already non-negative
    are kept as-is. Only samples with negative entries fall back to the
    slower per-row scipy NNLS, which enforces c >= 0 exactly.

    Args:
        F: Pigment ratio matrix (n_classes, n_pigments). Each row is one
           phytoplankton class's pigment:Tchla ratios.
        S: Sample matrix (n_samples, n_pigments). Measured pigment concentrations.
        weight_upper_bound: Cap on inverse-mean column weights.

    Returns:
        Dict with keys:
            C: Class abundances (n_samples, n_classes), all >= 0.
            F: The input F matrix (unchanged).
            rmse: Scalar RMSE of the reconstruction S vs C @ F.
    """
    weights = _compute_weights(S, weight_upper_bound)
    W = np.diag(weights)

    # Weight both F and S
    F_w = (F @ W).T   # (n_pigments, n_classes)
    S_w = S @ W        # (n_samples, n_pigments)

    # Vectorized unconstrained solve: F_w @ C.T = S_w.T
    # lstsq uses SVD internally — single LAPACK call for all samples
    C_T, _, _, _ = np.linalg.lstsq(F_w, S_w.T, rcond=None)
    C = C_T.T  # (n_samples, n_classes)

    # Fall back to per-sample NNLS only for rows that violate c >= 0
    neg_rows = np.where(np.any(C < 0, axis=1))[0]
    for i in neg_rows:
        C[i], _ = nnls(F_w, S_w[i])

    S_reconstructed = C @ F
    rmse = np.sqrt(np.mean((S - S_reconstructed) ** 2))

    return {"C": C, "F": F, "rmse": rmse}
