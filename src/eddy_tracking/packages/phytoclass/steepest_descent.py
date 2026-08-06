"""
Steepest descent refinement for F. Ports phytoclass::Steepest_Descent and its helpers (Replace_Rand / Randomise_elements / Fac_F_RR / Conduit / Minimise_elements_comb).
"""

import numpy as np

from eddy_tracking.packages.phytoclass.nnls_mf import nnls_mf


def _randomise_element(
    x: float,
    min_scaler: float,
    max_scaler: float,
    rng: np.random.Generator,
) -> float:
    """Mirrors R's Randomise_elements: a value below 0.001 is treated as 0.001 before scaling."""
    x_safe = max(x, 0.001)
    return float(np.round(rng.uniform(x_safe * min_scaler, x_safe * max_scaler), 4))


def _fac_f_rr(
    F_baseline: np.ndarray,
    rmse_baseline: float,
    S: np.ndarray,
    weights: np.ndarray,
    vary_rows: np.ndarray,
    vary_cols: np.ndarray,
    scaler_idx: int,
    rng: np.random.Generator,
) -> dict:
    """
    Try each (row, col) perturbation on its own, then apply every one that beat rmse_baseline and re-run NNLS. Mirrors R's Fac_F_RR.

    Improvements found one at a time need not compose, and R accepts the combined F regardless. When nothing improves, the search cascades to scaler_idx - 1 and stops at 1, which returns the baseline state.
    """
    # scaler_idx -> (low, high) multipliers on the current value, for example 2 -> (0.98, 1.02).
    scalers = {1: (0.99, 1.01), 2: (0.98, 1.02), 3: (0.97, 1.03)}
    min_s, max_s = scalers[scaler_idx]
    n_vary = len(vary_rows)
    new_values = np.empty(n_vary)
    improved = np.zeros(n_vary, dtype=bool)

    for k in range(n_vary):
        i, j = vary_rows[k], vary_cols[k]
        new_val = _randomise_element(F_baseline[i, j], min_s, max_s, rng)
        new_values[k] = new_val

        F_test = F_baseline.copy()
        F_test[i, j] = new_val
        test_rmse = nnls_mf(F_test, S, weights)["RMSE"]
        improved[k] = test_rmse < rmse_baseline

    if improved.any():
        F_combined = F_baseline.copy()
        F_combined[vary_rows[improved], vary_cols[improved]] = new_values[improved]
        return nnls_mf(F_combined, S, weights)

    if scaler_idx > 1:
        return _fac_f_rr(
            F_baseline, rmse_baseline, S, weights,
            vary_rows, vary_cols, scaler_idx - 1, rng,
        )

    return nnls_mf(F_baseline, S, weights)


def _minimise_elements_comb(
    F_current: np.ndarray,
    S: np.ndarray,
    weights: np.ndarray,
    vary_rows: np.ndarray,
    vary_cols: np.ndarray,
    scaler_idx: int,
    rng: np.random.Generator,
) -> dict:
    """Two passes of _fac_f_rr, keeping whichever has the lower RMSE. Mirrors R's Minimise_elements_comb and Conduit."""
    baseline = nnls_mf(F_current, S, weights)

    pass1 = _fac_f_rr(
        F_current, baseline["RMSE"], S, weights,
        vary_rows, vary_cols, scaler_idx, rng,
    )
    pass2 = _fac_f_rr(
        pass1["F"], pass1["RMSE"], S, weights,
        vary_rows, vary_cols, scaler_idx, rng,
    )

    if pass2["RMSE"] < pass1["RMSE"]:
        return pass2
    return pass1


def steepest_descent(
    F_initial: np.ndarray,
    S: np.ndarray,
    weights: np.ndarray,
    vary_rows: np.ndarray,
    vary_cols: np.ndarray,
    num_loops: int,
    rng: np.random.Generator,
) -> dict:
    """
    R's Steepest_Descent outer loop.

    Runs _minimise_elements_comb at scaler 3 up to num_loops times. While an iteration leaves the RMSE worse than the current one, it retries on R's scaler schedule: 5 calls at scaler 3, then 4 calls at scaler 1, then 91 calls at scaler 2, then break.

    Args:
        F_initial: full F matrix (n_classes, n_pigments) including the Tchla column.
        S: row-normalized sample matrix (n_samples, n_pigments).
        weights: (n_pigments,) NNLS weights, with the Tchla weight forced to 1.
        vary_rows, vary_cols: (n_vary,) coordinates of the non-zero entries in F's non-Tchla submatrix, so every vary_cols entry is below n_pigments - 1.
        num_loops: outer iteration count. R uses 10 when the SA temperature is above 0.3, else 2.
        rng: numpy RNG.

    Returns the dict from nnls_mf on the final F.
    """
    current = nnls_mf(F_initial, S, weights)

    for _ in range(num_loops):
        result = _minimise_elements_comb(
            current["F"], S, weights, vary_rows, vary_cols, 3, rng,
        )

        retry = 1
        while result["RMSE"] > current["RMSE"]:
            if retry <= 5:
                c1_num = 3
            elif retry < 10:
                c1_num = 1
            elif retry <= 100:
                c1_num = 2
            else:
                break
            result = _minimise_elements_comb(
                current["F"], S, weights, vary_rows, vary_cols, c1_num, rng,
            )
            retry += 1

        current = result

    return current
