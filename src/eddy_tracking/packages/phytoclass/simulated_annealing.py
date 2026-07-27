"""
Simulated annealing F matrix optimization. Ports phytoclass::simulated_annealing.

Matches the R implementation modulo RNG differences (R's Mersenne Twister vs
numpy's PCG64) - the algorithm is byte-for-byte equivalent, but trajectories
differ because the random draws differ.
"""

import numpy as np
import pandas as pd

from eddy_tracking.packages.phytoclass.matrix_checks import matrix_checks
from eddy_tracking.packages.phytoclass.nnls_mf import nnls_mf, nnls_mf_final
from eddy_tracking.packages.phytoclass.random_neighbour import random_neighbour
from eddy_tracking.packages.phytoclass.steepest_descent import steepest_descent
from eddy_tracking.packages.phytoclass.util import (
    bounded_weights,
    default_min_max,
    normalise_S,
    wrangling,
)


def _compute_place_coords(F_no_tchla: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (flat_indices, rows, cols) for non-zero entries in column-major order.

    flat_indices matches R's `place <- which(F > 0)` (0-indexed here, 1-indexed
    in R). rows/cols are the corresponding (row, col) coordinate arrays.
    """
    n_rows = F_no_tchla.shape[0]
    flat_F = F_no_tchla.flatten(order="F")
    flat_indices = np.flatnonzero(flat_F)
    rows = flat_indices % n_rows
    cols = flat_indices // n_rows
    return flat_indices, rows, cols


def simulated_annealing(
    S: pd.DataFrame,
    Fmat: pd.DataFrame,
    user_defined_min_max: pd.DataFrame,
    do_matrix_checks: bool = True,
    niter: int = 500,
    step: float = 0.009,
    weight_upper_bound: float = 30.0,
    seed: int | None = None,
    verbose: bool = False,
) -> dict:
    """
    Find an optimal F matrix via simulated annealing with steepest descent.

    Mirrors R's simulated_annealing() with these fixed choices:
        - niter = 500 SA iterations
        - step = 0.009, giving temperature schedule Temp = (1 - step)^k = 0.991^k
        - 120 random neighbors per iteration, bumped to 360 in the final 20
        - Steepest descent loop count: 10 when Temp > 0.3 else 2
        - Metropolis acceptance: exp(-(f_n_err - f_c_err)) < uniform(0, 1),
          with NO temperature division (matches R, not textbook SA)
        - Weighted NNLS, Tchla weight forced to 1
        - Binary initial F (all non-zero entries → 1)

    Args:
        S: sample matrix (n_samples, n_pigments), Tchla MUST be the last column.
           DataFrame or ndarray. Row-normalization is applied inside this
           function - pass raw concentrations.
        Fmat: pigment-to-Tchla ratio matrix (n_classes, n_pigments) with the
            same column order as S and row labels for phytoplankton classes.
            DataFrame preferred so class and pigment names are preserved.
        user_defined_min_max: DataFrame with columns Class, Pig_Abbrev, min,
            max. One row per non-zero (class, pigment) pair in Fmat. Tchla
            rows are ignored.
        niter, step, weight_upper_bound: SA hyperparameters (match R defaults).
        seed: numpy RNG seed. Hermetic (not affected by external numpy state).
        verbose: print per-iteration progress if True.

    Returns dict with keys matching R's output:
        F: final (row-normalized) F matrix as DataFrame
        RMSE: final reconstruction RMSE
        condition_number: kappa of Fn @ S^T
        class_abundances: DataFrame of absolute class abundances (mg/m^3 Chla)
        MAE: per-pigment mean absolute error
        residuals: S - C @ F residual matrix
        rmse_history: list of best-so-far RMSE at each iteration (Python addition)
    """
    if not isinstance(S, pd.DataFrame):
        raise TypeError("S must be a DataFrame with pigment column names.")
    if not isinstance(Fmat, pd.DataFrame):
        raise TypeError("Fmat must be a DataFrame with class row names and pigment column names.")

    if do_matrix_checks:
        S, Fmat = matrix_checks(S, Fmat)

    S_df = S
    class_names = list(Fmat.index)
    pigment_names = list(Fmat.columns)
    S_array = S.values.astype(float)
    Fmat = Fmat.values.astype(float)
    S = S_array

    if Fmat.shape[1] != S.shape[1]:
        raise ValueError(
            f"F matrix has {Fmat.shape[1]} columns but S has {S.shape[1]} - "
            "column counts must match and Tchla must be the last column in both."
        )

    rng = np.random.default_rng(seed)

    S_Chl = S[:, -1].copy()
    S_normalized = normalise_S(S)
    S_weights = bounded_weights(S_normalized, weight_upper_bound)

    Fmat_binary = (Fmat > 0).astype(float)

    Fd_binary = Fmat_binary[:, :-1]
    place_flat, vary_rows, vary_cols = _compute_place_coords(Fd_binary)

    min_vec_bounds, max_vec_bounds = default_min_max(
        user_defined_min_max, Fd_binary, class_names, pigment_names[:-1],
    )

    minF, maxF, SE, chlv = wrangling(Fmat_binary, min_vec_bounds, max_vec_bounds)

    nnls_initial = nnls_mf(Fmat_binary, S_normalized, S_weights)
    f_best = nnls_initial["F"].copy()
    f_current = nnls_initial["F"].copy()
    f_best_err = nnls_initial["RMSE"]
    f_current_err = nnls_initial["RMSE"]

    rmse_history = [f_best_err]
    step_decay = 1.0 - step

    for k in range(1, niter + 1):
        temperature = step_decay ** k

        num_loop = 300 if k > niter - 20 else 120

        D_list = []
        Dn = np.empty(num_loop)
        N_subset = place_flat
        for i in range(num_loop):
            temp_rand = random_neighbour(
                f_current, temperature, chlv,
                N_subset, place_flat, S_normalized, S_weights, minF, maxF, rng,
            )
            D_list.append(temp_rand)
            Dn[i] = temp_rand["RMSE"]

        best_idx = int(np.argmin(Dn))
        new_neighbour = D_list[best_idx]

        num_loop2 = 10 if temperature > 0.3 else 2
        new_neighbour = steepest_descent(
            new_neighbour["F"], S_normalized, S_weights,
            vary_rows, vary_cols, num_loop2, rng,
        )
        f_n = new_neighbour["F"]

        f_n_vect = f_n[vary_rows, vary_cols]
        oob_idx = np.where((f_n_vect < minF) | (f_n_vect > maxF))[0]

        while len(oob_idx) > 0:
            N_oob = place_flat[oob_idx]
            D2_list = []
            Dn2 = np.empty(num_loop)
            for i in range(num_loop):
                temp_rand = random_neighbour(
                    f_n, temperature, chlv,
                    N_oob, place_flat, S_normalized, S_weights, minF, maxF, rng,
                )
                D2_list.append(temp_rand)
                Dn2[i] = temp_rand["RMSE"]

            combined_Dn = np.concatenate([Dn, Dn2])
            combined_D = D_list + D2_list
            best_idx = int(np.argmin(combined_Dn))
            new_neighbour = combined_D[best_idx]
            f_n = new_neighbour["F"]

            f_n_vect = f_n[vary_rows, vary_cols]
            oob_idx = np.where((f_n_vect < minF) | (f_n_vect > maxF))[0]

        f_n_err = new_neighbour["RMSE"]
        delta = f_n_err - f_current_err
        if delta < 0 or np.exp(-delta) < rng.uniform(0.0, 1.0):
            f_current = f_n
            f_current_err = f_n_err

        if f_n_err < f_best_err:
            f_best = f_n.copy()
            f_best_err = f_n_err

        rmse_history.append(f_best_err)

        if verbose and (k % 50 == 0 or k == niter):
            print(f"  SA iter {k}/{niter}: RMSE {f_best_err:.6f}, Temp {temperature:.4f}")

    final = nnls_mf_final(f_best, S_normalized, S_Chl, S_weights)

    F_df = pd.DataFrame(final["F"], index=class_names, columns=pigment_names)
    abundances_df = pd.DataFrame(
        final["class_abundances"],
        index=S_df.index if S_df is not None else None,
        columns=class_names,
    )
    mae_series = pd.Series(final["MAE"], index=pigment_names)

    return {
        "F": F_df,
        "RMSE": final["RMSE"],
        "condition_number": final["condition_number"],
        "class_abundances": abundances_df,
        "MAE": mae_series,
        "residuals": final["residuals"],
        "rmse_history": rmse_history,
    }
