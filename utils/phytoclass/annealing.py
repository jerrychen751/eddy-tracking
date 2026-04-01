"""Simulated annealing + steepest descent for phytoclass F matrix optimization."""

import numpy as np
from utils.phytoclass.nnls_mf import nnls_factorize


def _random_neighbor(
    F_current: np.ndarray,
    bounds: dict[tuple[int, int], tuple[float, float]],
    temperature: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a neighbor F matrix by perturbing non-zero entries.

    Each non-zero entry is perturbed by:
        p_new = p_current + temperature * (max - min) * Uniform(-1, 1)
    clamped to [min, max].

    Args:
        F_current: Current F matrix.
        bounds: Dict mapping (row, col) to (min_val, max_val).
        temperature: SA temperature in [0, 1].
        rng: numpy random generator.

    Returns:
        New F matrix with perturbed values.
    """
    F_new = F_current.copy()
    for (i, j), (lo, hi) in bounds.items():
        perturbation = temperature * (hi - lo) * rng.uniform(-1, 1)
        F_new[i, j] = np.clip(F_current[i, j] + perturbation, lo, hi)
    return F_new


def _steepest_descent(
    F: np.ndarray,
    S: np.ndarray,
    bounds: dict[tuple[int, int], tuple[float, float]],
    n_loops: int,
    rng: np.random.Generator,
    weight_upper_bound: float,
) -> tuple[np.ndarray, float]:
    """
    Greedy local refinement: try small perturbations, keep improvements.

    For each non-zero F entry, tries a +/-3% multiplicative perturbation.
    Accepts only changes that reduce RMSE.

    Args:
        F: Current F matrix.
        S: Sample matrix.
        bounds: Ratio bounds.
        n_loops: Number of passes over all entries.
        rng: numpy random generator.
        weight_upper_bound: Weight cap for NNLS.

    Returns:
        Tuple of (improved F matrix, final RMSE).
    """
    best_F = F.copy()
    best_rmse = nnls_factorize(best_F, S, weight_upper_bound)["rmse"]

    for _ in range(n_loops):
        for (i, j), (lo, hi) in bounds.items():
            # Try a small multiplicative perturbation
            factor = rng.uniform(0.97, 1.03)
            candidate = best_F.copy()
            candidate[i, j] = np.clip(best_F[i, j] * factor, lo, hi)

            result = nnls_factorize(candidate, S, weight_upper_bound)
            if result["rmse"] < best_rmse:
                best_F = candidate
                best_rmse = result["rmse"]

    return best_F, best_rmse


def simulated_annealing(
    S: np.ndarray,
    F_structure: np.ndarray,
    bounds: dict[tuple[int, int], tuple[float, float]],
    n_iter: int = 500,
    n_neighbors: int = 120,
    cooling_step: float = 0.009,
    weight_upper_bound: float = 30.0,
    seed: int | None = None,
) -> dict:
    """
    Find the optimal F matrix via simulated annealing with steepest descent.

    Starting from a binary structure matrix, uses SA to globally search the
    ratio space defined by bounds, with local steepest descent refinement
    at each step. The Metropolis criterion allows temporary uphill moves
    to escape local minima.

    Args:
        S: Sample pigment matrix (n_samples, n_pigments).
        F_structure: Binary structure matrix (n_classes, n_pigments). 1 = pigment
            present, 0 = absent. Last column should be Tchla (always 1).
        bounds: Dict mapping (class_idx, pigment_idx) to (min_ratio, max_ratio)
            for each non-zero, non-Tchla entry in F_structure.
        n_iter: Number of SA iterations.
        n_neighbors: Number of random neighbors generated per iteration.
        cooling_step: Controls exponential cooling. Temperature = (1 - step)^k.
        weight_upper_bound: Cap on column weights for NNLS.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys:
            C: Optimized class abundances (n_samples, n_classes).
            F: Optimized pigment ratio matrix (n_classes, n_pigments).
            rmse: Final RMSE.
            rmse_history: List of RMSE values at each iteration.
    """
    rng = np.random.default_rng(seed)
    decay = 1.0 - cooling_step

    # Initialize F with midpoint of bounds for non-zero entries
    F_current = F_structure.astype(float).copy()
    for (i, j), (lo, hi) in bounds.items():
        F_current[i, j] = (lo + hi) / 2.0

    # Initial solve
    current_result = nnls_factorize(F_current, S, weight_upper_bound)
    current_rmse = current_result["rmse"]

    best_F = F_current.copy()
    best_rmse = current_rmse
    rmse_history = [current_rmse]

    for k in range(1, n_iter + 1):
        temperature = decay ** k

        # Generate neighbors and pick the best
        best_neighbor_F = None
        best_neighbor_rmse = np.inf

        # Use more neighbors in the final 20 iterations for fine-tuning
        n_nbr = n_neighbors * 3 if k > n_iter - 20 else n_neighbors

        for _ in range(n_nbr):
            candidate_F = _random_neighbor(F_current, bounds, temperature, rng)
            result = nnls_factorize(candidate_F, S, weight_upper_bound)
            if result["rmse"] < best_neighbor_rmse:
                best_neighbor_rmse = result["rmse"]
                best_neighbor_F = candidate_F

        # Steepest descent refinement
        # More local refinement at low temperature (exploitation phase)
        n_sd_loops = 2 if temperature > 0.3 else 10
        refined_F, refined_rmse = _steepest_descent(
            best_neighbor_F, S, bounds, n_sd_loops, rng, weight_upper_bound
        )

        # Metropolis acceptance criterion
        delta = refined_rmse - current_rmse
        if delta < 0 or rng.random() < np.exp(-delta / max(temperature, 1e-10)):
            F_current = refined_F
            current_rmse = refined_rmse

        # Track global best
        if refined_rmse < best_rmse:
            best_rmse = refined_rmse
            best_F = refined_F.copy()

        rmse_history.append(best_rmse)

    # Final solve with best F
    final_result = nnls_factorize(best_F, S, weight_upper_bound)
    best_C = final_result["C"]

    return {
        "C": best_C,
        "F": best_F,
        "rmse": best_rmse,
        "rmse_history": rmse_history,
    }
