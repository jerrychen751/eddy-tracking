"""
Random neighbor generation for the SA loop. Ports phytoclass::Random_neighbour.

Perturbs selected non-zero entries of F by a temperature-scaled uniform jump,
retries any values that land out of bounds, then runs NNLS_MF on the new F.
"""

import numpy as np

from utils.phytoclass.nnls_mf import nnls_mf


def random_neighbour(
    f_current: np.ndarray,
    temperature: float,
    chlv: np.ndarray,
    N: np.ndarray,
    place: np.ndarray,
    S: np.ndarray,
    S_weights: np.ndarray,
    minF: np.ndarray,
    maxF: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """
    Perturb F entries at positions N by Temp * (max - min) * uniform(-1, 1).

    Args:
        f_current: current full F (n_classes, n_pigments) including Tchla column.
        temperature: SA temperature in [0, 1].
        chlv: Tchla column (n_classes,) to append back after perturbation.
        N: subset of `place` listing which entries to perturb. 0-indexed
           column-major positions in f_current[:, :-1].
        place: all non-zero positions in f_current[:, :-1] (same indexing as N).
        S: sample matrix (n_samples, n_pigments).
        S_weights: per-pigment NNLS weights.
        minF, maxF: flat bound vectors aligned to `place`, already multiplied
                    by chlv inside wrangling().
        rng: numpy RNG.

    OOB handling mirrors R:
        - Retry up to 50 rounds of fresh uniform draws for any entries that
          land outside [minF, maxF].
        - Beyond round 50, sample from U(min(min*1.2, max*0.8), max(...))
          — shrunken bounds with sort-fixup for the corner case where the
          shrinkage inverts them.

    Returns the dict from nnls_mf on the new F.
    """
    Fd = f_current[:, :-1]
    Fd_flat = Fd.flatten(order="F")

    k = np.searchsorted(place, N)
    p_chg = Fd_flat[N]
    minF_k = minF[k]
    maxF_k = maxF[k]

    rand = np.round(rng.uniform(-1.0, 1.0, size=len(N)), 4)
    p_new = p_chg + temperature * (maxF_k - minF_k) * rand
    oob = np.where((p_new < minF_k) | (p_new > maxF_k))[0]

    loop = 0
    while len(oob) > 0:
        loop += 1
        nr = np.round(rng.uniform(-1.0, 1.0, size=len(oob)), 4)
        p_new[oob] = p_chg[oob] + temperature * (maxF_k[oob] - minF_k[oob]) * nr
        oob = np.where((p_new < minF_k) | (p_new > maxF_k))[0]

        if loop > 50 and len(oob) > 0:
            lo_raw = minF_k[oob] * 1.2
            hi_raw = maxF_k[oob] * 0.8
            lo = np.minimum(lo_raw, hi_raw)
            hi = np.maximum(lo_raw, hi_raw)
            p_new[oob] = np.round(rng.uniform(lo, hi), 4)
            oob = np.where((p_new < minF_k) | (p_new > maxF_k))[0]

    Fd_new_flat = Fd_flat.copy()
    Fd_new_flat[N] = p_new
    Fd_new = Fd_new_flat.reshape(Fd.shape, order="F")
    f_new = np.column_stack([Fd_new, chlv])

    return nnls_mf(f_new, S, S_weights)
