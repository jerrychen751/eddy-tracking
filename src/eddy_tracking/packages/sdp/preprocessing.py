"""
Rrs spectral preprocessing for the SDP model.

Implements the Kramer et al. (2022) preprocessing chain:
  1. Interpolate to 1 nm grid (on padded range 396-704 nm)
  2. 5 nm centered moving mean smoothing
  3. Trim 4 nm from each edge -> final 400-700 nm at 1 nm (301 values)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.interpolate import CubicSpline


def moving_mean(values: np.ndarray, window: int) -> np.ndarray:
    """
    Centered moving mean that requires a full window of finite values.

    Uses convolution to compute a sliding sum and a sliding count of finite elements. Only positions where the full window is finite get a value; everywhere else is NaN.

    Args:
        values: 1D input array (may contain NaNs).
        window: Odd positive integer for the window width.

    Returns:
        1D array of same length. NaN where the full window wasn't available.
    """
    if window <= 0 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")

    values = np.asarray(values, dtype=float)
    kernel = np.ones(window, dtype=float)

    finite = np.isfinite(values)
    sum_ = np.convolve(np.where(finite, values, 0.0), kernel, mode="same")
    count = np.convolve(finite.astype(float), kernel, mode="same")

    out = np.full_like(values, np.nan, dtype=float)
    full = count >= float(window)
    out[full] = sum_[full] / count[full]
    return out


def preprocess_rrs_spectrum(
    wavelengths_nm: np.ndarray,
    rrs: np.ndarray,
    *,
    interp_nm: int = 1,
    smooth_nm: int = 5,
    edge_trim_nm: int = 4,
    final_range_nm: Sequence[int] = (400, 700),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate, smooth, and trim a single Rrs spectrum to a fixed 1 nm grid.

    To make the final [400, 700] values valid after smoothing, interpolation is performed on an extended grid (final_range +/- edge_trim) which is then trimmed back to final_range. The four defaults match the Kramer et al. workflow.

    Args:
        wavelengths_nm: Native wavelength centers in nm (1D).
        rrs: Rrs values in sr^-1 at those wavelengths (1D, same shape).
        interp_nm: Interpolation step size in nm.
        smooth_nm: Moving-mean window width in nm.
        edge_trim_nm: Number of nm to trim from each edge after smoothing.
        final_range_nm: (min, max) wavelength range in nm for the output.

    Returns:
        (out_wavelengths, out_rrs) - both 1D arrays of length (final_max - final_min) / interp_nm + 1 (default: 301).
    """
    wl = np.asarray(wavelengths_nm, dtype=float)
    y = np.asarray(rrs, dtype=float)
    if wl.shape != y.shape:
        raise ValueError(f"wavelengths and rrs must have the same shape; got {wl.shape} vs {y.shape}")

    if len(final_range_nm) != 2:
        raise ValueError("final_range_nm must be a (min, max) pair")
    final_min, final_max = int(final_range_nm[0]), int(final_range_nm[1])
    if final_min >= final_max:
        raise ValueError(f"Invalid final_range_nm: {final_range_nm}")

    extended_min = final_min - int(edge_trim_nm)
    extended_max = final_max + int(edge_trim_nm)
    target = np.arange(extended_min, extended_max + 1, int(interp_nm), dtype=float)

    finite = np.isfinite(wl) & np.isfinite(y)
    if finite.sum() < 2:
        raise ValueError("Not enough finite wavelength/value pairs to interpolate")

    wl = wl[finite]
    y = y[finite]

    order = np.argsort(wl)
    wl = wl[order]
    y = y[order]

    # De-duplicate wavelength centers (keep the first occurrence).
    uniq_wl, uniq_idx = np.unique(wl, return_index=True)
    wl = uniq_wl
    y = y[uniq_idx]
    if wl.size < 2:
        raise ValueError("Not enough unique wavelength points to interpolate")

    interp = np.full_like(target, np.nan, dtype=float)
    in_range = (target >= wl.min()) & (target <= wl.max())
    if in_range.any():
        cs = CubicSpline(wl, y, extrapolate=False)
        interp[in_range] = cs(target[in_range])

    smoothed = moving_mean(interp, window=int(smooth_nm))

    keep = (target >= final_min) & (target <= final_max)
    out_wl = target[keep].astype(int)
    out_rrs = smoothed[keep]

    if out_wl[0] != final_min or out_wl[-1] != final_max:
        raise AssertionError("Internal error: output wavelength grid does not match requested final_range_nm")
    return out_wl.astype(float), out_rrs


def preprocess_rrs_batch(
    wavelengths_nm: np.ndarray,
    rrs_2d: np.ndarray,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply preprocess_rrs_spectrum to each row of a 2D Rrs array.

    Args:
        wavelengths_nm: 1D array of native wavelength centers in nm (shared by all rows).
        rrs_2d: 2D array of Rrs in sr^-1 with shape (n_obs, n_wavelengths), where each row is one observation (pixel).
        **kwargs: Forwarded to preprocess_rrs_spectrum (interp_nm, smooth_nm, etc.).

    Returns:
        (wavelengths, processed_2d) where wavelengths is the 1D output wavelength array and processed_2d has shape (n_obs, len(wavelengths)).
    """
    rrs_2d = np.asarray(rrs_2d, dtype=float)
    if rrs_2d.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {rrs_2d.shape}")
    if rrs_2d.shape[0] == 0:
        raise ValueError("Empty input array (0 rows)")

    wl_out = None
    rows = []
    for i in range(rrs_2d.shape[0]):
        wl_out, rrs_out = preprocess_rrs_spectrum(wavelengths_nm, rrs_2d[i], **kwargs)
        rows.append(rrs_out)

    return wl_out, np.stack(rows)  # n_obs arrays of (n_out_wavelengths,) -> (n_obs, n_out_wavelengths)
