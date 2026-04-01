"""SDP model prediction functions."""

from pathlib import Path

from utils.sdp.physics import get_rrs_residuals
import pandas as pd
import numpy as np

# Coefficient directory located alongside this module
_COEFF_DIR = Path(__file__).resolve().parent / "coefficients"


def run_sdp(
    rrs: pd.DataFrame,
    wl: np.ndarray,
    sst: np.ndarray,
    sss: np.ndarray,
    pigments: list[str] | None = None
) -> pd.DataFrame:
    """
    Predict pigment concentrations from Rrs spectra using trained SDP model.

    Computes Rrs residuals, takes 2nd derivative, then applies ensemble of 100 model permutations. Takes median over permutations. Negative predictions clipped to zero.

    Args:
        rrs: DataFrame with integer wavelength columns (400..700), each row is a spectrum.
        wl: 1D wavelength array matching the DataFrame columns.
        sst: Sea surface temperature array, one value per spectrum.
        sss: Sea surface salinity array, one value per spectrum.
        pigments: List of pigment names to predict, or None for all 13.

    Available pigments: Tchla, Zea, DVchla, ButFuco, HexFuco, Allo, MVchlb, Neo, Viola, Fuco, Chlc12, Chlc3, Perid.

    Returns:
        DataFrame with predicted concentrations (ug/L).
    """

    # All 13 pigments that were trained (from Kramer_Rrs_pigments.py)
    all_available_pigments = ['Tchla','Zea','DVchla','ButFuco','HexFuco','Allo','MVchlb',
                              'Neo','Viola','Fuco','Chlc12','Chlc3','Perid']

    # Use all pigments if none specified, otherwise use the requested ones
    if pigments is None:
        sdp_names = all_available_pigments.copy()
    else:
        # Validate requested pigments
        invalid_pigments = [p for p in pigments if p not in all_available_pigments]
        if invalid_pigments:
            raise ValueError(
                f"Invalid pigment names: {invalid_pigments}. "
                f"Available pigments: {all_available_pigments}"
            )
        sdp_names = pigments.copy()

    # Display names for output DataFrame columns
    display_names = {
        'Tchla': 'T chla',
        'Zea': 'Zea',
        'DVchla': 'DV chla',
        'ButFuco': 'ButFuco',
        'HexFuco': 'HexFuco',
        'Allo': 'Allo',
        'MVchlb': 'MV chlb',
        'Neo': 'Neo',
        'Viola': 'Viola',
        'Fuco': 'Fuco',
        'Chlc12': 'chl c1+c2',
        'Chlc3': 'chl c3',
        'Perid': 'Perid'
    }

    # Check if model coefficients exist before processing
    missing_coeffs = []
    for name in sdp_names:
        a_file = _COEFF_DIR / f'a_coefs_{name}.csv'
        c_file = _COEFF_DIR / f'c_coefs_{name}.csv'
        if not a_file.exists() or not c_file.exists():
            missing_coeffs.append(name)

    if missing_coeffs:
        raise FileNotFoundError(
            f"Model coefficients not found for pigments: {missing_coeffs}\n"
            f"Expected location: {_COEFF_DIR}\n"
            f"Copy the coefficient CSVs from the rrs-SDP-pigments repo."
        )

    rrs_residuals = get_rrs_residuals(rrs, sst, sss, wl)[1]
    rrs_residuals_d2 = np.diff(rrs_residuals, 2, axis=0).T

    sdp = np.zeros((rrs_residuals_d2.shape[0], len(sdp_names)))

    for p, name in enumerate(sdp_names):
        print(f"Predicting {name} ({p+1}/{len(sdp_names)})...")

        # Load coefficient files (already verified to exist above)
        a_file = _COEFF_DIR / f'a_coefs_{name}.csv'
        c_file = _COEFF_DIR / f'c_coefs_{name}.csv'

        a_coefs = pd.read_csv(a_file).values  # shape: (n_wl, 100)
        c_coefs = pd.read_csv(c_file).values.flatten()  # shape: (100,)

        # Matrix multiplication to compute all runs at once for all samples
        # Result: run_vals_all shape (n_samples, 100)
        run_vals_all = rrs_residuals_d2 @ a_coefs + c_coefs

        # Take median over runs axis (axis=1)
        median_run = np.median(run_vals_all, axis=1)

        # Enforce non-negative
        median_run[median_run < 0] = 0

        sdp[:, p] = median_run

    # Create column names from display_names mapping
    output_columns = [display_names[name] for name in sdp_names]
    return pd.DataFrame(sdp, columns=output_columns)  # type: ignore
