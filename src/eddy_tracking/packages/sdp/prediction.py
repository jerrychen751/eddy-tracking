"""SDP model prediction functions."""

from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from eddy_tracking.packages.sdp.ancillary import sample_ancillary
from eddy_tracking.packages.sdp.physics import (
    GSMInversionError,
    get_rrs_residuals,
)
from eddy_tracking.packages.sdp.preprocessing import preprocess_rrs_batch

_COEFF_DIR = Path(__file__).resolve().parent / "coefficients"
_PIGMENT_DISPLAY_NAMES = {
    "Tchla": "T chla",
    "Zea": "Zea",
    "DVchla": "DV chla",
    "ButFuco": "ButFuco",
    "HexFuco": "HexFuco",
    "Allo": "Allo",
    "MVchlb": "MV chlb",
    "Neo": "Neo",
    "Viola": "Viola",
    "Fuco": "Fuco",
    "Chlc12": "chl c1+c2",
    "Chlc3": "chl c3",
    "Perid": "Perid",
}
SDP_PIGMENT_COLUMNS = tuple(_PIGMENT_DISPLAY_NAMES.values())


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
        rrs: Above-surface remote-sensing reflectance in sr^-1, with integer wavelength columns (400..700), each row is a spectrum.
        wl: 1D wavelength array in nm matching the DataFrame columns.
        sst: Sea surface temperature in Celsius, one value per spectrum.
        sss: Sea surface salinity in PSU, one value per spectrum.
        pigments: List of pigment names to predict, or None for all 13.

    Available pigments: Tchla, Zea, DVchla, ButFuco, HexFuco, Allo, MVchlb, Neo, Viola, Fuco, Chlc12, Chlc3, Perid.

    Returns:
        DataFrame of concentrations in ug/L, one row per input spectrum and one column per requested pigment, labeled with the display name (Chlc12 becomes "chl c1+c2").
    """

    all_available_pigments = list(_PIGMENT_DISPLAY_NAMES)

    if pigments is None:
        sdp_names = all_available_pigments.copy()
    else:
        invalid_pigments = [p for p in pigments if p not in all_available_pigments]
        if invalid_pigments:
            raise ValueError(
                f"Invalid pigment names: {invalid_pigments}. "
                f"Available pigments: {all_available_pigments}"
            )
        sdp_names = pigments.copy()

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

    rrs_residuals = get_rrs_residuals(rrs, sst, sss, wl)[1]  # element 1 is RrsD, the above-surface residual
    rrs_residuals_d2 = np.diff(rrs_residuals, 2, axis=0).T  # (n_wavelengths, n_samples) differenced twice over wavelength -> (n_wavelengths - 2, n_samples) -> (n_samples, n_wavelengths - 2)

    sdp = np.zeros((rrs_residuals_d2.shape[0], len(sdp_names)))

    for p, name in enumerate(sdp_names):
        print(
            "status: predicting\n"
            f"pigment: {name}\n"
            f"pigment_number: {p + 1}\n"
            f"total_pigments: {len(sdp_names)}"
        )

        a_coefs, c_coefs = _load_coefficients(name)

        # (n_samples, n_wavelengths - 2) @ (n_wavelengths - 2, n_runs) -> (n_samples, n_runs)
        run_vals_all = rrs_residuals_d2 @ a_coefs + c_coefs

        median_run = np.median(run_vals_all, axis=1)

        median_run[median_run < 0] = 0

        sdp[:, p] = median_run

    output_columns = [_PIGMENT_DISPLAY_NAMES[name] for name in sdp_names]
    return pd.DataFrame(sdp, columns=output_columns)  # type: ignore


@cache
def _load_coefficients(pigment: str) -> tuple[np.ndarray, np.ndarray]:
    a_file = _COEFF_DIR / f"a_coefs_{pigment}.csv"
    c_file = _COEFF_DIR / f"c_coefs_{pigment}.csv"
    a_coefs = pd.read_csv(a_file).values  # (n_wavelengths - 2, n_runs), 299 x 100 on the 1 nm 400-700 grid
    c_coefs = pd.read_csv(c_file).values.flatten()  # (n_runs, 1) -> (n_runs,)
    return a_coefs, c_coefs


def run_sdp_on_pace_l2(
    pace_pixels: pd.DataFrame,
    sst_grid: pd.DataFrame,
    sss_grid: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict 13 pigments for each PACE Level-2 pixel.

    pace_pixels needs `datetime`, `latitude`, and `longitude` columns, and `wavelengths_nm` and `rrs_columns` in its `attrs`. sst_grid and sss_grid are the nearest-neighbor sources that sample_ancillary describes.

    The result preserves all input rows and columns. It adds `sst` in Celsius, `sss` in PSU, one column per pigment in ug/L, and `sdp_status`.

    `sdp_status` is "predicted" where the row carries pigment values, "invalid_rrs" where the spectrum has fewer than 2 finite Rrs values or preprocessing returns a non-finite value, "missing_ancillary" where the SST or SSS grid has no value at the pixel, and "gsm_nonconvergent" where the GSM inversion hits the scipy.optimize.fmin limits. The last three keep NaN pigment values.
    """
    wavelengths_nm, rrs_columns = _read_pace_spectral_schema(pace_pixels)
    required_columns = ("datetime", "latitude", "longitude")
    missing_columns = [
        column for column in required_columns if column not in pace_pixels
    ]
    if missing_columns:
        raise KeyError(f"pace_pixels missing required columns: {missing_columns}")

    result = pace_pixels.copy()
    pigment_values = np.full(
        (len(pace_pixels), len(SDP_PIGMENT_COLUMNS)),
        np.nan,
    )
    status = np.full(len(pace_pixels), "invalid_rrs", dtype=object)

    if pace_pixels.empty:
        return _add_sdp_columns(
            result,
            np.array([], dtype=float),
            np.array([], dtype=float),
            status,
            pigment_values,
            pace_pixels.attrs,
        )

    observation_times = (
        pd.to_datetime(pace_pixels["datetime"], utc=True)
        .dt.tz_localize(None)
        .to_numpy()
    )
    sst_values, sss_values = sample_ancillary(
        sst_grid,
        sss_grid,
        lons=pace_pixels["longitude"].to_numpy(dtype=float),
        lats=pace_pixels["latitude"].to_numpy(dtype=float),
        times=observation_times,
    )

    raw_rrs = pace_pixels.loc[:, rrs_columns].to_numpy(dtype=float)
    preprocessable = np.isfinite(raw_rrs).sum(axis=1) >= 2  # CubicSpline in preprocess_rrs_spectrum needs 2 finite points
    preprocessable_positions = np.flatnonzero(preprocessable)

    if len(preprocessable_positions):
        processed_wavelengths, processed_rrs = preprocess_rrs_batch(
            wavelengths_nm,
            raw_rrs[preprocessable],
        )
        finite_processed = np.isfinite(processed_rrs).all(axis=1)
        valid_spectrum_positions = preprocessable_positions[finite_processed]
        valid_spectra = processed_rrs[finite_processed]

        ancillary_valid = (
            np.isfinite(sst_values[valid_spectrum_positions])
            & np.isfinite(sss_values[valid_spectrum_positions])
        )
        missing_ancillary_positions = valid_spectrum_positions[~ancillary_valid]
        status[missing_ancillary_positions] = "missing_ancillary"

        prediction_positions = valid_spectrum_positions[ancillary_valid]
        if len(prediction_positions):
            predicted, nonconvergent = _predict_pace_spectra(
                valid_spectra[ancillary_valid],
                processed_wavelengths,
                sst_values[prediction_positions],
                sss_values[prediction_positions],
            )
            pigment_values[prediction_positions] = predicted
            status[prediction_positions] = "predicted"
            status[prediction_positions[nonconvergent]] = "gsm_nonconvergent"

    return _add_sdp_columns(
        result,
        sst_values,
        sss_values,
        status,
        pigment_values,
        pace_pixels.attrs,
    )


def _read_pace_spectral_schema(
    pace_pixels: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    wavelengths = pace_pixels.attrs.get("wavelengths_nm")
    rrs_columns = pace_pixels.attrs.get("rrs_columns")
    if wavelengths is None or rrs_columns is None:
        raise ValueError(
            "pace_pixels attrs must contain wavelengths_nm and rrs_columns"
        )

    wavelengths_nm = np.asarray(wavelengths, dtype=float)
    columns = list(rrs_columns)
    if wavelengths_nm.ndim != 1:
        raise ValueError("pace_pixels wavelengths_nm must be one-dimensional")
    if len(wavelengths_nm) != len(columns):
        raise ValueError(
            "pace_pixels wavelengths_nm and rrs_columns must have equal lengths"
        )

    missing_columns = [column for column in columns if column not in pace_pixels]
    if missing_columns:
        raise KeyError(f"pace_pixels missing Rrs columns: {missing_columns}")
    return wavelengths_nm, columns


def _predict_pace_spectra(
    rrs: np.ndarray,
    wavelengths_nm: np.ndarray,
    sst: np.ndarray,
    sss: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.full((len(rrs), len(SDP_PIGMENT_COLUMNS)), np.nan)
    nonconvergent = np.zeros(len(rrs), dtype=bool)

    def predict_positions(positions: np.ndarray) -> None:
        spectra = pd.DataFrame(
            rrs[positions],
            columns=wavelengths_nm.astype(int),
        )
        try:
            predicted = run_sdp(
                rrs=spectra,
                wl=wavelengths_nm,
                sst=sst[positions],
                sss=sss[positions],
            )
        except GSMInversionError:
            if len(positions) == 1:
                nonconvergent[positions[0]] = True
                return

            midpoint = len(positions) // 2
            predict_positions(positions[:midpoint])
            predict_positions(positions[midpoint:])
            return

        predictions[positions] = predicted.loc[
            :,
            list(SDP_PIGMENT_COLUMNS),
        ].to_numpy()

    predict_positions(np.arange(len(rrs)))
    return predictions, nonconvergent


def _add_sdp_columns(
    result: pd.DataFrame,
    sst: np.ndarray,
    sss: np.ndarray,
    status: np.ndarray,
    pigments: np.ndarray,
    attrs: dict,
) -> pd.DataFrame:
    result["sst"] = sst
    result["sss"] = sss
    result["sdp_status"] = status
    for column, values in zip(
        SDP_PIGMENT_COLUMNS,
        pigments.T,  # (n_rows, n_pigments) -> (n_pigments, n_rows)
        strict=True,
    ):
        result[column] = values
    result.attrs = attrs.copy()
    return result
