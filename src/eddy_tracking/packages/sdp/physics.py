"""
Seawater optical physics for the Kramer et al. (2022) SDP model.

Implements the GSM bio-optical inversion model (Gordon et al. 1988) and the Zhang et al. (2009) seawater scattering parameterization used to compute Rrs residuals for pigment prediction.
"""

import numpy as np
from scipy.optimize import fmin
import pandas as pd
from pathlib import Path


class GSMInversionError(RuntimeError):
    pass


def RInw(
    lambda_: int | float | np.ndarray,
    Tc: int | float,
    S: int | float,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Return (nsw, dnswds): the absolute seawater refractive index (Quan & Fry 1994 scaled by the Ciddor 1996 air index) and its derivative with respect to salinity, for lambda_ in nm, Tc in Celsius, and S in PSU."""
    n_air = (
        1.0 + (5792105.0 / (238.0185 - 1 / (lambda_ / 1e3) ** 2)
        + 167917.0 / (57.362 - 1 / (lambda_ / 1e3) ** 2)) / 1e8
    )
    n0 = 1.31405
    n1 = 1.779e-4
    n2 = -1.05e-6
    n3 = 1.6e-8
    n4 = -2.02e-6
    n5 = 15.868
    n6 = 0.01155
    n7 = -0.00423
    n8 = -4382
    n9 = 1.1455e6
    nsw = (
        n0 + (n1 + n2 * Tc + n3 * Tc ** 2) * S + n4 * Tc ** 2
        + (n5 + n6 * S + n7 * Tc) / lambda_ + n8 / lambda_ ** 2 + n9
        / lambda_ ** 3
    )
    nsw = nsw * n_air
    dnswds = (n1 + n2 * Tc + n3 * Tc ** 2 + n6 / lambda_) * n_air

    return nsw, dnswds


def BetaT(Tc: int | float, S: int | float) -> float:
    """Seawater isothermal compressibility in Pa^-1 (Millero 1980), for Tc in Celsius and S in PSU."""
    kw = (
        19652.21 + 148.4206 * Tc - 2.327105 * Tc ** 2 + 1.360477e-2
        * Tc ** 3 - 5.155288e-5 * Tc ** 4
    )
    Btw_cal = 1 / kw
    a0 = 54.6746 - 0.603459 * Tc + 1.09987e-2 * Tc ** 2-6.167e-5 * Tc ** 3
    b0 = 7.944e-2 + 1.6483e-2 * Tc - 5.3009e-4 * Tc ** 2

    Ks = kw + a0 * S + b0 * S ** 1.5
    IsoComp = 1 / Ks * 1e-5

    return IsoComp


def rho_sw(Tc: int | float, S: int | float) -> float:
    """Seawater density in kg/m^3 (UNESCO 1981), for Tc in Celsius and S in PSU."""
    a0 = 8.24493e-1
    a1 = -4.0899e-3
    a2 = 7.6438e-5
    a3 = -8.2467e-7
    a4 = 5.3875e-9
    a5 = -5.72466e-3
    a6 = 1.0227e-4
    a7 = -1.6546e-6
    a8 = 4.8314e-4
    b0 = 999.842594
    b1 = 6.793952e-2
    b2 = -9.09529e-3
    b3 = 1.001685e-4
    b4 = -1.120083e-6
    b5 = 6.536332e-9

    density_w = b0 + b1 * Tc + b2 * Tc ** 2 + b3 * Tc ** 3 + b4 * Tc ** 4 + b5 * Tc ** 5
    density_sw = (
        density_w + ((a0 + a1 * Tc + a2 * Tc ** 2 + a3 * Tc ** 3 + a4 * Tc ** 4) * S
        + (a5 + a6 * Tc + a7 * Tc ** 2) * S ** 1.5 + a8 * S ** 2)
    )
    return density_sw


def dlnasw_ds(Tc: int | float, S: int | float) -> float:
    """Partial derivative of ln(water activity) with respect to salinity, per PSU, for Tc in Celsius and S in PSU. Millero & Leung (1976) Table 19, reproduced from their Eq. 14, 22, 23, 88, and 107, then fitted to a polynomial."""

    dlnawds = (
        (-5.58651e-4 + 2.40452e-7 * Tc - 3.12165e-9 * Tc ** 2 + 2.40808e-11 * Tc ** 3)
        + 1.5 * (1.79613e-5 - 9.9422e-8 * Tc + 2.08919e-9 * Tc ** 2 - 1.39872e-11 * Tc ** 3) * S ** 0.5
        + 2 * (-2.31065e-6 - 1.37674e-9 * Tc - 1.93316e-11 * Tc ** 2) * S
    )
    return dlnawds


def PMH(n_wat: float | np.ndarray) -> float | np.ndarray:
    """Return the dimensionless PMH refractive-index density derivative for the absolute refractive index n_wat."""
    n_wat2 = n_wat ** 2
    n_density_derivative = (
        (n_wat2 - 1) * (1 + 2 / 3 * (n_wat2 + 2)
        * (n_wat / 3 - 1 / 3 / n_wat) ** 2)
    )
    return n_density_derivative

def betasw124_ZHH2009(
    lambda_: np.ndarray,
    S: int | float,
    Tc: int | float,
    delta: float = 0.039
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Seawater scattering at 124 degrees (Zhang et al. 2009).

    lambda_ is in nm, S in PSU, and Tc in Celsius. S and Tc must be scalars. delta defaults to 0.039 (Farinato & Roswell 1976).

    Returns: (betasw124, bsw, beta90sw, theta). betasw124 and beta90sw are volume scattering in m^-1 sr^-1, bsw is total scattering in m^-1, theta is the scattering angle in degrees from 0 to 180.
    """

    for param in [S, Tc]:
        if isinstance(param, np.ndarray):
            raise NotImplementedError("S and Tc must be scalar.")

    Na = 6.0221417930e23  # Avogadro's constant
    Kbz = 1.3806503e-23  # Boltzmann constant
    Tk = Tc + 273.15
    M0 = 18e-3  # Molecular weight of water in kg/mol

    theta = np.linspace(0.0, 180.0, 18_001)

    rad = theta * np.pi/180

    # nsw is the absolute refractive index of seawater, dnds its partial derivative with respect to salinity.
    nsw, dnds = RInw(lambda_, Tc, S)

    # The compressibility fit carries an error of about +/-0.004e-6 bar^-1 (Lepple & Millero 1971, pages 10-11).
    IsoComp = BetaT(Tc, S)

    density_sw = rho_sw(Tc, S)

    dlnawds = dlnasw_ds(Tc, S)

    DFRI = PMH(nsw)

    # volume scattering at 90 degree due to the density fluctuation
    beta_df = (
        np.pi * np.pi / 2 * ((lambda_ * 1e-9) ** (-4)) * Kbz * Tk * IsoComp * DFRI ** 2
        * (6 + 6 * delta) / (6 - 7 * delta)
    )

    # volume scattering at 90 degree due to the concentration fluctuation
    flu_con = S * M0 * dnds ** 2 / density_sw / (-dlnawds) / Na
    beta_cf = (
        2 * np.pi * np.pi * ((lambda_ * 1e-9) ** (-4)) * nsw ** 2
        * (flu_con) * (6 + 6 * delta)/(6 - 7 * delta)
    )

    beta90sw = beta_df + beta_cf
    bsw = 8 * np.pi/3 * beta90sw * (2 + delta) / (1 + delta)

    rad124 = int(np.searchsorted(theta, 124.0))

    betasw124 = (
        beta90sw * (1 + ((np.cos(rad[rad124])) ** 2) * (1 - delta) / (1 + delta))
    )

    return betasw124, bsw, beta90sw, theta

def gsm_cost(
    IOPs: np.ndarray,
    rrs: np.ndarray,
    aw: np.ndarray,
    bbw: np.ndarray,
    bbpstar: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    admstar: np.ndarray
) -> float:
    """Return the squared residual error for one GSM parameter vector IOPs = (chl, acdm at 443 nm in m^-1, bbp at 443 nm in m^-1) against the below-surface rrs of one spectrum."""
    g = np.array([0.0949, 0.0794])  # Constants from Gordon et al., 1988

    aph = A * IOPs[0]**B
    a = aw + aph + (IOPs[1] * admstar)
    bb = bbw + IOPs[2] * bbpstar
    x = bb / (a + bb)

    rrspred = (g[0] + g[1] * x) * x
    cost = np.sum((rrs - rrspred)**2)

    return cost

def gsm_invert(
    rrs: np.ndarray,
    aw: np.ndarray,
    bbw: np.ndarray,
    bbpstar: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    admstar: np.ndarray
) -> np.ndarray:
    """
    Fit GSM inherent optical properties for one below-surface rrs spectrum.

    Returns (chl, acdm at 443 nm in m^-1, bbp at 443 nm in m^-1). Raises GSMInversionError when the simplex search hits its iteration or evaluation limit.
    """

    IOPSinit = [0.15, 0.01, 0.0029]

    def cost_fn(IOPs_trial):
        return gsm_cost(IOPs_trial, rrs, aw, bbw, bbpstar, A, B, admstar)

    iops_opt, _, _, _, warnflag = fmin(
        cost_fn,
        IOPSinit,
        xtol=1e-9,
        ftol=1e-9,
        maxfun=2000,
        maxiter=2000,
        full_output=True,
        disp=False
    )

    if warnflag != 0:
        raise GSMInversionError(
            "GSM inversion failed to converge "
            f"(scipy.optimize.fmin warnflag={warnflag})"
        )

    return iops_opt

def get_rrs_residuals(
    Rrs: pd.DataFrame,
    temp: np.ndarray,
    sal: np.ndarray,
    wavelengths: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Rrs residuals (measured - modeled) using the Kramer et al. (2022) method.

    Converts the above-surface Rrs to below-surface rrs with the Lee et al. (2002) relation, inverts for IOPs with GSM (Gordon et al. 1988), then rebuilds Rrs from those IOPs.

    Args:
        Rrs: Above-surface remote-sensing reflectance in sr^-1, one row per spectrum, with integer wavelength columns that include 440, 490, and 555.
        temp: Sea surface temperature in Celsius, one value per row of Rrs.
        sal: Sea surface salinity in PSU, one value per row of Rrs.
        wavelengths: Wavelength centers in nm, matching the columns of Rrs.

    Returns:
        (rrsD, RrsD) - the below-surface and above-surface residuals, each of shape (n_wavelengths, n_samples).
    """

    required_wl = [440, 490, 555]
    missing = [w for w in required_wl if w not in Rrs.columns]
    if missing:
        raise ValueError(f"Rrs DataFrame missing required wavelength columns: {missing}")

    n = len(temp)

    # Gordon et al. (1988) needs below-surface rrs = Lu(0-)/Ed(0-), so Lee et al. (2002) converts the above-surface Rrs = Lu(0+)/Ed(0+).
    rrs = Rrs / (0.52 + 1.7 * Rrs)

    # Total absorption is seawater absorption (asw) plus phytoplankton absorption (aph) plus CDOM and detrital matter (acdm).
    coeff_dir = Path(__file__).resolve().parent / "reference_data"

    asw = pd.read_csv(coeff_dir / 'aw_mcf16_350_700_1nm.csv', header=0)
    assert asw.iloc[50, 0] == 400, (
        f"Expected wavelength 400 at row 50 of aw_mcf16 CSV, got {asw.iloc[50, 0]}"
    )
    asw = asw.iloc[50:, 1].values  # rows 50+ = 400-700 nm

    # aph = A * chl ** B, so the inversion solves for chl.
    AB_coefs = pd.read_csv(coeff_dir / 'aph_A_B_Coeffs_Sasha_RSE_paper.csv', header=0)
    A = AB_coefs.iloc[50:, 1].values
    B = AB_coefs.iloc[50:, 2].values

    # The acdm slope uses above-surface Rrs, not below-surface rrs.
    Rrs_490 = Rrs[490].values
    Rrs_555 = Rrs[555].values

    acdm_s = -(0.01447 + 0.00033 * Rrs_490 / Rrs_555)
    acdm = np.exp(np.outer(acdm_s, wavelengths - 443))  # (n_samples,) outer (n_wavelengths,) -> (n_samples, n_wavelengths)

    # Total backscattering is seawater backscattering (bbsw) plus particle backscattering (bbp).
    bsw = []

    for i in range(n):
        _, bsw_i, _, _ = betasw124_ZHH2009(wavelengths, float(sal[i]), float(temp[i]))
        bsw.append(bsw_i)

    bsw = np.array(bsw)  # n_samples arrays of (n_wavelengths,) -> (n_samples, n_wavelengths)
    bbsw = 0.5 * bsw.T  # (n_samples, n_wavelengths) -> (n_wavelengths, n_samples)

    # The bbp slope uses below-surface rrs, not above-surface Rrs.
    rrs_440 = rrs[440].values
    rrs_555 = rrs[555].values
    bbp_s = 2.0 * (1 - 1.2 * np.exp(-0.9 * rrs_440 / rrs_555))
    bbp = (443 / wavelengths.reshape(-1, 1)) ** bbp_s  # (n_wavelengths,) -> (n_wavelengths, 1), broadcast with bbp_s (n_samples,) -> (n_wavelengths, n_samples)

    IOPs = np.empty((n, 3))  # columns are chl, acdm at 443 nm in m^-1, bbp at 443 nm in m^-1

    for i in range(n):
        rrs_i = rrs.iloc[i, :].values
        asw_t = asw
        bbsw_i = bbsw[:, i]
        bbp_i = bbp[:, i]
        A_t = A
        B_t = B
        acdm_i = acdm[i, :]

        iops_i = gsm_invert(rrs_i, asw_t, bbsw_i, bbp_i, A_t, B_t, acdm_i)
        IOPs[i, :] = iops_i

    asw_ = asw[:, np.newaxis]  # (n_wavelengths,) -> (n_wavelengths, 1)
    A_ = A[:, np.newaxis]  # (n_wavelengths,) -> (n_wavelengths, 1)
    B_ = B[:, np.newaxis]  # (n_wavelengths,) -> (n_wavelengths, 1)

    a = asw_ + (A_ * (IOPs[:, 0]**B_)) + (acdm.T * IOPs[:, 1])  # acdm.T: (n_samples, n_wavelengths) -> (n_wavelengths, n_samples), so a is (n_wavelengths, n_samples)
    bb = bbsw + bbp * IOPs[:, 2]

    rrsP = bb / (a + bb)

    g1, g2 = 0.0949, 0.0794  # Gordon et al. (1988) coefficients

    modrrs = (g1 + g2 * rrsP) * rrsP

    # Inverse of the Lee et al. (2002) relation: below-surface rrs back to above-surface Rrs.
    modRrs = (0.52 * modrrs) / (1 - 1.7 * modrrs)

    rrsD = rrs.T - modrrs  # rrs.T: (n_samples, n_wavelengths) -> (n_wavelengths, n_samples)
    RrsD = Rrs.T - modRrs  # Rrs.T: (n_samples, n_wavelengths) -> (n_wavelengths, n_samples)

    return rrsD, RrsD
