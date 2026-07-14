"""
Visualize the SDP preprocessing chain on a real collocated PACE pixel.

Produces a 4-panel figure: raw Rrs + GSM model fit, residual (observed - modeled),
first spectral derivative of residual, second spectral derivative of residual.
Makes the case for 2nd derivative feature engineering visually obvious.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = "gulf_stream_20241001_20250701"
OUTPUT_PATH = REPO_ROOT / "visuals" / "sdp_derivative_cascade.png"


def main() -> None:
    """Render the SDP derivative cascade and write it to ``OUTPUT_PATH``."""
    repo_path = str(REPO_ROOT)
    sys.path.insert(0, repo_path)
    try:
        from utils.config import load_config, resolve_data_dir, resolve_output_dir
        from utils.sdp.ancillary import (
            load_sss_dataset,
            load_sst_dataset,
            sample_ancillary,
        )
        from utils.sdp.physics import get_rrs_residuals
        from utils.sdp.preprocessing import preprocess_rrs_batch
    finally:
        sys.path.remove(repo_path)

    cfg = load_config(EXPERIMENT)
    sst_dir = resolve_data_dir(cfg, "sst_dir")
    sss_dir = resolve_data_dir(cfg, "sss_dir")

    # Pool every collocated cyclone pixel across every eddy + date. The goal is
    # a high-SNR demonstration of the derivative cascade, not a per-eddy result.
    rrs_dir = resolve_output_dir(EXPERIMENT, "collocate_pace", "cyclone")
    rrs_files = sorted(rrs_dir.glob("eddy_*_rrs.parquet"))
    df = pd.concat([pd.read_parquet(fp) for fp in rrs_files], ignore_index=True)

    rrs_cols = [c for c in df.columns if c.startswith("Rrs_")]
    wavelengths = np.array([float(c.split("_")[1]) for c in rrs_cols])
    rrs_native = df[rrs_cols].values.mean(axis=0, keepdims=True)
    print(f"Pooled {len(df)} pixels across {len(rrs_files)} cyclones, {df['date'].nunique()} unique dates")

    wl_proc, rrs_proc = preprocess_rrs_batch(wavelengths, rrs_native)

    print("Loading SST/SSS grids...")
    sst_da = load_sst_dataset(sst_dir)
    sss_da = load_sss_dataset(sss_dir)

    # Sample SST/SSS at the eddy-mean location and the median date, since
    # we've averaged across the whole eddy's coverage period.
    center_lon = float(df["pixel_lon"].mean())
    center_lat = float(df["pixel_lat"].mean())
    median_date = df["date"].sort_values().iloc[len(df) // 2]
    sst, sss = sample_ancillary(
        sst_da, sss_da,
        lons=np.array([center_lon]),
        lats=np.array([center_lat]),
        times=np.array([pd.to_datetime(median_date)]),
    )
    print(f"Location: ({center_lat:.2f} N, {center_lon:.2f} E), SST={sst[0]:.2f} C, SSS={sss[0]:.2f} PSU")

    wl_int = wl_proc.astype(int)
    rrs_df = pd.DataFrame(rrs_proc, columns=wl_int)
    _, RrsD = get_rrs_residuals(rrs_df, sst, sss, wl_proc)

    residual = RrsD.values[:, 0]
    rrs_measured = rrs_proc[0]
    rrs_modeled = rrs_measured - residual

    d1 = np.diff(residual, 1)
    d2 = np.diff(residual, 2)
    wl_d1 = wl_proc[:-1] + 0.5
    wl_d2 = wl_proc[1:-1]

    # PACE has a wavelength gap covering the O2 A-band. Cubic-spline interpolation
    # across it produces a non-physical bump that the 2nd derivative amplifies.
    # Shade it so viewers see it's an instrument artifact, not a pigment feature.
    O2_GAP = (588, 613)

    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)

    def annotate(ax):
        ax.axvspan(*O2_GAP, color="#f0f0f0", zorder=0)
        ax.axvline(440, color="#2a7f2a", linestyle="--", linewidth=0.7, zorder=1, alpha=0.6)
        ax.axvline(675, color="#2a7f2a", linestyle="--", linewidth=0.7, zorder=1, alpha=0.6)
        ax.axvline(470, color="#b08020", linestyle="--", linewidth=0.7, zorder=1, alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax = axes[0]
    ax.plot(wl_proc, rrs_measured, color="#1f3b70", linewidth=1.6, label="Measured Rrs")
    ax.plot(wl_proc, rrs_modeled, color="#c23b22", linewidth=1.2, linestyle="--", label="GSM-modeled Rrs")
    annotate(ax)
    ax.set_ylabel("Rrs (sr$^{-1}$)")
    ax.set_title("(a) Measured Rrs vs. GSM forward-model fit", loc="left", fontsize=11)
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    ax = axes[1]
    ax.plot(wl_proc, residual, color="#1f3b70", linewidth=1.4)
    ax.axhline(0, color="#888888", linewidth=0.6)
    annotate(ax)
    ax.set_ylabel("Residual (sr$^{-1}$)")
    ax.set_title("(b) Residual = Rrs$_{measured}$ − Rrs$_{GSM}$ - baseline dominates, features are buried", loc="left", fontsize=11)

    ax = axes[2]
    ax.plot(wl_d1, d1, color="#1f3b70", linewidth=1.4)
    ax.axhline(0, color="#888888", linewidth=0.6)
    annotate(ax)
    ax.set_ylabel(r"d(Residual) / d$\lambda$")
    ax.set_title("(c) 1st derivative - constant offset killed, broad tilt survives as a DC-ish level", loc="left", fontsize=11)

    ax = axes[3]
    ax.plot(wl_d2, d2, color="#c23b22", linewidth=1.6)
    ax.axhline(0, color="#888888", linewidth=0.6)
    annotate(ax)
    ax.set_ylabel(r"d$^2$(Residual) / d$\lambda^2$")
    ax.set_title("(d) 2nd derivative - what SDP feeds to the linear regression", loc="left", fontsize=11)
    ax.set_xlabel("Wavelength (nm)")

    for ax in axes:
        ax.set_xlim(410, 695)

    fig.suptitle(
        f"SDP preprocessing cascade - all cyclone pixels pooled (N={len(df):,})\n"
        "green dashes: chl-a Soret (440) & Qy (675) bands   ·   tan dashes: chl-b (470)   ·   gray band: PACE O$_2$ gap",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=160)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
