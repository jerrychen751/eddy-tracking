"""
Run the SDP pigment model on collocated PACE Rrs observations.

For each per-eddy Rrs Parquet file produced by collocate_pace.py, preprocesses
the spectra (interpolate to 1nm, smooth, trim), samples SST/SSS at
each pixel location, runs the Kramer et al. (2022) SDP model, and
writes per-eddy pigment CSVs.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import load_config, resolve_data_dir, resolve_output_dir, METADATA_COLS
from utils.sdp.ancillary import load_sst_dataset, load_sss_dataset, sample_ancillary
from utils.sdp.preprocessing import preprocess_rrs_batch
from utils.sdp import run_sdp

parser = argparse.ArgumentParser()
parser.add_argument("experiment")
args = parser.parse_args()

cfg = load_config(args.experiment, "base.yaml")

PACE_DIR = resolve_data_dir(cfg, "pace_dir")
SST_DIR = resolve_data_dir(cfg, "sst_dir")
SSS_DIR = resolve_data_dir(cfg, "sss_dir")


def process_eddy(rrs_path: Path, out_path: Path, sst_da, sss_da) -> bool:
    """
    Run preprocessing + SDP on one eddy's Rrs Parquet file.

    Separates metadata from Rrs columns, preprocesses spectra to 1nm,
    samples nearest SST/SSS, runs the SDP model, and writes a pigments
    Parquet with metadata columns preserved.

    Returns True if the file was written, False if skipped.
    """
    if out_path.exists():
        print(f"Already exists: {out_path.name}")
        return False

    df = pd.read_parquet(rrs_path)

    # Separate metadata from Rrs columns
    rrs_cols = [c for c in df.columns if c.startswith("Rrs_")]
    wavelengths = np.array([float(c.split("_")[1]) for c in rrs_cols])
    rrs_raw = df[rrs_cols].values

    # Preprocess: interpolate to 1nm (cubic spline), smooth, trim → 301 values
    wl_processed, rrs_processed = preprocess_rrs_batch(wavelengths, rrs_raw)

    # Sample SST/SSS at each pixel's (lon, lat, date)
    sst_vals, sss_vals = sample_ancillary(
        sst_da, sss_da,
        lons=df["pixel_lon"].values,
        lats=df["pixel_lat"].values,
        times=pd.to_datetime(df["date"]).values,
    )

    # Drop rows where SST or SSS is NaN (land mask or temporal gaps in
    # 8-day composites). The GSM physics model needs both values to
    # compute seawater backscattering.
    valid = np.isfinite(sst_vals) & np.isfinite(sss_vals)
    n_dropped = (~valid).sum()
    if n_dropped > 0:
        print(f"Dropped {n_dropped}/{len(valid)} pixels with missing SST/SSS")

    if valid.sum() == 0:
        print(f"No valid pixels after SST/SSS filtering, skipping")
        return False

    df = df[valid].reset_index(drop=True)
    rrs_processed = rrs_processed[valid]
    sst_vals = sst_vals[valid]
    sss_vals = sss_vals[valid]

    # Build the integer-wavelength DataFrame that run_sdp expects
    wl_int = wl_processed.astype(int)
    rrs_df = pd.DataFrame(rrs_processed, columns=wl_int)

    pigments_df = run_sdp(
        rrs=rrs_df,
        wl=wl_processed,
        sst=sst_vals,
        sss=sss_vals,
    )

    # Attach metadata columns from the original Rrs CSV
    for i, col in enumerate(METADATA_COLS):
        pigments_df.insert(i, col, df[col].values)

    pigments_df.to_parquet(out_path, index=False)

    n_dates = df["date"].nunique()
    print(f"Wrote {out_path.name}: {len(pigments_df)} pixels, {n_dates} dates")
    return True


def main():
    # Load SST/SSS grids once — these cover the full date range
    print("Loading SST/SSS grids...")
    sst_da = load_sst_dataset(SST_DIR)
    sss_da = load_sss_dataset(SSS_DIR)

    n_written = 0
    for polarity in ("cyclone", "anticyclone"):
        rrs_dir = resolve_output_dir(args.experiment, "collocate_pace", polarity)
        out_dir = resolve_output_dir(args.experiment, "pigments", polarity)

        rrs_files = sorted(rrs_dir.glob("eddy_*_rrs.parquet"))
        if not rrs_files:
            print(f"[{polarity}] No Rrs files found in {rrs_dir}")
            continue

        print(f"[{polarity}] Processing {len(rrs_files)} eddies...")
        for fp in rrs_files:
            out_path = out_dir / fp.name.replace("_rrs.parquet", "_pigments.parquet")
            if process_eddy(fp, out_path, sst_da, sss_da):
                n_written += 1

    print(f"Done. {n_written} pigment files written.")


if __name__ == "__main__":
    main()
