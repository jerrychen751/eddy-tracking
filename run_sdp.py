"""
Run the SDP pigment model on collocated PACE Rrs observations.

For each per-eddy Rrs Parquet file, preprocesses the spectra, samples SST/SSS,
runs the Kramer et al. (2022) model, and writes a pigment Parquet file.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import (
    METADATA_COLS,
    load_config,
    resolve_data_dir,
    resolve_output_dir,
)
from eddy_tracking.packages.sdp import run_sdp
from eddy_tracking.packages.sdp.ancillary import sample_ancillary
from eddy_tracking.packages.sdp.preprocessing import preprocess_rrs_batch
from eddy_tracking.preprocess.sss import read_multiple_sss
from eddy_tracking.preprocess.sst import read_multiple_sst


def process_eddy(
    rrs_path: Path,
    out_path: Path,
    sst_df: pd.DataFrame,
    sss_df: pd.DataFrame,
) -> bool:
    """
    Write pigments for one eddy unless output exists or ancillary data is absent.

    Returns ``True`` only when a new Parquet file is written.
    """
    if out_path.exists():
        print(f"Already exists: {out_path.name}")
        return False

    observations = pd.read_parquet(rrs_path)

    rrs_columns = [column for column in observations if column.startswith("Rrs_")]
    wavelengths = np.array(
        [float(column.split("_")[1]) for column in rrs_columns]
    )
    raw_rrs = observations[rrs_columns].values

    processed_wavelengths, processed_rrs = preprocess_rrs_batch(
        wavelengths, raw_rrs
    )

    sst_values, sss_values = sample_ancillary(
        sst_df,
        sss_df,
        lons=observations["pixel_lon"].values,
        lats=observations["pixel_lat"].values,
        times=pd.to_datetime(observations["date"]).values,
    )

    # The GSM physics model needs both ancillary values for backscattering.
    valid_pixels = np.isfinite(sst_values) & np.isfinite(sss_values)
    n_dropped = int((~valid_pixels).sum())
    if n_dropped:
        print(
            f"Dropped {n_dropped}/{len(valid_pixels)} pixels with missing SST/SSS"
        )

    if not valid_pixels.any():
        print("No valid pixels after SST/SSS filtering, skipping")
        return False

    observations = observations[valid_pixels].reset_index(drop=True)
    processed_rrs = processed_rrs[valid_pixels]
    sst_values = sst_values[valid_pixels]
    sss_values = sss_values[valid_pixels]

    integer_wavelengths = processed_wavelengths.astype(int)
    rrs_frame = pd.DataFrame(processed_rrs, columns=integer_wavelengths)

    pigments = run_sdp(
        rrs=rrs_frame,
        wl=processed_wavelengths,
        sst=sst_values,
        sss=sss_values,
    )

    for col_idx, column in enumerate(METADATA_COLS):
        pigments.insert(col_idx, column, observations[column].values)

    pigments.to_parquet(out_path, index=False)

    n_dates = observations["date"].nunique()
    print(f"Wrote {out_path.name}: {len(pigments)} pixels, {n_dates} dates")
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    return parser.parse_args()


def main(experiment: str | None = None) -> None:
    """Process all collocated eddies and write missing pigment Parquet files."""
    if experiment is None:
        experiment = _parse_args().experiment

    cfg = load_config(experiment)
    sst_dir = resolve_data_dir(cfg, "sst_dir")
    sss_dir = resolve_data_dir(cfg, "sss_dir")

    print("Loading SST/SSS grids...")
    sst_df = read_multiple_sst(sst_dir)
    sss_df = read_multiple_sss(sss_dir)

    n_written = 0
    for polarity in ("cyclone", "anticyclone"):
        rrs_dir = resolve_output_dir(experiment, "collocate_pace", polarity)
        out_dir = resolve_output_dir(experiment, "pigments", polarity)

        rrs_files = sorted(rrs_dir.glob("eddy_*_rrs.parquet"))
        if not rrs_files:
            print(f"[{polarity}] No Rrs files found in {rrs_dir}")
            continue

        print(f"[{polarity}] Processing {len(rrs_files)} eddies...")
        for rrs_path in rrs_files:
            out_path = out_dir / rrs_path.name.replace(
                "_rrs.parquet", "_pigments.parquet"
            )
            if process_eddy(rrs_path, out_path, sst_df, sss_df):
                n_written += 1

    print(f"Done. {n_written} pigment files written.")


if __name__ == "__main__":
    main()
