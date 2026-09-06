"""
Run the SDP pigment model on collocated PACE Rrs observations.

For each per-eddy Rrs Parquet file, preprocesses the spectra, samples SST/SSS, runs the Kramer et al. (2022) model, and writes a pigment Parquet file.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import cast

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
        print(
            f"output_file: {out_path.name}\n"
            "status: already_exists"
        )
        return False

    observations = pd.read_parquet(rrs_path)

    rrs_columns = [column for column in observations if column.startswith("Rrs_")]
    wavelengths = np.array(
        [float(column.split("_")[1]) for column in rrs_columns]
    )
    raw_rrs = observations[rrs_columns].to_numpy()

    # raw_rrs (n_pixels, n_native_wavelengths) -> processed_rrs (n_pixels, n_processed_wavelengths)
    processed_wavelengths, processed_rrs = preprocess_rrs_batch(
        wavelengths, raw_rrs
    )

    sst_values, sss_values = sample_ancillary(
        sst_df,
        sss_df,
        lons=observations["pixel_lon"].to_numpy(),
        lats=observations["pixel_lat"].to_numpy(),
        times=pd.to_datetime(observations["date"]).to_numpy(),
    )

    # The GSM physics model needs both ancillary values for backscattering.
    valid_pixels = np.isfinite(sst_values) & np.isfinite(sss_values)
    n_dropped = int((~valid_pixels).sum())
    if n_dropped:
        print(
            f"pixels_dropped: {n_dropped}\n"
            f"total_pixels: {len(valid_pixels)}\n"
            "reason: missing_sst_sss"
        )

    if not valid_pixels.any():
        print(
            "status: skipped\n"
            "reason: no_valid_pixels_after_sst_sss_filter"
        )
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
        pigments.insert(col_idx, column, observations[column].to_numpy())  # pyright: ignore[reportAttributeAccessIssue]

    pigments.to_parquet(out_path, index=False)

    n_dates = observations["date"].nunique()  # pyright: ignore[reportAttributeAccessIssue]
    print(
        f"output_file: {out_path.name}\n"
        f"pixels_written: {len(pigments)}\n"
        f"dates: {n_dates}"
    )
    return True


def load_worker_ancillary(sst_dir: Path, sss_dir: Path) -> None:
    """Load the ancillary grids once in each SDP worker process."""
    global _worker_ancillary
    _worker_ancillary = (
        read_multiple_sst(sorted(sst_dir.glob("*.nc"))),
        read_multiple_sss(sorted(sss_dir.glob("*.nc4"))),
    )


def process_eddy_in_worker(rrs_path: Path, out_path: Path) -> bool:
    """Process one eddy after load_worker_ancillary initializes this process."""
    sst_df, sss_df = _worker_ancillary
    return process_eddy(rrs_path, out_path, sst_df, sss_df)


def main(experiment: str | None = None) -> None:
    """Process all collocated eddies and write missing pigment Parquet files."""
    if experiment is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("experiment")
        experiment = cast(str, parser.parse_args().experiment)

    cfg = load_config(experiment)
    sst_dir = resolve_data_dir(cfg, "sst_dir")
    sss_dir = resolve_data_dir(cfg, "sss_dir")
    max_workers = cfg.get("run_sdp", {}).get("max_workers", 1)

    tasks: list[tuple[Path, Path]] = []
    for polarity in ("cyclone", "anticyclone"):
        rrs_dir = resolve_output_dir(experiment, "collocate_pace", polarity)
        out_dir = resolve_output_dir(experiment, "pigments", polarity)

        rrs_files = sorted(rrs_dir.glob("eddy_*_rrs.parquet"))
        if not rrs_files:
            print(
                f"polarity: {polarity}\n"
                "status: skipped\n"
                "reason: no_rrs_files\n"
                f"rrs_dir: {rrs_dir}"
            )
            continue

        print(
            f"polarity: {polarity}\n"
            "status: processing\n"
            f"eddies: {len(rrs_files)}"
        )
        for rrs_path in rrs_files:
            out_path = out_dir / rrs_path.name.replace(
                "_rrs.parquet", "_pigments.parquet"
            )
            tasks.append((rrs_path, out_path))

    print("status: loading_sst_sss_grids")
    n_written = 0
    if max_workers == 1:
        sst_df = read_multiple_sst(sorted(sst_dir.glob("*.nc")))
        sss_df = read_multiple_sss(sorted(sss_dir.glob("*.nc4")))
        for rrs_path, out_path in tasks:
            if process_eddy(rrs_path, out_path, sst_df, sss_df):
                n_written += 1
    else:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=load_worker_ancillary,
            initargs=(sst_dir, sss_dir),
        ) as executor:
            futures = [
                executor.submit(process_eddy_in_worker, rrs_path, out_path)
                for rrs_path, out_path in tasks
            ]
            for future in as_completed(futures):
                if future.result():
                    n_written += 1

    print(
        "status: complete\n"
        f"pigment_files_written: {n_written}"
    )


if __name__ == "__main__":
    main()
