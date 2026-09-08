"""
Run the SDP pigment model on collocated PACE Rrs observations.

For each per-eddy Rrs Parquet file, preprocesses the spectra, samples SST/SSS, runs the Kramer et al. (2022) model, and writes a pigment Parquet file.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import cast

import pandas as pd

from eddy_tracking.config import (
    METADATA_COLS,
    load_config,
    resolve_data_dir,
    resolve_output_dir,
)
from eddy_tracking.packages.sdp import run_sdp_on_pace_l3
from eddy_tracking.preprocess.ancillary import read_ancillary_grids


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

    pigments, observations = run_sdp_on_pace_l3(pd.read_parquet(rrs_path), sst_df, sss_df)
    if pigments.empty:
        print(
            "status: skipped\n"
            "reason: no_valid_pixels"
        )
        return False

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
    _worker_ancillary = read_ancillary_grids(sst_dir, sss_dir)


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
        sst_df, sss_df = read_ancillary_grids(sst_dir, sss_dir)
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
