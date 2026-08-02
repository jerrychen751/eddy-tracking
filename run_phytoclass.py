"""
Run Phytoclass PFT decomposition on SDP pigment outputs.

Processes each eddy independently in parallel: loads its pigment Parquet file,
runs SA + NNLS, and writes a per-eddy PFT Parquet file.
Eddies are sorted largest-first so the long-running ones start early and
smaller eddies fill in the remaining workers as they free up.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from utils.config import (
    METADATA_COLS,
    load_config,
    resolve_config_file,
    resolve_output_dir,
)
from eddy_tracking.packages.phytoclass import run_phytoclass


@dataclass(frozen=True)
class PhytoclassSettings:
    """Model inputs and tuning parameters shared by all eddy workers."""

    f_matrix: pd.DataFrame
    min_max: pd.DataFrame
    seed: int | None
    cluster_threshold: float
    min_cluster_size: int
    n_iter: int


def process_eddy(
    pigment_path: Path,
    out_path: Path,
    settings: PhytoclassSettings,
) -> str | None:
    """Write one eddy's PFT output, or return ``None`` when skipped."""
    if out_path.exists():
        return None

    pigments = pd.read_parquet(pigment_path)
    if pigments.empty:
        return None

    metadata = pigments[METADATA_COLS].copy()
    pigment_columns = [
        column for column in pigments.columns if column not in METADATA_COLS
    ]

    pft_abundances = run_phytoclass(
        pigments[pigment_columns],
        f_matrix=settings.f_matrix,
        min_max=settings.min_max,
        seed=settings.seed,
        cluster_threshold=settings.cluster_threshold,
        min_cluster_size=settings.min_cluster_size,
        n_iter=settings.n_iter,
        n_jobs=1,  # The outer pool owns eddy-level parallelism.
    )

    for col_idx, column in enumerate(METADATA_COLS):
        pft_abundances.insert(col_idx, column, metadata[column].values)

    pft_abundances.to_parquet(out_path, index=False)

    n_dates = pigments["date"].nunique()
    return (
        f"output_file: {out_path.name}\n"
        f"pixels_written: {len(pigments)}\n"
        f"dates: {n_dates}"
    )


def process_polarity(
    experiment: str,
    polarity: str,
    max_workers: int,
    settings: PhytoclassSettings,
) -> int:
    """Process one polarity in parallel and return the number of files written."""
    pigment_dir = resolve_output_dir(experiment, "pigments", polarity)
    out_dir = resolve_output_dir(experiment, "pft", polarity)

    pigment_files = list(pigment_dir.glob("eddy_*_pigments.parquet"))
    if not pigment_files:
        print(
            f"polarity: {polarity}\n"
            "status: skipped\n"
            "reason: no_pigment_files\n"
            f"pigment_dir: {pigment_dir}"
        )
        return 0

    pigment_files.sort(key=lambda path: path.stat().st_size, reverse=True)
    print(
        f"polarity: {polarity}\n"
        f"eddies: {len(pigment_files)}\n"
        f"workers: {max_workers}"
    )

    n_written = 0
    n_failed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for pigment_path in pigment_files:
            out_path = out_dir / pigment_path.name.replace(
                "_pigments.parquet", "_pfts.parquet"
            )
            future = executor.submit(process_eddy, pigment_path, out_path, settings)
            futures[future] = pigment_path.stem
        for future in as_completed(futures):
            eddy_name = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                n_failed += 1
                print(
                    "status: failed\n"
                    f"eddy: {eddy_name}\n"
                    f"error: {exc}"
                )
                continue
            if result:
                print(result)
                n_written += 1

    if n_failed:
        raise RuntimeError(
            f"[{polarity}] {n_failed}/{len(pigment_files)} eddies failed"
        )
    return n_written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    return parser.parse_args()


def main(experiment: str | None = None) -> None:
    """Run PFT decomposition for both polarities and write missing outputs."""
    if experiment is None:
        experiment = _parse_args().experiment

    cfg = load_config(experiment)
    model_cfg = cfg["phytoclass"]
    settings = PhytoclassSettings(
        f_matrix=pd.read_csv(
            resolve_config_file(experiment, model_cfg["f_matrix"]),
            index_col="class",
        ),
        min_max=pd.read_csv(resolve_config_file(experiment, model_cfg["min_max"])),
        seed=model_cfg["seed"],
        cluster_threshold=model_cfg["cluster_threshold"],
        min_cluster_size=model_cfg["min_cluster_size"],
        n_iter=model_cfg["n_iter"],
    )
    max_workers = min(os.cpu_count() or 1, model_cfg["max_workers"])
    n_written = 0
    for polarity in ("cyclone", "anticyclone"):
        n_written += process_polarity(experiment, polarity, max_workers, settings)

    print(
        "status: complete\n"
        f"pft_files_written: {n_written}"
    )


if __name__ == "__main__":
    main()
