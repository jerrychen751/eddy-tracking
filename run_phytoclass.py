"""
Run Phytoclass PFT decomposition on SDP pigment outputs.

Processes each eddy independently in parallel: loads its pigment CSV
(all dates, all pixels), runs SA + NNLS, and writes a per-eddy PFT CSV.
Eddies are sorted largest-first so the long-running ones start early and
smaller eddies fill in the remaining workers as they free up.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import load_config, resolve_config_file, resolve_output_dir, METADATA_COLS
from utils.phytoclass import run_phytoclass

parser = argparse.ArgumentParser()
parser.add_argument("experiment")
args = parser.parse_args()

cfg = load_config(args.experiment)

SEED = cfg["phytoclass"]["seed"]
CLUSTER_THRESHOLD = cfg["phytoclass"]["cluster_threshold"]
MIN_CLUSTER_SIZE = cfg["phytoclass"]["min_cluster_size"]
N_ITER = cfg["phytoclass"]["n_iter"]
MAX_WORKERS = cfg["phytoclass"]["max_workers"]

F_MATRIX_PATH = resolve_config_file(args.experiment, cfg["phytoclass"]["f_matrix"])
MIN_MAX_PATH = resolve_config_file(args.experiment, cfg["phytoclass"]["min_max"])
F_MATRIX = pd.read_csv(F_MATRIX_PATH, index_col="class")
MIN_MAX = pd.read_csv(MIN_MAX_PATH)


def process_eddy(pigment_path: Path, out_path: Path) -> str | None:
    """
    Run phytoclass on a single eddy's pigment file and write PFT output.

    Returns a status string, or None if skipped (already exists or empty).
    """
    if out_path.exists():
        return None

    df = pd.read_parquet(pigment_path)
    if df.empty:
        return None

    meta = df[METADATA_COLS].copy()
    pigment_cols = [c for c in df.columns if c not in METADATA_COLS]

    pfts = run_phytoclass(
        df[pigment_cols],
        f_matrix=F_MATRIX,
        min_max=MIN_MAX,
        seed=SEED,
        cluster_threshold=CLUSTER_THRESHOLD,
        min_cluster_size=MIN_CLUSTER_SIZE,
        n_iter=N_ITER,
        n_jobs=1,  # prevent nested ProcessPoolExecutor deadlock
    )

    for i, col in enumerate(METADATA_COLS):
        pfts.insert(i, col, meta[col].values)

    pfts.to_parquet(out_path, index=False)

    n_dates = df["date"].nunique()
    return f"{out_path.name}: {len(df)} pixels, {n_dates} dates"


def process_polarity(experiment: str, polarity: str, max_workers: int) -> int:
    pigment_dir = resolve_output_dir(experiment, "pigments", polarity)
    out_dir = resolve_output_dir(experiment, "pft", polarity)

    pigment_files = list(pigment_dir.glob("eddy_*_pigments.parquet"))
    if not pigment_files:
        print(f"[{polarity}] No pigment files found in {pigment_dir}")
        return 0

    # Sort largest first so long-running eddies start immediately
    pigment_files.sort(key=lambda f: f.stat().st_size, reverse=True)
    print(f"[{polarity}] {len(pigment_files)} eddies, {max_workers} workers")

    n_written = 0
    n_failed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for fp in pigment_files:
            out_path = out_dir / fp.name.replace("_pigments.parquet", "_pfts.parquet")
            futures[executor.submit(process_eddy, fp, out_path)] = fp.stem
        for future in as_completed(futures):
            stem = futures[future]
            try:
                result = future.result()
            except Exception as e:
                n_failed += 1
                print(f"  FAILED {stem}: {e}")
                continue
            if result:
                print(f"  {result}")
                n_written += 1

    if n_failed > 0:
        raise RuntimeError(f"[{polarity}] {n_failed}/{len(pigment_files)} eddies failed")
    return n_written


def main():
    max_workers = min(os.cpu_count() or 1, MAX_WORKERS)
    n_written = 0
    for polarity in ("cyclone", "anticyclone"):
        n_written += process_polarity(args.experiment, polarity, max_workers)

    print(f"Done. {n_written} PFT files written.")


if __name__ == "__main__":
    main()
