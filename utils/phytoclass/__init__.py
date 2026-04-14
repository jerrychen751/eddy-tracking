"""
Phytoclass: pigment-based phytoplankton community composition via SA + NNLS.

Port of Hayward et al. 2023 (phytoclass R package) to Python, aiming for
byte-for-byte parity with R where feasible. See each submodule for the
specific R source file it mirrors.

Public API:
    - simulated_annealing(S, Fmat, user_defined_min_max, ...): main SA algorithm
    - matrix_checks(S, Fmat): pre-flight sanity filter
    - cluster_samples(S, ...): sample pre-clustering helper
    - run_phytoclass(...): convenience wrapper for the pipeline (clusters then
      runs SA per cluster, returns a single per-sample PFT DataFrame).
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from utils.phytoclass.cluster import cluster_samples
from utils.phytoclass.matrix_checks import matrix_checks
from utils.phytoclass.simulated_annealing import simulated_annealing


SDP_TO_INTERNAL: dict[str, str] = {
    "T chla": "Tchla",
    "Zea": "Zea",
    "DV chla": "DV_chla",
    "ButFuco": "ButFuco",
    "HexFuco": "HexFuco",
    "Allo": "Allo",
    "MV chlb": "MV_chlb",
    "Neo": "Neo",
    "Viola": "Viola",
    "Fuco": "Fuco",
    "chl c1+c2": "Chlc12",
    "chl c3": "Chlc3",
    "Perid": "Perid",
}


def _sa_on_cluster(
    sub_df: pd.DataFrame,
    f_matrix: pd.DataFrame,
    min_max: pd.DataFrame,
    n_iter: int,
    seed: int | None,
) -> pd.DataFrame:
    """Run simulated_annealing on one sample cluster and return class abundances."""
    result = simulated_annealing(
        S=sub_df,
        Fmat=f_matrix,
        user_defined_min_max=min_max,
        do_matrix_checks=False,
        niter=n_iter,
        seed=seed,
        verbose=False,
    )
    return result["class_abundances"]


def run_phytoclass(
    pigments_df: pd.DataFrame,
    f_matrix: pd.DataFrame,
    min_max: pd.DataFrame,
    as_fraction: bool = False,
    cluster_threshold: float = 1.5,
    min_cluster_size: int = 30,
    n_iter: int = 500,
    seed: int | None = None,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """
    End-to-end pipeline: rename SDP columns, sample-cluster, run SA per cluster.

    This is the Python-pipeline convenience wrapper. It is not part of the
    R phytoclass API — R expects the caller to handle clustering separately
    and then call simulated_annealing() per cluster.

    Args:
        pigments_df: SDP pigment DataFrame (per-sample rows). Expected SDP
            columns include "T chla", "Perid", "ButFuco", "Fuco", "HexFuco",
            "Allo", "Zea", "MV chlb", "Neo", "Viola", "chl c1+c2", "chl c3",
            "DV chla". Columns not in SDP_TO_INTERNAL are ignored. The final
            "T chla" column becomes the row's Tchla (required).
        f_matrix: DataFrame with class names as index, internal pigment names
            as columns (matching SDP_TO_INTERNAL values), Tchla as the final
            column. Loaded from the experiment config.
        min_max: DataFrame with columns Class, Pig_Abbrev, min, max. One row
            per non-zero (class, pigment) pair in f_matrix. Loaded from the
            experiment config.
        as_fraction: If True, return row-normalized fractions of total Chla.
            If False, return absolute class abundances in Chla units.
        cluster_threshold: cophenetic distance cutoff for fcluster.
        min_cluster_size: minimum samples per cluster; small clusters are
            merged into their nearest large-cluster centroid.
        n_iter: SA iterations per cluster.
        seed: per-cluster seeds are derived as seed + cluster_index.
        n_jobs: parallel worker count. Defaults to min(n_clusters, cpu_count).
            Set to 1 to avoid nested ProcessPoolExecutor inside a worker.

    Returns:
        DataFrame with one column per class, same row order as pigments_df.
    """
    col_map = {
        sdp: internal
        for sdp, internal in SDP_TO_INTERNAL.items()
        if sdp in pigments_df.columns and internal in f_matrix.columns
    }
    if "T chla" not in pigments_df.columns:
        raise ValueError("pigments_df must contain a 'T chla' column (total chlorophyll a).")

    renamed = pigments_df[list(col_map.keys())].rename(columns=col_map)

    f_cols = list(f_matrix.columns)
    for fc in f_cols:
        if fc not in renamed.columns:
            renamed[fc] = 0.0
    renamed = renamed[f_cols]

    if renamed.isna().any().any():
        raise ValueError(
            "pigments_df contains NaN in at least one F-matrix pigment column — "
            "drop or impute before calling run_phytoclass()."
        )

    S_filtered, F_filtered = matrix_checks(renamed, f_matrix)

    class_names = list(f_matrix.index)
    n_samples = len(pigments_df)
    C_out = np.zeros((n_samples, len(class_names)))

    clusters = cluster_samples(
        S_filtered.values,
        min_samples=min_cluster_size,
        distance_threshold=cluster_threshold,
    )
    n_clusters = len(clusters)
    print(
        f"Phytoclass: {n_samples} samples -> {n_clusters} clusters, "
        f"SA({n_iter} iter)"
    )

    cluster_seeds = [
        (seed + i if seed is not None else None) for i in range(n_clusters)
    ]
    cluster_frames = [
        S_filtered.iloc[row_indices] for row_indices, _ in clusters
    ]

    if n_jobs is None:
        n_jobs = min(n_clusters, os.cpu_count() or 1)

    out_by_cluster: dict[int, pd.DataFrame] = {}
    if n_jobs == 1 or n_clusters == 1:
        for idx, sub_df in enumerate(cluster_frames):
            print(f"  cluster {idx + 1}/{n_clusters}: {len(sub_df)} samples...")
            result_df = _sa_on_cluster(
                sub_df, F_filtered, min_max, n_iter, cluster_seeds[idx]
            )
            out_by_cluster[idx] = result_df
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(
                    _sa_on_cluster,
                    sub_df, F_filtered, min_max, n_iter, cluster_seeds[idx],
                ): idx
                for idx, sub_df in enumerate(cluster_frames)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    out_by_cluster[idx] = future.result()
                except Exception as exc:
                    print(f"  cluster {idx + 1}/{n_clusters} FAILED: {exc}")
                    raise

    col_idx_by_name = {name: i for i, name in enumerate(class_names)}
    for idx, (row_indices, _) in enumerate(clusters):
        result_df = out_by_cluster[idx]
        for col_name in result_df.columns:
            if col_name in col_idx_by_name:
                C_out[row_indices, col_idx_by_name[col_name]] = result_df[col_name].values

    out = pd.DataFrame(C_out, columns=class_names, index=pigments_df.index)
    if as_fraction:
        row_sums = out.sum(axis=1)
        row_sums = row_sums.replace(0, np.nan)
        out = out.div(row_sums, axis=0).fillna(0)
    return out


__all__ = [
    "run_phytoclass",
    "simulated_annealing",
    "cluster_samples",
    "matrix_checks",
    "SDP_TO_INTERNAL",
]
