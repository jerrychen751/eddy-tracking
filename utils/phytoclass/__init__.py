"""
Phytoclass: pigment-based phytoplankton community composition via SA + NNLS.

This package follows Hayward et al. (2023) and preserves parity with the
reference R implementation where feasible.

Public API:
    - ``run_phytoclass``: cluster samples and estimate per-sample PFT abundances.
    - ``simulated_annealing``: run the core simulated annealing algorithm.
    - ``matrix_checks``: validate and filter model matrices.
    - ``cluster_samples``: group samples before model execution.
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
    cluster_df: pd.DataFrame,
    f_matrix: pd.DataFrame,
    min_max: pd.DataFrame,
    n_iter: int,
    seed: int | None,
) -> pd.DataFrame:
    """Run simulated annealing for one cluster and return class abundances."""
    result = simulated_annealing(
        S=cluster_df,
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
    Estimate phytoplankton class abundances from SDP pigment concentrations.

    Args:
        pigments_df: Per-sample SDP pigments. ``T chla`` is required; columns
            outside ``SDP_TO_INTERNAL`` are ignored.
        f_matrix: Class-by-pigment contribution matrix using internal pigment
            names from ``SDP_TO_INTERNAL``.
        min_max: DataFrame with columns Class, Pig_Abbrev, min, max. One row
            per non-zero class-pigment pair in ``f_matrix``.
        as_fraction: Return row-normalized fractions instead of Chla units.
        cluster_threshold: Cophenetic distance cutoff for sample clustering.
        min_cluster_size: Minimum samples per cluster; small clusters are
            merged into their nearest large-cluster centroid.
        n_iter: Simulated annealing iterations per cluster.
        seed: Base seed. Each cluster uses ``seed + cluster_idx``.
        n_jobs: Worker count. Defaults to the smaller of cluster count and CPU
            count. Use 1 when this function already runs inside a worker.

    Returns:
        DataFrame with one column per class, same row order as pigments_df.

    Side effects:
        Prints cluster progress and starts worker processes when ``n_jobs > 1``.
    """
    column_map = {
        sdp: internal
        for sdp, internal in SDP_TO_INTERNAL.items()
        if sdp in pigments_df.columns and internal in f_matrix.columns
    }
    if "T chla" not in pigments_df.columns:
        raise ValueError(
            "pigments_df must contain a 'T chla' column (total chlorophyll a)."
        )

    pigment_matrix = pigments_df[list(column_map)].rename(columns=column_map)

    f_matrix_columns = list(f_matrix.columns)
    for column in f_matrix_columns:
        if column not in pigment_matrix.columns:
            pigment_matrix[column] = 0.0
    pigment_matrix = pigment_matrix[f_matrix_columns]

    if pigment_matrix.isna().any().any():
        raise ValueError(
            "pigments_df contains NaN in at least one F-matrix pigment column - "
            "drop or impute before calling run_phytoclass()."
        )

    checked_pigments, checked_f_matrix = matrix_checks(pigment_matrix, f_matrix)

    class_names = list(f_matrix.index)
    n_samples = len(pigments_df)
    class_abundances = np.zeros((n_samples, len(class_names)))

    clusters = cluster_samples(
        checked_pigments.values,
        min_samples=min_cluster_size,
        distance_threshold=cluster_threshold,
    )
    n_clusters = len(clusters)
    print(
        f"Phytoclass: {n_samples} samples -> {n_clusters} clusters, "
        f"SA({n_iter} iter)"
    )

    cluster_seeds = [
        seed + cluster_idx if seed is not None else None
        for cluster_idx in range(n_clusters)
    ]
    cluster_frames = [
        checked_pigments.iloc[row_indices] for row_indices, _ in clusters
    ]

    if n_jobs is None:
        n_jobs = min(n_clusters, os.cpu_count() or 1)

    result_by_cluster: dict[int, pd.DataFrame] = {}
    if n_jobs == 1 or n_clusters == 1:
        for cluster_idx, cluster_df in enumerate(cluster_frames):
            print(
                f"  cluster {cluster_idx + 1}/{n_clusters}: "
                f"{len(cluster_df)} samples..."
            )
            cluster_result = _sa_on_cluster(
                cluster_df,
                checked_f_matrix,
                min_max,
                n_iter,
                cluster_seeds[cluster_idx],
            )
            result_by_cluster[cluster_idx] = cluster_result
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(
                    _sa_on_cluster,
                    cluster_df,
                    checked_f_matrix,
                    min_max,
                    n_iter,
                    cluster_seeds[cluster_idx],
                ): cluster_idx
                for cluster_idx, cluster_df in enumerate(cluster_frames)
            }
            for future in as_completed(futures):
                cluster_idx = futures[future]
                try:
                    result_by_cluster[cluster_idx] = future.result()
                except Exception as exc:
                    print(
                        f"  cluster {cluster_idx + 1}/{n_clusters} FAILED: {exc}"
                    )
                    raise

    class_idx_by_name = {name: idx for idx, name in enumerate(class_names)}
    for cluster_idx, (row_indices, _) in enumerate(clusters):
        cluster_result = result_by_cluster[cluster_idx]
        for class_name in cluster_result.columns:
            if class_name in class_idx_by_name:
                class_idx = class_idx_by_name[class_name]
                class_abundances[row_indices, class_idx] = cluster_result[
                    class_name
                ].values

    output = pd.DataFrame(
        class_abundances,
        columns=class_names,
        index=pigments_df.index,
    )
    if as_fraction:
        total_abundance = output.sum(axis=1).replace(0, np.nan)
        output = output.div(total_abundance, axis=0).fillna(0)
    return output


__all__ = [
    "run_phytoclass",
    "simulated_annealing",
    "cluster_samples",
    "matrix_checks",
    "SDP_TO_INTERNAL",
]
