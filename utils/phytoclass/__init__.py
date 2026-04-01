"""Phytoclass: pigment-based phytoplankton community composition via SA + NNLS.

Port of the R phytoclass package (Hayward et al. 2023) to Python.
Resolves phytoplankton functional type (PFT) composition from accessory
pigment concentrations using simulated annealing + NNLS matrix factorization.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from utils.phytoclass.config import (
    SDP_TO_INTERNAL,
    load_default_bounds,
    load_default_f_matrix,
)
from utils.phytoclass.nnls_mf import nnls_factorize
from utils.phytoclass.annealing import simulated_annealing
from utils.phytoclass.cluster import cluster_samples


def _build_bounds_dict(
    F_df: pd.DataFrame,
    bounds_df: pd.DataFrame,
) -> dict[tuple[int, int], tuple[float, float]]:
    """
    Convert the bounds CSV into a {(row, col): (min, max)} dict
    aligned to the F matrix indices.

    Args:
        F_df: F matrix DataFrame (classes x pigments), indexed by class name.
        bounds_df: Bounds DataFrame with columns: class, pigment, min, max.

    Returns:
        Dict mapping (class_idx, pigment_idx) to (min_val, max_val).
    """
    class_names = list(F_df.index)
    col_names = list(F_df.columns)
    result = {}
    for _, row in bounds_df.iterrows():
        if row["class"] in class_names and row["pigment"] in col_names:
            i = class_names.index(row["class"])
            j = col_names.index(row["pigment"])
            result[(i, j)] = (row["min"], row["max"])
    return result


def run_phytoclass(
    pigments_df: pd.DataFrame,
    as_fraction: bool = False,
    cluster_threshold: float = 1.5,
    min_cluster_size: int = 30,
    n_iter: int = 500,
    n_neighbors: int = 120,
    seed: int | None = None,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """
    Estimate phytoplankton community composition from pigment concentrations.

    Takes a DataFrame of SDP-predicted pigment concentrations and returns
    Chla-weighted abundances (or fractions) for each phytoplankton functional type.

    Pipeline:
        1. Map SDP column names to internal names
        2. Cluster samples by pigment-ratio similarity (Ward's linkage)
        3. Run simulated annealing + NNLS per cluster to optimize F matrix
           (clusters are processed in parallel across CPU cores)
        4. Combine results across clusters

    Args:
        pigments_df: DataFrame with SDP pigment columns. Must include at minimum:
            "T chla", "Perid", "ButFuco", "Fuco", "HexFuco", "Allo", "Zea",
            "MV chlb", "Neo", "Viola". Other columns are ignored.
        as_fraction: If True, return fractions of total Chla (rows sum to 1).
            If False, return absolute Chla per class (mg/m3).
        cluster_threshold: Ward's linkage distance threshold for sample clustering.
        min_cluster_size: Minimum samples per cluster.
        n_iter: SA iterations per cluster.
        n_neighbors: Random neighbors per SA iteration.
        seed: Random seed for reproducibility. Each cluster receives a
            deterministic derived seed (seed + cluster_index).
        n_jobs: Number of parallel worker processes. Defaults to
            min(n_clusters, cpu_count). Set to 1 for sequential execution
            and when calling from inside a ProcessPoolExecutor worker to
            avoid nested process pools.

    Returns:
        DataFrame with one column per PFT class (Diatoms, Dinoflagellates,
        Haptophytes, Cryptophytes, Green_algae, Cyanobacteria). Same row
        count as input. Values are Chla (mg/m3) or fractions.
    """
    # Load defaults
    F_df, _ = load_default_f_matrix()
    bounds_df = load_default_bounds()

    # Map SDP names to internal names and select matching columns
    col_map = {sdp: internal for sdp, internal in SDP_TO_INTERNAL.items()
                if internal in F_df.columns and sdp in pigments_df.columns}

    # Build the sample matrix S with columns ordered to match F
    f_cols = list(F_df.columns)  # includes Tchla at the end
    sdp_for_col = {}
    for sdp_name, internal_name in col_map.items():
        sdp_for_col[internal_name] = sdp_name

    S_cols_ordered = []
    for fc in f_cols:
        if fc in sdp_for_col:
            S_cols_ordered.append(sdp_for_col[fc])
        else:
            # Column in F but not in SDP — skip by zeroing in F
            S_cols_ordered.append(None)

    # Remove F columns that have no SDP counterpart
    keep_mask = [c is not None for c in S_cols_ordered]
    f_cols_kept = [f_cols[i] for i in range(len(f_cols)) if keep_mask[i]]
    sdp_cols_kept = [S_cols_ordered[i] for i in range(len(S_cols_ordered)) if keep_mask[i]]

    F_arr = F_df[f_cols_kept].values.astype(float)
    S_arr = pigments_df[sdp_cols_kept].values.astype(float)

    # Validate: no NaN values
    if np.any(np.isnan(S_arr)):
        raise ValueError(
            "Input contains NaN values. Drop or impute before calling run_phytoclass()."
        )

    # Build bounds dict aligned to the kept columns
    bounds_dict = _build_bounds_dict(
        pd.DataFrame(F_arr, index=F_df.index, columns=f_cols_kept),
        bounds_df,
    )

    class_names = list(F_df.index)
    n_samples = S_arr.shape[0]
    n_classes = F_arr.shape[0]

    # Allocate output
    C_all = np.zeros((n_samples, n_classes))

    # Cluster samples by pigment-ratio similarity
    clusters = cluster_samples(S_arr, min_samples=min_cluster_size,
                                distance_threshold=cluster_threshold)

    n_clusters = len(clusters)
    print(f"Phytoclass: {n_samples} samples -> {n_clusters} clusters, "
          f"SA({n_iter} iter, {n_neighbors} neighbors)")

    # Derive per-cluster seeds so each cluster is deterministic but independent
    cluster_seeds = [
        (seed + i if seed is not None else None)
        for i in range(n_clusters)
    ]

    # Run SA in parallel across clusters
    if n_jobs is None:
        n_jobs = min(n_clusters, os.cpu_count() or 1)

    if n_jobs == 1 or n_clusters == 1:
        # Sequential — avoids ProcessPoolExecutor overhead for trivial cases
        for idx, (row_indices, S_cluster) in enumerate(clusters):
            print(f"Cluster {idx + 1}/{n_clusters}: {len(row_indices)} samples...")
            result = simulated_annealing(
                S_cluster, F_arr, bounds_dict,
                n_iter=n_iter, n_neighbors=n_neighbors,
                seed=cluster_seeds[idx],
            )
            C_all[row_indices] = result["C"]
            print(f"Cluster {idx + 1}/{n_clusters}: done (RMSE {result['rmse']:.6f})")
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {}
            for idx, (row_indices, S_cluster) in enumerate(clusters):
                future = executor.submit(
                    simulated_annealing,
                    S_cluster, F_arr, bounds_dict,
                    n_iter=n_iter, n_neighbors=n_neighbors,
                    seed=cluster_seeds[idx],
                )
                futures[future] = (idx, row_indices, len(row_indices))

            for future in as_completed(futures):
                idx, row_indices, n = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"Cluster {idx + 1}/{n_clusters} FAILED: {e}")
                    continue
                C_all[row_indices] = result["C"]
                print(f"Cluster {idx + 1}/{n_clusters}: {n} samples, RMSE {result['rmse']:.6f}")

    # Build output DataFrame
    out = pd.DataFrame(C_all, columns=class_names, index=pigments_df.index)

    if as_fraction:
        row_sums = out.sum(axis=1)
        row_sums = row_sums.replace(0, np.nan)
        out = out.div(row_sums, axis=0).fillna(0)

    return out


__all__ = ["run_phytoclass"]
