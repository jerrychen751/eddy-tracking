"""Pre-clustering of pigment samples for phytoclass analysis."""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


def cluster_samples(
    S: np.ndarray,
    min_samples: int = 14,
    distance_threshold: float = 1.5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Cluster pigment samples by community similarity using Ward's linkage.

    Normalizes pigments to Tchla (last column), computes Euclidean distance,
    and applies Ward's hierarchical clustering with a distance threshold.
    Clusters smaller than min_samples are merged into their nearest neighbor.

    Args:
        S: Sample matrix (n_samples, n_pigments). Last column must be Tchla.
        min_samples: Minimum samples per cluster. Clusters below this are merged.
        distance_threshold: Cophenetic distance cutoff for fcluster.

    Returns:
        List of (row_indices, sub_matrix) tuples, one per cluster.
        row_indices is a 1D int array of positions in the original S.
    """
    tchla = S[:, -1]
    safe_tchla = np.where(tchla > 0, tchla, 1e-10)
    ratios = S[:, :-1] / safe_tchla[:, np.newaxis]

    # Ward's linkage on Euclidean distance of pigment ratios
    condensed = pdist(ratios, metric="euclidean")
    Z = linkage(condensed, method="ward")
    labels = fcluster(Z, t=distance_threshold, criterion="distance")

    # Merge small clusters into nearest large cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    large_mask = counts >= min_samples
    large_labels = set(unique_labels[large_mask])

    if not large_labels:
        # Everything is too small — return as one cluster
        return [(np.arange(len(S)), S)]

    # For each small cluster, find the large cluster whose centroid is closest
    centroids = {}
    for lbl in large_labels:
        centroids[lbl] = ratios[labels == lbl].mean(axis=0)

    for lbl in unique_labels:
        if lbl in large_labels:
            continue
        small_centroid = ratios[labels == lbl].mean(axis=0)
        best_lbl = min(
            large_labels,
            key=lambda large_lbl: np.linalg.norm(small_centroid - centroids[large_lbl]),
        )
        labels[labels == lbl] = best_lbl

    # Build output list with row indices
    final_labels = np.unique(labels)
    return [
        (np.where(labels == lbl)[0], S[labels == lbl])
        for lbl in final_labels
    ]
