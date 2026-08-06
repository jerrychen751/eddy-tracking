"""
Sample pre-clustering for phytoclass. Ports phytoclass::Cluster.

R cuts the dendrogram with dynamicTreeCut::cutreeDynamic, which has no Python port. This module cuts at the caller's cophenetic distance instead, then merges every cluster below min_samples into the nearest large-cluster centroid, so cluster membership differs from R on the same input.
"""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import boxcox


def _standardize_pigment_ratios(S: np.ndarray) -> np.ndarray:
    """
    Divide every pigment column by the Tchla column, then Box-Cox each column. Mirrors R's Cluster standardise() helper.

    Returns (n_samples, n_pigments - 1). A column Box-Cox rejects keeps its untransformed values, so the output can mix transformed and raw scales.
    """
    pigments = S[:, :-1].astype(float).copy()  # (n_samples, n_pigments) -> (n_samples, n_pigments - 1)
    tchla = S[:, -1].astype(float)  # (n_samples, n_pigments) -> (n_samples,)
    # Mirrors R's standardise(), which substitutes 1e-6 for a 0 pigment before taking the ratio.
    pigments[pigments == 0] = 1e-6
    safe_tchla = np.where(tchla > 0, tchla, 1e-10)
    ratios = pigments / safe_tchla[:, np.newaxis]  # safe_tchla: (n_samples,) -> (n_samples, 1)

    standardized = np.empty_like(ratios)
    for j in range(ratios.shape[1]):
        col = ratios[:, j]
        # scipy.stats.boxcox needs strictly positive input and raises ValueError on a constant column.
        col_shifted = col if col.min() > 0 else col - col.min() + 1e-10
        try:
            bc, _ = boxcox(col_shifted)
            standardized[:, j] = bc
        except ValueError:
            standardized[:, j] = col_shifted
    return standardized


def cluster_samples(
    S: np.ndarray,
    min_samples: int = 14,
    distance_threshold: float = 1.5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Cluster pigment samples by pigment-ratio similarity.

    Args:
        S: sample matrix (n_samples, n_pigments) with Tchla as the last column.
        min_samples: minimum cluster size. A smaller cluster is merged into its nearest large-cluster centroid.
        distance_threshold: cophenetic distance cutoff passed to fcluster.

    Returns:
        List of (row_indices, S_subset) tuples, one per final cluster. Every row of S appears in exactly one output cluster. When no cluster reaches min_samples, every row comes back as a single cluster.
    """
    standardized = _standardize_pigment_ratios(S)

    if len(S) < 2:
        return [(np.arange(len(S)), S)]

    condensed = pdist(standardized, metric="euclidean")  # (n_samples, n_pigments - 1) -> (n_samples * (n_samples - 1) / 2,)
    # scipy's "ward" runs the Lance-Williams update on the distances, which matches R's ward.D2, not ward.D.
    Z = linkage(condensed, method="ward")
    labels = fcluster(Z, t=distance_threshold, criterion="distance")

    unique_labels, counts = np.unique(labels, return_counts=True)
    large_labels = set(unique_labels[counts >= min_samples])

    if not large_labels:
        return [(np.arange(len(S)), S)]

    centroids = {
        lbl: standardized[labels == lbl].mean(axis=0) for lbl in large_labels
    }

    for lbl in unique_labels:
        if lbl in large_labels:
            continue
        small_centroid = standardized[labels == lbl].mean(axis=0)
        nearest = min(
            large_labels,
            key=lambda large_lbl: np.linalg.norm(small_centroid - centroids[large_lbl]),
        )
        labels[labels == lbl] = nearest

    final_labels = np.unique(labels)
    return [
        (np.where(labels == lbl)[0], S[labels == lbl])
        for lbl in final_labels
    ]
