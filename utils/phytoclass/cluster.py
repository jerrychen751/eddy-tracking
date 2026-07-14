"""
Sample pre-clustering for phytoclass. Ports phytoclass::Cluster with one
documented departure.

R's Cluster() does:
    1. Divide each pigment column by Tchla (per-sample normalization)
    2. Box-Cox standardize each pigment column via bestNormalize::boxcox()
    3. stats::dist(euclidean) + stats::hclust(ward.D2) on the standardized rows
    4. Cut the dendrogram with dynamicTreeCut::cutreeDynamic(cutHeight=70,
       method="hybrid", minClusterSize=min_samples, deepSplit=4, pamStage=TRUE,
       respectSmallClusters=TRUE)

The Python port keeps steps 1-3 (Box-Cox via scipy.stats.boxcox, Ward linkage
via scipy which is mathematically equivalent to R's ward.D2), but cannot
replicate step 4 exactly - there is no pure-Python port of dynamicTreeCut.
Instead, we cut with scipy.cluster.hierarchy.fcluster at a user-specified
cophenetic distance threshold and post-hoc merge small clusters into their
nearest large-cluster centroid. Clusters produced this way are similar to R's
for typical datasets but not byte-identical.
"""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import boxcox


def _standardize_pigment_ratios(S: np.ndarray) -> np.ndarray:
    """
    Divide every pigment column by the Tchla column, then Box-Cox each column.

    Mirrors R's Cluster standardise() helper: it first replaces 0 entries with
    1e-6 (to avoid divide-by-zero inside Box-Cox), divides by Tchla, then
    applies Box-Cox column by column with a per-column lambda chosen to
    maximize log-likelihood.
    """
    pigments = S[:, :-1].astype(float).copy()
    tchla = S[:, -1].astype(float)
    pigments[pigments == 0] = 1e-6
    safe_tchla = np.where(tchla > 0, tchla, 1e-10)
    ratios = pigments / safe_tchla[:, np.newaxis]

    standardized = np.empty_like(ratios)
    for j in range(ratios.shape[1]):
        col = ratios[:, j]
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
        S: sample matrix (n_samples, n_pigments) with Tchla as last column.
        min_samples: minimum cluster size; smaller clusters get merged into
            their nearest large-cluster centroid.
        distance_threshold: cophenetic distance cutoff passed to fcluster.

    Returns:
        List of (row_indices, S_subset) tuples, one per final cluster. Every
        row of S appears in exactly one output cluster.
    """
    standardized = _standardize_pigment_ratios(S)

    if len(S) < 2:
        return [(np.arange(len(S)), S)]

    condensed = pdist(standardized, metric="euclidean")
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
