"""
Hierarchical clustering of diagnostic pigments from SDP output.

Follows Kramer & Siegel (2019): compute pigment:TChla ratios, take the pairwise
Pearson correlation matrix, convert to 1 - r distance, and run Ward's linkage.
Produces a dendrogram, linkage matrix, and flat cluster assignments - intended
to validate which PFT biomarkers co-vary in the dataset, informing F-matrix
pruning or merging for PhytoClass.

Two modes:
  Default (pooled): all pixels from all eddies concatenated. The resulting
    correlations are dominated by between-eddy regime shifts (biomass,
    season, water type) and lose fine-grained within-eddy PFT structure.
  --per-eddy: compute a 12x12 correlation matrix per eddy independently,
    then average across eddies. Isolates within-eddy PFT covariance, which
    matches the scope of PhytoClass (it runs per-eddy too).
"""

import argparse
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

from utils.config import METADATA_COLS, resolve_output_dir

# T chla is the sum across all chlorophyll-a forms and trivially correlates
# with everything; exclude it so biomarker groupings stand out.
EXCLUDE_COLS = {"T chla"}

DISPLAY_NAMES = {
    "DV chla":    "DV Chl-a",
    "MV chlb":    "MV Chl-b",
    "chl c1+c2":  "Chl-c1+c2",
    "chl c3":     "Chl-c3",
    "Fuco":       "Fucoxanthin",
    "ButFuco":    "But-Fuco",
    "HexFuco":    "Hex-Fuco",
    "Perid":      "Peridinin",
    "Allo":       "Alloxanthin",
    "Zea":        "Zeaxanthin",
    "Viola":      "Violaxanthin",
    "Neo":        "Neoxanthin",
}

# Literature-based primary PFT assignment per diagnostic pigment.
# Used to label clusters with their expected PFT group(s).
PIGMENT_TO_PFT = {
    "Fuco":      "Diatoms",
    "chl c1+c2": "Diatoms",
    "HexFuco":   "Haptophytes",
    "ButFuco":   "Haptophytes",
    "chl c3":    "Haptophytes",
    "Zea":       "Cyanobacteria",
    "DV chla":   "Cyanobacteria",
    "Perid":     "Dinoflagellates",
    "MV chlb":   "Green algae",
    "Viola":     "Green algae",
    "Neo":       "Green algae",
    "Allo":      "Cryptophytes",
}

# Order for presenting joined PFT labels (matches typical bio-optics convention)
PFT_ORDER = [
    "Diatoms", "Haptophytes", "Cyanobacteria",
    "Dinoflagellates", "Green algae", "Cryptophytes",
]


def build_cluster_pft_label(pigments_in_cluster: list[str]) -> str:
    """
    Build a ' + '-joined PFT label for a cluster, preserving PFT_ORDER.
    """
    pfts = {PIGMENT_TO_PFT.get(p, "?") for p in pigments_in_cluster}
    return " + ".join([p for p in PFT_ORDER if p in pfts])


def iter_eddy_files(experiment: str) -> Iterator[tuple[Path, pd.DataFrame]]:
    """
    Yield (path, dataframe) for each eddy pigment parquet, both polarities.
    """
    for polarity in ("cyclone", "anticyclone"):
        pig_dir = resolve_output_dir(experiment, "pigments", polarity)
        files = sorted(pig_dir.glob("eddy_*_pigments.parquet"))
        print(f"[{polarity}] {len(files)} eddy files in {pig_dir}")
        for fp in files:
            yield fp, pd.read_parquet(fp)


def get_pigment_cols(df: pd.DataFrame) -> list[str]:
    """Return modeled pigment columns, excluding metadata and total Chla."""
    return [c for c in df.columns if c not in METADATA_COLS and c not in EXCLUDE_COLS]


def clean_pigments(df: pd.DataFrame, pigment_cols: list[str]) -> pd.DataFrame:
    """
    Drop rows with T chla <= 0 or any non-finite pigment value.
    """
    valid = (df["T chla"] > 0) & np.all(np.isfinite(df[pigment_cols].to_numpy()), axis=1)
    return df[valid]


def compute_pooled_correlation(experiment: str) -> tuple[list[str], np.ndarray]:
    """
    Concatenate all pigment data, compute a single pooled 12x12 correlation matrix.
    """
    frames = [df for _, df in iter_eddy_files(experiment)]
    if not frames:
        raise RuntimeError("No pigment files found - did run_sdp.py finish?")
    df = pd.concat(frames, ignore_index=True)
    print(f"Pooled observations: {len(df)}")

    pigment_cols = get_pigment_cols(df)
    df_clean = clean_pigments(df, pigment_cols)
    print(f"Clean observations: {len(df_clean)} ({len(df) - len(df_clean)} dropped)")

    ratios = df_clean[pigment_cols].div(df_clean["T chla"], axis=0)
    corr = ratios.corr().values
    return pigment_cols, corr


def compute_per_eddy_mean_correlation(
    experiment: str, min_pixels: int
) -> tuple[list[str], np.ndarray, int]:
    """
    Compute a 12x12 correlation matrix per eddy, then average across eddies.

    Eddies with fewer than min_pixels clean pixels, or whose correlation matrix
    contains any NaN (e.g. a pigment is constant within the eddy), are skipped.
    """
    pigment_cols_ref: list[str] | None = None
    per_eddy_corrs: list[np.ndarray] = []
    n_skipped = 0

    for fp, df in iter_eddy_files(experiment):
        pigment_cols = get_pigment_cols(df)
        if pigment_cols_ref is None:
            pigment_cols_ref = pigment_cols
        elif pigment_cols != pigment_cols_ref:
            raise RuntimeError(f"Pigment columns differ in {fp}")

        df_clean = clean_pigments(df, pigment_cols)
        if len(df_clean) < min_pixels:
            n_skipped += 1
            continue

        ratios = df_clean[pigment_cols].div(df_clean["T chla"], axis=0)
        corr = ratios.corr().values
        if np.any(np.isnan(corr)):
            n_skipped += 1
            continue

        per_eddy_corrs.append(corr)

    if not per_eddy_corrs:
        raise RuntimeError("No eddies with enough valid pixels for per-eddy clustering")

    print(f"Per-eddy correlation matrices: {len(per_eddy_corrs)} kept, {n_skipped} skipped")
    mean_corr = np.mean(per_eddy_corrs, axis=0)
    assert pigment_cols_ref is not None
    return pigment_cols_ref, mean_corr, len(per_eddy_corrs)


def compute_linkage_from_correlation(corr: np.ndarray) -> np.ndarray:
    """
    Convert a correlation matrix to 1 - r distance and run Ward's linkage.
    """
    dist = 1 - corr
    # Clip diagonal noise and ensure symmetry before squareform
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    return linkage(condensed, method="ward")


def save_dendrogram(
    Z: np.ndarray,
    pigment_names: list[str],
    out_path: Path,
    color_threshold: float,
    title: str,
) -> None:
    """Render and save a pigment dendrogram with PFT cluster labels."""
    labels = [DISPLAY_NAMES.get(p, p) for p in pigment_names]
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    dendrogram(Z, labels=labels, ax=ax, leaf_rotation=45, leaf_font_size=9,
               color_threshold=color_threshold)
    ax.axhline(color_threshold, color="0.6", linestyle="--", linewidth=0.8)
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_ylabel("1 - Pearson correlation", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)

    # Annotate each flat cluster with its PFT label above the leaves
    assignments = fcluster(Z, t=color_threshold, criterion="distance")
    display_to_pigment = {DISPLAY_NAMES.get(p, p): p for p in pigment_names}
    tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    tick_positions = ax.get_xticks()

    # Group tick x positions by cluster id
    cluster_tick_xs: dict[int, list[float]] = {}
    cluster_members: dict[int, list[str]] = {}
    for xpos, leaf_label in zip(tick_positions, tick_labels):
        pigment = display_to_pigment.get(leaf_label, leaf_label)
        cid = assignments[pigment_names.index(pigment)]
        cluster_tick_xs.setdefault(cid, []).append(xpos)
        cluster_members.setdefault(cid, []).append(pigment)

    y_anno = ax.get_ylim()[1] * 1.02
    ax.set_ylim(top=ax.get_ylim()[1] * 1.18)
    for cid, xs in cluster_tick_xs.items():
        mid = (min(xs) + max(xs)) / 2
        label = build_cluster_pft_label(cluster_members[cid])
        ax.text(mid, y_anno, label, ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="0.2",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="0.7", linewidth=0.6))

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved dendrogram: {out_path}")


def save_cluster_table(
    Z: np.ndarray,
    pigment_names: list[str],
    threshold: float,
    out_path: Path,
) -> None:
    """Write pigment cluster assignments and print their PFT groupings."""
    assignments = fcluster(Z, t=threshold, criterion="distance")

    # Per-pigment table (with literature PFT)
    table = pd.DataFrame({
        "pigment": pigment_names,
        "display_name": [DISPLAY_NAMES.get(p, p) for p in pigment_names],
        "literature_pft": [PIGMENT_TO_PFT.get(p, "?") for p in pigment_names],
        "cluster": assignments,
    })

    # Derive each cluster's overall PFT label from its members
    cluster_to_label: dict[int, str] = {}
    for cid in sorted(set(assignments)):
        members = [p for p, c in zip(pigment_names, assignments) if c == cid]
        cluster_to_label[cid] = build_cluster_pft_label(members)
    table["cluster_pft_group"] = table["cluster"].map(cluster_to_label)

    table = table.sort_values(["cluster", "pigment"]).reset_index(drop=True)
    table.to_csv(out_path, index=False)

    n_clusters = table["cluster"].nunique()
    print(f"{n_clusters} clusters at threshold {threshold}: {out_path}")
    print(table.to_string(index=False))

    print("\nCluster -> PFT group:")
    for cid, label in cluster_to_label.items():
        members = table[table["cluster"] == cid]["display_name"].tolist()
        print(f"  Cluster {cid} ({label}): {', '.join(members)}")


def main() -> None:
    """Cluster experiment pigments and write the selected analysis artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="experiment name under configs/")
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="1-correlation distance threshold for flat clusters")
    parser.add_argument("--per-eddy", action="store_true",
                        help="Compute correlation per eddy and average (matches PhytoClass scope)")
    parser.add_argument("--min-pixels", type=int, default=50,
                        help="Minimum clean pixels per eddy in --per-eddy mode")
    args = parser.parse_args()

    if args.per_eddy:
        pigment_cols, mean_corr, n_eddies = compute_per_eddy_mean_correlation(
            args.experiment, min_pixels=args.min_pixels,
        )
        Z = compute_linkage_from_correlation(mean_corr)
        suffix = "_per_eddy"
        title = (f"Hierarchical Clustering of Diagnostic Pigments "
                 f"(per-eddy, avg over {n_eddies} eddies)")
        # Also persist the averaged correlation matrix for inspection
        np.save(resolve_output_dir(args.experiment, "pigments_clustering")
                / f"mean_correlation{suffix}.npy", mean_corr)
    else:
        pigment_cols, corr = compute_pooled_correlation(args.experiment)
        Z = compute_linkage_from_correlation(corr)
        suffix = ""
        title = "Hierarchical Clustering of Diagnostic Pigments (Kramer method, pooled)"

    out_dir = resolve_output_dir(args.experiment, "pigments_clustering")
    np.save(out_dir / f"linkage{suffix}.npy", Z)
    save_dendrogram(Z, pigment_cols, out_dir / f"dendrogram{suffix}.png",
                    args.threshold, title)
    save_cluster_table(Z, pigment_cols, args.threshold,
                       out_dir / f"cluster_assignments{suffix}.csv")

    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
