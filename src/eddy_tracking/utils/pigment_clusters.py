# Hierarchical clustering of the SDP pigment ratios over the target eddy-composites, and its PFT labels.
# A cluster is named after the PFTs whose diagnostic pigments it holds, the way Kramer and Siegel (2019) name the branches of their trees: the marker pigment places a group, while pigments shared by several groups (Chlc12, Chlc3, Viola) only follow whichever markers they correlate with.
# research_paper/pigments_clustering.ipynb draws and checks the clustering and research_paper/figures.ipynb groups its pigment panels by the labeled clusters; both read them from here, so a change of data or cut height reaches both.

from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from eddy_tracking.utils.analysis_period import restrict_to_analysis_period
from eddy_tracking.validation.quality import keep_valid_sdp_rows

SDP_PIGMENTS: tuple[str, ...] = ("Tchla", "Fuco", "HexFuco", "ButFuco", "Perid", "Chlc12", "Chlc3", "Allo", "MV_chlb", "Neo", "Viola", "Zea", "DV_chla")
# Every SDP pigment but Tchla, each divided by Tchla before clustering
RATIO_PIGMENTS: tuple[str, ...] = SDP_PIGMENTS[1:]
# Inside the gap between the merges at 0.68 and 0.76 of the ratio tree, the six-cluster level: the lowest cut at which every split still separates a new diagnostic-pigment group (cyanobacteria from haptophytes and pelagophytes, and Perid on its own), the "maximum number of taxonomically relevant clusters" rule of Kramer and Siegel (2019). The three-cluster cut at 1.5 (gap 1.45 to 2.60) is the more stable robustness check.
CLUSTER_CUT_HEIGHT = 0.72

# Kramer and Siegel (2019), Table 1: every group in which a pigment is often present, always present, or unique
PIGMENT_GROUPS: dict[str, tuple[str, ...]] = {
    "Fuco": ("Diatoms", "Dinoflagellates", "Chrysophytes", "Pelagophytes", "Haptophytes"),
    "HexFuco": ("Haptophytes",),
    "ButFuco": ("Pelagophytes", "Haptophytes"),
    "Perid": ("Dinoflagellates",),
    "Chlc12": ("Diatoms", "Dinoflagellates", "Chrysophytes", "Pelagophytes", "Haptophytes", "Cryptophytes"),
    "Chlc3": ("Diatoms", "Dinoflagellates", "Haptophytes"),
    "Allo": ("Cryptophytes",),
    "MV_chlb": ("Green algae",),
    "Neo": ("Green algae",),
    "Viola": ("Chrysophytes", "Green algae"),
    "Zea": ("Chrysophytes", "Pelagophytes", "Green algae", "Cyanobacteria"),
    "DV_chla": ("Cyanobacteria",),
}
# Diagnostic (marker) pigment of each PFT among the 12 SDP accessory pigments: the markers Kramer and Siegel (2019) name for the global groups (Fuco, Perid, HexFuco, MVchlb with the green accessory carotenoids, DVchla and Zea) plus the markers they use for the local groups (Allo for cryptophytes, ButFuco for the ButFuco-bearing flagellates), as in the diagnostic pigment analysis of Vidussi et al. (2001) and Uitz et al. (2006). Neo is listed only for green algae in Table 1. Chrysophytes have no pigment of their own in this set, so no cluster can be named after them.
DIAGNOSTIC_PIGMENTS: dict[str, str] = {
    "Fuco": "Diatoms",
    "Perid": "Dinoflagellates",
    "ButFuco": "Pelagophytes",
    "HexFuco": "Haptophytes",
    "Allo": "Cryptophytes",
    "MV_chlb": "Green algae",
    "Neo": "Green algae",
    "Zea": "Cyanobacteria",
    "DV_chla": "Cyanobacteria",
}
# The same table read the other way: the carrier pigments of each PFT
GROUP_PIGMENTS: dict[str, tuple[str, ...]] = {
    "Diatoms": ("Fuco", "Chlc12", "Chlc3"),
    "Dinoflagellates": ("Fuco", "Perid", "Chlc12", "Chlc3"),
    "Chrysophytes": ("Fuco", "Chlc12", "Viola", "Zea"),
    "Pelagophytes": ("Fuco", "ButFuco", "Chlc12", "Zea"),
    "Haptophytes": ("Fuco", "HexFuco", "ButFuco", "Chlc12", "Chlc3"),
    "Cryptophytes": ("Chlc12", "Allo"),
    "Green algae": ("MV_chlb", "Neo", "Viola", "Zea"),
    "Cyanobacteria": ("Zea", "DV_chla"),
}


class PigmentCluster(NamedTuple):
    cluster_id: int
    # PFTs with at least one diagnostic pigment in this cluster, in GROUP_PIGMENTS order; empty when none
    labels: tuple[str, ...]
    # Member pigments in RATIO_PIGMENTS order
    pigments: tuple[str, ...]
    # For each label, its carrier pigments that sit in this cluster
    carriers: dict[str, tuple[str, ...]]
    # For each label, its diagnostic pigments that sit in this cluster, the ones that earned the label
    markers: dict[str, tuple[str, ...]]

    @property
    def label(self) -> str:
        return ", ".join(self.labels)

    @property
    def carrier_pigments(self) -> tuple[str, ...]:
        carried = {pigment for pigments in self.carriers.values() for pigment in pigments}
        return tuple(pigment for pigment in self.pigments if pigment in carried)


def load_target_composites(data_dir: Path, cfg: dict[str, Any]) -> pd.DataFrame:
    eddy_tracks = restrict_to_analysis_period(pd.read_parquet(data_dir / "silver/gulf_stream/eddy_movement.parquet"), cfg)
    pigment_table = keep_valid_sdp_rows(restrict_to_analysis_period(pd.read_parquet(data_dir / "gold/eddy_pigment_table.parquet"), cfg))
    pigment_table["polarity"] = pigment_table["polarity"].map({0: "anticyclone", 1: "cyclone"})  # pyright: ignore[reportArgumentType]
    pigment_table = pigment_table.rename(columns={f"eddy_mean_{pigment}": pigment for pigment in SDP_PIGMENTS})
    # Target eddies are cyclones that go from north to south of the axis and anticyclones from south to north
    is_target = eddy_tracks["movement"].eq(eddy_tracks["polarity"].map({"cyclone": "NS", "anticyclone": "SN"}))  # pyright: ignore[reportArgumentType]
    return pigment_table.merge(eddy_tracks.loc[is_target, ["polarity", "track_id"]], on=["polarity", "track_id"])


# Pearson correlation between the pigment / Tchla ratios, one sample per eddy-composite interior mean
def correlate_ratios(composites: pd.DataFrame, pigments: tuple[str, ...] = RATIO_PIGMENTS) -> pd.DataFrame:
    return cast(pd.DataFrame, composites[list(pigments)].div(composites["Tchla"], axis=0)).corr()


def build_ward_linkage(correlation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # sqrt(2(1 - r)) is the Euclidean distance between standardized pigment vectors, which Ward requires; 2 - r - r.T keeps it symmetric
    distance = squareform(np.sqrt(2 - correlation - correlation.T), checks=False)
    return distance, linkage(distance, method="ward")


# Clusters come in the order of their first pigment in clusters, so the order does not depend on how fcluster numbers them
def label_clusters(clusters: pd.Series) -> tuple[PigmentCluster, ...]:
    clusters_in_order = pd.unique(clusters.to_numpy())
    result = []
    for cluster_id in clusters_in_order:
        members = tuple(clusters.loc[clusters.eq(cluster_id)].index)
        markers = {group: tuple(pigment for pigment in members if DIAGNOSTIC_PIGMENTS.get(pigment) == group) for group in GROUP_PIGMENTS}
        labels = tuple(group for group, inside in markers.items() if inside)
        carriers = {group: tuple(pigment for pigment in members if pigment in GROUP_PIGMENTS[group]) for group in labels}
        result.append(PigmentCluster(int(cluster_id), labels, members, carriers, {group: markers[group] for group in labels}))
    return tuple(result)


def resolve_pigment_clusters(data_dir: Path, cfg: dict[str, Any], cut_height: float = CLUSTER_CUT_HEIGHT) -> tuple[PigmentCluster, ...]:
    correlation = correlate_ratios(load_target_composites(data_dir, cfg))
    _, tree = build_ward_linkage(correlation.to_numpy())
    clusters = pd.Series(fcluster(tree, t=cut_height, criterion="distance"), index=correlation.index, name="cluster")
    return label_clusters(clusters)
