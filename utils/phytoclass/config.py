"""
Default configuration: F matrix, ratio bounds, and column name mappings.
"""

from pathlib import Path
import pandas as pd
import numpy as np

_DEFAULTS_DIR = Path(__file__).resolve().parent / "defaults"

# Mapping from SDP display names -> internal phytoclass names (underscored, no spaces)
SDP_TO_INTERNAL = {
    "T chla": "Tchla",
    "Zea": "Zea",
    "DV chla": "DV_chla",    # Marker for Cyanobacteria (Prochlorococcus component)
    "ButFuco": "ButFuco",
    "HexFuco": "HexFuco",
    "Allo": "Allo",
    "MV chlb": "MV_chlb",
    "Neo": "Neo",
    "Viola": "Viola",
    "Fuco": "Fuco",
    "chl c1+c2": "Chlc12",   # Accessory pigment for Diatoms
    "chl c3": "Chlc3",       # Marker for Haptophytes
    "Perid": "Perid",
}


def load_default_f_matrix() -> tuple[pd.DataFrame, list[str]]:
    """
    Load the default binary F matrix.

    Returns:
        Tuple of (F DataFrame indexed by class name, list of pigment column names).
    """
    df = pd.read_csv(_DEFAULTS_DIR / "f_matrix.csv", index_col="class")
    pigment_cols = [c for c in df.columns if c != "Tchla"]
    return df, pigment_cols


def load_default_bounds() -> pd.DataFrame:
    """
    Load the default min/max ratio bounds.

    Returns:
        DataFrame with columns: class, pigment, min, max.
    """
    return pd.read_csv(_DEFAULTS_DIR / "min_max.csv")
