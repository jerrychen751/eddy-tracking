"""
Config loader for the eddy-tracking project.

Reads YAML files from configs/<experiment>/ and provides helpers
for resolving data (input) and output directories.
"""

import yaml
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Metadata columns shared by collocate_pace, run_sdp, and run_phytoclass.
# Any change to the collocation output schema must be reflected here.
METADATA_COLS: list[str] = [
    "track_id", "date", "pixel_lon", "pixel_lat",
    "center_lon", "center_lat", "coverage",
]

def _validate_experiment(experiment: str) -> None:
    if Path(experiment).is_absolute():
        raise ValueError(f"experiment must not be an absolute path: {experiment!r}")
    if ".." in Path(experiment).parts:
        raise ValueError(f"experiment name must not contain '..': {experiment!r}")


def resolve_data_dir(cfg: dict[str, Any], dir_key: str) -> Path:
    """
    Returns the bronze-layer (raw download) directory for a data subdir.

    Builds the path as: data / <dataset> / bronze / <dir_key value>.
    Creates the directory if it does not exist yet.

    Args:
        cfg: The full merged config dict (must contain a "base" key).
        dir_key: Key name in cfg["base"]["data"] (e.g. "swot_dir").
    """
    base = cfg["base"]
    dest = PROJECT_ROOT / base["data"]["root"] / base["dataset"] / "bronze" / base["data"][dir_key]
    dest.mkdir(parents=True, exist_ok=True)
    return dest

def resolve_output_dir(experiment: str, *stages: str) -> Path:
    """
    Returns a silver-layer (intermediate processed) directory for a stage.

    Builds the path as: data / <experiment> / silver / <stage1> / <stage2> / ...
    Creates the directory if it does not exist yet.

    Args:
        experiment: Name of the experiment subfolder under configs/.
        stages: One or more path segments (e.g. "eddy_id", "anticyclone").
    """
    _validate_experiment(experiment)
    dest = PROJECT_ROOT / "data" / experiment / "silver"
    for stage in stages:
        dest = dest / stage
    dest.mkdir(parents=True, exist_ok=True)
    return dest

def resolve_gold_dir(experiment: str, *parts: str) -> Path:
    """
    Returns a gold-layer (analysis-ready) path: data / <experiment> / gold / <parts>.

    Creates the gold/ base directory; the returned path may be a file (e.g. the
    table parquet) or a subdirectory the caller creates.
    """
    _validate_experiment(experiment)
    dest = PROJECT_ROOT / "data" / experiment / "gold"
    dest.mkdir(parents=True, exist_ok=True)
    for part in parts:
        dest = dest / part
    return dest

def load_config(experiment: str) -> dict[str, Any]:
    """
    Load configs/<experiment>/config.yaml, parsed and keyed by stage section.

    Sections are the top-level keys (cfg["base"], cfg["eddy_id"], ...).
    """
    _validate_experiment(experiment)
    cfg_path = PROJECT_ROOT / "configs" / experiment / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {cfg_path}")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def resolve_config_file(experiment: str, relative_path: str) -> Path:
    """
    Returns an absolute Path to a file living inside the experiment's config dir.

    Used by pipeline scripts to resolve paths that phytoclass.yaml declares
    relative to its own directory (e.g. f_matrix.csv, min_max.csv). Keeps
    config portable across machines and across experiment renames.

    Args:
        experiment: Name of the experiment subfolder under configs/.
        relative_path: Path relative to configs/<experiment>/.
    """
    _validate_experiment(experiment)
    return PROJECT_ROOT / "configs" / experiment / relative_path
