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
    Returns a Path to the local directory for a data subdirectory.

    Builds the path as: data / <dataset> / <dir_key value>.
    Creates the directory if it does not exist yet.

    Args:
        cfg: The full merged config dict (must contain a "base" key).
        dir_key: Key name in cfg["base"]["data"] (e.g. "swot_dir").
    """
    base = cfg["base"]
    dataset: str = base["dataset"]
    root_dir: str = base["data"]["root"]
    dest = PROJECT_ROOT / root_dir / dataset / base["data"][dir_key]
    dest.mkdir(parents=True, exist_ok=True)
    return dest

def resolve_output_dir(experiment: str, *stages: str) -> Path:
    """
    Returns a Path to an output directory namespaced by experiment.

    Builds the path as: outputs / <experiment> / <stage1> / <stage2> / ...
    Creates the directory if it does not exist yet.

    Args:
        experiment: Name of the experiment (e.g. "gulf_stream_cyclonic").
        stages: One or more path segments (e.g. "eddy_id", "anticyclone").
    """
    _validate_experiment(experiment)
    dest = PROJECT_ROOT / "outputs" / experiment
    for stage in stages:
        dest = dest / stage
    dest.mkdir(parents=True, exist_ok=True)
    return dest

def load_config(experiment: str, *filenames: str) -> dict[str, Any]:
    """
    Returns a dictionary of the specified config files (expects file extension).

    The filename (without extension) is the key, and the parsed YAML content
    is the value. Config files are read from configs/<experiment>/.

    Args:
        experiment: Name of the experiment subfolder under configs/.
        filenames: YAML filenames to load (e.g. "base.yaml", "eddy_id.yaml").
    """
    _validate_experiment(experiment)
    cfg_dir = PROJECT_ROOT / "configs" / experiment
    if not cfg_dir.exists():
        raise FileNotFoundError(
            f"Experiment config directory not found: {cfg_dir}"
        )

    cfg = {}
    for name in filenames:
        fp = cfg_dir / name
        with open(fp) as f:
            cfg[Path(name).stem] = yaml.safe_load(f)

    return cfg
