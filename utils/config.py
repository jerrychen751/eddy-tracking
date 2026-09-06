"""Load experiment configuration and resolve medallion-layer paths."""

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Column order is positional: it must match the np.column_stack order in collocate_pace.py and the enumerate-insert loop in run_sdp.py.
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
    """Return the configured bronze directory, creating it if needed."""
    base = cfg["base"]
    data_dir = (
        PROJECT_ROOT
        / base["data"]["root"]
        / base["dataset"]
        / "bronze"
        / base["data"][dir_key]
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def resolve_output_dir(experiment: str, *stages: str) -> Path:
    """Return a silver-stage directory, creating it and its parents if needed."""
    _validate_experiment(experiment)
    output_dir = PROJECT_ROOT / "data" / experiment / "silver"
    for stage in stages:
        output_dir /= stage
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_gold_dir(experiment: str, *parts: str) -> Path:
    """Return a gold-layer path, creating only the experiment's gold directory."""
    _validate_experiment(experiment)
    gold_dir = PROJECT_ROOT / "data" / experiment / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    for part in parts:
        gold_dir /= part
    return gold_dir


def load_config(experiment: str) -> dict[str, Any]:
    """Load ``configs/<experiment>/config.yaml`` by stage section."""
    _validate_experiment(experiment)
    config_path = PROJECT_ROOT / "configs" / experiment / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    with config_path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)
