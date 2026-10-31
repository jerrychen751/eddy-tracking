"""Load a study's analysis settings and the settings records of its outputs, and resolve medallion-layer paths."""

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Column order is positional: it must match the np.column_stack order in collocate_pace.py and the enumerate-insert loop in run_sdp.py.
METADATA_COLS: list[str] = [
    "track_id", "date", "pixel_lon", "pixel_lat",
    "center_lon", "center_lat", "radius_km", "inside_contour", "coverage",
]


def _validate_experiment(experiment: str) -> None:
    if Path(experiment).is_absolute():
        raise ValueError(f"experiment must not be an absolute path: {experiment!r}")
    if ".." in Path(experiment).parts:
        raise ValueError(f"experiment name must not contain '..': {experiment!r}")


def resolve_bronze_dir(experiment: str, *parts: str) -> Path:
    _validate_experiment(experiment)
    bronze_dir = PROJECT_ROOT / "data" / experiment / "bronze"
    for part in parts:
        bronze_dir /= part
    return bronze_dir


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


# configs/<experiment>/analysis.yaml holds the settings that belong to the analysis rather than to a tool; the tools' own settings sit in the settings.yaml beside each output
def load_analysis_config(experiment: str) -> dict[str, Any]:
    _validate_experiment(experiment)
    config_path = PROJECT_ROOT / "configs" / experiment / "analysis.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


# The settings of the latest run of a tool recorded for an output folder (its settings.yaml) or an output file (<name>.settings.yaml beside it)
def read_settings(output: Path, tool: str) -> dict[str, Any]:
    record_path = output.with_name(f"{output.name}.settings.yaml") if output.suffix else output / "settings.yaml"
    with record_path.open(encoding="utf-8") as record_file:
        return [run for run in yaml.safe_load(record_file) if run["tool"] == tool][-1]["settings"]
