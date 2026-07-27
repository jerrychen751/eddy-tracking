"""
Track eddies across daily identification files using PET Correspondances.

Builds frame-to-frame eddy correspondences from the daily .nc files produced
by eddy_id.py, applies the configured minimum track duration, interpolates
virtual observations, smooths positions, and writes merged tracks to Zarr.
"""

import argparse
import shutil
from pathlib import Path

from utils.config import load_config, resolve_output_dir
from eddy_tracking.packages.py_eddy_tracker.tracking import Correspondances


def _remove_path(path: Path) -> None:
    """Remove a zarr directory or a stray same-named file from a prior write."""
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def track(
    id_dir: Path,
    track_dir: Path,
    *,
    virtual: int,
    min_track_days: int,
    median_half_window: int,
    loess_half_window: int,
) -> None:
    """Track daily eddy IDs and replace the output after a temporary write."""
    polarity = track_dir.name
    identification_files = sorted(id_dir.glob("*.nc"))
    if not identification_files:
        print(f"[{polarity}] No .nc files found in {id_dir}, skipping")
        return

    print(f"[{polarity}] Tracking {len(identification_files)} daily files...")

    correspondences = Correspondances(
        datasets=identification_files,
        virtual=virtual,
    )
    correspondences.track()

    correspondences.prepare_merging()
    correspondences.longer_than(min_track_days)
    tracked_observations = correspondences.merge(raw_data=False)

    # PET marks missing detections with time == 0 before interpolation.
    virtual_mask = tracked_observations.time == 0
    tracked_observations.virtual[:] = virtual_mask
    tracked_observations.filled_by_interpolation(
        tracked_observations.virtual == 1
    )

    tracked_observations.position_filter(
        median_half_window=median_half_window,
        loess_half_window=loess_half_window,
    )

    output_path = track_dir / f"{polarity}_tracks.zarr"
    temporary_path = track_dir / f"{polarity}_tracks.tmp.zarr"
    _remove_path(temporary_path)
    tracked_observations.write_file(filename=str(temporary_path))
    # Preserve the current output until its replacement is fully written.
    _remove_path(output_path)
    temporary_path.rename(output_path)
    print(f"[{polarity}] Wrote {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    return parser.parse_args()


def main(experiment: str | None = None) -> None:
    """Track both polarities for an experiment and write their Zarr datasets."""
    if experiment is None:
        experiment = _parse_args().experiment

    cfg = load_config(experiment)
    tracking_cfg = cfg["eddy_track"]
    filter_cfg = tracking_cfg["position_filter"]

    for polarity in ("anticyclone", "cyclone"):
        id_dir = resolve_output_dir(experiment, "eddy_id", polarity)
        track_dir = resolve_output_dir(experiment, "eddy_track", polarity)
        track_dir.mkdir(parents=True, exist_ok=True)

        track(
            id_dir,
            track_dir,
            virtual=tracking_cfg["virtual"],
            min_track_days=tracking_cfg["min_track_days"],
            median_half_window=filter_cfg["median_half_window"],
            loess_half_window=filter_cfg["loess_half_window"],
        )


if __name__ == "__main__":
    main()
