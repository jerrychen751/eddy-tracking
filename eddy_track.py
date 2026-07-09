"""
Track eddies across daily identification files using PET Correspondances.

Builds frame-to-frame eddy correspondences from the daily .nc files produced
by eddy_id.py, filters to tracks longer than MIN_TRACK_DAYS, interpolates
virtual observations, smooths positions, and writes merged tracks to a Zarr
file.
"""

import argparse
import shutil
from pathlib import Path

from utils.py_eddy_tracker.tracking import Correspondances

from utils.config import resolve_output_dir, load_config

parser = argparse.ArgumentParser()
parser.add_argument("experiment")
args = parser.parse_args()

cfg = load_config(args.experiment)

ANTICYCLONE_ID_DIR = resolve_output_dir(args.experiment, "eddy_id", "anticyclone")
CYCLONE_ID_DIR = resolve_output_dir(args.experiment, "eddy_id", "cyclone")

ANTICYCLONE_TRACK_DIR = resolve_output_dir(args.experiment, "eddy_track", "anticyclone")
CYCLONE_TRACK_DIR = resolve_output_dir(args.experiment, "eddy_track", "cyclone")

VIRTUAL = cfg["eddy_track"]["virtual"]
MIN_TRACK_DAYS = cfg["eddy_track"]["min_track_days"]
MEDIAN_HALF_WINDOW = cfg["eddy_track"]["position_filter"]["median_half_window"]
LOESS_HALF_WINDOW = cfg["eddy_track"]["position_filter"]["loess_half_window"]


def _remove_path(path: Path) -> None:
    """Remove a zarr directory or a stray same-named file from a prior write."""
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def track(id_dir: Path, track_dir: Path) -> None:
    """
    Run PET tracking on all daily eddy ID files in id_dir, write result to track_dir.

    Steps: build correspondences across consecutive days, filter to tracks
    longer than MIN_TRACK_DAYS, interpolate virtual observations, and write
    merged tracks to a Zarr file.
    """
    name = track_dir.name
    files = sorted(id_dir.glob("*.nc"))
    if not files:
        print(f"[{name}] No .nc files found in {id_dir}, skipping")
        return

    print(f"[{name}] Tracking {len(files)} daily files...")

    corr = Correspondances(
        datasets=files,
        virtual=VIRTUAL,
    )
    corr.track()

    corr.prepare_merging()
    corr.longer_than(MIN_TRACK_DAYS)
    tracked = corr.merge(raw_data=False)

    # Interpolate virtual observations (timesteps where the eddy existed
    # but wasn't detected). PET marks these with time == 0.
    virtual_mask = tracked.time == 0
    tracked.virtual[:] = virtual_mask
    tracked.filled_by_interpolation(tracked.virtual == 1)

    tracked.position_filter(
        median_half_window=MEDIAN_HALF_WINDOW,
        loess_half_window=LOESS_HALF_WINDOW,
    )

    out_path = track_dir / f"{name}_tracks.zarr"
    tmp_path = track_dir / f"{name}_tracks.tmp.zarr"
    _remove_path(tmp_path)
    tracked.write_file(filename=str(tmp_path))
    # Only remove the old zarr after the new one is fully written
    _remove_path(out_path)
    tmp_path.rename(out_path)
    print(f"[{name}] Wrote {out_path}")


def main():
    ANTICYCLONE_TRACK_DIR.mkdir(parents=True, exist_ok=True)
    CYCLONE_TRACK_DIR.mkdir(parents=True, exist_ok=True)

    track(ANTICYCLONE_ID_DIR, ANTICYCLONE_TRACK_DIR)
    track(CYCLONE_ID_DIR, CYCLONE_TRACK_DIR)


if __name__ == "__main__":
    main()
