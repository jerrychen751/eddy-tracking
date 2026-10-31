"""
Track eddies across daily identification files using PET Correspondances.

Builds frame-to-frame eddy correspondences from the daily .nc files produced by eddy_id.py, applies the --min-track-days minimum track duration, interpolates virtual observations, and writes merged tracks to Zarr.
"""

import argparse
import shutil
from pathlib import Path
from typing import cast

from numpy import concatenate, ones

from eddy_tracking.packages.py_eddy_tracker.generic import distance_grid
from eddy_tracking.packages.py_eddy_tracker.observations.observation import EddiesObservations
from eddy_tracking.packages.py_eddy_tracker.tracking import Correspondances
from eddy_tracking.utils.cli import write_settings


def _remove_path(path: Path) -> None:
    """Remove a zarr directory or a stray same-named file from a prior write."""
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


class SpeedLimitedObservations(EddiesObservations):
    # PET alone accepts any detection within 125 km of a virtual observation's extrapolated position, however many days the eddy has been missing.
    # This keeps that test and also requires the jump from the last real detection to fit inside both speed radii plus a maximum drift per elapsed day.
    __slots__ = ("max_speed_km_per_day", "days_since_detection", "detected_lon", "detected_lat")

    def __init__(self, *args, max_speed_km_per_day=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_speed_km_per_day = max_speed_km_per_day
        self.days_since_detection = None
        self.detected_lon = None
        self.detected_lat = None

    def merge(self, other):
        # Correspondances.track appends the virtual observations to the previous day's detections, and the merged copy keeps only the shared fields, so record each row's gap here.
        eddies = super().merge(other)
        eddies.max_speed_km_per_day = self.max_speed_km_per_day
        # A virtual observation with segment_size k sits k daily steps past its last detection and is matched against the day after, k + 1 days from that detection.
        segment_size = other["segment_size"]
        eddies.days_since_detection = concatenate((ones(len(self)), segment_size + 1))
        eddies.detected_lon = concatenate((self.lon, other.lon - segment_size * other["dlon"]))
        eddies.detected_lat = concatenate((self.lat, other.lat - segment_size * other["dlat"]))
        return eddies

    def mask_function(self, other, distance):
        if self.days_since_detection is None:
            # Before the first virtual observation exists, every row is a detection from the previous day.
            days, lon, lat = ones(len(self)), self.lon, self.lat
        else:
            days, lon, lat = self.days_since_detection, self.detected_lon, self.detected_lat
        jump_km = distance_grid(lon, lat, other.lon, other.lat)
        # load_file passes the speed through class_kw and merge copies it, so only the bare constructor inside new_like leaves it None.
        max_speed_km_per_day = cast(float, self.max_speed_km_per_day)
        # radius_s is in meters.
        reach_km = (
            self.radius_s[:, None] / 1000
            + other.radius_s[None, :] / 1000
            + max_speed_km_per_day * days[:, None]
        )
        return super().mask_function(other, distance) & (jump_km <= reach_km)


def track(
    id_dir: Path,
    track_dir: Path,
    *,
    virtual: int,
    max_speed_km_per_day: float,
    min_track_days: int,
) -> None:
    """Track daily eddy IDs and replace the output after a temporary write."""
    polarity = track_dir.name
    identification_files = sorted(id_dir.glob("*.nc"))
    if not identification_files:
        print(
            f"polarity: {polarity}\n"
            "status: skipped\n"
            "reason: no_nc_files\n"
            f"input_dir: {id_dir}"
        )
        return

    print(
        f"polarity: {polarity}\n"
        "status: tracking\n"
        f"daily_files: {len(identification_files)}"
    )

    correspondences = Correspondances(
        datasets=identification_files,
        virtual=virtual,
        class_method=SpeedLimitedObservations,
        class_kw={"max_speed_km_per_day": max_speed_km_per_day},
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

    # PET's position_filter is left out because it smooths only lon and lat, which moved centers off their own speed contours.

    output_path = track_dir / f"{polarity}_tracks.zarr"
    temporary_path = track_dir / f"{polarity}_tracks.tmp.zarr"
    _remove_path(temporary_path)
    tracked_observations.write_file(filename=str(temporary_path))
    # Preserve the current output until its replacement is fully written.
    _remove_path(output_path)
    temporary_path.rename(output_path)
    print(
        f"polarity: {polarity}\n"
        f"output_path: {output_path}\n"
        "status: written"
    )


def main(argv: list[str]) -> None:
    """Track both polarities and write their Zarr datasets."""
    parser = argparse.ArgumentParser(prog="eddy-tracking track-eddies")
    parser.add_argument("--eddies", type=Path, required=True, help="folder identify-eddies wrote, holding cyclone/ and anticyclone/")
    parser.add_argument("--virtual-days", type=int, required=True, help="consecutive missed days before a track breaks")
    parser.add_argument("--max-speed-km-per-day", type=float, required=True, help="drift allowed per elapsed day beyond the two speed radii")
    parser.add_argument("--min-track-days", type=int, required=True, help="tracks shorter than this are dropped")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    write_settings(args.out, "track-eddies", args, operational=set())

    for polarity in ("anticyclone", "cyclone"):
        track_dir = args.out / polarity
        track_dir.mkdir(parents=True, exist_ok=True)

        track(
            args.eddies / polarity,
            track_dir,
            virtual=args.virtual_days,
            max_speed_km_per_day=args.max_speed_km_per_day,
            min_track_days=args.min_track_days,
        )
