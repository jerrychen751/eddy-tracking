"""Download PACE OCI L3 mapped Rrs files for one experiment."""

import argparse

from utils.config import load_config, resolve_data_dir
from utils.download_pace_l3 import download_pace_l3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    return parser.parse_args()


def main(experiment: str | None = None) -> None:
    """Download and save PACE files, exiting if any date fails."""
    if experiment is None:
        experiment = _parse_args().experiment

    cfg = load_config(experiment)
    longitude_range = tuple(cfg["base"]["region"]["lon_range"])
    latitude_range = tuple(cfg["base"]["region"]["lat_range"])
    date_range = tuple(cfg["base"]["time"]["rrs_date_range"])
    temporal_resolution = cfg["base"]["download"]["pace"].get(
        "temporal_resolution", "DAY"
    )

    n_saved, n_skipped, n_errors = download_pace_l3(
        date_range=date_range,
        lon_range=longitude_range,
        lat_range=latitude_range,
        out_dir=resolve_data_dir(cfg, "pace_dir"),
        temporal_res=temporal_resolution,
    )

    print(f"Done. {n_saved} saved, {n_skipped} skipped, {n_errors} errors.")
    if n_errors:
        raise SystemExit(f"{n_errors} date(s) failed to download")


if __name__ == "__main__":
    main()
