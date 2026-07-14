"""Download AQUA MODIS SST and SMAP SSS files for one experiment."""

import argparse

from utils.config import load_config, resolve_data_dir
from utils.download_ancillary import (
    download_aqua_sst_8d_4km,
    download_smap_sss_8d,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("--only", choices=["sst", "sss"])
    return parser.parse_args()


def main(experiment: str | None = None, only: str | None = None) -> None:
    """Download configured SST and SSS files, or only the selected dataset."""
    if experiment is None:
        args = _parse_args()
        experiment = args.experiment
        only = args.only

    cfg = load_config(experiment)
    longitude_range = tuple(cfg["base"]["region"]["lon_range"])
    latitude_range = tuple(cfg["base"]["region"]["lat_range"])
    date_range = tuple(cfg["base"]["time"]["rrs_date_range"])

    run_sst = only in (None, "sst")
    run_sss = only in (None, "sss")
    n_sst = 0
    n_sss_saved = 0
    n_sss_failed = 0

    if run_sst:
        n_sst = download_aqua_sst_8d_4km(
            date_range=date_range,
            lon_range=longitude_range,
            lat_range=latitude_range,
            out_dir=resolve_data_dir(cfg, "sst_dir"),
            raw_tmp=resolve_data_dir(cfg, "sst_raw_tmp"),
            collection_id=cfg["base"]["download"]["sst"]["collection_id"],
            download_threads=cfg["base"]["download"]["sst"][
                "download_threads"
            ],
        )

    if run_sss:
        n_sss_saved, n_sss_failed = download_smap_sss_8d(
            date_range=date_range,
            lon_range=longitude_range,
            lat_range=latitude_range,
            out_dir=resolve_data_dir(cfg, "sss_dir"),
            raw_tmp=resolve_data_dir(cfg, "sss_raw_tmp"),
            collection_id=cfg["base"]["download"]["sss"]["collection_id"],
            batch_days=10,
            num_requests_workers=cfg["base"]["download"]["sss"][
                "num_requests_workers"
            ],
            download_chunk_size_mb=cfg["base"]["download"]["sss"][
                "download_chunk_size_mb"
            ],
        )

    print(f"Done. {n_sst} SST + {n_sss_saved} SSS files saved.")
    if n_sss_failed:
        raise SystemExit(f"{n_sss_failed} SSS window(s) failed to download")


if __name__ == "__main__":
    main()
