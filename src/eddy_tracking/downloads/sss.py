"""Download SMAP SSS files through Harmony."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
from pathlib import Path

from eddy_tracking.downloads.auth import login_harmony


def download_smap_sss_8d(
    date_range: tuple[str, str],
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    out_dir: Path,
    raw_tmp: Path,
    collection_id: str,
    batch_days: int = 10,
    num_requests_workers: int = 4,
    download_chunk_size_mb: int = 50,
) -> tuple[int, int]:
    """
    Download SMAP L3 8-day SSS through Harmony, one request per batch_days window, which keeps each request under the Harmony size limit.

    Creates out_dir and raw_tmp, writes one NetCDF per granule into out_dir, and deletes raw_tmp on exit.
    Sets the NUM_REQUESTS_WORKERS and DOWNLOAD_CHUNK_SIZE environment variables.
    Returns (files_saved, failed_windows).
    """
    import os as _os

    # harmony reads NUM_REQUESTS_WORKERS and DOWNLOAD_CHUNK_SIZE at import time, so set them before the harmony import below.
    _os.environ["NUM_REQUESTS_WORKERS"] = str(num_requests_workers)
    _os.environ["DOWNLOAD_CHUNK_SIZE"] = str(download_chunk_size_mb * 1024 * 1024)

    from harmony import BBox, Collection, Request

    print(
        "status: downloading_sss\n"
        "product: smap_l3_8_day_running_mean"
    )
    client = login_harmony()
    collection = Collection(id=collection_id)

    start = dt.datetime.strptime(date_range[0], "%Y-%m-%d")
    end = dt.datetime.strptime(date_range[1], "%Y-%m-%d")

    windows = []
    window_start = start
    while window_start < end:
        window_end = min(window_start + dt.timedelta(days=batch_days), end)
        windows.append((window_start, window_end))
        window_start = window_end

    out_dir = Path(out_dir)
    raw_tmp = Path(raw_tmp)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_tmp.mkdir(parents=True, exist_ok=True)

    saved = 0
    failed = 0

    try:
        for i, (win_start, win_end) in enumerate(windows, 1):
            label = win_start.strftime("%Y-%m")
            print(
                f"window: {i}\n"
                f"total_windows: {len(windows)}\n"
                f"month: {label}\n"
                f"start_date: {win_start.date()}\n"
                f"end_date: {win_end.date()}"
            )

            request = Request(
                collection=collection,
                spatial=BBox(lon_range[0], lat_range[0], lon_range[1], lat_range[1]),
                temporal={"start": win_start, "stop": win_end},
                granule_name=["*8DAYS*"],
                max_results=200,
                skip_preview=True,
            )

            try:
                job_id = client.submit(request)
                futures = client.download_all(
                    job_id, directory=str(raw_tmp), overwrite=True
                )
                raw_files = [Path(f.result()) for f in futures]
            except Exception as exc:
                print(
                    "status: error\n"
                    f"error: {exc}"
                )
                failed += 1
                continue

            for raw_path in raw_files:
                # Harmony stages each file as "{item_id}_{original_name}", such as "12345_SMAP_L3_SSS_20250114_8DAYS_V5.0.nc", and item_id changes per job.
                stable_name = re.sub(r"^\d+_", "", raw_path.name)
                out_path = out_dir / stable_name
                if out_path.exists():
                    continue
                shutil.copy2(raw_path, out_path)
                saved += 1

            for f in raw_tmp.iterdir():
                f.unlink()
    finally:
        shutil.rmtree(raw_tmp, ignore_errors=True)

    return saved, failed


def main(experiment: str | None = None) -> None:
    """Download configured SSS files and exit if a window fails."""
    if experiment is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("experiment")
        experiment = parser.parse_args().experiment

    from utils.config import load_config, resolve_data_dir

    cfg = load_config(experiment)
    n_saved, n_failed = download_smap_sss_8d(
        date_range=tuple(cfg["base"]["time"]["rrs_date_range"]),
        lon_range=tuple(cfg["base"]["region"]["lon_range"]),
        lat_range=tuple(cfg["base"]["region"]["lat_range"]),
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

    print(
        "status: download_finished\n"
        f"sss_files_saved: {n_saved}\n"
        f"windows_failed: {n_failed}"
    )
    if n_failed:
        raise SystemExit(f"{n_failed} SSS window(s) failed to download")


if __name__ == "__main__":
    main()
