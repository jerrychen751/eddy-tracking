"""Download SMAP SSS files through Harmony."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
from pathlib import Path

from eddy_tracking.downloads.auth import login_harmony
from eddy_tracking.utils.cli import add_region_arguments, write_settings


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
    Download SMAP L3 8-day SSS through Harmony, one request per batch_days date range, which keeps each request under the Harmony size limit.

    Creates out_dir and raw_tmp, writes one NetCDF per granule into out_dir, and deletes raw_tmp on exit. Sets the NUM_REQUESTS_WORKERS and DOWNLOAD_CHUNK_SIZE environment variables. Returns (files_saved, failed_date_ranges).
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

    batch_date_ranges = []
    batch_start = start
    while batch_start < end:
        batch_end = min(batch_start + dt.timedelta(days=batch_days), end)
        batch_date_ranges.append((batch_start, batch_end))
        batch_start = batch_end

    out_dir = Path(out_dir)
    raw_tmp = Path(raw_tmp)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_tmp.mkdir(parents=True, exist_ok=True)

    saved = 0
    failed = 0

    try:
        for i, (batch_start, batch_end) in enumerate(batch_date_ranges, 1):
            label = batch_start.strftime("%Y-%m")
            print(
                f"batch: {i}\n"
                f"total_batches: {len(batch_date_ranges)}\n"
                f"month: {label}\n"
                f"start_date: {batch_start.date()}\n"
                f"end_date: {batch_end.date()}"
            )

            request = Request(
                collection=collection,
                spatial=BBox(lon_range[0], lat_range[0], lon_range[1], lat_range[1]),
                temporal={"start": batch_start, "stop": batch_end},
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


def main(argv: list[str]) -> None:
    """Download the --dates SSS files and exit if a date range fails."""
    parser = argparse.ArgumentParser(prog="eddy-tracking download-sss")
    add_region_arguments(parser)
    parser.add_argument("--collection-id", required=True, help="CMR collection id of SMAP L3 8-day SSS, such as C2208422957-POCLOUD")
    parser.add_argument("--requests-workers", type=int, default=4, help="parallel Harmony download connections")
    parser.add_argument("--download-chunk-size-mb", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    write_settings(args.out, "download-sss", args, operational={"requests_workers", "download_chunk_size_mb"})
    args.out.mkdir(parents=True, exist_ok=True)

    # Harmony stages each batch in raw_tmp/ under --out, which download_smap_sss_8d deletes on exit; readers glob only the top level of --out
    n_saved, n_failed = download_smap_sss_8d(
        date_range=tuple(args.dates),
        lon_range=tuple(args.lon),
        lat_range=tuple(args.lat),
        out_dir=args.out,
        raw_tmp=args.out / "raw_tmp",
        collection_id=args.collection_id,
        batch_days=10,
        num_requests_workers=args.requests_workers,
        download_chunk_size_mb=args.download_chunk_size_mb,
    )

    print(
        "status: download_finished\n"
        f"sss_files_saved: {n_saved}\n"
        f"date_ranges_failed: {n_failed}"
    )
    if n_failed:
        raise SystemExit(f"{n_failed} SSS date range(s) failed to download")
