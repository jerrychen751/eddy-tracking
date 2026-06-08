"""Download daily L4 SSH files from AVISO FTP in parallel, then trim to region."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from ftplib import FTP
from pathlib import Path
import os
import re
import sys
import tempfile

from dotenv import load_dotenv
import xarray as xr

from utils.config import load_config, resolve_data_dir

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("experiment")
args = parser.parse_args()

# Config from YAML
cfg = load_config(args.experiment, "base.yaml")

REMOTE_DIR = cfg["base"]["download"]["swot"]["ftp_dir"]
LOCAL_DIR = resolve_data_dir(cfg, "swot_dir")
MAX_WORKERS = cfg["base"]["download"]["swot"]["max_workers"]
DATE_RANGE = tuple(cfg["base"]["time"]["eddy_date_range"])
LON_RANGE = tuple(cfg["base"]["region"]["lon_range"])
LAT_RANGE = tuple(cfg["base"]["region"]["lat_range"])

# Credentials from .env — fail early with a clear message rather than a bare KeyError
for _var in ("FTP_HOST", "FTP_USER", "FTP_PASSWORD"):
    if not os.environ.get(_var):
        sys.exit(f"Missing required env var: {_var} (set in .env or environment)")

HOST = os.environ["FTP_HOST"]
USER = os.environ["FTP_USER"]
PASSWORD = os.environ["FTP_PASSWORD"]

def list_remote_files_with_sizes() -> list[tuple[str, int | None]]:
    """
    Like list_remote_files, but also returns file sizes in bytes.
    """
    with FTP(HOST, USER, PASSWORD) as ftp:
        files = [f for f in ftp.nlst(REMOTE_DIR) if f.endswith('.nc')]
        ftp.voidcmd("TYPE I")  # switch to binary mode (must be after nlst)
        sized = [(f, ftp.size(f)) for f in files]

    return sized

def filter_by_date_range(
    files: list[tuple[str, int | None]],
    date_range: tuple[str | None, str | None],
) -> list[tuple[str, int | None]]:
    start = datetime.strptime(date_range[0], '%Y-%m-%d') if date_range[0] else None
    end = datetime.strptime(date_range[1], '%Y-%m-%d') if date_range[1] else None
    filtered = []
    for path, size in files:
        match = re.search(r'\d{8}', Path(path).name)
        if not match:
            continue
        obs_date = datetime.strptime(match.group(), '%Y%m%d')
        if (start and obs_date < start) or (end and obs_date > end):
            continue
        filtered.append((path, size))
    return filtered

def trim_file(raw_path: Path, out_path: Path) -> None:
    """
    Subset a global NetCDF to the configured lon/lat region.

    Uses a temp file + atomic rename to prevent corrupt partial writes.
    Deletes raw_path after a successful write.
    """
    tmp_path = out_path.with_suffix(".tmp.nc")
    with xr.open_dataset(raw_path) as ds:
        trimmed = ds.sel(
            longitude=slice(*LON_RANGE),
            latitude=slice(*LAT_RANGE),
        )
        trimmed.to_netcdf(tmp_path)
    raw_path.unlink()
    tmp_path.rename(out_path)


def download_one(remote_path: str) -> str:
    """
    Download a single file from FTP, trim to region, and save.

    Downloads to a temp file first, then trims. Skips if already exists.
    """
    fn = Path(remote_path).name
    local_fp = LOCAL_DIR / fn

    if local_fp.exists():
        return f"[skip] {fn}"

    # Download full-globe file to a temp location, then trim
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with FTP(HOST, USER, PASSWORD) as ftp:
            ftp.cwd(REMOTE_DIR)
            with open(tmp_path, 'wb') as f:
                ftp.retrbinary(f"RETR {fn}", f.write)
        trim_file(tmp_path, local_fp)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        local_fp.unlink(missing_ok=True)
        return f"[ERROR] {fn}: {e}"

    return f"[done] {fn}"

def main():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    # Perform remote lookup
    print("Listing remote files...")
    files = list_remote_files_with_sizes()
    files.sort(key=lambda x: Path(x[0]).name)
    print(f"Found {len(files)} files on server")

    # Filter by date range
    files = filter_by_date_range(files, DATE_RANGE)
    print(f"Filtered to {len(files)} files in date range {DATE_RANGE}")

    to_download = [path for path, _ in files]
    total = sum(size for _, size in files if size is not None)
    print(f"Total download size: {(total / 1024**3):.2f} GB")
    print(f"Downloading {len(to_download)} files")

    failures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(download_one, f): f for f in to_download}
        for future in as_completed(future_to_file):
            result = future.result()
            print(result)
            if result.startswith("[ERROR]"):
                failures.append(result)

    if failures:
        print(f"\n{len(failures)} files failed:")
        for msg in failures:
            print(f"  {msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()
