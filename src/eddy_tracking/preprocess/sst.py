from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def read_multiple_sst(sst_dir: Path) -> pd.DataFrame:
    """Load SST composites into one in-memory time-indexed DataFrame."""
    sst_dir = Path(sst_dir)
    files = sorted(sst_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No SST .nc files in {sst_dir}")

    arrays = []
    for f in files:
        start, end = _parse_sst_time_window(f.name)
        midpoint = start + (end - start) / 2
        with xr.open_dataset(f) as ds:
            da = ds["sst"].load().expand_dims(time=[np.datetime64(midpoint)])
        arrays.append(da)

    return xr.concat(arrays, dim="time").to_dataframe(name="sst").sort_index()


def _parse_sst_time_window(filename: str) -> tuple[dt.date, dt.date]:
    match = re.search(r"(\d{8})_(\d{8})", filename)
    if not match:
        raise ValueError(f"Cannot parse SST date window from: {filename}")
    start = dt.datetime.strptime(match.group(1), "%Y%m%d").date()
    end = dt.datetime.strptime(match.group(2), "%Y%m%d").date()
    return start, end
