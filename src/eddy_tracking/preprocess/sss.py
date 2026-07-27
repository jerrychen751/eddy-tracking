from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def read_multiple_sss(sss_dir: Path) -> pd.DataFrame:
    """Load SSS composites into one in-memory time-indexed DataFrame."""
    sss_dir = Path(sss_dir)
    files = sorted(sss_dir.glob("*.nc4"))
    if not files:
        raise FileNotFoundError(f"No SSS .nc4 files in {sss_dir}")

    arrays = []
    for f in files:
        center = _parse_sss_center_date(f.name)
        with xr.open_dataset(f) as ds:
            da = ds["smap_sss"].load().expand_dims(time=[np.datetime64(center)])
        arrays.append(da)

    return (
        xr.concat(arrays, dim="time")
        .to_dataframe(name="smap_sss")
        .sort_index()
    )


def _parse_sss_center_date(filename: str) -> dt.date:
    match = re.search(r"SSS_(\d{8})_", filename)
    if not match:
        raise ValueError(f"Cannot parse SSS date from: {filename}")
    return dt.datetime.strptime(match.group(1), "%Y%m%d").date()
