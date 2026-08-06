from __future__ import annotations

import datetime as dt
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def read_multiple_sss(fps: Sequence[Path | str]) -> pd.DataFrame:
    """Load SSS composite files into one in-memory time-indexed DataFrame."""
    files = [Path(fp) for fp in fps]
    if not files:
        raise ValueError("fps must contain at least one SSS file")

    arrays = []
    for f in files:
        # "SMAP_L3_SSS_20250114_8DAYS_V5.0.nc" carries the center date of its 8-day window.
        match = re.search(r"SSS_(\d{8})_", f.name)
        if not match:
            raise ValueError(f"Cannot parse SSS date from: {f.name}")
        center = dt.datetime.strptime(match.group(1), "%Y%m%d").date()
        with xr.open_dataset(f) as ds:
            # (n_lat, n_lon) -> (1, n_lat, n_lon)
            da = ds["smap_sss"].load().expand_dims(time=[np.datetime64(center)])
        arrays.append(da)

    # len(fps) arrays of (1, n_lat, n_lon) -> (len(fps), n_lat, n_lon)
    return (
        xr.concat(arrays, dim="time")
        .to_dataframe(name="smap_sss")
        .sort_index()
    )
