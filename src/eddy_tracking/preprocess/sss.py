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
