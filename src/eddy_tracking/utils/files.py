from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import xarray as xr


def write_parquet_atomically(table: pd.DataFrame, out_path: Path) -> None:
    with NamedTemporaryFile(dir=out_path.parent, suffix=".parquet", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        table.to_parquet(temporary_path, index=False)
        temporary_path.replace(out_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_netcdf_atomically(dataset: xr.Dataset, out_path: Path, encoding: dict | None = None) -> None:
    tmp_path = out_path.with_suffix(".tmp.nc")
    try:
        dataset.to_netcdf(tmp_path, encoding=encoding)
        tmp_path.rename(out_path)
    # BaseException, so an interrupt also removes the partial file: the readers glob every *.nc in a bronze directory
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
