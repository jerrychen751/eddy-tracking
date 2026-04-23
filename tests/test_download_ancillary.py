import numpy as np
import xarray as xr

from utils.download_ancillary import (
    infer_lat_lon_names,
    strip_harmony_prefix,
)


def test_strip_harmony_prefix_removes_numeric_prefix():
    assert strip_harmony_prefix("1234567_SMAP_L3_SSS_20250101.nc4") == "SMAP_L3_SSS_20250101.nc4"


def test_strip_harmony_prefix_leaves_unprefixed_names_alone():
    assert strip_harmony_prefix("AQUA_MODIS.20240929_20241006.L3m.8D.SST.sst.4km.nc") == \
        "AQUA_MODIS.20240929_20241006.L3m.8D.SST.sst.4km.nc"


def test_infer_lat_lon_names_picks_short_variants():
    ds = xr.Dataset(coords={"lat": [1, 2], "lon": [3, 4]})
    assert infer_lat_lon_names(ds) == ("lat", "lon")


def test_infer_lat_lon_names_picks_long_variants():
    ds = xr.Dataset(coords={"latitude": [1, 2], "longitude": [3, 4]})
    assert infer_lat_lon_names(ds) == ("latitude", "longitude")
