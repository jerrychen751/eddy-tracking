from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest
import xarray as xr

from eddy_tracking.downloads import sst


def test_download_aqua_sst_uses_opendap_and_writes_bbox_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filename = "AQUA_MODIS.20250101_20250108.L3m.8D.SST.sst.4km.nc"
    download_url = f"https://example.test/getfile/{filename}"
    opendap_url = f"https://example.test/opendap/{filename}"
    granule = Mock()
    granule.data_links.return_value = [download_url]
    granule.get.return_value = {
        "RelatedUrls": [
            {
                "Subtype": "OPENDAP DATA",
                "URL": opendap_url,
            }
        ]
    }
    remote_ds = xr.Dataset(
        data_vars={
            "sst": (
                ("lat", "lon"),
                [
                    [0, 1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10, 11],
                    [12, 13, 14, 15, 16, 17],
                    [18, 19, 20, 21, 22, 23],
                    [24, 25, 26, 27, 28, 29],
                ],
            )
        },
        coords={
            "lat": [45.0, 40.0, 35.0, 30.0, 25.0],
            "lon": [-85.0, -80.0, -75.0, -70.0, -65.0, -60.0],
        },
    )
    real_open_dataset = xr.open_dataset
    open_dataset = Mock(return_value=nullcontext(remote_ds))
    login = Mock()
    configure_auth = Mock()
    search_data = Mock(return_value=[granule])
    monkeypatch.setattr(sst, "login_earthdata", login)
    monkeypatch.setattr(sst, "configure_obdaac_opendap_auth", configure_auth)
    monkeypatch.setattr(sst.earthaccess, "search_data", search_data)
    monkeypatch.setattr(sst.xr, "open_dataset", open_dataset)

    out_dir = tmp_path / "sst"
    saved = sst.download_aqua_sst_8d_4km(
        date_range=("2025-01-01", "2025-01-08"),
        lon_range=(-81.0, -56.0),
        lat_range=(29.0, 44.0),
        out_dir=out_dir,
        collection_id="C1615905770-OB_DAAC",
    )

    assert saved == 1
    login.assert_called_once_with()
    configure_auth.assert_called_once_with()
    search_data.assert_called_once_with(
        concept_id="C1615905770-OB_DAAC",
        temporal=("2025-01-01", "2025-01-08"),
        count=-1,
    )
    open_dataset.assert_called_once_with(opendap_url, engine="netcdf4")

    output_path = out_dir / filename
    with real_open_dataset(output_path) as output_ds:
        assert output_ds["lat"].values.tolist() == [40.0, 35.0, 30.0]
        assert output_ds["lon"].values.tolist() == [
            -80.0,
            -75.0,
            -70.0,
            -65.0,
            -60.0,
        ]
        assert output_ds["sst"].shape == (3, 5)


def test_opendap_url_builds_fallback_from_filename() -> None:
    granule = Mock()
    granule.get.return_value = {"RelatedUrls": []}
    filename = "AQUA_MODIS.20250101_20250108.L3m.8D.SST.sst.4km.nc"

    url = sst._opendap_url(granule, filename)

    assert url == (
        "https://oceandata.sci.gsfc.nasa.gov/opendap/MODISA/L3SMI/"
        f"2025/0101/{filename}"
    )
