from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from ftplib import FTP
from pathlib import Path
from typing import TYPE_CHECKING

import earthaccess
from dotenv import load_dotenv

if TYPE_CHECKING:
    import xarray as xr
    from harmony import Client


def login_earthdata() -> earthaccess.Auth:
    """Log in to NASA Earthdata from ~/.netrc, which the caller must create first."""
    return earthaccess.login(strategy="netrc")


def configure_obdaac_opendap_auth() -> None:
    """Write a .dodsrc under the system temp dir and point the DAPRCFILE env var at it, so netCDF4 authenticates to OB.DAAC OPeNDAP from ~/.netrc. The directory is private to this process and goes away at exit, because libcurl rewrites the whole cookie jar at handle cleanup and run_pipeline runs the PACE and the SST download stages at the same time."""
    netrc_path = Path.home() / ".netrc"
    if not netrc_path.exists():
        raise FileNotFoundError(
            "~/.netrc with Earthdata Login credentials is required for OPeNDAP access"
        )

    dap_dir = Path(tempfile.mkdtemp(prefix="obdaac_opendap_"))
    atexit.register(shutil.rmtree, dap_dir, True)
    rc_path = dap_dir / ".dodsrc"
    rc_path.write_text(
        f"HTTP.NETRC={netrc_path}\n"
        f"HTTP.COOKIEJAR={dap_dir / 'urs_cookies.txt'}\n"
        "HTTP.DEFLATE=1\n"
    )
    os.environ["DAPRCFILE"] = str(rc_path)


def open_obdaac_dataset(url: str) -> xr.Dataset:
    """
    Open one OB.DAAC OPeNDAP URL, and try again after a pause when the open fails. OB.DAAC throttles by client address and answers HTTP 429 with a body that netCDF4 reports as "NetCDF: Access failure", so a burst of requests, or two download stages against oceandata.sci.gsfc.nasa.gov at the same time, makes every open fail. The pause starts at 30 seconds and doubles over 4 attempts. Any other OSError, such as a 404 on a URL built from the granule filename, raises at once. xarray retries the later hyperslab read itself, through robust_getitem, so this covers the open only. The caller owns the returned dataset and must close it.
    """
    import xarray as xr

    delay_seconds = 30
    attempt = 1
    while True:
        try:
            return xr.open_dataset(url, engine="netcdf4")
        except OSError as exc:
            if "Access failure" not in str(exc) or attempt == 4:
                raise
            print(
                "status: throttled\n"
                f"attempt: {attempt}\n"
                f"retry_in_seconds: {delay_seconds}"
            )
            time.sleep(delay_seconds)
            delay_seconds *= 2
            attempt += 1


def login_harmony() -> Client:
    """Create a Harmony client for server-side SSS subsetting."""
    from harmony import Client

    return Client()


@dataclass(frozen=True)
class AvisoCredentials:
    host: str
    user: str
    password: str = field(repr=False)


def load_aviso_credentials() -> AvisoCredentials:
    load_dotenv()
    for variable_name in ("FTP_HOST", "FTP_USER", "FTP_PASSWORD"):
        if not os.environ.get(variable_name):
            raise SystemExit(
                f"Missing required env var: {variable_name} "
                "(set in .env or environment)"
            )

    return AvisoCredentials(
        host=os.environ["FTP_HOST"],
        user=os.environ["FTP_USER"],
        password=os.environ["FTP_PASSWORD"],
    )


def login_aviso(host: str, user: str, password: str) -> FTP:
    return FTP(host, user, password)
