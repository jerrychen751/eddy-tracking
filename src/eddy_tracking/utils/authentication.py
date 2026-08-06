from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from ftplib import FTP
from pathlib import Path
from typing import TYPE_CHECKING

import earthaccess
from dotenv import load_dotenv

if TYPE_CHECKING:
    from harmony import Client


def login_earthdata() -> earthaccess.Auth:
    """Log in to NASA Earthdata from ~/.netrc, which the caller must create first."""
    return earthaccess.login(strategy="netrc")


def configure_obdaac_opendap_auth() -> None:
    """Write a .dodsrc under the system temp dir and point the DAPRCFILE env var at it, so netCDF4 authenticates to OB.DAAC OPeNDAP from ~/.netrc."""
    netrc_path = Path.home() / ".netrc"
    if not netrc_path.exists():
        raise FileNotFoundError(
            "~/.netrc with Earthdata Login credentials is required for OPeNDAP access"
        )

    dap_dir = Path(tempfile.gettempdir()) / "obdaac_opendap"
    dap_dir.mkdir(exist_ok=True)
    rc_path = dap_dir / ".dodsrc"
    rc_path.write_text(
        f"HTTP.NETRC={netrc_path}\n"
        f"HTTP.COOKIEJAR={dap_dir / 'urs_cookies.txt'}\n"
        "HTTP.DEFLATE=1\n"
    )
    os.environ["DAPRCFILE"] = str(rc_path)


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
