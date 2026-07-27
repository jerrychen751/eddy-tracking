import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from eddy_tracking.utils import authentication


def test_login_earthdata_delegates_to_earthaccess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_auth = Mock()
    login = Mock(return_value=expected_auth)
    monkeypatch.setattr(authentication.earthaccess, "login", login)

    assert authentication.login_earthdata() is expected_auth
    login.assert_called_once_with(strategy="netrc")


def test_configure_obdaac_opendap_auth_uses_netrc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    netrc_path = home_dir / ".netrc"
    netrc_path.write_text("machine example.test\n")
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    monkeypatch.setattr(authentication.Path, "home", lambda: home_dir)
    monkeypatch.setattr(authentication.tempfile, "gettempdir", lambda: str(temp_dir))
    monkeypatch.delenv("DAPRCFILE", raising=False)

    authentication.configure_obdaac_opendap_auth()

    dap_dir = temp_dir / "obdaac_opendap"
    rc_path = dap_dir / ".dodsrc"
    assert os.environ["DAPRCFILE"] == str(rc_path)
    assert rc_path.read_text() == (
        f"HTTP.NETRC={netrc_path}\n"
        f"HTTP.COOKIEJAR={dap_dir / 'urs_cookies.txt'}\n"
        "HTTP.DEFLATE=1\n"
    )


def test_configure_obdaac_opendap_auth_requires_netrc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(authentication.Path, "home", lambda: tmp_path)

    with pytest.raises(FileNotFoundError, match="Earthdata Login credentials"):
        authentication.configure_obdaac_opendap_auth()


def test_authenticate_harmony_returns_validated_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_client = Mock()
    client_type = Mock(return_value=expected_client)
    harmony_module = SimpleNamespace(Client=client_type)
    monkeypatch.setitem(sys.modules, "harmony", harmony_module)

    assert authentication.login_harmony() is expected_client
    client_type.assert_called_once_with()


def test_load_aviso_credentials_reads_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_dotenv = Mock()
    monkeypatch.setattr(authentication, "load_dotenv", load_dotenv)
    monkeypatch.setenv("FTP_HOST", "ftp.example.test")
    monkeypatch.setenv("FTP_USER", "researcher")
    monkeypatch.setenv("FTP_PASSWORD", "secret")

    credentials = authentication.load_aviso_credentials()

    load_dotenv.assert_called_once_with()
    assert credentials.host == "ftp.example.test"
    assert credentials.user == "researcher"
    assert credentials.password == "secret"
    assert "secret" not in repr(credentials)


def test_load_aviso_credentials_reports_first_missing_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(authentication, "load_dotenv", Mock())
    monkeypatch.delenv("FTP_HOST", raising=False)
    monkeypatch.delenv("FTP_USER", raising=False)
    monkeypatch.delenv("FTP_PASSWORD", raising=False)

    with pytest.raises(SystemExit, match="Missing required env var: FTP_HOST"):
        authentication.load_aviso_credentials()


def test_connect_aviso_ftp_passes_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_connection = Mock()
    ftp_type = Mock(return_value=expected_connection)
    monkeypatch.setattr(authentication, "FTP", ftp_type)

    connection = authentication.login_aviso(
        "ftp.example.test",
        "researcher",
        "secret",
    )

    assert connection is expected_connection
    ftp_type.assert_called_once_with("ftp.example.test", "researcher", "secret")
