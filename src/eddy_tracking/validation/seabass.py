"""
Module to read NASA SeaBASS files.
"""

from pathlib import Path

import pandas as pd


def read_sb(path: Path | str, below_detection: str = "nan") -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Read one SeaBASS file into a DataFrame and its header dict. below_detection sets how to treat the below-detection-limit sentinel.

    Returns (df, header). df holds one row per sample, columns named by /fields.
    df.attrs["units"] maps each field to its /units value.
    """
    if below_detection not in ("nan", "zero"):
        raise ValueError(f"below_detection must be 'nan' or 'zero', got {below_detection!r}")

    path = Path(path)

    # Read the header block, keep every /key=value, and count lines up to /end_header.
    header: dict[str, str] = {}
    n_skip = 0
    with open(path) as f:
        for line in f:
            n_skip += 1
            s = line.strip()
            if s.startswith('"') and s.endswith('"'):
                s = s[1:-1]  # some files quote every header line
            if s == "/end_header":
                break
            if not s or s.startswith("!"):
                continue  # blank line or comment
            if s.startswith("/") and "=" in s:
                key, value = s[1:].split("=", 1)
                header[key] = value

    fields = [f.strip() for f in header["fields"].split(",")]
    units = [unit.strip() for unit in header["units"].split(",")]
    if len(fields) != len(units):
        raise ValueError(
            f"{path} has {len(fields)} /fields values and {len(units)} /units values"
        )
    delim = header["delimiter"].strip()
    sep = {"comma": ",", "space": r"\s+", "tab": "\t"}.get(delim, delim)

    # Sentinels that always mean "no usable value".
    na_values = [header[k] for k in ("missing", "above_detection_limit") if k in header]
    below = header.get("below_detection_limit")
    if below_detection == "nan" and below is not None:
        na_values.append(below)

    df = pd.read_csv(
        path,
        sep=sep,
        skiprows=n_skip,
        header=None,
        names=fields,
        na_values=na_values,
        engine="python" if sep == r"\s+" else "c",
    )

    if below_detection == "zero" and below is not None:
        df = df.replace(float(below), 0.0)

    df.attrs["units"] = dict(zip(fields, units, strict=True))
    return df, header


def read_hplc_dir(hplc_dir: Path | str, below_detection: str = "nan") -> pd.DataFrame:
    """
    Read every .sb file under hplc_dir into one combined DataFrame.

    Adds a 'cruise' and 'source_file' column so each row keeps its origin.
    Adds a 'datetime' column when the files carry 'date' and 'time' fields.
    combined.attrs["units"] contains compatible units from all files.
    """
    hplc_dir = Path(hplc_dir)
    frames: list[pd.DataFrame] = []
    combined_units: dict[str, str] = {}
    for path in sorted(hplc_dir.glob("*.sb")):
        df, header = read_sb(path, below_detection=below_detection)
        for field, unit in df.attrs["units"].items():
            if field in combined_units and combined_units[field] != unit:
                raise ValueError(
                    f"{path} uses {unit!r} for {field!r}, "
                    f"but another file uses {combined_units[field]!r}"
                )
            combined_units[field] = unit
        df["cruise"] = header.get("cruise")
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No .sb files under {hplc_dir}")

    combined = pd.concat(frames, ignore_index=True)
    if "date" in combined.columns and "time" in combined.columns:
        combined["datetime"] = pd.to_datetime(
            combined["date"].astype(str) + " " + combined["time"].astype(str),
            format="%Y%m%d %H:%M:%S",
            errors="coerce",
        )
    combined.attrs["units"] = combined_units
    return combined
