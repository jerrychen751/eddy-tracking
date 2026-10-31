"""Build validated eddy means from standard PACE chlor_a without SDP inputs."""

import argparse
import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from eddy_tracking.preprocess.tracks import (
    collect_eddies_for_date_range,
    load_date_eddy_index,
    load_track_observations,
    mask_pixels_inside_contour,
)
from eddy_tracking.utils.cli import write_settings
from eddy_tracking.utils.files import write_parquet_atomically


def collect_chlorophyll_files(
    directory: Path, date_range: tuple[dt.date, dt.date],
) -> list[tuple[Path, dt.date, dt.date]]:
    """Return nonoverlapping 8-day BGC files that intersect the requested dates; reject absent or ambiguous input."""
    files = []
    for path in sorted(directory.glob("*.nc")):
        match = re.fullmatch(
            r"PACE_OCI\.(\d{8})_(\d{8})\.L3m\.8D\.BGC\..+\.4km\.nc", path.name,
        )
        if match is None:
            raise ValueError(f"Unexpected 8-day PACE BGC filename: {path}")
        start, end = [dt.datetime.strptime(value, "%Y%m%d").date() for value in match.groups()]
        if not 0 <= (end - start).days < 8:
            raise ValueError(f"Invalid PACE composite dates: {path}")
        if end < date_range[0] or start > date_range[1]:
            continue
        files.append((path, start, end))
    files.sort(key=lambda item: item[1])
    if not files:
        raise FileNotFoundError(f"No PACE BGC composites intersect {date_range} in {directory}")
    for previous, current in zip(files, files[1:]):
        if current[1] <= previous[2]:
            raise ValueError(f"Overlapping PACE composites: {previous[0]} and {current[0]}")
    return files


def read_chlorophyll_field(path: Path) -> xr.DataArray:
    """Read chlor_a in mg/m³; reject the wrong variable, dimensions, units, or coordinate grid."""
    with xr.open_dataset(path) as dataset:
        if "chlor_a" not in dataset:
            raise ValueError(f"Missing chlor_a variable: {path}")
        field = dataset["chlor_a"]
        if set(field.dims) != {"lat", "lon"} or field.ndim != 2:
            raise ValueError(f"chlor_a must have lat and lon dimensions: {path}")
        if field.attrs.get("units") != "mg m^-3":
            raise ValueError(f"chlor_a units must be mg m^-3: {path}")
        for name, low, high in [("lat", -90, 90), ("lon", -180, 180)]:
            if name not in field.coords or field[name].dims != (name,):
                raise ValueError(f"Missing one-dimensional {name} coordinates: {path}")
            values = field[name].to_numpy()
            delta = np.diff(values)
            if (
                values.size == 0 or not np.isfinite(values).all()
                or np.any(values < low) or np.any(values > high)
                or not (np.all(delta > 0) or np.all(delta < 0))
            ):
                raise ValueError(f"Invalid or nonmonotonic {name} coordinates: {path}")
        return field.transpose("lat", "lon").load()


def read_movement_table(path: Path, observations: pd.DataFrame) -> pd.DataFrame:
    """Read one movement row per physical track; reject duplicate identities or inconsistent physical lifetimes."""
    movement = pd.read_parquet(path)
    identity_columns = ["polarity", "track_id"]
    required = identity_columns + ["birth_date", "death_date", "birth_side", "death_side", "movement"]
    if not set(required).issubset(movement.columns):
        raise ValueError(f"Missing movement columns: {path}")
    if movement[required].isna().to_numpy().any() or movement.duplicated(identity_columns).any():
        raise ValueError(f"Null values or duplicate track identities in movement table: {path}")
    if (
        not movement["birth_side"].isin(["N", "S", ""]).to_numpy().all()
        or not movement["death_side"].isin(["N", "S", ""]).to_numpy().all()
    ):
        raise ValueError(f"Invalid endpoint sides in movement table: {path}")
    expected_classes = (movement["birth_side"] + movement["death_side"]).where(
        movement["birth_side"].ne("") & movement["death_side"].ne(""), "",
    )
    if not movement["movement"].eq(expected_classes).to_numpy().all():
        raise ValueError(f"Movement classes disagree with endpoint sides: {path}")
    lifetimes = observations.groupby(identity_columns)["date"].agg(["min", "max"]).reset_index()
    joined = movement.merge(lifetimes, on=identity_columns, how="outer", validate="one_to_one", indicator=True)
    if (
        not joined["_merge"].eq("both").to_numpy().all()
        or not joined["birth_date"].eq(joined["min"]).to_numpy().all()
        or not joined["death_date"].eq(joined["max"]).to_numpy().all()
        or not joined["death_date"].gt(joined["birth_date"]).to_numpy().all()
    ):
        raise ValueError(f"Movement identities or lifetimes disagree with physical tracks: {path}")
    return movement


def build_chlorophyll_table(
    pace_bgc_dir: Path,
    track_dir: Path,
    axis_dir: Path,
    date_range: tuple[dt.date, dt.date],
    selected_ids: set[int] | None,
    min_coverage: float,
    min_pixels: int,
) -> pd.DataFrame:
    """Return eddy means, coverage, and physical ages after input checks; retain gaps as absent observations."""
    if not np.isfinite(min_coverage) or not 0 < min_coverage <= 1:
        raise ValueError("--min-coverage must lie in (0, 1].")
    if min_pixels < 1:
        raise ValueError("--min-pixels must be a positive integer.")
    if date_range[0] > date_range[1]:
        raise ValueError("--dates must be chronological.")
    files = collect_chlorophyll_files(pace_bgc_dir, date_range)
    observations = load_track_observations(track_dir)
    movement = read_movement_table(axis_dir / "eddy_movement.parquet", observations)
    date_index = load_date_eddy_index(track_dir, selected_ids, date_range=date_range)
    rows = []
    for path, start, end in files:
        field = read_chlorophyll_field(path)
        for eddy in collect_eddies_for_date_range(date_index, start, end):
            if (
                eddy.contour_lon.ndim != 1 or eddy.contour_lon.size < 3
                or eddy.contour_lon.shape != eddy.contour_lat.shape
                or not np.isfinite(eddy.contour_lon).all()
                or not np.isfinite(eddy.contour_lat).all()
            ):
                raise ValueError(f"Invalid contour for {eddy.polarity} track {eddy.track_id} in {path.name}")
            interior = field.where(
                (field["lon"] >= eddy.contour_lon.min())
                & (field["lon"] <= eddy.contour_lon.max())
                & (field["lat"] >= eddy.contour_lat.min())
                & (field["lat"] <= eddy.contour_lat.max()), drop=True,
            )
            inside = mask_pixels_inside_contour(interior["lon"].to_numpy(), interior["lat"].to_numpy(), eddy.contour_lon, eddy.contour_lat)
            n_inside = int(inside.sum())
            if n_inside == 0:
                continue
            values = interior.to_numpy()[inside]
            valid_values = values[np.isfinite(values) & (values > 0)]
            coverage = len(valid_values) / n_inside
            if coverage < min_coverage or len(valid_values) < min_pixels:
                continue
            rows.append({
                "polarity": eddy.polarity, "track_id": eddy.track_id,
                "date": pd.Timestamp(start + (end - start) / 2).as_unit("ns"),
                "eddy_chl_mean": float(valid_values.mean()),
                "n_chl_pixels": len(valid_values), "valid_frac": coverage,
            })
    if not rows:
        raise ValueError("No eddy chlorophyll observations pass the coverage and pixel thresholds.")
    identity_columns = ["polarity", "track_id"]
    chl = pd.DataFrame(rows).merge(
        movement[identity_columns + ["movement", "birth_date", "death_date"]],
        on=identity_columns, how="left", validate="many_to_one",
    )
    if chl.duplicated(identity_columns + ["date"]).any():
        raise ValueError("Duplicate chlorophyll observations for an eddy and composite date.")
    if chl[["movement", "birth_date", "death_date"]].isna().to_numpy().any():
        raise ValueError("A chlorophyll observation has no physical track record.")
    lifetime_days = (chl["death_date"] - chl["birth_date"]).dt.days
    chl["age_frac"] = ((chl["date"] - chl["birth_date"]).dt.days / lifetime_days).clip(0, 1)
    if not chl["age_frac"].between(0, 1).all():
        raise ValueError("Physical track ages must be finite and lie in [0, 1].")
    return chl


def main(argv: list[str]) -> None:
    """Validate inputs and atomically replace eddy_chlor_a.parquet under --out; preserve the prior table on failure."""
    parser = argparse.ArgumentParser(prog="eddy-tracking collocate-chlorophyll")
    parser.add_argument("--pace-bgc", type=Path, required=True, help="folder of PACE L3 8-day BGC files")
    parser.add_argument("--tracks", type=Path, required=True, help="folder track-eddies wrote")
    parser.add_argument("--axis", type=Path, required=True, help="folder trace-axis wrote")
    parser.add_argument("--dates", nargs=2, required=True, metavar=("START", "END"), help="ISO dates, both inclusive; eddy-days outside them are left out")
    parser.add_argument("--track-ids", nargs="+", required=True, help="track ids to collocate in both polarities, or all")
    parser.add_argument("--min-coverage", type=float, required=True, help="fraction of interior pixels that must hold a positive chlor_a")
    parser.add_argument("--min-pixels", type=int, required=True, help="interior pixels that must hold a positive chlor_a")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    write_settings(args.out, "collocate-chlorophyll", args, operational=set())
    args.out.mkdir(parents=True, exist_ok=True)

    date_range = (dt.date.fromisoformat(args.dates[0]), dt.date.fromisoformat(args.dates[1]))
    selected_ids = None if args.track_ids == ["all"] else {int(track_id) for track_id in args.track_ids}
    chl = build_chlorophyll_table(args.pace_bgc, args.tracks, args.axis, date_range, selected_ids, args.min_coverage, args.min_pixels)
    out_path = args.out / "eddy_chlor_a.parquet"
    write_parquet_atomically(chl, out_path)
    print(f"output_file: {out_path}\neddy_composites: {len(chl)}")
