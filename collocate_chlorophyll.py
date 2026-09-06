"""Build validated eddy means from standard PACE chlor_a without SDP inputs."""

import argparse
import datetime as dt
import re
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path as PolygonPath

from collocate_pace import EddyObs, build_date_eddy_index, collect_eddies_for_window
from eddy_tracking.packages.py_eddy_tracker.observations.tracking import TrackEddiesObservations
from gulf_stream import load_track_observations
from utils.config import PROJECT_ROOT, load_config, resolve_output_dir


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


def build_chlorophyll_table(experiment: str) -> pd.DataFrame:
    """Return eddy means, coverage, and physical ages after input checks; retain gaps as absent observations."""
    cfg = load_config(experiment)
    settings = cfg["collocate_chlorophyll"]
    min_coverage = settings["min_coverage"]
    min_pixels = settings["min_pixels"]
    if not np.isfinite(min_coverage) or not 0 < min_coverage <= 1:
        raise ValueError("collocate_chlorophyll.min_coverage must lie in (0, 1].")
    if isinstance(min_pixels, bool) or not isinstance(min_pixels, int) or min_pixels < 1:
        raise ValueError("collocate_chlorophyll.min_pixels must be a positive integer.")
    collocation_cfg = cfg["collocate_pace"]
    if collocation_cfg["temporal_resolution"] != "8D":
        raise ValueError("Standard chlorophyll collocation requires 8-day composites.")
    dates = collocation_cfg["date_range"]
    if len(dates) != 2:
        raise ValueError("collocate_pace.date_range must contain a start and an end date.")
    date_range = (dt.date.fromisoformat(dates[0]), dt.date.fromisoformat(dates[1]))
    if date_range[0] > date_range[1]:
        raise ValueError("collocate_pace.date_range must be chronological.")
    data_dir = PROJECT_ROOT / "data" / experiment
    pace_dir = PROJECT_ROOT / cfg["base"]["data"]["root"] / cfg["base"]["dataset"] / "bronze" / cfg["base"]["data"]["pace_bgc_dir"]
    files = collect_chlorophyll_files(pace_dir, date_range)
    track_dir = data_dir / "silver/eddy_track"
    observations = load_track_observations(track_dir / "cyclone", track_dir / "anticyclone")
    movement = read_movement_table(data_dir / "silver/gulf_stream/eddy_movement.parquet", observations)
    configured_ids = collocation_cfg.get("track_ids")
    selected_ids = set(configured_ids) if configured_ids else None
    date_index: dict[dt.date, list[EddyObs]] = defaultdict(list)
    for polarity in ("cyclone", "anticyclone"):
        tracked = TrackEddiesObservations.load_file(str(track_dir / polarity / f"{polarity}_tracks.zarr"))
        polarity_index = build_date_eddy_index(
            tracked, polarity, selected_ids, collocation_cfg.get("region"), date_range,
        )
        for day, eddies in polarity_index.items():
            date_index[day].extend(eddies)
    rows = []
    for path, start, end in files:
        field = read_chlorophyll_field(path)
        for eddy in collect_eddies_for_window(date_index, start, end):
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
            lon_grid, lat_grid = np.meshgrid(interior["lon"].to_numpy(), interior["lat"].to_numpy())
            polygon = PolygonPath(np.column_stack([eddy.contour_lon, eddy.contour_lat]))
            inside = polygon.contains_points(np.column_stack([lon_grid.ravel(), lat_grid.ravel()]))
            n_inside = int(inside.sum())
            if n_inside == 0:
                continue
            values = interior.to_numpy().ravel()[inside]
            valid_values = values[np.isfinite(values) & (values > 0)]
            coverage = len(valid_values) / n_inside
            if coverage < min_coverage or len(valid_values) < min_pixels:
                continue
            rows.append({
                "polarity": eddy.polarity, "track_id": eddy.track_id,
                "date": pd.Timestamp(start + (end - start) / 2),
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


def main(experiment: str | None = None) -> None:
    """Validate inputs and atomically replace silver/pace_chl/eddy_chlor_a.parquet; preserve the prior table on failure."""
    if experiment is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("experiment")
        experiment = cast(str, parser.parse_args().experiment)
    chl = build_chlorophyll_table(experiment)
    out_path = resolve_output_dir(experiment, "pace_chl") / "eddy_chlor_a.parquet"
    with NamedTemporaryFile(dir=out_path.parent, suffix=".parquet", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        chl.to_parquet(temporary_path, index=False)
        temporary_path.replace(out_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"output_file: {out_path}\neddy_composites: {len(chl)}")


if __name__ == "__main__":
    main()
