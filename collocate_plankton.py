"""Build the gold table of Copernicus Marine plankton field means inside each tracked eddy, one row per eddy and 8-day composite window."""

import argparse
import datetime as dt
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
from utils.config import PROJECT_ROOT, load_config, resolve_data_dir, resolve_gold_dir


def build_plankton_table(experiment: str) -> pd.DataFrame:
    """
    Return one row per tracked eddy and NASA 8-day period over the eddy tracking window of the experiment. Each period starts on day 1 of the year and every eighth day after it, and the last period of a year ends on December 31. A pixel's composite is its mean over the days of the period with data, and each field mean covers the composite pixels inside the eddy boundary nearest the period midpoint that hold that field. The uncertainty of a field averages over the same pixels as the field. A row needs CHL on at least collocate_plankton.min_coverage of the interior pixels and on at least min_pixels of them; the other fields may have fewer pixels, down to zero, which leaves their mean absent. The flags field only marks land, so it is not aggregated. Concentrations are in mg/m³, uncertainties in percent, and date is the period midpoint.
    """
    cfg = load_config(experiment)
    settings = cfg["collocate_plankton"]
    first_day, last_day = [dt.date.fromisoformat(value) for value in cfg["base"]["time"]["eddy_date_range"]]
    data_dir = PROJECT_ROOT / "data" / experiment
    track_dir = data_dir / "silver/eddy_track"
    date_index: dict[dt.date, list[EddyObs]] = defaultdict(list)
    for polarity in ("cyclone", "anticyclone"):
        tracked = TrackEddiesObservations.load_file(str(track_dir / polarity / f"{polarity}_tracks.zarr"))
        for day, eddies in build_date_eddy_index(tracked, polarity, date_range=(first_day, last_day)).items():
            date_index[day].extend(eddies)
    windows = []
    for year in range(first_day.year, last_day.year + 1):
        start = dt.date(year, 1, 1)
        while start.year == year:
            end = min(start + dt.timedelta(days=7), dt.date(year, 12, 31))
            if end >= first_day and start <= last_day:
                windows.append((start, end))
            start = end + dt.timedelta(days=1)
    concentrations = ["CHL", "DIATO", "DINO", "GREEN", "HAPTO", "MICRO", "NANO", "PICO", "PROCHLO", "PROKAR"]
    fields = xr.open_mfdataset(
        sorted(resolve_data_dir(cfg, "plankton_dir").glob("plankton_*.nc")), combine="by_coords",
    )[concentrations + [f"{field}_uncertainty" for field in concentrations]]
    lon = fields["longitude"].to_numpy()
    lat = fields["latitude"].to_numpy()
    rows = []
    for start, end in windows:
        composite = fields.sel(time=slice(str(start), str(end))).mean("time").load()  # (n_days, n_lat, n_lon) -> (n_lat, n_lon) per field
        for eddy in collect_eddies_for_window(date_index, start, end):
            box = composite.isel(
                longitude=np.flatnonzero((lon >= eddy.contour_lon.min()) & (lon <= eddy.contour_lon.max())),
                latitude=np.flatnonzero((lat >= eddy.contour_lat.min()) & (lat <= eddy.contour_lat.max())),
            )
            lon_grid, lat_grid = np.meshgrid(box["longitude"].to_numpy(), box["latitude"].to_numpy())  # (n_box_lon,) + (n_box_lat,) -> (n_box_lat, n_box_lon) each
            polygon = PolygonPath(np.column_stack([eddy.contour_lon, eddy.contour_lat]))
            inside = polygon.contains_points(np.column_stack([lon_grid.ravel(), lat_grid.ravel()]))  # (n_box_lat*n_box_lon, 2) -> (n_box_lat*n_box_lon,)
            n_inside = int(inside.sum())
            if n_inside == 0:
                continue
            chl_valid = np.isfinite(box["CHL"].to_numpy().ravel()[inside])
            if chl_valid.sum() < settings["min_pixels"] or chl_valid.mean() < settings["min_coverage"]:
                continue
            row = {
                "polarity": eddy.polarity, "track_id": eddy.track_id,
                "date": pd.Timestamp(start + (end - start) / 2),
                "center_lon": eddy.center_lon, "center_lat": eddy.center_lat,
                "n_pixels": n_inside, "valid_frac": float(chl_valid.mean()),
            }
            for field in concentrations:
                values = box[field].to_numpy().ravel()[inside]
                uncertainty = box[f"{field}_uncertainty"].to_numpy().ravel()[inside]
                valid = np.isfinite(values)
                row[field] = float(values[valid].mean()) if valid.any() else np.nan
                row[f"{field}_uncertainty"] = float(np.nanmean(uncertainty[valid])) if valid.any() else np.nan
                row[f"{field}_n_pixels"] = int(valid.sum())
            rows.append(row)
    identity_columns = ["polarity", "track_id"]
    movement = pd.read_parquet(data_dir / "silver/gulf_stream/eddy_movement.parquet")
    table = pd.DataFrame(rows).merge(
        movement[identity_columns + ["movement", "birth_date", "death_date"]], on=identity_columns, how="left",
    )
    lifetime_days = (table["death_date"] - table["birth_date"]).dt.days
    table["age_frac"] = ((table["date"] - table["birth_date"]).dt.days / lifetime_days).clip(0, 1)
    return table


def main(experiment: str | None = None) -> None:
    """Write gold/eddy_plankton_table.parquet through a temporary file, so a failed build leaves the prior table in place."""
    if experiment is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("experiment")
        experiment = cast(str, parser.parse_args().experiment)
    table = build_plankton_table(experiment)
    out_path = resolve_gold_dir(experiment, "eddy_plankton_table.parquet")
    with NamedTemporaryFile(dir=out_path.parent, suffix=".parquet", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        table.to_parquet(temporary_path, index=False)
        temporary_path.replace(out_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"output_file: {out_path}\neddy_composites: {len(table)}")


if __name__ == "__main__":
    main()
